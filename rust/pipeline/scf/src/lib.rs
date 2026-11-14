//! The Roothaan step at a fixed operator, `F C = S C ε` solved through the
//! orthogonalizer `X = S^(-1/2)`, as tasks of a [`Dag`](fb_core_planner::Dag):
//! what the C++ SCF does once per iteration. The C++
//! `src/pipeline/scf/scf_loop.cppm`'s `solve_fock` is the reference design,
//! without the density it forms after the orbitals.
//!
//! [`add_solve_fock`] adds, for the rows of a symmetric operator F and of X:
//!
//! - **`fock transform`**, on the worker's stream: `F' = X F X` by two
//!   gemms, `F X` into a scratch row and `Xᵀ (F X)` into the output row F'
//!   (X is symmetric, so `Xᵀ F X` is `X F X`).
//! - **`syevd`** and **`syevd check`**,
//!   [`add_diagonalize`](fb_recipes_diagonalize::add_diagonalize)'s two
//!   tasks on F': `F' = C' diag(ε) C'ᵀ`, the orbital energies ε ascending.
//! - **`back transform`**, on the worker's stream: `C = X C'` by a gemm into
//!   the output row C.
//!
//! The operator is a row, not a matrix of this crate: the core Hamiltonian
//! here, the Fock matrix of each SCF iteration later, with no change to the
//! step.
//!
//! **One closed-shell iteration's dense nodes**, steps 2, 3 and 6 of the
//! C++ loop that the C++ module's README lists under its step, each a task
//! over rows, shaped to sit in a loop body
//! ([`Dag::add_loop`](fb_core_planner::Dag::add_loop)):
//!
//! - [`occupied_orbitals`]: `nocc`, half the molecule's electron count, at
//!   planning; an odd count is refused (no open shell is ported).
//! - [`add_density`]: **`density`**, `D = 2 C_occ C_occᵀ` from C's first
//!   `nocc` columns, writing the loop's carried D and occupied factor.
//! - [`add_fock`]: **`fock`**, `F = H + J - K/2` and
//!   `E = tr(D (H + F))/2 + E_nuc`; **`residual`**,
//!   `e = Xᵀ (F D S - S D F) X`; and **`energy and norm`**, E and `‖e‖_F`
//!   in one pinned row for the loop's decision.
//!
//! J and K are input rows: the density-fitted build is
//! `fb_integrals_eri_density_fitting_jk`'s.
//!
//! [`Run`] is the whole run at the core Hamiltonian, the core guess's
//! orbitals: an XYZ file and a basis name planned once (the nuclei, the
//! records, built once for both S and H, S, H, X and the step), laid out
//! in one arena per kind, and run on one stream, returning ε, C and the
//! timing record.
//!
//! | C++ | Here |
//! |---|---|
//! | `solve_fock(res, st, open, fock, x, n_alpha, n_beta)`: `gemmwich`, `diagonalize`, `gemm` | [`add_solve_fock`]: `fock transform`, the eigensolver's two tasks, `back transform` |
//! | `gemmwich(blas, n, x, fock, st.work, st.tmp)`, `X^T F X` through an n × n work | `fock transform`'s two gemms on the worker's handle ([`DeviceStream::blas`](fb_core_thread_pool::DeviceStream::blas)), `F X` its scratch row |
//! | `diagonalize<double>(res, n, st.tmp, st.eps)`, F' overwritten by C', one stream sync for the flag | [`add_diagonalize`](fb_recipes_diagonalize::add_diagonalize) on F', C' and ε its output rows, the sync its `syevd check` |
//! | `st.eps`, `st.c`, the loop's buffers | ε and C, output rows |
//! | `syevd_lwork`, sized into the workspace by the footprint | `lwork` handed in, [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) asked of the device before planning |
//! | D from the occupied columns, `gemm<N, T>(…, n_alpha, kTwo, c, c, kZero, d)` | [`add_density`]'s `density`: C's first `nocc` columns copied into the carried factor by a geam, then the same gemm from it into the carried D |
//! | Da, Db, ROHF's `roothaan_fock`, `canonicalize_open`, `publish_open` | not ported: an odd electron count is refused by [`occupied_orbitals`] |
//! | `DfFockBuilder::build`'s `F = H; F += J; F -= K/2` (copy and two axpys) | [`add_fock`]'s `fock`: `H + J`, then `- K/2`, by two geams, the same sums in the same order |
//! | `electronic_energy`, two `dot`s into host scalars, each a stream sync | `fock`'s two gemvs (`vec(D)` as an n² × 1 matrix) into one device scalar, copied down to a pinned row; E_nuc added by `energy and norm` |
//! | `commutator`: `F D`, `(F D) S`, `geam(N, T)`, `gemmwich`, `nrm2` into a host scalar | `residual`: the same gemms and geam, `Xᵀ (M - Mᵀ) X` by two gemms, `‖e‖²` by a gemv down to a pinned row, its root taken by `energy and norm` |
//! | the core guess, `ScfStep` over the pipeline's S, H and X | [`Run`], one plan from the XYZ file and the basis |

#![forbid(unsafe_code)]

mod iteration;
mod run;
mod tasks;

pub use iteration::{add_density, add_fock, occupied_orbitals};
pub use run::Run;
pub use tasks::add_solve_fock;
