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
//! | D (or Da, Db) from the occupied columns | not ported: the density is the SCF's, and the core guess's orbitals need none yet |
//! | the core guess, `ScfStep` over the pipeline's S, H and X | [`Run`], one plan from the XYZ file and the basis |

mod run;
mod tasks;

pub use run::Run;
pub use tasks::add_solve_fock;
