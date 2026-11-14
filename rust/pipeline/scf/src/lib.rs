//! `src/pipeline/scf` (module `free_breakfast.pipeline.scf`), its
//! closed-shell density-fitted path: restricted Hartree-Fock iterated to
//! self-consistency with Pulay DIIS on the Fock matrix, as tasks of a
//! [`Dag`](fb_core_planner::Dag) whose loop replays one iteration's
//! sub-DAG until its decision says converged. The C++
//! `src/pipeline/scf`'s `scf_loop` and `DfFockBuilder` are the reference
//! design, and its README "The step" the algorithm at length.
//!
//! [`add_scf`] is the step: the guess's factor before the loop, the body
//! (J and K over the fitted B, F, E and the residual, DIIS, the Roothaan
//! step, the density), the decision, and the converged F's orbitals after
//! it; its doc has each task. [`ScfStep`] holds the C++ step's options the
//! loop reads. [`Run`] is the whole run for the tests: an XYZ file, a basis
//! and a fitting set planned once, from the SAD guess, returning an
//! [`ScfOutput`] and the timing record.
//!
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
//! in the tests' plan of its orbitals, the Fock matrix of each SCF
//! iteration and the converged one in [`add_scf`], with no change to the
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
//! | `ScfStep::run`, `scf_loop` with a `DfFockBuilder` | [`add_scf`], the loop a planned sub-DAG and its decision a CPU node |
//! | `ScfStep`'s `max_iterations`, `energy_tolerance`, `commutator_tolerance`, `diis_history`, an option out of range `InvalidConfig` | [`ScfStep`], checked at planning |
//! | `fock_rebuild_interval`, `tau_j`, incremental builds, `DirectFockBuilder` | not ported: every DF build is full, and the direct ERIs are not ported |
//! | ROHF, `roothaan_fock`, `publish_open`, `molden` | not ported: an odd electron count is refused |
//! | `guess_exchange`: `syevd(D)`, `C̃ = V (Λ/2)^(1/2)` over the eigenvalues above 1e-12 of the largest, `df_exchange` over `n_alpha` columns at a time, summed | `syevd` on the guess, `guess factor` (the floor 1e-12 absolute, applied on the device by `fb_recipes_sys::inverse_sqrt`), the largest `nocc` columns the carried factor, the other `n - nocc` one K before the loop that `guess exchange` adds to the first iteration's and then zeroes |
//! | a loaded guess that is not PSD: its negative part left out, with a warning | left out, and the same warning from `guess check`, a CPU node reading the eigenvalues `guess factor` copies to a pinned row |
//! | the guess rescaled by `N / Σ Z` for an ion | not ported: it moves only the first iterate, and [`Run`] plans a neutral molecule |
//! | `previous = 0.0`, the loop's local | the pinned row `last build`, which the decision carries, with the last build's `‖e‖` and the iteration count beside E (`docs/memory.md`, "Nested plans") |
//! | `ScfStep`'s keys read from the config's `[scf]` table | [`ScfStep`] itself, deserialized by `fb-pipeline-config` |
//! | converged: `full && iteration > 1 && \|dE\| < energy_tolerance && norm < commutator_tolerance` | the decision's test, from its second iteration (index 1) on |
//! | on convergence, `solve_fock` on F as built, not on the extrapolation | the same after the loop, `published density` its D; the last body iteration's extrapolated step is computed and left |
//! | a singular extrapolation: logged, `diis.reset()`, the pair pushed again with every later residual scaled by `1/\|[F,D]\|` | logged, [`Diis::restart`](fb_recipes_diis::Diis::restart)`(1/‖e‖)` from the decision: the pair kept and every later residual scaled, the kept pair's move done by the next push before its own, so the next iteration extrapolates over two pairs as the C++'s does |
//! | `err_max_iterations_exceeded`, naming the tolerances | the loop's bound, `MaxIterationsExceeded` from the executor |
//! | `FBF_LOG_INFO` per iteration and on convergence | `log::info!` from the decision, the same lines |
//! | the core guess's orbitals, a test vehicle of the Roothaan step | not here: the tests build that plan themselves |

#![forbid(unsafe_code)]

mod iteration;
mod run;
mod scf;
mod tasks;

pub use iteration::{add_density, add_fock, occupied_orbitals};
pub use run::{Run, ScfOutput};
pub use scf::{ScfStep, add_scf};
pub use tasks::add_solve_fock;
