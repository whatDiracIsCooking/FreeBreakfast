//! The closed-shell DF-RHF loop: one iteration's tasks as a loop's body,
//! replayed until its decision finds the energy and the residual converged.

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::{dgmm, geam};
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use fb_integrals_eri_density_fitting_jk::{add_df_coulomb, add_df_exchange};
use fb_molecule::{Molecule, nuclear_repulsion_energy};
use fb_pipeline_eri::DensityFitted;
use fb_recipes_diagonalize::add_diagonalize;
use fb_recipes_diis::{DIIS_MAX_HISTORY_LIMIT, Diis, add_diis};
use fb_recipes_sys::inverse_sqrt;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;
use wwr_sys::wwr_blas_side_mode::{WWR_BLAS_SIDE_LEFT as LEFT, WWR_BLAS_SIDE_RIGHT as RIGHT};

use super::iteration::{add_density, add_fock, add_published_density, occupied_orbitals};
use super::tasks::add_solve_fock;

/// An eigenvalue of the guess density at or under this, halved, is dropped
/// from the guess's factor: what it drops is the eigensolver's round-off
/// on a positive semidefinite D. The C++ `kDensityFactorFloor` is the same
/// 1e-12 relative to the largest eigenvalue; here it is absolute, the
/// kernel's floor, since the eigenvalues stay on the device. A SAD
/// density's largest is of order one, and its round-off `n ε` of that.
const GUESS_FACTOR_FLOOR: f64 = 0.5e-12;

/// The closed-shell SCF's options: the C++ `ScfStep`'s keys that the
/// density-fitted closed-shell loop reads, each at the C++ default
/// ([`ScfStep::default`]).
///
/// The C++ `fock_rebuild_interval` and `tau_j` are the integral-direct
/// builder's, which every DF build ignores, and `molden` a product writer:
/// none is ported.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfStep {
    /// The loop's bound: the most Fock builds before the run fails with
    /// `MaxIterationsExceeded`. Default 100; at least 1.
    pub max_iterations: usize,
    /// `|ΔE|` between two Fock builds under which the energy is converged,
    /// in hartree. Default 1e-10; non-negative.
    pub energy_tolerance: f64,
    /// `‖Xᵀ (F D S - S D F) X‖_F` under which the residual is converged.
    /// Default 1e-7; non-negative.
    pub commutator_tolerance: f64,
    /// The most (F, e) pairs DIIS extrapolates over. Default 8; 1 to
    /// [`DIIS_MAX_HISTORY_LIMIT`].
    pub diis_history: usize,
}

impl Default for ScfStep {
    fn default() -> ScfStep {
        ScfStep {
            max_iterations: 100,
            energy_tolerance: 1e-10,
            commutator_tolerance: 1e-7,
            diis_history: 8,
        }
    }
}

impl ScfStep {
    /// `InvalidConfig` naming the first option out of range, as the C++
    /// step refuses them.
    fn check(&self) -> Status {
        if self.max_iterations < 1 {
            return Err(Error::config(format!(
                "[scf] max_iterations is {}, below 1",
                self.max_iterations
            )));
        }
        let refused = |t: f64| t.is_nan() || t < 0.0;
        if refused(self.energy_tolerance) || refused(self.commutator_tolerance) {
            return Err(Error::config(format!(
                "[scf] energy_tolerance ({}) and commutator_tolerance ({}) must be non-negative",
                self.energy_tolerance, self.commutator_tolerance
            )));
        }
        if !(1..=DIIS_MAX_HISTORY_LIMIT).contains(&self.diis_history) {
            return Err(Error::config(format!(
                "[scf] diis_history is {}, outside 1..{DIIS_MAX_HISTORY_LIMIT}",
                self.diis_history
            )));
        }
        Ok(())
    }
}

/// Add the closed-shell DF-RHF of `molecule` to `dag` on GPU worker
/// `worker`, from the guess density `guess`, and return
/// `(scalars, [ε, C, D])`: a pinned row of two doubles, `[E, ‖e‖_F]` of the
/// last Fock build (E in hartree, electronic plus nuclear repulsion), and
/// device output rows of the `n` orbital energies ascending, the orbitals
/// (`n²` doubles, column-major, column `j` the orbital of `ε[j]`) and
/// `D = 2 C_occ C_occᵀ` (`n²`), all of the converged Fock matrix as built.
///
/// `h`, `s` and `x` are earlier tasks' outputs of `n²` doubles: the core
/// Hamiltonian, the overlap and `X = S^(-1/2)`. `guess` is the guess
/// density, `n²` doubles, symmetric positive semidefinite (a SAD guess;
/// `fb_pipeline_guess::add_guess_upload`'s row); the loop carries D in it,
/// so it holds the last iteration's density after the run. `eri` is the
/// density-fitted integrals of the same basis (`fb_pipeline_eri::add_df_eri`).
/// `lwork` is [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) at `n`.
///
/// **Before the loop**, the guess enters as its factor, as the C++ first
/// build does, since K is built from a factor of D, never from D: the
/// eigendecomposition `D = U diag(w) Uᵀ` ([`add_diagonalize`] on the rows
/// `"guess U"`, `"guess w"`), then **`guess factor`** (stream):
/// `C̃ = U diag((w/2)^(1/2))`, an eigenvalue at or under 1e-12 a zero
/// column, so `D = 2 C̃ C̃ᵀ` to round-off. The guess's rank is about its
/// minimal basis, above `nocc`, but the loop's factor row is `n × nocc`,
/// the width of every later K. So C̃'s `nocc` columns of the largest
/// eigenvalues become the carried factor `"C occ"`, and the other `n - nocc`
/// the row `"C rest"`, whose K ([`add_df_exchange`], once) the first
/// iteration adds to its own: the C++ takes the same factor `nocc` columns
/// at a time and sums their K. Then `diis history` ([`add_diis`], the rings
/// of `"DIIS F"`), and **`energy before`** (CPU): the decision's carried
/// last energy, 0 as the C++ starts it.
///
/// **The body**, one iteration: `df coulomb` (J of D) and `df exchange` (K
/// of the factor); **`guess exchange`** (stream), when `nocc < n`: K plus
/// the rest's K into `"K of D"`, then the rest's K zeroed, so only the first
/// iteration has it; [`add_fock`]'s `fock`, `residual` and `energy and
/// norm`; `diis push` ([`Diis::add_push`]) of (F, e); [`add_solve_fock`] on
/// the extrapolated F; [`add_density`], which writes D and the factor for
/// the next iteration. **The decision**, `converged` (CPU): converged when
/// `|ΔE| < energy_tolerance` and `‖e‖ < commutator_tolerance`, from the
/// second iteration on; otherwise a singular DIIS extrapolation restarts
/// the history ([`Diis::reset`]). Each iteration is logged at `info`.
///
/// **After the loop**, the published orbitals are the converged F's as
/// built, not of its extrapolation, as in the C++: [`add_solve_fock`] on
/// F, then **`published density`** (stream), D from C's first `nocc`
/// columns by one gemm.
///
/// # Errors
/// At planning, before anything is added: `InvalidConfig` for an option
/// out of range ([`ScfStep`]) or an electron count that is not a closed
/// shell ([`occupied_orbitals`]). Then the kernel library's load check
/// ([`add_diis`], [`add_df_exchange`]), as `GpuRuntime`; `dag` may hold
/// some of the tasks after it. At run time, the tasks' errors, and
/// `MaxIterationsExceeded` when the loop has not converged in
/// `max_iterations` builds.
///
/// # Panics
/// If `n²` does not fit an `i32`, the BLAS's integer, or as
/// [`add_df_exchange`]; or as [`Dag::add`], if a row is of another worker's
/// table.
#[allow(
    clippy::too_many_arguments,
    reason = "the step's inputs, each a row or a product of its own"
)]
#[allow(
    clippy::type_complexity,
    reason = "the step's four products as rows, in the crate's tuple style"
)]
#[track_caller]
pub fn add_scf(
    dag: &mut Dag<'_>,
    worker: usize,
    step: &ScfStep,
    molecule: &Molecule,
    n: usize,
    lwork: usize,
    eri: &DensityFitted,
    [h, s, x]: [RowId<Device, f64>; 3],
    guess: RowId<Device, f64>,
) -> Result<(RowId<Pinned, f64>, [RowId<Device, f64>; 3])> {
    step.check()?;
    let nocc = occupied_orbitals(molecule)?;
    assert!(
        n.checked_mul(n).is_some_and(|n2| i32::try_from(n2).is_ok()),
        "add_scf: n = {n}: n² does not fit BLAS's i32"
    );
    let naux = eri.aux.nao_sphe() as usize;
    let e_nuc = nuclear_repulsion_energy(molecule);

    // Before the loop: the guess's factor, the rest's K, DIIS's rings and
    // the decision's last energy.
    let (u, w) = add_diagonalize(dag, worker, "guess", n, lwork, guess);
    let factor = dag.add("guess factor", worker, GuessFactor { u, w, n, nocc });
    let rest_k = match factor.rest {
        Some(rest) => Some(add_df_exchange(
            dag,
            worker,
            n,
            naux,
            n - nocc,
            eri.b,
            rest,
        )?),
        None => None,
    };
    let diis = add_diis(dag, worker, "DIIS F", n * n, step.diis_history)?;
    let previous = dag.add("energy before", worker, EnergyBefore);

    // The body: one iteration.
    let body = dag.len();
    let j = add_df_coulomb(dag, worker, n, naux, eri.b, guess);
    let mut k = add_df_exchange(dag, worker, n, naux, nocc, eri.b, factor.c_occ)?;
    if let Some(rest) = rest_k {
        k = dag.add("guess exchange", worker, GuessExchange { k, rest, n });
    }
    let (f, e, scalars) = add_fock(dag, worker, n, e_nuc, h, j, k, guess, s, x);
    let (f_star, singular) = diis.add_push(dag, f, e);
    let (_, c_next) = add_solve_fock(dag, worker, n, lwork, f_star, x);
    add_density(dag, worker, n, nocc, c_next, guess, factor.c_occ);
    dag.add_loop(
        "converged",
        worker,
        body,
        step.max_iterations,
        Converged {
            scalars,
            singular,
            previous,
            diis,
            energy_tolerance: step.energy_tolerance,
            commutator_tolerance: step.commutator_tolerance,
        },
    );

    // After the loop: the converged F's own orbitals and their density.
    let (eps, c) = add_solve_fock(dag, worker, n, lwork, f, x);
    let d = add_published_density(dag, worker, n, nocc, c);
    Ok((scalars, [eps, c, d]))
}

/// GPU: `C̃ = U diag((w/2)^(1/2))`, its last `nocc` columns into the
/// carried factor and the rest into a row of their own.
struct GuessFactor {
    u: RowId<Device, f64>,
    w: RowId<Device, f64>,
    n: usize,
    nocc: usize,
}

#[derive(Clone, Copy)]
struct GuessFactorRows {
    half: RowId<Device, f64>,
    root: RowId<Device, f64>,
    weights: RowId<Device, f64>,
    factor: RowId<Device, f64>,
    c_occ: RowId<Device, f64>,
    rest: Option<RowId<Device, f64>>,
}

impl Task for GuessFactor {
    type Rows = GuessFactorRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.u);
        device.read(self.w);
        let n = self.n;
        GuessFactorRows {
            half: device.scratch("w/2", n),
            root: device.scratch("(w/2)^(-1/2)", n),
            weights: device.scratch("(w/2)^(1/2)", n),
            factor: device.scratch("guess factor", n * n),
            c_occ: device.output("C occ", n * self.nocc),
            rest: (self.nocc < n).then(|| device.output("C rest", n * (n - self.nocc))),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("nocc", self.nocc)]
    }

    fn run(
        &self,
        rows: &GuessFactorRows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let blas = worker.blas();
        // add_scf checked that n² fits an i32.
        let n = self.n as i32;
        let (u, w) = (device.get(self.u), device.get(self.w));
        let (mut half, mut root, mut weights, mut factor, mut c_occ) = (
            device.take_mut(rows.half),
            device.take_mut(rows.root),
            device.take_mut(rows.weights),
            device.take_mut(rows.factor),
            device.take_mut(rows.c_occ),
        );
        // Each call follows the one before on the stream. (w/2)^(1/2) as
        // (w/2) (w/2)^(-1/2): the kernel maps an eigenvalue at or under the
        // floor, or a negative one, to 0, and so drops its column.
        geam(blas, OP_N, OP_N, n, 1, 0.5, &w, n, 0.0, &w, n, &mut half, n)?;
        copy::d2d(&mut root, &half, worker)?;
        inverse_sqrt(worker, self.n, &mut root, GUESS_FACTOR_FLOOR)?;
        dgmm(blas, LEFT, n, 1, &half, n, &root, 1, &mut weights, n)?;
        dgmm(blas, RIGHT, n, n, &u, n, &weights, 1, &mut factor, n)?;
        // Ascending: the largest eigenvalues are the last nocc columns.
        let (head, tail) = split_at_mut(&mut factor, self.n * (self.n - self.nocc));
        copy::d2d(&mut c_occ, &tail, worker)?;
        if let Some(rest) = rows.rest {
            copy::d2d(&mut device.take_mut(rest), &head, worker)?;
        }
        Ok(())
    }
}

/// GPU, in the body: `K of D = K + K rest`, then `K rest = 0`, so the
/// rest of the guess's factor enters the first iteration's K alone.
struct GuessExchange {
    k: RowId<Device, f64>,
    rest: RowId<Device, f64>,
    n: usize,
}

impl Task for GuessExchange {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.k);
        device.write(self.rest);
        device.output("K of D", self.n * self.n)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        total: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let n = self.n as i32;
        let k = device.get(self.k);
        let (mut rest, mut total) = (device.take_mut(self.rest), device.take_mut(*total));
        // The zero follows the sum on the stream.
        geam(
            worker.blas(),
            OP_N,
            OP_N,
            n,
            n,
            1.0,
            &k,
            n,
            1.0,
            &rest,
            n,
            &mut total,
            n,
        )?;
        memset::zero(&mut rest, worker)
    }
}

/// CPU, before the loop: the decision's last energy, 0.
struct EnergyBefore;

impl Task for EnergyBefore {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("E before", 1)
    }

    fn run(
        &self,
        row: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        host.take_mut(*row)[0] = 0.0;
        Ok(())
    }
}

/// The loop's decision: the iteration logged, convergence tested, and a
/// singular DIIS subspace restarted.
struct Converged {
    scalars: RowId<Pinned, f64>,
    singular: RowId<Pinned, i32>,
    previous: RowId<Pinned, f64>,
    diis: Diis,
    energy_tolerance: f64,
    commutator_tolerance: f64,
}

impl Decision for Converged {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>) {
        host.read(self.scalars);
        host.read(self.singular);
        host.write(self.previous);
    }

    fn decide(&self, _: &(), host: &Bufs<'_, Pinned>, iteration: usize) -> Result<ControlFlow<()>> {
        let [energy, norm] = host.get(self.scalars)[..] else {
            unreachable!("the scalars row is [E, |e|]")
        };
        let previous = host.take_mut(self.previous);
        let delta = energy - previous[0];
        previous[0] = energy;
        log::info!(
            "scf: iteration {:3}  E = {energy:.12}  dE = {delta:+.3e}  |[F,D]| = {norm:.3e}",
            iteration + 1
        );
        if iteration > 0 && delta.abs() < self.energy_tolerance && norm < self.commutator_tolerance
        {
            log::info!(
                "scf: converged in {} iterations, E = {energy:.12}",
                iteration + 1
            );
            return Ok(ControlFlow::Break(()));
        }
        if host.get(self.singular)[0] != 0 && norm > 0.0 {
            // The push already returned F as built, the plain step; the
            // next push starts a fresh subspace.
            log::info!("scf: DIIS subspace singular at |[F,D]| = {norm:.3e}; restarted");
            self.diis.reset();
        }
        Ok(ControlFlow::Continue(()))
    }
}
