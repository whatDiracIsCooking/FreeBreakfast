//! The closed-shell RHF loop over the fitted integrals, with Pulay DIIS on
//! the Fock matrix: a looping step, each trip one Fock build.
//!
//! One trip, all of it on the library stream: J and K of the carried D by
//! the [`EriManager`](crate::steps::eri::EriManager), `F = H + J - K/2`,
//! `E = tr(D (H + F))/2`, the residual `e = Xᵀ (F D S - S D F) X` and its
//! norm, the DIIS push and extrapolation, and the Roothaan step on the
//! extrapolated F, which leaves the next D. Then the step waits for the
//! card, reads E and `‖e‖` on the host, and says `Again` or `Done`. On
//! convergence it solves once more on F as built, so the orbitals are those
//! of the density the energy belongs to, as the old loop's are.
//!
//! **Crudest form.** K is built from D itself, `Σ_Q B_Q D B_Q`, with no
//! occupied factor, so the old loop's guess factor is gone. The DIIS is the
//! old push without its kept-pair restart: a singular subspace is emptied,
//! and every later residual scaled by `1/‖e‖` as the old loop does.

use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::blas::{geam, gemm, gemv};
use fb_core_wwr::copy::d2d;
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use fb_recipes_sys::diis_solve;

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, ScfTable, Shared};
use crate::device::{DeviceResources, download};
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::core_hamiltonian::CoreHamiltonianStep;
use crate::steps::eri::EriStep;
use crate::steps::guess::GuessStep;
use crate::steps::orthogonalizer::OrthogonalizerStep;
use crate::steps::overlap::OverlapStep;
use crate::steps::{GPU0, OP_N, OP_T, syevd_workspace};

/// The step.
#[derive(Clone, Copy)]
pub struct ScfStep;

/// The DIIS history: the step's own, never shared with another solver's.
struct Diis {
    cap: usize,
    /// The iterates, one `n²` column per pair.
    iterates: DeviceBuf<f64>,
    /// The residuals, likewise, each scaled by `scale` as it entered.
    residuals: DeviceBuf<f64>,
    /// The residuals' Gram, `cap × cap`.
    gram: DeviceBuf<f64>,
    coefficients: DeviceBuf<f64>,
    singular: DeviceBuf<i32>,
    singular_host: PinnedBuf<i32>,
    // The cursor: host state.
    size: usize,
    head: usize,
    scale: f64,
}

impl Diis {
    fn reset(&mut self, scale: f64) {
        (self.size, self.head, self.scale) = (0, 0, scale);
    }

    /// The column the next pair takes, and the pairs held after it.
    fn advance(&mut self) -> (usize, usize) {
        if self.size < self.cap {
            self.size += 1;
            (self.size - 1, self.size)
        } else {
            let slot = self.head;
            self.head = (self.head + 1) % self.cap;
            (slot, self.cap)
        }
    }
}

/// The converged run's products, and everything the loop works in.
pub struct ScfOutput {
    // What readers and the driver take.
    energy: f64,
    residual: f64,
    iterations: usize,
    converged: bool,
    c: DeviceBuf<f64>,
    epsilon: DeviceBuf<f64>,
    d: DeviceBuf<f64>,
    c_host: PinnedBuf<f64>,
    epsilon_host: PinnedBuf<f64>,
    d_host: PinnedBuf<f64>,
    // The loop's own.
    j: DeviceBuf<f64>,
    k: DeviceBuf<f64>,
    f: DeviceBuf<f64>,
    f_star: DeviceBuf<f64>,
    f_ortho: DeviceBuf<f64>,
    error: DeviceBuf<f64>,
    t1: DeviceBuf<f64>,
    t2: DeviceBuf<f64>,
    /// `[E_electronic, ‖e‖²]` on the device, and their pinned copy.
    scalars: DeviceBuf<f64>,
    scalars_host: PinnedBuf<f64>,
    info: DeviceBuf<i32>,
    info_host: PinnedBuf<i32>,
    diis: Diis,
    last_energy: f64,
}

impl ScfOutput {
    /// The total energy, hartree: that of the density the last Fock matrix
    /// was built from.
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// `‖Xᵀ (F D S - S D F) X‖_F` at that build.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// The Fock builds.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Whether the loop converged.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The orbital energies on the host, ascending.
    pub fn epsilon(&self) -> &[f64] {
        &self.epsilon_host
    }

    /// The orbitals on the host, `n_ao²` column-major.
    pub fn c(&self) -> &[f64] {
        &self.c_host
    }

    /// The density on the host, `n_ao²` column-major.
    pub fn d(&self) -> &[f64] {
        &self.d_host
    }

    /// The density on the device, for a step that reads it there.
    pub fn d_device(&self) -> &DeviceBuf<f64> {
        &self.d
    }
}

/// The Roothaan step at the operator `f`: `F' = X F X`, its eigensolve, the
/// orbitals `C = X C'` and the density `D = 2 C_occ C_occᵀ`. On the library
/// stream; `t` and `f_ortho` are scratch.
#[allow(clippy::too_many_arguments, reason = "the step's matrices")]
fn roothaan(
    library: &mut DeviceResources,
    n: usize,
    n_occ: usize,
    f: &DeviceBuf<f64>,
    x: &DeviceBuf<f64>,
    t: &mut DeviceBuf<f64>,
    f_ortho: &mut DeviceBuf<f64>,
    epsilon: &mut DeviceBuf<f64>,
    info: &mut DeviceBuf<i32>,
    c: &mut DeviceBuf<f64>,
    d: &mut DeviceBuf<f64>,
) -> Status {
    let ni = n as i32;
    fn square(
        blas: &fb_core_wwr::blas::BlasHandle,
        a: &DeviceBuf<f64>,
        b: &DeviceBuf<f64>,
        out: &mut DeviceBuf<f64>,
        ni: i32,
    ) -> Status {
        gemm(
            blas, OP_N, OP_N, ni, ni, ni, 1.0, a, ni, b, ni, 0.0, out, ni,
        )
    }
    // X is symmetric, so Xᵀ F X is X (F X).
    square(library.blas(), f, x, t, ni)?;
    square(library.blas(), x, t, f_ortho, ni)?;
    library.syevd(ScfStep::NAME, n, f_ortho, epsilon, info)?;
    square(library.blas(), x, f_ortho, c, ni)?;
    gemm(
        library.blas(),
        OP_N,
        OP_T,
        ni,
        ni,
        n_occ as i32,
        2.0,
        &*c,
        ni,
        &*c,
        ni,
        0.0,
        d,
        ni,
    )
}

impl Step<MainBus> for ScfStep {
    const NAME: &'static str = "scf";
    const LOOPS: bool = true;
    type Output = ScfOutput;
    type Reads = (
        OverlapStep,
        CoreHamiltonianStep,
        OrthogonalizerStep,
        EriStep,
        GuessStep,
    );

    fn slot(bus: &MainBus) -> &Slot<ScfOutput, Lane> {
        &bus.scf
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<ScfOutput> {
        let n = shared.dims.n_ao;
        let options = shared.config.scf.clone().unwrap_or_default();
        let cap = options.diis_history;
        if !(1..=64).contains(&cap) {
            return Err(Error::config(format!(
                "[scf] diis_history is {cap}, outside 1..=64"
            )));
        }
        let square = || budget.device::<f64>(GPU0, n * n);
        let out = ScfOutput {
            energy: 0.0,
            residual: 0.0,
            iterations: 0,
            converged: false,
            c: square(),
            epsilon: budget.device(GPU0, n),
            d: square(),
            c_host: budget.pinned(GPU0, n * n),
            epsilon_host: budget.pinned(GPU0, n),
            d_host: budget.pinned(GPU0, n * n),
            j: square(),
            k: square(),
            f: square(),
            f_star: square(),
            f_ortho: square(),
            error: square(),
            t1: square(),
            t2: square(),
            scalars: budget.device(GPU0, 2),
            scalars_host: budget.pinned(GPU0, 2),
            info: budget.device(GPU0, 1),
            info_host: budget.pinned(GPU0, 1),
            diis: Diis {
                cap,
                iterates: budget.device(GPU0, n * n * cap),
                residuals: budget.device(GPU0, n * n * cap),
                gram: budget.device(GPU0, cap * cap),
                coefficients: budget.device(GPU0, cap),
                singular: budget.device(GPU0, 1),
                singular_host: budget.pinned(GPU0, 1),
                size: 0,
                head: 0,
                scale: 1.0,
            },
            last_energy: 0.0,
        };
        // Every eigensolve of the loop is at n.
        budget.workspace(GPU0, syevd_workspace(shared, n)?);
        Ok(out)
    }

    fn run(out: &mut ScfOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let shared = view.shared();
        let options: ScfTable = shared.config.scf.clone().unwrap_or_default();
        let (n, n_occ) = (shared.dims.n_ao, shared.dims.n_occ);
        let (ni, n2) = (n as i32, (n * n) as i32);
        let lane = view.lane().gpu();
        let overlap = view.read(OverlapStep);
        let core = view.read(CoreHamiltonianStep);
        let orthogonalizer = view.read(OrthogonalizerStep);
        let eri = view.read(EriStep);
        let (s, h, x) = (overlap.s(), core.h(), orthogonalizer.x());

        // A replay starts a fresh loop in the same buffers. Who clears what
        // between replays is an open question of the design; here the step
        // resets itself when it is entered after having converged.
        if out.converged {
            out.converged = false;
            out.iterations = 0;
            out.diis.reset(1.0);
        }
        let iteration = out.iterations;
        let (slot, held) = out.diis.advance();
        let scale = out.diis.scale;

        lane.library(shared.library(GPU0), |library| {
            if iteration == 0 {
                let guess = view.read(GuessStep);
                d2d(&mut out.d, guess.d(), library.stream())?;
            }
            // J and K of the carried D: the manager's, into this step's rows.
            eri.build_jk(library, &out.d, &mut out.j, &mut out.k)?;
            let blas = library.blas();
            let on = library.stream();
            let add = |alpha: f64, a: &DeviceBuf<f64>, beta: f64, b: &DeviceBuf<f64>, c: &mut _| {
                geam(blas, OP_N, OP_N, ni, ni, alpha, a, ni, beta, b, ni, c, ni)
            };
            let times = |a: &DeviceBuf<f64>, b: &DeviceBuf<f64>, c: &mut DeviceBuf<f64>| {
                gemm(blas, OP_N, OP_N, ni, ni, ni, 1.0, a, ni, b, ni, 0.0, c, ni)
            };
            // F = H + J - K/2.
            add(1.0, h, 1.0, &out.j, &mut out.t1)?;
            add(1.0, &out.t1, -0.5, &out.k, &mut out.f)?;
            // E_electronic = tr(D (H + F)) / 2, a dot of two n² vectors.
            add(1.0, h, 1.0, &out.f, &mut out.t1)?;
            let (mut energy, mut norm2) = split_at_mut(&mut out.scalars, 1);
            gemv(
                blas,
                OP_T,
                n2,
                1,
                0.5,
                &out.t1,
                n2,
                &out.d,
                1,
                0.0,
                &mut energy,
                1,
            )?;
            // e = Xᵀ (F D S - S D F) X: with M = F D S, M - Mᵀ.
            times(&out.f, &out.d, &mut out.t1)?;
            times(&out.t1, s, &mut out.t2)?;
            geam(
                blas,
                OP_N,
                OP_T,
                ni,
                ni,
                1.0,
                &out.t2,
                ni,
                -1.0,
                &out.t2,
                ni,
                &mut out.t1,
                ni,
            )?;
            times(&out.t1, x, &mut out.t2)?;
            times(x, &out.t2, &mut out.error)?;
            gemv(
                blas, OP_T, n2, 1, 1.0, &out.error, n2, &out.error, 1, 0.0, &mut norm2, 1,
            )?;
            download(&mut out.scalars_host, &out.scalars, on)?;

            // The DIIS push: F and the scaled residual into the ring's
            // column, the Gram of the held residuals, the bordered solve,
            // and F* = Σ c_i F_i.
            let diis = &mut out.diis;
            let column = slot * n * n;
            {
                let (_, mut tail) = split_at_mut(&mut diis.iterates, column);
                let (mut into, _) = split_at_mut(&mut tail, n * n);
                d2d(&mut into, &out.f, on)?;
                let (_, mut tail) = split_at_mut(&mut diis.residuals, column);
                let (mut into, _) = split_at_mut(&mut tail, n * n);
                geam(
                    blas, OP_N, OP_N, n2, 1, scale, &out.error, n2, 0.0, &out.error, n2, &mut into,
                    n2,
                )?;
            }
            let (m, cap) = (held as i32, diis.cap as i32);
            if held >= 2 {
                gemm(
                    blas,
                    OP_T,
                    OP_N,
                    m,
                    m,
                    n2,
                    1.0,
                    &diis.residuals,
                    n2,
                    &diis.residuals,
                    n2,
                    0.0,
                    &mut diis.gram,
                    cap,
                )?;
                diis_solve(
                    on,
                    held,
                    diis.cap,
                    slot,
                    &diis.gram,
                    &mut diis.coefficients,
                    &mut diis.singular,
                )?;
                gemv(
                    blas,
                    OP_N,
                    n2,
                    m,
                    1.0,
                    &diis.iterates,
                    n2,
                    &diis.coefficients,
                    1,
                    0.0,
                    &mut out.f_star,
                    1,
                )?;
            } else {
                d2d(&mut out.f_star, &out.f, on)?;
                memset::zero(&mut diis.singular, on)?;
            }
            download(&mut diis.singular_host, &diis.singular, on)?;

            // The Roothaan step on the extrapolation: the next D.
            roothaan(
                library,
                n,
                n_occ,
                &out.f_star,
                x,
                &mut out.t1,
                &mut out.f_ortho,
                &mut out.epsilon,
                &mut out.info,
                &mut out.c,
                &mut out.d,
            )?;
            download(&mut out.info_host, &out.info, library.stream())
        })?;

        // The decision reads E and ‖e‖ on the host: the step waits for the
        // card here, once per trip.
        lane.sync()?;
        if out.info_host[0] != 0 {
            return Err(Error::numerical(format!(
                "scf: syevd at iteration {iteration}: info = {}",
                out.info_host[0]
            )));
        }
        let energy = out.scalars_host[0] + shared.e_nuc;
        let norm = out.scalars_host[1].sqrt();
        let change = energy - out.last_energy;
        log::info!(
            "scf: iteration {iteration:3}  E = {energy:.12}  dE = {change:+.3e}  |e| = {norm:.3e}"
        );
        let converged = iteration >= 1
            && change.abs() < options.energy_tolerance
            && norm < options.commutator_tolerance;
        (out.energy, out.residual, out.last_energy) = (energy, norm, energy);
        out.iterations = iteration + 1;

        if converged {
            // The orbitals of F as built, not of the extrapolation.
            lane.library(shared.library(GPU0), |library| {
                roothaan(
                    library,
                    n,
                    n_occ,
                    &out.f,
                    x,
                    &mut out.t1,
                    &mut out.f_ortho,
                    &mut out.epsilon,
                    &mut out.info,
                    &mut out.c,
                    &mut out.d,
                )?;
                let on = library.stream();
                download(&mut out.info_host, &out.info, on)?;
                download(&mut out.epsilon_host, &out.epsilon, on)?;
                download(&mut out.c_host, &out.c, on)?;
                download(&mut out.d_host, &out.d, on)
            })?;
            lane.sync()?;
            if out.info_host[0] != 0 {
                return Err(Error::numerical(format!(
                    "scf: the converged F's syevd: info = {}",
                    out.info_host[0]
                )));
            }
            out.converged = true;
            log::info!(
                "scf: converged in {} iterations, E = {energy:.12}",
                out.iterations
            );
            return Ok(Next::Done);
        }
        if out.iterations >= options.max_iterations {
            return Err(Error::max_iterations_exceeded(format!(
                "scf: not converged in {} iterations (|dE| = {:.3e} against {:.1e}, |e| = \
                 {norm:.3e} against {:.1e})",
                out.iterations,
                change.abs(),
                options.energy_tolerance,
                options.commutator_tolerance
            )));
        }
        if out.diis.singular_host[0] != 0 {
            // The solve's pivot test is absolute and the Gram is squares of
            // the residuals: restart at the residual's scale, as the old
            // loop does (without its kept pair).
            log::info!("scf: DIIS subspace singular at iteration {iteration}; restarting");
            out.diis.reset(1.0 / norm);
        }
        Ok(Next::Again)
    }
}
