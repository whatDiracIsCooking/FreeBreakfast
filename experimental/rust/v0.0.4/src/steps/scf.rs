//! The closed-shell RHF loop over the fitted integrals, with Pulay DIIS on
//! the Fock matrix. An all-GPU step, through [`EriManager::build_jk`]; its
//! eigensolve and everything else run on gpu0's stream, under gpu0's lock.
//!
//! One iteration: J and K of the carried D by the manager, `F = H + J -
//! K/2`, `E = tr(D (H + F))/2`, the residual `e = Xᵀ (F D S - S D F) X`
//! and its norm, the DIIS step, and the Roothaan step on the extrapolated
//! F, which leaves the next D. Then the step waits for gpu0, reads E and
//! `‖e‖` on the host, and goes on or stops. On convergence it solves once
//! more on F as built, so the orbitals are those of the density the energy
//! belongs to.
//!
//! **What is an `Output` and what is scratch.** C, the orbital energies, D
//! and the energy outlive the `run`. F, J, K, the DIIS ring and syevd's
//! work buffer die with it: one region of gpu0's scratch, laid out by
//! [`layout`].
//!
//! **Crudest form.** K is built from D itself, with no occupied factor.
//! The loop is a plain loop inside `run`, not a condition loop in a
//! subflow: README.md says so.
//!
//! **A replay** starts from the orbitals the last run left: D carries
//! over in the `Output`, and nothing in scratch does.

use fb_core_error_handling::{Error, Result, Status};

use crate::budget::{Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, LockId, ManagerId, Span};
use crate::bus::{Held, Lane, MainBus, ScfTable, Shared};
use crate::device::{DeviceSlice, DeviceSliceMut, GpuResources, OP_N, OP_T, part, part_mut};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::GPU0;
use crate::steps::core_hamiltonian::CoreHamiltonianStep;
use crate::steps::eri::EriStep;
use crate::steps::guess::GuessStep;
use crate::steps::orthogonalizer::OrthogonalizerStep;
use crate::steps::overlap::OverlapStep;

/// The step.
#[derive(Clone, Copy)]
pub struct ScfStep;

/// The converged run's products.
pub struct ScfOutput {
    energy: f64,
    residual: f64,
    iterations: usize,
    converged: bool,
    c: DeviceBuf<f64>,
    epsilon: DeviceBuf<f64>,
    d: DeviceBuf<f64>,
    c_host: HeapSlab<f64>,
    epsilon_host: HeapSlab<f64>,
    d_host: HeapSlab<f64>,
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

    /// The Fock builds of the last run.
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

    /// The density on gpu0, for a step that reads it there.
    pub fn d_device(&self) -> &DeviceBuf<f64> {
        &self.d
    }
}

/// The step's one layout on gpu0: `mem_budget` declares its bytes, `run`
/// carves its spans.
struct ScfLayout {
    device_bytes: usize,
    pinned_bytes: usize,
    j: Span<f64>,
    k: Span<f64>,
    f: Span<f64>,
    f_star: Span<f64>,
    f_ortho: Span<f64>,
    error: Span<f64>,
    t1: Span<f64>,
    t2: Span<f64>,
    scalars: Span<f64>,
    info: Span<i32>,
    iterates: Span<f64>,
    residuals: Span<f64>,
    gram: Span<f64>,
    coefficients: Span<f64>,
    singular: Span<i32>,
    work: Span<f64>,
    h_scalars: Span<f64>,
    h_info: Span<i32>,
    h_singular: Span<i32>,
    h_c: Span<f64>,
    h_epsilon: Span<f64>,
    h_d: Span<f64>,
}

fn layout(n: usize, cap: usize, lwork: usize) -> ScfLayout {
    let n2 = n * n;
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    ScfLayout {
        j: dev.span(n2),
        k: dev.span(n2),
        f: dev.span(n2),
        f_star: dev.span(n2),
        f_ortho: dev.span(n2),
        error: dev.span(n2),
        t1: dev.span(n2),
        t2: dev.span(n2),
        scalars: dev.span(2),
        info: dev.span(1),
        iterates: dev.span(n2 * cap),
        residuals: dev.span(n2 * cap),
        gram: dev.span(cap * cap),
        coefficients: dev.span(cap),
        singular: dev.span(1),
        work: dev.span(lwork),
        h_scalars: pin.span(2),
        h_info: pin.span(1),
        h_singular: pin.span(1),
        h_c: pin.span(n2),
        h_epsilon: pin.span(n),
        h_d: pin.span(n2),
        device_bytes: dev.bytes(),
        pinned_bytes: pin.bytes(),
    }
}

/// The DIIS cursor: host state of one `run`. The history itself is a slice
/// of the step's scratch and lives exactly as long as the `run`.
struct Diis {
    cap: usize,
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

/// The Roothaan step at the operator `f`: `F' = X F X`, its eigensolve,
/// the orbitals `C = X C'` and the density `D = 2 C_occ C_occᵀ`. On the
/// GPU's stream; `t`, `f_ortho` and `work` are scratch.
#[allow(clippy::too_many_arguments, reason = "the step's matrices")]
fn roothaan(
    gpu: &mut GpuResources,
    n: usize,
    n_occ: usize,
    f: &impl DeviceSlice<f64>,
    x: &impl DeviceSlice<f64>,
    t: &mut impl DeviceSliceMut<f64>,
    f_ortho: &mut impl DeviceSliceMut<f64>,
    epsilon: &mut impl DeviceSliceMut<f64>,
    work: &mut impl DeviceSliceMut<f64>,
    info: &mut impl DeviceSliceMut<i32>,
    c: &mut impl DeviceSliceMut<f64>,
    d: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let ni = n as i32;
    // X is symmetric, so Xᵀ F X is X (F X).
    gpu.gemm(OP_N, OP_N, ni, ni, ni, 1.0, f, ni, x, ni, 0.0, t, ni)?;
    gpu.gemm(
        OP_N, OP_N, ni, ni, ni, 1.0, x, ni, &*t, ni, 0.0, f_ortho, ni,
    )?;
    gpu.syevd(n, f_ortho, epsilon, work, info)?;
    gpu.gemm(
        OP_N, OP_N, ni, ni, ni, 1.0, x, ni, &*f_ortho, ni, 0.0, c, ni,
    )?;
    gpu.gemm(
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
        // It calls an all-GPU method, so it is an all-GPU step, though its
        // eigensolve runs on gpu0.
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<ScfOutput> {
        let n = shared.dims.n_ao;
        let options = shared.config.scf.clone().unwrap_or_default();
        let cap = options.diis_history;
        if !(1..=64).contains(&cap) {
            return Err(Error::config(format!(
                "[scf] diis_history is {cap}, outside 1..=64"
            )));
        }
        // Every eigensolve of the loop is at n: one work buffer.
        let l = layout(n, cap, shared.syevd_lwork(n)?);
        budget.scratch(GPU0, Kind::Device, l.device_bytes);
        budget.scratch(GPU0, Kind::Pinned, l.pinned_bytes);
        // The manager and every GPU; the wrapper takes them in rank order.
        budget.locks(
            [LockId::Manager(ManagerId::Eri)]
                .into_iter()
                .chain((0..budget.gpus()).map(LockId::Gpu)),
        );
        Ok(ScfOutput {
            energy: 0.0,
            residual: 0.0,
            iterations: 0,
            converged: false,
            c: budget.device(GPU0, n * n),
            epsilon: budget.device(GPU0, n),
            d: budget.device(GPU0, n * n),
            c_host: budget.heap(n * n),
            epsilon_host: budget.heap(n),
            d_host: budget.heap(n * n),
        })
    }

    fn run(
        out: &mut ScfOutput,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        _: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let options: ScfTable = shared.config.scf.clone().unwrap_or_default();
        let (n, n_occ) = (shared.dims.n_ao, shared.dims.n_occ);
        let (ni, n2) = (n as i32, n * n);
        let n2i = n2 as i32;
        let overlap = view.read(OverlapStep);
        let core = view.read(CoreHamiltonianStep);
        let orthogonalizer = view.read(OrthogonalizerStep);
        let eri = view.read(EriStep);
        let (s, h, x) = (overlap.s(), core.h(), orthogonalizer.x());

        let cap = options.diis_history;
        let l = layout(n, cap, held.gpu(GPU0).syevd_lwork(n)?);
        let dev = |span| scratch.device(GPU0, span);
        let (mut j, mut k, mut f, mut f_star) = (dev(l.j), dev(l.k), dev(l.f), dev(l.f_star));
        let (mut f_ortho, mut error, mut t1, mut t2) =
            (dev(l.f_ortho), dev(l.error), dev(l.t1), dev(l.t2));
        let mut scalars = dev(l.scalars);
        let (mut iterates, mut residuals, mut gram, mut coefficients) = (
            dev(l.iterates),
            dev(l.residuals),
            dev(l.gram),
            dev(l.coefficients),
        );
        let mut work = dev(l.work);
        let mut info = scratch.device(GPU0, l.info);
        let mut singular = scratch.device(GPU0, l.singular);
        let h_scalars = scratch.pinned(GPU0, l.h_scalars);
        let h_info = scratch.pinned(GPU0, l.h_info);
        let h_singular = scratch.pinned(GPU0, l.h_singular);

        // A fresh loop in the same buffers. The first run starts from the
        // guess; a replay from the D the last run left in the `Output`.
        if !out.converged {
            let guess = view.read(GuessStep);
            held.gpu(GPU0).stream().copy_device(&mut out.d, guess.d())?;
        }
        out.converged = false;
        let mut diis = Diis {
            cap,
            size: 0,
            head: 0,
            scale: 1.0,
        };
        let mut last_energy = 0.0;
        let mut iteration = 0;
        loop {
            let (slot, held_pairs) = diis.advance();
            // J and K of the carried D: the manager's, into this step's
            // scratch.
            eri.build_jk(
                shared,
                held,
                &[part(&out.d, 0, n2)],
                &mut [part_mut(&mut j, 0, n2)],
                &mut [part_mut(&mut k, 0, n2)],
            )?;
            let gpu = held.gpu(GPU0);
            // F = H + J - K/2.
            gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, 1.0, &j, &mut t1)?;
            gpu.geam(OP_N, OP_N, ni, ni, 1.0, &t1, -0.5, &k, &mut f)?;
            // E_electronic = tr(D (H + F)) / 2, a dot of two n² vectors.
            gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, 1.0, &f, &mut t1)?;
            {
                let mut energy = part_mut(&mut scalars, 0, 1);
                gpu.gemv(OP_T, n2i, 1, 0.5, &t1, n2i, &out.d, 0.0, &mut energy)?;
            }
            // e = Xᵀ (F D S - S D F) X: with M = F D S, M - Mᵀ.
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &f, ni, &out.d, ni, 0.0, &mut t1, ni,
            )?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &t1, ni, s, ni, 0.0, &mut t2, ni,
            )?;
            gpu.geam(OP_N, OP_T, ni, ni, 1.0, &t2, -1.0, &t2, &mut t1)?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &t1, ni, x, ni, 0.0, &mut t2, ni,
            )?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, x, ni, &t2, ni, 0.0, &mut error, ni,
            )?;
            {
                let mut norm2 = part_mut(&mut scalars, 1, 1);
                gpu.gemv(OP_T, n2i, 1, 1.0, &error, n2i, &error, 0.0, &mut norm2)?;
            }
            gpu.stream().download(h_scalars, &scalars)?;

            // The DIIS step: F and the scaled residual into the ring's
            // column, the Gram of the held residuals, the bordered solve,
            // and F* = Σ c_i F_i.
            {
                let mut into = part_mut(&mut iterates, slot * n2, n2);
                gpu.stream().copy_device(&mut into, &f)?;
                let mut into = part_mut(&mut residuals, slot * n2, n2);
                gpu.geam(
                    OP_N, OP_N, n2i, 1, diis.scale, &error, 0.0, &error, &mut into,
                )?;
            }
            if held_pairs >= 2 {
                let m = held_pairs as i32;
                gpu.gemm(
                    OP_T, OP_N, m, m, n2i, 1.0, &residuals, n2i, &residuals, n2i, 0.0, &mut gram,
                    cap as i32,
                )?;
                gpu.diis_solve(
                    held_pairs,
                    cap,
                    slot,
                    &gram,
                    &mut coefficients,
                    &mut singular,
                )?;
                gpu.gemv(
                    OP_N,
                    n2i,
                    m,
                    1.0,
                    &iterates,
                    n2i,
                    &coefficients,
                    0.0,
                    &mut f_star,
                )?;
            } else {
                gpu.stream().copy_device(&mut f_star, &f)?;
                gpu.stream().zero(&mut singular)?;
            }
            gpu.stream().download(h_singular, &singular)?;

            // The Roothaan step on the extrapolation: the next D.
            roothaan(
                gpu,
                n,
                n_occ,
                &f_star,
                x,
                &mut t1,
                &mut f_ortho,
                &mut out.epsilon,
                &mut work,
                &mut info,
                &mut out.c,
                &mut out.d,
            )?;
            gpu.stream().download(h_info, &info)?;

            // The decision reads E and ‖e‖ on the host: the step waits for
            // gpu0 here, once per iteration.
            gpu.stream().sync()?;
            if h_info[0] != 0 {
                return Err(Error::numerical(format!(
                    "scf: syevd at iteration {iteration}: info = {}",
                    h_info[0]
                )));
            }
            let energy = h_scalars[0] + shared.e_nuc;
            let norm = h_scalars[1].sqrt();
            let change = energy - last_energy;
            log::info!(
                "scf: iteration {iteration:3}  E = {energy:.12}  dE = {change:+.3e}  |e| = \
                 {norm:.3e}"
            );
            let converged = iteration >= 1
                && change.abs() < options.energy_tolerance
                && norm < options.commutator_tolerance;
            (out.energy, out.residual, last_energy) = (energy, norm, energy);
            iteration += 1;
            out.iterations = iteration;

            if converged {
                // The orbitals of F as built, not of the extrapolation.
                roothaan(
                    gpu,
                    n,
                    n_occ,
                    &f,
                    x,
                    &mut t1,
                    &mut f_ortho,
                    &mut out.epsilon,
                    &mut work,
                    &mut info,
                    &mut out.c,
                    &mut out.d,
                )?;
                let (h_c, h_epsilon, h_d) = (
                    scratch.pinned(GPU0, l.h_c),
                    scratch.pinned(GPU0, l.h_epsilon),
                    scratch.pinned(GPU0, l.h_d),
                );
                let on = gpu.stream();
                on.download(h_info, &info)?;
                on.download(h_epsilon, &out.epsilon)?;
                on.download(h_c, &out.c)?;
                on.download(h_d, &out.d)?;
                on.sync()?;
                if h_info[0] != 0 {
                    return Err(Error::numerical(format!(
                        "scf: the converged F's syevd: info = {}",
                        h_info[0]
                    )));
                }
                out.c_host.copy_from_slice(h_c);
                out.epsilon_host.copy_from_slice(h_epsilon);
                out.d_host.copy_from_slice(h_d);
                out.converged = true;
                log::info!("scf: converged in {iteration} iterations, E = {energy:.12}");
                return Ok(());
            }
            if iteration >= options.max_iterations {
                return Err(Error::max_iterations_exceeded(format!(
                    "scf: not converged in {iteration} iterations (|dE| = {:.3e} against \
                     {:.1e}, |e| = {norm:.3e} against {:.1e})",
                    change.abs(),
                    options.energy_tolerance,
                    options.commutator_tolerance
                )));
            }
            if h_singular[0] != 0 {
                // The solve's pivot test is absolute and the Gram is
                // squares of the residuals: restart at the residual's
                // scale.
                log::info!("scf: DIIS subspace singular at iteration {iteration}; restarting");
                diis.reset(1.0 / norm);
            }
        }
    }
}
