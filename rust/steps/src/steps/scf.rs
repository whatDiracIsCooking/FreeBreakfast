//! The RHF loop over the fitted integrals, ROHF at a multiplicity above
//! 1, with Pulay DIIS on the Fock matrix: an all-GPU step through [`EriManager::build_jk`], its
//! eigensolve and everything else on gpu0's stream, under the manager's
//! lock and every GPU's (`docs/programming_model.md`, "The MainBus and its
//! steps", "The EriManager"; the prototype's `src/steps/scf.rs`).
//!
//! One iteration, a plain loop inside `run`: J and K of the carried D by
//! the manager, `F = H + J - K/2`, `E = tr(D (H + F))/2`, the residual
//! `e = Xᵀ (F D S - S D F) X` and its norm, the DIIS step over a slice of
//! the step's scratch, and the Roothaan step on the extrapolated F, which
//! leaves the next D. Then the step reads E, `‖e‖` and the two flags on the
//! host, through their downloads' token, and goes on or stops: the one
//! host read of a turn (`docs/gpu_layer.md`, "The v0.0.4 runtime"). On
//! convergence it solves once more on F as built, so the orbitals are
//! those of the density the energy belongs to.
//!
//! **What is an `Output` and what is scratch.** C, the orbital energies, D
//! and the energy outlive the `run`. F, J, K, the DIIS ring and syevd's
//! work buffer die with it: one region of gpu0's scratch, laid out by
//! [`layout`], and the host's reads one region of its pinned scratch.
//!
//! **K from a factor, every pass.** Every Roothaan step leaves the
//! orbitals C of its `D = 2 C Cᵀ`, so every J/K pass is given C's occupied
//! columns but a fresh run's first, whose D is the guess. That one is given
//! the guess's factor: `D = U diag(w) Uᵀ` by syevd on gpu0, and
//! `Y = U diag((w/2)^(1/2))` over the eigenvalues above [`GUESS_FLOOR`] of
//! the largest, so `D = 2 Y Yᵀ` less the dropped eigenvalues, and the pass
//! costs `rank / n_ao` of one from D. The rank, the width of that pass's
//! GEMMs, is read on the host once, before the loop's first turn, through
//! the eigenvalues' download: the loop's turn read, its turn the factor's
//! (`docs/gpu_layer.md`, "The v0.0.4 runtime"). The SAD guess is a sum of
//! atomic densities, so its rank is about its molecule's minimal basis:
//! gly10's is 237 of 734 AOs, beside 155 occupied orbitals. The energy is
//! still `tr(D (H + F))/2` of the guess's D.
//!
//! **An open shell** (`n_alpha > n_beta`, ROHF) changes the iteration's
//! J/K pass, F and its energy, and the density, and its published
//! orbitals' open block; the module `scf_open` has its equations. Its
//! J/K pass is given C's first `n_alpha` and first `n_beta` columns at
//! once, F is the Roothaan effective Fock matrix of the two spin Fock
//! matrices, and the residual, the DIIS step and the Roothaan step are the
//! closed shell's on it. Its scratch, the spin matrices, the second
//! density's J and K, the identity and the open block, follows the closed
//! shell's in the one layout, and its `Output` adds PySCF's effective Fock
//! matrix ([`ScfOutput::fock`]).
//!
//! **A replay** starts from the orbitals the last run left: C and D carry
//! over in the `Output`, and nothing in scratch does (v0.0.2's reset mode).
//!
//! [`EriManager::build_jk`]: crate::EriManager::build_jk

use std::time::{Duration, Instant};

use fb_core_budget::{Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, LockId, ManagerId, Span};
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, GpuResources, part, part_mut, sync_meter, wwr_blas_operation,
    wwr_blas_side_mode,
};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};
use fb_recipes_sys::FB_DIIS_SOLVE_MAX_DIM;

use crate::bus::{MainBus, ScfTable, Shared};
use crate::steps::GPU0;
use crate::steps::core_hamiltonian::CoreHamiltonianStep;
use crate::steps::eri::{Density, EriStep};
use crate::steps::guess::GuessStep;
use crate::steps::orthogonalizer::OrthogonalizerStep;
use crate::steps::overlap::OverlapStep;
use crate::steps::scf_open::{
    Open, OpenSpans, Pass, RoothaanScratch, canonical_open, pyscf_fock, roothaan_fock,
    spin_densities, spin_fock,
};

const OP_N: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
const OP_T: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
const LEFT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
const RIGHT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;

/// The guess's factor keeps the eigenvalues of D above this fraction of
/// the largest: the old path's and the C++ floor. The SAD density's
/// eigenvalues are its atoms' natural occupations, of order 1e-2 and up,
/// and round-off, under 1e-14 of the largest; the floor parts the two by
/// orders on either side, so the dropped eigenvalues are round-off and the
/// first pass's D is the guess's to round-off. Bounded however: at most
/// `n_ao` of them, each at most 1e-12 of the largest (2 for a closed-shell
/// atom), so at most `2e-12 n_ao` of the trace in all, 1.5e-9 at gly10,
/// a change in the first build's F and never in the converged energy,
/// which is the guess's fixed point to the loop's tolerance.
const GUESS_FLOOR: f64 = 1e-12;

/// The turn's read: the scalars' download token synced, after the turn's
/// last enqueue.
macro_rules! turn_read {
    ($e:expr) => {{
        #[allow(
            clippy::disallowed_methods,
            reason = "a loop deciding on the host reads its turn's scalars through their download's token, after the turn's last enqueue (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        let host = $e.sync();
        host
    }};
}

/// `[scf] stage_sync`'s sync of gpu0's stream at a stage end.
macro_rules! stage_sync {
    ($on:expr, $gpu:expr) => {
        if $on {
            #[allow(
                clippy::disallowed_methods,
                reason = "[scf] stage_sync, a measurement off by default, syncs gpu0's stream where an iteration would end a step were it three (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
            )]
            $gpu.stream().sync()?;
        }
    };
}

/// The step.
#[derive(Clone, Copy)]
pub struct ScfStep;

/// What the last run's loop took: a measurement, not a product.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopMeter {
    /// The loop's wall time, from its first J/K build to the converged
    /// orbitals on the host.
    pub wall: Duration,
    /// The stream syncs inside it, on every GPU.
    pub syncs: u64,
    /// The time blocked in them, summed over the GPUs' threads.
    pub blocked: Duration,
}

/// The converged run's products.
pub struct ScfOutput {
    energy: f64,
    residual: f64,
    iterations: usize,
    converged: bool,
    guess_rank: usize,
    meter: LoopMeter,
    c: DeviceBuf<f64>,
    epsilon: DeviceBuf<f64>,
    d: DeviceBuf<f64>,
    fock: Option<DeviceBuf<f64>>,
    c_host: HeapSlab<f64>,
    epsilon_host: HeapSlab<f64>,
    d_host: HeapSlab<f64>,
}

impl ScfOutput {
    /// The total energy, hartree: that of the density the last Fock matrix
    /// was built from.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// `‖Xᵀ (F D S - S D F) X‖_F` at that build.
    #[must_use]
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// The Fock builds of the last run.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The rank of the guess's factor, the columns of the first J/K pass of
    /// the last run that started from the guess; a replay leaves it.
    #[must_use]
    pub fn guess_rank(&self) -> usize {
        self.guess_rank
    }

    /// Whether the last run converged.
    #[must_use]
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The last run's loop, timed.
    #[must_use]
    pub fn meter(&self) -> LoopMeter {
        self.meter
    }

    /// The orbital energies on the host, ascending.
    #[must_use]
    pub fn epsilon(&self) -> &[f64] {
        &self.epsilon_host
    }

    /// The orbitals on the host, `n_ao²` column-major.
    #[must_use]
    pub fn c(&self) -> &[f64] {
        &self.c_host
    }

    /// The density on the host, `n_ao²` column-major.
    #[must_use]
    pub fn d(&self) -> &[f64] {
        &self.d_host
    }

    /// The density on gpu0, for a step that reads it there.
    #[must_use]
    pub fn d_device(&self) -> &DeviceBuf<f64> {
        &self.d
    }

    /// An open shell's effective Fock matrix on gpu0, `n_ao²`: PySCF's
    /// Roothaan matrix at the densities of the last build, whose
    /// occupation classes are C's columns `[0, n_beta)`, doubly occupied,
    /// `[n_beta, n_alpha)`, singly, and the rest (`Dims`); `None` for a
    /// closed shell.
    #[must_use]
    pub fn fock(&self) -> Option<&DeviceBuf<f64>> {
        self.fock.as_ref()
    }
}

/// The step's one layout on gpu0 and its pinned reads: `mem_budget`
/// declares its bytes, `run` carves its spans.
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
    h_w: Span<f64>,
    open: Option<OpenSpans>,
}

/// The layout at `n` AOs, a DIIS ring of `cap`, a work buffer of `lwork`
/// and, for an open shell, `m` singly occupied orbitals, whose tenants
/// follow the closed shell's.
fn layout(n: usize, cap: usize, lwork: usize, open: Option<usize>) -> ScfLayout {
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
        h_w: pin.span(n),
        open: open.map(|m| OpenSpans::new(&mut dev, &mut pin, n, m)),
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

/// The Roothaan step's matrices: the operator and X it reads, the scratch
/// it writes through, and the orbitals it leaves.
struct Roothaan<'a, F, X, T, O, E, W, I, C> {
    f: &'a F,
    x: &'a X,
    t: &'a mut T,
    f_ortho: &'a mut O,
    epsilon: &'a mut E,
    work: &'a mut W,
    info: &'a mut I,
    c: &'a mut C,
}

/// The Roothaan step at the operator `r.f`: `F' = X F X`, its eigensolve
/// and the orbitals `C = X C'`, on the GPU's stream.
fn roothaan<F, X, T, O, E, W, I, C>(
    gpu: &mut GpuResources,
    n: usize,
    r: Roothaan<'_, F, X, T, O, E, W, I, C>,
) -> Status
where
    F: DeviceSlice<f64>,
    X: DeviceSlice<f64>,
    T: DeviceSliceMut<f64>,
    O: DeviceSliceMut<f64>,
    E: DeviceSliceMut<f64>,
    W: DeviceSliceMut<f64>,
    I: DeviceSliceMut<i32>,
    C: DeviceSliceMut<f64>,
{
    // In range: mem_budget refused an n² past i32.
    let ni = n as i32;
    // X is symmetric, so Xᵀ F X is X (F X).
    gpu.gemm(OP_N, OP_N, ni, ni, ni, 1.0, r.f, ni, r.x, ni, 0.0, r.t, ni)?;
    gpu.gemm(
        OP_N, OP_N, ni, ni, ni, 1.0, r.x, ni, &*r.t, ni, 0.0, r.f_ortho, ni,
    )?;
    gpu.syevd(n, r.f_ortho, r.epsilon, r.work, r.info)?;
    gpu.gemm(
        OP_N,
        OP_N,
        ni,
        ni,
        ni,
        1.0,
        r.x,
        ni,
        &*r.f_ortho,
        ni,
        0.0,
        r.c,
        ni,
    )
}

/// A closed shell's density from its orbitals, `D = 2 C_occ C_occᵀ`, on
/// the GPU's stream.
fn closed_density(
    gpu: &mut GpuResources,
    n: usize,
    n_occ: usize,
    c: &impl DeviceSlice<f64>,
    d: &mut impl DeviceSliceMut<f64>,
) -> Status {
    // In range: mem_budget refused an n² past i32.
    let (ni, occ) = (n as i32, n_occ as i32);
    gpu.gemm(OP_N, OP_T, ni, ni, occ, 2.0, c, ni, c, ni, 0.0, d, ni)
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

    fn slot(bus: &MainBus) -> &Slot<ScfOutput> {
        &bus.scf
    }

    fn placement(_: &Shared) -> Placement {
        // It calls an all-GPU method, so it is an all-GPU step, though its
        // eigensolve runs on gpu0.
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<ScfOutput> {
        let n = shared.dims.n_ao;
        let options = shared
            .config
            .scf
            .clone()
            .ok_or_else(|| Error::config("the scf step runs with [scf]"))?;
        options.check()?;
        if n.checked_mul(n).is_none_or(|n2| i32::try_from(n2).is_err()) {
            return Err(Error::unsupported(format!(
                "scf: {n} AOs, whose n² is past the BLAS's i32"
            )));
        }
        // Every eigensolve of the loop is at n, and an open shell's
        // published orbitals add one at m: one work buffer for both.
        let m = open_width(&shared.dims);
        let lwork = match m {
            Some(m) => shared
                .syevd_lwork(budget, n)?
                .max(shared.syevd_lwork(budget, m)?),
            None => shared.syevd_lwork(budget, n)?,
        };
        let l = layout(n, options.diis_history, lwork, m);
        budget.scratch(GPU0, Kind::Device, l.device_bytes);
        budget.scratch(GPU0, Kind::Pinned, l.pinned_bytes);
        // The manager and every GPU; the wrapper takes them in rank order.
        budget.locks(
            std::iter::once(LockId::Manager(ManagerId::Eri))
                .chain((0..budget.gpus()).map(LockId::Gpu)),
        );
        Ok(ScfOutput {
            energy: 0.0,
            residual: 0.0,
            iterations: 0,
            converged: false,
            guess_rank: 0,
            meter: LoopMeter::default(),
            c: budget.device(GPU0, n * n),
            epsilon: budget.device(GPU0, n),
            d: budget.device(GPU0, n * n),
            fock: m.map(|_| budget.device(GPU0, n * n)),
            c_host: budget.heap(n * n),
            epsilon_host: budget.heap(n),
            d_host: budget.heap(n * n),
        })
    }

    fn run(
        out: &mut ScfOutput,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        if lane.gpus().is_empty() {
            return Err(Error::config("the scf step runs on a device"));
        }
        let options: &ScfTable = shared
            .config
            .scf
            .as_ref()
            .ok_or_else(|| Error::config("the scf step runs with [scf]"))?;
        let (n, n_occ) = (shared.dims.n_ao, shared.dims.n_beta);
        let (n_alpha, n_beta) = (shared.dims.n_alpha, shared.dims.n_beta);
        let m = open_width(&shared.dims);
        // In range: mem_budget refused an n² past i32.
        let (ni, n2) = (n as i32, n * n);
        let n2i = n2 as i32;
        let overlap = view.read(OverlapStep);
        let core = view.read(CoreHamiltonianStep);
        let orthogonalizer = view.read(OrthogonalizerStep);
        let eri = view.read(EriStep);
        let (s, h, x) = (overlap.s(), core.h(), orthogonalizer.x());

        let cap = options.diis_history;
        let lwork = match m {
            Some(m) => {
                let gpu = held.gpu(GPU0);
                gpu.syevd_lwork(n)?.max(gpu.syevd_lwork(m)?)
            }
            None => held.gpu(GPU0).syevd_lwork(n)?,
        };
        let mut l = layout(n, cap, lwork, m);
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
        // The turn's host ends, carved once and handed from token to token.
        let mut h_scalars = scratch.pinned(GPU0, l.h_scalars);
        let mut h_info = scratch.pinned(GPU0, l.h_info);
        let mut h_singular = scratch.pinned(GPU0, l.h_singular);
        let mut open = match l.open.take() {
            Some(spans) => Some(Open::carve(held.gpu(GPU0), scratch, spans, n)?),
            None => None,
        };

        // A fresh loop in the same buffers. The first run starts from the
        // guess and its factor; a replay from the C and D the last run left.
        let fresh = !out.converged;
        // The columns of `out.c` the next pass is given: the occupied
        // orbitals of `out.d` after any Roothaan step, so on a replay from
        // the start, and the guess's factor's rank on a fresh run's first.
        // An open shell's are its alpha factor's, its beta factor their
        // first `n_beta`; a closed shell's `n_alpha` is `n_beta`.
        let mut cols = n_alpha;
        let after_roothaan = if n_beta > 0 { Pass::Both } else { Pass::Alpha };
        let mut pass = if fresh {
            Pass::Spinless
        } else {
            after_roothaan
        };
        if fresh {
            let guess = view.read(GuessStep);
            let gpu = held.gpu(GPU0);
            // D = U diag(w) Uᵀ: U over t1, w into the orbital energies,
            // which the first Roothaan step overwrites.
            gpu.stream().copy_device(&mut out.d, guess.d())?;
            gpu.stream().copy_device(&mut t1, guess.d())?;
            gpu.syevd(n, &mut t1, &mut out.epsilon, &mut work, &mut info)?;
            let on = gpu.stream();
            let (read_w, read_info) = (
                on.download(scratch.pinned(GPU0, l.h_w), &out.epsilon)?,
                on.download(h_info, &info)?,
            );
            let h_w = turn_read!(read_w)?;
            h_info = turn_read!(read_info)?;
            if h_info[0] != 0 {
                return Err(Error::numerical(format!(
                    "scf: the guess's syevd: info = {}",
                    h_info[0]
                )));
            }
            // Ascending, so the kept eigenvalues are the last `rank`.
            let floor = GUESS_FLOOR * h_w.last().copied().unwrap_or(0.0).max(0.0);
            let rank = h_w.iter().filter(|&&w| w > floor).count();
            if rank == 0 {
                return Err(Error::numerical(
                    "scf: the guess density has no positive eigenvalue",
                ));
            }
            let first = n - rank;
            let dropped: f64 = h_w[..first].iter().map(|w| w.abs()).sum();
            log::info!(
                "scf: the guess's factor: rank {rank} of {n}, the dropped eigenvalues' |w| \
                 summing to {dropped:.1e}"
            );
            // In range: rank ≤ n, and n² fits an i32.
            let ri = rank as i32;
            // (w/2)^(1/2) as (w/2) (w/2)^(-1/2), over the kept tail, in F's
            // and t2's heads, then into ε's head; Y = U_tail diag of it,
            // into C's first `rank` columns.
            let tail = part(&out.epsilon, first, rank);
            let (mut half, mut root) = (part_mut(&mut t2, 0, rank), part_mut(&mut f, 0, rank));
            gpu.geam(
                OP_N, OP_N, ri, 1, 0.5, &tail, ri, 0.0, &tail, ri, &mut half, ri,
            )?;
            gpu.stream().copy_device(&mut root, &half)?;
            gpu.inverse_sqrt(&mut root, 0.5 * floor)?;
            let mut weights = part_mut(&mut out.epsilon, 0, rank);
            gpu.dgmm(LEFT, ri, 1, &half, ri, &root, &mut weights, ri)?;
            gpu.dgmm(
                RIGHT,
                ni,
                ri,
                &part(&t1, first * n, n * rank),
                ni,
                &part(&out.epsilon, 0, rank),
                &mut part_mut(&mut out.c, 0, n * rank),
                ni,
            )?;
            (cols, out.guess_rank) = (rank, rank);
        }
        if let Some(o) = open.as_mut() {
            // The spin densities of the first build: the guess's split
            // evenly, or the carried orbitals'.
            let gpu = held.gpu(GPU0);
            if fresh {
                let guess = view.read(GuessStep);
                gpu.geam(
                    OP_N,
                    OP_N,
                    ni,
                    ni,
                    0.5,
                    guess.d(),
                    ni,
                    0.0,
                    guess.d(),
                    ni,
                    &mut o.da,
                    ni,
                )?;
                gpu.stream().copy_device(&mut o.db, &o.da)?;
            } else {
                spin_densities(gpu, n, (n_alpha, n_beta), &out.c, o, &mut out.d)?;
            }
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
        let (start, (syncs, blocked)) = (Instant::now(), sync_meter());
        loop {
            let (slot, held_pairs) = diis.advance();
            // J and K of the carried D, from its factor: the manager's, into
            // this step's scratch.
            match open.as_mut() {
                // An open shell's alpha and beta factors, in one pass.
                Some(o) if pass == Pass::Both => eri.build_jk(
                    shared,
                    held,
                    &[
                        Density::Occupied(part(&out.c, 0, n * n_alpha), n_alpha),
                        Density::Occupied(part(&out.c, 0, n * n_beta), n_beta),
                    ],
                    &mut [part_mut(&mut j, 0, n2), part_mut(&mut o.j2, 0, n2)],
                    &mut [part_mut(&mut k, 0, n2), part_mut(&mut o.k2, 0, n2)],
                )?,
                _ => eri.build_jk(
                    shared,
                    held,
                    &[Density::Occupied(part(&out.c, 0, n * cols), cols)],
                    &mut [part_mut(&mut j, 0, n2)],
                    &mut [part_mut(&mut k, 0, n2)],
                )?,
            }
            let gpu = held.gpu(GPU0);
            stage_sync!(options.stage_sync, gpu);
            if let Some(o) = open.as_mut() {
                // Fa, Fb and their energy, then the Roothaan matrix, its
                // scratch the pass's J and K and what the residual
                // overwrites after it.
                spin_fock(
                    gpu,
                    n,
                    pass,
                    h,
                    (&j, &k),
                    &out.d,
                    o,
                    &mut t2,
                    &mut part_mut(&mut scalars, 0, 1),
                )?;
                roothaan_fock(
                    gpu,
                    n,
                    s,
                    [&o.fa, &o.fb, &o.da, &o.db, &o.identity],
                    RoothaanScratch {
                        fc: &mut j,
                        fo_half: &mut k,
                        pc: &mut o.j2,
                        po: &mut o.k2,
                        pv: &mut t1,
                        a: &mut f_ortho,
                        work: &mut t2,
                    },
                    &mut f,
                )?;
            } else {
                // F = H + J - K/2.
                gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, ni, 1.0, &j, ni, &mut t1, ni)?;
                gpu.geam(OP_N, OP_N, ni, ni, 1.0, &t1, ni, -0.5, &k, ni, &mut f, ni)?;
                // E_electronic = tr(D (H + F)) / 2, a dot of two n² vectors.
                gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, ni, 1.0, &f, ni, &mut t1, ni)?;
                gpu.gemv(
                    OP_T,
                    n2i,
                    1,
                    0.5,
                    &t1,
                    n2i,
                    &out.d,
                    0.0,
                    &mut part_mut(&mut scalars, 0, 1),
                )?;
            }
            // e = Xᵀ (F D S - S D F) X: with M = F D S, M - Mᵀ.
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &f, ni, &out.d, ni, 0.0, &mut t1, ni,
            )?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &t1, ni, s, ni, 0.0, &mut t2, ni,
            )?;
            gpu.geam(OP_N, OP_T, ni, ni, 1.0, &t2, ni, -1.0, &t2, ni, &mut t1, ni)?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, &t1, ni, x, ni, 0.0, &mut t2, ni,
            )?;
            gpu.gemm(
                OP_N, OP_N, ni, ni, ni, 1.0, x, ni, &t2, ni, 0.0, &mut error, ni,
            )?;
            gpu.gemv(
                OP_T,
                n2i,
                1,
                1.0,
                &error,
                n2i,
                &error,
                0.0,
                &mut part_mut(&mut scalars, 1, 1),
            )?;
            stage_sync!(options.stage_sync, gpu);

            // The DIIS step: F and the scaled residual into the ring's
            // column, the Gram of the held residuals, the bordered solve,
            // and F* = Σ c_i F_i.
            gpu.stream()
                .copy_device(&mut part_mut(&mut iterates, slot * n2, n2), &f)?;
            gpu.geam(
                OP_N,
                OP_N,
                n2i,
                1,
                diis.scale,
                &error,
                n2i,
                0.0,
                &error,
                n2i,
                &mut part_mut(&mut residuals, slot * n2, n2),
                n2i,
            )?;
            if held_pairs >= 2 {
                // In range: held_pairs and cap are under the solve's 65.
                let (m, ld) = (held_pairs as i32, cap as i32);
                gpu.gemm(
                    OP_T,
                    OP_N,
                    m,
                    m,
                    n2i,
                    1.0,
                    &part(&residuals, 0, n2 * held_pairs),
                    n2i,
                    &part(&residuals, 0, n2 * held_pairs),
                    n2i,
                    0.0,
                    &mut gram,
                    ld,
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
                    &part(&iterates, 0, n2 * held_pairs),
                    n2i,
                    &part(&coefficients, 0, held_pairs),
                    0.0,
                    &mut f_star,
                )?;
            } else {
                gpu.stream().copy_device(&mut f_star, &f)?;
                gpu.stream().zero(&mut singular)?;
            }
            stage_sync!(options.stage_sync, gpu);

            // The Roothaan step on the extrapolation: the next D.
            roothaan(
                gpu,
                n,
                Roothaan {
                    f: &f_star,
                    x,
                    t: &mut t1,
                    f_ortho: &mut f_ortho,
                    epsilon: &mut out.epsilon,
                    work: &mut work,
                    info: &mut info,
                    c: &mut out.c,
                },
            )?;
            match open.as_mut() {
                Some(o) => spin_densities(gpu, n, (n_alpha, n_beta), &out.c, o, &mut out.d)?,
                None => closed_density(gpu, n, n_occ, &out.c, &mut out.d)?,
            }
            (cols, pass) = (n_alpha, after_roothaan);

            // The turn's read, after its last enqueue: E, ‖e‖² and the two
            // flags, under one sync (the later tokens are covered by it).
            let on = gpu.stream();
            let (read_scalars, read_singular, read_info) = (
                on.download(h_scalars, &scalars)?,
                on.download(h_singular, &singular)?,
                on.download(h_info, &info)?,
            );
            h_scalars = turn_read!(read_scalars)?;
            h_singular = turn_read!(read_singular)?;
            h_info = turn_read!(read_info)?;
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
                if let (Some(o), Some(fock)) = (open.as_ref(), out.fock.as_mut()) {
                    // PySCF's Roothaan matrix at this build's densities,
                    // from F as built and its Po, before either is
                    // overwritten.
                    pyscf_fock(gpu, n, (&f, &o.k2), o, (&mut t1, &mut t2), fock)?;
                }
                // The orbitals of F as built, not of the extrapolation.
                roothaan(
                    gpu,
                    n,
                    Roothaan {
                        f: &f,
                        x,
                        t: &mut t1,
                        f_ortho: &mut f_ortho,
                        epsilon: &mut out.epsilon,
                        work: &mut work,
                        info: &mut info,
                        c: &mut out.c,
                    },
                )?;
                match open.as_mut() {
                    Some(o) => {
                        spin_densities(gpu, n, (n_alpha, n_beta), &out.c, o, &mut out.d)?;
                        // Fc of this build is still in J's tenant.
                        canonical_open(
                            gpu,
                            n,
                            (n_alpha, n_beta),
                            &j,
                            o,
                            &mut work,
                            &mut out.c,
                            &mut out.epsilon,
                        )?;
                    }
                    None => closed_density(gpu, n, n_occ, &out.c, &mut out.d)?,
                }
                let on = gpu.stream();
                let (h_c, h_epsilon, h_d) = (
                    on.download(scratch.pinned(GPU0, l.h_c), &out.c)?,
                    on.download(scratch.pinned(GPU0, l.h_epsilon), &out.epsilon)?,
                    on.download(scratch.pinned(GPU0, l.h_d), &out.d)?,
                );
                let read_info = on.download(h_info, &info)?;
                let read_open = match open.as_mut() {
                    Some(o) => Some(
                        on.download(
                            o.h_info
                                .take()
                                .expect("the open flag's pinned end, read once"),
                            &o.info,
                        )?,
                    ),
                    None => None,
                };
                // The solver's flag, with the orbitals it vouches for, read
                // on the host after the step's last enqueue: the solver-flag
                // site, which waits for no part of the step still to come.
                #[allow(
                    clippy::disallowed_methods,
                    reason = "a step reading a solver's flag after its last enqueue (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
                )]
                let (h_info, h_c, h_epsilon, h_d, h_open) = (
                    read_info.sync()?,
                    h_c.sync()?,
                    h_epsilon.sync()?,
                    h_d.sync()?,
                    read_open.map(|r| r.sync()).transpose()?,
                );
                if h_info[0] != 0 {
                    return Err(Error::numerical(format!(
                        "scf: the converged F's syevd: info = {}",
                        h_info[0]
                    )));
                }
                if let Some(h_open) = h_open.filter(|h| h[0] != 0) {
                    return Err(Error::numerical(format!(
                        "scf: the open block's syevd: info = {}",
                        h_open[0]
                    )));
                }
                out.c_host.copy_from_slice(&h_c);
                out.epsilon_host.copy_from_slice(&h_epsilon);
                out.d_host.copy_from_slice(&h_d);
                out.converged = true;
                let (now_syncs, now_blocked) = sync_meter();
                out.meter = LoopMeter {
                    wall: start.elapsed(),
                    syncs: now_syncs - syncs,
                    blocked: now_blocked.saturating_sub(blocked),
                };
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

/// An open shell's singly occupied orbitals, `n_alpha - n_beta`; `None`
/// for a closed shell.
fn open_width(dims: &crate::bus::Dims) -> Option<usize> {
    (dims.n_alpha > dims.n_beta).then(|| dims.n_alpha - dims.n_beta)
}

/// `[scf] diis_history`'s ceiling: the bordered solve's largest order less
/// its border.
pub(crate) const DIIS_HISTORY_MAX: usize = FB_DIIS_SOLVE_MAX_DIM as usize - 1;
