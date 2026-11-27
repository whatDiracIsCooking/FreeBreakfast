//! Pipek-Mezey localization of the SCF's occupied orbitals: a gpu0 step
//! under gpu0's lock, its sweeps the kernels of `fb-localization-sys` and
//! its products two gemms a turn, on gpu0's guarded stream.
//!
//! **What it maximizes.** Over orthogonal `U` of one block of occupied
//! orbitals `C_b` (`n_ao × m`),
//!
//! ```text
//! L_p(U) = Σ_i Σ_A (Q^A_ii)^p
//! Q^A_ij = 1/2 Σ_{μ ∈ A} (C_μi W_μj + C_μj W_μi),   W = S C_b
//! ```
//!
//! the symmetrized Mulliken populations of atom A, which sum to `δ_ij`
//! over the atoms for S-orthonormal orbitals. `p = 2` is Pipek and Mezey's
//! (J. Chem. Phys. 90, 4916 (1989)), with an exact angle per pair; `p = 4`
//! is Knizia's IBO exponent (J. Chem. Theory Comput. 9, 4834 (2013), eq.
//! 4, the angle of Appendix D), a rotation kept only where it raises `L_4`.
//! Over Mulliken populations, not IAO ones: `mode = "ibo"` is refused.
//! An atom owns its shells' spherical AOs as one range, in atom order.
//!
//! **A sweep** visits every pair once, as `m - 1` rounds of disjoint pairs
//! of a round-robin tournament, each round three launches on the
//! population tensor and U ([`fb_localization_sys::pm_sweep`]).
//!
//! **One turn, a plain loop inside `run`.** Turn 0 builds the populations
//! of `C_b` as given and `U = I`; each later turn is one sweep. Every turn
//! then enqueues `L`, the turn's orbitals `T = C_b U`, and the sort key,
//! the diagonal of `G = Uᵀ diag(ε_b) U`, and reads `L`, G and T on the host
//! through their downloads' tokens: the loop's turn read
//! (`docs/gpu_layer.md`, "The v0.0.4 runtime"). Converged when a sweep
//! moves `L` by less than `functional_tolerance`; one orbital has no pair
//! and stops after turn 0.
//!
//! **The order.** `G_ii` is `⟨i|F|i⟩` of the localized orbital `i`: the
//! SCF's C diagonalizes the Fock matrix its last Roothaan step solved
//! within each block, `C_bᵀ F C_b = diag(ε_b)`, so `C_b U`'s is
//! `Uᵀ diag(ε_b) U`. The last turn's columns are stably sorted by it on
//! the host, copied into the host copy in that order, and on the device
//! from T into the `Output`'s block.
//!
//! **An open shell's key is the same diagonal.** ROHF's published C is
//! the eigenvectors of the effective Fock matrix it iterated on, its
//! singly occupied block rotated to diagonalize `Fc = (Fa + Fb)/2` there
//! with `ε` its eigenvalues (`ScfOutput`'s module doc), and PySCF's
//! Roothaan matrix ([`ScfOutput::fock`](crate::ScfOutput::fock)) is `Fc`
//! in both occupied diagonal blocks, so within each block `Uᵀ diag(ε_b) U`
//! is that matrix's `⟨i|F|i⟩`, to the residual's order, with no product
//! over F. It is also the order the C++ step and the old path's test held
//! the open shell to, `Σ_k (C_bᵀ S C_local)_ki² ε_k`;
//! `open_localization_reference.py` sorts by nothing, its test matching
//! orbitals one to one.
//!
//! **A block is a run of C's columns**, `[b0, b1)`, with its own layout,
//! after the previous block's in the step's one: a closed shell's one
//! block is the doubly occupied `[0, n_beta)`; an open shell's are
//! `[0, n_beta)` and `[n_beta, n_alpha)`, each localized among itself and
//! the first left out when it is empty. Its arrays are sized by the
//! block's width (`docs/programming_model.md`, "The MainBus and its
//! steps"). `L` before and after and the sweeps are summed over the
//! blocks.
//!
//! **Inputs.** S is the overlap's on gpu0; each block's C and orbital
//! energies come from the SCF's host copies, staged and uploaded.
//!
//! | Step | Runs on | Locks | `Output` | Scratch |
//! |---|---|---|---|---|
//! | [`LocalizationStep`] | gpu0 | gpu0 | `C_local` (`n_ao × n_alpha`) on the host, `L` before and after and the sweeps, summed over the blocks | each block's: its input, `W`, the population tensor and U with their pongs, the angles, the tables, G and T; their pinned ends |

use contracts::requires;
use fb_core_budget::{Budget, HeapSlab, Kind, Layout, Lease, LockId, Span};
use fb_core_device::{
    DeviceSlice, GpuResources, part, part_mut, wwr_blas_operation, wwr_blas_side_mode,
};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};
use fb_core_log::le;
use fb_localization_sys::{
    Angles, FB_PM_MAX_OCCUPIED, pm_functional, pm_identity, pm_populations, pm_schedule,
    pm_schedule_len, pm_sweep,
};

use crate::bus::{Dims, LocalizationTable, MainBus, Shared};
use crate::steps::GPU0;
use crate::steps::overlap::OverlapStep;
use crate::steps::scf::ScfStep;

const OP_N: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
const OP_T: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
const LEFT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;

/// The loop's turn read: the token synced, after the turn's last enqueue.
macro_rules! turn_read {
    ($e:expr) => {{
        #[allow(
            clippy::disallowed_methods,
            reason = "a loop deciding on the host reads its turn's results through their download's token, after the turn's last enqueue (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        let host = $e.sync();
        host
    }};
}

/// The step.
#[derive(Clone, Copy)]
pub struct LocalizationStep;

/// The localized orbitals and what the loop measured.
pub struct LocalizationOutput {
    c_host: HeapSlab<f64>,
    before: f64,
    after: f64,
    sweeps: usize,
    n_cols: usize,
}

impl LocalizationOutput {
    /// `C_local` on the host, `n_ao × n_cols` column-major, each block's
    /// columns ascending in `⟨i|F|i⟩`.
    #[must_use]
    pub fn c(&self) -> &[f64] {
        &self.c_host
    }

    /// The localized columns: `n_alpha`, a closed shell's `n_beta`.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// `L` of the orbitals as the SCF left them, summed over the blocks.
    #[must_use]
    pub fn functional_before(&self) -> f64 {
        self.before
    }

    /// `L` of the localized orbitals, summed over the blocks.
    #[must_use]
    pub fn functional_after(&self) -> f64 {
        self.after
    }

    /// The sweeps the last run took, summed over the blocks; 0 for one
    /// orbital in each.
    #[must_use]
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }
}

/// One block's layout at `n` AOs, `n_atom` atoms and width `m`: the
/// device scratch and its pinned ends, after what `dev` and `pin` hold.
/// `mem_budget` declares the layouts' bytes, `run` carves its spans: the
/// input's two the caller's, which stages it, the rest
/// [`localize_block`]'s.
struct BlockLayout {
    /// C's block, `n × m`, then its `m` orbital energies.
    input: Span<f64>,
    /// The input's pinned end.
    h_input: Span<f64>,
    spans: BlockSpans,
}

/// The tenants [`localize_block`] carves.
struct BlockSpans {
    w: Span<f64>,
    pops: Span<f64>,
    pops_pong: Span<f64>,
    u: Span<f64>,
    u_pong: Span<f64>,
    cos: Span<f64>,
    sin: Span<f64>,
    partner: Span<i32>,
    offsets: Span<i32>,
    schedule: Span<i32>,
    v: Span<f64>,
    /// `L`, then G, `m × m`.
    scalars: Span<f64>,
    t: Span<f64>,
    h_offsets: Span<i32>,
    h_schedule: Span<i32>,
    h_scalars: Span<f64>,
    h_t: Span<f64>,
    h_order: Span<i32>,
}

/// The block layout (the module doc's), appended to `dev` and `pin`.
fn block_layout(
    dev: &mut Layout,
    pin: &mut Layout,
    n: usize,
    n_atom: usize,
    m: usize,
) -> BlockLayout {
    let (nm, m2, tensor) = (n * m, m * m, n_atom * m * m);
    let input = dev.span(nm + m);
    let h_input = pin.span(nm + m);
    let spans = BlockSpans {
        w: dev.span(nm),
        pops: dev.span(tensor),
        pops_pong: dev.span(tensor),
        u: dev.span(m2),
        u_pong: dev.span(m2),
        cos: dev.span(m),
        sin: dev.span(m),
        partner: dev.span(m),
        offsets: dev.span(n_atom + 1),
        schedule: dev.span(pm_schedule_len(m)),
        v: dev.span(m2),
        scalars: dev.span(1 + m2),
        t: dev.span(nm),
        h_offsets: pin.span(n_atom + 1),
        h_schedule: pin.span(pm_schedule_len(m)),
        h_scalars: pin.span(1 + m2),
        h_t: pin.span(nm),
        h_order: pin.span(m),
    };
    BlockLayout {
        input,
        h_input,
        spans,
    }
}

/// What one block's loop measured.
struct BlockInfo {
    before: f64,
    after: f64,
    sweeps: usize,
}

/// The blocks of C's columns, `[b0, b1)`: the doubly occupied
/// `[0, n_beta)`, left out when empty, then an open shell's singly
/// occupied `[n_beta, n_alpha)`.
fn blocks(dims: &Dims) -> impl Iterator<Item = (usize, usize)> {
    [(0, dims.n_beta), (dims.n_beta, dims.n_alpha)]
        .into_iter()
        .filter(|(b0, b1)| b1 > b0)
}

/// Refuse what the step cannot localize: no occupied orbital, and a block
/// past the kernels' count or the BLAS's i32.
fn check_dims(dims: &Dims) -> Status {
    if dims.n_alpha == 0 {
        return Err(Error::unsupported(
            "localization: no occupied orbital to localize",
        ));
    }
    let n = dims.n_ao;
    let fits = |k: Option<usize>| k.is_some_and(|k| i32::try_from(k).is_ok());
    for (b0, b1) in blocks(dims) {
        let m = b1 - b0;
        if m > FB_PM_MAX_OCCUPIED as usize
            || !fits(n.checked_mul(m))
            || !fits(m.checked_mul(m).and_then(|m2| m2.checked_mul(dims.n_atom)))
        {
            return Err(Error::unsupported(format!(
                "localization: a block of {m} occupied orbitals over {n} AOs and {} atoms, \
                 past the kernels' {FB_PM_MAX_OCCUPIED} or the i32 the products take",
                dims.n_atom
            )));
        }
    }
    Ok(())
}

/// The block's inputs as handed to [`localize_block`]: the options, the
/// overlap, the uploaded input (C's block, `n × m`, then its `m` orbital
/// energies) and the block's place.
struct Block<'a, S, I> {
    options: &'a LocalizationTable,
    s: &'a S,
    input: &'a I,
    /// AOs.
    n: usize,
    /// The block's width.
    m: usize,
}

/// Localize one block whose C and orbital energies the caller has
/// uploaded into `block.input`, and write its sorted columns
/// into `c_host` (`n × m`): the module doc's
/// loop, on gpu0's guarded stream.
///
/// # Errors
/// An enqueue's or a copy's failure; `MaxIterationsExceeded` when
/// `max_sweeps` sweeps have not converged.
#[requires(
    le!(block.n * block.m, c_host.len()),
    "localize_block: c_host holds n × m"
)]
#[requires(
    le!(block.n * block.m + block.m, block.input.len()),
    "localize_block: input holds n × m and m"
)]
fn localize_block<S, I>(
    gpu: &mut GpuResources,
    shared: &Shared,
    scratch: &Lease<'_>,
    spans: BlockSpans,
    block: &Block<'_, S, I>,
    c_host: &mut [f64],
) -> Result<BlockInfo>
where
    S: DeviceSlice<f64>,
    I: DeviceSlice<f64>,
{
    let (n, m, n_atom) = (block.n, block.m, shared.dims.n_atom);
    let options = block.options;
    let exponent = options.exponent;
    // In range: check_dims refused an n m, an m² n_atom past i32.
    let (ni, mi) = (n as i32, m as i32);
    let dev = |span| scratch.device(GPU0, span);
    let c_b = part(block.input, 0, n * m);
    let eps = part(block.input, n * m, m);
    let (mut w, mut pops, mut pops_pong) = (dev(spans.w), dev(spans.pops), dev(spans.pops_pong));
    let (mut u, mut u_pong, mut v) = (dev(spans.u), dev(spans.u_pong), dev(spans.v));
    let (mut cos, mut sin) = (dev(spans.cos), dev(spans.sin));
    let mut partner = scratch.device(GPU0, spans.partner);
    let mut offsets_dev = scratch.device(GPU0, spans.offsets);
    let (mut scalars, mut t) = (dev(spans.scalars), dev(spans.t));

    // The atoms' AO offsets, straight into their pinned end: atom A's
    // start at its first shell's AO, the end at n.
    let basis = &shared.basis;
    let mut h_offsets = scratch.pinned(GPU0, spans.h_offsets);
    let n_shells = basis.nshells();
    for (a, o) in h_offsets.iter_mut().enumerate() {
        // In range: a is at most n_atom, a u32, and the offsets at most n.
        let shell = if a < n_atom {
            basis.atom_shell_offset(a as u32)
        } else {
            n_shells
        };
        *o = if shell < n_shells {
            basis.ao_offset_sphe(shell) as i32
        } else {
            n as i32
        };
    }
    // Each token is dropped: its pinned end is a tenant of the lease,
    // carved until the lease drops after the wrapper's step-end sync.
    let (schedule, uploaded) = pm_schedule(
        gpu.stream(),
        m,
        scratch.pinned(GPU0, spans.h_schedule),
        scratch.device(GPU0, spans.schedule),
    )?;
    drop(uploaded);
    // W = S C_b.
    gpu.gemm(
        OP_N, OP_N, ni, mi, ni, 1.0, block.s, ni, &c_b, ni, 0.0, &mut w, ni,
    )?;
    let uploaded = pm_populations(
        gpu.stream(),
        m,
        n,
        &c_b,
        &w,
        h_offsets,
        &mut offsets_dev,
        &mut pops,
    )?;
    drop(uploaded);
    pm_identity(gpu.stream(), m, &mut u)?;

    let (mut h_scalars, mut h_t) = (
        scratch.pinned(GPU0, spans.h_scalars),
        scratch.pinned(GPU0, spans.h_t),
    );
    let (mut before, mut last, mut sweeps) = (0.0, 0.0, 0);
    loop {
        if sweeps > 0 {
            pm_sweep(
                gpu.stream(),
                exponent,
                n_atom,
                &schedule,
                options.rotation_gain_threshold,
                (&mut pops, &mut pops_pong),
                (&mut u, &mut u_pong),
                Angles {
                    cos: &mut cos,
                    sin: &mut sin,
                    partner: &mut partner,
                },
            )?;
        }
        // L, T = C_b U, and G = Uᵀ (diag(ε_b) U) behind L.
        pm_functional(
            gpu.stream(),
            exponent,
            n_atom,
            m,
            &pops,
            &mut part_mut(&mut scalars, 0, 1),
        )?;
        gpu.gemm(
            OP_N, OP_N, ni, mi, mi, 1.0, &c_b, ni, &u, mi, 0.0, &mut t, ni,
        )?;
        gpu.dgmm(LEFT, mi, mi, &u, mi, &eps, &mut v, mi)?;
        gpu.gemm(
            OP_T,
            OP_N,
            mi,
            mi,
            mi,
            1.0,
            &u,
            mi,
            &v,
            mi,
            0.0,
            &mut part_mut(&mut scalars, 1, m * m),
            mi,
        )?;
        // The turn's read, after its last enqueue: one sync covers both.
        let on = gpu.stream();
        let (read_scalars, read_t) = (on.download(h_scalars, &scalars)?, on.download(h_t, &t)?);
        h_scalars = turn_read!(read_scalars)?;
        h_t = turn_read!(read_t)?;
        let l = h_scalars[0];
        if sweeps == 0 {
            (before, last) = (l, l);
            if m < 2 {
                break;
            }
        } else {
            let change = (l - last).abs();
            last = l;
            log::debug!("localization: sweep {sweeps:3}  L = {l:.12}  dL = {change:.3e}");
            if change < options.functional_tolerance {
                break;
            }
            if sweeps == options.max_sweeps {
                return Err(Error::max_iterations_exceeded(format!(
                    "localization: L still moved {change:e} in sweep {sweeps}"
                )));
            }
        }
        sweeps += 1;
    }

    // The last turn's columns, stably sorted by ⟨i|F|i⟩ = G_ii.
    let mut order = scratch.pinned(GPU0, spans.h_order);
    for (j, o) in order.iter_mut().enumerate() {
        // In range: j < m, which fits an i32.
        *o = j as i32;
    }
    let key = |j: i32| h_scalars[1 + j as usize * (m + 1)];
    order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
    for (j, &from) in order.iter().enumerate() {
        let from = from as usize;
        c_host[j * n..(j + 1) * n].copy_from_slice(&h_t[from * n..(from + 1) * n]);
    }
    log::info!("localization: converged in {sweeps} sweeps, L = {last:.12} (from {before:.12})");
    Ok(BlockInfo {
        before,
        after: last,
        sweeps,
    })
}

impl Step<MainBus> for LocalizationStep {
    const NAME: &'static str = "localization";
    type Output = LocalizationOutput;
    type Reads = (OverlapStep, ScfStep);

    fn slot(bus: &MainBus) -> &Slot<LocalizationOutput> {
        &bus.localization
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<LocalizationOutput> {
        let options = shared
            .config
            .localization
            .as_ref()
            .ok_or_else(|| Error::config("the localization step runs with [localization]"))?;
        options.check()?;
        let dims = &shared.dims;
        check_dims(dims)?;
        let n = dims.n_ao;
        let (mut dev, mut pin) = (Layout::new(), Layout::new());
        for (b0, b1) in blocks(dims) {
            block_layout(&mut dev, &mut pin, n, dims.n_atom, b1 - b0);
        }
        budget.scratch(GPU0, Kind::Device, dev.bytes());
        budget.scratch(GPU0, Kind::Pinned, pin.bytes());
        budget.locks([LockId::Gpu(GPU0)]);
        Ok(LocalizationOutput {
            c_host: budget.heap(n * dims.n_alpha),
            before: 0.0,
            after: 0.0,
            sweeps: 0,
            n_cols: dims.n_alpha,
        })
    }

    fn run(
        out: &mut LocalizationOutput,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        _: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let options = shared
            .config
            .localization
            .as_ref()
            .ok_or_else(|| Error::config("the localization step runs with [localization]"))?;
        let dims = &shared.dims;
        let n = dims.n_ao;
        let overlap = view.read(OverlapStep);
        let scf = view.read(ScfStep);
        let gpu = held.gpu(GPU0);
        (out.before, out.after, out.sweeps) = (0.0, 0.0, 0);
        let (mut dev, mut pin) = (Layout::new(), Layout::new());
        for (b0, b1) in blocks(dims) {
            let m = b1 - b0;
            let BlockLayout {
                input,
                h_input,
                spans,
            } = block_layout(&mut dev, &mut pin, n, dims.n_atom, m);
            // The block's C and orbital energies, from the SCF's host
            // copies; the token is dropped, its pinned end a tenant of the
            // lease.
            let mut staged = scratch.pinned(GPU0, h_input);
            staged[..n * m].copy_from_slice(&scf.c()[n * b0..n * b1]);
            staged[n * m..].copy_from_slice(&scf.epsilon()[b0..b1]);
            let mut input = scratch.device(GPU0, input);
            drop(gpu.stream().upload(&mut input, staged)?);
            let info = localize_block(
                gpu,
                shared,
                scratch,
                spans,
                &Block {
                    options,
                    s: overlap.s(),
                    input: &input,
                    n,
                    m,
                },
                &mut out.c_host[n * b0..n * b1],
            )?;
            out.before += info.before;
            out.after += info.after;
            out.sweeps += info.sweeps;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    //! The sweep body at the old crate's fixtures' bar (tier 4): PySCF's
    //! RHF S and occupied C of water (five orbitals, the schedule's
    //! padding) and borane (four) in cc-pVDZ, and PySCF's Pipek-Mezey of
    //! them over Mulliken populations at both exponents, `tests/data/pm_*.npy`,
    //! written by `tests/data/pm_reference.py`.
    //! A probe step on the MainBus uploads them and calls
    //! [`localize_block`], as [`LocalizationStep`] does with the SCF's.
    //!
    //! **The bars, by analysis, one run** (the old crate's): L before
    //! `1e-12` relative, `n_ao ε` on each population through a power of at
    //! most 4 and a sum of at most 15 terms, two decades over; L after
    //! `2e-12` absolute, both codes within their own last change of the
    //! maximum (here `functional_tolerance = 1e-13`, PySCF's `conv_tol =
    //! 1e-12`), each converging at least linearly.
    #![allow(
        clippy::disallowed_methods,
        reason = "a test reads its fixtures on the host, outside any step's run"
    )]
    #![allow(
        clippy::disallowed_macros,
        reason = "a test's host data, outside any step's run"
    )]

    use super::*;
    use std::path::{Path, PathBuf};

    use fb_core_budget::{Backing, DeviceBuf};
    use fb_core_flow::{Dag, Executor};

    use crate::bus::Config;

    /// The probe: the fixture's S and C uploaded, one block localized.
    #[derive(Clone, Copy)]
    pub(crate) struct ProbeStep;

    /// The probe's result.
    pub(crate) struct ProbeOutput {
        s: DeviceBuf<f64>,
        c_host: HeapSlab<f64>,
        info: Option<BlockInfo>,
    }

    fn data(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(file)
    }

    /// The case's name, from its XYZ file's stem.
    fn case(shared: &Shared) -> String {
        let xyz = &shared.config.molecule.xyz;
        xyz.file_stem().unwrap().to_string_lossy().into_owned()
    }

    fn fixture(file: &str) -> (Vec<f64>, Vec<usize>) {
        let a = fb_io_npy::read_npy(&data(&format!("{file}.npy"))).unwrap();
        (a.data, a.shape)
    }

    impl Step<MainBus> for ProbeStep {
        const NAME: &'static str = "localization probe";
        type Output = ProbeOutput;
        type Reads = ();

        fn slot(bus: &MainBus) -> &Slot<ProbeOutput> {
            &bus.localization_probe
        }

        fn placement(_: &Shared) -> Placement {
            Placement::Gpu0
        }

        fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<ProbeOutput> {
            let dims = &shared.dims;
            let (n, m) = (dims.n_ao, dims.n_beta);
            let (mut dev, mut pin) = (Layout::new(), Layout::new());
            block_layout(&mut dev, &mut pin, n, dims.n_atom, m);
            budget.scratch(GPU0, Kind::Device, dev.bytes());
            budget.scratch(GPU0, Kind::Pinned, pin.bytes() + n * n * 8 + 64);
            budget.locks([LockId::Gpu(GPU0)]);
            Ok(ProbeOutput {
                s: budget.device(GPU0, n * n),
                c_host: budget.heap(n * m),
                info: None,
            })
        }

        fn run(
            out: &mut ProbeOutput,
            view: &View<'_, MainBus, Self>,
            held: &mut Held<'_>,
            _: &Lane,
            scratch: &Lease<'_>,
        ) -> Status {
            let shared = view.shared();
            let (n, m) = (shared.dims.n_ao, shared.dims.n_beta);
            let name = case(shared);
            let (s, _) = fixture(&format!("pm_{name}_s"));
            let (c, shape) = fixture(&format!("pm_{name}_c"));
            assert_eq!(shape, [n, m], "{name}: the fixture's C");
            let (mut dev, mut pin) = (Layout::new(), Layout::new());
            let BlockLayout {
                input,
                h_input,
                spans,
            } = block_layout(&mut dev, &mut pin, n, shared.dims.n_atom, m);
            // S's staging, after the block's own pinned tenants.
            let s_span = pin.span::<f64>(n * n);
            let gpu = held.gpu(GPU0);
            let mut staged_s = scratch.pinned(GPU0, s_span);
            staged_s.copy_from_slice(&s);
            drop(gpu.stream().upload(&mut out.s, staged_s)?);
            // The fixture has no orbital energies: zero keeps the order.
            let mut staged = scratch.pinned(GPU0, h_input);
            staged[..n * m].copy_from_slice(&c);
            staged[n * m..].fill(0.0);
            let mut input = scratch.device(GPU0, input);
            drop(gpu.stream().upload(&mut input, staged)?);
            let info = localize_block(
                gpu,
                shared,
                scratch,
                spans,
                &Block {
                    options: shared.config.localization.as_ref().unwrap(),
                    s: &out.s,
                    input: &input,
                    n,
                    m,
                },
                &mut out.c_host,
            )?;
            out.info = Some(info);
            Ok(())
        }
    }

    /// L before and after of the case's fixture at `exponent`.
    fn localize(case: &str, exponent: i32) -> (f64, f64, usize) {
        let text = format!(
            "[molecule]\nxyz = \"{case}.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
             [localization]\nexponent = {exponent}\nfunctional_tolerance = 1e-13\n"
        );
        let config = Config::parse(&text, &data("")).unwrap();
        let mut dag = Dag::new();
        dag.add(ProbeStep);
        let plan = dag
            .build(MainBus::setup(config, Backing::Real).unwrap())
            .unwrap();
        let executor = Executor::new(2);
        plan.budget(&executor).unwrap();
        plan.execute(&executor).unwrap();
        let out = plan.output(ProbeStep);
        let info = out.info.as_ref().unwrap();
        (info.before, info.after, info.sweeps)
    }

    /// L before and after against PySCF's, for water's odd count and
    /// borane's even one at both exponents. Catches a wrong angle or sign
    /// in a rotation, the padding mishandled (water), a round that skips
    /// or repeats an orbital, and a sweep loop that stops short. Needs a
    /// GPU.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn the_sweeps_match_pyscf_at_both_exponents() {
        for case in ["h2o", "bh3"] {
            let (functional, _) = fixture(&format!("pm_{case}_functional"));
            for (k, p) in [2, 4].into_iter().enumerate() {
                let (before, after, sweeps) = localize(case, p);
                let (want_before, want_after) = (functional[2 * k], functional[2 * k + 1]);
                let miss_before = (before - want_before).abs() / want_before;
                let miss_after = (after - want_after).abs();
                println!(
                    "{case} p={p}: {sweeps} sweeps; |dL before| {miss_before:.1e} relative, \
                     |dL after| {miss_after:.1e}"
                );
                assert!(
                    miss_before <= 1e-12,
                    "{case} p={p}: L before {before} vs {want_before}"
                );
                assert!(
                    miss_after <= 2e-12,
                    "{case} p={p}: L after {after} vs {want_after}"
                );
                assert!(sweeps >= 1, "{case} p={p}: no sweep");
            }
        }
    }
}
