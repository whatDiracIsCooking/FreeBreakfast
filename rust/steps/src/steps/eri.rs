//! The density-fitted integrals: one all-GPU step whose `Output` is the
//! [`EriManager`], the metric's Cholesky factor L on every GPU, the fitted
//! tensor
//!
//! ```text
//! B[Q][μν] = Σ_P L⁻¹[Q][P] (P|μν)
//! ```
//!
//! on the host heap, n_aux rows of the n_col spherical pair columns, and
//! per GPU the ERI buffer with its pinned staging and its streams' state
//! (`docs/programming_model.md`, "The EriManager"; `docs/gpu_layer.md`,
//! "The v0.0.4 runtime"; the prototype's `src/steps/eri.rs`).
//!
//! **B is built over pairs.** A pair column of B depends on every P, so a
//! batch of P cannot be fitted alone; a batch of pairs can. The pair
//! batches, a class's run of pairs each, partition the pair columns
//! (checked when the list is made), and each GPU takes a contiguous range
//! of them, so it owns one contiguous range of columns in every row of B:
//! the rows are cut at the GPUs' column boundaries by `split_at_mut`, and
//! each GPU scatters into its own slices, safe code with no shared pointer.
//! Each GPU forms (μν|P) in its ring, `ssperi3<la, lb, lP>`, spherical in
//! all three indices, fits it against its own L, and downloads it.
//!
//! **The screen.** At a positive `[eri] schwarz_threshold`, each class's
//! records' dressed Schwarz bounds are formed on the GPU
//! ([`record_bound`]), read on the host, and summed per shell pair: a pair
//! whose sum is under the threshold has no cells, and its columns of B are
//! zero. The sum is at least `sqrt((mn|mn))` for every Cartesian `m`, `n`
//! of the pair, so a dropped column's fit is that small. B keeps every
//! pair's columns, at the unscreened bound `Dims` gives, so the budget does
//! not depend on the threshold.
//!
//! **The build's ring.** `[eri] ring` pair batches in flight per GPU, two
//! or one, each ring buffer on a stream of its own carrying its batch's
//! cells, its three-centre kernels and its download. The fit, a blocked
//! triangular solve, goes on the GPU's stream: it waits on the event the
//! ring stream recorded after the kernels, and the download waits on the
//! event recorded after the fit. The host waits on a ring buffer's
//! download token, which syncs that buffer's stream alone, before it
//! scatters that batch into B and writes the buffer's pinned bytes again.
//! The metric's factor goes on the GPU's stream before the loop, and
//! overlaps the first batch's kernels.
//!
//! **B is contracted over aux functions.** [`EriManager::build_jk`] runs a
//! J/K pass: each GPU contracts its own fixed share of B's rows, and the
//! fragments are reduced onto gpu0 through the host. Each aux batch goes
//! through one of the pass's ring buffers, each with a pinned end and an
//! upload stream of the manager's own: the host copies the batch's rows
//! into the buffer's pinned end once that buffer's last upload has run (a
//! wait on its token, its stream alone); the upload and the gather into
//! square matrices go on the buffer's stream, after the event the GPU's
//! stream recorded behind the contraction that last read the buffer; the
//! contraction goes on the GPU's stream, after the event recorded behind
//! the gather. So a batch's host copy and its upload overlap the batch
//! before's contraction. The pass joins its streams before it returns.
//!
//! **The ERI buffer** is one device buffer per GPU within `[eri]
//! mem_per_gpu`, charged at what it uses, with its pinned staging; both are
//! carved into tenants ([`DeviceBuf::tenant`], [`PinnedBuf::tenant`]).
//! During `eri`'s run they hold the build's ring; afterwards a pass's
//! gather tables, a copy of D, the J and K fragments, one aux batch's work
//! and the pass's ring.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use contracts::debug_requires;
use fb_core_budget::{
    Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, LockId, ManagerId, PinnedBuf, PinnedTenant,
    Span,
};
use fb_core_device::{
    CHOLESKY_TILE, DeviceSlice, Event, GpuResources, Part, PartMut, Pending, Stream, part,
    part_mut, wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode,
};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{GpuLane, Held, Lane, Placement, Slot, Step, View, fan_out};
use fb_core_log::{eq, le};
use fb_integrals_sys::{
    FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_PACKED_CELL_FIELDS, RECORD_F64_FIELDS,
    RECORD_I32_FIELDS, mmd_records, packed_gather, record_bound, three_center_spherical_class,
};

use crate::bus::{ClassDim, Dims, MainBus, Shared, for_each_pair};
use crate::steps::GPU0;
use crate::steps::metric::MetricStep;

/// `[eri] mem_per_gpu`'s default, bytes: each GPU's ERI buffer. A fixed
/// constant, not a fraction of the cap, so a device-free budget is exact.
/// A placeholder for a measured value.
pub const DEFAULT_MEM_PER_GPU: u64 = 2 << 30;

/// The most one pinned buffer of a J/K pass's ring holds: an aux batch of
/// B's rows on its way up. Page-locking costs about 0.9 s per GiB, and the
/// ring page-locks one for each buffer of its depth.
const PASS_STAGING_BYTES: usize = 128 << 20;

/// The most one pinned buffer of the build's ring holds: a fitted pair
/// batch on its way down.
const BUILD_STAGING_BYTES: usize = 64 << 20;

/// Whether a J/K pass from C makes `K += 2 M Mᵀ` a syrk into K's lower
/// triangle, mirrored once the pass is done, or a gemm: whichever measured
/// faster on each backend, read from the backend the kernel libraries were
/// built for (`rust/kernel_lib.rs`). gly10 cc-pVDZ, release, one run each
/// on one card in one session (#1858): on the RTX 3080 the syrk takes the
/// SCF loop from 3.180 s to 2.775 s an iteration (−13%, at half the
/// flops); on the gfx1200 rocBLAS's dsyrk takes it from 4.634 s to 5.342 s
/// (+15%), so HIP keeps the gemm.
const K_BY_SYRK: bool = matches!(env!("FB_KERNELS_BACKEND").as_bytes(), b"CUDA");

/// The fields of one aux record and of one cell, as the kernel reads them.
const AUX_FIELDS: usize = FB_THREE_CENTER_AUX_FIELDS as usize;
const CELL_FIELDS: usize = FB_THREE_CENTER_PACKED_CELL_FIELDS as usize;

/// The reason the screen's one host wait carries.
macro_rules! screen_wait {
    ($e:expr) => {{
        #[allow(
            clippy::disallowed_methods,
            reason = "the eri build's screen reads its record bounds through their download's token, on the first ring stream, before its first batch (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        let host = $e.sync();
        host
    }};
}

/// The reason every host wait in the rings carries.
macro_rules! ring_wait {
    ($e:expr) => {{
        #[allow(
            clippy::disallowed_methods,
            reason = "a manager's ring waits on one slot's copy before it reads or rewrites that slot's host end (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        let host = $e.sync();
        host
    }};
}

/// The step.
#[derive(Clone, Copy)]
pub struct EriStep;

/// A density a J/K pass contracts, on gpu0.
#[derive(Clone, Copy)]
pub enum Density<'a> {
    /// D itself, `n_ao × n_ao`, symmetric.
    Dense(Part<'a, f64>),
    /// The occupied orbitals C of `D = 2 C Cᵀ`, `n_ao × n_occ`
    /// column-major, and `n_occ`. K then costs `n_occ / n_ao` of what it
    /// does from D.
    Occupied(Part<'a, f64>, usize),
}

impl<'a> Density<'a> {
    fn matrix(self) -> Part<'a, f64> {
        match self {
            Density::Dense(m) | Density::Occupied(m, _) => m,
        }
    }

    fn n_occ(self) -> Option<usize> {
        match self {
            Density::Dense(_) => None,
            Density::Occupied(_, n_occ) => Some(n_occ),
        }
    }
}

/// One GPU's part of the manager's state.
struct GpuState {
    /// The ERI buffer.
    buf: DeviceBuf<u8>,
    /// Its pinned staging.
    pinned: PinnedBuf<u8>,
    /// The manager's own streams, one per buffer of a pass's ring, and its
    /// events: per build buffer two (kernels, fit), one after the records,
    /// per pass buffer two (gathered, contracted), and one for D and one
    /// after the contractions. Made on the first run.
    streams: Vec<Stream>,
    events: Vec<Event>,
}

/// The fitted integrals and what builds J and K from them: `eri`'s
/// `Output`. A reader declares it like any other, and a reader that calls
/// [`Self::build_jk`] also declares the manager's lock and every GPU's,
/// which the wrapper takes. B and L are immutable after `eri`; the state
/// behind the `Mutex` changes only under the manager's lock.
pub struct EriManager {
    n_aux: usize,
    n_col: usize,
    width: usize,
    batches: usize,
    depth: usize,
    jk_width: usize,
    l: Vec<DeviceBuf<f64>>,
    b: HeapSlab<f64>,
    state: Mutex<Vec<GpuState>>,
    /// The J/K passes made since B was built: zeroed by `eri`'s run,
    /// counted by [`Self::build_jk`] under the manager's lock.
    passes: AtomicUsize,
    /// The shell pairs the screen dropped in the last build.
    dropped: usize,
}

impl EriManager {
    /// The fitting functions: B's rows.
    #[must_use]
    pub fn n_aux(&self) -> usize {
        self.n_aux
    }

    /// The spherical pair columns: B's row length.
    #[must_use]
    pub fn n_col(&self) -> usize {
        self.n_col
    }

    /// B, `n_aux` rows of `n_col`, pair-fastest.
    #[must_use]
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// `gpu`'s copy of the metric's Cholesky factor, in the lower triangle.
    #[must_use]
    pub fn l(&self, gpu: usize) -> &DeviceBuf<f64> {
        &self.l[gpu]
    }

    /// The pair batch width `[eri] mem_per_gpu` gave, in pair columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The pair batches B was built in.
    #[must_use]
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The aux batch width of a J/K pass, in rows of B.
    #[must_use]
    pub fn jk_width(&self) -> usize {
        self.jk_width
    }

    /// The shell pairs `[eri] schwarz_threshold` dropped from B's build:
    /// their columns are zero.
    #[must_use]
    pub fn dropped_pairs(&self) -> usize {
        self.dropped
    }

    /// The J/K passes [`Self::build_jk`] has finished since B was last
    /// built: an SCF's iterations, when it is the one reader.
    #[must_use]
    pub fn passes(&self) -> usize {
        self.passes.load(Ordering::Relaxed)
    }

    /// J and K for each density, on gpu0, into the caller's `j` and `k`.
    /// Every GPU uploads and contracts its own aux batches through the
    /// pass's ring, and the fragments are reduced onto gpu0 through the
    /// host. It takes no lock: the GPU guards and the manager's come in
    /// `held`, taken by the wrapper; `shared` is the reader's.
    ///
    /// # Errors
    /// A launch's, a copy's or the library's failure.
    ///
    /// # Panics
    /// If the manager's state is held elsewhere: the caller did not declare
    /// the manager's lock.
    #[debug_requires(
        le!(1, d.len()) && le!(d.len(), densities(&shared.dims)),
        "one density, or an open shell's two: the pass's buffers are sized for them"
    )]
    #[debug_requires(
        eq!(j.len(), d.len()) && eq!(k.len(), d.len()),
        "one J and one K per density"
    )]
    #[debug_requires(
        held.taken().contains(&LockId::Manager(ManagerId::Eri)),
        "the caller declared the manager's lock"
    )]
    #[track_caller]
    pub fn build_jk(
        &self,
        shared: &Shared,
        held: &mut Held<'_>,
        d: &[Density<'_>],
        j: &mut [PartMut<'_, f64>],
        k: &mut [PartMut<'_, f64>],
    ) -> Status {
        let mut state = self
            .state
            .try_lock()
            .expect("the manager's state is reached under the manager's lock alone");
        let dims = &shared.dims;
        let n2 = dims.n_ao * dims.n_ao;
        let layout = jk_layout(dims, self.jk_width, self.depth);
        let mut gpus = held.all_gpus();
        let n_gpus = gpus.len();
        let shares: Vec<Range<usize>> = (0..n_gpus).map(|g| share(self.n_aux, g, n_gpus)).collect();

        // Each density, or its factor, reaches the other GPUs through the
        // host: a download from gpu0 on the manager's stream after the
        // caller's work, then one host end uploaded to every other GPU
        // under one token.
        if n_gpus > 1 {
            let ready = &state[GPU0].events[state[GPU0].events.len() - 2];
            ready.record(gpus[GPU0].stream())?;
            state[GPU0].streams[0].wait(ready)?;
            for (i, density) in d.iter().enumerate() {
                let m = density.matrix();
                let st0 = &mut state[GPU0];
                let h_d = st0
                    .pinned
                    .tenant(&layout.p.span::<f64>(layout.h_d[i], m.len()));
                let h_d = ring_wait!(st0.streams[0].download(h_d, &m)?)?;
                let mut here: Vec<_> = state
                    .iter_mut()
                    .skip(1)
                    .map(|st| st.buf.tenant(&layout.r.span::<f64>(layout.d[i], m.len())))
                    .collect();
                let mut copies = state[1].streams[0].upload(&mut here[0], h_d)?;
                for (st, dst) in state.iter().skip(2).zip(here.iter_mut().skip(1)) {
                    copies = copies.and_upload(&st.streams[0], dst)?;
                }
                drop(ring_wait!(copies)?);
            }
        }
        let b = &self.b[..];
        let work: Vec<_> = gpus.iter_mut().zip(state.iter_mut()).zip(&shares).collect();
        let (jk_width, l) = (self.jk_width, &layout);
        fan_out(work, |((gpu, st), share)| {
            let source = (gpu.gpu() == GPU0).then_some(d);
            gpu_pass(shared, b, share.clone(), jk_width, gpu, st, l, source, d)
        })?;

        // The reduction onto gpu0. One GPU: its fragments are the result,
        // copied on its stream. Several: downloaded on each GPU's manager
        // stream after its contractions, summed on the host in gpu0's
        // pinned staging, and uploaded.
        let gpu0 = &mut *gpus[GPU0];
        for (i, (j, k)) in j.iter_mut().zip(k.iter_mut()).enumerate() {
            if n_gpus == 1 {
                let st0 = &mut state[GPU0];
                let jj = st0.buf.tenant(&layout.r.span::<f64>(layout.j[i], n2));
                let kk = st0.buf.tenant(&layout.r.span::<f64>(layout.k[i], n2));
                gpu0.stream().copy_device(j, &jj)?;
                gpu0.stream().copy_device(k, &kk)?;
                continue;
            }
            let mut sums: Option<(PinnedTenant<f64>, PinnedTenant<f64>)> = None;
            for st in state.iter_mut() {
                let s = &st.streams[0];
                let (jj, kk) = (
                    st.buf.tenant(&layout.r.span::<f64>(layout.j[i], n2)),
                    st.buf.tenant(&layout.r.span::<f64>(layout.k[i], n2)),
                );
                let h_j = st.pinned.tenant(&layout.p.span::<f64>(layout.h_j[i], n2));
                let h_k = st.pinned.tenant(&layout.p.span::<f64>(layout.h_k[i], n2));
                let (h_j, h_k) = (s.download(h_j, &jj)?, s.download(h_k, &kk)?);
                let (h_j, h_k) = (ring_wait!(h_j)?, ring_wait!(h_k)?);
                match &mut sums {
                    None => sums = Some((h_j, h_k)),
                    Some((sj, sk)) => {
                        sj.iter_mut().zip(h_j.iter()).for_each(|(s, x)| *s += x);
                        sk.iter_mut().zip(h_k.iter()).for_each(|(s, x)| *s += x);
                    }
                }
            }
            let (sj, sk) = sums.expect("at least one GPU");
            let s0 = &state[GPU0].streams[0];
            let (sj, sk) = (s0.upload(j, sj)?, s0.upload(k, sk)?);
            drop((ring_wait!(sj)?, ring_wait!(sk)?));
        }
        self.passes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// One buffer's regions in order: a manager buffer's layout, whose tenants
/// are carved at a region's start with an exact length.
#[derive(Default)]
struct Regions(Vec<usize>);

impl Regions {
    /// A region of `len` elements of `U`; its index.
    fn add<U>(&mut self, len: usize) -> usize {
        self.0.push(len * size_of::<U>());
        self.0.len() - 1
    }

    /// The buffer's bytes.
    fn bytes(&self) -> usize {
        let mut l = Layout::new();
        for &b in &self.0 {
            let _ = l.span::<u8>(b);
        }
        l.bytes()
    }

    /// The first `len` elements of region `at`, as a span to carve.
    #[debug_requires(len * size_of::<U>() <= self.0[at], "the tenant fits its region")]
    fn span<U>(&self, at: usize, len: usize) -> Span<U> {
        let mut l = Layout::new();
        for &b in &self.0[..at] {
            let _ = l.span::<u8>(b);
        }
        l.span(len)
    }
}

/// A ring buffer of the build: its regions.
struct RingSlot {
    cells: usize,
    batch: usize,
    h_batch: usize,
    /// One pinned region per aux momentum: that momentum's cells.
    h_cells: Vec<usize>,
}

/// The ERI buffer and its staging during the build, at a pair batch width
/// of `w` columns and a ring of `depth`.
struct RingLayout {
    r: Regions,
    p: Regions,
    slots: Vec<RingSlot>,
}

/// The pairs of class `c` in one batch of `w` pair columns.
fn pairs_per_batch(d: &Dims, c: usize, w: usize) -> usize {
    (w / d.classes[c].sph()).clamp(1, d.classes[c].n_pair.max(1))
}

fn ring_layout(d: &Dims, w: usize, depth: usize) -> RingLayout {
    let per = |c: usize| pairs_per_batch(d, c, w);
    let classes = 0..d.classes.len();
    let columns = classes.clone().map(|c| per(c) * d.classes[c].sph()).max();
    let cells_of = |n_shells: usize| classes.clone().map(|c| per(c) * n_shells).max();
    let all_cells = cells_of(d.n_aux_shell).unwrap_or(0);
    let (mut r, mut p) = (Regions::default(), Regions::default());
    let slots = (0..depth)
        .map(|_| RingSlot {
            cells: r.add::<i32>(CELL_FIELDS * all_cells),
            batch: r.add::<f64>(d.n_aux * columns.unwrap_or(0)),
            h_batch: p.add::<f64>(d.n_aux * columns.unwrap_or(0)),
            h_cells: d
                .aux_shells_of_l
                .iter()
                .map(|&n| p.add::<i32>(CELL_FIELDS * cells_of(n).unwrap_or(0)))
                .collect(),
        })
        .collect();
    RingLayout { r, p, slots }
}

/// The ERI buffer and its staging during a J/K pass, at an aux batch width
/// of `w` rows and a ring of `depth`.
struct JkLayout {
    r: Regions,
    p: Regions,
    weights: usize,
    d: Vec<usize>,
    j: Vec<usize>,
    k: Vec<usize>,
    bd: usize,
    gamma: usize,
    tables: Vec<usize>,
    staging: Vec<usize>,
    squares: Vec<usize>,
    h_d: Vec<usize>,
    h_j: Vec<usize>,
    h_k: Vec<usize>,
    h_tables: Vec<usize>,
    h_packed: Vec<usize>,
}

/// The densities a J/K pass takes at most, and the pass's buffers are
/// sized for: an open shell's two spin factors, a closed shell's one.
fn densities(d: &Dims) -> usize {
    if d.n_alpha > d.n_beta { 2 } else { 1 }
}

fn jk_layout(d: &Dims, w: usize, depth: usize) -> JkLayout {
    let (n2, n_col) = (d.n_ao * d.n_ao, d.n_col);
    let (mut r, mut p) = (Regions::default(), Regions::default());
    let each =
        |r: &mut Regions| -> Vec<usize> { (0..densities(d)).map(|_| r.add::<f64>(n2)).collect() };
    let ring = |r: &mut Regions, len: usize, of_i32: bool| -> Vec<usize> {
        (0..depth)
            .map(|_| {
                if of_i32 {
                    r.add::<i32>(len)
                } else {
                    r.add::<f64>(len)
                }
            })
            .collect()
    };
    JkLayout {
        weights: r.add::<f64>(n_col),
        d: each(&mut r),
        j: each(&mut r),
        k: each(&mut r),
        bd: r.add::<f64>(w * n2),
        gamma: r.add::<f64>(w),
        tables: ring(&mut r, 2 * n_col, true),
        staging: ring(&mut r, w * n_col, false),
        squares: ring(&mut r, w * n2, false),
        h_d: each(&mut p),
        h_j: each(&mut p),
        h_k: each(&mut p),
        h_tables: ring(&mut p, 2 * n_col, true),
        h_packed: ring(&mut p, w * n_col, false),
        r,
        p,
    }
}

/// The step's scratch on one GPU, laid out: the shell table, the aux
/// records, each class's records tables, records and record bounds, the
/// pair prefixes the cells read, the Cholesky tile and flag, and the
/// gather weights. The bounds are laid out at every threshold, so the
/// budget does not depend on it.
struct ScratchLayout {
    device: Layout,
    pinned: Layout,
    shells: Span<f64>,
    h_shells: Span<f64>,
    aux: Span<f64>,
    h_aux: Span<f64>,
    tables: Vec<Span<i32>>,
    h_tables: Vec<Span<i32>>,
    rec_f: Vec<Span<f64>>,
    rec_i: Vec<Span<i32>>,
    bounds: Vec<Span<f64>>,
    h_bounds: Vec<Span<f64>>,
    h_prefix: Span<i32>,
    h_diagonal: Span<i32>,
    tile: Span<f64>,
    flag: Span<i32>,
    h_flag: Span<i32>,
    h_weights: Span<f64>,
}

fn scratch_layout(d: &Dims) -> ScratchLayout {
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    let n_shells = 2 * d.n_prim + 3 * d.n_shell;
    let table = |n_pair: usize| 3 * d.n_shell + 3 * n_pair + 2;
    ScratchLayout {
        shells: dev.span(n_shells),
        h_shells: pin.span(n_shells),
        aux: dev.span(AUX_FIELDS * d.n_aux_prim),
        h_aux: pin.span(AUX_FIELDS * d.n_aux_prim),
        tables: d
            .classes
            .iter()
            .map(|c| dev.span(table(c.n_pair)))
            .collect(),
        h_tables: d
            .classes
            .iter()
            .map(|c| pin.span(table(c.n_pair)))
            .collect(),
        rec_f: d
            .classes
            .iter()
            .map(|c| dev.span(RECORD_F64_FIELDS * c.n_rec))
            .collect(),
        rec_i: d
            .classes
            .iter()
            .map(|c| dev.span(RECORD_I32_FIELDS * c.n_rec))
            .collect(),
        bounds: d.classes.iter().map(|c| dev.span(c.n_rec)).collect(),
        h_bounds: d.classes.iter().map(|c| pin.span(c.n_rec)).collect(),
        h_prefix: pin.span(d.n_shell_pair + d.classes.len()),
        h_diagonal: pin.span(d.n_shell_pair),
        tile: dev.span(CHOLESKY_TILE * CHOLESKY_TILE),
        flag: dev.span(1),
        h_flag: pin.span(1),
        h_weights: pin.span(d.n_col),
        device: dev,
        pinned: pin,
    }
}

/// A pair batch: a run of one class's pairs, and where its columns lie in
/// B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Batch {
    class: usize,
    first_pair: usize,
    n_pair: usize,
    first_col: usize,
}

impl Batch {
    /// Its columns.
    fn cols(&self, d: &Dims) -> usize {
        self.n_pair * d.classes[self.class].sph()
    }
}

/// The pair batches at width `w`, class by class: a batch never spans two
/// classes. Checked to partition the pair columns as it is made.
fn batches(d: &Dims, w: usize) -> Vec<Batch> {
    let mut out = Vec::new();
    let mut col = 0;
    for (c, class) in d.classes.iter().enumerate() {
        let per = pairs_per_batch(d, c, w);
        for first_pair in (0..class.n_pair).step_by(per) {
            out.push(Batch {
                class: c,
                first_pair,
                n_pair: per.min(class.n_pair - first_pair),
                first_col: col + first_pair * class.sph(),
            });
        }
        col += class.n_pair * class.sph();
    }
    check_partition(d, &out);
    out
}

/// A class's place in the build's launch order, largest kernel stack
/// first: `(la + lb, la)`. At every aux momentum, `ssperi3`'s stack per lane
/// does not fall as the rank rises, nor, within a class, as the momentum
/// does (`tests::the_stack_rank_orders_ssperis_stack`).
fn stack_rank(class: &ClassDim) -> (i32, i32) {
    (class.la + class.lb, class.la)
}

/// That `batches` partition the pair columns: each opens where the one
/// before it ends, and the last ends at `n_col`. What lets the GPUs write
/// B at once, each its own columns.
///
/// # Panics
/// At the first batch that does not, naming it.
fn check_partition(d: &Dims, batches: &[Batch]) {
    let mut end = 0;
    for (i, b) in batches.iter().enumerate() {
        assert!(
            b.first_col == end,
            "eri: pair batch {i} opens at column {}, not {end}: the batches do not partition B's columns",
            b.first_col
        );
        end += b.cols(d);
    }
    assert!(
        end == d.n_col,
        "eri: the pair batches end at column {end}, not {}",
        d.n_col
    );
}

/// GPU `g` of `n`'s share of `q` items: the `g`-th of `n` contiguous
/// ranges, the first `q % n` one longer.
fn share(q: usize, g: usize, n: usize) -> Range<usize> {
    let at = |k: usize| k * (q / n) + k.min(q % n);
    at(g)..at(g + 1)
}

/// The class index of `(la, lb)` in the dimensions' ascending list.
fn class_of(d: &Dims, la: i32, lb: i32) -> usize {
    d.classes
        .iter()
        .position(|c| (c.la, c.lb) == (la, lb))
        .expect("every pair's class was counted")
}

impl EriStep {
    /// The least `[eri] mem_per_gpu` holds a batch at, bytes: the build's
    /// ring at a width of one pair, or a J/K pass's ring at aux batches of
    /// one row, whichever is larger. A byte under it, the plan is refused.
    #[must_use]
    pub fn least_mem_per_gpu(d: &Dims, ring: usize) -> u64 {
        let least = d.classes.iter().map(|c| c.sph()).max().unwrap_or(1);
        (ring_layout(d, least, ring).r.bytes()).max(jk_layout(d, 1, ring).r.bytes()) as u64
    }
}

/// The widest of `lo..=hi` that `fits`, `lo` when none past it does.
fn widest(lo: usize, hi: usize, fits: &dyn Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (lo, hi.max(lo));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

impl Step<MainBus> for EriStep {
    const NAME: &'static str = "eri";
    /// One stream per buffer of the build's ring.
    const STREAMS: usize = 2;
    type Output = EriManager;
    type Reads = (MetricStep,);

    fn slot(bus: &MainBus) -> &Slot<EriManager> {
        &bus.eri
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<EriManager> {
        let d = &shared.dims;
        let (q, n2) = (d.n_aux, d.n_ao * d.n_ao);
        let table = shared
            .config
            .eri
            .as_ref()
            .ok_or_else(|| Error::config("the eri step runs with [eri]"))?;
        let depth = table.ring;
        if !(1..=2).contains(&depth) {
            return Err(Error::config(format!(
                "[eri] ring = {depth}, which is neither 1 nor 2"
            )));
        }
        let threshold = table.schwarz_threshold;
        if !(threshold >= 0.0 && threshold.is_finite()) {
            return Err(Error::config(format!(
                "[eri] schwarz_threshold = {threshold}, which is not a finite number at least 0"
            )));
        }
        let limit = table.mem_per_gpu.unwrap_or(DEFAULT_MEM_PER_GPU);
        let needs = EriStep::least_mem_per_gpu(d, depth);
        if needs > limit {
            return Err(Error::config(format!(
                "[eri] mem_per_gpu = {limit} holds no batch: the ring at a width of one pair \
                 and a J/K pass at one aux row need {needs} bytes"
            )));
        }
        let n_gpus = budget.gpus();
        // The pair batch width: the widest whose ring fits the limit and
        // whose fitted batch fits one pinned buffer.
        let least = d.classes.iter().map(|c| c.sph()).max().unwrap_or(1);
        let most = d
            .classes
            .iter()
            .map(|c| c.n_pair * c.sph())
            .max()
            .unwrap_or(1)
            .min(i32::MAX as usize / q.max(1))
            .min(BUILD_STAGING_BYTES / (8 * q.max(1)))
            .max(least);
        let width = widest(least, most, &|w| {
            ring_layout(d, w, depth).r.bytes() as u64 <= limit
        });
        // The aux batch width: the widest whose pass fits the limit; never
        // past a share, nor past what one pinned buffer holds.
        let longest = (0..n_gpus).map(|g| share(q, g, n_gpus).len()).max();
        let most_p = longest
            .unwrap_or(1)
            .min(i32::MAX as usize / n2.max(1))
            .min(PASS_STAGING_BYTES / (8 * d.n_col.max(1)))
            .max(1);
        let jk_width = widest(1, most_p, &|p| {
            jk_layout(d, p, depth).r.bytes() as u64 <= limit
        });
        if (width == least && most > least) || (jk_width == 1 && most_p > 1) {
            log::warn!("step 'eri': [eri] mem_per_gpu = {limit} gives a batch width of one");
        }

        let sl = scratch_layout(d);
        let (ring, jk) = (ring_layout(d, width, depth), jk_layout(d, jk_width, depth));
        let mut l = Vec::new();
        let mut gpus = Vec::new();
        for gpu in 0..n_gpus {
            budget.scratch(gpu, Kind::Device, sl.device.bytes());
            budget.scratch(gpu, Kind::Pinned, sl.pinned.bytes());
            l.push(budget.device(gpu, q * q));
            // What is charged is what is used, never the limit.
            gpus.push(GpuState {
                buf: budget.device(gpu, ring.r.bytes().max(jk.r.bytes())),
                pinned: budget.pinned(gpu, ring.p.bytes().max(jk.p.bytes())),
                streams: Vec::new(),
                events: Vec::new(),
            });
        }
        budget.locks(
            std::iter::once(LockId::Manager(ManagerId::Eri)).chain((0..n_gpus).map(LockId::Gpu)),
        );
        Ok(EriManager {
            n_aux: q,
            n_col: d.n_col,
            width,
            batches: batches(d, width).len(),
            depth,
            jk_width,
            l,
            // B's home: one slab on the host heap, at the unscreened bound.
            b: budget.heap(d.n_col * q),
            state: Mutex::new(gpus),
            passes: AtomicUsize::new(0),
            dropped: 0,
        })
    }

    fn run(
        out: &mut EriManager,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let d = &shared.dims;
        if lane.gpus().is_empty() {
            return Err(Error::config("the eri step runs on a device"));
        }
        let metric = view.read(MetricStep);
        let all = batches(d, out.width);
        let n_gpus = lane.gpus().len();
        // Each GPU's contiguous run of batches, and so of columns.
        let runs: Vec<Range<usize>> = (0..n_gpus).map(|g| share(all.len(), g, n_gpus)).collect();
        let col_of = |i: usize| all.get(i).map_or(d.n_col, |b| b.first_col);
        // B's rows cut at the GPUs' column boundaries: each GPU's slices
        // are its own columns of every row.
        let mut rows: Vec<Vec<&mut [f64]>> = (0..n_gpus).map(|_| Vec::new()).collect();
        for row in out.b.chunks_mut(d.n_col.max(1)) {
            let mut rest = row;
            for (g, run) in runs.iter().enumerate() {
                let (mine, after) = rest.split_at_mut(col_of(run.end) - col_of(run.start));
                rows[g].push(mine);
                rest = after;
            }
        }
        let (width, depth) = (out.width, out.depth);
        // B is built again: no pass has read it yet.
        *out.passes.get_mut() = 0;
        let state = out.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let work: Vec<_> = held
            .all_gpus()
            .into_iter()
            .zip(lane.gpus())
            .zip(state.iter_mut())
            .zip(out.l.iter_mut())
            .zip(rows)
            .zip(&runs)
            .collect();
        // Whoever fans out, joins: every GPU's batches are in B, and its
        // streams joined, before this returns.
        let dropped = fan_out(work, |(((((gpu, lane), st), l), rows), run)| {
            let mine = &all[run.clone()];
            let first = col_of(run.start);
            gpu_build(
                shared,
                &metric.gpus()[lane.gpu()],
                gpu,
                lane,
                st,
                l,
                scratch,
                (width, depth),
                mine,
                (rows, first),
            )
        })?;
        // Every GPU screens every pair alike; gpu0's count is the build's.
        out.dropped = dropped[GPU0];
        if out.dropped > 0 {
            log::info!(
                "step 'eri': [eri] schwarz_threshold drops {} of {} shell pairs",
                out.dropped,
                d.n_shell_pair
            );
        }
        Ok(())
    }
}

/// Scatter a fitted batch, `n_aux × cols` aux fastest, into `rows` (each
/// row's slice of this GPU's columns, from column `first`) at the batch's
/// columns: the transpose, in 64 × 64 tiles so both ends stay in cache.
fn scatter(rows: &mut [&mut [f64]], first: usize, cols: usize, fitted: &[f64]) {
    const TILE: usize = 64;
    let q = rows.len();
    for c0 in (0..cols).step_by(TILE) {
        let c1 = (c0 + TILE).min(cols);
        for q0 in (0..q).step_by(TILE) {
            for (qi, row) in rows
                .iter_mut()
                .enumerate()
                .take((q0 + TILE).min(q))
                .skip(q0)
            {
                for c in c0..c1 {
                    row[first + c] = fitted[qi + q * c];
                }
            }
        }
    }
}

/// One GPU's part of the build: its records, its screen, its factor, its
/// pair batches through the ring into its columns of B, and the gather
/// weights a J/K pass reads. Returns the shell pairs the screen dropped,
/// of all of them.
#[allow(
    clippy::too_many_arguments,
    reason = "a step's inner task, spelled out"
)]
fn gpu_build(
    shared: &Shared,
    metric: &DeviceBuf<f64>,
    gpu: &mut GpuResources,
    lane: &GpuLane,
    st: &mut GpuState,
    l: &mut DeviceBuf<f64>,
    scratch: &Lease<'_>,
    (width, depth): (usize, usize),
    mine: &[Batch],
    (mut rows, first_col): (Vec<&mut [f64]>, usize),
) -> Result<usize> {
    let g = lane.gpu();
    let d = &shared.dims;
    let basis = &shared.basis;
    let aux = shared
        .aux
        .as_ref()
        .ok_or_else(|| Error::config("the eri step runs with [eri]'s fitting set"))?;
    let (q, n_shell, n_prim) = (d.n_aux, d.n_shell, d.n_prim);
    let sl = scratch_layout(d);
    let streams = lane.streams();
    let s0 = &streams[0];
    let physical = gpu.physical();
    if st.events.is_empty() {
        for _ in 0..2 * depth + 1 + 2 * depth + 2 {
            st.events.push(Event::new(physical));
        }
        for _ in 0..depth {
            st.streams.push(Stream::new(physical));
        }
    }

    // The shell table and the aux records, built in pinned scratch and
    // uploaded on the first stream.
    let mut h_shells = scratch.pinned(g, sl.h_shells);
    let mut prim = 0;
    for (s, shell) in basis.shells().iter().enumerate() {
        let c = shell.center();
        for p in shell.primitives() {
            h_shells[prim] = p.exponent;
            h_shells[n_prim + prim] = p.coefficient;
            prim += 1;
        }
        h_shells[2 * n_prim + s] = c.x;
        h_shells[2 * n_prim + n_shell + s] = c.y;
        h_shells[2 * n_prim + 2 * n_shell + s] = c.z;
    }
    let mut shells = scratch.device(g, sl.shells);
    drop(s0.upload(&mut shells, h_shells)?);
    let mut h_aux = scratch.pinned(g, sl.h_aux);
    let n_aux_rec = d.n_aux_prim;
    // Each aux shell's record range and first function, and the shells of
    // each momentum: O(aux shells) of setup-sized host scratch.
    let mut aux_rec = Vec::new();
    let mut aux_sph = Vec::new();
    let mut aux_of_l: Vec<Vec<usize>> = d.aux_shells_of_l.iter().map(|_| Vec::new()).collect();
    let mut r = 0;
    aux_rec.push(0i32);
    for (s, shell) in aux.shells().iter().enumerate() {
        let c = shell.center();
        for p in shell.primitives() {
            let weight = p.coefficient * (std::f64::consts::PI / p.exponent).powf(1.5);
            for (f, v) in [c.x, c.y, c.z, p.exponent, weight].into_iter().enumerate() {
                h_aux[f * n_aux_rec + r] = v;
            }
            r += 1;
        }
        aux_rec.push(r as i32);
        aux_sph.push(aux.ao_offset_sphe(s as u32) as i32);
        aux_of_l[shell.angular_momentum().unsigned_abs() as usize].push(s);
    }
    let mut d_aux = scratch.device(g, sl.aux);
    drop(s0.upload(&mut d_aux, h_aux)?);

    // Each class's pairs and record prefix, by a walk that writes in
    // place: the prefixes the cells read, kept on the host, and each
    // class's records tables.
    let first_pair: Vec<usize> = d
        .classes
        .iter()
        .scan(0, |at, c| {
            let first = *at;
            *at += c.n_pair;
            Some(first)
        })
        .collect();
    let mut prefix = scratch.pinned(g, sl.h_prefix);
    let mut diagonal = scratch.pinned(g, sl.h_diagonal);
    let mut tables: Vec<_> = sl
        .h_tables
        .into_iter()
        .map(|span| scratch.pinned(g, span))
        .collect();
    let mut rank_of_l = [0i32; 8];
    for (c, t) in tables.iter_mut().enumerate() {
        let mut p = 0i32;
        t[0] = 0;
        for (s, shell) in basis.shells().iter().enumerate() {
            p += shell.nprimitives() as i32;
            t[s + 1] = p;
            t[n_shell + 1 + s] = basis.ao_offset_cart(s as u32) as i32;
            if c == 0 {
                let l = shell.angular_momentum().unsigned_abs() as usize;
                t[2 * n_shell + 1 + s] = rank_of_l[l];
                rank_of_l[l] += 1;
            }
        }
    }
    for c in 1..tables.len() {
        let (head, tail) = tables.split_at_mut(c);
        tail[0][2 * n_shell + 1..3 * n_shell + 1]
            .copy_from_slice(&head[0][2 * n_shell + 1..3 * n_shell + 1]);
    }
    let mut filled: Vec<usize> = d.classes.iter().map(|_| 0).collect();
    let mut records: Vec<i32> = d.classes.iter().map(|_| 0).collect();
    for c in 0..d.classes.len() {
        prefix[first_pair[c] + c] = 0;
    }
    for_each_pair(basis, |la, lb, bra, ket| {
        let c = class_of(d, la, lb);
        let (k, n) = (filled[c], d.classes[c].n_pair);
        let t = &mut tables[c];
        let base = 3 * n_shell + 1;
        t[base + k] = records[c];
        records[c] += (basis.shells()[bra as usize].nprimitives()
            * basis.shells()[ket as usize].nprimitives()) as i32;
        t[base + k + 1] = records[c];
        t[base + n + 1 + k] = bra as i32;
        t[base + 2 * n + 1 + k] = ket as i32;
        prefix[first_pair[c] + c + k + 1] = records[c];
        diagonal[first_pair[c] + k] = i32::from(bra == ket);
        filled[c] += 1;
    });
    // Each class's records, formed on the device from its tables.
    let mut rec_f: Vec<_> = sl.rec_f.into_iter().map(|s| scratch.device(g, s)).collect();
    let mut rec_i: Vec<_> = sl.rec_i.into_iter().map(|s| scratch.device(g, s)).collect();
    for (c, ((t, span), (f, i))) in tables
        .into_iter()
        .zip(sl.tables)
        .zip(rec_f.iter_mut().zip(rec_i.iter_mut()))
        .enumerate()
    {
        let class = &d.classes[c];
        let mut t_dev = scratch.device(g, span);
        drop(mmd_records(
            s0,
            (n_shell, n_prim),
            &shells,
            (class.n_pair, class.n_rec),
            t,
            &mut t_dev,
            f,
            i,
        )?);
    }
    // The screen: each record's bound on the first stream, after its
    // records, read on the host and summed per shell pair, before any
    // batch's cells are written. At 0 nothing is formed or read.
    let threshold = shared
        .config
        .eri
        .as_ref()
        .map_or(0.0, |t| t.schwarz_threshold);
    #[allow(
        clippy::disallowed_macros,
        reason = "small scratch: one flag per shell pair, a byte each, the size of the records' tables beside it"
    )]
    let mut kept = vec![true; d.n_shell_pair];
    if threshold > 0.0 {
        let mut downs = Vec::new();
        for (c, (span, h_span)) in sl.bounds.into_iter().zip(sl.h_bounds).enumerate() {
            let class = &d.classes[c];
            let mut bound = scratch.device(g, span);
            record_bound(s0, (class.la, class.lb), class.n_rec, &rec_f[c], &mut bound)?;
            downs.push(s0.download(scratch.pinned(g, h_span), &bound)?);
        }
        for (c, down) in downs.into_iter().enumerate() {
            let bound = screen_wait!(down)?;
            let base = first_pair[c] + c;
            for k in 0..d.classes[c].n_pair {
                let (r0, r1) = (prefix[base + k] as usize, prefix[base + k + 1] as usize);
                let sum: f64 = bound[r0..r1].iter().sum();
                // A NaN bound keeps its pair.
                kept[first_pair[c] + k] = sum.is_nan() || sum >= threshold;
            }
        }
    }
    let dropped = kept.iter().filter(|&&k| !k).count();
    let ready = &st.events[2 * depth];
    ready.record(s0)?;
    for s in streams.iter().take(depth).skip(1) {
        // A ring buffer's stream reads the records the first one formed.
        s.wait(ready)?;
    }

    // The metric's factor, on the GPU's stream before the loop: it
    // overlaps the first batch's kernels.
    let mut tile = scratch.device(g, sl.tile);
    let mut flag = scratch.device(g, sl.flag);
    gpu.stream().copy_device(l, metric)?;
    gpu.cholesky(q, l, &mut tile, &mut flag)?;

    // The pair batches, through the ring.
    let ring = ring_layout(d, width, depth);
    let mut slot_dev: Vec<_> = ring
        .slots
        .iter()
        .map(|s| {
            let cells_len = ring.r.0[s.cells] / size_of::<i32>();
            let batch_len = ring.r.0[s.batch] / size_of::<f64>();
            (
                st.buf.tenant(&ring.r.span::<i32>(s.cells, cells_len)),
                st.buf.tenant(&ring.r.span::<f64>(s.batch, batch_len)),
            )
        })
        .collect();
    type Flight<'s> = (
        Batch,
        Pending<'s, PinnedTenant<f64>, false>,
        Vec<Pending<'s, PinnedTenant<i32>>>,
    );
    let mut in_flight: Vec<Option<Flight<'_>>> = (0..depth).map(|_| None).collect();
    let mut h_batches: Vec<Option<PinnedTenant<f64>>> = ring
        .slots
        .iter()
        .map(|s| {
            let len = ring.p.0[s.h_batch] / size_of::<f64>();
            Some(st.pinned.tenant(&ring.p.span::<f64>(s.h_batch, len)))
        })
        .collect();
    let land = |flight: Flight<'_>, rows: &mut [&mut [f64]]| -> Result<PinnedTenant<f64>> {
        let (batch, down, cells) = flight;
        // The host waits on the buffer's download, its stream alone.
        let fitted = ring_wait!(down)?;
        for cell in cells {
            drop(ring_wait!(cell)?);
        }
        let cols = batch.cols(d);
        scatter(rows, batch.first_col - first_col, cols, &fitted[..q * cols]);
        Ok(fitted)
    };
    // Largest kernel stack first: the batches by their class's
    // (la + lb, la), descending, and each batch's momenta from the highest,
    // so the first launch on each ring stream is the largest it makes, and
    // a smaller one after it on the same queue fits the stack that launch
    // took (`docs/memory.md`, "Caps and margins"). Each batch's columns are
    // its own, so the order moves no element of B.
    let mut order: Vec<&Batch> = mine.iter().collect();
    order.sort_by_key(|b| std::cmp::Reverse(stack_rank(&d.classes[b.class])));
    for (turn, batch) in order.into_iter().enumerate() {
        let s = turn % depth;
        let stream = &streams[s];
        if let Some(flight) = in_flight[s].take() {
            h_batches[s] = Some(land(flight, &mut rows)?);
        }
        let class = &d.classes[batch.class];
        let ns = class.sph();
        let cols = batch.cols(d);
        let base = first_pair[batch.class] + batch.class;
        let pairs = &prefix[base..base + class.n_pair + 1];
        let diag = &diagonal[first_pair[batch.class]..first_pair[batch.class] + class.n_pair];
        let (cells, tensor) = &mut slot_dev[s];
        // The batch's pairs the screen keeps. A dropped pair has no cells,
        // so its block is not written: the batch is zeroed first, on the
        // buffer's stream after its last download.
        let mine: Vec<usize> = (0..batch.n_pair)
            .filter(|&i| kept[first_pair[batch.class] + batch.first_pair + i])
            .collect();
        if mine.len() < batch.n_pair {
            stream.zero(&mut part_mut(tensor, 0, q * cols))?;
        }
        // The cells: this batch's kept pairs crossed with every aux shell
        // of one momentum, each destination relative to the batch, written
        // into an exact-length tenant of that momentum's pinned region.
        let mut uploads = Vec::new();
        for (lp, shells_l) in aux_of_l.iter().enumerate().rev() {
            let n_cell = mine.len() * shells_l.len();
            if n_cell == 0 {
                continue;
            }
            let span = ring
                .p
                .span::<i32>(ring.slots[s].h_cells[lp], CELL_FIELDS * n_cell);
            let mut h = st.pinned.tenant(&span);
            let mut k = 0;
            for &i in &mine {
                let p = batch.first_pair + i;
                for &sh in shells_l {
                    let fields = [
                        pairs[p],
                        pairs[p + 1],
                        aux_rec[sh],
                        aux_rec[sh + 1],
                        (i * ns * q) as i32 + aux_sph[sh],
                        diag[p],
                    ];
                    for (f, v) in fields.into_iter().enumerate() {
                        h[f * n_cell + k] = v;
                    }
                    k += 1;
                }
            }
            uploads.push(three_center_spherical_class(
                stream,
                (class.la, class.lb, lp as i32),
                class.n_rec,
                &rec_f[batch.class],
                n_aux_rec,
                &d_aux,
                n_cell,
                h,
                cells,
                (cols, q),
                tensor,
            )?);
        }
        let (kernels, fit) = (&st.events[2 * s], &st.events[2 * s + 1]);
        kernels.record(stream)?;
        // The fit, in place, on the GPU's stream: B ← L⁻¹ B, the tree's
        // own blocked solve, after this batch's kernels. A unit test's
        // negative control alone removes the wait.
        #[cfg(test)]
        let keep = !tests::DROP_THE_FIT_EVENT.load(std::sync::atomic::Ordering::SeqCst);
        #[cfg(not(test))]
        let keep = true;
        if keep {
            gpu.stream().wait(kernels)?;
        }
        let mut fitted = part_mut(tensor, 0, q * cols);
        gpu.trsm(
            wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            wwr_blas_operation::WWR_BLAS_OP_N,
            wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
            q as i32,
            cols as i32,
            1.0,
            &*l,
            q as i32,
            &mut fitted,
            q as i32,
        )?;
        fit.record(gpu.stream())?;
        // The download, on the ring buffer's stream, after the fit.
        stream.wait(fit)?;
        let h_batch = h_batches[s].take().expect("the slot's pinned end is home");
        let down = stream.download(h_batch, &part(tensor, 0, q * cols))?;
        in_flight[s] = Some((*batch, down, uploads));
    }
    for flight in in_flight.into_iter().flatten() {
        drop(land(flight, &mut rows)?);
    }
    drop((h_batches, slot_dev));

    // The gather weights a J/K pass reads, written once: 1 on a diagonal
    // pair's column, 2 on any other.
    let jk = jk_layout(d, 1, depth);
    let mut h_weights = scratch.pinned(g, sl.h_weights);
    pair_columns(shared, |col, _, _, w| h_weights[col] = w);
    let mut weights = st.buf.tenant(&jk.r.span::<f64>(jk.weights, d.n_col));
    drop(gpu.stream().upload(&mut weights, h_weights)?);

    // The factorization's flag, read on the host after the GPU's last
    // enqueue: the solver-flag site (docs/gpu_layer.md, "The v0.0.4
    // runtime").
    let h_flag = scratch.pinned(g, sl.h_flag);
    #[allow(
        clippy::disallowed_methods,
        reason = "a step reading a solver's flag after its last enqueue (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
    )]
    let h_flag = gpu.stream().download(h_flag, &flag)?.sync()?;
    if h_flag[0] != 0 {
        return Err(Error::numerical(format!(
            "eri: on gpu {g} the metric's leading minor {} is not positive",
            h_flag[0]
        )));
    }
    Ok(dropped)
}

/// Every pair column of B in order: `f(column, row m, column n, weight)`,
/// `(m, n)` its spherical AOs, the weight 1 on a diagonal pair and 2 on any
/// other. What a pass's gather tables and weights are written from.
fn pair_columns(shared: &Shared, mut f: impl FnMut(usize, i32, i32, f64)) {
    let (d, basis) = (&shared.dims, &shared.basis);
    let first_col: Vec<usize> = d
        .classes
        .iter()
        .scan(0, |at, c| {
            let first = *at;
            *at += c.n_pair * c.sph();
            Some(first)
        })
        .collect();
    let mut filled: Vec<usize> = d.classes.iter().map(|_| 0).collect();
    for_each_pair(basis, |la, lb, bra, ket| {
        let c = class_of(d, la, lb);
        let (na, nb) = (2 * la + 1, 2 * lb + 1);
        let (off_a, off_b) = (
            basis.ao_offset_sphe(bra) as i32,
            basis.ao_offset_sphe(ket) as i32,
        );
        let mut col = first_col[c] + filled[c] * d.classes[c].sph();
        for ia in 0..na {
            for ib in 0..nb {
                f(
                    col,
                    off_a + ia,
                    off_b + ib,
                    if bra == ket { 1.0 } else { 2.0 },
                );
                col += 1;
            }
        }
        filled[c] += 1;
    });
}

/// One GPU's J/K pass: its share of B's rows contracted with each density
/// of `d`, D or the `n_occ` occupied orbitals of `D = 2 C Cᵀ`, into its
/// fragments of J and K, each density's in turn on one aux batch's
/// squares: the batch is uploaded and gathered once for all of them. gpu0
/// reads the caller's matrices (`here`), every other GPU its copies. The uploads and gathers go on the manager's streams, one per
/// ring buffer, the contractions on the GPU's stream, joined by the ring's
/// events; everything on the manager's streams has run when it returns.
#[allow(clippy::too_many_arguments, reason = "a manager's inner task")]
fn gpu_pass(
    shared: &Shared,
    b: &[f64],
    share: Range<usize>,
    width: usize,
    gpu: &mut GpuResources,
    st: &mut GpuState,
    l: &JkLayout,
    here: Option<&[Density<'_>]>,
    densities: &[Density<'_>],
) -> Status {
    let d = &shared.dims;
    let (n, n_col) = (d.n_ao, d.n_col);
    let n2 = n * n;
    let depth = l.staging.len();
    let weights = st.buf.tenant(&l.r.span::<f64>(l.weights, n_col));
    let staged: Vec<_> = densities
        .iter()
        .zip(&l.d)
        .map(|(m, &at)| st.buf.tenant(&l.r.span::<f64>(at, m.matrix().len())))
        .collect();
    let mut jk: Vec<_> =
        l.j.iter()
            .zip(&l.k)
            .take(densities.len())
            .map(|(&j, &k)| {
                (
                    st.buf.tenant(&l.r.span::<f64>(j, n2)),
                    st.buf.tenant(&l.r.span::<f64>(k, n2)),
                )
            })
            .collect();
    let mut bd = st.buf.tenant(&l.r.span::<f64>(l.bd, width * n2));
    let mut gamma = st.buf.tenant(&l.r.span::<f64>(l.gamma, width));
    let dms: Vec<(Part<'_, f64>, Option<usize>)> = densities
        .iter()
        .zip(&staged)
        .enumerate()
        .map(|(i, (m, staged))| {
            let dm = here.map_or_else(|| part(staged, 0, m.matrix().len()), |h| h[i].matrix());
            (dm, m.n_occ())
        })
        .collect();
    for (jj, kk) in &mut jk {
        gpu.stream().zero(jj)?;
        gpu.stream().zero(kk)?;
    }
    let mut slots: Vec<_> = (0..depth)
        .map(|s| {
            (
                st.buf.tenant(&l.r.span::<i32>(l.tables[s], 2 * n_col)),
                st.buf.tenant(&l.r.span::<f64>(l.staging[s], width * n_col)),
                st.buf.tenant(&l.r.span::<f64>(l.squares[s], width * n2)),
            )
        })
        .collect();
    // Each buffer's pinned ends: the gather tables, written once a pass,
    // and the rows of B.
    let mut homes: Vec<Option<(PinnedTenant<i32>, PinnedTenant<f64>)>> = (0..depth)
        .map(|s| {
            let mut t = st.pinned.tenant(&l.p.span::<i32>(l.h_tables[s], 2 * n_col));
            pair_columns(shared, |c, m, nn, _| {
                t[c] = m;
                t[n_col + c] = nn;
            });
            let p = st
                .pinned
                .tenant(&l.p.span::<f64>(l.h_packed[s], width * n_col));
            Some((t, p))
        })
        .collect();
    type Flight<'s> = (
        Pending<'s, PinnedTenant<f64>>,
        Pending<'s, PinnedTenant<i32>>,
    );
    let mut in_flight: Vec<Option<Flight<'_>>> = (0..depth).map(|_| None).collect();
    let (ni, n2i) = (n as i32, n2 as i32);
    let op_n = wwr_blas_operation::WWR_BLAS_OP_N;
    let op_t = wwr_blas_operation::WWR_BLAS_OP_T;
    let lower = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
    let n_events = st.events.len();
    let pass_events = n_events - 2 - 2 * depth;
    for (turn, first) in share.clone().step_by(width.max(1)).enumerate() {
        let s = turn % depth;
        let up = &st.streams[s];
        let rows = width.min(share.end - first);
        let len = n_col * rows;
        let (gathered, contracted) = (
            &st.events[pass_events + 2 * s],
            &st.events[pass_events + 2 * s + 1],
        );
        if let Some((packed, tables)) = in_flight[s].take() {
            // The buffer's last upload and gather have run: its pinned ends
            // come home. Its device end waits for the contraction.
            homes[s] = Some((ring_wait!(tables)?, ring_wait!(packed)?));
            up.wait(contracted)?;
        }
        let (tables, mut packed) = homes[s].take().expect("the buffer's pinned ends are home");
        packed[..len].copy_from_slice(&b[n_col * first..][..len]);
        let (t_dev, staging, squares) = &mut slots[s];
        let mut stage = part_mut(staging, 0, len);
        let packed = up.upload(&mut stage, packed)?;
        let mut sq = part_mut(squares, 0, rows * n2);
        let tables = packed_gather(
            up,
            (n, n_col, rows),
            tables,
            t_dev,
            &weights,
            &part(staging, 0, len),
            &mut sq,
        )?;
        gathered.record(up)?;
        gpu.stream().wait(gathered)?;
        let sq = part(squares, 0, rows * n2);
        let r = rows as i32;
        for (&(dm, n_occ), (jj, kk)) in dms.iter().zip(&mut jk) {
            if let Some(n_occ) = n_occ {
                // With D = 2 C Cᵀ: every B_Q C in one batched gemm, the blocks
                // side by side as one n × (n_occ rows) matrix M; gamma_Q =
                // B_Q · D = 2 (B_Q C) · C, J += Σ_Q gamma_Q B_Q, K += 2 M Mᵀ:
                // a syrk on CUDA, a gemm on HIP ([`K_BY_SYRK`]).
                let (oi, no) = (n_occ as i32, (n * n_occ) as i32);
                gpu.gemm_strided_batched(
                    op_n,
                    op_n,
                    ni,
                    oi,
                    ni,
                    1.0,
                    &sq,
                    ni,
                    i64::from(n2i),
                    &dm,
                    ni,
                    0,
                    0.0,
                    &mut bd,
                    ni,
                    i64::from(no),
                    r,
                )?;
                gpu.gemv(op_t, no, r, 2.0, &bd, no, &dm, 0.0, &mut gamma)?;
                gpu.gemv(op_n, n2i, r, 1.0, &sq, n2i, &gamma, 1.0, jj)?;
                if K_BY_SYRK {
                    gpu.syrk(lower, op_n, ni, oi * r, 2.0, &bd, ni, 1.0, kk, ni)?;
                } else {
                    gpu.gemm(
                        op_n,
                        op_t,
                        ni,
                        ni,
                        oi * r,
                        2.0,
                        &bd,
                        ni,
                        &bd,
                        ni,
                        1.0,
                        kk,
                        ni,
                    )?;
                }
            } else {
                // gamma_Q = B_Q · D, J += Σ_Q gamma_Q B_Q; K += Σ_Q (B_Q D)
                // B_Qᵀ, every B_Q D in one batched gemm, then the blocks side by
                // side as one n × (n rows) matrix times B's.
                gpu.gemv(op_t, n2i, r, 1.0, &sq, n2i, &dm, 0.0, &mut gamma)?;
                gpu.gemv(op_n, n2i, r, 1.0, &sq, n2i, &gamma, 1.0, jj)?;
                gpu.gemm_strided_batched(
                    op_n,
                    op_n,
                    ni,
                    ni,
                    ni,
                    1.0,
                    &sq,
                    ni,
                    i64::from(n2i),
                    &dm,
                    ni,
                    0,
                    0.0,
                    &mut bd,
                    ni,
                    i64::from(n2i),
                    r,
                )?;
                gpu.gemm(
                    op_n,
                    op_t,
                    ni,
                    ni,
                    ni * r,
                    1.0,
                    &bd,
                    ni,
                    &sq,
                    ni,
                    1.0,
                    kk,
                    ni,
                )?;
            }
        }
        contracted.record(gpu.stream())?;
        in_flight[s] = Some((packed, tables));
    }
    // Whoever fans out, joins: the manager's streams have run, and the
    // pinned ends are home for the next pass.
    for flight in in_flight.into_iter().flatten() {
        let (packed, tables) = flight;
        drop((ring_wait!(tables)?, ring_wait!(packed)?));
    }
    drop((homes, slots));
    for ((_, n_occ), (_, kk)) in dms.iter().zip(&mut jk) {
        if K_BY_SYRK && n_occ.is_some() {
            // K from C holds its lower triangle alone: the upper from it,
            // on the GPU's stream after the last syrk, so the fragment is
            // read in full.
            gpu.mirror_triangle(lower, ni, kk, ni)?;
        }
    }
    // The fragments are read on the manager's first stream: it waits for
    // the contractions.
    let done = &st.events[n_events - 1];
    done.record(gpu.stream())?;
    st.streams[0].wait(done)?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    //! The pair batches' partition (tier 3), and J and K from C against J
    //! and K from `D = 2 C Cᵀ` on water (tier 4), through a probe step that
    //! reads the manager and calls [`EriManager::build_jk`] as a reader
    //! does, under the locks the wrapper takes.
    #![allow(
        clippy::disallowed_methods,
        reason = "a test reads its result back on the host: the harness waits, not a task"
    )]
    #![allow(
        clippy::disallowed_macros,
        reason = "a test's host data, outside any step's run"
    )]

    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    use fb_core_budget::{Backing, Caps, Ledger};
    use fb_core_flow::{Bus, Dag, Executor};

    use crate::bus::Config;

    /// The negative control of the event before the fit: set, the fit no
    /// longer waits for its batch's kernels on the ring buffer's stream.
    /// Only [`removing_the_event_before_the_fit_fails_under_the_nan_fill`]
    /// sets it, under [`CONTROL`].
    pub(crate) static DROP_THE_FIT_EVENT: AtomicBool = AtomicBool::new(false);

    /// Held by every device case of this module, so the control's switch
    /// is never set while another builds B.
    static CONTROL: Mutex<()> = Mutex::new(());

    fn water(eri: &str) -> Config {
        let text = format!(
            "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
             [eri]\naux_basis = \"cc-pvdz-jkfit\"\n{eri}"
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        Config::parse(&text, &dir).unwrap()
    }

    /// The batches partition the columns at every width from one pair up
    /// past the widest class, and a list with a gap is refused, naming the
    /// batch.
    #[test]
    fn the_pair_batches_partition_the_columns() {
        let bus = MainBus::setup(water(""), Backing::Dry).unwrap();
        let d = &bus.shared().dims;
        let widest = d.classes.iter().map(|c| c.n_pair * c.sph()).max().unwrap();
        for w in [1, 9, 25, 100, widest, widest + 1] {
            let list = batches(d, w);
            assert_eq!(
                list.iter().map(|b| b.cols(d)).sum::<usize>(),
                d.n_col,
                "w {w}"
            );
        }
        let mut gap = batches(d, 25);
        gap[1].first_col += 1;
        let message = fb_test_support_message(|| check_partition(d, &gap));
        assert!(
            message.contains("eri: pair batch 1 opens at column"),
            "{message}"
        );
    }

    /// `ssperi3<la, lb, lp>`'s kernel stack per lane, bytes: each class's
    /// `.private_segment_fixed_size` in the gfx1200 code objects
    /// (`llvm-readelf --notes`), row `(la, lb)` in the launcher's order
    /// (`la <= lb`), column `lp`.
    const STACK: [((i32, i32), [u32; 7]); 10] = [
        ((0, 0), [0, 0, 0, 0, 0, 0, 0]),
        ((0, 1), [0, 0, 0, 0, 24, 188, 1232]),
        ((0, 2), [0, 0, 56, 244, 1040, 1840, 2432]),
        ((0, 3), [0, 164, 836, 1260, 2464, 3040, 3376]),
        ((1, 1), [0, 0, 288, 552, 1440, 2160, 2800]),
        ((1, 2), [304, 816, 1632, 2192, 3360, 4032, 4304]),
        ((1, 3), [1296, 2048, 3504, 3744, 4000, 4880, 5792]),
        ((2, 2), [1680, 2128, 3536, 4000, 4528, 5392, 6416]),
        ((2, 3), [2672, 3616, 5440, 5920, 6720, 8048, 9232]),
        ((3, 3), [4176, 5952, 7232, 8704, 10032, 11504, 13072]),
    ];

    /// The build's launch order is largest kernel stack first: within a
    /// class the stack rises with the aux momentum, and at every momentum
    /// a class of higher [`stack_rank`] has at least the stack of one
    /// below it, so each ring stream's first launch is its largest.
    #[test]
    fn the_stack_rank_orders_ssperis_stack() {
        let rank = |(la, lb): (i32, i32)| {
            stack_rank(&ClassDim {
                la,
                lb,
                n_pair: 1,
                n_rec: 1,
            })
        };
        for (class, row) in STACK {
            assert!(row.is_sorted(), "{class:?}: {row:?}");
        }
        for (a, row_a) in STACK {
            for (b, row_b) in STACK {
                if rank(a) < rank(b) {
                    for lp in 0..7 {
                        assert!(
                            row_a[lp] <= row_b[lp],
                            "{a:?} ranks below {b:?} with more stack at lp {lp}"
                        );
                    }
                }
            }
        }
    }

    /// The panic message of `f`.
    fn fb_test_support_message(f: impl FnOnce()) -> String {
        let payload =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("the call panics");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default()
    }

    /// The probe: J and K from C and from D, on gpu0.
    pub(crate) struct ProbeOutput {
        c: DeviceBuf<f64>,
        d: DeviceBuf<f64>,
        jk: [DeviceBuf<f64>; 4],
    }

    #[derive(Clone, Copy)]
    struct Probe;

    /// C, a fixed `n × n_occ` of mixed sign.
    fn c_of(n: usize, n_occ: usize) -> Vec<f64> {
        (0..n * n_occ)
            .map(|i| ((i * 7919 % 1013) as f64 / 1013.0 - 0.5) * 0.4)
            .collect()
    }

    impl Step<MainBus> for Probe {
        const NAME: &'static str = "probe";
        type Output = ProbeOutput;
        type Reads = (EriStep,);

        fn slot(bus: &MainBus) -> &Slot<ProbeOutput> {
            &bus.probe
        }

        fn placement(_: &Shared) -> Placement {
            Placement::AllGpus
        }

        fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<ProbeOutput> {
            let (n, o) = (shared.dims.n_ao, shared.dims.n_occ);
            let mut l = Layout::new();
            let _ = l.span::<f64>(n * o);
            budget.scratch(GPU0, Kind::Pinned, l.bytes());
            budget.locks(
                std::iter::once(LockId::Manager(ManagerId::Eri))
                    .chain((0..budget.gpus()).map(LockId::Gpu)),
            );
            Ok(ProbeOutput {
                c: budget.device(GPU0, n * o),
                d: budget.device(GPU0, n * n),
                jk: std::array::from_fn(|_| budget.device(GPU0, n * n)),
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
            let (n, o) = (shared.dims.n_ao, shared.dims.n_occ);
            let mut staged = scratch.pinned(GPU0, Layout::new().span::<f64>(n * o));
            staged.copy_from_slice(&c_of(n, o));
            let gpu = held.gpu(GPU0);
            drop(gpu.stream().upload(&mut out.c, staged)?);
            let (ni, oi) = (n as i32, o as i32);
            let op_n = wwr_blas_operation::WWR_BLAS_OP_N;
            let op_t = wwr_blas_operation::WWR_BLAS_OP_T;
            gpu.gemm(
                op_n, op_t, ni, ni, oi, 2.0, &out.c, ni, &out.c, ni, 0.0, &mut out.d, ni,
            )?;
            let eri = view.read(EriStep);
            let [jc, kc, jd, kd] = &mut out.jk;
            eri.build_jk(
                shared,
                held,
                &[Density::Occupied(part(&out.c, 0, n * o), o)],
                &mut [part_mut(jc, 0, n * n)],
                &mut [part_mut(kc, 0, n * n)],
            )?;
            eri.build_jk(
                shared,
                held,
                &[Density::Dense(part(&out.d, 0, n * n))],
                &mut [part_mut(jd, 0, n * n)],
                &mut [part_mut(kd, 0, n * n)],
            )
        }
    }

    /// Water's J and K from C and from D, `[J_C, K_C, J_D, K_D]`, through
    /// the probe on `devices`, at forced batches (twice the least that
    /// holds a batch, so several pair batches and aux batches go through
    /// each ring).
    fn jk_on(devices: &str) -> Vec<Vec<f64>> {
        let _turn = CONTROL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dims = MainBus::setup(water(""), Backing::Dry)
            .unwrap()
            .shared()
            .dims
            .clone();
        let mem = 2 * EriStep::least_mem_per_gpu(&dims, 2);
        let config = water(&format!(
            "mem_per_gpu = {mem}\n[resources]\ndevices = {devices}\n"
        ));
        let bus = MainBus::setup(config, Backing::Real).unwrap();
        let mut dag = Dag::new();
        dag.add(MetricStep).add(EriStep).add(Probe);
        let plan = dag.build(bus).unwrap();
        let executor = Executor::new(2);
        plan.budget(&executor).unwrap();
        plan.execute(&executor).unwrap();
        let n = plan.bus().shared().dims.n_ao;
        let eri = plan.output(EriStep);
        println!(
            "water on {devices}: {} pair batches, aux batches of {} rows of {}",
            eri.batches(),
            eri.jk_width(),
            eri.n_aux()
        );
        assert!(
            eri.jk_width() < eri.n_aux() && eri.batches() > 1,
            "several batches"
        );
        let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
        let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
        let budget = ledger.budget("test");
        let stream = Stream::new(0);
        let probe = plan.output(Probe);
        probe
            .jk
            .iter()
            .map(|m| {
                let host = budget.pinned::<f64>(0, n * n);
                stream.download(host, m).unwrap().sync().unwrap().to_vec()
            })
            .collect()
    }

    /// `got` against `want` at 1e-12 of `want`'s largest element, every
    /// element of `got` finite.
    fn within(what: &str, got: &[f64], want: &[f64]) {
        let largest = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        println!("{what}: max difference {worst:e}, max element {largest:e}");
        assert!(largest > 1e-3, "{what}: empty");
        // f64::max passes a NaN over, so the fold alone would not see one.
        assert!(got.iter().all(|v| v.is_finite()), "{what}: not finite");
        assert!(
            worst <= 1e-12 * largest,
            "{what}: {worst:e} > 1e-12 × {largest:e}"
        );
    }

    /// J and K from C equal J and K from D = 2 C Cᵀ to 1e-12 of the
    /// largest element: one contraction in two association orders, FP64
    /// reassociation over the aux and AO sums alone.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn j_and_k_from_c_equal_j_and_k_from_d_on_water() {
        let jk = jk_on("[0]");
        within("J from C against D", &jk[0], &jk[2]);
        within("K from C against D", &jk[1], &jk[3]);
    }

    /// Two logical GPUs on the card against one: B built in two column
    /// runs, each GPU's share of the aux rows contracted apart, D sent
    /// through the host and the fragments summed there. The same bar: the
    /// split reassociates the aux sum alone.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn two_gpus_of_one_card_give_one_gpus_j_and_k() {
        let (one, two) = (jk_on("[0]"), jk_on("[0, 0]"));
        for (i, what) in ["J from C", "K from C", "J from D", "K from D"]
            .iter()
            .enumerate()
        {
            within(&format!("{what}, two GPUs against one"), &two[i], &one[i]);
        }
    }

    /// Water's B at forced batches (the least `mem_per_gpu` of the ring of
    /// two: many pair batches through each ring buffer), NaN-filled, on
    /// the card: with the event before each fit, without it, and with it
    /// restored. Without it the fit reads its tensor while the kernels on
    /// the ring buffer's stream are still writing it, NaN from the fill or
    /// the last batch's integrals, so B is wrong; restored, B is the first
    /// run's to the bar on B, 1e-10 of its largest element. The control
    /// asserts the failure.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn removing_the_event_before_the_fit_fails_under_the_nan_fill() {
        let _turn = CONTROL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let least = EriStep::least_mem_per_gpu(
            &MainBus::setup(water(""), Backing::Dry)
                .unwrap()
                .shared()
                .dims,
            2,
        );
        let b = |drop_the_event: bool| {
            let mut config = water(&format!("mem_per_gpu = {least}\n"));
            config.resources.nan_fill = true;
            DROP_THE_FIT_EVENT.store(drop_the_event, Ordering::SeqCst);
            let run = crate::run(config, Backing::Real);
            DROP_THE_FIT_EVENT.store(false, Ordering::SeqCst);
            let run = run.unwrap();
            let eri = run.plan.output(EriStep);
            assert!(eri.batches() > 6, "the build is batched");
            eri.b().to_vec()
        };
        let first = b(false);
        let largest = first.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let off = |other: &[f64]| {
            other.iter().zip(&first).fold(0.0f64, |m, (a, b)| {
                if (a - b).is_nan() {
                    f64::INFINITY
                } else {
                    m.max((a - b).abs())
                }
            })
        };
        let (without, restored) = (off(&b(true)), off(&b(false)));
        println!(
            "B: max |B| = {largest:.6}; without the event max |ΔB| = {without:.1e}, restored \
             {restored:.1e} (bar {:.1e})",
            1e-10 * largest
        );
        assert!(largest > 0.1);
        assert!(without > 1e-10 * largest, "B is right without the event");
        assert!(restored <= 1e-10 * largest, "B is wrong with it restored");
    }
}
