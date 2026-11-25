//! The density-fitted integrals, one all-GPU step whose `Output` is the
//! [`EriManager`]: the metric's Cholesky factor L on every GPU, the fitted
//! tensor
//!
//! ```text
//! B[Q][μν] = Σ_P L⁻¹[Q][P] (P|μν)
//! ```
//!
//! on the host heap, and per GPU the ERI buffer with the cache's state.
//!
//! **B is built over pairs.** A pair column of B depends on every P, so a
//! batch of P cannot be fitted alone; a batch of pairs can. Each GPU takes
//! its own pair batches, forms (μν|P) in its ring, fits it against its own
//! L, and downloads it into B's slab. (μν|P) is spherical from the kernel,
//! `ssperi3<la, lb, lP>`: no bra transform runs, and no array here has a
//! Cartesian index.
//!
//! **The ring.** Two pair batches in flight per GPU, each ring buffer with
//! a stream of its own carrying its batch's upload, its three-centre
//! kernels and its download. The fit goes on the GPU's stream, under the
//! GPU's lock: it waits on the event its ring stream recorded after the
//! kernels, and the download waits on the event recorded after the fit.
//! The host waits on a ring buffer's last event, not a whole stream,
//! before it scatters that batch into B and writes the buffer's pinned
//! bytes again.
//!
//! **B is contracted over aux functions.** [`EriManager::build_jk`] runs a
//! J/K pass over aux batches: each GPU contracts its own fixed share, the
//! cached rows first, and the fragments are reduced onto gpu0 through the
//! host. The cache fills on the first pass, from the host copy.
//!
//! **The ERI buffer** is one device buffer per GPU within `[eri]
//! mem_per_gpu`. During `eri`'s run it holds the ring; afterwards the
//! gather tables, a copy of D, the J and K fragments, one aux batch's
//! work, and the cache.

use std::ops::Range;
use std::sync::Mutex;

use fb_core_error_handling::{Error, Result, Status};

use crate::budget::{
    Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, LockId, ManagerId, PinnedBuf, Span, bytes,
};
use crate::bus::{Dims, GpuLane, Held, Lane, MainBus, Shared, for_each_pair, n_sph_of};
use crate::device::launch::{
    self, AUX_FIELDS, CELL_FIELDS, PairSource, RECORD_F64_FIELDS, RECORD_I32_FIELDS,
};
use crate::device::{
    CHOLESKY_TILE, Event, GpuResources, OP_N, OP_T, Part, PartMut, part, part_mut,
};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::metric::MetricStep;
use crate::steps::{GPU0, fan_out};

/// The default step limit: each GPU's ERI buffer. A fixed constant, not a
/// fraction of the cap, so a device-free budget is exact. A PLACEHOLDER
/// for a measured value.
pub const DEFAULT_MEM_PER_GPU: u64 = 2 << 30;

/// The step.
#[derive(Clone, Copy)]
pub struct EriStep;

/// One GPU's part of the manager's mutable state.
struct GpuState {
    /// The ERI buffer.
    buf: DeviceBuf<u8>,
    /// Its pinned staging.
    pinned: PinnedBuf<u8>,
    /// The aux rows of this GPU's share the cache holds: the first ones.
    cache_rows: usize,
    /// Whether the cache holds them yet: filled on the first J/K pass.
    cache_filled: bool,
    /// The ring's events, three per ring buffer (kernels, fit, download),
    /// and the one after the records: made on the first run.
    events: Vec<Event>,
}

/// The manager's state: what changes after `eri`, and only under the
/// manager's lock.
struct EriState {
    gpus: Vec<GpuState>,
    passes: usize,
    uploaded: usize,
}

/// The fitted integrals and what builds J and K from them. `eri`'s
/// `Output`: its readers declare it like any other, and a reader that
/// calls [`Self::build_jk`] also declares the manager's lock, which the
/// wrapper takes. B and L are immutable after `eri`; the state changes
/// only under that lock.
pub struct EriManager {
    n_aux: usize,
    n_col: usize,
    width: usize,
    batches: usize,
    depth: usize,
    jk_width: usize,
    shares: Vec<Range<usize>>,
    l: Vec<DeviceBuf<f64>>,
    b: HeapSlab<f64>,
    state: Mutex<EriState>,
}

impl EriManager {
    /// The fitting functions: B's rows.
    pub fn n_aux(&self) -> usize {
        self.n_aux
    }

    /// The spherical pair columns: B's row length.
    pub fn n_col(&self) -> usize {
        self.n_col
    }

    /// B, `n_aux` rows of `n_col`, pair-fastest.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// `gpu`'s copy of the metric's Cholesky factor, in the lower triangle.
    pub fn l(&self, gpu: usize) -> &DeviceBuf<f64> {
        &self.l[gpu]
    }

    /// The pair batch width the step limit picked, in pair columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The pair batches B was built in.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// The aux batch width of a J/K pass.
    pub fn jk_width(&self) -> usize {
        self.jk_width
    }

    /// The aux rows `gpu` contracts, fixed in budget mode.
    pub fn share(&self, gpu: usize) -> Range<usize> {
        self.shares[gpu].clone()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, EriState> {
        // Never blocks: the holder of the manager's lock is the one user.
        self.state
            .try_lock()
            .expect("the manager's state is reached under the manager's lock alone")
    }

    /// The aux rows of `gpu`'s share its cache holds.
    pub fn cached_rows(&self, gpu: usize) -> usize {
        self.state().gpus[gpu].cache_rows
    }

    /// The J/K passes made since B was built.
    pub fn passes(&self) -> usize {
        self.state().passes
    }

    /// The aux rows the last J/K pass uploaded, over every GPU.
    pub fn uploaded_last_pass(&self) -> usize {
        self.state().uploaded
    }

    /// J and K for each density, on gpu0. Every GPU contracts its own aux
    /// batches, the cached ones first; the fragments are reduced onto gpu0
    /// through the host. The caller owns its results: this writes the J
    /// and K it is passed, reads the densities it is passed, and keeps
    /// nothing of the caller's. It takes no lock: the GPU guards come in
    /// `held`, held by the wrapper, and so does the manager's.
    ///
    /// v0.0.4 sizes for one density.
    ///
    /// # Errors
    /// A launch's or the library's failure.
    ///
    /// # Panics
    /// If the caller did not declare the manager's lock and every GPU's.
    pub fn build_jk(
        &self,
        shared: &Shared,
        held: &mut Held<'_>,
        d: &[Part<'_, f64>],
        j: &mut [PartMut<'_, f64>],
        k: &mut [PartMut<'_, f64>],
    ) -> Status {
        assert!(
            held.holds_manager(ManagerId::Eri),
            "build_jk: the caller did not declare the EriManager's lock"
        );
        assert!(
            d.len() == 1 && j.len() == 1 && k.len() == 1,
            "build_jk: v0.0.4 sizes for one density"
        );
        let dims = &shared.dims;
        let n2 = dims.n_ao * dims.n_ao;
        let mut state = self.state();
        let state = &mut *state;
        let mut gpus = held.all_gpus();
        let n_gpus = gpus.len();
        let layouts: Vec<JkLayout> = state
            .gpus
            .iter()
            .map(|g| jk_layout(dims, self.jk_width, g.cache_rows))
            .collect();

        // D for the other GPUs goes through the host, in gpu0's pinned
        // staging; they only read it there.
        let h_d: &[f64] = {
            let staged = state.gpus[GPU0].pinned.tenant(&layouts[GPU0].h_d);
            // SAFETY: the tenant is gpu0's pinned staging, which outlives
            // the pass; it is written once below, before the fan-out, and
            // only read by the passes.
            unsafe { std::slice::from_raw_parts(staged.as_ptr(), staged.len()) }
        };
        if n_gpus > 1 {
            let gpu0 = &mut *gpus[GPU0];
            let _current = gpu0.current();
            let h_d = state.gpus[GPU0].pinned.tenant(&layouts[GPU0].h_d);
            gpu0.stream().download(h_d, &d[0])?;
            gpu0.stream().sync()?;
        }
        let d0 = d[0];
        let (b, jk_width) = (&self.b[..], self.jk_width);
        let work: Vec<_> = gpus
            .iter_mut()
            .zip(state.gpus.iter_mut())
            .zip(&layouts)
            .zip(&self.shares)
            .collect();
        let uploaded = fan_out(work, |(((gpu, st), l), share)| {
            let source = if gpu.gpu() == GPU0 {
                Density::Here(d0)
            } else {
                Density::Staged(h_d)
            };
            gpu_pass(
                dims,
                b,
                share.clone(),
                jk_width,
                &mut **gpu,
                st,
                l,
                source,
                n_gpus > 1,
            )
        })?;

        // The reduction onto gpu0. One GPU: its fragments are the result,
        // copied on its stream. Several: summed on the host and copied up.
        let gpu0 = &mut *gpus[GPU0];
        let _current = gpu0.current();
        let st0 = &state.gpus[GPU0];
        if n_gpus == 1 {
            let (jj, kk) = (
                st0.buf.tenant(&layouts[GPU0].j),
                st0.buf.tenant(&layouts[GPU0].k),
            );
            gpu0.stream().copy_device(&mut j[0], &part(&jj, 0, n2))?;
            gpu0.stream().copy_device(&mut k[0], &part(&kk, 0, n2))?;
        } else {
            let (sum_j, sum_k) = (
                st0.pinned.tenant(&layouts[GPU0].h_j),
                st0.pinned.tenant(&layouts[GPU0].h_k),
            );
            for (st, l) in state.gpus.iter().zip(&layouts).skip(1) {
                let (fj, fk): (&[f64], &[f64]) =
                    (st.pinned.tenant(&l.h_j), st.pinned.tenant(&l.h_k));
                for (sum, x) in sum_j.iter_mut().zip(fj) {
                    *sum += x;
                }
                for (sum, x) in sum_k.iter_mut().zip(fk) {
                    *sum += x;
                }
            }
            gpu0.stream().upload(&mut j[0], sum_j)?;
            gpu0.stream().upload(&mut k[0], sum_k)?;
            // The pinned sums are the next pass's fragments too.
            gpu0.stream().sync()?;
        }
        state.passes += 1;
        state.uploaded = uploaded.iter().sum();
        Ok(())
    }
}

/// Where a GPU's pass reads D: the caller's, on gpu0, or the host copy it
/// uploads.
#[derive(Clone, Copy)]
enum Density<'a> {
    Here(Part<'a, f64>),
    Staged(&'a [f64]),
}

/// One GPU's J/K pass: its share of B's rows contracted with D into its
/// fragments of J and K, on the GPU's stream throughout, in the order it
/// is enqueued. Returns the rows it uploaded.
#[allow(clippy::too_many_arguments, reason = "a manager's inner task")]
fn gpu_pass(
    dims: &Dims,
    b: &[f64],
    share: Range<usize>,
    width: usize,
    gpu: &mut GpuResources,
    st: &mut GpuState,
    l: &JkLayout,
    d: Density<'_>,
    download: bool,
) -> Result<usize> {
    let _current = gpu.current();
    let (n, n_col) = (dims.n_ao, dims.n_col);
    let n2 = n * n;
    let (rows_of, cols_of, weights) = (
        st.buf.tenant(&l.rows),
        st.buf.tenant(&l.cols),
        st.buf.tenant(&l.weights),
    );
    let (mut d_here, mut jj, mut kk) = (
        st.buf.tenant(&l.d),
        st.buf.tenant(&l.j),
        st.buf.tenant(&l.k),
    );
    let (mut squares, mut bd, mut gamma) = (
        st.buf.tenant(&l.squares),
        st.buf.tenant(&l.bd),
        st.buf.tenant(&l.gamma),
    );
    let (mut staging, mut cache) = (st.buf.tenant(&l.staging), st.buf.tenant(&l.cache));
    let h_packed = st.pinned.tenant(&l.h_packed);
    let d = match d {
        Density::Here(d) => d,
        Density::Staged(h_d) => {
            gpu.stream().upload(&mut d_here, h_d)?;
            part(&d_here, 0, n2)
        }
    };
    gpu.stream().zero(&mut jj)?;
    gpu.stream().zero(&mut kk)?;

    let (ni, n2i) = (n as i32, n2 as i32);
    // `rows` packed rows of B, on the device, into the fragments.
    let mut contract = |gpu: &mut GpuResources, packed: &Part<'_, f64>, rows: usize| -> Status {
        let mut sq = part_mut(&mut squares, 0, rows * n2);
        gpu.stream().zero(&mut sq)?;
        launch::packed_gather(
            gpu.stream(),
            n,
            n_col,
            rows,
            &rows_of,
            &cols_of,
            &weights,
            packed,
            &mut sq,
        )?;
        let r = rows as i32;
        // gamma_Q = B_Q · D, then J += Σ_Q gamma_Q B_Q.
        gpu.gemv(OP_T, n2i, r, 1.0, &sq, n2i, &d, 0.0, &mut gamma)?;
        gpu.gemv(OP_N, n2i, r, 1.0, &sq, n2i, &gamma, 1.0, &mut jj)?;
        // K += Σ_Q (B_Q D) B_Qᵀ: every B_Q D in one batched gemm, then the
        // blocks side by side as one n × (n rows) matrix times B's.
        gpu.gemm_strided_batched(
            ni,
            ni,
            ni,
            1.0,
            &sq,
            ni,
            i64::from(n2i),
            &d,
            ni,
            0,
            0.0,
            &mut bd,
            ni,
            i64::from(n2i),
            r,
        )?;
        gpu.gemm(
            OP_N,
            OP_T,
            ni,
            ni,
            ni * r,
            1.0,
            &bd,
            ni,
            &sq,
            ni,
            1.0,
            &mut kk,
            ni,
        )
    };

    let cached = st.cache_rows.min(share.len());
    let mut uploaded = 0;
    // The first pass fills the cache, from the host copy: the build's pair
    // batches are the wrong index order to fill it.
    if !st.cache_filled {
        for first in (0..cached).step_by(width) {
            let rows = width.min(cached - first);
            let from = n_col * (share.start + first);
            h_packed[..rows * n_col].copy_from_slice(&b[from..from + rows * n_col]);
            let mut into = part_mut(&mut cache, first * n_col, rows * n_col);
            gpu.stream().upload(&mut into, &h_packed[..rows * n_col])?;
            // The pinned bytes are the next batch's too.
            gpu.stream().sync()?;
            uploaded += rows;
        }
        st.cache_filled = true;
    }
    for first in (0..cached).step_by(width) {
        let rows = width.min(cached - first);
        let packed = part(&cache, first * n_col, rows * n_col);
        contract(gpu, &packed, rows)?;
    }
    // What is not cached is uploaded on every pass, one batch in flight
    // with a blocking sync per batch.
    for first in (share.start + cached..share.end).step_by(width) {
        let rows = width.min(share.end - first);
        h_packed[..rows * n_col].copy_from_slice(&b[n_col * first..n_col * (first + rows)]);
        gpu.stream()
            .upload(&mut staging, &h_packed[..rows * n_col])?;
        let packed = part(&staging, 0, rows * n_col);
        contract(gpu, &packed, rows)?;
        gpu.stream().sync()?;
        uploaded += rows;
    }
    if download {
        gpu.stream().download(st.pinned.tenant(&l.h_j), &jj)?;
        gpu.stream().download(st.pinned.tenant(&l.h_k), &kk)?;
        gpu.stream().sync()?;
    }
    Ok(uploaded)
}

/// The ERI buffer during a J/K pass, and its pinned staging, at an aux
/// batch width of `p` rows with `cached` rows in the cache.
struct JkLayout {
    device_bytes: usize,
    pinned_bytes: usize,
    rows: Span<i32>,
    cols: Span<i32>,
    weights: Span<f64>,
    d: Span<f64>,
    j: Span<f64>,
    k: Span<f64>,
    squares: Span<f64>,
    bd: Span<f64>,
    gamma: Span<f64>,
    staging: Span<f64>,
    cache: Span<f64>,
    h_rows: Span<i32>,
    h_cols: Span<i32>,
    h_weights: Span<f64>,
    h_d: Span<f64>,
    h_j: Span<f64>,
    h_k: Span<f64>,
    h_packed: Span<f64>,
}

fn jk_layout(d: &Dims, p: usize, cached: usize) -> JkLayout {
    let (n2, n_col) = (d.n_ao * d.n_ao, d.n_col);
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    JkLayout {
        rows: dev.span(n_col),
        cols: dev.span(n_col),
        weights: dev.span(n_col),
        d: dev.span(n2),
        j: dev.span(n2),
        k: dev.span(n2),
        squares: dev.span(p * n2),
        bd: dev.span(p * n2),
        gamma: dev.span(p),
        staging: dev.span(p * n_col),
        cache: dev.span(cached * n_col),
        h_rows: pin.span(n_col),
        h_cols: pin.span(n_col),
        h_weights: pin.span(n_col),
        h_d: pin.span(n2),
        h_j: pin.span(n2),
        h_k: pin.span(n2),
        h_packed: pin.span(p * n_col),
        device_bytes: dev.bytes(),
        pinned_bytes: pin.bytes(),
    }
}

/// One ring buffer: a pair batch's cells and its `n_aux × width` tensor,
/// and their pinned ends.
struct RingSlot {
    cells: Span<i32>,
    batch: Span<f64>,
    h_cells: Span<i32>,
    h_batch: Span<f64>,
}

/// The ERI buffer during the build, at a pair batch width of `w` columns.
struct RingLayout {
    device_bytes: usize,
    pinned_bytes: usize,
    slots: Vec<RingSlot>,
}

/// The pairs of class `c` in one batch of `w` pair columns.
fn pairs_per_batch(d: &Dims, c: usize, w: usize) -> usize {
    (w / d.classes[c].sph()).clamp(1, d.classes[c].n_pair.max(1))
}

fn ring_layout(d: &Dims, w: usize, depth: usize) -> RingLayout {
    let per = |c: usize| pairs_per_batch(d, c, w);
    let cells = (0..d.classes.len())
        .map(|c| per(c) * d.n_aux_shell)
        .max()
        .unwrap_or(0);
    let columns = (0..d.classes.len())
        .map(|c| per(c) * d.classes[c].sph())
        .max()
        .unwrap_or(0);
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    let slots = (0..depth)
        .map(|_| RingSlot {
            cells: dev.span(CELL_FIELDS * cells),
            batch: dev.span(d.n_aux * columns),
            h_cells: pin.span(CELL_FIELDS * cells),
            h_batch: pin.span(d.n_aux * columns),
        })
        .collect();
    RingLayout {
        slots,
        device_bytes: dev.bytes(),
        pinned_bytes: pin.bytes(),
    }
}

/// The step's scratch on one GPU: the shell tables and the aux records,
/// pinned and device, and on the device each class's records, the
/// Cholesky tile and the flag.
struct ScratchLayout {
    device_bytes: usize,
    pinned_bytes: usize,
    f: Span<f64>,
    i: Span<i32>,
    rec_begin: Span<i32>,
    pairs: Span<u32>,
    aux: Span<f64>,
    class_f: Vec<Span<f64>>,
    class_i: Vec<Span<i32>>,
    tile: Span<f64>,
    flag: Span<i32>,
    h_f: Span<f64>,
    h_i: Span<i32>,
    h_rec_begin: Span<i32>,
    h_pairs: Span<u32>,
    h_aux: Span<f64>,
    h_flag: Span<i32>,
}

fn scratch_layout(d: &Dims) -> ScratchLayout {
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    let n_f = 2 * d.n_prim + 3 * d.n_shell;
    let n_i = 3 * d.n_shell + 1;
    let n_rec_begin = d.n_shell_pair + d.classes.len();
    ScratchLayout {
        f: dev.span(n_f),
        i: dev.span(n_i),
        rec_begin: dev.span(n_rec_begin),
        pairs: dev.span(2 * d.n_shell_pair),
        aux: dev.span(AUX_FIELDS * d.n_aux_prim),
        class_f: d
            .classes
            .iter()
            .map(|c| dev.span(RECORD_F64_FIELDS * c.n_rec))
            .collect(),
        class_i: d
            .classes
            .iter()
            .map(|c| dev.span(RECORD_I32_FIELDS * c.n_rec))
            .collect(),
        tile: dev.span(CHOLESKY_TILE * CHOLESKY_TILE),
        flag: dev.span(1),
        h_f: pin.span(n_f),
        h_i: pin.span(n_i),
        h_rec_begin: pin.span(n_rec_begin),
        h_pairs: pin.span(2 * d.n_shell_pair),
        h_aux: pin.span(AUX_FIELDS * d.n_aux_prim),
        h_flag: pin.span(1),
        device_bytes: dev.bytes(),
        pinned_bytes: pin.bytes(),
    }
}

/// The least `[eri] mem_per_gpu` holds a batch at: the ring at a width of
/// one pair, and one aux batch of one row with no cache. A byte under it,
/// the plan is refused naming `[eri]`.
pub fn least_mem_per_gpu(d: &Dims, ring: usize) -> u64 {
    let least = d.classes.iter().map(|c| c.sph()).max().unwrap_or(1);
    (ring_layout(d, least, ring).device_bytes).max(jk_layout(d, 1, 0).device_bytes) as u64
}

/// A pair batch: a run of one class's pairs, and where its columns lie in
/// B.
#[derive(Clone, Copy, Debug)]
struct Batch {
    class: usize,
    first_pair: usize,
    n_pair: usize,
    first_col: usize,
}

/// The pair batches at width `w`, class by class: a batch never spans two.
/// They partition the pair columns: each batch owns `[first_col, first_col
/// + n_pair sph)` and no two meet, which is what lets several GPUs scatter
/// into B at once.
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
    let mut end = 0;
    for batch in &out {
        assert_eq!(batch.first_col, end, "the batches partition the columns");
        end += batch.n_pair * d.classes[batch.class].sph();
    }
    assert_eq!(end, d.n_col, "the batches cover every column");
    out
}

/// GPU `g` of `n`'s share of `q` aux rows: the `g`-th of `n` contiguous
/// ranges, the first `q % n` one longer. Fixed in budget mode.
fn share(q: usize, g: usize, n: usize) -> Range<usize> {
    let at = |k: usize| k * (q / n) + k.min(q % n);
    at(g)..at(g + 1)
}

/// B's rows, written by several GPUs at once, each its own columns.
struct Columns {
    ptr: *mut f64,
    n_col: usize,
    n_aux: usize,
}

// SAFETY: every batch owns its columns of B and no other batch's
// (`batches` checks the partition), so the GPUs' writes never meet.
unsafe impl Sync for Columns {}

impl Columns {
    /// Scatter a fitted batch, `n_aux × columns` with the aux index
    /// fastest, into B's columns from `first`: the transpose.
    fn scatter(&self, first: usize, columns: usize, batch: &[f64]) {
        assert!(first + columns <= self.n_col && batch.len() >= self.n_aux * columns);
        for c in 0..columns {
            for q in 0..self.n_aux {
                // THE SECOND UNSAFE OF THE MEMORY MODEL. The invariant:
                // the pair batches partition the pair columns, so columns
                // `[first, first + columns)` are this batch's alone.
                unsafe {
                    *self.ptr.add(first + c + self.n_col * q) = batch[q + self.n_aux * c];
                }
            }
        }
    }
}

impl Step<MainBus> for EriStep {
    const NAME: &'static str = "eri";
    /// One stream per ring buffer.
    const STREAMS: usize = 2;
    type Output = EriManager;
    type Reads = (MetricStep,);

    fn slot(bus: &MainBus) -> &Slot<EriManager, Lane> {
        &bus.eri
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<EriManager> {
        let d = &shared.dims;
        let (q, n2) = (d.n_aux, d.n_ao * d.n_ao);
        let table = shared.config.eri.as_ref().expect("eri runs with its table");
        let depth = table.ring;
        if !(1..=2).contains(&depth) {
            return Err(Error::config(format!("[eri] ring is {depth}, not 1 or 2")));
        }
        let limit = table.mem_per_gpu_mib.map_or(DEFAULT_MEM_PER_GPU, |mib| {
            (mib * (1u64 << 20) as f64) as u64
        });
        let n_gpus = budget.gpus();

        // The pair batch width: the widest whose ring fits the limit.
        let least = d.classes.iter().map(|c| c.sph()).max().unwrap_or(1);
        let most = d
            .classes
            .iter()
            .map(|c| c.n_pair * c.sph())
            .max()
            .unwrap_or(1)
            .min(i32::MAX as usize / q.max(1))
            .max(least);
        let ring_at = |w: usize| ring_layout(d, w, depth).device_bytes as u64;
        let jk_at = |p: usize| jk_layout(d, p, 0).device_bytes as u64;
        let needs = least_mem_per_gpu(d, depth);
        if needs > limit {
            return Err(Error::allocation(format!(
                "step 'eri': [eri] mem_per_gpu of {} ({limit} B) holds no batch: the ring at a \
                 width of one pair and one aux batch need at least {} ({needs} B)",
                bytes(limit),
                bytes(needs)
            )));
        }
        let widest = |least: usize, most: usize, fits: &dyn Fn(usize) -> bool| {
            let (mut lo, mut hi) = (least, most.max(least));
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if fits(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            lo
        };
        let width = widest(least, most, &|w| ring_at(w) <= limit);
        // The aux batch width: the widest whose work takes at most half
        // the buffer, so the cache has the rest; never past a share.
        let shares: Vec<Range<usize>> = (0..n_gpus).map(|g| share(q, g, n_gpus)).collect();
        let longest = shares.iter().map(Range::len).max().unwrap_or(1).max(1);
        let most_p = longest.min(i32::MAX as usize / n2.max(1)).max(1);
        let jk_width = widest(1, most_p, &|p| jk_at(p) <= limit / 2);
        if width == least && most > least || jk_width == 1 && most_p > 1 {
            log::warn!(
                "step 'eri': [eri] mem_per_gpu of {} gives a batch width of one",
                bytes(limit)
            );
        }

        let sl = scratch_layout(d);
        let ring = ring_layout(d, width, depth);
        let mut l = Vec::new();
        let mut gpus = Vec::new();
        for (gpu, share) in shares.iter().enumerate() {
            budget.scratch(gpu, Kind::Device, sl.device_bytes);
            budget.scratch(gpu, Kind::Pinned, sl.pinned_bytes);
            l.push(budget.device(gpu, q * q));
            // The cache: as many of this GPU's rows as the rest of the
            // buffer holds.
            let room = (limit - jk_at(jk_width)) as usize / (8 * d.n_col.max(1));
            let mut cache_rows = room.min(share.len());
            while cache_rows > 0 && jk_layout(d, jk_width, cache_rows).device_bytes as u64 > limit {
                cache_rows -= 1;
            }
            let jk = jk_layout(d, jk_width, cache_rows);
            // What is charged is what is used, never the limit.
            gpus.push(GpuState {
                buf: budget.device(gpu, ring.device_bytes.max(jk.device_bytes)),
                pinned: budget.pinned(gpu, ring.pinned_bytes.max(jk.pinned_bytes)),
                cache_rows,
                cache_filled: false,
                events: Vec::new(),
            });
        }
        budget.locks(
            [LockId::Manager(ManagerId::Eri)]
                .into_iter()
                .chain((0..n_gpus).map(LockId::Gpu)),
        );
        Ok(EriManager {
            n_aux: q,
            n_col: d.n_col,
            width,
            batches: batches(d, width).len(),
            depth,
            jk_width,
            shares,
            l,
            // B's home: one slab on the host heap, at the unscreened bound.
            b: budget.heap(d.n_col * q),
            state: Mutex::new(EriState {
                gpus,
                passes: 0,
                uploaded: 0,
            }),
        })
    }

    fn run(
        out: &mut EriManager,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let metric = view.read(MetricStep);
        let all = batches(&shared.dims, out.width);
        let n_gpus = lane.gpus.len();
        let b = Columns {
            ptr: out.b.as_mut_ptr(),
            n_col: out.n_col,
            n_aux: out.n_aux,
        };
        let (width, depth) = (out.width, out.depth);
        // The owner reaches the state through `&mut`, with no lock.
        let state = out.state.get_mut().unwrap_or_else(|e| e.into_inner());
        (state.passes, state.uploaded) = (0, 0);
        let work: Vec<_> = held
            .all_gpus()
            .into_iter()
            .zip(lane.gpus.iter())
            .zip(state.gpus.iter_mut())
            .zip(out.l.iter_mut())
            .collect();
        // Whoever fans out, joins: every GPU's work is finished, and its
        // batches are in B, before this returns.
        fan_out(work, |(((gpu, lane), st), l)| {
            let mine = all.iter().skip(lane.gpu()).step_by(n_gpus);
            gpu_build(
                shared,
                metric.on(lane.gpu()),
                gpu,
                lane,
                st,
                l,
                scratch,
                width,
                depth,
                mine,
                &b,
            )
        })?;
        Ok(())
    }
}

/// One GPU's part of the build: its records, its factor, its pair batches
/// through the ring, and the gather tables a J/K pass reads.
#[allow(
    clippy::too_many_arguments,
    reason = "a step's inner task, spelled out"
)]
fn gpu_build<'a>(
    shared: &Shared,
    metric: &DeviceBuf<f64>,
    gpu: &mut GpuResources,
    lane: &GpuLane,
    st: &mut GpuState,
    l: &mut DeviceBuf<f64>,
    scratch: &Lease<'_>,
    width: usize,
    depth: usize,
    mine: impl Iterator<Item = &'a Batch>,
    b: &Columns,
) -> Status {
    let _current = lane.current();
    let g = lane.gpu();
    let d = &shared.dims;
    let basis = &shared.basis;
    let aux = shared.aux.as_ref().expect("eri runs with a fitting set");
    let (q, n_shell, n_prim) = (d.n_aux, d.n_shell, d.n_prim);
    let sl = scratch_layout(d);
    let s0 = lane.stream(0);

    // The shell tables, built straight into pinned scratch.
    let (h_f, h_i) = (scratch.pinned(g, sl.h_f), scratch.pinned(g, sl.h_i));
    let (h_rec_begin, h_pairs) = (
        scratch.pinned(g, sl.h_rec_begin),
        scratch.pinned(g, sl.h_pairs),
    );
    let mut rank_of_l = [0i32; 8];
    let mut prim = 0;
    for (s, shell) in basis.shells().iter().enumerate() {
        let c = shell.center();
        h_i[s] = prim as i32;
        for p in shell.primitives() {
            h_f[prim] = p.exponent;
            h_f[n_prim + prim] = p.coefficient;
            prim += 1;
        }
        h_f[2 * n_prim + s] = c.x;
        h_f[2 * n_prim + n_shell + s] = c.y;
        h_f[2 * n_prim + 2 * n_shell + s] = c.z;
        h_i[n_shell + 1 + s] = basis.ao_offset_cart(s as u32) as i32;
        let l = shell.angular_momentum() as usize;
        h_i[2 * n_shell + 1 + s] = rank_of_l[l];
        rank_of_l[l] += 1;
    }
    h_i[n_shell] = prim as i32;
    // Each class's pairs and record prefix: a counting walk gave each
    // class's length (the dimensions), so each class has its offset and
    // the walk writes in place.
    let first_pair: Vec<usize> = d
        .classes
        .iter()
        .scan(0, |at, c| {
            let first = *at;
            *at += c.n_pair;
            Some(first)
        })
        .collect();
    let mut filled = vec![0usize; d.classes.len()];
    let mut records = vec![0i32; d.classes.len()];
    for_each_pair(basis, |la, lb, bra, ket| {
        let c = d.class_of(la, lb);
        let (k, n) = (filled[c], d.classes[c].n_pair);
        h_pairs[2 * first_pair[c] + k] = bra;
        h_pairs[2 * first_pair[c] + n + k] = ket;
        h_rec_begin[first_pair[c] + c + k] = records[c];
        records[c] += (basis.shells()[bra as usize].nprimitives()
            * basis.shells()[ket as usize].nprimitives()) as i32;
        h_rec_begin[first_pair[c] + c + k + 1] = records[c];
        filled[c] += 1;
    });
    // The aux records, and each aux shell's record range and first
    // function: O(aux shells) of host scratch, within the policy.
    let h_aux = scratch.pinned(g, sl.h_aux);
    let n_aux_rec = d.n_aux_prim;
    let (mut aux_rec, mut aux_sph) = (vec![0i32], Vec::with_capacity(d.n_aux_shell));
    let mut aux_of_l = vec![Vec::new(); d.aux_shells_of_l.len()];
    let mut r = 0;
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
        aux_of_l[shell.angular_momentum() as usize].push(s);
    }

    let (mut d_f, mut d_i) = (scratch.device(g, sl.f), scratch.device(g, sl.i));
    let (mut d_rec_begin, mut d_pairs) =
        (scratch.device(g, sl.rec_begin), scratch.device(g, sl.pairs));
    let mut d_aux = scratch.device(g, sl.aux);
    s0.upload(&mut d_f, h_f)?;
    s0.upload(&mut d_i, h_i)?;
    s0.upload(&mut d_rec_begin, h_rec_begin)?;
    s0.upload(&mut d_pairs, h_pairs)?;
    s0.upload(&mut d_aux, h_aux)?;
    // Each class's records, formed on the device from the tables.
    let mut class_f = Vec::with_capacity(d.classes.len());
    for (c, class) in d.classes.iter().enumerate() {
        let mut f = scratch.device(g, sl.class_f[c]);
        let mut i = scratch.device(g, sl.class_i[c]);
        let source = PairSource {
            n_pairs: class.n_pair,
            n_shell,
            n_prim,
            f: &d_f,
            i: &d_i,
            rec_begin: &part(&d_rec_begin, first_pair[c] + c, class.n_pair + 1),
            pairs: &part(&d_pairs, 2 * first_pair[c], 2 * class.n_pair),
        };
        launch::mmd_records(
            s0,
            &format!("fb_mmd_records ({}, {})", class.la, class.lb),
            class.n_rec,
            &source,
            &mut f,
            &mut i,
        )?;
        class_f.push(f);
    }
    // The events: three per ring buffer and one after the records, made
    // on the first run and kept.
    if st.events.is_empty() {
        for _ in 0..3 * depth + 1 {
            st.events.push(Event::new(lane.physical())?);
        }
    }
    let ready = &st.events[3 * depth];
    ready.record(s0)?;
    for s in 1..depth {
        // A ring buffer's stream reads the records the first one formed.
        lane.stream(s).wait(ready)?;
    }

    // The metric's factor, on the GPU's stream before the loop: it
    // overlaps the first batch's kernels.
    let mut tile = scratch.device(g, sl.tile);
    let mut flag = scratch.device(g, sl.flag);
    let h_flag = scratch.pinned(g, sl.h_flag);
    gpu.stream().copy_device(l, metric)?;
    gpu.cholesky(q, l, &mut tile, &mut flag)?;
    gpu.stream().download(h_flag, &flag)?;

    // The pair batches, through the ring.
    let ring = ring_layout(d, width, depth);
    let mut in_flight: Vec<Option<(Batch, usize)>> = vec![None; depth];
    let land = |s: usize, flight: (Batch, usize)| -> Status {
        // The host waits on the ring buffer's last event, not a stream.
        st.events[3 * s + 2].sync()?;
        let (batch, ws) = flight;
        let h_batch: &[f64] = st.pinned.tenant(&ring.slots[s].h_batch);
        b.scatter(batch.first_col, ws, &h_batch[..q * ws]);
        Ok(())
    };
    for (turn, batch) in mine.enumerate() {
        let s = turn % depth;
        let stream = lane.stream(s);
        if let Some(flight) = in_flight[s].take() {
            land(s, flight)?;
        }
        let class = &d.classes[batch.class];
        let ns = class.sph();
        let ws = batch.n_pair * ns;
        assert!(
            q * ws <= i32::MAX as usize,
            "a batch fits the kernels' index"
        );
        // The cells: this batch's pairs crossed with every aux shell, by
        // aux momentum, each destination relative to the batch. Written
        // into the ring buffer's pinned bytes: its last download is done.
        let h_cells = st.pinned.tenant(&ring.slots[s].h_cells);
        let base = first_pair[batch.class];
        let rec_begin = &h_rec_begin[base + batch.class..];
        let n_class = class.n_pair;
        let mut launches = Vec::with_capacity(aux_of_l.len());
        let mut first = 0;
        for (lp, shells) in aux_of_l.iter().enumerate() {
            if shells.is_empty() {
                continue;
            }
            let n_cell = batch.n_pair * shells.len();
            let mut k = 0;
            for i in 0..batch.n_pair {
                let p = batch.first_pair + i;
                let diagonal = h_pairs[2 * base + p] == h_pairs[2 * base + n_class + p];
                for &sh in shells {
                    let fields = [
                        rec_begin[p],
                        rec_begin[p + 1],
                        aux_rec[sh],
                        aux_rec[sh + 1],
                        (i * ns * q) as i32 + aux_sph[sh],
                        i32::from(diagonal),
                    ];
                    for (f, v) in fields.into_iter().enumerate() {
                        h_cells[CELL_FIELDS * first + f * n_cell + k] = v;
                    }
                    k += 1;
                }
            }
            launches.push((lp as i32, first, n_cell));
            first += n_cell;
        }
        let mut cells = st.buf.tenant(&ring.slots[s].cells);
        let mut tensor = st.buf.tenant(&ring.slots[s].batch);
        stream.upload(&mut cells, &h_cells[..CELL_FIELDS * first])?;
        let view = launch::class_view(class.la, class.lb, class.n_rec, &class_f[batch.class]);
        for (lp, first, n_cell) in launches {
            launch::ssperi3(
                stream,
                &view,
                lp,
                n_aux_rec,
                &d_aux,
                n_cell,
                &part(&cells, CELL_FIELDS * first, CELL_FIELDS * n_cell),
                ws,
                q,
                &mut tensor,
            )?;
        }
        let (kernels, fit, down) = (
            &st.events[3 * s],
            &st.events[3 * s + 1],
            &st.events[3 * s + 2],
        );
        kernels.record(stream)?;
        // The fit, in place, on the GPU's stream: B ← L⁻¹ B, the tree's
        // own blocked solve, after this batch's kernels.
        gpu.stream().wait(kernels)?;
        let mut fitted = part_mut(&mut tensor, 0, q * ws);
        gpu.trsm_lower(
            true,
            false,
            q as i32,
            ws as i32,
            1.0,
            &*l,
            q as i32,
            &mut fitted,
            q as i32,
        )?;
        fit.record(gpu.stream())?;
        // The download, on the ring buffer's stream, after the fit.
        stream.wait(fit)?;
        let h_batch = st.pinned.tenant(&ring.slots[s].h_batch);
        stream.download(&mut h_batch[..q * ws], &fitted)?;
        down.record(stream)?;
        in_flight[s] = Some((*batch, ws));
    }
    for (s, flight) in in_flight.into_iter().enumerate() {
        if let Some(flight) = flight {
            land(s, flight)?;
        }
    }
    // The factorization's flag is read on the host: the GPU's stream is
    // done by now but for the flag's copy.
    gpu.stream().sync()?;
    if h_flag[0] != 0 {
        return Err(Error::numerical(format!(
            "eri: on gpu {g} the metric's leading minor {} is not positive",
            h_flag[0]
        )));
    }

    // The ring is done: the ERI buffer is now a J/K pass's. Its gather
    // tables are written once here; the cache fills on the first pass.
    let jk = jk_layout(d, 1, 0);
    let (h_rows, h_cols, h_weights) = (
        st.pinned.tenant(&jk.h_rows),
        st.pinned.tenant(&jk.h_cols),
        st.pinned.tenant(&jk.h_weights),
    );
    let first_col: Vec<usize> = d
        .classes
        .iter()
        .scan(0, |at, c| {
            let first = *at;
            *at += c.n_pair * c.sph();
            Some(first)
        })
        .collect();
    let mut filled = vec![0usize; d.classes.len()];
    for_each_pair(basis, |la, lb, bra, ket| {
        let c = d.class_of(la, lb);
        let (na, nb) = (n_sph_of(la), n_sph_of(lb));
        let (off_a, off_b) = (
            basis.ao_offset_sphe(bra) as i32,
            basis.ao_offset_sphe(ket) as i32,
        );
        let mut col = first_col[c] + filled[c] * na * nb;
        for ia in 0..na as i32 {
            for ib in 0..nb as i32 {
                h_rows[col] = off_a + ia;
                h_cols[col] = off_b + ib;
                h_weights[col] = if bra == ket { 1.0 } else { 2.0 };
                col += 1;
            }
        }
        filled[c] += 1;
    });
    s0.upload(&mut st.buf.tenant(&jk.rows), h_rows)?;
    s0.upload(&mut st.buf.tenant(&jk.cols), h_cols)?;
    s0.upload(&mut st.buf.tenant(&jk.weights), h_weights)?;
    // A replay rebuilds B: what the cache held is the last geometry's.
    st.cache_filled = false;
    Ok(())
}
