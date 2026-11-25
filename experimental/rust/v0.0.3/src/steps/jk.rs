//! The DF J and K build, a batched all-GPU step over **batches of aux
//! functions P**:
//!
//! ```text
//! J[m, n] = Σ_Q B_Q[m, n] Σ_ls B_Q[l, s] D[l, s]
//! K[m, n] = Σ_Q (B_Q D B_Q)[m, n]
//! ```
//!
//! B is on the host heap; each card takes its share of the batches,
//! streams them up, contracts them into its own fragments of J and K, and
//! the fragments are reduced onto gpu0.
//!
//! The shared code is [`build_jk`]: a plain function over a view of B, the
//! density, the J and K to write (its caller's), its caller's scratch and
//! the lanes of the cards its caller holds. It has no state, so no lock of
//! its own. The callee gives the size ([`JkLayout`]), the caller declares
//! it: [`JkStep`] is the one caller here, and adds nothing of its own.
//!
//! **Crudest form.** One batch in flight per card, a blocking sync per
//! batch; every pass uploads all of B; the reduction goes through the
//! host (each card's fragments downloaded, summed, and the sum copied up
//! to gpu0), with no collectives; K is built from D, with no occupied
//! factor.

use fb_core_error_handling::{Result, Status};
use fb_core_wwr::blas::{gemm, gemm_strided_batched, gemv};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at_mut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_integrals_sys::fb_packed_gather;

use crate::budget::{Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, ScratchBuf, Span};
use crate::bus::{Dims, MainBus, Shared, n_sph_of};
use crate::device::{CardLane, Lane, download, launched, upload};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::eri::EriStep;
use crate::steps::guess::GuessStep;
use crate::steps::{GPU0, OP_N, OP_T, fan_out, limit, widest};

/// The step.
#[derive(Clone, Copy)]
pub struct JkStep;

/// J and K of the guess density, on gpu0, and a copy on the host heap for
/// whoever holds the plan after the run (a test; an SCF would not want it).
pub struct JkOutput {
    width: usize,
    batches: usize,
    j: DeviceBuf<f64>,
    k: DeviceBuf<f64>,
    j_host: HeapSlab<f64>,
    k_host: HeapSlab<f64>,
}

impl JkOutput {
    /// J on gpu0, `n_ao × n_ao`.
    pub fn j(&self) -> &DeviceBuf<f64> {
        &self.j
    }

    /// K on gpu0.
    pub fn k(&self) -> &DeviceBuf<f64> {
        &self.k
    }

    /// J on the host.
    pub fn j_host(&self) -> &[f64] {
        &self.j_host
    }

    /// K on the host.
    pub fn k_host(&self) -> &[f64] {
        &self.k_host
    }

    /// The batch width the step's limit picked, in aux functions.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The batches of aux functions of one pass.
    pub fn batches(&self) -> usize {
        self.batches
    }
}

/// One card's scratch for a J and K pass at a batch width: the callee's
/// size function and its layout in one. Pure in the dimensions, the width
/// and whether the card is gpu0.
pub struct JkLayout {
    /// The device bytes: what the caller declares.
    pub device_bytes: usize,
    /// The pinned bytes.
    pub pinned_bytes: usize,
    rows: Span<i32>,
    cols: Span<i32>,
    weights: Span<f64>,
    packed: Span<f64>,
    squares: Span<f64>,
    bd: Span<f64>,
    gamma: Span<f64>,
    /// D, and the card's fragments of J and K: empty on gpu0, which reads
    /// the caller's D and writes the caller's J and K.
    d: Span<f64>,
    j: Span<f64>,
    k: Span<f64>,
    h_rows: Span<i32>,
    h_cols: Span<i32>,
    h_weights: Span<f64>,
    h_packed: Span<f64>,
    /// D staged on the host (gpu0's region holds it for every card), and
    /// the card's fragments brought down.
    h_d: Span<f64>,
    h_j: Span<f64>,
    h_k: Span<f64>,
}

impl JkLayout {
    /// The layout of `card` at a width of `p` aux functions.
    pub fn new(d: &Dims, p: usize, card: usize) -> JkLayout {
        let (n2, n_col) = (d.n_ao * d.n_ao, d.n_col);
        let own = if card == GPU0 { 0 } else { n2 };
        let (mut dev, mut pin) = (Layout::new(), Layout::new());
        JkLayout {
            rows: dev.span(n_col),
            cols: dev.span(n_col),
            weights: dev.span(n_col),
            packed: dev.span(p * n_col),
            squares: dev.span(p * n2),
            bd: dev.span(p * n2),
            gamma: dev.span(p),
            d: dev.span(own),
            j: dev.span(own),
            k: dev.span(own),
            h_rows: pin.span(n_col),
            h_cols: pin.span(n_col),
            h_weights: pin.span(n_col),
            h_packed: pin.span(p * n_col),
            h_d: pin.span(if card == GPU0 { n2 } else { 0 }),
            h_j: pin.span(n2),
            h_k: pin.span(n2),
            device_bytes: dev.bytes(),
            pinned_bytes: pin.bytes(),
        }
    }
}

/// Each packed column's `(m, n)` in the `n_ao × n_ao` matrix and its
/// weight, 1 on a diagonal shell pair's block and 2 on the others: what
/// `fb_packed_gather` reads. A function of the orbital basis alone.
pub fn write_gather_tables(
    shared: &Shared,
    rows: &mut [i32],
    cols: &mut [i32],
    weights: &mut [f64],
) {
    let basis = &shared.basis;
    let mut c = 0;
    for class in shared.tables.classes() {
        let (na, nb) = (n_sph_of(class.la) as i32, n_sph_of(class.lb) as i32);
        for (&a, &b) in class.bra.iter().zip(&class.ket) {
            let (off_a, off_b) = (
                basis.ao_offset_sphe(a) as i32,
                basis.ao_offset_sphe(b) as i32,
            );
            for ia in 0..na {
                for ib in 0..nb {
                    rows[c] = off_a + ia;
                    cols[c] = off_b + ib;
                    weights[c] = if a == b { 1.0 } else { 2.0 };
                    c += 1;
                }
            }
        }
    }
    assert_eq!(
        c, shared.dims.n_col,
        "the columns are the dimensions' n_col"
    );
}

/// Build J and K of the symmetric density `d` (on gpu0) over `b`, the host
/// B of `n_aux` rows, into the caller's `j` and `k` on gpu0 and their host
/// copies, in batches of `width` aux functions shared out over `lanes`,
/// working in the caller's `scratch` (declared from [`JkLayout`] at the
/// same width). It writes only what it is passed, and returns with every
/// card's fragments reduced: whoever fans out, joins.
///
/// # Errors
/// A launch's or a BLAS call's failure.
#[allow(clippy::too_many_arguments, reason = "the shared build, spelled out")]
pub fn build_jk(
    shared: &Shared,
    b: &[f64],
    n_aux: usize,
    width: usize,
    d: &DeviceBuf<f64>,
    j: &mut DeviceBuf<f64>,
    k: &mut DeviceBuf<f64>,
    j_host: &mut [f64],
    k_host: &mut [f64],
    lanes: &mut [CardLane],
    scratch: &Lease<'_>,
) -> Status {
    let dims = &shared.dims;
    let n_cards = lanes.len();
    let batches: Vec<(usize, usize)> = (0..n_aux)
        .step_by(width)
        .map(|q0| (q0, width.min(n_aux - q0)))
        .collect();
    let empty = || Layout::new().span::<f64>(0);
    let mut layouts: Vec<JkLayout> = (0..n_cards)
        .map(|c| JkLayout::new(dims, width, c))
        .collect();

    // D for the other cards goes through the host, in gpu0's pinned
    // region; the other cards only read it there.
    let h_d = scratch.pinned(GPU0, std::mem::replace(&mut layouts[GPU0].h_d, empty()));
    if n_cards > 1 {
        let gpu0 = &lanes[GPU0];
        let _current = gpu0.current();
        download(h_d, d, gpu0.stream())?;
        gpu0.sync()?;
    }
    let h_d: &[f64] = h_d;

    // Each card's fragments come down into its own pinned tenants.
    let mut down: Vec<Fragments<'_>> = layouts
        .iter_mut()
        .enumerate()
        .map(|(c, l)| Fragments {
            j: scratch.pinned(c, std::mem::replace(&mut l.h_j, empty())),
            k: scratch.pinned(c, std::mem::replace(&mut l.h_k, empty())),
        })
        .collect();
    // gpu0 writes the caller's J and K; the other cards their fragments.
    let mut targets: Vec<Option<(&mut DeviceBuf<f64>, &mut DeviceBuf<f64>)>> =
        (0..n_cards).map(|_| None).collect();
    targets[GPU0] = Some((j, k));
    let cards: Vec<_> = lanes
        .iter_mut()
        .zip(layouts)
        .zip(targets.iter_mut())
        .zip(down.iter_mut())
        .collect();
    fan_out(cards, |(((lane, l), own), down)| {
        let mine = batches.iter().skip(lane.card()).step_by(n_cards);
        card_pass(shared, b, d, h_d, lane, l, own, down, scratch, mine)
    })?;

    // The reduction onto gpu0, through the host: sum the fragments, keep
    // the host copy, and with more than one card copy the sums up.
    j_host.fill(0.0);
    k_host.fill(0.0);
    for fragment in &down {
        for (sum, x) in j_host.iter_mut().zip(fragment.j.iter()) {
            *sum += x;
        }
        for (sum, x) in k_host.iter_mut().zip(fragment.k.iter()) {
            *sum += x;
        }
    }
    if n_cards > 1 {
        let staged = &mut down[GPU0];
        staged.j.copy_from_slice(j_host);
        staged.k.copy_from_slice(k_host);
        let (j, k) = targets[GPU0]
            .take()
            .expect("gpu0 holds the caller's J and K");
        let gpu0 = &lanes[GPU0];
        let _current = gpu0.current();
        upload(j, staged.j, gpu0.stream())?;
        upload(k, staged.k, gpu0.stream())?;
    }
    Ok(())
}

/// A card's fragments of J and K, brought down to its pinned scratch.
struct Fragments<'a> {
    j: &'a mut [f64],
    k: &'a mut [f64],
}

/// One card's tenants of a pass.
struct Work<'a> {
    rows: ScratchBuf<'a, i32>,
    cols: ScratchBuf<'a, i32>,
    weights: ScratchBuf<'a, f64>,
    packed: ScratchBuf<'a, f64>,
    squares: ScratchBuf<'a, f64>,
    bd: ScratchBuf<'a, f64>,
    gamma: ScratchBuf<'a, f64>,
    h_packed: &'a mut [f64],
}

/// One card's pass: its tables, its D, its batches, its fragments down.
#[allow(
    clippy::too_many_arguments,
    reason = "a step's inner task, spelled out"
)]
fn card_pass<'a>(
    shared: &Shared,
    b: &[f64],
    d: &DeviceBuf<f64>,
    h_d: &[f64],
    lane: &mut CardLane,
    l: JkLayout,
    own: &mut Option<(&mut DeviceBuf<f64>, &mut DeviceBuf<f64>)>,
    down: &mut Fragments<'_>,
    scratch: &Lease<'_>,
    batches: impl Iterator<Item = &'a (usize, usize)>,
) -> Status {
    let _current = lane.current();
    let (card, stream) = (lane.card(), lane.stream());
    let (h_rows, h_cols, h_weights) = (
        scratch.pinned(card, l.h_rows),
        scratch.pinned(card, l.h_cols),
        scratch.pinned(card, l.h_weights),
    );
    write_gather_tables(shared, h_rows, h_cols, h_weights);
    let mut work = Work {
        rows: scratch.device(card, l.rows),
        cols: scratch.device(card, l.cols),
        weights: scratch.device(card, l.weights),
        packed: scratch.device(card, l.packed),
        squares: scratch.device(card, l.squares),
        bd: scratch.device(card, l.bd),
        gamma: scratch.device(card, l.gamma),
        h_packed: scratch.pinned(card, l.h_packed),
    };
    upload(&mut work.rows, h_rows, stream)?;
    upload(&mut work.cols, h_cols, stream)?;
    upload(&mut work.weights, h_weights, stream)?;
    match own {
        Some((j, k)) => contract(
            shared, b, d, &mut **j, &mut **k, lane, &mut work, down, batches,
        ),
        None => {
            let mut d_here = scratch.device(card, l.d);
            let (mut j, mut k) = (scratch.device(card, l.j), scratch.device(card, l.k));
            upload(&mut d_here, h_d, stream)?;
            contract(
                shared, b, &d_here, &mut j, &mut k, lane, &mut work, down, batches,
            )
        }
    }
}

/// A card's batches, contracted with `d` into `j` and `k` on that card,
/// and the two brought down.
#[allow(
    clippy::too_many_arguments,
    reason = "a step's inner task, spelled out"
)]
fn contract<'a>(
    shared: &Shared,
    b: &[f64],
    d: &impl DeviceSlice<f64>,
    j: &mut impl DeviceSliceMut<f64>,
    k: &mut impl DeviceSliceMut<f64>,
    lane: &CardLane,
    work: &mut Work<'_>,
    down: &mut Fragments<'_>,
    batches: impl Iterator<Item = &'a (usize, usize)>,
) -> Status {
    let (stream, blas) = (lane.stream(), lane.blas());
    let dims = &shared.dims;
    let (n, n_col) = (dims.n_ao as i32, dims.n_col);
    let n2 = n * n;
    memset::zero(j, stream)?;
    memset::zero(k, stream)?;
    for &(q0, p) in batches {
        // The batch of B, streamed from the host: aux functions
        // [q0, q0 + p) are one contiguous run of its rows.
        let rows = &b[n_col * q0..n_col * (q0 + p)];
        work.h_packed[..rows.len()].copy_from_slice(rows);
        upload(&mut work.packed, &work.h_packed[..rows.len()], stream)?;
        // Each row unpacked into its symmetric n_ao × n_ao square.
        let (mut squares, _) = split_at_mut(&mut work.squares, p * n2 as usize);
        memset::zero(&mut squares, stream)?;
        // SAFETY: the gather tables are this card's uploads, n_col rows,
        // columns and weights, every index inside n_ao; the packed batch
        // is p × n_col and the slab p × n_ao², both this step's scratch,
        // at least that long by the layout.
        let status = unsafe {
            fb_packed_gather(
                stream.stream(),
                n,
                n_col as i32,
                p as i32,
                work.rows.as_ptr(),
                work.cols.as_ptr(),
                work.weights.as_ptr(),
                work.packed.as_ptr(),
                squares.as_mut_ptr(),
            )
        };
        launched("fb_packed_gather", status)?;
        let p = p as i32;
        // gamma_Q = B_Q · D, then J += Σ_Q gamma_Q B_Q.
        gemv(
            blas,
            OP_T,
            n2,
            p,
            1.0,
            &squares,
            n2,
            d,
            1,
            0.0,
            &mut work.gamma,
            1,
        )?;
        gemv(
            blas,
            OP_N,
            n2,
            p,
            1.0,
            &squares,
            n2,
            &work.gamma,
            1,
            1.0,
            j,
            1,
        )?;
        // K += Σ_Q (B_Q D) B_Qᵀ: every B_Q D in one batched gemm, then the
        // blocks side by side as one n × (n p) matrix times B's.
        gemm_strided_batched(
            blas,
            OP_N,
            OP_N,
            n,
            n,
            n,
            1.0,
            &squares,
            n,
            i64::from(n2),
            d,
            n,
            0,
            0.0,
            &mut work.bd,
            n,
            i64::from(n2),
            p,
        )?;
        gemm(
            blas,
            OP_N,
            OP_T,
            n,
            n,
            n * p,
            1.0,
            &work.bd,
            n,
            &squares,
            n,
            1.0,
            k,
            n,
        )?;
        // One batch in flight: the pinned bytes are the next batch's too.
        lane.sync()?;
    }
    download(down.j, j, stream)?;
    download(down.k, k, stream)?;
    lane.sync()
}

impl Step<MainBus> for JkStep {
    const NAME: &'static str = "jk";
    type Output = JkOutput;
    type Reads = (EriStep, GuessStep);

    fn slot(bus: &MainBus) -> &Slot<JkOutput, Lane> {
        &bus.jk
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<JkOutput> {
        let d = &shared.dims;
        let n2 = d.n_ao * d.n_ao;
        let table = shared.config.jk.as_ref().expect("jk runs with its table");
        let cap = (0..budget.cards())
            .map(|c| budget.device_cap(c))
            .min()
            .unwrap_or(u64::MAX);
        // gpu0's layout is the smaller; the width is held to the larger.
        let last = budget.cards() - 1;
        let most = d.n_aux.clamp(1, i32::MAX as usize / n2.max(1));
        let width = widest(Self::NAME, limit(table.limit_mib, cap), 1, most, |p| {
            JkLayout::new(d, p, last).device_bytes as u64
        })?;
        // The callee gives the size, the caller declares it, a row per card.
        for card in 0..budget.cards() {
            let l = JkLayout::new(d, width, card);
            budget.scratch(card, Kind::Device, l.device_bytes);
            budget.scratch(card, Kind::Pinned, l.pinned_bytes);
            budget.library(card);
        }
        Ok(JkOutput {
            width,
            batches: d.n_aux.div_ceil(width),
            j: budget.device(GPU0, n2),
            k: budget.device(GPU0, n2),
            j_host: budget.heap(n2),
            k_host: budget.heap(n2),
        })
    }

    fn run(
        out: &mut JkOutput,
        view: &View<'_, MainBus, Self>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let eri = view.read(EriStep);
        let guess = view.read(GuessStep);
        build_jk(
            view.shared(),
            eri.b(),
            eri.n_aux(),
            out.width,
            guess.d(),
            &mut out.j,
            &mut out.k,
            &mut out.j_host,
            &mut out.k_host,
            &mut lane.cards,
            scratch,
        )
    }
}
