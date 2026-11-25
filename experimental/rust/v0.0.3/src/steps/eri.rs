//! The density-fitted integrals, one all-GPU step: the metric (P|Q) and
//! its Cholesky factor L, resident on every card, and
//!
//! ```text
//! B[Q][mn] = Σ_P L⁻¹[Q][P] (P|mn)
//! ```
//!
//! built in **batches of orbital shell pairs** and kept on the host heap,
//! never whole on a card. A column mn of B depends on every P, so a batch
//! of P cannot be fitted alone; a batch of pairs can. Each card takes its
//! share of the batches, forms (P|mn), fits it against its own copy of L,
//! and downloads the batch. No card needs anything from another.
//!
//! B's layout is the tree's packed one
//! (`fb_integrals_eri_density_fitting_pair_map`): `n_aux` rows of `n_col`
//! spherical pair columns, pair-fastest, element `(column, Q)` at
//! `column + n_col · Q`, every shell pair kept (no screen), the classes in
//! the tables' order.
//!
//! One batch, on its card's stream: the cells written into pinned scratch
//! and copied up; the packed `speri3` kernels, one launch per aux
//! momentum, the aux index already spherical; the orbital pair's
//! Cartesian-to-spherical transform, one batched gemm against the class's
//! `U_a ⊗ U_b`; the fit, `fb_recipes_sys::trsm` against L, in place; the
//! download; a sync; and the host's scatter into B's rows.
//!
//! The solve is the tree's own: the Cholesky tile kernel with
//! `fb_recipes_sys::trsm` on the panel and a syrk, as
//! `rust/recipes/cholesky` drives it, and `fb_recipes_sys::trsm` for the
//! fit. No vendor trsm, potrs or getrs.
//!
//! **Crudest form.** One batch in flight per card, with a blocking sync
//! per batch; the transpose is the host's scatter, not a device `geam`;
//! every card computes its own metric and factor.

use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::blas::{geam, gemm, gemm_strided_batched, syrk};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at, split_at_mut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_integrals_sys::{fb_metric_class, fb_three_center_packed_class};
use fb_recipes_sys::{FB_CHOLESKY_TILE, cholesky_diagonal, symmetrize, trsm};
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_side_mode};

use crate::budget::{Budget, DeviceBuf, HeapSlab, Kind, Layout, Lease, Span};
use crate::bus::{Dims, MainBus, Shared, n_cart_of, n_sph_of};
use crate::device::{CardLane, Lane, download, launched, upload};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::records::{RecordsOutput, RecordsStep};
use crate::steps::{OP_N, OP_T, fan_out, limit, widest};

const LOWER: wwr_blas_fill_mode = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
const NON_UNIT: wwr_blas_diag_type = wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT;
const LEFT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
const RIGHT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;
const BLOCK: usize = FB_CHOLESKY_TILE as usize;
/// The fields of an aux record, of a metric pair and of a packed cell.
const AUX_FIELDS: usize = 5;
const PAIR_FIELDS: usize = 6;
const CELL_FIELDS: usize = 6;

/// The step.
#[derive(Clone, Copy)]
pub struct EriStep;

/// One card's metric and factor: O(N²), resident whole.
pub struct CardMetric {
    /// M = (P|Q), spherical, symmetric.
    metric: DeviceBuf<f64>,
    /// L in the lower triangle, `M = L Lᵀ`.
    factor: DeviceBuf<f64>,
}

impl CardMetric {
    /// The metric.
    pub fn metric(&self) -> &DeviceBuf<f64> {
        &self.metric
    }

    /// Its Cholesky factor.
    pub fn factor(&self) -> &DeviceBuf<f64> {
        &self.factor
    }
}

/// The fitted integrals: the metric and its factor on every card, and B on
/// the host heap. An ordinary `Output`: steps read it through their view.
pub struct EriOutput {
    n_aux: usize,
    n_col: usize,
    width: usize,
    batches: usize,
    cards: Vec<CardMetric>,
    b: HeapSlab<f64>,
}

impl EriOutput {
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

    /// `card`'s metric and factor.
    pub fn card(&self, card: usize) -> &CardMetric {
        &self.cards[card]
    }

    /// The batch width the step's limit picked, in Cartesian pair columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The batches of pairs B was built in.
    pub fn batches(&self) -> usize {
        self.batches
    }
}

/// One card's scratch, laid out at a batch width: the step's one layout
/// function, for `mem_budget`'s bytes and `run`'s spans.
struct EriLayout {
    device_bytes: usize,
    pinned_bytes: usize,
    aux: Span<f64>,
    pairs: Span<i32>,
    u_aux: Span<f64>,
    metric_cart: Span<f64>,
    metric_half: Span<f64>,
    tile: Span<f64>,
    flag: Span<i32>,
    kron: Span<f64>,
    cells: Span<i32>,
    cart: Span<f64>,
    sph: Span<f64>,
    h_aux: Span<f64>,
    h_pairs: Span<i32>,
    h_u_aux: Span<f64>,
    h_kron: Span<f64>,
    h_flag: Span<i32>,
    h_cells: Span<i32>,
    h_down: Span<f64>,
}

/// The pairs of class `c` in one batch of `w` Cartesian columns.
fn pairs_per_batch(d: &Dims, c: usize, w: usize) -> usize {
    (w / d.classes[c].cart()).clamp(1, d.classes[c].n_pair.max(1))
}

fn layout(d: &Dims, w: usize) -> EriLayout {
    let (q, q_cart) = (d.n_aux, d.n_aux_cart);
    let n_metric_pair = d.n_aux_shell * (d.n_aux_shell + 1) / 2;
    let kron: usize = d.classes.iter().map(|c| c.cart() * c.sph()).sum();
    // A batch's cells: its pairs crossed with every aux shell.
    let cells = (0..d.classes.len())
        .map(|c| pairs_per_batch(d, c, w) * d.n_aux_shell)
        .max()
        .unwrap_or(0);
    let tile = q.min(BLOCK);
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    EriLayout {
        aux: dev.span(AUX_FIELDS * d.n_aux_prim),
        pairs: dev.span(PAIR_FIELDS * n_metric_pair),
        u_aux: dev.span(q_cart * q),
        metric_cart: dev.span(q_cart * q_cart),
        metric_half: dev.span(q_cart * q),
        tile: dev.span(tile * tile),
        flag: dev.span(1),
        kron: dev.span(kron),
        cells: dev.span(CELL_FIELDS * cells),
        cart: dev.span(q * w),
        sph: dev.span(q * w),
        h_aux: pin.span(AUX_FIELDS * d.n_aux_prim),
        h_pairs: pin.span(PAIR_FIELDS * n_metric_pair),
        h_u_aux: pin.span(q_cart * q),
        h_kron: pin.span(kron),
        h_flag: pin.span(1),
        h_cells: pin.span(CELL_FIELDS * cells),
        h_down: pin.span(q * w),
        device_bytes: dev.bytes(),
        pinned_bytes: pin.bytes(),
    }
}

/// A batch: a run of one class's pairs, and where its columns lie in B.
#[derive(Clone, Copy, Debug)]
struct Batch {
    class: usize,
    first_pair: usize,
    n_pair: usize,
    first_col: usize,
}

/// The batches at width `w`, class by class: a batch never spans two.
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
    out
}

/// B's rows, written by several cards at once, each its own columns.
struct Columns {
    ptr: *mut f64,
    n_col: usize,
    n_aux: usize,
}

// SAFETY: every batch owns its columns of B and no other batch's, so the
// cards' writes never meet (`Columns::scatter`).
unsafe impl Sync for Columns {}

impl Columns {
    /// Scatter a fitted batch, `n_aux × columns` with the aux index
    /// fastest, into B's columns from `first`: the transpose.
    fn scatter(&self, first: usize, columns: usize, batch: &[f64]) {
        assert!(first + columns <= self.n_col && batch.len() >= self.n_aux * columns);
        for c in 0..columns {
            for q in 0..self.n_aux {
                // SAFETY: inside B, `n_col × n_aux`; columns
                // `[first, first + columns)` are this batch's alone.
                unsafe {
                    *self.ptr.add(first + c + self.n_col * q) = batch[q + self.n_aux * c];
                }
            }
        }
    }
}

/// Small host data `run` derives from the two bases: O(shells) and one
/// transform block per momentum. Host scratch within the policy.
struct Host {
    /// Each aux shell's first record (`n_aux_shell + 1` long), momentum,
    /// first Cartesian function and first spherical function.
    rec_begin: Vec<i32>,
    momentum: Vec<usize>,
    cart_offset: Vec<i32>,
    sph_offset: Vec<i32>,
    /// The aux shells of each momentum.
    of_l: Vec<Vec<usize>>,
    /// Per orbital momentum, U's block: `n_cart × n_sph`, column-major.
    u: Vec<Vec<f64>>,
}

impl Host {
    fn new(shared: &Shared) -> Host {
        let aux = shared.aux.as_ref().expect("eri runs with a fitting set");
        let (mut rec_begin, mut momentum) = (vec![0i32], Vec::new());
        let (mut cart_offset, mut sph_offset) = (Vec::new(), Vec::new());
        let mut of_l = vec![Vec::new(); shared.dims.aux_shells_of_l.len()];
        for (s, shell) in (0u32..).zip(aux.shells()) {
            rec_begin.push(rec_begin[s as usize] + shell.nprimitives() as i32);
            momentum.push(shell.angular_momentum() as usize);
            cart_offset.push(aux.ao_offset_cart(s) as i32);
            sph_offset.push(aux.ao_offset_sphe(s) as i32);
            of_l[shell.angular_momentum() as usize].push(s as usize);
        }
        let basis = &shared.basis;
        let n_cart = basis.nao_cart() as usize;
        let mut full = vec![0.0; n_cart * basis.nao_sphe() as usize];
        fb_basis::spherical_transform_matrix(basis, &mut full);
        let mut u: Vec<Vec<f64>> = Vec::new();
        for (s, shell) in (0u32..).zip(basis.shells()) {
            let l = shell.angular_momentum();
            if u.len() <= l as usize {
                u.resize(l as usize + 1, Vec::new());
            }
            if u[l as usize].is_empty() {
                let (nc, ns) = (n_cart_of(l), n_sph_of(l));
                let (r0, c0) = (
                    basis.ao_offset_cart(s) as usize,
                    basis.ao_offset_sphe(s) as usize,
                );
                let mut block = vec![0.0; nc * ns];
                for sp in 0..ns {
                    for c in 0..nc {
                        block[c + nc * sp] = full[(r0 + c) + n_cart * (c0 + sp)];
                    }
                }
                u[l as usize] = block;
            }
        }
        Host {
            rec_begin,
            momentum,
            cart_offset,
            sph_offset,
            of_l,
            u,
        }
    }
}

impl Step<MainBus> for EriStep {
    const NAME: &'static str = "eri";
    type Output = EriOutput;
    type Reads = (RecordsStep,);

    fn slot(bus: &MainBus) -> &Slot<EriOutput, Lane> {
        &bus.eri
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<EriOutput> {
        let d = &shared.dims;
        let q = d.n_aux;
        let table = shared.config.eri.as_ref().expect("eri runs with its table");
        // The width: the widest batch whose scratch fits the step's own
        // limit on the tightest card. One batch per class is the most.
        let cap = (0..budget.cards())
            .map(|c| budget.device_cap(c))
            .min()
            .unwrap_or(u64::MAX);
        let least = d.classes.iter().map(|c| c.cart()).max().unwrap_or(1);
        let most = d
            .classes
            .iter()
            .map(|c| c.n_pair * c.cart())
            .max()
            .unwrap_or(1)
            .min(i32::MAX as usize / q.max(1));
        let width = widest(Self::NAME, limit(table.limit_mib, cap), least, most, |w| {
            layout(d, w).device_bytes as u64
        })?;
        let at = layout(d, width);
        let cards = (0..budget.cards())
            .map(|card| {
                budget.scratch(card, Kind::Device, at.device_bytes);
                budget.scratch(card, Kind::Pinned, at.pinned_bytes);
                budget.library(card);
                CardMetric {
                    metric: budget.device(card, q * q),
                    factor: budget.device(card, q * q),
                }
            })
            .collect();
        Ok(EriOutput {
            n_aux: q,
            n_col: d.n_col,
            width,
            batches: batches(d, width).len(),
            cards,
            // B's home: one slab on the host heap, at the unscreened bound.
            b: budget.heap(d.n_col * q),
        })
    }

    fn run(
        out: &mut EriOutput,
        view: &View<'_, MainBus, Self>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let records = view.read(RecordsStep);
        let host = Host::new(shared);
        let all = batches(&shared.dims, out.width);
        let n_cards = lane.cards.len();
        let b = Columns {
            ptr: out.b.as_mut_ptr(),
            n_col: out.n_col,
            n_aux: out.n_aux,
        };
        let width = out.width;
        // Whoever fans out, joins: every card's work is finished, and its
        // batches are in B, before this returns.
        fan_out(
            lane.cards.iter_mut().zip(out.cards.iter_mut()).collect(),
            |(lane, metric)| {
                let mine = all.iter().skip(lane.card()).step_by(n_cards);
                card_run(
                    shared, &records, &host, lane, metric, scratch, width, mine, &b,
                )
            },
        )?;
        Ok(())
    }
}

/// One card's part: its metric and factor, then its batches.
#[allow(
    clippy::too_many_arguments,
    reason = "a step's inner task, spelled out"
)]
fn card_run<'a>(
    shared: &Shared,
    records: &RecordsOutput,
    host: &Host,
    lane: &mut CardLane,
    out: &mut CardMetric,
    scratch: &Lease<'_>,
    width: usize,
    batches: impl Iterator<Item = &'a Batch>,
    b: &Columns,
) -> Status {
    let _current = lane.current();
    let (card, stream, blas) = (lane.card(), lane.stream(), lane.blas());
    let d = &shared.dims;
    let aux = shared.aux.as_ref().expect("eri runs with a fitting set");
    let (q, q_cart, n_rec) = (d.n_aux, d.n_aux_cart, d.n_aux_prim);
    let l = layout(d, width);

    // The records and work lists, built straight into pinned scratch.
    let h_aux = scratch.pinned(card, l.h_aux);
    let mut r = 0;
    for shell in aux.shells() {
        let c = shell.center();
        for p in shell.primitives() {
            let weight = p.coefficient * (std::f64::consts::PI / p.exponent).powf(1.5);
            for (f, v) in [c.x, c.y, c.z, p.exponent, weight].into_iter().enumerate() {
                h_aux[f * n_rec + r] = v;
            }
            r += 1;
        }
    }
    // The metric's aux shell pairs P <= Q, by class lp <= lq: a counting
    // pass gives each class's first pair, so no temporary is needed.
    let h_pairs = scratch.pinned(card, l.h_pairs);
    let n_l = host.of_l.len();
    let class_of = |i: usize, j: usize| {
        let (p, q) = if host.momentum[i] > host.momentum[j] {
            (j, i)
        } else {
            (i, j)
        };
        (host.momentum[p] * n_l + host.momentum[q], p, q)
    };
    let mut count = vec![0usize; n_l * n_l];
    for i in 0..d.n_aux_shell {
        for j in i..d.n_aux_shell {
            count[class_of(i, j).0] += 1;
        }
    }
    let mut first = vec![0usize; n_l * n_l];
    for c in 1..n_l * n_l {
        first[c] = first[c - 1] + count[c - 1];
    }
    let mut filled = vec![0usize; n_l * n_l];
    for i in 0..d.n_aux_shell {
        for j in i..d.n_aux_shell {
            let (c, p, q) = class_of(i, j);
            let fields = [
                host.rec_begin[p],
                host.rec_begin[p + 1],
                host.rec_begin[q],
                host.rec_begin[q + 1],
                host.cart_offset[p],
                host.cart_offset[q],
            ];
            for (f, v) in fields.into_iter().enumerate() {
                h_pairs[PAIR_FIELDS * first[c] + f * count[c] + filled[c]] = v;
            }
            filled[c] += 1;
        }
    }
    let h_u_aux = scratch.pinned(card, l.h_u_aux);
    fb_basis::spherical_transform_matrix(aux, h_u_aux);
    // Each class's U_a ⊗ U_b: element ((a, b), (s, t)) is U_a[a, s] U_b[b, t].
    let h_kron = scratch.pinned(card, l.h_kron);
    let mut kron_at = Vec::with_capacity(d.classes.len());
    let mut at = 0;
    for class in &d.classes {
        kron_at.push(at);
        let (ua, ub) = (&host.u[class.la as usize], &host.u[class.lb as usize]);
        let (nca, ncb) = (n_cart_of(class.la), n_cart_of(class.lb));
        let (nsa, nsb) = (n_sph_of(class.la), n_sph_of(class.lb));
        for s in 0..nsa {
            for t in 0..nsb {
                for a in 0..nca {
                    for bb in 0..ncb {
                        h_kron[at + (a * ncb + bb) + nca * ncb * (s * nsb + t)] =
                            ua[a + nca * s] * ub[bb + ncb * t];
                    }
                }
            }
        }
        at += class.cart() * class.sph();
    }

    let mut aux_dev = scratch.device(card, l.aux);
    let mut pairs_dev = scratch.device(card, l.pairs);
    let mut u_aux = scratch.device(card, l.u_aux);
    let mut kron = scratch.device(card, l.kron);
    upload(&mut aux_dev, h_aux, stream)?;
    upload(&mut pairs_dev, h_pairs, stream)?;
    upload(&mut u_aux, h_u_aux, stream)?;
    upload(&mut kron, h_kron, stream)?;

    // The Cartesian metric (P|Q): plain stores, each with its mirror.
    let mut metric_cart = scratch.device(card, l.metric_cart);
    for c in (0..n_l * n_l).filter(|&c| count[c] > 0) {
        let (lp, lq) = ((c / n_l) as i32, (c % n_l) as i32);
        // SAFETY: the records and pairs are this step's uploads, laid out
        // as the kernel's header says, every pair's block inside the
        // n_aux_cart² metric; the metric is this step's scratch, of that
        // size by the layout.
        let status = unsafe {
            fb_metric_class(
                stream.stream(),
                lp,
                lq,
                n_rec as i32,
                aux_dev.as_ptr(),
                count[c] as i32,
                pairs_dev.as_ptr().wrapping_add(PAIR_FIELDS * first[c]),
                q_cart as i32,
                metric_cart.as_mut_ptr(),
            )
        };
        launched(&format!("fb_metric_class ({lp}, {lq})"), status)?;
    }

    // M = U_auxᵀ (P|Q) U_aux, averaged with its transpose; then a copy of
    // it factored in place, M = L Lᵀ.
    let (qi, qc) = (q as i32, q_cart as i32);
    let mut metric_half = scratch.device(card, l.metric_half);
    gemm(
        blas,
        OP_N,
        OP_N,
        qc,
        qi,
        qc,
        1.0,
        &metric_cart,
        qc,
        &u_aux,
        qc,
        0.0,
        &mut metric_half,
        qc,
    )?;
    gemm(
        blas,
        OP_T,
        OP_N,
        qi,
        qi,
        qc,
        1.0,
        &u_aux,
        qc,
        &metric_half,
        qc,
        0.0,
        &mut out.metric,
        qi,
    )?;
    symmetrize(stream, q, &mut out.metric, q)?;
    geam(
        blas,
        OP_N,
        OP_N,
        qi,
        qi,
        1.0,
        &out.metric,
        qi,
        0.0,
        &out.metric,
        qi,
        &mut out.factor,
        qi,
    )?;
    let mut tile = scratch.device(card, l.tile);
    let mut flag = scratch.device(card, l.flag);
    memset::zero(&mut flag, stream)?;
    // The tree's blocked right-looking Cholesky (rust/recipes/cholesky):
    // the diagonal tile by a kernel, the panel by the tree's own trsm, the
    // trailing matrix by a syrk.
    for j in (0..q).step_by(BLOCK) {
        let jb = BLOCK.min(q - j);
        {
            let (_, mut diagonal) = split_at_mut(&mut out.factor, j * (q + 1));
            cholesky_diagonal(stream, jb, j, &mut diagonal, q, &mut tile, &mut flag)?;
        }
        let rest = q - j - jb;
        if rest == 0 {
            break;
        }
        let (mut head, mut trailing) = split_at_mut(&mut out.factor, (j + jb) * (q + 1));
        let (_, mut panel) = split_at_mut(&mut head, j * (q + 1) + jb);
        let (jb, rest) = (jb as i32, rest as i32);
        trsm(
            blas, RIGHT, LOWER, OP_T, NON_UNIT, rest, jb, 1.0, &tile, jb, &mut panel, qi,
        )?;
        syrk(
            blas,
            LOWER,
            OP_N,
            rest,
            jb,
            -1.0,
            &panel,
            qi,
            1.0,
            &mut trailing,
            qi,
        )?;
    }
    // The factorization's flag is read on the host: the card waits.
    let h_flag = scratch.pinned(card, l.h_flag);
    download(h_flag, &flag, stream)?;
    lane.sync()?;
    if h_flag[0] != 0 {
        return Err(Error::numerical(format!(
            "eri: on card {card} the metric's leading minor {} is not positive",
            h_flag[0]
        )));
    }

    // The batches of pairs.
    let h_cells = scratch.pinned(card, l.h_cells);
    let h_down = scratch.pinned(card, l.h_down);
    let mut cells = scratch.device(card, l.cells);
    let mut cart = scratch.device(card, l.cart);
    let mut sph = scratch.device(card, l.sph);
    for batch in batches {
        let class = &d.classes[batch.class];
        let pairs = &shared.tables.classes()[batch.class];
        let (nc, ns) = (class.cart(), class.sph());
        let (wc, ws) = (batch.n_pair * nc, batch.n_pair * ns);

        // The cells: this batch's pairs crossed with every aux shell, by
        // aux momentum, each destination relative to the batch.
        let mut launches = Vec::with_capacity(host.of_l.len());
        let mut first = 0;
        for (lp, shells) in host.of_l.iter().enumerate() {
            if shells.is_empty() {
                continue;
            }
            let n_cell = batch.n_pair * shells.len();
            let mut k = 0;
            for i in 0..batch.n_pair {
                let p = batch.first_pair + i;
                for &sh in shells {
                    let fields = [
                        pairs.rec_begin[p],
                        pairs.rec_begin[p + 1],
                        host.rec_begin[sh],
                        host.rec_begin[sh + 1],
                        (i * nc * q) as i32 + host.sph_offset[sh],
                        i32::from(pairs.bra[p] == pairs.ket[p]),
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
        upload(&mut cells, &h_cells[..CELL_FIELDS * first], stream)?;
        let view = records.classes(card)[batch.class].view();
        for (lp, first, n_cell) in launches {
            // SAFETY: the class's records are the records step's on this
            // card, finished before this step ran; the aux records and the
            // cells are this step's uploads, each cell's block inside the
            // batch's `n_aux × wc` tensor; the tensor is this step's
            // scratch, `n_aux × width` by the layout, and `wc <= width`.
            let status = unsafe {
                fb_three_center_packed_class(
                    stream.stream(),
                    &view,
                    lp,
                    n_rec as i32,
                    aux_dev.as_ptr(),
                    n_cell as i32,
                    cells.as_ptr().wrapping_add(CELL_FIELDS * first),
                    wc as i32,
                    qi,
                    cart.as_mut_ptr(),
                )
            };
            launched(
                &format!(
                    "fb_three_center_packed_class ({}, {}, {lp})",
                    class.la, class.lb
                ),
                status,
            )?;
        }
        // The pair index to spherical: each pair's n_aux × nc block times
        // the class's U_a ⊗ U_b.
        let (_, kron_class) = split_at(&kron, kron_at[batch.class]);
        gemm_strided_batched(
            blas,
            OP_N,
            OP_N,
            qi,
            ns as i32,
            nc as i32,
            1.0,
            &cart,
            qi,
            (q * nc) as i64,
            &kron_class,
            nc as i32,
            0,
            0.0,
            &mut sph,
            qi,
            (q * ns) as i64,
            batch.n_pair as i32,
        )?;
        // The fit, in place: B ← L⁻¹ B, the tree's own blocked solve.
        trsm(
            blas,
            LEFT,
            LOWER,
            OP_N,
            NON_UNIT,
            qi,
            ws as i32,
            1.0,
            &out.factor,
            qi,
            &mut sph,
            qi,
        )?;
        download(&mut h_down[..q * ws], &sph, stream)?;
        // One batch in flight: the pinned bytes are the next batch's too.
        lane.sync()?;
        b.scatter(batch.first_col, ws, &h_down[..q * ws]);
    }
    Ok(())
}
