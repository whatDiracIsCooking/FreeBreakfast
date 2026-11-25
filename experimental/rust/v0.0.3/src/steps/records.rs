//! The primitive pair records of each shell-pair class of the orbital
//! basis, formed on every card from the host's shell tables: what every
//! integral kernel over AO pairs reads. An all-GPU step: each card builds
//! the same records for itself, so no card needs anything from another.
//!
//! The records are the `Output`, on every card. The shell tables they are
//! formed from are scratch, pinned for the staging and device for what
//! `fb_mmd_records` reads: `run` builds them straight into its pinned
//! region and uploads them, on every run.

use fb_core_error_handling::{Result, Status};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_sys::{
    fb_mmd_records, fb_primitive_pair_class_view, fb_primitive_pair_records,
    fb_primitive_pair_source,
};

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease, Span};
use crate::bus::{Dims, MainBus, Shared};
use crate::device::{Lane, launched, upload};
use crate::flow::{Placement, Slot, Step, View};

/// The f64 fields `fb_mmd_records` writes per record: zeta, P, PA, PB, s,
/// ea, eb.
const F64_FIELDS: usize = 13;
/// The i32 fields: off_a, off_b, rank_a, rank_b.
const I32_FIELDS: usize = 4;

/// The step.
#[derive(Clone, Copy)]
pub struct RecordsStep;

/// One class's records on one card: `F64_FIELDS` segments of `n_rec`
/// doubles and `I32_FIELDS` of `n_rec` ints.
pub struct ClassRecords {
    la: i32,
    lb: i32,
    n_rec: i32,
    f: DeviceBuf<f64>,
    i: DeviceBuf<i32>,
}

impl ClassRecords {
    /// What a class kernel takes: every field the records step formed, the
    /// three bounds null. The pointers are this step's device memory, valid
    /// while the reader holds its view.
    pub(crate) fn view(&self) -> fb_primitive_pair_class_view {
        let n = self.n_rec as usize;
        let f = |k: usize| self.f.as_ptr().wrapping_add(k * n);
        let i = |k: usize| self.i.as_ptr().wrapping_add(k * n);
        fb_primitive_pair_class_view {
            la: self.la,
            lb: self.lb,
            n_rec: self.n_rec,
            zeta: f(0),
            p_x: f(1),
            p_y: f(2),
            p_z: f(3),
            pa_x: f(4),
            pa_y: f(5),
            pa_z: f(6),
            pb_x: f(7),
            pb_y: f(8),
            pb_z: f(9),
            s: f(10),
            ea: f(11),
            eb: f(12),
            bound: std::ptr::null(),
            bound_ab: std::ptr::null(),
            bound_ba: std::ptr::null(),
            off_a: i(0),
            off_b: i(1),
            rank_a: i(2),
            rank_b: i(3),
        }
    }
}

/// Every card's records, the classes in the order of the host tables'.
pub struct RecordsOutput {
    cards: Vec<Vec<ClassRecords>>,
}

impl RecordsOutput {
    /// `card`'s classes.
    pub fn classes(&self, card: usize) -> &[ClassRecords] {
        &self.cards[card]
    }
}

/// The shell tables, as one region: the same layout for the pinned staging
/// and for the device copy.
struct Source {
    bytes: usize,
    /// exponent, coefficient (one per primitive), c_x, c_y, c_z (per shell).
    f: Span<f64>,
    /// prim_begin (shells + 1), ao_offset, rank (per shell), then each
    /// class's rec_begin (pairs + 1).
    i: Span<i32>,
    /// bra, ket (one per pair, the classes concatenated).
    u: Span<u32>,
}

/// The step's one layout function: `mem_budget` declares its bytes, `run`
/// carves its spans.
fn layout(d: &Dims) -> Source {
    let mut l = Layout::new();
    let f = l.span(2 * d.n_prim + 3 * d.n_shell);
    let i = l.span((d.n_shell + 1) + 2 * d.n_shell + d.n_shell_pair + d.classes.len());
    let u = l.span(2 * d.n_shell_pair);
    Source {
        bytes: l.bytes(),
        f,
        i,
        u,
    }
}

impl Step<MainBus> for RecordsStep {
    const NAME: &'static str = "records";
    type Output = RecordsOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<RecordsOutput, Lane> {
        &bus.records
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<RecordsOutput> {
        let d = &shared.dims;
        let cards = (0..budget.cards())
            .map(|card| {
                budget.scratch(card, Kind::Device, layout(d).bytes);
                budget.scratch(card, Kind::Pinned, layout(d).bytes);
                d.classes
                    .iter()
                    .map(|c| ClassRecords {
                        la: c.la,
                        lb: c.lb,
                        n_rec: c.n_rec as i32,
                        f: budget.device(card, F64_FIELDS * c.n_rec),
                        i: budget.device(card, I32_FIELDS * c.n_rec),
                    })
                    .collect()
            })
            .collect();
        Ok(RecordsOutput { cards })
    }

    fn run(
        out: &mut RecordsOutput,
        view: &View<'_, MainBus, Self>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let t = &shared.tables;
        let s = t.shells();
        let d = &shared.dims;
        let (n_shell, n_prim, n_pair) = (d.n_shell, d.n_prim, d.n_shell_pair);
        // Cheap and the same on every card: one after the other.
        for (lane, classes) in lane.cards.iter().zip(&mut out.cards) {
            let _current = lane.current();
            let card = lane.card();
            let stream = lane.stream();
            let (host, dev) = (layout(d), layout(d));
            let (hf, hi, hu) = (
                scratch.pinned(card, host.f),
                scratch.pinned(card, host.i),
                scratch.pinned(card, host.u),
            );
            let (mut df, mut di, mut du) = (
                scratch.device(card, dev.f),
                scratch.device(card, dev.i),
                scratch.device(card, dev.u),
            );

            // Stage the tables, straight into the pinned region.
            let mut at = 0;
            for field in [&s.exponent, &s.coefficient, &s.c_x, &s.c_y, &s.c_z] {
                hf[at..at + field.len()].copy_from_slice(field);
                at += field.len();
            }
            let mut at = 0;
            for field in [&s.prim_begin, &s.ao_offset, &s.rank] {
                hi[at..at + field.len()].copy_from_slice(field);
                at += field.len();
            }
            let rec_begin_at = at;
            let mut pair = 0;
            for c in t.classes() {
                let n = c.bra.len();
                hi[at..=at + n].copy_from_slice(&c.rec_begin);
                hu[pair..pair + n].copy_from_slice(&c.bra);
                hu[n_pair + pair..n_pair + pair + n].copy_from_slice(&c.ket);
                at += n + 1;
                pair += n;
            }
            upload(&mut df, hf, stream)?;
            upload(&mut di, hi, stream)?;
            upload(&mut du, hu, stream)?;

            // Form each class's records.
            let (f, i, u) = (df.as_ptr(), di.as_ptr(), du.as_ptr());
            let (mut first, mut rec_at) = (0, rec_begin_at);
            for (c, class) in t.classes().iter().zip(classes.iter_mut()) {
                let n_pairs = c.bra.len();
                let src = fb_primitive_pair_source {
                    n_pairs: n_pairs as i32,
                    bra: u.wrapping_add(first),
                    ket: u.wrapping_add(n_pair + first),
                    rec_begin: i.wrapping_add(rec_at),
                    prim_begin: i,
                    exponent: f,
                    coefficient: f.wrapping_add(n_prim),
                    c_x: f.wrapping_add(2 * n_prim),
                    c_y: f.wrapping_add(2 * n_prim + n_shell),
                    c_z: f.wrapping_add(2 * n_prim + 2 * n_shell),
                    ao_offset: i.wrapping_add(n_shell + 1),
                    rank: i.wrapping_add(2 * n_shell + 1),
                };
                let n = class.n_rec as usize;
                let (pf, pi) = (class.f.as_mut_ptr(), class.i.as_mut_ptr());
                let g = |k: usize| pf.wrapping_add(k * n);
                let h = |k: usize| pi.wrapping_add(k * n);
                let records = fb_primitive_pair_records {
                    zeta: g(0),
                    p_x: g(1),
                    p_y: g(2),
                    p_z: g(3),
                    pa_x: g(4),
                    pa_y: g(5),
                    pa_z: g(6),
                    pb_x: g(7),
                    pb_y: g(8),
                    pb_z: g(9),
                    s: g(10),
                    ea: g(11),
                    eb: g(12),
                    off_a: h(0),
                    off_b: h(1),
                    rank_a: h(2),
                    rank_b: h(3),
                };
                // SAFETY: the source arrays are this step's uploads,
                // enqueued on this stream above, each as long as the host
                // table it was staged from, which `RecordTables` checked
                // against the pairs; the outputs are this class's `n_rec`
                // records per field, by `mem_budget`; the two structs are
                // host values the call reads before it returns.
                let status =
                    unsafe { fb_mmd_records(stream.stream(), class.n_rec, &src, &records) };
                launched(&format!("fb_mmd_records ({}, {})", c.la, c.lb), status)?;
                first += n_pairs;
                rec_at += n_pairs + 1;
            }
        }
        Ok(())
    }
}
