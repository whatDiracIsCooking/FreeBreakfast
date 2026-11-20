//! The primitive pair records of each shell-pair class of the orbital
//! basis, formed on the device from the host's shell tables: what every
//! integral kernel over AO pairs reads.
//!
//! The old tree's `fb_integrals_sys::add_record_tasks`, without the arena:
//! the tables are staged in pinned buffers, copied up, and `fb_mmd_records`
//! is launched per class into this step's own buffers. The screening bounds
//! the old task also forms are left out: nothing here screens.

use fb_core_error_handling::Result;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_sys::{
    fb_mmd_records, fb_primitive_pair_class_view, fb_primitive_pair_records,
    fb_primitive_pair_source,
};

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::{launched, upload};
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::GPU0;

/// The f64 fields `fb_mmd_records` writes per record: zeta, P, PA, PB, s,
/// ea, eb.
const F64_FIELDS: usize = 13;
/// The i32 fields: off_a, off_b, rank_a, rank_b.
const I32_FIELDS: usize = 4;

/// The step.
#[derive(Clone, Copy)]
pub struct RecordsStep;

/// One class's records: `F64_FIELDS` segments of `n_rec` doubles and
/// `I32_FIELDS` of `n_rec` ints.
pub struct ClassRecords {
    la: i32,
    lb: i32,
    n_rec: i32,
    f: DeviceBuf<f64>,
    i: DeviceBuf<i32>,
}

impl ClassRecords {
    /// The bra shells' momentum.
    pub fn la(&self) -> i32 {
        self.la
    }

    /// The ket shells' momentum, at least `la`.
    pub fn lb(&self) -> i32 {
        self.lb
    }

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

/// The classes' records, and the shell tables they are formed from.
pub struct RecordsOutput {
    classes: Vec<ClassRecords>,
    /// exponent, coefficient (one per primitive), c_x, c_y, c_z (per shell).
    src_f64: DeviceBuf<f64>,
    src_f64_host: PinnedBuf<f64>,
    /// prim_begin (shells + 1), ao_offset, rank (per shell), then each
    /// class's rec_begin (pairs + 1).
    src_i32: DeviceBuf<i32>,
    src_i32_host: PinnedBuf<i32>,
    /// bra, ket (one per pair, the classes concatenated).
    src_u32: DeviceBuf<u32>,
    src_u32_host: PinnedBuf<u32>,
}

impl RecordsOutput {
    /// The classes, in the order of the host tables' (`Shared::tables`).
    pub fn classes(&self) -> &[ClassRecords] {
        &self.classes
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
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<RecordsOutput> {
        let t = &shared.tables;
        let s = t.shells();
        let (n_shell, n_prim, n_pair) = (s.ao_offset.len(), s.exponent.len(), t.n_pairs());
        let n_f64 = 2 * n_prim + 3 * n_shell;
        let n_i32 = (n_shell + 1) + 2 * n_shell + n_pair + t.classes().len();
        let classes = t
            .classes()
            .iter()
            .map(|c| {
                let n = c.n_rec() as usize;
                ClassRecords {
                    la: c.la,
                    lb: c.lb,
                    n_rec: c.n_rec(),
                    f: budget.device(GPU0, F64_FIELDS * n),
                    i: budget.device(GPU0, I32_FIELDS * n),
                }
            })
            .collect();
        Ok(RecordsOutput {
            classes,
            src_f64: budget.device(GPU0, n_f64),
            src_f64_host: budget.pinned(GPU0, n_f64),
            src_i32: budget.device(GPU0, n_i32),
            src_i32_host: budget.pinned(GPU0, n_i32),
            src_u32: budget.device(GPU0, 2 * n_pair),
            src_u32_host: budget.pinned(GPU0, 2 * n_pair),
        })
    }

    fn run(out: &mut RecordsOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let t = &view.shared().tables;
        let s = t.shells();
        let (n_shell, n_prim, n_pair) = (s.ao_offset.len(), s.exponent.len(), t.n_pairs());
        let stream = view.lane().gpu().stream();

        // Stage the tables.
        let mut at = 0;
        for field in [&s.exponent, &s.coefficient, &s.c_x, &s.c_y, &s.c_z] {
            out.src_f64_host[at..at + field.len()].copy_from_slice(field);
            at += field.len();
        }
        let mut at = 0;
        for field in [&s.prim_begin, &s.ao_offset, &s.rank] {
            out.src_i32_host[at..at + field.len()].copy_from_slice(field);
            at += field.len();
        }
        let rec_begin_at = at;
        let mut pair = 0;
        for c in t.classes() {
            let n = c.bra.len();
            out.src_i32_host[at..=at + n].copy_from_slice(&c.rec_begin);
            out.src_u32_host[pair..pair + n].copy_from_slice(&c.bra);
            out.src_u32_host[n_pair + pair..n_pair + pair + n].copy_from_slice(&c.ket);
            at += n + 1;
            pair += n;
        }
        upload(&mut out.src_f64, &out.src_f64_host, stream)?;
        upload(&mut out.src_i32, &out.src_i32_host, stream)?;
        upload(&mut out.src_u32, &out.src_u32_host, stream)?;

        // Form each class's records.
        let (f, i, u) = (
            out.src_f64.as_ptr(),
            out.src_i32.as_ptr(),
            out.src_u32.as_ptr(),
        );
        let (mut first, mut rec_at) = (0, rec_begin_at);
        for (c, class) in t.classes().iter().zip(&mut out.classes) {
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
            // SAFETY: the source arrays are this step's uploads, enqueued on
            // this stream above, each as long as the host table it was
            // staged from, which `RecordTables` checked against the pairs;
            // the outputs are this class's `n_rec` records per field, by
            // `mem_budget`; the two structs are host values the call reads
            // before it returns.
            let status = unsafe { fb_mmd_records(stream.stream(), class.n_rec, &src, &records) };
            launched(&format!("fb_mmd_records ({}, {})", c.la, c.lb), status)?;
            first += n_pairs;
            rec_at += n_pairs + 1;
        }
        Ok(Next::Done)
    }
}
