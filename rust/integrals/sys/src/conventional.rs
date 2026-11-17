//! The conventional ERI's device path: the packed four-index tensor of a
//! basis, whose records the records tasks formed ([`ClassRows`]) from
//! tables the caller hands in too ([`RecordTables`]), and its
//! exchange-ordered gather, as three tasks ([`add_conventional_tasks`]);
//! and the packing of a density and the unpacking of a packed result
//! around the GEMVs that read them ([`pack`], [`unpack`]).
//!
//! **The packing.** Cartesian AO pairs `m >= n` are numbered
//! `pidx(m, n) = m (m + 1) / 2 + n`, symmetric in its arguments, and there
//! are `n_pp = n_cart (n_cart + 1) / 2` of them. The packed ERI is the
//! `n_pp × n_pp` matrix
//!
//! ```text
//! ERI_pp[pidx(m, n), pidx(l, s)] = (mn|ls)
//! X_pp[pidx(l, s), pidx(m, n)]   = ((ml|ns) + (ms|nl)) / 2
//! ```
//!
//! each exactly symmetric, so its row-major and column-major layouts are
//! one. For a symmetric D, `J_pp = ERI_pp Dp` and `K_pp = X_pp Dp`, with
//! `Dp[pidx(l, s)] = D_ls + D_sl` off the diagonal and `D_ll` on it, and
//! `J_mn = J_pp[pidx(m, n)]` (FelizJueves's `eri/full_eri/packed.py`).
//!
//! The tasks, added in this order:
//!
//! - **`conventional tables`**, on the CPU: every class's shell pairs (the
//!   record range and the two AO offsets of each), every class of the
//!   tensor's tasks, the pair index `pidx` and its inverse `(pm, pn)`, all
//!   formed at planning, written into pinned rows.
//! - **`conventional eri`**, on the worker's stream: the pairs, the tasks
//!   and `pidx` uploaded into scratch, every class's ζ, P, P − A, P − B and
//!   `s` ([`Fields::CONVENTIONAL`]) read in place from its records rows, the
//!   tensor zeroed and one [`fb_conventional_eri_class`] per quartet class
//!   into `ERI_pp`, its output.
//! - **`exchange gather`**, on the worker's stream: `pm`, `pn` and `pidx`
//!   uploaded into scratch, and [`fb_conventional_exchange`] from `ERI_pp`
//!   into `X_pp`, its output.
//!
//! **The tasks of the tensor.** A quartet class pairs the records class
//! `X` with a class `Y` not before it in the tables' order (ascending in
//! `(la, lb)`), and its tasks are every (bra pair, ket pair) when
//! `X ≠ Y`, and the pairs `bra >= ket` when `X = Y`: every unordered pair
//! of shell pairs is one task, once, so every element of `ERI_pp` is
//! owned by one task (FelizJueves's `eri_classes`, whose staging the
//! tables' is: every shell pair `i <= j` once, swapped to the lower
//! momentum first). The kernels' stores are plain, so the tensor is
//! bit-reproducible and exactly symmetric.
//!
//! As the three-center integrals' cells are, every index a pair holds is
//! fixed at planning from tables [`RecordTables::new`] checked: a pair's
//! record range is its prefix in its class, inside the class's `n_rec`,
//! and its offsets its shells', each opening a block inside `n_cart`.
//! That the records are the tables' is a contract, checked always, since
//! the ranges index the records' rows.

use std::sync::Arc;

use contracts::{debug_ensures, requires};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_log::{eq, fits_i32, le};
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::Stream;

use crate::records::{ClassRows, Fields, RecordTables};
use crate::{
    FB_CONVENTIONAL_MAX_AO, FB_CONVENTIONAL_MAX_MOMENTUM, FB_CONVENTIONAL_PAIR_FIELDS,
    FB_CONVENTIONAL_TASK_FIELDS, fb_conventional_eri_class, fb_conventional_exchange,
    fb_conventional_pack, fb_conventional_unpack, launched,
};

/// The fields of one shell pair.
const PAIR_FIELDS: usize = FB_CONVENTIONAL_PAIR_FIELDS as usize;
/// The fields of one task.
const TASK_FIELDS: usize = FB_CONVENTIONAL_TASK_FIELDS as usize;

/// The packed pair count of `n` Cartesian AOs, `n (n + 1) / 2`, or `None`
/// past `usize`.
pub fn packed_pairs(n: usize) -> Option<usize> {
    n.checked_add(1)?.checked_mul(n).map(|x| x / 2)
}

/// One quartet class: its bra and ket records classes, by index, its first
/// task and its task count.
#[derive(Debug, Clone, Copy)]
struct Quartet {
    bra: usize,
    ket: usize,
    begin: usize,
    n_task: usize,
}

/// What the tasks need, formed at planning.
#[derive(Debug)]
struct Staging {
    n_cart: usize,
    n_pp: usize,
    /// The records classes, whose fields the launches read in place.
    classes: Vec<ClassRows>,
    /// Each class's first pair in `pairs`, and its pair count.
    class_pairs: Vec<(usize, usize)>,
    /// Every class's pairs, class by class, each class's field-major.
    pairs: Vec<i32>,
    /// The quartet classes, in launch order.
    quartets: Vec<Quartet>,
    /// Every task, quartet by quartet, each quartet's field-major.
    tasks: Vec<i32>,
    /// `pidx[m n_cart + n]`.
    pidx: Vec<i32>,
    /// The AO pair of each packed index, `pm >= pn`.
    pm: Vec<i32>,
    pn: Vec<i32>,
}

/// Add the packed ERI of a basis and its exchange-ordered gather to `dag`
/// on GPU worker `worker`, and return them, `[ERI_pp, X_pp]`: two output
/// rows of `n_pp²` doubles each, `n_pp = n_cart (n_cart + 1) / 2`
/// ([`packed_pairs`]), over the Cartesian AOs, as the module doc lays them
/// out.
///
/// `classes` are the basis's records ([`add_record_tasks`]) and `tables`
/// the tables they were formed from, whose shell pairs give each pair's
/// record range and AO offsets. The tasks are `conventional tables` (CPU),
/// `conventional eri` and `exchange gather` (stream), the module doc's.
/// Both rows are in the plan's table at planning, so a basis whose pair
/// does not fit the card is refused by admission, with the rows named.
///
/// A class above the kernels' table ([`FB_CONVENTIONAL_MAX_MOMENTUM`]) is
/// the launcher's refusal at run time, `Unsupported` naming the class; a
/// caller refuses one at planning.
///
/// [`add_record_tasks`]: crate::add_record_tasks
///
/// # Panics
/// If a quartet class's task count does not fit an `i32`; or a class is of
/// another worker's or plan's table ([`Dag::add_gpu`]).
#[requires(tables.formed(classes), "classes are the records of tables")]
#[requires(
    le!(tables.n_cart(), FB_CONVENTIONAL_MAX_AO as usize),
    "n_cart is within the kernels' FB_CONVENTIONAL_MAX_AO"
)]
#[requires(
    fits_i32!(packed_pairs(tables.n_cart()).and_then(|n| n.checked_mul(n))),
    "n_pp² fits the kernels' 32-bit index"
)]
#[debug_requires(!classes.is_empty(), "a basis of at least one shell")]
#[debug_requires(
    classes.iter().all(|c| c.lb() <= FB_CONVENTIONAL_MAX_MOMENTUM),
    "every class is in the kernels' table"
)]
#[track_caller]
pub fn add_conventional_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[ClassRows],
    tables: &RecordTables,
) -> [RowId<Device, f64>; 2] {
    let staging = Arc::new(staging(classes, tables));
    let tables = dag.add_cpu(
        "conventional tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
        },
    );
    let eri = dag
        .add_gpu(
            "conventional eri",
            worker,
            Tensor {
                staging: Arc::clone(&staging),
                tables,
            },
        )
        .eri;
    let x = dag
        .add_gpu(
            "exchange gather",
            worker,
            Gather {
                staging,
                tables,
                eri,
            },
        )
        .x;
    [eri, x]
}

/// The staging of [`add_conventional_tasks`]: the pairs, the tasks and the
/// pair index formed from the tables, every extent it asserts checked.
#[debug_ensures(
    ret.quartets.iter().map(|q| q.n_task).sum::<usize>()
        == tables.n_pairs() * (tables.n_pairs() + 1) / 2,
    "every unordered pair of shell pairs is one task"
)]
#[debug_ensures(
    ret.quartets.iter().try_fold(0, |at, q| (q.begin == at).then_some(at + q.n_task))
        == Some(ret.tasks.len() / TASK_FIELDS),
    "the quartet classes tile the tasks in order"
)]
#[track_caller]
fn staging(classes: &[ClassRows], tables: &RecordTables) -> Staging {
    // Each class's pairs, in enumeration order: a pair's record range, in
    // its class's records, is its prefix, and its offsets are its two
    // shells'. Every entry is an i32 already (RecordTables::new).
    let offset = &tables.shells().ao_offset;
    let mut pairs = Vec::new();
    let mut class_pairs = Vec::new();
    for c in tables.classes() {
        let n = c.bra.len();
        class_pairs.push((pairs.len() / PAIR_FIELDS, n));
        let mut run = vec![0; PAIR_FIELDS * n];
        for p in 0..n {
            let fields = [
                c.rec_begin[p],
                c.rec_begin[p + 1],
                offset[c.bra[p] as usize],
                offset[c.ket[p] as usize],
            ];
            for (f, v) in fields.into_iter().enumerate() {
                run[f * n + p] = v;
            }
        }
        pairs.extend(run);
    }

    // The quartet classes: X with every Y not before it, each X = Y the
    // pairs bra >= ket. Pair indices are each class's own, inside its
    // pair count, which fits an i32 (RecordTables::new).
    let mut quartets = Vec::new();
    let mut tasks = Vec::new();
    let mut n_tasks = 0;
    for (x, &(_, nx)) in class_pairs.iter().enumerate() {
        for (y, &(_, ny)) in class_pairs.iter().enumerate().skip(x) {
            let run: Vec<[usize; 2]> = if x == y {
                (0..nx).flat_map(|b| (0..=b).map(move |k| [b, k])).collect()
            } else {
                (0..nx).flat_map(|b| (0..ny).map(move |k| [b, k])).collect()
            };
            let n_task = run.len();
            assert!(
                i32::try_from(n_task).is_ok(),
                "add_conventional_tasks: class ({}, {}|{}, {}) has {n_task} tasks, past an i32 \
                 count",
                classes[x].la(),
                classes[x].lb(),
                classes[y].la(),
                classes[y].lb()
            );
            let mut fields = vec![0; TASK_FIELDS * n_task];
            for (t, [b, k]) in run.into_iter().enumerate() {
                fields[t] = b as i32;
                fields[n_task + t] = k as i32;
            }
            tasks.extend(fields);
            quartets.push(Quartet {
                bra: x,
                ket: y,
                begin: n_tasks,
                n_task,
            });
            n_tasks += n_task;
        }
    }

    // The pair index and its inverse; n_pp² fits an i32 (the contracts), so
    // every index does.
    let n = tables.n_cart();
    let pidx = (0..n * n)
        .map(|i| {
            let (a, b) = (i / n, i % n);
            let (m, k) = (a.max(b), a.min(b));
            (m * (m + 1) / 2 + k) as i32
        })
        .collect();
    let pm = (0..n)
        .flat_map(|m| (0..=m).map(move |_| m as i32))
        .collect();
    let pn = (0..n).flat_map(|m| (0..=m).map(|k| k as i32)).collect();
    Staging {
        n_cart: n,
        n_pp: n * (n + 1) / 2,
        classes: classes.to_vec(),
        class_pairs,
        pairs,
        quartets,
        tasks,
        pidx,
        pm,
        pn,
    }
}

/// The staged tables, of one kind: pinned as the CPU task's outputs, device
/// as the stream tasks' scratch.
struct TableRows<K> {
    pairs: RowId<K, i32>,
    tasks: RowId<K, i32>,
    pidx: RowId<K, i32>,
    pm: RowId<K, i32>,
    pn: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

/// CPU: the pairs, the tasks and the pair index, formed at planning, into
/// pinned rows.
struct Tables {
    staging: Arc<Staging>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let s = &self.staging;
        TableRows {
            pairs: host.output("conventional pairs", s.pairs.len()),
            tasks: host.output("conventional tasks", s.tasks.len()),
            pidx: host.output("pidx", s.pidx.len()),
            pm: host.output("pm", s.pm.len()),
            pn: host.output("pn", s.pn.len()),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        memcpy_h2p(host, rows.pairs, &s.pairs);
        memcpy_h2p(host, rows.tasks, &s.tasks);
        memcpy_h2p(host, rows.pidx, &s.pidx);
        memcpy_h2p(host, rows.pm, &s.pm);
        memcpy_h2p(host, rows.pn, &s.pn);
        Ok(())
    }
}

/// GPU: the pairs, the tasks and `pidx` uploaded, the tensor zeroed and
/// every quartet class launched into it from its classes' own records rows.
struct Tensor {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct TensorRows {
    pairs: RowId<Device, i32>,
    tasks: RowId<Device, i32>,
    pidx: RowId<Device, i32>,
    eri: RowId<Device, f64>,
}

impl GpuTask for Tensor {
    type Rows = TensorRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> TensorRows {
        let t = self.tables;
        host.read(t.pairs);
        host.read(t.tasks);
        host.read(t.pidx);
        let s = &self.staging;
        for c in &s.classes {
            c.read(device, Fields::CONVENTIONAL);
        }
        TensorRows {
            pairs: device.scratch("conventional pairs", s.pairs.len()),
            tasks: device.scratch("conventional tasks", s.tasks.len()),
            pidx: device.scratch("pidx", s.pidx.len()),
            eri: device.output("ERI_pp", s.n_pp * s.n_pp),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.staging.n_cart)]
    }

    fn run(
        &self,
        rows: &TensorRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let s = &self.staging;
        let mut pairs = device.take_mut(rows.pairs);
        memcpy_p2d(&mut pairs, host, self.tables.pairs, worker)?;
        let mut tasks = device.take_mut(rows.tasks);
        memcpy_p2d(&mut tasks, host, self.tables.tasks, worker)?;
        let mut pidx = device.take_mut(rows.pidx);
        memcpy_p2d(&mut pidx, host, self.tables.pidx, worker)?;
        let mut eri = device.take_mut(rows.eri);
        // Every element is owned by one task; the zero makes the tensor's
        // value independent of that, should a class ever be skipped.
        memset::zero(&mut eri, worker)?;

        // In range, as every cast below: n_cart is within
        // FB_CONVENTIONAL_MAX_AO (the contracts), each class's pair count is
        // an i32 (RecordTables::new) and each quartet's task count one (the
        // staging asserts it).
        let n = s.n_cart as i32;
        for q in &s.quartets {
            let (bra, ket) = (&s.classes[q.bra], &s.classes[q.ket]);
            let (bra_at, n_bpair) = s.class_pairs[q.bra];
            let (ket_at, n_kpair) = s.class_pairs[q.ket];
            let bra_view = bra.view(device, Fields::CONVENTIONAL);
            let ket_view = ket.view(device, Fields::CONVENTIONAL);
            // SAFETY: each view's arrays are its class's n_rec-long rows of
            // Fields::CONVENTIONAL, every record field the launcher reads (a
            // test scans it), the rest null, written by the class's records
            // task before this task's stream work (ClassRows' argument); the
            // bra and ket pairs are PAIR_FIELDS segments of n_bpair and
            // n_kpair int32s inside the pairs row, the tasks TASK_FIELDS
            // segments of n_task inside the tasks row and pidx n_cart²
            // int32s, all copied above on this stream from the tables task,
            // which wrote the staging formed at planning: every record range
            // a pair's prefix in its class, so inside [0, n_rec) (the
            // classes are the tables', a contract), its offsets the pair's
            // shells', each opening a block inside [0, n_cart)
            // (RecordTables::new), every task's indices inside its two pair
            // counts, and every pidx entry inside [0, n_pp); eri is this
            // task's n_pp² doubles; the views are host structs that outlive
            // the call, which only enqueues on the task's stream.
            let status = unsafe {
                fb_conventional_eri_class(
                    worker.stream(),
                    &bra_view,
                    &ket_view,
                    n_bpair as i32,
                    pairs.as_ptr().wrapping_add(PAIR_FIELDS * bra_at),
                    n_kpair as i32,
                    pairs.as_ptr().wrapping_add(PAIR_FIELDS * ket_at),
                    q.n_task as i32,
                    tasks.as_ptr().wrapping_add(TASK_FIELDS * q.begin),
                    n,
                    pidx.as_ptr(),
                    eri.as_mut_ptr(),
                )
            };
            launched(
                "fb_conventional_eri_class",
                &[bra.la(), bra.lb(), ket.la(), ket.lb()],
                status,
            )?;
        }
        Ok(())
    }
}

/// GPU: `pm`, `pn` and `pidx` uploaded, and `X_pp` gathered from `ERI_pp`.
struct Gather {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
    eri: RowId<Device, f64>,
}

#[derive(Clone, Copy)]
struct GatherRows {
    pm: RowId<Device, i32>,
    pn: RowId<Device, i32>,
    pidx: RowId<Device, i32>,
    x: RowId<Device, f64>,
}

impl GpuTask for Gather {
    type Rows = GatherRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> GatherRows {
        let t = self.tables;
        host.read(t.pm);
        host.read(t.pn);
        host.read(t.pidx);
        device.read(self.eri);
        let s = &self.staging;
        GatherRows {
            pm: device.scratch("pm", s.pm.len()),
            pn: device.scratch("pn", s.pn.len()),
            pidx: device.scratch("pidx", s.pidx.len()),
            x: device.output("X_pp", s.n_pp * s.n_pp),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.staging.n_cart)]
    }

    fn run(
        &self,
        rows: &GatherRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let s = &self.staging;
        let mut pm = device.take_mut(rows.pm);
        memcpy_p2d(&mut pm, host, self.tables.pm, worker)?;
        let mut pn = device.take_mut(rows.pn);
        memcpy_p2d(&mut pn, host, self.tables.pn, worker)?;
        let mut pidx = device.take_mut(rows.pidx);
        memcpy_p2d(&mut pidx, host, self.tables.pidx, worker)?;
        let eri = device.get(self.eri);
        let x = device.take_mut(rows.x);
        // SAFETY: pm and pn are n_pp int32s and pidx n_cart² int32s, copied
        // above on this stream from the tables task, which wrote the staging:
        // every pm, pn an AO inside [0, n_cart) and every pidx entry inside
        // [0, n_pp); eri is the tensor task's n_pp² doubles, written before
        // this task's stream work, and x this task's own n_pp² doubles, so
        // apart; n_cart is within FB_CONVENTIONAL_MAX_AO (the contracts);
        // the launch only enqueues on the task's stream.
        let status = unsafe {
            fb_conventional_exchange(
                worker.stream(),
                s.n_cart as i32,
                pm.as_ptr(),
                pn.as_ptr(),
                pidx.as_ptr(),
                eri.as_ptr(),
                x.as_mut_ptr(),
            )
        };
        launched("fb_conventional_exchange", &[], status)
    }
}

/// Packs the `n_cart × n_cart` density `d` into `dp`, `n_pp` doubles, on
/// `stream`: [`fb_conventional_pack`], `dp[pidx(l, s)] = d_ls + d_sl` off
/// the diagonal and `d_ll` on it (the module doc's `Dp`). `d` is read as
/// stored, and either of its layouts gives the same `dp`.
///
/// # Errors
/// The launcher's refusal or launch error, as
/// [`fb_kernel_abi::launch_status`] maps it.
#[requires(
    le!(1, n_cart) && le!(n_cart, FB_CONVENTIONAL_MAX_AO as usize),
    "n_cart is within the kernels' FB_CONVENTIONAL_MAX_AO"
)]
#[requires(eq!(Some(d.len()), n_cart.checked_mul(n_cart)), "d is n_cart² doubles")]
#[requires(eq!(Some(dp.len()), packed_pairs(n_cart)), "dp is n_pp doubles")]
#[track_caller]
pub fn pack(
    stream: &Stream,
    n_cart: usize,
    d: &impl DeviceSlice<f64>,
    dp: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let dp = dp.as_mut_ptr();
    // SAFETY: d is n_cart² doubles and dp n_pp, by the contracts, which also
    // bound n_cart within the launcher's FB_CONVENTIONAL_MAX_AO, so the cast
    // is exact; dp has no other user until the enqueued work finishes
    // (DeviceSliceMut's promise), so apart from d; the call only enqueues.
    let status = unsafe { fb_conventional_pack(stream.stream(), n_cart as i32, d.as_ptr(), dp) };
    launched("fb_conventional_pack", &[], status)
}

/// Unpacks `vp`, `n_pp` doubles, into the symmetric `n_cart × n_cart`
/// matrix `v` on `stream`: [`fb_conventional_unpack`],
/// `v_mn = vp[pidx(m, n)]`, so `v_mn` and `v_nm` are the same bits.
///
/// # Errors
/// As [`pack`].
#[requires(
    le!(1, n_cart) && le!(n_cart, FB_CONVENTIONAL_MAX_AO as usize),
    "n_cart is within the kernels' FB_CONVENTIONAL_MAX_AO"
)]
#[requires(eq!(Some(vp.len()), packed_pairs(n_cart)), "vp is n_pp doubles")]
#[requires(eq!(Some(v.len()), n_cart.checked_mul(n_cart)), "v is n_cart² doubles")]
#[track_caller]
pub fn unpack(
    stream: &Stream,
    n_cart: usize,
    vp: &impl DeviceSlice<f64>,
    v: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let v = v.as_mut_ptr();
    // SAFETY: as pack's, with vp n_pp doubles and v n_cart².
    let status = unsafe { fb_conventional_unpack(stream.stream(), n_cart as i32, vp.as_ptr(), v) };
    launched("fb_conventional_unpack", &[], status)
}
