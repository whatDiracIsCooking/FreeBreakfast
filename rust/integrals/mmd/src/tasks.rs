//! The two tasks: the tables staged in pinned rows on the CPU, and every
//! class formed from them on a stream.

use std::sync::Arc;

use fb_core_arena::{Bufs, Device, Element, Kind, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_sys::{
    FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, FB_LAUNCH_UNSUPPORTED,
    FB_PAIR_BOUND_MAX_MOMENTUM, FB_RECORD_BOUND_MAX_MOMENTUM, fb_launch_status, fb_mmd_bound,
    fb_mmd_order, fb_mmd_order_workspace_bytes, fb_mmd_records, fb_ordered_pair_bound,
    fb_primitive_pair_class_view, fb_primitive_pair_records, fb_primitive_pair_source,
};
use wwr_sys::{wwr_memcpy_kind, wwr_status, wwr_stream};

use super::tables::Tables;

/// One class's records on the device: its momenta, its record count and a
/// row per field, each `n_rec` long, all outputs of the records task. A
/// reader declares the fields it reads with
/// [`Layout::read`](fb_core_arena::Layout::read).
///
/// The fields are the C++ `PrimitivePairClass`'s (its README's design has
/// each one), in descending ζ order, equal ζ
/// in enumeration order (shell pair, then bra primitive, then ket
/// primitive).
#[derive(Clone, Copy, Debug)]
pub struct ClassRows {
    /// The bra shell's momentum, at most `lb`.
    pub la: i32,
    /// The ket shell's momentum.
    pub lb: i32,
    /// The number of records, positive: every primitive product of every
    /// shell pair of the class, none screened.
    pub n_rec: i32,
    /// ζ = α + β.
    pub zeta: RowId<Device, f64>,
    /// P: x.
    pub p_x: RowId<Device, f64>,
    /// P: y.
    pub p_y: RowId<Device, f64>,
    /// P: z.
    pub p_z: RowId<Device, f64>,
    /// P − A: x.
    pub pa_x: RowId<Device, f64>,
    /// P − A: y.
    pub pa_y: RowId<Device, f64>,
    /// P − A: z.
    pub pa_z: RowId<Device, f64>,
    /// P − B: x.
    pub pb_x: RowId<Device, f64>,
    /// P − B: y.
    pub pb_y: RowId<Device, f64>,
    /// P − B: z.
    pub pb_z: RowId<Device, f64>,
    /// c_a c_b (π/ζ)^{3/2}.
    pub s: RowId<Device, f64>,
    /// α, the bra exponent.
    pub ea: RowId<Device, f64>,
    /// β, the ket exponent.
    pub eb: RowId<Device, f64>,
    /// The undressed Schwarz bound, `pair_bound(ζ, la + lb)`.
    pub bound: RowId<Device, f64>,
    /// The dressed Schwarz bound as stored.
    pub bound_ab: RowId<Device, f64>,
    /// The dressed Schwarz bound swapped.
    pub bound_ba: RowId<Device, f64>,
    /// The bra shell's first Cartesian AO.
    pub off_a: RowId<Device, i32>,
    /// The ket shell's first Cartesian AO.
    pub off_b: RowId<Device, i32>,
    /// The bra shell's rank among the basis's shells of momentum `la`.
    pub rank_a: RowId<Device, i32>,
    /// The ket shell's rank among those of momentum `lb`.
    pub rank_b: RowId<Device, i32>,
}

/// Add the records of `basis` to `dag` on GPU worker `worker`: a CPU task,
/// `mmd tables`, that writes the shell table and every class's shell pairs
/// into pinned rows, and a stream task, `mmd records`, that copies them to
/// device scratch and forms every class from them. Returns each class's
/// rows, ascending in `(la, lb)`; a basis with no shells adds nothing and
/// returns none.
///
/// Every size is the basis's, fixed here at planning: a class's `n_rec` is
/// the sum of its shell pairs' primitive products. Aborts past `i32::MAX`
/// primitives, or records in one class, as the C++ does.
#[track_caller]
pub fn add_records(dag: &mut Dag<'_>, worker: usize, basis: &fb_basis::BasisSet) -> Vec<ClassRows> {
    if basis.nshells() == 0 {
        return Vec::new();
    }
    let tables = Arc::new(Tables::new(basis));
    let staged = dag.add(
        "mmd tables",
        worker,
        Stage {
            tables: Arc::clone(&tables),
        },
    );
    dag.add("mmd records", worker, Records { tables, staged })
        .classes
}

/// The eleven tables, one row each, of one kind: pinned as the CPU task's
/// outputs, device as the stream task's scratch.
struct TableRows<K> {
    prim_begin: RowId<K, i32>,
    exponent: RowId<K, f64>,
    coefficient: RowId<K, f64>,
    c_x: RowId<K, f64>,
    c_y: RowId<K, f64>,
    c_z: RowId<K, f64>,
    ao_offset: RowId<K, i32>,
    rank: RowId<K, i32>,
    bra: RowId<K, u32>,
    ket: RowId<K, u32>,
    /// Every class's prefix, one after the other: class `c`'s
    /// `n_pairs + 1` entries start at its first pair's index plus `c`.
    rec_begin: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

impl<K: Kind> TableRows<K> {
    /// Declare the eleven rows, as outputs or as scratch.
    fn declare(l: &mut Layout<'_, K>, t: &Tables, output: bool) -> Self {
        fn row<K: Kind, T: Element>(
            l: &mut Layout<'_, K>,
            name: &str,
            len: usize,
            output: bool,
        ) -> RowId<K, T> {
            if output {
                l.output(name, len)
            } else {
                l.scratch(name, len)
            }
        }
        let s = &t.shells;
        let n_pairs = t.n_pairs();
        TableRows {
            prim_begin: row(l, "prim_begin", s.prim_begin.len(), output),
            exponent: row(l, "exponent", s.exponent.len(), output),
            coefficient: row(l, "coefficient", s.coefficient.len(), output),
            c_x: row(l, "c_x", s.c_x.len(), output),
            c_y: row(l, "c_y", s.c_y.len(), output),
            c_z: row(l, "c_z", s.c_z.len(), output),
            ao_offset: row(l, "ao_offset", s.ao_offset.len(), output),
            rank: row(l, "rank", s.rank.len(), output),
            bra: row(l, "bra", n_pairs, output),
            ket: row(l, "ket", n_pairs, output),
            rec_begin: row(l, "rec_begin", n_pairs + t.classes.len(), output),
        }
    }
}

/// CPU: the host tables written into pinned rows.
struct Stage {
    tables: Arc<Tables>,
}

impl Task for Stage {
    type Rows = TableRows<Pinned>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        TableRows::declare(host, &self.tables, true)
    }

    fn run(
        &self,
        rows: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let s = &self.tables.shells;
        host.take_mut(rows.prim_begin)
            .copy_from_slice(&s.prim_begin);
        host.take_mut(rows.exponent).copy_from_slice(&s.exponent);
        host.take_mut(rows.coefficient)
            .copy_from_slice(&s.coefficient);
        host.take_mut(rows.c_x).copy_from_slice(&s.c_x);
        host.take_mut(rows.c_y).copy_from_slice(&s.c_y);
        host.take_mut(rows.c_z).copy_from_slice(&s.c_z);
        host.take_mut(rows.ao_offset).copy_from_slice(&s.ao_offset);
        host.take_mut(rows.rank).copy_from_slice(&s.rank);
        let bra = host.take_mut(rows.bra);
        let ket = host.take_mut(rows.ket);
        let rec_begin = host.take_mut(rows.rec_begin);
        let (mut p, mut r) = (0, 0);
        for c in &self.tables.classes {
            let n = c.bra.len();
            bra[p..p + n].copy_from_slice(&c.bra);
            ket[p..p + n].copy_from_slice(&c.ket);
            rec_begin[r..=r + n].copy_from_slice(&c.rec_begin);
            p += n;
            r += n + 1;
        }
        Ok(())
    }
}

/// GPU: the tables copied to device scratch, then each class, in order, on
/// the one stream: its order (key and stable sort, in scratch sized to the
/// largest class), its records, its bound and its two dressed bounds.
struct Records {
    tables: Arc<Tables>,
    staged: TableRows<Pinned>,
}

#[derive(Clone)]
struct RecordsRows {
    tables: TableRows<Device>,
    key: RowId<Device, f64>,
    perm: RowId<Device, u32>,
    workspace: RowId<Device, u8>,
    classes: Vec<ClassRows>,
}

impl Task for Records {
    type Rows = RecordsRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        host: &mut Layout<'_, Pinned>,
    ) -> RecordsRows {
        let s = &self.staged;
        host.read(s.prim_begin);
        host.read(s.exponent);
        host.read(s.coefficient);
        host.read(s.c_x);
        host.read(s.c_y);
        host.read(s.c_z);
        host.read(s.ao_offset);
        host.read(s.rank);
        host.read(s.bra);
        host.read(s.ket);
        host.read(s.rec_begin);
        let t = &self.tables;
        let tables = TableRows::declare(device, t, false);
        let n_max = t.n_max() as usize;
        let key = device.scratch("order key", n_max);
        let perm = device.scratch("order perm", n_max);
        // SAFETY: a host function of its argument alone.
        let bytes = unsafe { fb_mmd_order_workspace_bytes(t.n_max()) };
        let workspace = device.scratch("order workspace", bytes);
        let classes = t
            .classes
            .iter()
            .map(|c| {
                let (la, lb, n) = (c.la, c.lb, c.n_rec() as usize);
                let mut f64_row =
                    |field: &str| device.output::<f64>(&format!("{field} ({la},{lb})"), n);
                let (zeta, p_x, p_y, p_z) = (
                    f64_row("zeta"),
                    f64_row("p_x"),
                    f64_row("p_y"),
                    f64_row("p_z"),
                );
                let (pa_x, pa_y, pa_z) = (f64_row("pa_x"), f64_row("pa_y"), f64_row("pa_z"));
                let (pb_x, pb_y, pb_z) = (f64_row("pb_x"), f64_row("pb_y"), f64_row("pb_z"));
                let (s, ea, eb) = (f64_row("s"), f64_row("ea"), f64_row("eb"));
                let (bound, bound_ab, bound_ba) =
                    (f64_row("bound"), f64_row("bound_ab"), f64_row("bound_ba"));
                let mut i32_row =
                    |field: &str| device.output::<i32>(&format!("{field} ({la},{lb})"), n);
                let (off_a, off_b, rank_a, rank_b) = (
                    i32_row("off_a"),
                    i32_row("off_b"),
                    i32_row("rank_a"),
                    i32_row("rank_b"),
                );
                ClassRows {
                    la,
                    lb,
                    n_rec: c.n_rec(),
                    zeta,
                    p_x,
                    p_y,
                    p_z,
                    pa_x,
                    pa_y,
                    pa_z,
                    pb_x,
                    pb_y,
                    pb_z,
                    s,
                    ea,
                    eb,
                    bound,
                    bound_ab,
                    bound_ba,
                    off_a,
                    off_b,
                    rank_a,
                    rank_b,
                }
            })
            .collect();
        RecordsRows {
            tables,
            key,
            perm,
            workspace,
            classes,
        }
    }

    fn run(
        &self,
        rows: &RecordsRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let stream = stream.stream();
        let (d, s) = (&rows.tables, &self.staged);
        let prim_begin = upload(stream, device, d.prim_begin, host.get(s.prim_begin))?;
        let exponent = upload(stream, device, d.exponent, host.get(s.exponent))?;
        let coefficient = upload(stream, device, d.coefficient, host.get(s.coefficient))?;
        let c_x = upload(stream, device, d.c_x, host.get(s.c_x))?;
        let c_y = upload(stream, device, d.c_y, host.get(s.c_y))?;
        let c_z = upload(stream, device, d.c_z, host.get(s.c_z))?;
        let ao_offset = upload(stream, device, d.ao_offset, host.get(s.ao_offset))?;
        let rank = upload(stream, device, d.rank, host.get(s.rank))?;
        let bra = upload(stream, device, d.bra, host.get(s.bra))?;
        let ket = upload(stream, device, d.ket, host.get(s.ket))?;
        let rec_begin = upload(stream, device, d.rec_begin, host.get(s.rec_begin))?;

        let key = device.take_mut(rows.key);
        let perm = device.take_mut(rows.perm);
        let workspace = device.take_mut(rows.workspace);
        let mut first = 0;
        for (c, (pairs, cls)) in self.tables.classes.iter().zip(&rows.classes).enumerate() {
            let n_pairs = pairs.bra.len();
            let src = fb_primitive_pair_source {
                n_pairs: n_pairs as i32,
                // In bounds: the class's pairs, and its prefix, lie inside
                // the concatenated rows (TableRows::rec_begin).
                bra: bra.wrapping_add(first),
                ket: ket.wrapping_add(first),
                rec_begin: rec_begin.wrapping_add(first + c),
                prim_begin,
                exponent,
                coefficient,
                c_x,
                c_y,
                c_z,
                ao_offset,
                rank,
            };
            first += n_pairs;
            let n = cls.n_rec;
            // SAFETY: key and perm are n_max >= n long, the workspace is the
            // largest class's query, every array of src is device memory
            // the uploads above wrote on this stream, and none overlap: each
            // is a row of its own.
            launch("fb_mmd_order", cls, unsafe {
                fb_mmd_order(
                    stream,
                    n,
                    &src,
                    key.as_mut_ptr(),
                    perm.as_mut_ptr(),
                    workspace.as_mut_ptr().cast(),
                    workspace.len(),
                )
            })?;
            let f = |row| device.take_mut(row);
            let (zeta, p_x, p_y, p_z) = (f(cls.zeta), f(cls.p_x), f(cls.p_y), f(cls.p_z));
            let (pa_x, pa_y, pa_z) = (f(cls.pa_x), f(cls.pa_y), f(cls.pa_z));
            let (pb_x, pb_y, pb_z) = (f(cls.pb_x), f(cls.pb_y), f(cls.pb_z));
            let (sv, ea, eb) = (f(cls.s), f(cls.ea), f(cls.eb));
            let (bound, bound_ab, bound_ba) = (f(cls.bound), f(cls.bound_ab), f(cls.bound_ba));
            let g = |row| device.take_mut(row);
            let (off_a, off_b, rank_a, rank_b) =
                (g(cls.off_a), g(cls.off_b), g(cls.rank_a), g(cls.rank_b));
            let out = fb_primitive_pair_records {
                zeta: zeta.as_mut_ptr(),
                p_x: p_x.as_mut_ptr(),
                p_y: p_y.as_mut_ptr(),
                p_z: p_z.as_mut_ptr(),
                pa_x: pa_x.as_mut_ptr(),
                pa_y: pa_y.as_mut_ptr(),
                pa_z: pa_z.as_mut_ptr(),
                pb_x: pb_x.as_mut_ptr(),
                pb_y: pb_y.as_mut_ptr(),
                pb_z: pb_z.as_mut_ptr(),
                s: sv.as_mut_ptr(),
                ea: ea.as_mut_ptr(),
                eb: eb.as_mut_ptr(),
                off_a: off_a.as_mut_ptr(),
                off_b: off_b.as_mut_ptr(),
                rank_a: rank_a.as_mut_ptr(),
                rank_b: rank_b.as_mut_ptr(),
            };
            // SAFETY: perm is the order just written, and every array of out
            // is an n-long row of its own.
            launch("fb_mmd_records", cls, unsafe {
                fb_mmd_records(stream, n, &src, perm.as_ptr(), &out)
            })?;

            // A class above a family's table keeps that family's bounds
            // zero, as the C++ does; every build refuses such a class.
            if cls.lb <= FB_PAIR_BOUND_MAX_MOMENTUM {
                // SAFETY: zeta and bound are n-long rows of their own.
                launch("fb_mmd_bound", cls, unsafe {
                    fb_mmd_bound(stream, cls.la, cls.lb, n, zeta.as_ptr(), bound.as_mut_ptr())
                })?;
            } else {
                zero(stream, bound.as_mut_ptr(), n)?;
            }
            if cls.lb <= FB_RECORD_BOUND_MAX_MOMENTUM {
                let view = fb_primitive_pair_class_view {
                    la: cls.la,
                    lb: cls.lb,
                    n_rec: n,
                    zeta: zeta.as_ptr(),
                    p_x: p_x.as_ptr(),
                    p_y: p_y.as_ptr(),
                    p_z: p_z.as_ptr(),
                    pa_x: pa_x.as_ptr(),
                    pa_y: pa_y.as_ptr(),
                    pa_z: pa_z.as_ptr(),
                    pb_x: pb_x.as_ptr(),
                    pb_y: pb_y.as_ptr(),
                    pb_z: pb_z.as_ptr(),
                    s: sv.as_ptr(),
                    ea: ea.as_ptr(),
                    eb: eb.as_ptr(),
                    bound: bound.as_ptr(),
                    bound_ab: bound_ab.as_ptr(),
                    bound_ba: bound_ba.as_ptr(),
                    off_a: off_a.as_ptr(),
                    off_b: off_b.as_ptr(),
                    rank_a: rank_a.as_ptr(),
                    rank_b: rank_b.as_ptr(),
                };
                // The swap: la and lb, and P - A and P - B, exchanged.
                let swapped = fb_primitive_pair_class_view {
                    la: cls.lb,
                    lb: cls.la,
                    pa_x: view.pb_x,
                    pa_y: view.pb_y,
                    pa_z: view.pb_z,
                    pb_x: view.pa_x,
                    pb_y: view.pa_y,
                    pb_z: view.pa_z,
                    ..view
                };
                // SAFETY: the view's arrays are the class's n-long rows,
                // written above on this stream, and each bound is a row of
                // its own that the kernel only writes.
                launch("fb_ordered_pair_bound", cls, unsafe {
                    fb_ordered_pair_bound(stream, &view, bound_ab.as_mut_ptr())
                })?;
                // SAFETY: as above, the view swapped.
                launch("fb_ordered_pair_bound (swapped)", cls, unsafe {
                    fb_ordered_pair_bound(stream, &swapped, bound_ba.as_mut_ptr())
                })?;
            } else {
                zero(stream, bound_ab.as_mut_ptr(), n)?;
                zero(stream, bound_ba.as_mut_ptr(), n)?;
            }
        }
        Ok(())
    }
}

/// `src`, a pinned row, copied on `stream` to the device row `dst` of the
/// same length; returns `dst`'s pointer, for the launches to read.
fn upload<T: Element>(
    stream: wwr_stream,
    device: &Bufs<'_, Device>,
    dst: RowId<Device, T>,
    src: &[T],
) -> Result<*const T, Error> {
    let dst = device.take_mut(dst);
    debug_assert_eq!(dst.len(), src.len());
    // SAFETY: two rows of `src.len()` elements, on the task's stream. The
    // pinned row is not rewritten before the copy completes: its bytes'
    // next owner is ordered after this task's event by a handoff.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            size_of_val(src),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            stream,
        )
    })?;
    Ok(dst.as_ptr())
}

/// `n` doubles at `ptr` zeroed on `stream`.
fn zero(stream: wwr_stream, ptr: *mut f64, n: i32) -> Status {
    // SAFETY: `ptr` is an n-long device row, on the task's stream.
    gpu_status(unsafe {
        wwr_sys::wwr_memset_async(ptr.cast(), 0, n as usize * size_of::<f64>(), stream)
    })
}

/// A launch wrapper's status as a `Status`, naming the wrapper and the
/// class (`docs/gpu_layer.md`, "Statuses"): a refusal as `Unsupported` or
/// `InvalidConfig`, a launch error by its `wwr_status` name.
#[track_caller]
fn launch(what: &str, cls: &ClassRows, status: fb_launch_status) -> Status {
    let class = format!("{what} ({}, {})", cls.la, cls.lb);
    const WWR: [wwr_status; 8] = [
        wwr_status::WWR_STATUS_SUCCESS,
        wwr_status::WWR_STATUS_INVALID_VALUE,
        wwr_status::WWR_STATUS_INVALID_DEVICE,
        wwr_status::WWR_STATUS_NO_DEVICE,
        wwr_status::WWR_STATUS_NOT_READY,
        wwr_status::WWR_STATUS_MEMORY_ALLOCATION,
        wwr_status::WWR_STATUS_NOT_SUPPORTED,
        wwr_status::WWR_STATUS_UNKNOWN,
    ];
    match status {
        FB_LAUNCH_SUCCESS => Ok(()),
        FB_LAUNCH_UNSUPPORTED => Err(Error::unsupported(format!("{class}: no kernel"))),
        FB_LAUNCH_INVALID_ARGUMENT => Err(Error::config(format!("{class}: refused its arguments"))),
        code => match usize::try_from(code).ok().and_then(|i| WWR.get(i)) {
            Some(&wwr) => gpu_status(wwr).map_err(|e| e.with_context(class)),
            None => Err(Error::gpu(format!("{class}: unknown launch status {code}"))),
        },
    }
}
