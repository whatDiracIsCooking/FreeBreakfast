//! The packed three-center integrals' device path: the Cartesian (mn|P) of
//! a batch of an orbital basis's shell pairs, each pair's block in columns
//! of its own, side by side, as two tasks
//! ([`add_packed_three_center_tasks`]).
//!
//! The tasks, added in this order:
//!
//! - **`packed three-center tables`**, on the CPU: the aux records, formed
//!   at planning, and the cells of its trip's batch, written from the
//!   staging into two pinned rows.
//! - **`packed three-center`**, on the worker's stream: both rows uploaded
//!   into scratch, then one [`fb_three_center_packed_class`] launch per
//!   class `(la, lb, lP)` with a cell, each class's bra fields read in
//!   place from its records rows ([`Fields::THREE_CENTER`]), into the
//!   packed tensor, its output.
//!
//! A cell is the deleted square path's (#1721): a shell pair's
//! record range, its prefix in its class, and an aux shell's, with its
//! destination in place of the three AO offsets, the pair's first column
//! times the aux count plus the aux shell's offset, and whether the pair is
//! a shell's with itself, whose block is then stored whole. Every
//! destination is formed from the first batch's columns, at planning,
//! which every batch shares, its pairs a prefix of the first's classes
//! pair for pair, so each block lies inside the output by construction.

use std::ops::Range;
use std::sync::Arc;

use contracts::{debug_ensures, requires};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::handle::DeviceHandleStream;

use crate::metric::n_cart_of;
use crate::records::{ClassRows, Fields, RecordTables};
use crate::{
    AuxTable, FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_PACKED_CELL_FIELDS,
    fb_three_center_packed_class, launched,
};

/// The fields of one aux record.
const AUX_FIELDS: usize = FB_THREE_CENTER_AUX_FIELDS as usize;
/// The fields of one packed cell.
const CELL_FIELDS: usize = FB_THREE_CENTER_PACKED_CELL_FIELDS as usize;

/// One class `(la, lb, lP)` of cells: the orbital class it crosses, by
/// index, the aux momentum, its first cell and its cell count.
#[derive(Debug, Clone, Copy)]
struct CellClass {
    class: usize,
    lp: i32,
    begin: usize,
    n_cell: usize,
}

/// What the tasks need, formed at planning: the first trip's shape, of
/// which every trip's is a prefix, and every trip's pairs' own fields,
/// from which the tables task writes its trip's cells.
#[derive(Debug)]
struct Staging {
    /// A trip's columns.
    n_col: usize,
    /// The aux basis's Cartesian functions: each column's length.
    n_aux: usize,
    /// The orbital classes, whose bra fields the launches read in place.
    classes: Vec<ClassRows>,
    /// The aux records, field-major: the aux row.
    aux: Vec<f64>,
    n_aux_rec: usize,
    /// The cell classes of the first trip, in launch order.
    cells: Vec<CellClass>,
    /// Per cell class, its pairs' places in a trip.
    cell_pairs: Vec<Range<usize>>,
    /// The cells of the first trip, every class's: the cells row holds
    /// `CELL_FIELDS` times it.
    n_cells: usize,
    /// A trip's pairs, every trip's but the last's, which may be fewer.
    per_trip: usize,
    /// Each place's first column in a trip, the same in every trip.
    column: Vec<usize>,
    /// Every trip's pairs in order, each its record range and whether it
    /// is a shell's with itself.
    pair: Vec<([i32; 2], bool)>,
    /// The aux shells of each momentum, and each shell's record range
    /// (`n_shells + 1` long) and first Cartesian function.
    aux_of_l: Vec<Vec<usize>>,
    aux_rec_begin: Vec<i32>,
    aux_ao_offset: Vec<i32>,
}

impl Staging {
    /// Trip `trip`'s pairs: `per_trip`, or what is left for the last.
    fn trip_pairs(&self, trip: usize) -> usize {
        self.per_trip.min(self.pair.len() - trip * self.per_trip)
    }

    /// Cell class `c`'s places in a trip of `pairs` pairs, a prefix of the
    /// first trip's, and its cells there: those places times its aux
    /// shells.
    fn trip_cells(&self, c: usize, pairs: usize) -> (Range<usize>, usize) {
        let places = &self.cell_pairs[c];
        let places = places.start.min(pairs)..places.end.min(pairs);
        let n_cell = places.len() * self.aux_of_l[self.cells[c].lp as usize].len();
        (places, n_cell)
    }

    /// Trip `trip`'s cells into `out`, the cells row: per cell class at
    /// `CELL_FIELDS` times its first cell in the first trip, field-major
    /// over the trip's cells of it, the pair outer and the aux shell
    /// inner. In range: every destination is below `n_col n_aux`, which
    /// fits an i32 ([`staging`]).
    ///
    /// # Panics
    /// If `trip` is past the last trip.
    fn write_cells(&self, trip: usize, out: &mut [i32]) {
        let (base, pairs) = (trip * self.per_trip, self.trip_pairs(trip));
        for (c, cc) in self.cells.iter().enumerate() {
            let shells = &self.aux_of_l[cc.lp as usize];
            let (places, n_cell) = self.trip_cells(c, pairs);
            let run = &mut out[CELL_FIELDS * cc.begin..CELL_FIELDS * (cc.begin + n_cell)];
            let mut k = 0;
            for place in places {
                let ([rec0, rec1], diagonal) = self.pair[base + place];
                for &sh in shells {
                    let fields = [
                        rec0,
                        rec1,
                        self.aux_rec_begin[sh],
                        self.aux_rec_begin[sh + 1],
                        (self.column[place] * self.n_aux) as i32 + self.aux_ao_offset[sh],
                        i32::from(diagonal),
                    ];
                    for (f, v) in fields.into_iter().enumerate() {
                        run[f * n_cell + k] = v;
                    }
                    k += 1;
                }
            }
        }
    }
}

/// Add the Cartesian (mn|P) of a batch of shell pairs of an orbital basis
/// and an auxiliary one to `dag` on GPU worker `worker`, and return it,
/// packed: an output row of `n_col · aux.n_cart()` doubles, `n_col` the
/// batch's columns, column `c`'s `aux.n_cart()` aux functions contiguous,
/// element `(c, P)` at `c · aux.n_cart() + P`.
///
/// `pairs` are batches of `batch` shell pairs side by side, the last of
/// what is left, each pair by its index among the pairs of `tables` in
/// their order (class by class, as [`RecordTables::classes`] lists them),
/// batch `k` the `k`-th `batch` of them; outside a loop the batch is the
/// first, and in a counted loop of `pairs.len().div_ceil(batch)` trips
/// the tasks compute batch `k` in trip `k`, from their views' iteration.
/// Each pair of a batch owns one
/// block of columns, side by side from column 0 in the batch's order: a
/// pair of momenta `(la, lb)` owns `n_cart(la) · n_cart(lb)` columns,
/// element `(a, b)` of its block, `a` its bra shell's Cartesian and `b`
/// its ket's, at its first column plus `a · n_cart(lb) + b`. A shell's
/// pair with itself owns its full square block; any other pair's `(m, n)`
/// stands for `(n, m)` too. So a range of the packed shell-pair map's
/// kept pairs (`fb_integrals_eri_density_fitting_pair_map`) has that
/// map's Cartesian columns, each relative to the range's first, and every
/// element is bitwise the square tensor's `(m, n, P)`
/// ([`fb_three_center_class`](crate::fb_three_center_class)'s Cartesian
/// tensor, which this module's test builds as the witness): the same
/// kernels' arithmetic, only the stores moved.
///
/// Every batch's classes are a prefix of the first's, pair for pair, so
/// its columns and its launches are the front of the first's: the rows
/// and the launches are formed once, at planning, from the first batch,
/// and only the cells' record ranges are each trip's, written by the
/// tables task when it runs, with each launch's cell count, which a short
/// last batch cuts. The output keeps the first batch's length; a short
/// batch writes its own columns, the front of it, and leaves the rest.
///
/// `classes` are the orbital basis's records ([`add_record_tasks`]) and
/// `tables` the tables they were formed from; `aux` the auxiliary basis's
/// table. The tasks are `packed three-center tables` (CPU) and
/// `packed three-center` (stream), which reads the bra fields of every
/// class the batch has a pair of ([`Fields::THREE_CENTER`]) and holds the
/// uploaded tables as scratch beside the output. Every element of the
/// output is written, each by one cell's plain store, so nothing is zeroed.
///
/// A class above the kernels' table is the launcher's refusal at run time,
/// `Unsupported` naming the class, as for the square path; a caller
/// refuses one at planning.
///
/// [`add_record_tasks`]: crate::add_record_tasks
///
/// # Panics
/// If `n_col · aux.n_cart()` does not fit the kernels' 32-bit index, or a
/// class's cell count does not fit an `i32`, both counts the body forms;
/// if a batch's classes are not a prefix of the first's, pair for pair,
/// which the first's columns and launches would not fit; or a class is
/// of another worker's or plan's table ([`Dag::add_gpu`]); and when the
/// plan runs, if a trip is past the last batch.
#[requires(tables.formed(classes), "classes are the records of tables")]
#[debug_requires(batch > 0, "a batch of at least one pair")]
#[debug_requires(!pairs.is_empty(), "at least one batch")]
#[debug_requires(
    pairs.chunks(batch).all(|b| b.is_sorted_by(|a, b| a < b)),
    "each batch's pairs ascending, each once, in the tables' order"
)]
#[debug_requires(
    pairs.iter().all(|&p| p < tables.n_pairs()),
    "every pair is one of the tables'"
)]
#[debug_requires(aux.n_shells() > 0, "an aux basis of at least one shell")]
#[track_caller]
pub fn add_packed_three_center_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &AuxTable,
    pairs: &[usize],
    batch: usize,
) -> RowId<Device, f64> {
    let staging = Arc::new(staging(classes, tables, aux, pairs, batch));
    let tables = dag.add_cpu(
        "packed three-center tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
        },
    );
    dag.add_gpu(
        "packed three-center",
        worker,
        PackedThreeCenter { staging, tables },
    )
    .out
}

/// The staging of [`add_packed_three_center_tasks`]: a batch's shape,
/// formed from the first, and every batch's pairs' fields, every extent
/// it asserts checked.
///
/// # Panics
/// As [`add_packed_three_center_tasks`]: a count past an `i32`, or a
/// batch whose classes are not a prefix of the first's.
#[debug_ensures(
    ret.cells.iter().try_fold(0, |at, c| (c.begin == at).then_some(at + c.n_cell))
        == Some(ret.per_trip * aux.n_shells()),
    "every pair meets every aux shell in exactly one cell, the cell classes in order"
)]
#[track_caller]
fn staging(
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &AuxTable,
    pairs: &[usize],
    batch: usize,
) -> Staging {
    let n_aux = aux.n_cart();
    let n_l = aux.momentum.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut aux_of_l = vec![Vec::new(); n_l];
    for (s, &l) in aux.momentum.iter().enumerate() {
        aux_of_l[l as usize].push(s);
    }

    // Each pair's class, record range and diagonal flag, every batch's.
    let ends: Vec<usize> = tables
        .classes()
        .iter()
        .scan(0, |end, c| {
            *end += c.bra.len();
            Some(*end)
        })
        .collect();
    let mut of_class = Vec::with_capacity(pairs.len());
    let mut pair = Vec::with_capacity(pairs.len());
    for &p in pairs {
        let class = ends.partition_point(|&end| end <= p);
        let c = &tables.classes()[class];
        let q = p - (ends[class] - c.bra.len());
        of_class.push(class);
        pair.push(([c.rec_begin[q], c.rec_begin[q + 1]], c.bra[q] == c.ket[q]));
    }
    let per_trip = batch.min(pairs.len());
    assert!(
        of_class.chunks(per_trip).all(|b| b == &of_class[..b.len()]),
        "add_packed_three_center_tasks: every batch's pairs must be a prefix of the first's \
         classes, pair for pair"
    );

    // The first batch's columns, and its places class by class.
    let mut column = Vec::with_capacity(per_trip);
    let mut places: Vec<Range<usize>> = vec![0..0; classes.len()];
    let mut n_col = 0;
    for (place, &class) in of_class[..per_trip].iter().enumerate() {
        let c = &tables.classes()[class];
        if places[class].is_empty() {
            places[class] = place..place;
        }
        places[class].end = place + 1;
        column.push(n_col);
        n_col += n_cart_of(c.la) * n_cart_of(c.lb);
    }
    assert!(
        n_col
            .checked_mul(n_aux)
            .is_some_and(|n| i32::try_from(n).is_ok()),
        "add_packed_three_center_tasks: {n_col} columns of {n_aux} aux functions do not fit the \
         kernels' 32-bit index"
    );

    let mut cells = Vec::new();
    let mut cell_pairs = Vec::new();
    let mut n_cells = 0;
    for (class, (c, places)) in classes.iter().zip(&places).enumerate() {
        for (lp, shells) in aux_of_l.iter().enumerate() {
            let n_cell = places.len() * shells.len();
            if n_cell == 0 {
                continue;
            }
            assert!(
                i32::try_from(n_cell).is_ok(),
                "add_packed_three_center_tasks: class ({}, {}, {lp}) has {n_cell} cells, past an \
                 i32 count",
                c.la(),
                c.lb()
            );
            cells.push(CellClass {
                class,
                lp: lp as i32,
                begin: n_cells,
                n_cell,
            });
            cell_pairs.push(places.clone());
            n_cells += n_cell;
        }
    }

    let records = aux.field_major();
    // INVARIANT: an AuxTable's field-major records are AUX_FIELDS segments
    // of its n_rec, the launch's aux segments.
    debug_assert_eq!(records.len(), AUX_FIELDS * aux.n_rec());
    Staging {
        n_col,
        n_aux,
        classes: classes.to_vec(),
        aux: records,
        n_aux_rec: aux.n_rec(),
        cells,
        cell_pairs,
        n_cells,
        per_trip,
        column,
        pair,
        aux_of_l,
        aux_rec_begin: aux.rec_begin.clone(),
        aux_ao_offset: aux.ao_offset.clone(),
    }
}

/// The aux records row and the cells row, of one kind: pinned as the CPU
/// task's outputs, device as the stream task's scratch.
struct TableRows<K> {
    aux: RowId<K, f64>,
    cells: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

/// CPU: the aux records, formed at planning, and its trip's cells, from
/// the staging, into pinned rows.
struct Tables {
    staging: Arc<Staging>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let s = &self.staging;
        TableRows {
            aux: host.output("packed three-center aux", s.aux.len()),
            cells: host.output("packed three-center cells", CELL_FIELDS * s.n_cells),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        memcpy_h2p(host, rows.aux, &s.aux);
        s.write_cells(host.iteration(), host.take_mut(rows.cells));
        Ok(())
    }
}

/// GPU: the tables uploaded, and every class launched into the packed
/// tensor from its own records rows.
struct PackedThreeCenter {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct PackedRows {
    tables: TableRows<Device>,
    out: RowId<Device, f64>,
}

impl GpuTask for PackedThreeCenter {
    type Rows = PackedRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.tables.aux);
        host.read(self.tables.cells);
        let s = &self.staging;
        // Each class the batch has a pair of, once: its cell classes are
        // consecutive.
        let mut read = None;
        for cc in &s.cells {
            if read != Some(cc.class) {
                s.classes[cc.class].read(device, Fields::THREE_CENTER);
                read = Some(cc.class);
            }
        }
        PackedRows {
            tables: TableRows {
                aux: device.scratch("packed three-center aux", s.aux.len()),
                cells: device.scratch("packed three-center cells", CELL_FIELDS * s.n_cells),
            },
            out: device.output("(mn|P) packed", s.n_col * s.n_aux),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        let s = &self.staging;
        vec![("n_col", s.n_col), ("n_aux", s.n_aux)]
    }

    fn run(
        &self,
        rows: &PackedRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let s = &self.staging;
        let mut aux = device.take_mut(rows.tables.aux);
        memcpy_p2d(&mut aux, host, self.tables.aux, worker)?;
        let mut cells = device.take_mut(rows.tables.cells);
        memcpy_p2d(&mut cells, host, self.tables.cells, worker)?;
        let out = device.take_mut(rows.out);

        // In range, as every cast below: n_col n_aux fits an i32 (staging),
        // and the aux records and cells are counts the aux table and
        // staging hold in i32.
        let n_aux_rec = s.n_aux_rec as i32;
        let (n_col, n_aux) = (s.n_col as i32, s.n_aux as i32);
        let pairs = s.trip_pairs(device.iteration());
        for (c, cc) in s.cells.iter().enumerate() {
            // This trip's cells of the class, the tables task's count for
            // the same trip: none past a short last batch's pairs.
            let (_, n_cell) = s.trip_cells(c, pairs);
            if n_cell == 0 {
                continue;
            }
            let class = &s.classes[cc.class];
            let view = class.view(device, Fields::THREE_CENTER);
            // SAFETY: the view's arrays are the class's n_rec-long rows of
            // Fields::THREE_CENTER, every bra field the launcher reads (a
            // test scans it), the rest null, written by the class's records
            // task before this task's stream work (ClassRows' argument);
            // aux holds the AUX_FIELDS segments of n_aux_rec doubles, and
            // the class's cells are CELL_FIELDS segments of this trip's
            // n_cell int32s, at most the first trip's cc.n_cell, inside the
            // cells row, both copied above on this stream from the tables
            // task, which wrote this trip's n_cell cells of the class from
            // the staging formed at planning, the same trip_cells count,
            // whichever trip it was, every batch's pairs a prefix of the
            // first's classes pair for pair (staging asserts it), so its
            // columns are the front of the first's: every bra range a
            // pair's prefix in its class, so
            // inside [0, n_rec) (the classes are the tables', a contract),
            // its aux range from the AuxTable's prefix, and its destination
            // its pair's first column times n_aux plus the aux shell's
            // offset, so its block, n_cart(la) n_cart(lb) columns from the
            // pair's first, the aux shell's functions inside [0, n_aux),
            // lies inside out's n_col n_aux doubles; the view is a host
            // struct that outlives the call, which only enqueues on the
            // task's stream.
            let status = unsafe {
                fb_three_center_packed_class(
                    worker.stream(),
                    &view,
                    cc.lp,
                    n_aux_rec,
                    aux.as_ptr(),
                    n_cell as i32,
                    cells.as_ptr().wrapping_add(CELL_FIELDS * cc.begin),
                    n_col,
                    n_aux,
                    out.as_mut_ptr(),
                )
            };
            launched(
                "fb_three_center_packed_class",
                &[class.la(), class.lb(), cc.lp],
                status,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! The packed tensor against the square kernels' Cartesian tensor,
    //! bitwise: both from one records build of a basis with a shell of
    //! every orbital momentum through f on each of two centres, two
    //! primitives on one centre's, against an aux basis with a shell of
    //! every momentum through i on each of two centres, so every one of
    //! the kernels' 70 classes is launched, with diagonal, one-centre and
    //! two-centre pairs. The packed kernels are the square ones' arithmetic
    //! in the same form with plain stores (FelizJueves's THE PACKED
    //! OUTPUT), so the bar is equality of the bits, with no rounding to
    //! bound.

    use std::sync::{Arc, Mutex};

    use fb_core_arena::memcpy_d2p;

    use super::*;
    use crate::records::{ClassPairs, ShellTable};
    use crate::{FB_THREE_CENTER_CELL_FIELDS, add_record_tasks, fb_three_center_class};

    /// An orbital shell: its momentum, its centre and its primitives.
    type Shell = (i32, [f64; 3], Vec<(f64, f64)>);

    /// The orbital shells.
    fn shells() -> Vec<Shell> {
        let a = [0.0, 0.1, -0.2];
        let b = [0.4, -0.9, 1.3];
        let mut out = Vec::new();
        for l in 0..=3 {
            let e = 0.6 + 0.35 * f64::from(l);
            out.push((l, a, vec![(2.5 * e, 0.7), (e, 0.4)]));
            out.push((l, b, vec![(1.3 * e, 0.9)]));
        }
        out
    }

    /// The tables of [`shells`]: every pair `i <= j` once, staged with the
    /// lower momentum first, in classes `(la, lb)` ascending.
    fn tables() -> RecordTables {
        let shells = shells();
        let mut t = ShellTable {
            prim_begin: vec![0],
            ..ShellTable::default()
        };
        let mut n_cart = 0;
        let mut rank = [0; 4];
        for (l, c, prims) in &shells {
            for &(e, k) in prims {
                t.exponent.push(e);
                t.coefficient.push(k);
            }
            t.prim_begin.push(t.exponent.len() as i32);
            t.c_x.push(c[0]);
            t.c_y.push(c[1]);
            t.c_z.push(c[2]);
            t.ao_offset.push(n_cart as i32);
            t.rank.push(rank[*l as usize]);
            rank[*l as usize] += 1;
            n_cart += n_cart_of(*l);
        }
        let mut classes = Vec::new();
        for la in 0..=3 {
            for lb in la..=3 {
                let mut c = ClassPairs {
                    la,
                    lb,
                    bra: Vec::new(),
                    ket: Vec::new(),
                    rec_begin: vec![0],
                };
                for i in 0..shells.len() {
                    for j in i..shells.len() {
                        let (li, lj) = (shells[i].0, shells[j].0);
                        let (bra, ket) = if li <= lj { (i, j) } else { (j, i) };
                        if (shells[bra].0, shells[ket].0) != (la, lb) {
                            continue;
                        }
                        c.bra.push(bra as u32);
                        c.ket.push(ket as u32);
                        let n = shells[bra].2.len() * shells[ket].2.len();
                        c.rec_begin.push(c.rec_begin.last().unwrap() + n as i32);
                    }
                }
                classes.push(c);
            }
        }
        RecordTables::new(t, classes, n_cart)
    }

    /// The aux basis: a shell of every momentum through i on each of two
    /// centres, one of them an orbital centre.
    fn aux() -> AuxTable {
        let centres = [[0.0, 0.1, -0.2], [-0.7, 0.5, 0.8]];
        let n = 2 * (0..=6).map(n_cart_of).sum::<usize>();
        let mut aux = AuxTable::new(n);
        let mut at = 0;
        for (k, c) in centres.into_iter().enumerate() {
            for l in 0..=6 {
                let e = 0.9 + 0.2 * f64::from(l) + 0.15 * k as f64;
                let prims = if l % 2 == 0 {
                    vec![(e, 1.0)]
                } else {
                    vec![(3.0 * e, 0.6), (e, 0.5)]
                };
                aux.push_shell(l, c, at, prims);
                at += n_cart_of(l);
            }
        }
        aux
    }

    /// CPU: the square path's aux records and cells, every shell pair with
    /// every aux shell, its three offsets the pair's and the aux shell's.
    struct SquareTables {
        aux: Vec<f64>,
        cells: Vec<i32>,
    }

    impl CpuTask for SquareTables {
        type Rows = TableRows<Pinned>;

        fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
            TableRows {
                aux: host.output("square aux", self.aux.len()),
                cells: host.output("square cells", self.cells.len()),
            }
        }

        fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
            memcpy_h2p(host, rows.aux, &self.aux);
            memcpy_h2p(host, rows.cells, &self.cells);
            Ok(())
        }
    }

    /// GPU: the square kernels into the Cartesian `n_cart² n_aux` tensor.
    struct Square {
        tables: TableRows<Pinned>,
        classes: Vec<ClassRows>,
        /// Each launch: class index, aux momentum, first cell, cell count.
        launches: Vec<CellClass>,
        n_aux_rec: usize,
        n_cart: usize,
        n_aux: usize,
    }

    impl GpuTask for Square {
        type Rows = PackedRows;

        fn layout(
            &self,
            device: &mut Layout<'_, Device>,
            host: &mut Layout<'_, Pinned>,
        ) -> Self::Rows {
            host.read(self.tables.aux);
            host.read(self.tables.cells);
            for c in &self.classes {
                c.read(device, Fields::THREE_CENTER);
            }
            PackedRows {
                tables: TableRows {
                    aux: device.scratch("square aux", AUX_FIELDS * self.n_aux_rec),
                    cells: device.scratch(
                        "square cells",
                        FB_THREE_CENTER_CELL_FIELDS as usize
                            * self.launches.iter().map(|l| l.n_cell).sum::<usize>(),
                    ),
                },
                out: device.output("square", self.n_cart * self.n_cart * self.n_aux),
            }
        }

        fn run(
            &self,
            rows: &PackedRows,
            device: &Bufs<'_, Device>,
            host: &CopyOnly<'_, Pinned>,
            worker: &DeviceStream,
        ) -> Status {
            let mut aux = device.take_mut(rows.tables.aux);
            memcpy_p2d(&mut aux, host, self.tables.aux, worker)?;
            let mut cells = device.take_mut(rows.tables.cells);
            memcpy_p2d(&mut cells, host, self.tables.cells, worker)?;
            let out = device.take_mut(rows.out);
            for l in &self.launches {
                let class = &self.classes[l.class];
                let view = class.view(device, Fields::THREE_CENTER);
                // SAFETY: as the square path's launch (three_center.rs): the
                // cells are the test's, every range a pair's prefix or an
                // aux shell's, every offset a shell's inside the tensor.
                let status = unsafe {
                    fb_three_center_class(
                        worker.stream(),
                        &view,
                        l.lp,
                        self.n_aux_rec as i32,
                        aux.as_ptr(),
                        l.n_cell as i32,
                        cells
                            .as_ptr()
                            .wrapping_add(FB_THREE_CENTER_CELL_FIELDS as usize * l.begin),
                        self.n_cart as i32,
                        self.n_aux as i32,
                        out.as_mut_ptr(),
                    )
                };
                launched(
                    "fb_three_center_class",
                    &[class.la(), class.lb(), l.lp],
                    status,
                )?;
            }
            Ok(())
        }
    }

    /// GPU: a device row copied into a pinned one; CPU: that one read out.
    struct Download(RowId<Device, f64>, usize);
    struct Collect(RowId<Pinned, f64>, Arc<Mutex<Vec<f64>>>);

    impl GpuTask for Download {
        type Rows = RowId<Pinned, f64>;

        fn layout(
            &self,
            device: &mut Layout<'_, Device>,
            host: &mut Layout<'_, Pinned>,
        ) -> Self::Rows {
            device.read(self.0);
            host.output("host copy", self.1)
        }

        fn run(
            &self,
            pinned: &Self::Rows,
            device: &Bufs<'_, Device>,
            host: &CopyOnly<'_, Pinned>,
            stream: &DeviceStream,
        ) -> Status {
            memcpy_d2p(host, *pinned, device.get(self.0), stream)
        }
    }

    impl CpuTask for Collect {
        type Rows = ();

        fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
            host.read(self.0);
        }

        fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
            *self.1.lock().unwrap() = host.get(self.0).to_vec();
            Ok(())
        }
    }

    /// `row`, `len` doubles, brought to the host after the plan runs.
    fn download(dag: &mut Dag<'_>, row: RowId<Device, f64>, len: usize) -> Arc<Mutex<Vec<f64>>> {
        let pinned = dag.add_gpu("download", 0, Download(row, len));
        let out = Arc::new(Mutex::new(Vec::new()));
        dag.add_cpu("collect", 0, Collect(pinned, Arc::clone(&out)));
        out
    }

    /// Every pair of `pairs`' block in `packed`, laid out from column 0,
    /// against the square tensor's elements, bit for bit; the count of
    /// elements compared.
    fn compare(packed: &[f64], square: &[f64], pairs: &[usize], n_aux: usize) -> usize {
        let t = tables();
        let offset = &t.shells().ao_offset;
        let n = t.n_cart();
        let mut where_ = Vec::new();
        for c in t.classes() {
            for p in 0..c.bra.len() {
                where_.push((
                    offset[c.bra[p] as usize] as usize,
                    offset[c.ket[p] as usize] as usize,
                    n_cart_of(c.la),
                    n_cart_of(c.lb),
                ));
            }
        }
        let mut column = 0;
        let mut misses = Vec::new();
        for &pair in pairs {
            let (oa, ob, na, nb) = where_[pair];
            for a in 0..na {
                for b in 0..nb {
                    for q in 0..n_aux {
                        let got = packed[(column + a * nb + b) * n_aux + q];
                        let want = square[((oa + a) * n + ob + b) * n_aux + q];
                        if got.to_bits() != want.to_bits() {
                            misses
                                .push(format!("pair {pair} ({a}, {b}, {q}): {got:e} != {want:e}"));
                        }
                    }
                }
            }
            column += na * nb;
        }
        assert_eq!(packed.len(), column * n_aux, "the packed tensor's length");
        assert!(
            misses.is_empty(),
            "{} of {} elements differ:\n{}",
            misses.len(),
            packed.len(),
            misses[..misses.len().min(20)].join("\n")
        );
        packed.len()
    }

    /// Every pair as one batch, and a sub-range with pairs screened out
    /// (every other pair of 7..31, columns relative to its first), each
    /// bitwise the square kernels' Cartesian tensor, every element nonzero
    /// somewhere. Needs a GPU.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn packed_blocks_are_the_square_tensor_bitwise() {
        let aux = aux();
        let n_aux = aux.n_cart();
        let t = tables();
        let n = t.n_cart();
        let every: Vec<usize> = (0..t.n_pairs()).collect();
        let some: Vec<usize> = (7..31).step_by(2).collect();

        // The square path's cells, one launch per (class, lP).
        let mut cells = Vec::new();
        let mut launches = Vec::new();
        let offset = &t.shells().ao_offset;
        for (class, c) in t.classes().iter().enumerate() {
            for lp in 0..=6 {
                let shells: Vec<usize> = (0..aux.n_shells())
                    .filter(|&s| aux.momentum[s] == lp)
                    .collect();
                let n_cell = c.bra.len() * shells.len();
                if n_cell == 0 {
                    continue;
                }
                let mut run = vec![0; FB_THREE_CENTER_CELL_FIELDS as usize * n_cell];
                let mut k = 0;
                for p in 0..c.bra.len() {
                    for &s in &shells {
                        let f = [
                            c.rec_begin[p],
                            c.rec_begin[p + 1],
                            aux.rec_begin[s],
                            aux.rec_begin[s + 1],
                            offset[c.bra[p] as usize],
                            offset[c.ket[p] as usize],
                            aux.ao_offset[s],
                        ];
                        for (i, v) in f.into_iter().enumerate() {
                            run[i * n_cell + k] = v;
                        }
                        k += 1;
                    }
                }
                launches.push(CellClass {
                    class,
                    lp,
                    begin: cells.len() / FB_THREE_CENTER_CELL_FIELDS as usize,
                    n_cell,
                });
                cells.extend(run);
            }
        }
        assert_eq!(launches.len(), 70, "every class of the kernels' table");

        let mut dag = Dag::new();
        let classes = add_record_tasks(&mut dag, 0, tables());
        let tables_rows = dag.add_cpu(
            "square tables",
            0,
            SquareTables {
                aux: aux.field_major(),
                cells,
            },
        );
        let square = dag
            .add_gpu(
                "square",
                0,
                Square {
                    tables: tables_rows,
                    classes: classes.clone(),
                    launches,
                    n_aux_rec: aux.n_rec(),
                    n_cart: n,
                    n_aux,
                },
            )
            .out;
        let all =
            add_packed_three_center_tasks(&mut dag, 0, &classes, &t, &aux, &every, every.len());
        let part =
            add_packed_three_center_tasks(&mut dag, 0, &classes, &t, &aux, &some, some.len());
        let col = |pairs: &[usize]| -> usize {
            let mut sizes = Vec::new();
            for c in t.classes() {
                sizes.extend(std::iter::repeat_n(
                    n_cart_of(c.la) * n_cart_of(c.lb),
                    c.bra.len(),
                ));
            }
            pairs.iter().map(|&p| sizes[p]).sum()
        };
        let square = download(&mut dag, square, n * n * n_aux);
        let all = download(&mut dag, all, col(&every) * n_aux);
        let part = download(&mut dag, part, col(&some) * n_aux);

        let mut instance = dag.plan().instantiate(&[0]);
        let mut backend = instance.backend();
        instance.run(&mut backend, 1).result.unwrap();

        let square = square.lock().unwrap();
        assert!(square.iter().all(|v| v.is_finite()) && square.iter().any(|&v| v != 0.0));
        let checked = compare(&all.lock().unwrap(), &square, &every, n_aux);
        // A diagonal block is the full square, and every off-diagonal pair
        // once: the lower triangle at shell-pair granularity.
        assert_eq!(checked / n_aux, {
            let sizes: Vec<usize> = shells().iter().map(|s| n_cart_of(s.0)).collect();
            let mut cols = 0;
            for i in 0..sizes.len() {
                for j in i..sizes.len() {
                    cols += sizes[i] * sizes[j];
                }
            }
            cols
        });
        let part_checked = compare(&part.lock().unwrap(), &square, &some, n_aux);
        println!(
            "packed: {checked} elements of every pair and {part_checked} of the sub-range, bitwise \
             the square tensor"
        );
    }
}
