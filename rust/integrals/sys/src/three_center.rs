//! The three-center integrals' device path: (mn|P) of an orbital basis,
//! whose records the records tasks formed ([`ClassRows`]) from tables the
//! caller hands in too ([`RecordTables`]), and an auxiliary one
//! ([`AuxTable`]), as three tasks ([`add_three_center_tasks`]).
//!
//! The tasks, added in this order:
//!
//! - **`download three-center bra`**, the planner's copy node
//!   ([`Dag::download_rows`]) on the worker's stream: every class's bra
//!   fields (ζ, P, P − A, P − B, `s`) copied into pinned rows named after
//!   the field and the class, one node and one event.
//! - **`three-center tables`**, on the CPU: each class's bra fields packed
//!   field-major into one pinned row as they are, then the aux records,
//!   formed at planning, behind them; and the cells, formed at planning,
//!   into another.
//! - **`three-center`**, on the worker's stream: both rows uploaded into
//!   scratch; U and U_aux read from the caller's device rows (the plan's
//!   `upload U` rows, copied up once for every reader), one
//!   [`fb_three_center_class`] launch per class that has a cell into the
//!   Cartesian tensor (scratch), the three spherical GEMMs and
//!   [`fb_three_center_symmetrize`] into (mn|P), its output.
//!
//! An `eri3` thread walks its shell pair's records as one run. The records
//! are in enumeration order, each shell pair's one contiguous run of its
//! class (`ClassRows`' order), so nothing is regrouped: a pair's run is
//! its record prefix in the tables, and its AO offsets are its shells'
//! there. Every count and every index a cell holds is therefore fixed at
//! planning, from tables [`RecordTables::new`] checked: a pair's record
//! range inside its class, its offsets each opening a block inside the
//! orbital basis's `n_cart`, the aux shell's range and offset from the
//! [`AuxTable`]. That the records are the tables' is a contract, checked
//! always, since the ranges index the records' rows.

use std::sync::Arc;

use contracts::{debug_ensures, requires};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_log::{fits_i32, le};
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::{gemm, gemm_strided_batched};
use fb_core_wwr::handle::DeviceHandleStream;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

use crate::records::{ClassRows, RecordTables};
use crate::{
    AuxTable, FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_AUX_MAX_MOMENTUM,
    FB_THREE_CENTER_BRA_FIELDS, FB_THREE_CENTER_CELL_FIELDS, fb_three_center_class,
    fb_three_center_symmetrize, launched,
};

/// The fields of one bra record: the first of [`ClassRows::f64_rows`].
const BRA_FIELDS: usize = FB_THREE_CENTER_BRA_FIELDS as usize;
/// The fields of one aux record.
const AUX_FIELDS: usize = FB_THREE_CENTER_AUX_FIELDS as usize;
/// The fields of one cell.
const CELL_FIELDS: usize = FB_THREE_CENTER_CELL_FIELDS as usize;
/// The bra fields' names, as the download's rows are named.
const BRA_NAMES: [&str; BRA_FIELDS] = [
    "zeta", "p_x", "p_y", "p_z", "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z", "s",
];

/// One class `(la, lb, lP)` of cells: the orbital class it crosses, by
/// index, the aux momentum, its first cell and its cell count.
#[derive(Debug, Clone, Copy)]
struct CellClass {
    class: usize,
    lp: i32,
    begin: usize,
    n_cell: usize,
}

/// What the tasks need, formed at planning.
#[derive(Debug)]
struct Staging {
    n_cart: usize,
    n_sph: usize,
    n_aux_cart: usize,
    n_aux_sph: usize,
    /// Each orbital class's momenta, record count, and first record among
    /// all classes'.
    classes: Vec<(i32, i32, usize, usize)>,
    /// Every class's records.
    n_bra: usize,
    /// The aux records, field-major: the records row behind the bra
    /// records.
    tail: Vec<f64>,
    n_aux_rec: usize,
    /// The cell classes, in launch order.
    cells: Vec<CellClass>,
    /// Every cell, field-major: the cells row.
    cell_fields: Vec<i32>,
    n_cells: usize,
}

impl Staging {
    /// The records row's length: the bra records, then the tail.
    fn n_records(&self) -> usize {
        BRA_FIELDS * self.n_bra + self.tail.len()
    }
}

/// Add the spherical three-center integrals (mn|P) of an orbital basis and
/// an auxiliary one to `dag` on GPU worker `worker`, and return them: an
/// output row of `n_sph² n_aux_sph` doubles, element `(m, n, P)` at
/// `P + n_aux_sph (n + n_sph m)`, P fastest, exactly symmetric in m and n.
///
/// `classes` are the orbital basis's records ([`add_record_tasks`]) and
/// `tables` the tables they were formed from, whose shell pairs give each
/// pair's record range and AO offsets; `aux` the auxiliary basis's table;
/// `[u, u_aux]` the device rows of the two bases' Cartesian-to-spherical
/// transforms, U `n_cart` × `n_sph` and U_aux `aux.n_cart()` × `n_aux_sph`,
/// column-major, written by earlier tasks of `dag` on `worker`
/// (`fb_recipes_matrix_ao::add_harmonics`'s), and `[n_sph, n_aux_sph]`
/// their column counts. The tasks are `download three-center bra` (stream,
/// the planner's copy node), `three-center tables` (CPU) and
/// `three-center` (stream), which reads U and U_aux and holds, as scratch,
/// the Cartesian tensor (`n_cart² aux.n_cart()` doubles) and the
/// transform's intermediate (`n_cart² n_aux_sph`), so all are in its table
/// at planning, beside the output.
///
/// A class above the kernels' table ([`FB_THREE_CENTER_MAX_MOMENTUM`](crate::FB_THREE_CENTER_MAX_MOMENTUM)
/// for the orbital shells, [`FB_THREE_CENTER_AUX_MAX_MOMENTUM`] for the
/// aux) is the launcher's refusal at run time, `Unsupported` naming the
/// class; a caller refuses one at planning.
///
/// [`add_record_tasks`]: crate::add_record_tasks
///
/// # Panics
/// If the classes' records together pass an `i32` count (the kernels'
/// record index); a class's cell count does not fit an `i32`; a class, U or
/// U_aux is of another worker's or plan's table ([`Dag::add_gpu`]); or, at
/// run time, U's row is shorter than `n_cart · n_sph` doubles or U_aux's
/// than `aux.n_cart() · n_aux_sph` (`fb_core_wwr::blas`'s GEMMs).
#[requires(tables.formed(classes), "classes are the records of tables")]
#[requires(
    fits_i32!(tables.n_cart().checked_pow(2).and_then(|n2| n2.checked_mul(aux.n_cart()))),
    "n_cart² aux.n_cart() fits the kernels' 32-bit index"
)]
#[requires(
    le!(1, n_sph[0]) && le!(n_sph[0], tables.n_cart()),
    "U has 1 to its n_cart rows of columns"
)]
#[requires(
    le!(1, n_sph[1]) && le!(n_sph[1], aux.n_cart()),
    "U_aux has 1 to its aux.n_cart() rows of columns"
)]
#[debug_requires(!classes.is_empty(), "an orbital basis of at least one shell")]
#[debug_requires(aux.n_shells() > 0, "an aux basis of at least one shell")]
#[track_caller]
pub fn add_three_center_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &AuxTable,
    u: [RowId<Device, f64>; 2],
    n_sph: [usize; 2],
) -> RowId<Device, f64> {
    let staging = Arc::new(staging(classes, tables, aux, n_sph));
    let downloaded = download(dag, worker, classes);
    let tables = dag.add_cpu(
        "three-center tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
            downloaded,
        },
    );
    dag.add_gpu("three-center", worker, ThreeCenter { staging, tables, u })
        .out
}

/// The staging of [`add_three_center_tasks`]: the cells formed from the
/// tables' shell pairs, every extent it asserts checked.
#[debug_ensures(
    ret.n_cells == tables.n_pairs() * aux.n_shells(),
    "every shell pair meets every aux shell in exactly one cell"
)]
#[debug_ensures(
    ret.cells.iter().try_fold(0, |at, c| (c.begin == at).then_some(at + c.n_cell)) == Some(ret.n_cells),
    "the cell classes tile 0..n_cells in order"
)]
#[debug_ensures(
    ret.n_bra == classes.iter().map(|c| c.n_rec() as usize).sum::<usize>(),
    "the bra records are every class's"
)]
#[track_caller]
fn staging(
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &AuxTable,
    [n_sph, n_aux_sph]: [usize; 2],
) -> Staging {
    let n_cart = tables.n_cart();
    let n_bra: usize = classes.iter().map(|c| c.n_rec() as usize).sum();
    assert!(
        i32::try_from(n_bra).is_ok(),
        "add_three_center_tasks: {n_bra} records, past the kernels' i32 record index"
    );
    let n_l = FB_THREE_CENTER_AUX_MAX_MOMENTUM.max(aux.momentum.iter().copied().max().unwrap_or(0))
        as usize
        + 1;
    let mut aux_of_l = vec![Vec::new(); n_l];
    for (s, &l) in aux.momentum.iter().enumerate() {
        aux_of_l[l as usize].push(s);
    }

    // Each class's pairs, in enumeration order: a pair's record range,
    // among every class's records, is its prefix behind the class's first
    // record, and its offsets are its two shells'. In range: n_bra fits an
    // i32 (above), and so does every AO offset (RecordTables::new).
    let offset = &tables.shells().ao_offset;
    let mut out = Vec::with_capacity(classes.len());
    let mut cells = Vec::new();
    let mut cell_classes = Vec::new();
    let mut n_cells = 0;
    let mut first = 0;
    for (class, (c, pairs)) in classes.iter().zip(tables.classes()).enumerate() {
        let n_rec = c.n_rec() as usize;
        out.push((c.la(), c.lb(), n_rec, first));
        let runs: Vec<[i32; 4]> = (0..pairs.bra.len())
            .map(|p| {
                [
                    first as i32 + pairs.rec_begin[p],
                    first as i32 + pairs.rec_begin[p + 1],
                    offset[pairs.bra[p] as usize],
                    offset[pairs.ket[p] as usize],
                ]
            })
            .collect();
        for (lp, shells) in aux_of_l.iter().enumerate() {
            let n_cell = runs.len() * shells.len();
            if n_cell == 0 {
                continue;
            }
            assert!(
                i32::try_from(n_cell).is_ok(),
                "add_three_center_tasks: class ({}, {}, {lp}) has {n_cell} cells, past an i32 \
                 count",
                c.la(),
                c.lb()
            );
            // The class's cells, field-major, the pair outer and the aux
            // shell inner.
            let mut run = vec![0; CELL_FIELDS * n_cell];
            let mut k = 0;
            for &[rec0, rec1, oa, ob] in &runs {
                for &sh in shells {
                    let fields = [
                        rec0,
                        rec1,
                        aux.rec_begin[sh],
                        aux.rec_begin[sh + 1],
                        oa,
                        ob,
                        aux.ao_offset[sh],
                    ];
                    for (f, v) in fields.into_iter().enumerate() {
                        run[f * n_cell + k] = v;
                    }
                    k += 1;
                }
            }
            cells.extend(run);
            cell_classes.push(CellClass {
                class,
                lp: lp as i32,
                begin: n_cells,
                n_cell,
            });
            n_cells += n_cell;
        }
        first += n_rec;
    }

    let tail = aux.field_major();
    // INVARIANT: an AuxTable's field-major records are AUX_FIELDS segments
    // of its n_rec, the launch's aux segments.
    debug_assert_eq!(tail.len(), AUX_FIELDS * aux.n_rec());
    Staging {
        n_cart,
        n_sph,
        n_aux_cart: aux.n_cart(),
        n_aux_sph,
        classes: out,
        n_bra,
        tail,
        n_aux_rec: aux.n_rec(),
        cells: cell_classes,
        cell_fields: cells,
        n_cells,
    }
}

/// Each class's bra fields copied into pinned rows by one planner copy
/// node, `download three-center bra`, each row named after its field and
/// class: one array of rows per class.
fn download(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[ClassRows],
) -> Vec<[RowId<Pinned, f64>; BRA_FIELDS]> {
    let name = |what: &str, c: &ClassRows| format!("{what} ({},{})", c.la(), c.lb());
    let fields = classes
        .iter()
        .flat_map(|c| (0..BRA_FIELDS).map(move |f| (c, f)))
        .map(|(c, f)| (c.f64_rows()[f], name(BRA_NAMES[f], c), c.n_rec() as usize));
    let fields = dag.download_rows(worker, "three-center bra", fields);
    (0..classes.len())
        .map(|c| std::array::from_fn(|f| fields[BRA_FIELDS * c + f]))
        .collect()
}

/// The records row (the bra records, then the aux records) and the cells
/// row, of one kind: pinned as the CPU task's outputs, device as the
/// stream task's scratch.
struct TableRows<K> {
    records: RowId<K, f64>,
    cells: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

/// CPU: the downloaded bra fields packed field-major, the aux records
/// behind them, and the cells, into pinned rows.
struct Tables {
    staging: Arc<Staging>,
    downloaded: Vec<[RowId<Pinned, f64>; BRA_FIELDS]>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        for fields in &self.downloaded {
            for &row in fields {
                host.read(row);
            }
        }
        let s = &self.staging;
        TableRows {
            records: host.output("three-center records", s.n_records()),
            cells: host.output("three-center cells", CELL_FIELDS * s.n_cells),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        let records = host.take_mut(rows.records);
        let n_bra = s.n_bra;
        records[BRA_FIELDS * n_bra..].copy_from_slice(&s.tail);
        for (fields, &(_, _, n_rec, base)) in self.downloaded.iter().zip(&s.classes) {
            for (f, &row) in fields.iter().enumerate() {
                records[f * n_bra + base..f * n_bra + base + n_rec].copy_from_slice(host.get(row));
            }
        }
        host.take_mut(rows.cells).copy_from_slice(&s.cell_fields);
        Ok(())
    }
}

/// GPU: the tables uploaded, every class launched into the
/// Cartesian tensor, the three spherical GEMMs, and the (m, n) average into
/// (mn|P).
struct ThreeCenter {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
    /// U and U_aux, the caller's device rows.
    u: [RowId<Device, f64>; 2],
}

#[derive(Clone, Copy)]
struct ThreeCenterRows {
    tables: TableRows<Device>,
    cartesian: RowId<Device, f64>,
    work: RowId<Device, f64>,
    out: RowId<Device, f64>,
}

impl GpuTask for ThreeCenter {
    type Rows = ThreeCenterRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.tables.records);
        host.read(self.tables.cells);
        device.read(self.u[0]);
        device.read(self.u[1]);
        let s = &self.staging;
        ThreeCenterRows {
            tables: TableRows {
                records: device.scratch("three-center records", s.n_records()),
                cells: device.scratch("three-center cells", CELL_FIELDS * s.n_cells),
            },
            cartesian: device.scratch("(mn|P) cartesian", s.n_cart * s.n_cart * s.n_aux_cart),
            work: device.scratch("(mn|P) work", s.n_cart * s.n_cart * s.n_aux_sph),
            out: device.output("(mn|P)", s.n_sph * s.n_sph * s.n_aux_sph),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        let s = &self.staging;
        vec![("n_ao", s.n_sph), ("n_aux", s.n_aux_sph)]
    }

    fn run(
        &self,
        rows: &ThreeCenterRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let s = &self.staging;
        let mut records = device.take_mut(rows.tables.records);
        memcpy_p2d(&mut records, host, self.tables.records, worker)?;
        let mut cells = device.take_mut(rows.tables.cells);
        memcpy_p2d(&mut cells, host, self.tables.cells, worker)?;
        let (u, u_aux) = (device.get(self.u[0]), device.get(self.u[1]));
        let mut cartesian = device.take_mut(rows.cartesian);
        let mut work = device.take_mut(rows.work);
        let out = device.take_mut(rows.out);

        // In range, as every cast below: n_cart² n_aux_cart fits an i32
        // (staging), and the records and cells are counts the classes and
        // the aux table hold in i32.
        let (n_bra, n_aux_rec) = (s.n_bra as i32, s.n_aux_rec as i32);
        let (n, n_aux) = (s.n_cart as i32, s.n_aux_cart as i32);
        let aux = records.as_ptr().wrapping_add(BRA_FIELDS * s.n_bra);
        for cc in &s.cells {
            let (la, lb, ..) = s.classes[cc.class];
            // SAFETY: records holds the BRA_FIELDS segments of n_bra doubles
            // and then the AUX_FIELDS segments of n_aux_rec, and the class's
            // cells are CELL_FIELDS segments of n_cell int32s inside the
            // cells row, all copied above on this stream from the tables
            // task, which wrote the cells staging formed at planning: every
            // bra range a pair's prefix inside its class's records (the
            // classes are the tables', a contract), so inside [0, n_bra), its
            // aux range from the AuxTable's prefix, its orbital offsets the
            // pair's shells', each opening a block inside [0, n_cart)
            // (RecordTables::new), and its aux offset the AuxTable's, inside
            // [0, n_aux_cart); cartesian is this task's n_cart² n_aux_cart
            // doubles; the launch only enqueues on the task's stream.
            let status = unsafe {
                fb_three_center_class(
                    worker.stream(),
                    la,
                    lb,
                    cc.lp,
                    n_bra,
                    records.as_ptr(),
                    n_aux_rec,
                    aux,
                    cc.n_cell as i32,
                    cells.as_ptr().wrapping_add(CELL_FIELDS * cc.begin),
                    n,
                    n_aux,
                    cartesian.as_mut_ptr(),
                )
            };
            launched("fb_three_center_class", &[la, lb, cc.lp], status)?;
        }

        let handle = worker.blas();
        let (ns, nas) = (s.n_sph as i32, s.n_aux_sph as i32);
        // The aux index: work = U_auxᵀ cart, n_aux_sph × n², from the
        // n_aux_cart × n² view of the Cartesian tensor, P fastest.
        gemm(
            handle,
            WWR_BLAS_OP_T,
            WWR_BLAS_OP_N,
            nas,
            n * n,
            n_aux,
            1.0,
            &u_aux,
            n_aux,
            &cartesian,
            n_aux,
            0.0,
            &mut work,
            nas,
        )?;
        // The n index: for each Cartesian m, its n_aux_sph × n block of work
        // times U, into the Cartesian tensor as n_aux_sph × n_sph blocks, U
        // shared by the batch.
        gemm_strided_batched(
            handle,
            WWR_BLAS_OP_N,
            WWR_BLAS_OP_N,
            nas,
            ns,
            n,
            1.0,
            &work,
            nas,
            i64::from(nas) * i64::from(n),
            &u,
            n,
            0,
            0.0,
            &mut cartesian,
            nas,
            i64::from(nas) * i64::from(ns),
            n,
        )?;
        // The m index: the (n_aux_sph n_sph) × n view of the tensor times U,
        // into work.
        gemm(
            handle,
            WWR_BLAS_OP_N,
            WWR_BLAS_OP_N,
            nas * ns,
            ns,
            n,
            1.0,
            &cartesian,
            nas * ns,
            &u,
            n,
            0.0,
            &mut work,
            nas * ns,
        )?;
        // The two GEMMs round (m, n) and (n, m) in different orders; half of
        // each plus half of its mirror is exactly symmetric.
        let len = s.n_sph * s.n_sph * s.n_aux_sph;
        // INVARIANT: work is n_cart² n_aux_sph and n_sph <= n_cart (staging),
        // and out is the n_sph² n_aux_sph the layout declared.
        debug_assert!(work.len() >= len && out.len() == len);
        // SAFETY: work holds at least n_sph² n_aux_sph doubles and out
        // exactly that many (above), two rows of this task, so apart; the
        // launch only enqueues on the task's stream, after the GEMMs that
        // wrote work.
        let status = unsafe {
            fb_three_center_symmetrize(worker.stream(), ns, nas, work.as_ptr(), out.as_mut_ptr())
        };
        launched("fb_three_center_symmetrize", &[], status)
    }
}
