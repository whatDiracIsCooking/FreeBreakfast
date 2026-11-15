//! The three-center integrals' device path: (mn|P) of an orbital basis,
//! whose records the records tasks formed ([`ClassRows`]) from tables the
//! caller hands in too ([`RecordTables`]), and an auxiliary one
//! ([`AuxTable`]), as two tasks ([`add_three_center_tasks`]).
//!
//! The tasks, added in this order:
//!
//! - **`three-center tables`**, on the CPU: the aux records and the cells,
//!   both formed at planning, written into two pinned rows.
//! - **`three-center`**, on the worker's stream: both rows uploaded into
//!   scratch; every class's bra fields (ζ, P, P − A, P − B, `s`,
//!   [`Fields::THREE_CENTER`]) read in place from its records rows, edges
//!   of this task; U and U_aux read from the caller's device rows (the
//!   plan's `upload U` rows, copied up once for every reader), one
//!   [`fb_three_center_class`] launch per class that has a cell into the
//!   Cartesian tensor (scratch), the three spherical GEMMs and
//!   [`fb_three_center_symmetrize`] into (mn|P), its output.
//!
//! An `eri3` thread walks its shell pair's records as one run. The records
//! are in enumeration order, each shell pair's one contiguous run of its
//! class (`ClassRows`' order), so a launch reads its class's rows as the
//! records task wrote them: a pair's run is its record prefix in the
//! tables, and its AO offsets are its shells' there. Every count and every
//! index a cell holds is therefore fixed at planning, from tables
//! [`RecordTables::new`] checked: a pair's record range inside its class,
//! its offsets each opening a block inside the orbital basis's `n_cart`,
//! the aux shell's range and offset from the [`AuxTable`]. That the records
//! are the tables' is a contract, checked always, since the ranges index
//! the records' rows.

use std::sync::Arc;

use contracts::{debug_ensures, requires};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_log::{fits_i32, le};
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::{gemm, gemm_strided_batched};
use fb_core_wwr::handle::DeviceHandleStream;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

use crate::records::{ClassRows, Fields, RecordTables};
use crate::{
    AuxTable, FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_AUX_MAX_MOMENTUM,
    FB_THREE_CENTER_CELL_FIELDS, fb_three_center_class, fb_three_center_symmetrize, launched,
};

/// The fields of one aux record.
const AUX_FIELDS: usize = FB_THREE_CENTER_AUX_FIELDS as usize;
/// The fields of one cell.
const CELL_FIELDS: usize = FB_THREE_CENTER_CELL_FIELDS as usize;

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
    /// The orbital classes, whose bra fields the launches read in place.
    classes: Vec<ClassRows>,
    /// The aux records, field-major: the aux row.
    aux: Vec<f64>,
    n_aux_rec: usize,
    /// The cell classes, in launch order.
    cells: Vec<CellClass>,
    /// Every cell, field-major: the cells row.
    cell_fields: Vec<i32>,
    n_cells: usize,
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
/// their column counts. The tasks are `three-center tables` (CPU) and
/// `three-center` (stream), which reads every class's
/// [`Fields::THREE_CENTER`], U and U_aux, and holds, as scratch, the
/// Cartesian tensor (`n_cart² aux.n_cart()` doubles) and the transform's
/// intermediate (`n_cart² n_aux_sph`), so all are in its table at
/// planning, beside the output.
///
/// A class above the kernels' table ([`FB_THREE_CENTER_MAX_MOMENTUM`](crate::FB_THREE_CENTER_MAX_MOMENTUM)
/// for the orbital shells, [`FB_THREE_CENTER_AUX_MAX_MOMENTUM`] for the
/// aux) is the launcher's refusal at run time, `Unsupported` naming the
/// class; a caller refuses one at planning.
///
/// [`add_record_tasks`]: crate::add_record_tasks
///
/// # Panics
/// If a class's cell count does not fit an `i32`; a class, U or U_aux is
/// of another worker's or plan's table ([`Dag::add_gpu`]); or, at run time,
/// U's row is shorter than `n_cart · n_sph` doubles or U_aux's than
/// `aux.n_cart() · n_aux_sph` (`fb_core_wwr::blas`'s GEMMs).
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
    let tables = dag.add_cpu(
        "three-center tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
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
#[track_caller]
fn staging(
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &AuxTable,
    [n_sph, n_aux_sph]: [usize; 2],
) -> Staging {
    let n_l = FB_THREE_CENTER_AUX_MAX_MOMENTUM.max(aux.momentum.iter().copied().max().unwrap_or(0))
        as usize
        + 1;
    let mut aux_of_l = vec![Vec::new(); n_l];
    for (s, &l) in aux.momentum.iter().enumerate() {
        aux_of_l[l as usize].push(s);
    }

    // Each class's pairs, in enumeration order: a pair's record range, in
    // its class's records, is its prefix, and its offsets are its two
    // shells'. Every entry is an i32 already (RecordTables::new).
    let offset = &tables.shells().ao_offset;
    let mut cells = Vec::new();
    let mut cell_classes = Vec::new();
    let mut n_cells = 0;
    for (class, (c, pairs)) in classes.iter().zip(tables.classes()).enumerate() {
        let runs: Vec<[i32; 4]> = (0..pairs.bra.len())
            .map(|p| {
                [
                    pairs.rec_begin[p],
                    pairs.rec_begin[p + 1],
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
    }

    let records = aux.field_major();
    // INVARIANT: an AuxTable's field-major records are AUX_FIELDS segments
    // of its n_rec, the launch's aux segments.
    debug_assert_eq!(records.len(), AUX_FIELDS * aux.n_rec());
    Staging {
        n_cart: tables.n_cart(),
        n_sph,
        n_aux_cart: aux.n_cart(),
        n_aux_sph,
        classes: classes.to_vec(),
        aux: records,
        n_aux_rec: aux.n_rec(),
        cells: cell_classes,
        cell_fields: cells,
        n_cells,
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

/// CPU: the aux records and the cells, formed at planning, into pinned
/// rows.
struct Tables {
    staging: Arc<Staging>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let s = &self.staging;
        TableRows {
            aux: host.output("three-center aux", s.aux.len()),
            cells: host.output("three-center cells", CELL_FIELDS * s.n_cells),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        memcpy_h2p(host, rows.aux, &s.aux);
        memcpy_h2p(host, rows.cells, &s.cell_fields);
        Ok(())
    }
}

/// GPU: the tables uploaded, every class launched into the Cartesian
/// tensor from its own records rows, the three spherical GEMMs, and the
/// (m, n) average into (mn|P).
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
        host.read(self.tables.aux);
        host.read(self.tables.cells);
        let s = &self.staging;
        for c in &s.classes {
            c.read(device, Fields::THREE_CENTER);
        }
        device.read(self.u[0]);
        device.read(self.u[1]);
        ThreeCenterRows {
            tables: TableRows {
                aux: device.scratch("three-center aux", s.aux.len()),
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
        let mut aux = device.take_mut(rows.tables.aux);
        memcpy_p2d(&mut aux, host, self.tables.aux, worker)?;
        let mut cells = device.take_mut(rows.tables.cells);
        memcpy_p2d(&mut cells, host, self.tables.cells, worker)?;
        let (u, u_aux) = (device.get(self.u[0]), device.get(self.u[1]));
        let mut cartesian = device.take_mut(rows.cartesian);
        let mut work = device.take_mut(rows.work);
        let out = device.take_mut(rows.out);

        // In range, as every cast below: n_cart² n_aux_cart fits an i32
        // (the contracts), and the aux records and cells are counts the
        // aux table and staging hold in i32.
        let n_aux_rec = s.n_aux_rec as i32;
        let (n, n_aux) = (s.n_cart as i32, s.n_aux_cart as i32);
        for cc in &s.cells {
            let class = &s.classes[cc.class];
            let view = class.view(device, Fields::THREE_CENTER);
            // SAFETY: the view's arrays are the class's n_rec-long rows of
            // Fields::THREE_CENTER, every bra field the launcher reads (a
            // test scans it), the rest null, written by the class's records
            // task before this task's stream work (ClassRows' argument);
            // aux holds the AUX_FIELDS segments of n_aux_rec doubles, and
            // the class's cells are CELL_FIELDS segments of n_cell int32s
            // inside the cells row, both copied above on this stream from
            // the tables task, which wrote the cells staging formed at
            // planning: every bra range a pair's prefix in its class, so
            // inside [0, n_rec) (the classes are the tables', a contract),
            // its aux range from the AuxTable's prefix, its orbital offsets
            // the pair's shells', each opening a block inside [0, n_cart)
            // (RecordTables::new), and its aux offset the AuxTable's,
            // inside [0, n_aux_cart); cartesian is this task's
            // n_cart² n_aux_cart doubles; the view is a host struct that
            // outlives the call, which only enqueues on the task's stream.
            let status = unsafe {
                fb_three_center_class(
                    worker.stream(),
                    &view,
                    cc.lp,
                    n_aux_rec,
                    aux.as_ptr(),
                    cc.n_cell as i32,
                    cells.as_ptr().wrapping_add(CELL_FIELDS * cc.begin),
                    n,
                    n_aux,
                    cartesian.as_mut_ptr(),
                )
            };
            launched(
                "fb_three_center_class",
                &[class.la(), class.lb(), cc.lp],
                status,
            )?;
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
