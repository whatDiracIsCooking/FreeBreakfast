//! The three-center integrals' device path: (mn|P) of an orbital basis,
//! whose records the records tasks formed ([`ClassRows`]), and an auxiliary
//! one ([`AuxTable`]), as three tasks ([`add_three_center_tasks`]).
//!
//! The tasks, added in this order:
//!
//! - **`three-center download`**, on the worker's stream: each class's
//!   bra fields (ζ, P, P − A, P − B, `s`) and its two AO offsets copied
//!   into pinned rows, its outputs.
//! - **`three-center tables`**, on the CPU: each class's records regrouped
//!   by shell pair, the pairs in their first record's order and the
//!   records' descending-ζ order kept within each, since an `eri3` thread
//!   walks its pair's records as one run and the records task interleaves
//!   the pairs of a class; then the aux records, U and U_aux, formed at
//!   planning, behind them in one pinned row; and the cells, one per
//!   (shell pair, aux shell), per class `(la, lb, lP)`, the pair outer and
//!   the aux shell inner, into another.
//! - **`three-center`**, on the worker's stream: both rows uploaded into
//!   scratch, one [`fb_three_center_class`] launch per class that has a
//!   cell into the Cartesian tensor (scratch), the three spherical GEMMs
//!   and [`fb_three_center_symmetrize`] into (mn|P), its output.
//!
//! Every count is fixed at planning: the records' (each class's `n_rec`),
//! the aux table's, and the shell pairs per class, which the caller gives.
//! What only a run knows is the order of the records within a class; the
//! regroup checks it against the pair counts, and every index a cell holds
//! is one this module wrote: a pair's record range from the regroup, its
//! offsets each inside the orbital basis's `n_cart` (checked again here,
//! though the records task wrote them from checked tables), the aux
//! shell's range and offset from the [`AuxTable`].

use std::sync::Arc;

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::{Error, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::{gemm, gemm_strided_batched};
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::handle::DeviceHandleStream;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

use crate::metric::n_cart_of;
use crate::records::ClassRows;
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
    /// Each orbital class's momenta, record count, first record among all
    /// classes', and shell pairs.
    classes: Vec<(i32, i32, usize, usize, usize)>,
    /// Every class's records.
    n_bra: usize,
    /// The aux records, field-major, then U and U_aux: the records row
    /// behind the bra records.
    tail: Vec<f64>,
    n_aux_rec: usize,
    /// The aux shells of each momentum, by index.
    aux_of_l: Vec<Vec<usize>>,
    aux_rec_begin: Vec<i32>,
    aux_offset: Vec<i32>,
    cells: Vec<CellClass>,
    n_cells: usize,
}

impl Staging {
    /// The records row's length: the bra records, then the tail.
    fn n_records(&self) -> usize {
        BRA_FIELDS * self.n_bra + self.tail.len()
    }

    /// Where U starts in the records row.
    fn u_at(&self) -> usize {
        BRA_FIELDS * self.n_bra + AUX_FIELDS * self.n_aux_rec
    }
}

/// Add the spherical three-center integrals (mn|P) of an orbital basis and
/// an auxiliary one to `dag` on GPU worker `worker`, and return them: an
/// output row of `n_sph² n_aux_sph` doubles, element `(m, n, P)` at
/// `P + n_aux_sph (n + n_sph m)`, P fastest, exactly symmetric in m and n.
///
/// `classes` are the orbital basis's records ([`add_record_tasks`]), and
/// `n_pairs[c]` the shell pairs of `classes[c]`; `aux` the auxiliary
/// basis's table; `u`, the orbital basis's Cartesian-to-spherical transform,
/// `n_cart` × `n_sph` column-major, and `u_aux` the aux basis's,
/// `aux.n_cart()` × `n_aux_sph`. The tasks are `three-center download`
/// (stream), `three-center tables` (CPU) and `three-center` (stream), the
/// Cartesian tensor
/// (`n_cart² aux.n_cart()` doubles) and the transform's intermediate
/// (`n_cart² n_aux_sph`) the last one's scratch, so both are in its table
/// at planning, beside the output.
///
/// A class above the kernels' table ([`FB_THREE_CENTER_MAX_MOMENTUM`](crate::FB_THREE_CENTER_MAX_MOMENTUM)
/// for the orbital shells, [`FB_THREE_CENTER_AUX_MAX_MOMENTUM`] for the
/// aux) is the launcher's refusal at run time, `Unsupported` naming the
/// class; a caller refuses one at planning. A regroup that finds another
/// count of pairs in a class than `n_pairs` gives is `InvalidConfig` at
/// run time: the records were not the basis's the counts describe.
///
/// [`add_record_tasks`]: crate::add_record_tasks
///
/// # Panics
/// If `n_pairs` is not one count per class; the classes are not over one
/// `n_cart`, or their records together pass an `i32` count (the kernels'
/// record index); `n_cart² aux.n_cart()` does not fit an `i32` (the kernels'
/// index); `u` is not `n_cart` rows of at most `n_cart` columns, or `u_aux`
/// `aux.n_cart()` rows of at most `aux.n_cart()`; a class's cell count does
/// not fit an `i32`; or a class is of another worker's or plan's table
/// ([`Dag::add`]).
#[debug_requires(!classes.is_empty(), "an orbital basis of at least one shell")]
#[debug_requires(aux.n_shells() > 0, "an aux basis of at least one shell")]
#[track_caller]
pub fn add_three_center_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[ClassRows],
    n_pairs: &[usize],
    aux: &AuxTable,
    u: Vec<f64>,
    u_aux: Vec<f64>,
) -> RowId<Device, f64> {
    let staging = Arc::new(staging(classes, n_pairs, aux, u, u_aux));
    let downloaded = dag.add(
        "three-center download",
        worker,
        Download {
            classes: classes.to_vec(),
        },
    );
    let tables = dag.add(
        "three-center tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
            downloaded,
        },
    );
    dag.add("three-center", worker, ThreeCenter { staging, tables })
        .out
}

/// The staging of [`add_three_center_tasks`], every extent it asserts
/// checked.
#[track_caller]
fn staging(
    classes: &[ClassRows],
    n_pairs: &[usize],
    aux: &AuxTable,
    u: Vec<f64>,
    u_aux: Vec<f64>,
) -> Staging {
    assert_eq!(
        n_pairs.len(),
        classes.len(),
        "add_three_center_tasks: {} pair counts for {} classes",
        n_pairs.len(),
        classes.len()
    );
    let n_cart = classes[0].n_cart();
    assert!(
        classes.iter().all(|c| c.n_cart() == n_cart),
        "add_three_center_tasks: the classes are not over one basis's Cartesian AOs"
    );
    let n_aux_cart = aux.n_cart();
    assert!(
        n_cart
            .checked_mul(n_cart)
            .and_then(|n2| n2.checked_mul(n_aux_cart))
            .is_some_and(|n| i32::try_from(n).is_ok()),
        "add_three_center_tasks: {n_cart}² x {n_aux_cart} Cartesian elements do not fit the \
         kernels' 32-bit index"
    );
    // The transform writes U's columns over the Cartesian tensor and the
    // intermediate, which hold as many only when there are no more of them
    // than Cartesian functions.
    let columns = |what: &str, u: &[f64], rows: usize| {
        let cols = u.len() / rows.max(1);
        assert!(
            rows > 0 && u.len() == rows * cols && cols <= rows,
            "add_three_center_tasks: {what} holds {} doubles, not {rows} rows of at most {rows} \
             columns",
            u.len()
        );
        cols
    };
    let n_sph = columns("U", &u, n_cart);
    let n_aux_sph = columns("U_aux", &u_aux, n_aux_cart);

    let n_bra: usize = classes.iter().map(|c| c.n_rec() as usize).sum();
    assert!(
        i32::try_from(n_bra).is_ok(),
        "add_three_center_tasks: {n_bra} records, past the kernels' i32 record index"
    );
    let mut first = 0;
    let classes: Vec<_> = classes
        .iter()
        .zip(n_pairs)
        .map(|(c, &pairs)| {
            let n_rec = c.n_rec() as usize;
            first += n_rec;
            (c.la(), c.lb(), n_rec, first - n_rec, pairs)
        })
        .collect();

    let n_l = FB_THREE_CENTER_AUX_MAX_MOMENTUM.max(aux.momentum.iter().copied().max().unwrap_or(0))
        as usize
        + 1;
    let mut aux_of_l = vec![Vec::new(); n_l];
    for (s, &l) in aux.momentum.iter().enumerate() {
        aux_of_l[l as usize].push(s);
    }
    let mut cells = Vec::new();
    let mut n_cells = 0;
    for (class, &(la, lb, _, _, pairs)) in classes.iter().enumerate() {
        for (lp, shells) in aux_of_l.iter().enumerate() {
            let n_cell = pairs * shells.len();
            if n_cell == 0 {
                continue;
            }
            assert!(
                i32::try_from(n_cell).is_ok(),
                "add_three_center_tasks: class ({la}, {lb}, {lp}) has {n_cell} cells, past an \
                 i32 count"
            );
            cells.push(CellClass {
                class,
                lp: lp as i32,
                begin: n_cells,
                n_cell,
            });
            n_cells += n_cell;
        }
    }

    let mut tail = aux.field_major();
    tail.extend(u);
    tail.extend(u_aux);
    Staging {
        n_cart,
        n_sph,
        n_aux_cart,
        n_aux_sph,
        classes,
        n_bra: first,
        tail,
        n_aux_rec: aux.n_rec(),
        aux_of_l,
        aux_rec_begin: aux.rec_begin.clone(),
        aux_offset: aux.ao_offset.clone(),
        cells,
        n_cells,
    }
}

/// One class's downloaded fields: the bra fields and the two AO offsets.
#[derive(Clone, Copy)]
struct Downloaded {
    fields: [RowId<Pinned, f64>; BRA_FIELDS],
    off: [RowId<Pinned, i32>; 2],
}

/// GPU: each class's bra fields and offsets copied into pinned rows.
struct Download {
    classes: Vec<ClassRows>,
}

impl Task for Download {
    type Rows = Vec<Downloaded>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        self.classes
            .iter()
            .map(|c| {
                let (la, lb, n) = (c.la(), c.lb(), c.n_rec() as usize);
                let f64_rows = c.f64_rows();
                let i32_rows = c.i32_rows();
                let fields = std::array::from_fn(|f| {
                    device.read(f64_rows[f]);
                    host.output(&format!("{} ({la},{lb})", BRA_NAMES[f]), n)
                });
                let off = std::array::from_fn(|f| {
                    device.read(i32_rows[f]);
                    host.output(&format!("{} ({la},{lb})", ["off_a", "off_b"][f]), n)
                });
                Downloaded { fields, off }
            })
            .collect()
    }

    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        for (c, d) in self.classes.iter().zip(rows) {
            for (&src, &dst) in c.f64_rows().iter().zip(&d.fields) {
                memcpy_d2p(host, dst, device.get(src), stream)?;
            }
            for (&src, &dst) in c.i32_rows().iter().zip(&d.off) {
                memcpy_d2p(host, dst, device.get(src), stream)?;
            }
        }
        Ok(())
    }
}

/// The records row (the bra records, the aux records, U and U_aux) and the
/// cells row, of one kind: pinned as the CPU task's outputs, device as the
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

/// CPU: the records regrouped by shell pair, and the cells formed from the
/// pairs, into pinned rows behind what planning formed.
struct Tables {
    staging: Arc<Staging>,
    downloaded: Vec<Downloaded>,
}

impl Task for Tables {
    type Rows = TableRows<Pinned>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        for d in &self.downloaded {
            for &row in &d.fields {
                host.read(row);
            }
            for &row in &d.off {
                host.read(row);
            }
        }
        let s = &self.staging;
        TableRows {
            records: host.output("three-center records", s.n_records()),
            cells: host.output("three-center cells", CELL_FIELDS * s.n_cells),
        }
    }

    fn run(
        &self,
        rows: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let s = &self.staging;
        let records = host.take_mut(rows.records);
        let cells = host.take_mut(rows.cells);
        let n_bra = s.n_bra;
        records[BRA_FIELDS * n_bra..].copy_from_slice(&s.tail);
        for (c, (d, &(la, lb, n_rec, base, n_pairs))) in
            self.downloaded.iter().zip(&s.classes).enumerate()
        {
            let off_a = host.get(d.off[0]);
            let off_b = host.get(d.off[1]);
            let mut order: Vec<usize> = (0..n_rec).collect();
            order.sort_by_key(|&r| (off_a[r], off_b[r]));
            // Each pair's run: its offsets and its record range in the
            // class's regrouped order.
            let mut pairs = Vec::with_capacity(n_pairs);
            let mut k = 0;
            while k < n_rec {
                let key = (off_a[order[k]], off_b[order[k]]);
                let end = k + order[k..]
                    .iter()
                    .take_while(|&&r| (off_a[r], off_b[r]) == key)
                    .count();
                let fits = |off: i32, l: i32| off >= 0 && off as usize + n_cart_of(l) <= s.n_cart;
                if !fits(key.0, la) || !fits(key.1, lb) {
                    return Err(Error::config(format!(
                        "three-center tables: a ({la}, {lb}) record's AOs ({}, {}) are not the \
                         orbital basis's {} Cartesian functions",
                        key.0, key.1, s.n_cart
                    )));
                }
                pairs.push((key, base + k, base + end));
                k = end;
            }
            if pairs.len() != n_pairs {
                return Err(Error::config(format!(
                    "three-center tables: class ({la}, {lb})'s records hold {} shell pairs, and \
                     the plan was made for {n_pairs}",
                    pairs.len()
                )));
            }
            for (f, &row) in d.fields.iter().enumerate() {
                let src = host.get(row);
                let dst = &mut records[f * n_bra + base..f * n_bra + base + n_rec];
                for (to, &r) in dst.iter_mut().zip(&order) {
                    *to = src[r];
                }
            }
            for cc in s.cells.iter().filter(|cc| cc.class == c) {
                let shells = &s.aux_of_l[cc.lp as usize];
                let run = &mut cells[CELL_FIELDS * cc.begin..CELL_FIELDS * (cc.begin + cc.n_cell)];
                let mut k = 0;
                for &((oa, ob), rec0, rec1) in &pairs {
                    for &sh in shells {
                        // In range: n_bra fits an i32 (staging), and so does
                        // every aux record count (AuxTable::push_shell).
                        let fields = [
                            rec0 as i32,
                            rec1 as i32,
                            s.aux_rec_begin[sh],
                            s.aux_rec_begin[sh + 1],
                            oa,
                            ob,
                            s.aux_offset[sh],
                        ];
                        for (f, v) in fields.into_iter().enumerate() {
                            run[f * cc.n_cell + k] = v;
                        }
                        k += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

/// GPU: the tables uploaded, every class launched into the Cartesian
/// tensor, the three spherical GEMMs, and the (m, n) average into (mn|P).
struct ThreeCenter {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct ThreeCenterRows {
    tables: TableRows<Device>,
    cartesian: RowId<Device, f64>,
    work: RowId<Device, f64>,
    out: RowId<Device, f64>,
}

impl Task for ThreeCenter {
    type Rows = ThreeCenterRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.tables.records);
        host.read(self.tables.cells);
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
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let s = &self.staging;
        let mut records = device.take_mut(rows.tables.records);
        memcpy_p2d(&records, host, self.tables.records, worker)?;
        let cells = device.take_mut(rows.tables.cells);
        memcpy_p2d(&cells, host, self.tables.cells, worker)?;
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
            // task, which wrote every cell's bra range inside [0, n_bra), its
            // aux range from the AuxTable's prefix, its orbital offsets each
            // checked to open a block inside [0, n_cart) and its aux offset
            // the AuxTable's, inside [0, n_aux_cart); cartesian is this
            // task's n_cart² n_aux_cart doubles; the launch only enqueues on
            // the task's stream.
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

        // U and U_aux, the records row's tail.
        let (_, mut tail) = split_at_mut(&mut records, s.u_at());
        let (u, u_aux) = split_at_mut(&mut tail, s.n_cart * s.n_sph);
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
