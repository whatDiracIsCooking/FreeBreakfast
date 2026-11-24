//! One card's share of the fully spherical packed build of B:
//! `../single_gpu_spherical/src/ssperi3.rs`'s `add_packed_b_spherical`,
//! forked to build a part of the pairs into a heap row of its own
//! ([`add_packed_b_parts`]).
//!
//! `B[Q][μν] = Σ_P L⁻¹[Q][P] (P|μν)` is independent per column once L
//! exists, so the cards split the columns and nothing is reduced. Card `w`
//! of `n` takes the `w`-th contiguous share of every class's kept pairs
//! ([`share`]), by pair count: the fit's trsm is most of a trip's device
//! time and costs the same per column, and a class's pairs all have the
//! same block. Its trips are that experiment's, unchanged: `packed
//! three-center tables` and `packed three-center spherical` over
//! FelizJueves's `ssperi3<la, lb, lP>`, `density fit`, the transpose and
//! the pitched download.
//!
//! What differs:
//!
//! - **A heap row per card.** The planner orders every writer of a row
//!   after the one before it, and the arena lets a row have one writer at
//!   a time, so two cards downloading into one B would run one after the
//!   other. Card `w`'s row holds its own columns alone, `naux` rows of
//!   [`Part::n_columns`], its classes side by side in the map's order;
//!   [`Part::spans`] says where each run of them sits in the map's
//!   columns. Every card's row is made before any card's loop: one made
//!   later would take a finished part's bytes, and wait for it.
//! - **The width is the card's.** Each card's `packed B batch` is an
//!   adaptive part on its own worker, sized by its own room over its own
//!   share.
//! - **L is the card's too.** A row crosses between workers by a
//!   collective alone, so the caller builds the metric and its factor on
//!   every card (`main.rs`), and hands each worker's in.
//!
//! The kernel library is that experiment's `libfbx_ssperi3_kernels.so`,
//! linked as it is (`build.rs`).

use std::ffi::c_void;
use std::sync::Arc;

use fb_basis::BasisSet;
use fb_core_arena::{
    Bufs, CopyOnly, DEVICE_ALIGNMENT, Device, Heap, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d,
};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Adaptive, CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::geam;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_eri_density_fitting_fit::add_batch_fit;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::{ClassRows, Fields, RecordTables, fb_primitive_pair_class_view};
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_T;

#[link(name = "fbx_ssperi3_kernels")]
unsafe extern "C" {
    /// `kernels/ssperi3_launch.cu`: class `(cls.la, cls.lb, lp)` of the
    /// fully spherical packed `(μν|Q)` into `eri3`, enqueued on `stream`;
    /// 0, a CUDA error code, or -1 (no kernel) / -2 (a bad argument).
    fn fbx_ssperi3_packed_class(
        stream: *mut c_void,
        cls: *const fb_primitive_pair_class_view,
        lp: i32,
        n_aux_rec: i32,
        aux: *const f64,
        n_cell: i32,
        cells: *const i32,
        n_col: i32,
        n_aux: i32,
        eri3: *mut f64,
    ) -> i32;
}

/// The kernels' momenta: orbital through f, aux through i.
const MAX_L: i32 = 3;
const AUX_MAX_L: i32 = 6;
/// An aux record's fields: x, y, z, α, `c (π/α)^{3/2}`.
const AUX_FIELDS: usize = 5;
/// A cell's fields: bra record range, aux record range, destination, diag.
const CELL_FIELDS: usize = 6;
/// The ring's depth, as rust/'s.
const DEPTH: usize = 2;

/// The aux basis as the kernels read it.
#[derive(Debug)]
struct Aux {
    /// The records, field-major: field `f` of record `r` at `f n_rec + r`.
    records: Vec<f64>,
    n_rec: usize,
    /// Each shell's record range (`n_shells + 1` long).
    rec_begin: Vec<i32>,
    /// Each shell's first spherical function.
    sph_offset: Vec<usize>,
    /// The shells of each momentum, in order.
    of_l: Vec<Vec<usize>>,
    /// The spherical functions.
    n_sph: usize,
}

impl Aux {
    fn new(aux: &BasisSet) -> Aux {
        let mut rows: Vec<[f64; AUX_FIELDS]> = Vec::new();
        let mut rec_begin = vec![0];
        let mut sph_offset = Vec::new();
        let mut of_l = vec![Vec::new(); AUX_MAX_L as usize + 1];
        let mut n_sph = 0;
        for (s, shell) in aux.shells().iter().enumerate() {
            let c = shell.center();
            let l = shell.angular_momentum();
            rows.extend(shell.primitives().iter().map(|p| {
                let (alpha, coef) = (p.exponent, p.coefficient);
                [
                    c.x,
                    c.y,
                    c.z,
                    alpha,
                    coef * (std::f64::consts::PI / alpha).powf(1.5),
                ]
            }));
            rec_begin.push(i32::try_from(rows.len()).expect("aux records fit an i32"));
            sph_offset.push(n_sph);
            of_l[l as usize].push(s);
            n_sph += 2 * l as usize + 1;
        }
        let n_rec = rows.len();
        let mut records = vec![0.0; AUX_FIELDS * n_rec];
        for (r, row) in rows.iter().enumerate() {
            for (f, &v) in row.iter().enumerate() {
                records[f * n_rec + r] = v;
            }
        }
        Aux {
            records,
            n_rec,
            rec_begin,
            sph_offset,
            of_l,
            n_sph,
        }
    }
}

/// One aux momentum's cells in a trip: its `lp`, and its first cell in the
/// cells row (the first trip's count, which a short trip's prefix keeps).
#[derive(Debug, Clone, Copy)]
struct CellGroup {
    lp: i32,
    begin: usize,
}

/// One class's staging: every trip of its loop is a prefix of the first.
#[derive(Debug)]
struct Staging {
    aux: Arc<Aux>,
    /// The class's records, which the launches read in place.
    class: ClassRows,
    /// The class's kept pairs, in order: record range and diagonal flag.
    pair: Vec<([i32; 2], bool)>,
    /// Pairs per trip, the last trip what is left.
    per_trip: usize,
    /// A pair's spherical block, columns.
    block: usize,
    /// One group per aux momentum present.
    groups: Vec<CellGroup>,
    /// The first trip's cells.
    n_cells: usize,
}

impl Staging {
    fn trip_pairs(&self, trip: usize) -> usize {
        self.per_trip.min(self.pair.len() - trip * self.per_trip)
    }

    /// A trip's spherical columns, `n_col` of the first trip.
    fn n_col(&self) -> usize {
        self.per_trip * self.block
    }

    /// Trip `trip`'s cells of each group, field-major per group, pair outer
    /// and aux shell inner, each destination its pair's spherical column in
    /// the trip times `n_aux` plus the aux shell's first function.
    fn write_cells(&self, trip: usize, out: &mut [i32]) {
        let (base, pairs) = (trip * self.per_trip, self.trip_pairs(trip));
        let n_aux = self.aux.n_sph;
        for g in &self.groups {
            let shells = &self.aux.of_l[g.lp as usize];
            let n_cell = pairs * shells.len();
            let run = &mut out[CELL_FIELDS * g.begin..CELL_FIELDS * (g.begin + n_cell)];
            let mut k = 0;
            for place in 0..pairs {
                let ([rec0, rec1], diagonal) = self.pair[base + place];
                for &sh in shells {
                    let fields = [
                        rec0,
                        rec1,
                        self.aux.rec_begin[sh],
                        self.aux.rec_begin[sh + 1],
                        (place * self.block * n_aux + self.aux.sph_offset[sh]) as i32,
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

/// The staging of map class `pairs` (a range of `map.pairs()`, one class
/// `(la, lb)`), in trips of `per_trip` pairs.
fn staging(
    classes: &[ClassRows],
    tables: &RecordTables,
    aux: &Arc<Aux>,
    map: &PairMap,
    pairs: std::ops::Range<usize>,
    per_trip: usize,
) -> Staging {
    let ends: Vec<usize> = tables
        .classes()
        .iter()
        .scan(0, |end, c| {
            *end += c.bra.len();
            Some(*end)
        })
        .collect();
    let mut class = None;
    let mut pair = Vec::with_capacity(pairs.len());
    for k in &map.pairs()[pairs.clone()] {
        let p = k.pair;
        let ci = ends.partition_point(|&end| end <= p);
        assert!(
            class.is_none_or(|c| c == ci),
            "a map class spans one record class"
        );
        class = Some(ci);
        let c = &tables.classes()[ci];
        let q = p - (ends[ci] - c.bra.len());
        pair.push(([c.rec_begin[q], c.rec_begin[q + 1]], c.bra[q] == c.ket[q]));
    }
    let ci = class.expect("a class of at least one pair");
    let sphe = map.sphe().blocks()[pairs.start].n;
    let per_trip = per_trip.min(pair.len());
    let mut groups = Vec::new();
    let mut n_cells = 0;
    for (lp, shells) in aux.of_l.iter().enumerate() {
        if shells.is_empty() {
            continue;
        }
        groups.push(CellGroup {
            lp: lp as i32,
            begin: n_cells,
        });
        n_cells += per_trip * shells.len();
    }
    let s = Staging {
        aux: Arc::clone(aux),
        class: classes[ci],
        pair,
        per_trip,
        block: sphe[0] * sphe[1],
        groups,
        n_cells,
    };
    assert!(
        s.n_col()
            .checked_mul(s.aux.n_sph)
            .is_some_and(|n| i32::try_from(n).is_ok()),
        "a trip's spherical columns times naux fit the kernels' 32-bit index"
    );
    s
}

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

/// CPU: the aux records and the trip's cells into pinned rows.
struct Tables {
    staging: Arc<Staging>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let s = &self.staging;
        TableRows {
            aux: host.output("packed three-center aux", s.aux.records.len()),
            cells: host.output("packed three-center cells", CELL_FIELDS * s.n_cells),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        memcpy_h2p(host, rows.aux, &s.aux.records);
        s.write_cells(host.iteration(), host.take_mut(rows.cells));
        Ok(())
    }
}

/// GPU: the tables uploaded, then one `ssperi3` launch per aux momentum.
struct PackedSpherical {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct PackedRows {
    tables: TableRows<Device>,
    out: RowId<Device, f64>,
}

impl GpuTask for PackedSpherical {
    type Rows = PackedRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.tables.aux);
        host.read(self.tables.cells);
        let s = &self.staging;
        s.class.read(device, Fields::THREE_CENTER);
        PackedRows {
            tables: TableRows {
                aux: device.scratch("packed three-center aux", s.aux.records.len()),
                cells: device.scratch("packed three-center cells", CELL_FIELDS * s.n_cells),
            },
            out: device.output("(mn|P) spherical packed", s.n_col() * s.aux.n_sph),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        let s = &self.staging;
        vec![("n_col", s.n_col()), ("n_aux", s.aux.n_sph)]
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

        // The class view: the bra fields ssperi3 reads (ClassRows::view's
        // first eleven f64 rows), every other pointer null.
        let f64_rows = s.class.f64_rows();
        let f = |bit: usize| device.get(f64_rows[bit]).as_ptr();
        let null = std::ptr::null();
        let view = fb_primitive_pair_class_view {
            la: s.class.la(),
            lb: s.class.lb(),
            n_rec: s.class.n_rec(),
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
            ea: null,
            eb: null,
            bound: null,
            bound_ab: null,
            bound_ba: null,
            off_a: std::ptr::null(),
            off_b: std::ptr::null(),
            rank_a: std::ptr::null(),
            rank_b: std::ptr::null(),
        };
        let pairs = s.trip_pairs(device.iteration());
        let (n_col, n_aux) = (s.n_col() as i32, s.aux.n_sph as i32);
        for g in &s.groups {
            let n_cell = pairs * s.aux.of_l[g.lp as usize].len();
            // SAFETY: as rust/'s add_packed_three_center_tasks: the view's
            // rows are the class's n_rec-long records rows this task
            // declared; aux and cells were copied above on this stream from
            // the tables task's rows; the group's n_cell cells lie inside the
            // cells row (at most the first trip's); every destination is a
            // place's spherical column, below this trip's pairs, times n_aux
            // plus an aux shell's first function, so its block lies inside
            // out's n_col n_aux doubles. The launch only enqueues.
            let status = unsafe {
                fbx_ssperi3_packed_class(
                    worker.stream(),
                    &view,
                    g.lp,
                    s.aux.n_rec as i32,
                    aux.as_ptr(),
                    n_cell as i32,
                    cells.as_ptr().wrapping_add(CELL_FIELDS * g.begin),
                    n_col,
                    n_aux,
                    out.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(Error::gpu(format!(
                    "fbx_ssperi3_packed_class ({}, {}, {}): status {status}",
                    view.la, view.lb, g.lp
                )));
            }
        }
        Ok(())
    }
}

/// One class of the map: its first pair and count, its spherical block,
/// its momenta.
#[derive(Clone, Copy, Debug)]
struct Shape {
    first: usize,
    pairs: usize,
    sph: [usize; 2],
    la: i32,
    lb: i32,
}

impl Shape {
    fn block(&self) -> usize {
        self.sph[0] * self.sph[1]
    }

    fn per_batch(&self, width: usize) -> usize {
        (width / self.block()).clamp(1, self.pairs)
    }
}

/// The adaptive part `packed B batch`, in spherical columns.
struct Batch {
    shapes: Vec<Shape>,
    naux: usize,
    aux_rec: usize,
    aux_shells: usize,
    cap: usize,
}

impl Batch {
    /// A batch of `p` pairs of class `c`: device bytes (the tables, the
    /// integrals, the fit's B, its transpose) and pinned bytes (the tables
    /// and the download window).
    fn rows(&self, c: usize, p: usize) -> (usize, usize) {
        let s = &self.shapes[c];
        let sph = p * s.block();
        let tables = pad(8 * self.aux_rec) + pad(4 * CELL_FIELDS * p * self.aux_shells);
        let device = tables + 3 * pad(8 * self.naux * sph);
        (device, tables + pad(8 * self.naux * sph))
    }

    fn ring(&self, c: usize, width: usize) -> usize {
        let p = self.shapes[c].per_batch(width);
        DEPTH * self.rows(c, p).0.next_multiple_of(DEVICE_ALIGNMENT)
    }
}

impl Adaptive for Batch {
    fn width_max(&self) -> usize {
        let whole = self.shapes.iter().map(|s| s.pairs * s.block()).max();
        whole.unwrap_or(1).min(self.cap).max(1)
    }

    fn bytes(&self, width: usize) -> usize {
        (0..self.shapes.len())
            .map(|c| self.ring(c, width))
            .max()
            .unwrap_or(0)
    }

    fn pinned_bytes(&self, width: usize) -> usize {
        (0..self.shapes.len())
            .map(|c| {
                let p = self.shapes[c].per_batch(width);
                DEPTH * self.rows(c, p).1.next_multiple_of(DEVICE_ALIGNMENT)
            })
            .max()
            .unwrap_or(0)
    }
}

fn pad(bytes: usize) -> usize {
    bytes.next_multiple_of(DEVICE_ALIGNMENT)
}

/// CPU: B's heap row.
struct HeapB {
    len: usize,
}

impl CpuTask for HeapB {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("B", self.len)
    }

    fn run(&self, _: &Self::Rows, _: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        Ok(())
    }
}

/// GPU: a batch of B, aux fastest, transposed to pair-fastest (rust/'s).
struct Transpose {
    b: RowId<Device, f64>,
    naux: usize,
    columns: usize,
    batch: usize,
}

impl GpuTask for Transpose {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.b);
        device.output("B batch pair-fastest", self.naux * self.batch)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("columns", self.batch), ("naux", self.naux)]
    }

    fn run(
        &self,
        out: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let count = self
            .batch
            .min(self.columns - device.iteration() * self.batch);
        let b = device.get(self.b);
        let mut out = device.take_mut(*out);
        let (n, count, ld) = (self.naux as i32, count as i32, self.batch as i32);
        geam(
            worker.blas(),
            WWR_BLAS_OP_T,
            WWR_BLAS_OP_T,
            count,
            n,
            1.0,
            &b,
            n,
            0.0,
            &b,
            n,
            &mut out,
            ld,
        )
    }
}

/// Card `w` of `n`'s share of a class of `pairs` kept pairs: the `w`-th of
/// `n` contiguous ranges, the first `pairs % n` one longer.
pub fn share(pairs: usize, w: usize, n: usize) -> std::ops::Range<usize> {
    let at = |k: usize| k * (pairs / n) + k.min(pairs % n);
    at(w)..at(w + 1)
}

/// One card's prefix: its orbital records and its L (`add_cholesky`'s).
pub type Prefix = (Vec<ClassRows>, (RowId<Device, f64>, RowId<Pinned, i32>));

/// One card's part of B.
#[derive(Debug)]
pub struct Part {
    /// Its heap row: `naux` rows of `n_columns`, element `(column, Q)` at
    /// `column + n_columns · Q`.
    pub b: RowId<Heap, f64>,
    /// Its columns.
    pub n_columns: usize,
    /// Each class's run of them: the first column in the map's spherical
    /// layout, the first in this part's row, and the length.
    pub spans: Vec<(usize, usize, usize)>,
}

/// `../single_gpu_spherical`'s `add_packed_b_spherical` over the GPU
/// workers `0..prefix.len()`: on worker `w` the same tasks, over its
/// [`share`] of every class's pairs, into a heap row of its own
/// (`"B part <w>"`). `prefix[w]` is worker `w`'s own records and L.
///
/// Every part's heap row is made before any loop. A row made after
/// another's last write would be laid out in its bytes, and so ordered
/// after that write: card 1's whole build after card 0's.
///
/// # Errors
/// `Unsupported` for a shell past the kernels' momenta or a pair's block
/// past their 32-bit index; `GpuRuntime` if rust/'s kernel library (whose
/// records the launches read) fails its load check.
pub fn add_packed_b_parts(
    dag: &mut Dag<'_>,
    basis: &BasisSet,
    aux: &BasisSet,
    map: &PairMap,
    prefix: &[Prefix],
) -> Result<Vec<Part>> {
    let n = prefix.len();
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    for (which, set, max) in [("orbital", basis, MAX_L), ("aux", aux, AUX_MAX_L)] {
        if let Some(shell) = set.shells().iter().find(|s| s.angular_momentum() > max) {
            return Err(Error::unsupported(format!(
                "add_packed_b_parts: an {which} shell of l = {}, past {max}",
                shell.angular_momentum()
            )));
        }
    }
    let table = Arc::new(Aux::new(aux));
    let naux = table.n_sph;
    let tables = record_tables(basis);
    let sphe = map.sphe().blocks();
    let cap = i32::MAX as usize / naux;
    // Each worker's share of each class, the classes it has no pair of
    // left out (a class of fewer pairs than cards), and its heap row.
    let mut parts = Vec::with_capacity(n);
    for worker in 0..n {
        let shapes: Vec<Shape> = map
            .classes()
            .iter()
            .filter_map(|c| {
                let mine = share(c.pairs.len(), worker, n);
                (!mine.is_empty()).then(|| Shape {
                    first: c.pairs.start + mine.start,
                    pairs: mine.len(),
                    sph: sphe[c.pairs.start].n,
                    la: c.la,
                    lb: c.lb,
                })
            })
            .collect();
        let largest = shapes.iter().map(Shape::block).max().unwrap_or(1);
        if largest > cap {
            return Err(Error::unsupported(format!(
                "add_packed_b_parts: a pair's block, {largest} columns x {naux}, is past the \
                 kernels' 32-bit index"
            )));
        }
        let batch = Batch {
            shapes,
            naux,
            aux_rec: AUX_FIELDS * table.n_rec,
            aux_shells: aux.shells().len(),
            cap,
        };
        let n_columns: usize = batch.shapes.iter().map(|s| s.pairs * s.block()).sum();
        let b = dag.add_cpu(
            format!("B part {worker}"),
            worker,
            HeapB {
                len: naux * n_columns,
            },
        );
        parts.push((batch, b, n_columns));
    }
    let mut out = Vec::with_capacity(n);
    for (worker, (part, b, n_columns)) in parts.into_iter().enumerate() {
        let (classes, factor) = &prefix[worker];
        let mut spans = Vec::with_capacity(part.shapes.len());
        if part.shapes.is_empty() {
            out.push(Part {
                b,
                n_columns,
                spans,
            });
            continue;
        }
        let width = dag.width("packed B batch", worker, &part);
        let mut local = 0;
        for shape in &part.shapes {
            let per = shape.per_batch(width);
            let trips = shape.pairs.div_ceil(per);
            let name = format!("B ({}, {}) batch", shape.la, shape.lb);
            let body = dag.len();
            let staging = Arc::new(staging(
                classes,
                &tables,
                &table,
                map,
                shape.first..shape.first + shape.pairs,
                per,
            ));
            let table_rows = dag.add_cpu(
                "packed three-center tables",
                worker,
                Tables {
                    staging: Arc::clone(&staging),
                },
            );
            let packed = dag
                .add_gpu(
                    "packed three-center spherical",
                    worker,
                    PackedSpherical {
                        staging,
                        tables: table_rows,
                    },
                )
                .out;
            let [columns, batch] = [shape.pairs * shape.block(), per * shape.block()];
            let fitted = add_batch_fit(dag, worker, naux, columns, batch, *factor, packed);
            let transposed = dag.add_gpu(
                format!("{name} transpose"),
                worker,
                Transpose {
                    b: fitted,
                    naux,
                    columns,
                    batch,
                },
            );
            dag.download_heap_pitched(
                worker,
                transposed,
                b,
                local..local + columns,
                batch,
                naux,
                n_columns,
                &name,
            );
            dag.add_counted(body, trips, DEPTH.min(trips));
            spans.push((sphe[shape.first].column, local, columns));
            local += columns;
        }
        out.push(Part {
            b,
            n_columns,
            spans,
        });
    }
    Ok(out)
}
