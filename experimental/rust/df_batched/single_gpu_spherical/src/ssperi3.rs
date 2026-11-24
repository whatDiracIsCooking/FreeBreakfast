//! The fully spherical packed build of B: rust/'s `add_packed_b`
//! (`rust/integrals/eri/density_fitting/three_center/src/build.rs`) with
//! the kernels swapped for FelizJueves's `ssperi3<la, lb, lP>` (#745) and
//! the spherical transform gone.
//!
//! `speri3` writes `(ab|Q)`, `a` and `b` Cartesian, so rust/ follows each
//! launch with `spherical transform`, two strided-batched GEMMs per class
//! against U. `ssperi3` writes `(μν|Q)`, all three indices real solid
//! harmonics, each pair's block at its spherical column of the pair map
//! (`PairMap::sphe`), which is the layout the fit's trsm reads. So a trip is
//!
//! - `packed three-center tables` (CPU) and `packed three-center
//!   spherical` (stream), this file's fork of
//!   `fb_integrals_sys::add_packed_three_center_tasks` with each cell's
//!   destination at its pair's spherical column;
//! - `density fit` (`add_batch_fit`, unchanged);
//! - `B (la, lb) batch transpose` and the pitched download, as there.
//!
//! No U for the orbital basis is built or uploaded, and no Cartesian batch
//! or transform scratch is on the card: a trip's device rows are the
//! tables and three rows of `naux` × the batch's spherical columns (the
//! integrals, the fit's B and its transpose). The batch width is counted
//! in spherical columns.
//!
//! What is forked, not reused, and why: the cells, the aux records and the
//! class view are `pub(crate)` in `fb_integrals_sys`, so they are formed
//! here the same way (`AuxTable::push_shell`'s weight, `ClassRows::view`'s
//! fields).

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

/// rust/'s `add_packed_b` over `ssperi3`: the same arguments but U's slots,
/// the same B (a heap row, `(column, Q)` at `column + n_columns · Q`), the
/// same tasks less `harmonics`, `upload U` and `spherical transform`.
///
/// # Errors
/// `Unsupported` for a shell past the kernels' momenta or a pair's block
/// past their 32-bit index; `GpuRuntime` if rust/'s kernel library (whose
/// records the launches read) fails its load check.
pub fn add_packed_b_spherical(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    aux: &BasisSet,
    map: &PairMap,
    factor: (RowId<Device, f64>, RowId<Pinned, i32>),
) -> Result<RowId<Heap, f64>> {
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    for (which, set, max) in [("orbital", basis, MAX_L), ("aux", aux, AUX_MAX_L)] {
        if let Some(shell) = set.shells().iter().find(|s| s.angular_momentum() > max) {
            return Err(Error::unsupported(format!(
                "add_packed_b_spherical: an {which} shell of l = {}, past {max}",
                shell.angular_momentum()
            )));
        }
    }
    let table = Arc::new(Aux::new(aux));
    let naux = table.n_sph;
    let tables = record_tables(basis);
    let sphe = map.sphe().blocks();
    let shapes: Vec<Shape> = map
        .classes()
        .iter()
        .map(|c| Shape {
            first: c.pairs.start,
            pairs: c.pairs.len(),
            sph: sphe[c.pairs.start].n,
            la: c.la,
            lb: c.lb,
        })
        .collect();
    let largest = shapes.iter().map(Shape::block).max().unwrap_or(1);
    let cap = i32::MAX as usize / naux;
    if largest > cap {
        return Err(Error::unsupported(format!(
            "add_packed_b_spherical: a pair's block, {largest} columns x {naux}, is past the \
             kernels' 32-bit index"
        )));
    }
    let part = Batch {
        shapes,
        naux,
        aux_rec: AUX_FIELDS * table.n_rec,
        aux_shells: aux.shells().len(),
        cap,
    };
    let n_columns = map.sphe().n_columns();
    let b = dag.add_cpu(
        "B",
        worker,
        HeapB {
            len: naux * n_columns,
        },
    );
    let width = dag.width("packed B batch", worker, &part);
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
        let fitted = add_batch_fit(dag, worker, naux, columns, batch, factor, packed);
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
        let first = sphe[shape.first].column;
        dag.download_heap_pitched(
            worker,
            transposed,
            b,
            first..first + columns,
            batch,
            naux,
            n_columns,
            &name,
        );
        dag.add_counted(body, trips, DEPTH.min(trips));
    }
    Ok(b)
}
