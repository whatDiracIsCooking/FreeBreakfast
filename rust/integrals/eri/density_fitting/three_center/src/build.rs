//! The build loop: a packed B made batch by batch of kept shell pairs, on
//! the device, into a heap row on the host ([`add_packed_b`]).

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, DEVICE_ALIGNMENT, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Adaptive, CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::geam;
use fb_integrals_eri_density_fitting_fit::add_batch_fit;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_sys::{ClassRows, FB_THREE_CENTER_AUX_FIELDS, FB_THREE_CENTER_PACKED_CELL_FIELDS};
use fb_recipes_matrix_ao::add_harmonics;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_T;

use super::{add_packed_three_center, checked_aux_table};

/// The full batches' ring: two copies of a batch's rows, so batch `k + 1`'s
/// tables and launches run under batch `k`'s download and unstage.
const DEPTH: usize = 2;

/// Add the build of the packed factor B of the orbital basis `basis` over
/// the fitting set `aux`, against the metric's factor `factor`, to `dag`
/// on GPU worker `worker`, and return B: a heap row of
/// `naux · n_columns` doubles named `"B"`, `naux` the aux basis's
/// spherical functions and `n_columns` the spherical columns of `map`,
/// element `(column, Q)` at `column + n_columns · Q`, the layout the pair
/// map's crate doc defines. No device row holds B whole: the device holds
/// one batch's rows beside the metric's factor and the records, and B is
/// on the host.
///
/// `basis`, `classes`, `aux`, `map`, `u` and `u_aux` are as for
/// [`add_packed_three_center`]; `factor` is
/// `fb_recipes_cholesky::add_cholesky`'s `(L, checked)` over the
/// `naux` × `naux` metric of `aux`, added before.
///
/// **The tasks**, in plan order: `harmonics` and `upload U` for each slot
/// still `None`; **`B`** (CPU), the heap row, which every batch writes its
/// columns of; then, per class `(la, lb)` of `map`, in the map's order,
/// its pairs in batches of one size, the last what is left, a counted
/// loop of a trip per batch and ring depth 2 (`docs/memory.md`, "Nested
/// plans"), each trip:
///
/// - [`add_packed_three_center`]'s `packed three-center tables`, `packed
///   three-center` and `spherical transform` over the trip's pairs;
/// - `density fit` ([`add_batch_fit`]), `naux` × the batch's spherical
///   columns, aux fastest;
/// - **`B (la, lb) batch transpose`** (stream), one `geam` into the
///   batch pair-fastest, `(c, Q)` at `c + columns · Q`;
/// - `Dag::download_heap_pitched`'s `download` and `unstage`, the
///   batch's `naux` runs of its columns into B at a pitch of `n_columns`.
///
/// A class whose pairs the batch does not divide ends in a short batch,
/// the loop's last trip, whose tasks take its count from their iteration
/// and work on the front of the first trip's rows, and whose download is
/// clamped to the class's columns. A batch never spans two classes: a
/// class's blocks are of one shape ([`PairMap::classes`]), so every trip
/// of a loop has the first's rows and launches, the last a prefix of
/// them, and the loops are the classes, whatever the basis's size. Each
/// batch's columns are fitted apart from every other's, so B is the
/// one-batch B column for column.
///
/// The batch's width is an adaptive part (`fb_core_planner::Adaptive`,
/// `docs/memory.md`, "Adaptive widths: one planner routine"), `packed B
/// batch`, in Cartesian columns: a class of blocks of `k` Cartesian
/// columns takes `width / k` pairs a batch, at least one and at most its
/// own. Its bytes are the largest class's ring, two copies of its
/// batch's device rows, and of its pinned rows. Under
/// [`Dag::plan_within`] the width is the widest that fits the room, and a
/// room short of one pair's batch leaves it at 1, a plan admission refuses
/// naming the loops; under [`Dag::plan`], every class in one batch.
///
/// First, before anything is added, the load check and the kernels'
/// limits of [`add_packed_three_center`].
///
/// # Errors
/// As [`add_packed_three_center`]: a `GpuRuntime` error with the load
/// check's message, or an `Unsupported` one naming the shell; or an
/// `Unsupported` one naming the count, if one pair's Cartesian block
/// times the aux basis's Cartesian functions does not fit the kernels'
/// 32-bit index. Nothing is added to `dag` on an error.
///
/// # Panics
/// As [`add_packed_three_center`]; and if `factor` is of another worker's
/// table ([`Dag::add_gpu`]).
#[debug_requires(basis.nshells() > 0, "an orbital basis of at least one shell")]
#[debug_requires(aux.nshells() > 0, "an aux basis of at least one shell")]
#[debug_requires(
    map.cart().nao() == basis.nao_cart() as usize,
    "map is a map of basis's pairs"
)]
#[debug_requires(!map.pairs().is_empty(), "map keeps a pair")]
#[allow(
    clippy::too_many_arguments,
    reason = "add_packed_three_center's, with the factor and without the batch"
)]
#[track_caller]
pub fn add_packed_b(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    aux: &BasisSet,
    map: &PairMap,
    factor: (RowId<Device, f64>, RowId<Pinned, i32>),
    u: &mut Option<RowId<Device, f64>>,
    u_aux: &mut Option<RowId<Device, f64>>,
) -> Result<RowId<Heap, f64>> {
    let table = checked_aux_table("add_packed_b", basis, aux)?;
    let naux = [aux.nao_cart() as usize, aux.nao_sphe() as usize];
    let (cart, sphe) = (map.cart().blocks(), map.sphe().blocks());
    let shapes: Vec<Shape> = map
        .classes()
        .iter()
        .map(|c| Shape {
            first: c.pairs.start,
            pairs: c.pairs.len(),
            cart: cart[c.pairs.start].n,
            sph: sphe[c.pairs.start].n,
            la: c.la,
            lb: c.lb,
        })
        .collect();
    let largest = shapes.iter().map(Shape::block).max().unwrap_or(1);
    // Every batch's Cartesian columns are at most the larger of the width
    // and one pair's block, so capping the width here bounds every count
    // add_packed_three_center refuses, and the transform's, fit's and
    // transpose's below it (add_packed_three_center's argument).
    let cap = i32::MAX as usize / naux[0];
    if largest > cap {
        return Err(Error::unsupported(format!(
            "add_packed_b: a pair's Cartesian block, {largest} columns x {} elements ({} bytes), \
             is past the kernels' 32-bit index",
            naux[0],
            8 * largest * naux[0]
        )));
    }
    let part = Batch {
        shapes,
        naux,
        aux_rec: FB_THREE_CENTER_AUX_FIELDS as usize * table.n_rec(),
        aux_shells: table.n_shells(),
        cap,
    };
    add_harmonics(dag, worker, basis, u);
    add_harmonics(dag, worker, aux, u_aux);
    let n_columns = map.sphe().n_columns();
    let b = dag.add_cpu(
        "B",
        worker,
        HeapB {
            len: naux[1] * n_columns,
        },
    );
    let width = dag.width("packed B batch", worker, &part);
    for shape in &part.shapes {
        let per = shape.per_batch(width);
        let trips = shape.pairs.div_ceil(per);
        let name = format!("B ({}, {}) batch", shape.la, shape.lb);
        let body = dag.len();
        // Never an error: the library and the momenta were checked above,
        // and the batch's counts are bounded by the cap.
        let packed = add_packed_three_center(
            dag,
            worker,
            basis,
            classes,
            aux,
            map,
            shape.first..shape.first + shape.pairs,
            per,
            u,
            u_aux,
        )?;
        let block = shape.sph[0] * shape.sph[1];
        let [columns, batch] = [shape.pairs * block, per * block];
        let fitted = add_batch_fit(dag, worker, naux[1], columns, batch, factor, packed);
        let transposed = dag.add_gpu(
            format!("{name} transpose"),
            worker,
            Transpose {
                b: fitted,
                naux: naux[1],
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
            naux[1],
            n_columns,
            &name,
        );
        dag.add_counted(body, trips, DEPTH.min(trips));
    }
    Ok(b)
}

/// One class of the map's kept pairs: its first pair and count, its
/// blocks' Cartesian and spherical shapes, and its momenta.
#[derive(Clone, Copy, Debug)]
struct Shape {
    first: usize,
    pairs: usize,
    cart: [usize; 2],
    sph: [usize; 2],
    la: i32,
    lb: i32,
}

impl Shape {
    /// A block's Cartesian columns.
    fn block(&self) -> usize {
        self.cart[0] * self.cart[1]
    }

    /// The pairs a batch of `width` Cartesian columns takes: at least one,
    /// at most the class's.
    fn per_batch(&self, width: usize) -> usize {
        (width / self.block()).clamp(1, self.pairs)
    }
}

/// The batch's adaptive part: the largest class's ring at a width.
struct Batch {
    shapes: Vec<Shape>,
    /// The aux basis's Cartesian and spherical functions.
    naux: [usize; 2],
    /// The aux records row's doubles.
    aux_rec: usize,
    /// The aux shells: a pair's cells.
    aux_shells: usize,
    /// The widest batch whose counts fit the kernels' 32-bit index.
    cap: usize,
}

impl Batch {
    /// Class `c`'s batch of `p` pairs, in bytes each padded to the device
    /// alignment: its device rows (the tables' copies and the packed
    /// tensor, the transform's two scratch rows and output, the fit's B and
    /// its transpose) and its pinned rows (the tables and the window).
    fn rows(&self, c: usize, p: usize) -> (usize, usize) {
        let s = &self.shapes[c];
        let [nc, ns] = self.naux;
        let (cart, sph) = (p * s.block(), p * s.sph[0] * s.sph[1]);
        let tables = pad(8 * self.aux_rec)
            + pad(4 * FB_THREE_CENTER_PACKED_CELL_FIELDS as usize * p * self.aux_shells);
        let device = tables
            + pad(8 * nc * cart)
            + pad(8 * ns * cart)
            + pad(8 * ns * p * s.cart[0] * s.sph[1])
            + 3 * pad(8 * ns * sph);
        (device, tables + pad(8 * ns * sph))
    }

    /// Class `c`'s ring at `width`, device bytes: two copies of its
    /// batch's rows, the body padded again, the counted loop's region
    /// (`docs/memory.md`, "Nested plans").
    fn ring(&self, c: usize, width: usize) -> usize {
        let p = self.shapes[c].per_batch(width);
        DEPTH * self.rows(c, p).0.next_multiple_of(DEVICE_ALIGNMENT)
    }
}

impl Adaptive for Batch {
    /// Every class in one batch, or the widest the kernels' index takes.
    fn width_max(&self) -> usize {
        let whole = self.shapes.iter().map(|s| s.pairs * s.block()).max();
        whole.unwrap_or(1).min(self.cap).max(1)
    }

    /// The largest class's ring: every class's loop runs in the same
    /// bytes in turn, and a ring of depth 2 is counted for each, the most
    /// a loop of any trip count holds, so the bytes stay monotone.
    fn bytes(&self, width: usize) -> usize {
        (0..self.shapes.len())
            .map(|c| self.ring(c, width))
            .max()
            .unwrap_or(0)
    }

    /// The largest class's pinned ring: two copies of its tables and its
    /// window.
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

/// `bytes` padded to the device alignment, which the pinned rows share.
fn pad(bytes: usize) -> usize {
    bytes.next_multiple_of(DEVICE_ALIGNMENT)
}

/// CPU: B's heap row, made for the batches to write their columns of;
/// every column is a kept pair's, so every element is written.
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

/// GPU: a batch of B, `naux` × `batch` aux fastest, transposed to
/// pair-fastest, `batch` × `naux`; batch `k` of the class's `columns`, the
/// last what is left, the front of each row, at the first batch's leading
/// dimension, the window's pitch.
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
        // This trip's columns: the batch's, or what is left for the last.
        let count = self
            .batch
            .min(self.columns - device.iteration() * self.batch);
        let b = device.get(self.b);
        let mut out = device.take_mut(*out);
        // In range: add_packed_b's cap bounds batch · naux_cart by an i32,
        // naux is at most naux_cart, and count at most batch. beta is zero,
        // so B's second operand only gives the call a pointer: C = Bᵀ
        // exactly, the first count rows of each of naux columns at the
        // batch's leading dimension.
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

#[cfg(test)]
mod tests {
    use fb_core_planner::Room;

    use super::*;

    /// Classes of every momentum pair through f, each of `pairs` pairs.
    fn batch(pairs: usize, naux: [usize; 2]) -> Batch {
        let n = |l: usize| [(l + 1) * (l + 2) / 2, 2 * l + 1];
        let mut shapes = Vec::new();
        for la in 0..=3 {
            for lb in la..=3 {
                let (a, b) = (n(la), n(lb));
                shapes.push(Shape {
                    first: shapes.len() * pairs,
                    pairs: pairs + la + 2 * lb,
                    cart: [a[0], b[0]],
                    sph: [a[1], b[1]],
                    la: la as i32,
                    lb: lb as i32,
                });
            }
        }
        Batch {
            shapes,
            naux,
            aux_rec: 9 * 40,
            aux_shells: 30,
            cap: i32::MAX as usize / naux[0],
        }
    }

    /// The part's bytes of both kinds are monotone in the width, so the
    /// planner's bisection is a linear scan's answer, over every width of
    /// a few shapes; and the width past which nothing grows is the widest
    /// class whole.
    #[test]
    fn the_bytes_are_monotone() {
        for (pairs, naux) in [(1, [10, 9]), (5, [116, 98]), (17, [300, 250])] {
            let part = batch(pairs, naux);
            let max = part.width_max();
            for w in 1..max {
                assert!(part.bytes(w) <= part.bytes(w + 1), "device at {w}");
                assert!(
                    part.pinned_bytes(w) <= part.pinned_bytes(w + 1),
                    "pinned at {w}"
                );
            }
            assert_eq!(
                part.bytes(max),
                part.bytes(10 * max),
                "nothing past the max"
            );
            for room in (0..40).map(|k| part.bytes(max) * k / 37) {
                let room = Room {
                    device: room,
                    pinned: usize::MAX,
                };
                let scan = (1..=max)
                    .rev()
                    .find(|&w| part.bytes(w) <= room.device)
                    .unwrap_or(1);
                assert_eq!(fb_core_planner::widest(&part, room), scan, "{room:?}");
            }
        }
    }
}
