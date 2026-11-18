//! The streamed fold: every J and K of one pass over a packed B that lives
//! in a heap row, read in aux batches ([`add_streamed_jk`]).

use contracts::debug_requires;
use fb_core_arena::{Bufs, CopyOnly, DEVICE_ALIGNMENT, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_log::fits_i32;
use fb_core_planner::{Adaptive, Dag, GpuTask};
use fb_core_thread_pool::{DeviceStream, LaneId};
use fb_core_wwr::blas::{gemm_strided_batched, gemv, syrk};
use fb_core_wwr::copy::d2d;
use fb_core_wwr::device_slice::{split_at, split_at_mut};
use fb_core_wwr::memset;
use fb_integrals_eri_density_fitting_pair_map::Packing;
use fb_integrals_sys::PackedTables;
use fb_recipes_symmetrize::symmetrize_from;
use wwr_sys::{wwr_blas_fill_mode, wwr_blas_operation};

/// The widest aux batch [`add_streamed_jk`] streams B in, its adaptive
/// part's [`width_max`](Adaptive::width_max): 64, CUDA's preferred batch
/// (`experimental/gemm_host_offload/README.md`: every power of two from 32
/// up costs at most 1% of the pipeline's time on the RTX 3080, a batch
/// below 32 grows like `1/b`). HIP's waits for the gfx1200 sweep and is 64
/// until then, so there is one value for both backends.
pub const BATCH_MAX: usize = 64;

/// The batch below which a narrowed plan is logged at `warn`
/// ([`Adaptive::warn_below`]): 32, where the measured cost starts to grow.
const BATCH_WARN_BELOW: usize = 32;

/// The batches' ring: two copies of a batch's rows, so batch `k + 1`'s
/// upload runs under batch `k`'s contractions.
const DEPTH: usize = 2;

/// One contraction of a streamed pass ([`add_streamed_jk`]), over the
/// orbital basis of the pass's packing, `nao` functions.
#[derive(Clone, Copy, Debug)]
pub enum Contraction {
    /// J of a symmetric density, a device row of `nao²` doubles,
    /// column-major: `J_mn = Σ_ls (mn|ls) D_ls`.
    Coulomb(RowId<Device, f64>),
    /// K of the density `D = 2 C Cᵀ`, from the factor C, a device row of
    /// `nao · nocc` doubles, column-major, `nao × nocc`:
    /// `K_mn = Σ_ls (ml|ns) D_ls`. K depends on C only through `C Cᵀ`, so
    /// any factor of a positive semidefinite density serves, a spin
    /// density `C_σ C_σᵀ` as `C_σ / √2`.
    Exchange {
        /// The factor C.
        c: RowId<Device, f64>,
        /// Its columns.
        nocc: usize,
    },
}

/// Add one pass over the packed factor `b` that serves every contraction
/// of `set` to `dag` on GPU worker `worker`, and return each one's result,
/// in `set`'s order: an output row of `nao²` doubles, column-major, named
/// `"J <i>"` or `"K <i>"` for the contraction at index `i`, `(m, n)` and
/// `(n, m)` of each the same bits. B crosses the bus once per pass,
/// however many contractions the set holds (RHF's one J and one K,
/// ROHF's J of the total density and a K per spin).
///
/// `b` is a heap row of `naux · n_columns` doubles in the layout the
/// pair-map crate defines (element `(column, Q)` at
/// `column + n_columns · Q`), `packing` the map's packing B was built over
/// (`nao` functions, `n_columns` columns) and `tables` its index tables on
/// the worker, made from the same packing by
/// [`fb_integrals_sys::add_packed_tables`] once, outside any loop the pass
/// is in. Every input is left as it was. With
/// `(mn|ls) ≈ Σ_Q B_Q,mn B_Q,ls`, a dropped shell pair's columns of B are
/// zero, so J and K are the resident [`add_df_coulomb`] and
/// [`add_df_exchange`] over B square with those columns zeroed.
///
/// [`add_df_coulomb`]: crate::add_df_coulomb
/// [`add_df_exchange`]: crate::add_df_exchange
///
/// **The tasks**, in plan order:
///
/// - **`df jk pack`** (stream, lane 0): each J's density packed,
///   `w_c D_mn` per column ([`PackedTables::pack`]), and every accumulator
///   zeroed: a packed J of `n_columns` doubles per J (`"J packed <i>"`),
///   K's lower triangle of `nao²` per K (`"K lower <i>"`).
/// - **The batches**, `naux.div_ceil(width)` of `width` aux functions, the
///   last the `naux - k width` left when `width` does not divide `naux`,
///   a counted loop of ring depth 2, or 1 when there is one batch
///   (`docs/memory.md`, "Nested plans"), each trip `stage B batch` (CPU) copying its range of the heap row
///   into its pinned window, the last range clamped to the row's end
///   ([`Dag::upload_heap`]), `upload B batch` (lane 1) copying the window
///   up, and **`df jk batch`** (stream, lane 0) contracting the trip's
///   functions, from its iteration: for each J,
///   `γ = B_bᵀ dp` and `J_p += B_b γ` by two gemvs over the packed batch
///   (`γ` the scratch `"gamma"`); for any K, the batch gathered once into
///   `width` square matrices ([`PackedTables::gather`], the scratch
///   `"slab"`), then for each K `X_q = B_q C` by one strided-batched gemm
///   (the scratch `"X"`) and `K += 2 X Xᵀ` by one syrk into the lower
///   triangle, X the `nao × (width · nocc)` matrix the `X_q` make side by
///   side.
/// - **`df jk finish`** (stream, lane 0): each packed J unpacked
///   ([`PackedTables::gather`] of one), each K's lower triangle copied out
///   and mirrored onto the upper ([`symmetrize_from`]).
///
/// The batch's width is an adaptive part (`fb_core_planner::Adaptive`,
/// `docs/memory.md`, "Adaptive widths: one planner routine"), `df jk
/// batch`: at most [`BATCH_MAX`], rounded down to a power of two, logged
/// at `warn` below 32. Its bytes are the batches' ring, its two copies of
/// a batch's device rows and of its pinned window; the finish's outputs
/// follow the ring, and admission counts them. Under [`Dag::plan_within`] the
/// width is the widest that fits the room; under [`Dag::plan`],
/// [`BATCH_MAX`]. No device row holds B whole.
///
/// The pass may lie in a decision loop's body, the SCF's, and is then
/// replayed whole every iteration; nothing it holds is carried.
///
/// First, before anything is added, the load checks of the two kernel
/// libraries its launches are in ([`fb_integrals_sys::check_library`],
/// [`fb_recipes_sys::check_library`]).
///
/// # Errors
/// A `GpuRuntime` error with a load check's message, if a loaded kernel
/// library is not the one its bindings describe.
///
/// # Panics
/// If an exchange's `naux · nocc` does not fit an `i32`, the BLAS's
/// integer, naming it; or a row is of another worker's table
/// ([`Dag::add_gpu`]); and when the plan runs, if `b` ends before the
/// last batch's range starts ([`Dag::upload_heap`]).
#[debug_requires(naux > 0, "a fitting set of at least one function")]
#[debug_requires(!set.is_empty(), "a pass has a contraction")]
#[debug_requires(
    set.iter().all(|c| !matches!(c, Contraction::Exchange { nocc: 0, .. })),
    "every factor has a column"
)]
#[requires(fits_i32!(packing.nao()), "nao fits the BLAS's i32")]
#[requires(
    fits_i32!(packing.n_columns()),
    "n_columns fits the BLAS's i32"
)]
#[requires(fits_i32!(naux), "naux fits the BLAS's i32")]
#[track_caller]
pub fn add_streamed_jk(
    dag: &mut Dag<'_>,
    worker: usize,
    packing: &Packing,
    tables: PackedTables,
    naux: usize,
    b: RowId<Heap, f64>,
    set: &[Contraction],
) -> Result<Vec<RowId<Device, f64>>> {
    let max_nocc = set
        .iter()
        .map(|c| match *c {
            Contraction::Coulomb(_) => 0,
            Contraction::Exchange { nocc, .. } => nocc,
        })
        .max()
        .unwrap_or(0);
    assert!(
        naux.checked_mul(max_nocc)
            .is_some_and(|n| i32::try_from(n).is_ok()),
        "add_streamed_jk: naux · nocc (naux = {naux}, nocc = {max_nocc}) does not fit the \
         BLAS's i32"
    );
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    let shape = Shape {
        nao: packing.nao(),
        n_columns: packing.n_columns(),
        coulomb: set.iter().any(|c| matches!(c, Contraction::Coulomb(_))),
        max_nocc,
    };

    let acc = dag.add_gpu(
        "df jk pack",
        worker,
        Pack {
            shape,
            tables,
            set: set.to_vec(),
        },
    );

    let width = dag.width("df jk batch", worker, &Batch { shape, naux });
    let body = dag.len();
    let batch = dag.upload_heap(
        LaneId { worker, lane: 1 },
        b,
        0..width * shape.n_columns,
        "B batch",
    );
    dag.add_gpu(
        "df jk batch",
        worker,
        Contract {
            shape,
            tables,
            naux,
            width,
            batch,
            set: set.to_vec(),
            acc: acc.clone(),
        },
    );
    // A fitting set of one batch or less is one trip, whose ring is one
    // copy: a counted loop's depth is at most its trips.
    let trips = naux.div_ceil(width);
    dag.add_counted(body, trips, DEPTH.min(trips));

    Ok(dag.add_gpu("df jk finish", worker, Finish { shape, tables, acc }))
}

/// What every task of a pass is sized by.
#[derive(Clone, Copy, Debug)]
struct Shape {
    nao: usize,
    n_columns: usize,
    /// Whether the set holds a J: the batch's γ.
    coulomb: bool,
    /// The widest factor of the set's K, 0 with none: the batch's X.
    max_nocc: usize,
}

impl Shape {
    /// `nao²`.
    fn square(&self) -> usize {
        self.nao * self.nao
    }

    /// The device rows of a batch of `functions` aux functions, in
    /// doubles: the uploaded batch, γ for a J, the slab and X for a K.
    fn batch_rows(&self, functions: usize) -> [usize; 4] {
        let k = self.max_nocc > 0;
        [
            functions * self.n_columns,
            if self.coulomb { functions } else { 0 },
            if k { functions * self.square() } else { 0 },
            functions * self.nao * self.max_nocc,
        ]
    }
}

/// The batch's adaptive part: the batches' ring at a width.
struct Batch {
    shape: Shape,
    naux: usize,
}

impl Adaptive for Batch {
    fn width_max(&self) -> usize {
        BATCH_MAX
    }

    /// Two copies of a batch's body, its rows each padded to the device
    /// alignment and the body's total padded again: the counted loop's
    /// region (`docs/memory.md`, "Nested plans"). A width past `naux` holds
    /// `naux` functions.
    fn bytes(&self, width: usize) -> usize {
        let body: usize = self
            .shape
            .batch_rows(width.min(self.naux))
            .into_iter()
            .filter(|&d| d > 0)
            .map(pad)
            .sum();
        DEPTH * body.next_multiple_of(DEVICE_ALIGNMENT)
    }

    /// Two copies of a batch's pinned window, padded as the device rows
    /// are: the counted loop's pinned region.
    fn pinned_bytes(&self, width: usize) -> usize {
        let window = self.shape.batch_rows(width.min(self.naux))[0];
        DEPTH * pad(window).next_multiple_of(DEVICE_ALIGNMENT)
    }

    fn warn_below(&self) -> usize {
        BATCH_WARN_BELOW
    }

    fn power_of_two(&self) -> bool {
        true
    }
}

/// `doubles` doubles' bytes, padded to the device alignment, which the
/// pinned rows share.
fn pad(doubles: usize) -> usize {
    (doubles * size_of::<f64>()).next_multiple_of(DEVICE_ALIGNMENT)
}

/// A contraction's accumulator and, for a J, its packed density.
#[derive(Clone, Copy, Debug)]
enum Acc {
    /// The packed density `dp` and the packed J.
    Coulomb {
        dp: RowId<Device, f64>,
        jp: RowId<Device, f64>,
    },
    /// K's lower triangle.
    Exchange { k: RowId<Device, f64> },
}

/// GPU: each J's density packed, every accumulator zeroed.
struct Pack {
    shape: Shape,
    tables: PackedTables,
    set: Vec<Contraction>,
}

impl GpuTask for Pack {
    type Rows = Vec<Acc>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Vec<Acc> {
        if self.shape.coulomb {
            self.tables.read(device);
        }
        self.set
            .iter()
            .enumerate()
            .map(|(i, c)| match *c {
                Contraction::Coulomb(d) => {
                    device.read(d);
                    Acc::Coulomb {
                        dp: device.output(&format!("D packed {i}"), self.shape.n_columns),
                        jp: device.output(&format!("J packed {i}"), self.shape.n_columns),
                    }
                }
                Contraction::Exchange { .. } => Acc::Exchange {
                    k: device.output(&format!("K lower {i}"), self.shape.square()),
                },
            })
            .collect()
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        dims(&self.shape, &self.set)
    }

    fn run(
        &self,
        acc: &Vec<Acc>,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        for (c, a) in self.set.iter().zip(acc) {
            match (*c, *a) {
                (Contraction::Coulomb(d), Acc::Coulomb { dp, jp }) => {
                    self.tables
                        .pack(worker, device, &device.get(d), &mut device.take_mut(dp))?;
                    memset::zero(&mut device.take_mut(jp), worker)?;
                }
                (_, Acc::Exchange { k }) => memset::zero(&mut device.take_mut(k), worker)?,
                (Contraction::Exchange { .. }, Acc::Coulomb { .. }) => {
                    unreachable!("the layout gave each contraction its own kind's rows")
                }
            }
        }
        Ok(())
    }
}

/// The dimensions a pass's task reports.
fn dims(shape: &Shape, set: &[Contraction]) -> Vec<(&'static str, usize)> {
    vec![
        ("nao", shape.nao),
        ("n_columns", shape.n_columns),
        ("contractions", set.len()),
    ]
}

/// GPU: one batch of aux functions contracted into every accumulator,
/// trip `k` of the batches' loop functions `k width ..`, `width` of them
/// or the `naux - k width` left in the last.
struct Contract {
    shape: Shape,
    tables: PackedTables,
    /// The fitting set's functions.
    naux: usize,
    /// A full batch's aux functions.
    width: usize,
    /// The uploaded batch, `width · n_columns` doubles, pair-fastest, of
    /// which the trip's functions are the head.
    batch: RowId<Device, f64>,
    set: Vec<Contraction>,
    acc: Vec<Acc>,
}

/// A batch task's scratch.
#[derive(Clone, Copy)]
struct Scratch {
    gamma: Option<RowId<Device, f64>>,
    slab_x: Option<(RowId<Device, f64>, RowId<Device, f64>)>,
}

impl GpuTask for Contract {
    type Rows = Scratch;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Scratch {
        device.read(self.batch);
        let k = self.shape.max_nocc > 0;
        if k {
            self.tables.read(device);
        }
        for (c, a) in self.set.iter().zip(&self.acc) {
            match (*c, *a) {
                (Contraction::Coulomb(_), Acc::Coulomb { dp, jp }) => {
                    device.read(dp);
                    device.write(jp);
                }
                (Contraction::Exchange { c, .. }, Acc::Exchange { k }) => {
                    device.read(c);
                    device.write(k);
                }
                _ => unreachable!("the pack gave each contraction its own kind's rows"),
            }
        }
        let [_, gamma, slab, x] = self.shape.batch_rows(self.width);
        Scratch {
            gamma: self.shape.coulomb.then(|| device.scratch("gamma", gamma)),
            slab_x: k.then(|| (device.scratch("slab", slab), device.scratch("X", x))),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        let mut dims = dims(&self.shape, &self.set);
        dims.push(("batch", self.width));
        dims
    }

    fn run(
        &self,
        scratch: &Scratch,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let Shape { nao, n_columns, .. } = self.shape;
        // The trip's functions: the upload clamped the last range to the
        // heap row's end, and left the rest of the device row stale.
        let count = self.width.min(self.naux - device.iteration() * self.width);
        let batch = device.get(self.batch);
        let (batch, _) = split_at(&batch, count * n_columns);
        // add_streamed_jk checked that nao, n_columns, naux and naux · nocc
        // fit an i32, and the batch is at most naux functions; the handle is
        // the worker's, bound to its stream, so every call runs in order.
        let (nao_i, columns, functions) = (nao as i32, n_columns as i32, count as i32);
        // Each scratch row asked of the view once: it keeps a row's flag
        // until it drops.
        let mut gamma = scratch.gamma.map(|g| device.take_mut(g));
        let mut slab_x = scratch
            .slab_x
            .map(|(slab, x)| (device.take_mut(slab), device.take_mut(x)));
        if let Some((slab, _)) = &mut slab_x {
            let (mut slab, _) = split_at_mut(slab, count * nao * nao);
            self.tables
                .gather(worker, device, count, &batch, &mut slab)?;
        }
        for (c, a) in self.set.iter().zip(&self.acc) {
            match (*c, *a, &mut gamma, &mut slab_x) {
                (_, Acc::Coulomb { dp, jp }, Some(gamma), _) => {
                    gemv(
                        worker.blas(),
                        wwr_blas_operation::WWR_BLAS_OP_T,
                        columns,
                        functions,
                        1.0,
                        &batch,
                        columns,
                        &device.get(dp),
                        1,
                        0.0,
                        gamma,
                        1,
                    )?;
                    gemv(
                        worker.blas(),
                        wwr_blas_operation::WWR_BLAS_OP_N,
                        columns,
                        functions,
                        1.0,
                        &batch,
                        columns,
                        &*gamma,
                        1,
                        1.0,
                        &mut device.take_mut(jp),
                        1,
                    )?;
                }
                (Contraction::Exchange { c, nocc }, Acc::Exchange { k }, _, Some((slab, x))) => {
                    let nocc = nocc as i32;
                    gemm_strided_batched(
                        worker.blas(),
                        wwr_blas_operation::WWR_BLAS_OP_N,
                        wwr_blas_operation::WWR_BLAS_OP_N,
                        nao_i,
                        nocc,
                        nao_i,
                        1.0,
                        &*slab,
                        nao_i,
                        (nao * nao) as i64,
                        &device.get(c),
                        nao_i,
                        0,
                        0.0,
                        x,
                        nao_i,
                        i64::from(nao_i * nocc),
                        functions,
                    )?;
                    syrk(
                        worker.blas(),
                        wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                        wwr_blas_operation::WWR_BLAS_OP_N,
                        nao_i,
                        functions * nocc,
                        2.0,
                        &*x,
                        nao_i,
                        1.0,
                        &mut device.take_mut(k),
                        nao_i,
                    )?;
                }
                _ => unreachable!("the layout declared each contraction's scratch"),
            }
        }
        Ok(())
    }
}

/// GPU: each packed J unpacked, each K's lower triangle copied out and
/// mirrored.
struct Finish {
    shape: Shape,
    tables: PackedTables,
    acc: Vec<Acc>,
}

impl GpuTask for Finish {
    type Rows = Vec<RowId<Device, f64>>;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        _: &mut Layout<'_, Pinned>,
    ) -> Vec<RowId<Device, f64>> {
        if self.shape.coulomb {
            self.tables.read(device);
        }
        let square = self.shape.square();
        self.acc
            .iter()
            .enumerate()
            .map(|(i, a)| match *a {
                Acc::Coulomb { jp, .. } => {
                    device.read(jp);
                    device.output(&format!("J {i}"), square)
                }
                Acc::Exchange { k } => {
                    device.read(k);
                    device.output(&format!("K {i}"), square)
                }
            })
            .collect()
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("nao", self.shape.nao),
            ("n_columns", self.shape.n_columns),
            ("contractions", self.acc.len()),
        ]
    }

    fn run(
        &self,
        out: &Vec<RowId<Device, f64>>,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        for (a, &o) in self.acc.iter().zip(out) {
            let mut o = device.take_mut(o);
            match *a {
                Acc::Coulomb { jp, .. } => {
                    self.tables
                        .gather(worker, device, 1, &device.get(jp), &mut o)?;
                }
                Acc::Exchange { k } => {
                    d2d(&mut o, &device.get(k), worker)?;
                    symmetrize_from(
                        worker,
                        wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                        self.shape.nao,
                        &mut o,
                        self.shape.nao,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use fb_core_planner::{Adaptive, Room, widest};
    use proptest::prelude::*;

    use super::{BATCH_MAX, Batch, Shape};

    proptest! {
        /// The batch's bytes, device and pinned, are monotone in the
        /// width, so the planner's bisection finds the widest that fits:
        /// `widest` against a linear scan over random shapes and rooms.
        #[test]
        fn widest_is_the_linear_scans(
            nao in 1_usize..400,
            n_columns in 1_usize..80_000,
            naux in 1_usize..3_000,
            coulomb in any::<bool>(),
            max_nocc in 0_usize..200,
            device in 0_usize..2_000_000_000,
            pinned in 0_usize..100_000_000,
        ) {
            let part = Batch {
                shape: Shape { nao, n_columns, coulomb, max_nocc },
                naux,
            };
            let room = Room { device, pinned };
            let mut scan = 1;
            for w in 1..=BATCH_MAX {
                prop_assert!(part.bytes(w) >= part.bytes(w - 1));
                prop_assert!(part.pinned_bytes(w) >= part.pinned_bytes(w - 1));
                if part.bytes(w) <= device && part.pinned_bytes(w) <= pinned {
                    scan = w;
                }
            }
            prop_assert_eq!(widest(&part, room), 1 << scan.ilog2());
        }
    }
}
