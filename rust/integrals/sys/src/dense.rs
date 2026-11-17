//! The dense conventional ERI's device path: row blocks of the two dense
//! matrices of a basis's Cartesian ERI, formed from the records between the
//! basis and itself ([`MixedClassRows`], every ordered shell pair) by two
//! tasks ([`add_dense_tasks`]).
//!
//! **The matrices.** With every Cartesian AO pair unpacked,
//! `(m, n) ↦ m n_cart + n`,
//!
//! ```text
//! M_J[(m,n), (l,s)] = (mn|ls)        J = M_J vec(D)
//! M_K[(m,n), (l,s)] = (ml|ns)        K = M_K vec(D)
//! ```
//!
//! each dense `n_cart² × n_cart²` and row-major, `vec(D)[(l,s)] = D_ls`,
//! for any D, symmetric or not (FelizJueves's `eri/full_eri/dense.py`). A
//! row's `m` is its bra's first AO in both layouts, so the shells
//! `[sh0, sh1)` of `m` own the contiguous rows `[m0 n_cart, m1 n_cart)` of
//! either, every column: the **row block** [`add_dense_tasks`] writes,
//! `(m1 - m0) n_cart` rows of `n_cart²` doubles, `m0` and `m1` the AOs
//! the shells open and close. Blocks over disjoint shell ranges are
//! disjoint, and together they are the matrix.
//!
//! **No symmetry.** A task is one ordered shell quartet `(A B|C D)`, every
//! one of them, its block stored element by element at `M_J[(a,b), (c,d)]`
//! or at `M_K[(a,c), (b,d)]`, so every element has one owner and one plain
//! store: the blocks are bit-reproducible, and an element's value is its
//! task's whatever the block it lies in. The price is up to 8× the unique
//! quartets' work, and both layouts each compute it.
//!
//! The tasks, added in this order:
//!
//! - **`dense tables`**, on the CPU: every launch's bra pairs (the record
//!   range and the two AO offsets of each), every class's ket pairs and
//!   each ket class's task table, all formed at planning, written into
//!   pinned rows.
//! - **`dense eri`**, on the worker's stream: the tables uploaded into
//!   scratch, every class's ζ, P, P − A, P − B and `s`
//!   ([`Fields::CONVENTIONAL`]) read in place from its records rows, both
//!   blocks zeroed and one [`fb_dense_eri_class`] per launch into each,
//!   `M_J`'s and `M_K`'s, its outputs.
//!
//! **The launches.** The kernels index a block in 32 bits, so the block is
//! cut at shell boundaries into **sub-blocks**, each of rows ·
//! `n_cart²` within `i32` ([`DenseBlocks::blocks`]), and every launch
//! writes one sub-block. A launch is a (bra class, ket class) pair over a
//! batch of the sub-block's bra pairs of that class whose first shell is in
//! it, crossed with every ket pair of the ket class: tasks
//! `t ↦ (t / n_kpair, t % n_kpair)`, a prefix of the ket class's task
//! table. A batch is at most `max(1, TASKS_PER_LAUNCH / n_kpair)` bra pairs,
//! so a table is about `TASKS_PER_LAUNCH` tasks per ket class however
//! large the basis, where one task per (bra pair, ket pair) would be
//! `n_pairs²` of them.
//!
//! As the packed tensor's are, every index a pair holds is formed at
//! planning from tables [`RecordTables::mixed`] checked: a pair's record
//! range is its prefix in its class, and both its offsets its shells', each
//! opening a block inside `n_cart`; a bra pair's first shell is in the
//! sub-block's shells, so its rows inside the sub-block. That the records
//! are the tables' is a contract, checked always.

use std::ops::Range;
use std::sync::Arc;

use contracts::{debug_ensures, requires};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_log::{eq, fits_i32, le, lt};
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::Stream;

use crate::records::{Fields, MixedClassRows, RecordTables};
use crate::{
    FB_CONVENTIONAL_MAX_MOMENTUM, FB_CONVENTIONAL_PAIR_FIELDS, FB_CONVENTIONAL_TASK_FIELDS,
    FB_DENSE_LAYOUT_J, FB_DENSE_LAYOUT_K, fb_dense_eri_class, fb_dense_row_dot, launched,
};

/// The fields of one shell pair.
const PAIR_FIELDS: usize = FB_CONVENTIONAL_PAIR_FIELDS as usize;
/// The fields of one task.
const TASK_FIELDS: usize = FB_CONVENTIONAL_TASK_FIELDS as usize;
/// About the tasks of one launch, and of one ket class's task table: a bra
/// batch is `max(1, TASKS_PER_LAUNCH / n_kpair)` pairs. 2^16 threads are
/// 512 blocks of four warps, a few waves of a card at these kernels'
/// occupancy, and a table of half a MiB.
const TASKS_PER_LAUNCH: usize = 1 << 16;

/// A worker's row blocks of the dense matrices, [`add_dense_tasks`]'
/// product.
#[derive(Clone, Debug)]
pub struct DenseBlocks {
    /// `M_J`'s rows `[m0 n_cart, m1 n_cart)`: an output row of
    /// `(m1 - m0) n_cart³` doubles, row-major, `n_cart²` to a row.
    pub j: RowId<Device, f64>,
    /// `M_K`'s rows, laid out as `j`.
    pub k: RowId<Device, f64>,
    /// The sub-blocks, as the AOs `m` of each, consecutive from `m0` to
    /// `m1`, each cut at a shell boundary and each of rows · `n_cart²`
    /// within `i32`: what the kernels wrote by, and what a reader indexing
    /// the block in 32 bits, as the BLAS does, reads by.
    pub blocks: Vec<Range<usize>>,
}

/// Add the row blocks of the dense ERI matrices `M_J` and `M_K` of a basis
/// over the shells `shells` of `m` to `dag` on GPU worker `worker`, and
/// return them ([`DenseBlocks`]): two output rows over the Cartesian AOs,
/// laid out as the module doc has them.
///
/// `classes` are the records between the basis and itself
/// ([`add_mixed_record_tasks`]) and `tables` the tables they were formed
/// from, whose ordered shell pairs give each pair's record range and AO
/// offsets. The tasks are `dense tables` (CPU) and `dense eri` (stream),
/// the module doc's. Both rows are in the plan's table at planning, so a
/// block that does not fit the card is refused by admission, with the rows
/// named.
///
/// A class above the kernels' table ([`FB_CONVENTIONAL_MAX_MOMENTUM`]) is
/// the launcher's refusal at run time, `Unsupported` naming the class; a
/// caller refuses one at planning.
///
/// [`add_mixed_record_tasks`]: crate::add_mixed_record_tasks
///
/// # Panics
/// If one shell's rows alone pass the kernels' 32-bit index (rows ·
/// `n_cart²`), so no sub-block holds it; if the staged tables pass an
/// `i32` count; or a class is of another worker's or plan's table
/// ([`Dag::add_gpu`]).
#[requires(
    tables.formed_with_itself(classes),
    "classes are the records of tables, a basis with itself"
)]
#[requires(
    fits_i32!(tables.n_cart().checked_mul(tables.n_cart())),
    "n_cart² fits the kernels' 32-bit index"
)]
#[requires(
    lt!(shells.start, shells.end) && le!(shells.end, tables.n_bra_shells()),
    "shells is a nonempty range of the basis's"
)]
#[debug_requires(
    classes
        .iter()
        .all(|c| c.la().max(c.lb()) <= FB_CONVENTIONAL_MAX_MOMENTUM),
    "every class is in the kernels' table"
)]
#[track_caller]
pub fn add_dense_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    classes: &[MixedClassRows],
    tables: &RecordTables,
    shells: Range<usize>,
) -> DenseBlocks {
    let staging = Arc::new(staging(classes, tables, shells));
    let blocks = staging.blocks.iter().map(|b| b.aos.clone()).collect();
    let rows = dag.add_cpu(
        "dense tables",
        worker,
        Tables {
            staging: Arc::clone(&staging),
        },
    );
    let out = dag.add_gpu(
        "dense eri",
        worker,
        Tensor {
            staging,
            tables: rows,
        },
    );
    DenseBlocks {
        j: out.j,
        k: out.k,
        blocks,
    }
}

/// One launch: a batch of a sub-block's bra pairs of one class, crossed
/// with every ket pair of another.
#[derive(Debug, Clone, Copy)]
struct Launch {
    bra: usize,
    ket: usize,
    /// The batch's first pair in `pairs`, and the stride of its segment's
    /// fields.
    bra_at: usize,
    bra_stride: usize,
    n_bpair: usize,
    /// The ket class's segment in `pairs`.
    ket_at: usize,
    n_kpair: usize,
    /// The ket class's task table in `tasks`, and its stride.
    task_at: usize,
    task_stride: usize,
}

/// One sub-block: its AOs `m` and its launches.
#[derive(Debug)]
struct Block {
    aos: Range<usize>,
    launches: Vec<Launch>,
}

/// What the tasks need, formed at planning.
#[derive(Debug)]
struct Staging {
    n_cart: usize,
    /// The worker's AOs `m`, `[m0, m1)`.
    aos: Range<usize>,
    /// The records classes, whose fields the launches read in place.
    classes: Vec<MixedClassRows>,
    blocks: Vec<Block>,
    /// Every segment of pairs, each field-major.
    pairs: Vec<i32>,
    /// Every ket class's task table, each field-major.
    tasks: Vec<i32>,
}

/// The staging of [`add_dense_tasks`]: the sub-blocks, their launches and
/// the tables they index, every extent the launcher trusts formed here.
#[debug_ensures(
    ret.blocks.first().map(|b| b.aos.start) == Some(ret.aos.start)
        && ret.blocks.windows(2).all(|w| w[0].aos.end == w[1].aos.start)
        && ret.blocks.last().map(|b| b.aos.end) == Some(ret.aos.end),
    "the sub-blocks tile the worker's AOs in order"
)]
#[track_caller]
fn staging(classes: &[MixedClassRows], tables: &RecordTables, shells: Range<usize>) -> Staging {
    let n = tables.n_cart();
    let n_bra_shells = tables.n_bra_shells();
    let offset = &tables.shells().ao_offset;
    // The bra basis's shells' AOs: offsets ascend in basis order and the
    // last shell closes at n_cart (RecordTables::mixed).
    let shell_end = |s: usize| {
        if s + 1 < n_bra_shells {
            offset[s + 1] as usize
        } else {
            n
        }
    };
    let aos = offset[shells.start] as usize..shell_end(shells.end - 1);

    // The sub-blocks: shells taken greedily while rows · n² fits an i32.
    let n_col = n * n;
    let max_aos = i32::MAX as usize / (n * n_col);
    let mut cuts = vec![shells.start];
    for s in shells.clone() {
        let first = offset[*cuts.last().expect("a cut")] as usize;
        let width = shell_end(s) - offset[s] as usize;
        assert!(
            width <= max_aos,
            "add_dense_tasks: shell {s}'s {width} AOs are {} rows of {n_col} doubles, past the \
             kernels' 32-bit index",
            width * n
        );
        if shell_end(s) - first > max_aos {
            cuts.push(s);
        }
    }
    cuts.push(shells.end);

    let mut pairs = Vec::new();
    // Every class's ket pairs, one segment each, and its task table.
    let mut kets = Vec::new();
    let mut tasks = Vec::new();
    for c in tables.classes() {
        let n_k = c.bra.len();
        let at = push_segment(&mut pairs, tables, c, &(0..n_k).collect::<Vec<_>>());
        let batch = (TASKS_PER_LAUNCH / n_k.max(1)).max(1);
        let stride = batch * n_k;
        let task_at = tasks.len();
        tasks.reserve(TASK_FIELDS * stride);
        tasks.extend((0..stride).map(|t| (t / n_k) as i32));
        tasks.extend((0..stride).map(|t| (t % n_k) as i32));
        kets.push((at, n_k, batch, task_at, stride));
    }

    let mut blocks = Vec::new();
    for cut in cuts.windows(2) {
        let range = cut[0]..cut[1];
        let block_aos = offset[cut[0]] as usize..shell_end(cut[1] - 1);
        let mut launches = Vec::new();
        for (bra, c) in tables.classes().iter().enumerate() {
            let chosen: Vec<usize> = (0..c.bra.len())
                .filter(|&p| range.contains(&(c.bra[p] as usize)))
                .collect();
            if chosen.is_empty() {
                continue;
            }
            let at = push_segment(&mut pairs, tables, c, &chosen);
            for (ket, &(ket_at, n_kpair, batch, task_at, task_stride)) in kets.iter().enumerate() {
                if n_kpair == 0 {
                    continue;
                }
                for first in (0..chosen.len()).step_by(batch) {
                    launches.push(Launch {
                        bra,
                        ket,
                        bra_at: at + first,
                        bra_stride: chosen.len(),
                        n_bpair: batch.min(chosen.len() - first),
                        ket_at,
                        n_kpair,
                        task_at,
                        task_stride,
                    });
                }
            }
        }
        blocks.push(Block {
            aos: block_aos,
            launches,
        });
    }
    assert!(
        i32::try_from(pairs.len()).is_ok() && i32::try_from(tasks.len()).is_ok(),
        "add_dense_tasks: {} pair and {} task entries, past an i32 count",
        pairs.len(),
        tasks.len()
    );
    Staging {
        n_cart: n,
        aos,
        classes: classes.to_vec(),
        blocks,
        pairs,
        tasks,
    }
}

/// The pairs `chosen` of class `c` appended to `pairs` as one field-major
/// segment: each pair's record range, in its class's records, and the AO
/// offsets of its two shells. Returns the segment's first entry. Every
/// entry is an i32 already (RecordTables::mixed).
fn push_segment(
    pairs: &mut Vec<i32>,
    tables: &RecordTables,
    c: &crate::ClassPairs,
    chosen: &[usize],
) -> usize {
    let offset = &tables.shells().ao_offset;
    let at = pairs.len();
    let n = chosen.len();
    let mut run = vec![0; PAIR_FIELDS * n];
    for (i, &p) in chosen.iter().enumerate() {
        let fields = [
            c.rec_begin[p],
            c.rec_begin[p + 1],
            offset[c.bra[p] as usize],
            offset[c.ket[p] as usize],
        ];
        for (f, v) in fields.into_iter().enumerate() {
            run[f * n + i] = v;
        }
    }
    pairs.extend(run);
    at
}

/// The staged tables, of one kind: pinned as the CPU task's outputs, device
/// as the stream task's scratch.
struct TableRows<K> {
    pairs: RowId<K, i32>,
    tasks: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

/// CPU: the pairs and the task tables, formed at planning, into pinned rows.
struct Tables {
    staging: Arc<Staging>,
}

impl CpuTask for Tables {
    type Rows = TableRows<Pinned>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let s = &self.staging;
        TableRows {
            pairs: host.output("dense pairs", s.pairs.len()),
            tasks: host.output("dense tasks", s.tasks.len()),
        }
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let s = &self.staging;
        memcpy_h2p(host, rows.pairs, &s.pairs);
        memcpy_h2p(host, rows.tasks, &s.tasks);
        Ok(())
    }
}

/// GPU: the tables uploaded, both blocks zeroed and every launch of both
/// layouts made, each class's records read in place.
struct Tensor {
    staging: Arc<Staging>,
    tables: TableRows<Pinned>,
}

#[derive(Clone, Copy)]
struct TensorRows {
    tables: TableRows<Device>,
    j: RowId<Device, f64>,
    k: RowId<Device, f64>,
}

impl GpuTask for Tensor {
    type Rows = TensorRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> TensorRows {
        host.read(self.tables.pairs);
        host.read(self.tables.tasks);
        let s = &self.staging;
        for c in &s.classes {
            c.rows().read(device, Fields::CONVENTIONAL);
        }
        let len = s.aos.len() * s.n_cart * s.n_cart * s.n_cart;
        TensorRows {
            tables: TableRows {
                pairs: device.scratch("dense pairs", s.pairs.len()),
                tasks: device.scratch("dense tasks", s.tasks.len()),
            },
            j: device.output("M_J", len),
            k: device.output("M_K", len),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        let s = &self.staging;
        vec![("n_cart", s.n_cart), ("m", s.aos.len())]
    }

    fn run(
        &self,
        rows: &TensorRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let s = &self.staging;
        let mut pairs = device.take_mut(rows.tables.pairs);
        memcpy_p2d(&mut pairs, host, self.tables.pairs, worker)?;
        let mut tasks = device.take_mut(rows.tables.tasks);
        memcpy_p2d(&mut tasks, host, self.tables.tasks, worker)?;
        let views: Vec<_> = s
            .classes
            .iter()
            .map(|c| c.rows().view(device, Fields::CONVENTIONAL))
            .collect();
        let n = s.n_cart;
        for (layout, out) in [(FB_DENSE_LAYOUT_J, rows.j), (FB_DENSE_LAYOUT_K, rows.k)] {
            let mut eri = device.take_mut(out);
            // Every element is owned by one task; the zero makes the block's
            // value independent of that, should a class ever be skipped.
            memset::zero(&mut eri, worker)?;
            for b in &s.blocks {
                // The sub-block's first element, inside the block: its AOs
                // are inside the worker's (the staging tiles them).
                let at = (b.aos.start - s.aos.start) * n * n * n;
                for l in &b.launches {
                    let (bra, ket) = (&s.classes[l.bra], &s.classes[l.ket]);
                    // SAFETY: each view's arrays are its class's n_rec-long
                    // rows of Fields::CONVENTIONAL, every record field the
                    // launcher reads (a test scans it), the rest null, written
                    // by the class's records task before this task's stream
                    // work; the bra batch is n_bpair pairs from bra_at of a
                    // PAIR_FIELDS segment of stride bra_stride, the ket class
                    // a segment of n_kpair, and its task table TASK_FIELDS
                    // fields of task_stride >= n_bpair n_kpair tasks
                    // (t / n_kpair, t % n_kpair), all inside the pairs and
                    // tasks rows copied above on this stream from the tables
                    // task, which wrote the staging formed at planning: every
                    // record range a pair's prefix in its class, so inside
                    // [0, n_rec) (the classes are the tables', a contract),
                    // both offsets the pair's shells', each opening a block
                    // inside [0, n_cart) (RecordTables::mixed), and every bra
                    // pair's first shell in the sub-block's shells, so its
                    // rows inside the sub-block; eri from `at` holds the
                    // sub-block's rows of n_cart² doubles, n_cart² and its
                    // rows · n_cart² within i32 (the contract and the
                    // staging), inside this task's block; the views are host
                    // structs that outlive the call, which only enqueues on
                    // the task's stream.
                    let status = unsafe {
                        fb_dense_eri_class(
                            worker.stream(),
                            layout,
                            &views[l.bra],
                            &views[l.ket],
                            l.n_bpair as i32,
                            l.bra_stride as i32,
                            pairs.as_ptr().wrapping_add(l.bra_at),
                            l.n_kpair as i32,
                            pairs.as_ptr().wrapping_add(l.ket_at),
                            l.task_stride as i32,
                            tasks.as_ptr().wrapping_add(l.task_at),
                            n as i32,
                            b.aos.start as i32,
                            (b.aos.len() * n) as i32,
                            eri.as_mut_ptr().wrapping_add(at),
                        )
                    };
                    launched(
                        "fb_dense_eri_class",
                        &[bra.la(), bra.lb(), ket.la(), ket.lb()],
                        status,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// `y = M x` on `stream`, `M` the `n_row × n_col` row-major matrix `m`:
/// [`fb_dense_row_dot`], each entry of `y` its row's products summed in an
/// order fixed by `n_col` alone, so the same bits whatever `n_row` and the
/// rows beside it. What a row block of `M_J` or `M_K` is read by
/// ([`DenseBlocks`]), so that J and K over blocks on several devices are
/// one device's bits.
///
/// # Errors
/// The launcher's refusal or launch error, as
/// [`fb_kernel_abi::launch_status`] maps it.
#[requires(le!(1, n_row) && le!(1, n_col), "the counts are positive")]
#[requires(
    fits_i32!(n_row.checked_mul(n_col)),
    "n_row n_col fits the kernel's 32-bit index"
)]
#[requires(eq!(Some(m.len()), n_row.checked_mul(n_col)), "m is n_row n_col doubles")]
#[requires(eq!(x.len(), n_col), "x is n_col doubles")]
#[requires(eq!(y.len(), n_row), "y is n_row doubles")]
#[track_caller]
pub fn dense_row_dot(
    stream: &Stream,
    n_row: usize,
    n_col: usize,
    m: &impl DeviceSlice<f64>,
    x: &impl DeviceSlice<f64>,
    y: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let y = y.as_mut_ptr();
    // SAFETY: m is n_row n_col doubles, x n_col and y n_row, by the
    // contracts, which also bound n_row n_col within an i32, so the casts
    // are exact; y has no other user until the enqueued work finishes
    // (DeviceSliceMut's promise), so apart from m and x; the call only
    // enqueues.
    let status = unsafe {
        fb_dense_row_dot(
            stream.stream(),
            n_row as i32,
            n_col as i32,
            m.as_ptr(),
            x.as_ptr(),
            y,
        )
    };
    launched("fb_dense_row_dot", &[], status)
}
