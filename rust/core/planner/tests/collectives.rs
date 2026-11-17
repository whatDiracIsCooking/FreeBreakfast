//! The collectives on real cards: a matrix whose rows are partitioned over
//! the GPU workers, one worker per device, multiplied into a vector every
//! iteration of a decision loop (`docs/programming_model.md`,
//! "Collectives").
//!
//! The pattern multi-GPU J and K by GEMV take: the vector is the privileged
//! lane's, broadcast to every worker at the top of the iteration; each
//! worker multiplies its own rows, held for the whole plan; the pieces are
//! gathered back onto worker 0, which updates the carried vector. Each
//! worker also carries a row of its own across the iterations, its
//! pieces' running sum, gathered after the loop; and each worker's
//! transpose product of its rows with that sum is reduced onto worker 0.
//!
//! **The bar is zero, by analysis.** Every entry of the matrix is -1, 0 or
//! 1 and of the start vector an integer of magnitude at most 2, so every
//! product and every partial sum, in whatever order a library forms it, is
//! an integer far below 2^53 (the largest, the transpose product's, at
//! most `2 n^4` per iterate, four of them summed and `n` of those, so
//! `8 n^5 < 2^44` for `n = 257`) and so exact.
//! Each worker count's results are then bitwise the host's, and so one
//! another's: the gather's pieces are disjoint and the reduce's partials
//! integers, so neither's order can show.

use std::ops::{ControlFlow, Range};
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, Decision, GpuTask, Partition};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemv;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::device_count;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// The matrix's order.
const N: usize = 257;
/// The loop's iterations.
const ITERATIONS: usize = 4;

/// Entry `(i, j)` of the matrix: -1, 0 or 1.
fn entry(i: usize, j: usize) -> f64 {
    ((i * 7 + j * 3 + i * j) % 3) as f64 - 1.0
}

/// Entry `j` of the start vector.
fn start(j: usize) -> f64 {
    (j % 5) as f64 - 2.0
}

/// What the plan computes, on the host: the vector after the loop, the
/// pieces' running sum, and the transpose product of the matrix with it.
fn reference() -> [Vec<f64>; 3] {
    let product = |x: &[f64]| -> Vec<f64> {
        (0..N)
            .map(|i| (0..N).map(|j| entry(i, j) * x[j]).sum())
            .collect()
    };
    let mut x: Vec<f64> = (0..N).map(start).collect();
    let mut sum = vec![0.0; N];
    for _ in 0..ITERATIONS {
        let y = product(&x);
        sum.iter_mut().zip(&y).for_each(|(s, y)| *s += y);
        x = y;
    }
    let transpose = (0..N)
        .map(|j| (0..N).map(|i| entry(i, j) * sum[i]).sum())
        .collect();
    [x, sum, transpose]
}

/// CPU: a worker's rows of the matrix, column-major, `rows.len()` by `N`,
/// or the start vector, into a pinned output.
struct Fill {
    rows: Option<Range<usize>>,
}

impl CpuTask for Fill {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let len = self.rows.as_ref().map_or(N, |r| r.len() * N);
        h.output("filled", len)
    }

    fn run(&self, out: &Self::Rows, h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let out = h.take_mut(*out);
        match &self.rows {
            Some(rows) => {
                for j in 0..N {
                    for (k, i) in rows.clone().enumerate() {
                        out[j * rows.len() + k] = entry(i, j);
                    }
                }
            }
            None => out.iter_mut().enumerate().for_each(|(j, e)| *e = start(j)),
        }
        Ok(())
    }
}

/// GPU: a worker's running sum, zeroed before the loop.
struct Zero(usize);

impl GpuTask for Zero {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.output("sum", self.0)
    }

    fn run(
        &self,
        sum: &Self::Rows,
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memset::zero(&mut d.take_mut(*sum), stream)
    }
}

/// GPU: a worker's piece of the product, `piece = M_w x`, and its running
/// sum carried across the iterations, `sum += M_w x`. A worker of no rows
/// launches nothing.
struct Multiply {
    rows: usize,
    m: RowId<Device, f64>,
    x: RowId<Device, f64>,
    sum: RowId<Device, f64>,
}

impl GpuTask for Multiply {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.m);
        d.read(self.x);
        d.write(self.sum);
        d.output("piece", self.rows)
    }

    fn run(
        &self,
        piece: &Self::Rows,
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        if self.rows == 0 {
            return Ok(());
        }
        let (m, x) = (d.get(self.m), d.get(self.x));
        let (rows, n) = (self.rows as i32, N as i32);
        let blas = stream.blas();
        gemv(
            blas,
            OP_N,
            rows,
            n,
            1.0,
            &m,
            rows,
            &x,
            1,
            0.0,
            &mut d.take_mut(*piece),
            1,
        )?;
        gemv(
            blas,
            OP_N,
            rows,
            n,
            1.0,
            &m,
            rows,
            &x,
            1,
            1.0,
            &mut d.take_mut(self.sum),
            1,
        )
    }
}

/// GPU: a worker's partial of the transpose product, `M_w^T sum_w`, every
/// worker's of length `N`; zero for a worker of no rows.
struct Transpose {
    rows: usize,
    m: RowId<Device, f64>,
    sum: RowId<Device, f64>,
}

impl GpuTask for Transpose {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.m);
        d.read(self.sum);
        d.output("partial", N)
    }

    fn run(
        &self,
        partial: &Self::Rows,
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let mut partial = d.take_mut(*partial);
        if self.rows == 0 {
            return memset::zero(&mut partial, stream);
        }
        let (m, sum) = (d.get(self.m), d.get(self.sum));
        let (rows, n) = (self.rows as i32, N as i32);
        gemv(
            stream.blas(),
            OP_T,
            rows,
            n,
            1.0,
            &m,
            rows,
            &sum,
            1,
            0.0,
            &mut partial,
            1,
        )
    }
}

/// GPU, on worker 0: the gathered product becomes the next iterate, the
/// carried vector's last write in the body.
struct Carry {
    product: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

impl GpuTask for Carry {
    type Rows = ();

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.product);
        d.write(self.x);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2d(&d.take_mut(self.x), d.get(self.product), stream)
    }
}

/// Stops after `ITERATIONS`.
struct Count;

impl Decision for Count {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {}

    fn decide(
        &self,
        _: &(),
        _: &Bufs<'_, Pinned>,
        _: &Bufs<'_, Heap>,
        i: usize,
    ) -> Result<ControlFlow<()>> {
        Ok(if i + 1 < ITERATIONS {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    }
}

/// CPU: the three results, copied out once the executor waited on their
/// downloads.
struct Keep {
    rows: [RowId<Pinned, f64>; 3],
    into: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Keep {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        for row in self.rows {
            h.read(row);
        }
    }

    fn run(&self, _: &(), h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.into.lock().unwrap() = self.rows.iter().map(|&r| h.get(r).to_vec()).collect();
        Ok(())
    }
}

/// The plan over `split.parts()` workers, worker `w` on device `w`, run
/// once: the vector, the running sum and the transpose product, each read
/// back from worker 0.
fn run(split: &Partition) -> Vec<Vec<f64>> {
    let workers = split.parts();
    let out = Arc::new(Mutex::new(Vec::new()));
    let mut dag = Dag::new();
    let mut m = Vec::new();
    let mut sums = Vec::new();
    for w in 0..workers {
        dag.blas_workspace(w);
        let rows = split.range(w);
        let filled = dag.add_cpu(
            "fill M",
            w,
            Fill {
                rows: Some(rows.clone()),
            },
        );
        m.push(dag.upload(w, filled, "M", rows.len() * N));
        sums.push(dag.add_gpu("zero sum", w, Zero(rows.len())));
    }
    let start = dag.add_cpu("fill x", 0, Fill { rows: None });
    let x = dag.upload(0, start, "x", N);

    let body = dag.len();
    let xs = dag.broadcast("x", 0, x, N, workers);
    let pieces: Vec<_> = (0..workers)
        .map(|w| {
            let task = Multiply {
                rows: split.range(w).len(),
                m: m[w],
                x: xs[w],
                sum: sums[w],
            };
            dag.add_gpu("multiply", w, task)
        })
        .collect();
    let product = dag.gather("product", 0, &pieces, split);
    dag.add_gpu("carry", 0, Carry { product, x });
    dag.add_loop("count", 0, body, ITERATIONS, Count);

    let sum = dag.gather("sum", 0, &sums, split);
    let partials: Vec<_> = (0..workers)
        .map(|w| {
            let task = Transpose {
                rows: split.range(w).len(),
                m: m[w],
                sum: sums[w],
            };
            dag.add_gpu("transpose", w, task)
        })
        .collect();
    let transpose = dag.reduce("transpose", 0, &partials, N);
    let rows = [(x, "x"), (sum, "sum"), (transpose, "transpose")]
        .map(|(row, name)| (row, name.to_string(), N));
    let down = dag.download_rows(0, "results", rows);
    dag.add_cpu(
        "keep",
        0,
        Keep {
            rows: [down[0], down[1], down[2]],
            into: Arc::clone(&out),
        },
    );

    let planned = dag.plan();
    assert_eq!(planned.ranks(), workers);
    let devices: Vec<i32> = (0..workers as i32).collect();
    let mut instance = planned.instantiate(&devices);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 2);
    record.result.unwrap();
    out.lock().unwrap().clone()
}

/// Each of the three results, bitwise against the host's.
fn check(got: &[Vec<f64>], workers: usize) {
    let want = reference();
    for (k, name) in ["the vector", "the running sum", "the transpose product"]
        .iter()
        .enumerate()
    {
        assert!(got[k] == want[k], "{name} on {workers} worker(s)");
    }
}

/// Needs a GPU. One worker: every collective a clique of one rank.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_partitioned_product_in_a_loop_on_one_device() {
    check(&run(&Partition::even(N, 1)), 1);
}

/// The slow-case switch (`docs/testing.md`, "Slow cases").
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// The uneven splits a run over `visible` devices tries: two workers, and
/// four, one with no rows, where four are visible.
fn uneven(visible: usize) -> Partition {
    if visible >= 4 {
        Partition::new(&[60, 0, 120, 77])
    } else {
        Partition::new(&[100, 157])
    }
}

/// Needs two GPUs: the plan on four workers where four devices are
/// visible, split unevenly with a worker of no rows, or else on two,
/// bitwise the host's. The family's representative.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_a_partitioned_product_in_a_loop_across_devices() {
    let visible = device_count().unwrap();
    assert!(visible >= 2, "{visible} devices");
    let split = uneven(visible);
    check(&run(&split), split.parts());
}

/// Needs two GPUs. Slow, past the bar with a clique made per run: one
/// worker, two split evenly and unevenly, and four where four are visible,
/// each bitwise the host's and the one worker's, compared in this one
/// process.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_every_split_is_bitwise_one_workers() {
    if fast_tests() {
        return;
    }
    let visible = device_count().unwrap();
    assert!(visible >= 2, "{visible} devices");
    let one = run(&Partition::even(N, 1));
    check(&one, 1);
    let mut splits = vec![Partition::even(N, 2), Partition::new(&[100, 157])];
    if visible >= 4 {
        splits.extend([Partition::even(N, 4), uneven(visible)]);
    }
    for split in &splits {
        let got = run(split);
        let ranges: Vec<_> = (0..split.parts()).map(|w| split.range(w)).collect();
        println!("{} workers over {ranges:?}: bitwise", split.parts());
        check(&got, split.parts());
        assert!(got == one, "{} workers against one", split.parts());
    }
}
