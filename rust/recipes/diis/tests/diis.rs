//! DIIS on a card (tier 4, device 0), each case one run of one plan: the
//! history before a loop, and in the loop's body the pair's upload, the
//! push and its output's readback, the decision reading the output and the
//! singular flag after every push. The pairs are the committed
//! `tests/data/iterates.npy` and `residuals.npy` (written by
//! `tests/data/fixture.py`), the linear map `linear_a.npy` and
//! `linear_b.npy`; the C++ `test/recipes/diis_gpu.cpp` is what the cases
//! mirror.
//!
//! **The bar of the extrapolation, by analysis.** The reference solves the
//! bordered system of the pairs in age order on the host by Gauss-Jordan
//! elimination with partial pivoting and sums `Σ c_i F_i` in that order;
//! the card solves the same system in the ring's rotated order, a
//! symmetric permutation of it, from a Gram its gemvs summed in their own
//! order. Both are backward stable: each Gram entry carries a relative
//! error of at most `len u` (`u = 2⁻⁵³`), the elimination of order
//! `m + 1 <= 5` a few `u` more, so to first order either side's
//! coefficients are within `κ (len + m + 1) u` of the exact ones,
//! relatively, `κ` the bordered matrix's ∞-norm condition number. On the
//! committed pairs `κ` is at most 153.4, at the first extrapolating push
//! (asserted at every push, under [`KAPPA`]), which puts each side within
//! `160 · 42 · 1.1e-16 = 7.5e-13`, and the sum of at most four terms
//! bounded by one adds `4 u`. `1e-12` relative to `|F*|_∞` is the bar: it
//! holds unless the two sides' worst cases both occur, in opposite
//! directions, and a wrong system (a stale Gram entry, a column out of
//! place) misses it by orders of magnitude. The run measured at most
//! `5e-16`.

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fb_core_arena::{
    Bufs, Device, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p, memcpy_h2p, memcpy_p2d,
};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::gemv;
use fb_core_wwr::memset;
use fb_recipes_diis::{Diis, add_diis};
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N;

/// The most the bordered matrix's ∞-norm condition number may be on the
/// pairs a case extrapolates over, for the bar's analysis to hold.
const KAPPA: f64 = 160.0;

/// The committed `name` under `tests/data`, as its shape and its data,
/// column-major (`fb_io_npy`'s order).
fn fixture(name: &str) -> (Vec<usize>, Vec<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    let a = fb_io_npy::read_npy(&path).unwrap();
    (a.shape, a.data)
}

/// The committed pushes, `(iterate, residual)` each, the files' rows cut
/// to `len`.
fn pairs(len: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let (shape, f) = fixture("iterates.npy");
    let (_, e) = fixture("residuals.npy");
    let row = |a: &[f64], k: usize| (0..len).map(|i| a[k + i * shape[0]]).collect();
    (0..shape[0]).map(|k| (row(&f, k), row(&e, k))).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The bordered matrix of `residuals`, row-major, of order `m + 1`.
fn bordered(residuals: &[&[f64]]) -> Vec<f64> {
    let m = residuals.len();
    let dim = m + 1;
    let mut b = vec![-1.0; dim * dim];
    for i in 0..m {
        for j in 0..m {
            b[i * dim + j] = dot(residuals[i], residuals[j]);
        }
    }
    b[m * dim + m] = 0.0;
    b
}

/// `b x = rhs` for each column of `rhs` (row-major, `dim` × `k`), by
/// Gauss-Jordan elimination with partial pivoting.
fn solve(mut b: Vec<f64>, mut rhs: Vec<f64>, dim: usize, k: usize) -> Vec<f64> {
    for col in 0..dim {
        let pivot = (col..dim)
            .max_by(|&r, &s| b[r * dim + col].abs().total_cmp(&b[s * dim + col].abs()))
            .unwrap();
        for j in 0..dim {
            b.swap(col * dim + j, pivot * dim + j);
        }
        for j in 0..k {
            rhs.swap(col * k + j, pivot * k + j);
        }
        let diag = b[col * dim + col];
        for r in (0..dim).filter(|&r| r != col) {
            let factor = b[r * dim + col] / diag;
            for j in col..dim {
                b[r * dim + j] -= factor * b[col * dim + j];
            }
            for j in 0..k {
                rhs[r * k + j] -= factor * rhs[col * k + j];
            }
        }
    }
    for r in 0..dim {
        for j in 0..k {
            rhs[r * k + j] /= b[r * dim + r];
        }
    }
    rhs
}

/// The reference: the bordered system of the pairs in age order solved on
/// the host, `Σ c_i F_i`, and the system's ∞-norm condition number.
fn extrapolate(pairs: &[&(Vec<f64>, Vec<f64>)]) -> (Vec<f64>, f64) {
    let m = pairs.len();
    let dim = m + 1;
    let residuals: Vec<&[f64]> = pairs.iter().map(|p| p.1.as_slice()).collect();
    let b = bordered(&residuals);
    let mut rhs = vec![0.0; dim];
    rhs[m] = -1.0;
    let c = solve(b.clone(), rhs, dim, 1);
    let mut identity = vec![0.0; dim * dim];
    for i in 0..dim {
        identity[i * dim + i] = 1.0;
    }
    let inverse = solve(b.clone(), identity, dim, dim);
    let norm = |a: &[f64]| {
        (0..dim)
            .map(|i| {
                a[i * dim..(i + 1) * dim]
                    .iter()
                    .map(|x| x.abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max)
    };
    let kappa = norm(&b) * norm(&inverse);
    let mut out = vec![0.0; pairs[0].0.len()];
    for (ci, p) in c.iter().zip(pairs) {
        for (o, f) in out.iter_mut().zip(&p.0) {
            *o += ci * f;
        }
    }
    (out, kappa)
}

/// `|got - want|_∞ / |want|_∞`.
fn relative(got: &[f64], want: &[f64]) -> f64 {
    let diff = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max);
    diff / want.iter().map(|w| w.abs()).fold(0.0, f64::max)
}

/// CPU, in the body: the next pair written into two pinned rows.
struct Feed {
    pairs: Arc<Vec<(Vec<f64>, Vec<f64>)>>,
    next: Arc<AtomicUsize>,
}

impl Task for Feed {
    type Rows = (RowId<Pinned, f64>, RowId<Pinned, f64>);
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        let len = self.pairs[0].0.len();
        (h.output("F host", len), h.output("e host", len))
    }

    fn run(
        &self,
        &(f, e): &Self::Rows,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let pair = &self.pairs[self.next.fetch_add(1, Ordering::Relaxed)];
        memcpy_h2p(h, f, &pair.0);
        memcpy_h2p(h, e, &pair.1);
        Ok(())
    }
}

/// GPU, in the body: the pair up to two device rows.
struct Up((RowId<Pinned, f64>, RowId<Pinned, f64>), usize);

impl Task for Up {
    type Rows = (RowId<Device, f64>, RowId<Device, f64>);
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        let (f, e) = self.0;
        h.read(f);
        h.read(e);
        (d.output("F", self.1), d.output("e", self.1))
    }

    fn run(
        &self,
        &(f, e): &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_p2d(&d.take_mut(f), h, self.0.0, stream)?;
        memcpy_p2d(&d.take_mut(e), h, self.0.1, stream)
    }
}

/// GPU, in the body: a device row down to a pinned one.
struct Down(RowId<Device, f64>, usize);

impl Task for Down {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("out host", self.1)
    }

    fn run(
        &self,
        &out: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(h, out, d.get(self.0), stream)
    }
}

/// What the decision saw after each push: the output and the flag.
type Seen = Arc<Mutex<Vec<(Vec<f64>, i32)>>>;

/// The decision: the output and the flag kept, the history reset after
/// push `reset_after` if given, and the loop ended after `pushes`.
struct Record {
    out: RowId<Pinned, f64>,
    singular: RowId<Pinned, i32>,
    seen: Seen,
    diis: Diis,
    reset_after: Option<usize>,
    pushes: usize,
}

impl Decision for Record {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>) {
        h.read(self.out);
        h.read(self.singular);
    }

    fn decide(&self, _: &(), h: &Bufs<'_, Pinned>, iteration: usize) -> Result<ControlFlow<()>> {
        let seen = (h.get(self.out).to_vec(), h.get(self.singular)[0]);
        self.seen.lock().unwrap().push(seen);
        if self.reset_after == Some(iteration) {
            self.diis.reset();
        }
        Ok(if iteration + 1 == self.pushes {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }
}

/// `pairs` pushed in order, one per iteration of one run, into a history
/// of `cap`, reset after push `reset_after` if given: what the decision saw
/// after each.
fn run_pushes(
    pairs: Vec<(Vec<f64>, Vec<f64>)>,
    cap: usize,
    reset_after: Option<usize>,
) -> Vec<(Vec<f64>, i32)> {
    let len = pairs[0].0.len();
    let pushes = pairs.len();
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let diis = add_diis(&mut dag, 0, "F*", len, cap).unwrap();
    let body = dag.len();
    let host = dag.add(
        "feed",
        0,
        Feed {
            pairs: Arc::new(pairs),
            next: Arc::new(AtomicUsize::new(0)),
        },
    );
    let (f, e) = dag.add("up", 0, Up(host, len));
    let (out, singular) = diis.add_push(&mut dag, f, e);
    let out = dag.add("down", 0, Down(out, len));
    dag.add_loop(
        "record",
        0,
        body,
        pushes,
        Record {
            out,
            singular,
            seen: Arc::clone(&seen),
            diis,
            reset_after,
            pushes,
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    seen.lock().unwrap().clone()
}

/// Needs a GPU. Eleven pushes into a ring of four, so the ring fills at the
/// fourth and then wraps nearly twice: from the second push on, the output
/// is the host's age-ordered extrapolation over the last four pairs within
/// `1e-12` relative, and no subspace is singular. The first push leaves
/// the output the iterate itself, bit for bit (the C++ left its in-place
/// output untouched), with the flag 0.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_extrapolation_is_the_hosts_through_the_ring_wrap() {
    const CAP: usize = 4;
    let pairs = pairs(37);
    let seen = run_pushes(pairs.clone(), CAP, None);
    assert_eq!(seen.len(), pairs.len());
    assert_eq!(seen[0], (pairs[0].0.clone(), 0), "the first push");
    for (step, (out, singular)) in seen.iter().enumerate().skip(1) {
        assert_eq!(*singular, 0, "push {step} was singular");
        let held: Vec<_> = pairs[(step + 1).saturating_sub(CAP)..=step]
            .iter()
            .collect();
        let (want, kappa) = extrapolate(&held);
        assert!(kappa <= KAPPA, "push {step}: κ = {kappa}");
        let error = relative(out, &want);
        assert!(error <= 1e-12, "push {step}: relative error {error:e}");
    }
}

/// Needs a GPU. Two equal residuals make the Gram rank one, so the
/// bordered system is singular: the flag is 1 and the output is the newest
/// iterate, bit for bit.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_singular_subspace_falls_back_to_the_newest_iterate() {
    let mut pairs = pairs(16);
    pairs.truncate(2);
    pairs[1].1 = pairs[0].1.clone();
    let seen = run_pushes(pairs.clone(), 4, None);
    assert_eq!(seen[1], (pairs[1].0.clone(), 1));
}

/// Needs a GPU. Five pushes into a ring of three, then a reset from the
/// decision: the next push is a first one again, its output the iterate,
/// and the one after extrapolates over the two pairs since the reset and
/// no others, within the bar. The Gram entries from before are never read.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_reset_drops_the_subspace_and_keeps_the_rows() {
    let pairs = pairs(37);
    let pairs = pairs[..7].to_vec();
    let seen = run_pushes(pairs.clone(), 3, Some(4));
    assert_eq!(
        seen[5],
        (pairs[5].0.clone(), 0),
        "the first push after the reset"
    );
    let (want, kappa) = extrapolate(&[&pairs[5], &pairs[6]]);
    assert!(kappa <= KAPPA, "κ = {kappa}");
    let error = relative(&seen[6].0, &want);
    assert!(error <= 1e-12, "relative error {error:e}");
    assert_eq!(seen[6].1, 0);
}

/// The order of the linear map.
const N: usize = 6;

/// [`Upload`]'s rows: the map, `A - I`, `b` and `x`, the last carried by
/// the loop.
#[derive(Clone, Copy)]
struct Fixed {
    a: RowId<Device, f64>,
    a_minus_i: RowId<Device, f64>,
    b: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

/// CPU, before the loop: the map, `A - I` and `b` into pinned rows, the
/// matrices column-major.
struct Start;

impl Task for Start {
    type Rows = [RowId<Pinned, f64>; 3];
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        [
            h.output("A host", N * N),
            h.output("A - I host", N * N),
            h.output("b host", N),
        ]
    }

    fn run(
        &self,
        &[a, a_minus_i, b]: &Self::Rows,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let (_, column_major) = fixture("linear_a.npy");
        let shifted: Vec<f64> = column_major
            .iter()
            .enumerate()
            .map(|(k, v)| if k % (N + 1) == 0 { v - 1.0 } else { *v })
            .collect();
        memcpy_h2p(h, a, &column_major);
        memcpy_h2p(h, a_minus_i, &shifted);
        memcpy_h2p(h, b, &fixture("linear_b.npy").1);
        Ok(())
    }
}

/// GPU, before the loop: [`Fixed`]'s rows.
struct Upload([RowId<Pinned, f64>; 3]);

impl Task for Upload {
    type Rows = Fixed;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Fixed {
        for row in self.0 {
            h.read(row);
        }
        Fixed {
            a: d.output("A", N * N),
            a_minus_i: d.output("A - I", N * N),
            b: d.output("b", N),
            x: d.output("x", N),
        }
    }

    fn run(
        &self,
        rows: &Fixed,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let [a, a_minus_i, b] = self.0;
        memcpy_p2d(&d.take_mut(rows.a), h, a, stream)?;
        memcpy_p2d(&d.take_mut(rows.a_minus_i), h, a_minus_i, stream)?;
        memcpy_p2d(&d.take_mut(rows.b), h, b, stream)?;
        memset::zero(&mut d.take_mut(rows.x), stream)
    }
}

/// GPU, in the body: `F = A x + b` and `e = F - x = (A - I) x + b`, and
/// `e` down to a pinned row for the decision.
struct Apply(Fixed);

impl Task for Apply {
    type Rows = (RowId<Device, f64>, RowId<Device, f64>, RowId<Pinned, f64>);
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0.a);
        d.read(self.0.a_minus_i);
        d.read(self.0.b);
        d.read(self.0.x);
        (d.output("F", N), d.output("e", N), h.output("e host", N))
    }

    fn run(
        &self,
        &(f, e, e_host): &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let n = N as i32;
        let (mut f, mut e) = (d.take_mut(f), d.take_mut(e));
        for (out, map) in [(&mut f, self.0.a), (&mut e, self.0.a_minus_i)] {
            memcpy_d2d(out, d.get(self.0.b), stream)?;
            gemv(
                stream.blas(),
                WWR_BLAS_OP_N,
                n,
                n,
                1.0,
                &d.get(map),
                n,
                &d.get(self.0.x),
                1,
                1.0,
                out,
                1,
            )?;
        }
        memcpy_d2p(h, e_host, &e, stream)
    }
}

/// GPU, the body's last device task: `x = F*`, the carried row's writer.
struct CarryX {
    out: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

impl Task for CarryX {
    type Rows = ();
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.out);
        d.write(self.x);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2d(&d.take_mut(self.x), d.get(self.out), stream)
    }
}

/// The decision: stops once `|e|_2` of the iterate the iteration started
/// from is below `1e-10`, keeping each norm.
struct Converged {
    e: RowId<Pinned, f64>,
    norms: Arc<Mutex<Vec<f64>>>,
}

impl Decision for Converged {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>) {
        h.read(self.e);
    }

    fn decide(&self, _: &(), h: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
        let e = h.get(self.e);
        let norm = dot(e, e).sqrt();
        self.norms.lock().unwrap().push(norm);
        Ok(if norm < 1e-10 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }
}

/// Needs a GPU. `x ← A x + b`, `F = A x + b` and `e = F - x` pushed each
/// iteration and `x` the push's output, carried, from `x = 0`, the history
/// keeping every pair. On a linear map DIIS is GMRES, so the residual of
/// the iterate reaches zero, to rounding, within `N + 1` pushes, where the
/// plain iteration's spectral radius of 0.9 leaves it near `0.9⁷ ≈ 0.48`
/// of where it started. The loop's bound is `N + 2` iterations: the
/// iteration after the `N + 1`-th push is the last whose residual may stop
/// it, so a slower convergence fails the run with `MaxIterationsExceeded`.
/// The bar, `1e-10` on `|e|_2`, is the C++ test's: GMRES's exact answer
/// after `N` steps is the fixed point, and what is left is the rounding of
/// a solve of order 7 with `|x| < 10`, near `1e-14`.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_linear_fixed_point_converges_within_n_plus_one_pushes() {
    let norms = Arc::new(Mutex::new(Vec::new()));
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let host = dag.add("start", 0, Start);
    let fixed = dag.add("upload", 0, Upload(host));
    let diis = add_diis(&mut dag, 0, "x*", N, N + 2).unwrap();
    let body = dag.len();
    let x = fixed.x;
    let (f, e, e_host) = dag.add("A x + b", 0, Apply(fixed));
    let (out, _) = diis.add_push(&mut dag, f, e);
    dag.add("x = x*", 0, CarryX { out, x });
    dag.add_loop(
        "converged",
        0,
        body,
        N + 2,
        Converged {
            e: e_host,
            norms: Arc::clone(&norms),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    let norms = norms.lock().unwrap();
    record
        .result
        .unwrap_or_else(|e| panic!("{e}; the residuals were {norms:?}"));
    assert!(*norms.last().unwrap() < 1e-10, "{norms:?}");
}
