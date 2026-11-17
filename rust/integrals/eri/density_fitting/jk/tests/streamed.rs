//! The streamed pass on a card (tier 4, device 0), one run per case: a
//! synthetic packed B in a heap row, `naux = 150` aux functions over the
//! spherical packing of one s, p, d and f basis on three centres (22
//! functions), two symmetric densities D₁, D₂ and two factors C₁ (22 × 5),
//! C₂ (22 × 3), all from fixed seeds, and [`add_streamed_jk`]'s J and K
//! against the resident [`add_df_coulomb`] and [`add_df_exchange`] over
//! the same B unpacked to the square, its dropped pairs' columns zero.
//!
//! **The bar, by analysis, one run.** The two compute the same sums,
//! `Σ_Q B_Q,mn (Σ_ls B_Q,ls D_ls)` and `2 Σ_Q (B_Q C)(B_Q C)ᵀ`, in another
//! order: the streamed one cuts Q into batches and adds a batch's partial
//! sums into the accumulator, and sums J over packed columns with
//! weights of 2 where the resident one adds `(m, n)` and `(n, m)`. Each
//! element is a dot product of at most `naux nao² ≈ 7·10⁴` terms of mixed
//! sign, whose forward error in either order is at most `γ_k Σ|terms|`
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//! §3.1); the GEMMs' blocked sums keep the effective `k` near
//! `√k`-growth, and `experimental/gemm_host_offload` measured 6e-14 of the
//! largest element for exactly this cut. The bar is the issue's
//! `1e-12 · max |ref|`, every element.
//!
//! One pass serving two J and two K is held **bitwise** to two passes of
//! one J and one K each: each contraction's calls are the same calls on
//! the same batches in the same order, so nothing but a misplaced row can
//! move a bit. K's `(m, n)` and `(n, m)` are the same bits.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, Decision, Plan, Room};
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_jk::{
    Contraction, add_df_coulomb, add_df_exchange, add_streamed_jk,
};
use fb_integrals_eri_density_fitting_pair_map::{Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::add_packed_tables;

/// The fitting set's size: two full batches of 64 and a ragged last one
/// of 22 at the widest batch.
const NAUX: usize = 150;
/// The two factors' columns.
const NOCC: [usize; 2] = [5, 3];
/// The bar's multiple of `max |ref|`.
const BAR: f64 = 1e-12;

/// `n` values of mixed sign and magnitude from `seed`.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// A symmetric `n × n` matrix from `seed`.
fn symmetric(n: usize, seed: u64) -> Vec<f64> {
    let v = values(n * n, seed);
    let mut a = vec![0.0; n * n];
    for m in 0..n {
        for k in 0..=m {
            a[m * n + k] = v[m * n + k];
            a[k * n + m] = v[m * n + k];
        }
    }
    a
}

/// The basis's map: every pair kept, or every fifth pair dropped, a
/// diagonal one among them.
fn map(screened: bool) -> PairMap {
    let at = |x: f64, z: f64| Double3 { x, y: 0.25 * x, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(0, &[3.0, 0.5], &[0.6, 0.4], at(0.0, 0.0)),
        Shell::zipped(2, &[0.8], &[1.0], at(0.0, 0.0)),
        Shell::zipped(1, &[1.2], &[1.0], at(1.4, 0.3)),
        Shell::zipped(3, &[0.6], &[1.0], at(1.4, 0.3)),
        Shell::zipped(0, &[0.9], &[1.0], at(-1.1, 1.0)),
        Shell::zipped(2, &[1.5], &[1.0], at(-1.1, 1.0)),
    ]);
    let tables = record_tables(&basis);
    let bounds: Vec<f64> = (0..tables.n_pairs())
        .map(|p| if screened && p % 5 == 2 { 1e-12 } else { 1.0 })
        .collect();
    let map = PairMap::new(&tables, Some(&bounds), 1e-10);
    assert_eq!(map.pairs().len() < tables.n_pairs(), screened);
    map
}

/// The synthetic inputs over one packing.
struct Inputs {
    packing: Packing,
    /// B packed, `column + n_columns Q`.
    packed: Vec<f64>,
    /// B square, the resident layout `Q + naux (n + nao m)`, a dropped
    /// pair's elements zero.
    square: Vec<f64>,
    d: [Vec<f64>; 2],
    c: [Vec<f64>; 2],
}

fn inputs(packing: &Packing) -> Inputs {
    let (nao, nc) = (packing.nao(), packing.n_columns());
    // Random columns, each B_Q then made symmetric as a fitted factor is,
    // a diagonal block's (a, b) and (b, a) columns one value: its unpack
    // (from the lower triangle) packed again.
    let mut packed = values(NAUX * nc, 1);
    let mut square = vec![0.0; NAUX * nao * nao];
    let mut bq = vec![0.0; nao * nao];
    for (q, column) in packed.chunks_mut(nc).enumerate() {
        packing.unpack(column, &mut bq);
        packing.pack(&bq, column);
        for (mn, &v) in bq.iter().enumerate() {
            square[q + NAUX * mn] = v;
        }
    }
    Inputs {
        packing: packing.clone(),
        packed,
        square,
        d: [symmetric(nao, 2), symmetric(nao, 3)],
        c: [values(nao * NOCC[0], 4), values(nao * NOCC[1], 5)],
    }
}

/// CPU: the packed B written into a heap row.
struct Source(Vec<f64>);

impl CpuTask for Source {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("B", self.0.len())
    }

    fn run(&self, row: &Self::Rows, _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        heap.take_mut(*row).copy_from_slice(&self.0);
        Ok(())
    }
}

/// CPU: one input written into a pinned row.
struct Stage {
    name: &'static str,
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output(self.name, self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// CPU: the downloaded results read out, in order.
struct Collect {
    rows: Vec<RowId<Pinned, f64>>,
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        for &row in &self.rows {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap() = self.rows.iter().map(|&r| host.get(r).to_vec()).collect();
        Ok(())
    }
}

/// The decision of a loop of a fixed count of iterations.
struct Times(usize);

impl Decision for Times {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {}

    fn decide(
        &self,
        _: &(),
        _: &Bufs<'_, Pinned>,
        _: &Bufs<'_, Heap>,
        iteration: usize,
    ) -> Result<ControlFlow<()>> {
        Ok(if iteration + 1 == self.0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }
}

/// What a run returns: the batch width it was planned at, and each
/// result, `nao²` doubles: the one pass's J₁, K₁, J₂, K₂, then the two
/// passes' J₁, K₁ and J₂, K₂, then the resident tasks' J₁, K₁, J₂, K₂.
struct Ran {
    width: usize,
    results: Vec<Vec<f64>>,
}

/// Plan the passes and the resident tasks over `inp` on device 0 in a
/// device room of `room` bytes, the pinned room unbounded (none:
/// [`Dag::plan`], the widest batch), the passes in a decision loop of
/// `iterations` iterations when that is above 0, the results collected
/// into `out`; return the plan and the one pass's batch width.
fn plan(
    inp: &Inputs,
    room: Option<usize>,
    iterations: usize,
    out: &Arc<Mutex<Vec<Vec<f64>>>>,
) -> (Plan<'static>, usize) {
    let (nao, nc) = (inp.packing.nao(), inp.packing.n_columns());
    let declare = |dag: &mut Dag<'static>| {
        dag.blas_workspace(0);
        let b = dag.add_cpu("source", 0, Source(inp.packed.clone()));
        let p = inp.packing.clone();
        let tables = add_packed_tables(dag, 0, nao, nc, move |r, c, w| p.write_tables(r, c, w));
        let mut up = |name: &'static str, data: &[f64]| {
            let len = data.len();
            let data = data.to_vec();
            let staged = dag.add_cpu(format!("stage {name}"), 0, Stage { name, data });
            dag.upload(0, staged, name, len)
        };
        let square = up("B square", &inp.square);
        let d = [up("D1", &inp.d[0]), up("D2", &inp.d[1])];
        let c = [up("C1", &inp.c[0]), up("C2", &inp.c[1])];
        let set = |i: usize| {
            [
                Contraction::Coulomb(d[i]),
                Contraction::Exchange {
                    c: c[i],
                    nocc: NOCC[i],
                },
            ]
        };

        let body = dag.len();
        let both = [set(0), set(1)].concat();
        let mut rows = add_streamed_jk(dag, 0, &inp.packing, tables, NAUX, b, &both).unwrap();
        for i in 0..2 {
            rows.extend(add_streamed_jk(dag, 0, &inp.packing, tables, NAUX, b, &set(i)).unwrap());
        }
        if iterations > 0 {
            dag.add_loop("again", 0, body, iterations, Times(iterations));
        }
        for i in 0..2 {
            rows.push(add_df_coulomb(dag, 0, nao, NAUX, square, d[i]));
            rows.push(add_df_exchange(dag, 0, nao, NAUX, NOCC[i], square, c[i]).unwrap());
        }

        let named = rows
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, format!("result {i}"), nao * nao));
        let down = dag.download_rows(0, "results", named);
        dag.add_cpu(
            "collect",
            0,
            Collect {
                rows: down,
                out: Arc::clone(out),
            },
        );
        Ok(())
    };
    let planned = match room {
        None => {
            let mut dag = Dag::new();
            declare(&mut dag).unwrap();
            dag.plan()
        }
        Some(device) => {
            let room = Room {
                device,
                pinned: usize::MAX,
            };
            Dag::plan_within(|_| Ok(room), declare).unwrap().0
        }
    };
    // The first width line is the one pass's.
    let width = planned
        .to_string()
        .lines()
        .find_map(|l| l.strip_prefix("width df jk batch: "))
        .and_then(|l| l.split(' ').next())
        .and_then(|w| w.parse().ok())
        .expect("the plan has the batch's width line");
    (planned, width)
}

/// [`plan`], run once.
fn run(inp: &Inputs, room: Option<usize>, iterations: usize) -> Ran {
    let out = Arc::new(Mutex::new(Vec::new()));
    let (planned, width) = plan(inp, room, iterations, &out);
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let results = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(results.len(), 12, "the collect did not run");
    Ran { width, results }
}

/// Every streamed J and K within `BAR · max |ref|` of the resident one,
/// each J and K exactly symmetric, and, if `bitwise`, the one pass bitwise
/// the two.
fn check(ran: &Ran, nao: usize, bitwise: bool) {
    let r = &ran.results;
    let names = ["J1", "K1", "J2", "K2"];
    for (i, name) in names.iter().enumerate() {
        let (got, want) = (&r[i], &r[8 + i]);
        let max = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut miss = 0.0_f64;
        for (g, w) in got.iter().zip(want) {
            // Not f64::max, which would drop a NaN.
            let d = (g - w).abs();
            if d.is_nan() || d > miss {
                miss = d;
            }
        }
        println!(
            "width {}: {name} max |Δ| = {miss:e} of max |ref| {max:e}",
            ran.width
        );
        assert!(
            !miss.is_nan() && miss <= BAR * max,
            "{name}: max |Δ| = {miss:e} > {BAR:e} · {max:e}"
        );
        for (j, (a, b)) in got.iter().zip(&r[4 + i]).enumerate().filter(|_| bitwise) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name}: element {j} of the one pass {a:e}, of its own pass {b:e}"
            );
        }
        for n in 0..nao {
            for m in 0..n {
                let (lower, upper) = (got[n + nao * m], got[m + nao * n]);
                assert_eq!(
                    lower.to_bits(),
                    upper.to_bits(),
                    "{name}({n}, {m}) = {lower:e} but ({m}, {n}) = {upper:e}"
                );
            }
        }
    }
}

/// Every pair kept, at the widest batch (two full batches of 64 in the
/// ring and the ragged last one of 22): J and K within the bar of the
/// resident tasks', exactly symmetric, the one pass bitwise the two.
/// Catches a batch read at the wrong offset of the heap row, a packed
/// weight or a diagonal block counted twice, the gather's slab read in
/// the wrong order by the batched gemm, an accumulator overwritten rather
/// than added to, the ragged last batch lost or read past its 22
/// functions, and two contractions of one pass sharing a row they should
/// not. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn every_pair_kept_is_the_resident_within_the_bar() {
    let map = map(false);
    let inp = inputs(map.sphe());
    let ran = run(&inp, None, 0);
    assert_eq!(ran.width, 64);
    check(&ran, inp.packing.nao(), true);
}

/// A screened map, a diagonal pair dropped among others: J and K those of
/// the resident tasks over B with the dropped pairs' columns zero, at the
/// bar. Catches a dropped pair's elements left as the slab's previous
/// batch's or J unpacked over a pair it does not hold. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_screened_map_is_the_resident_with_dropped_pairs_zero() {
    let map = map(true);
    let inp = inputs(map.sphe());
    let ran = run(&inp, None, 0);
    check(&ran, inp.packing.nao(), true);
}

/// The screened map planned in a room that narrows the one pass's batch
/// to 16 (the plan's `warn`, `tests/plan.rs`), so the ring turns over many
/// times: the same bar. Catches a ring copy reused before its batch is
/// contracted. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_narrowed_batch_meets_the_same_bar() {
    let map = map(true);
    let inp = inputs(map.sphe());
    let ran = run(&inp, Some(narrowed_room(&inp)), 0);
    assert_eq!(ran.width, 16);
    // The two passes' parts are bisected on their own, to widths of their
    // own, so they sum in other batches: the bar alone.
    check(&ran, inp.packing.nao(), false);
}

/// The largest room in which [`plan`]'s one pass is narrowed below 32,
/// to 16: found by bisection, planning alone.
fn narrowed_room(inp: &Inputs) -> usize {
    let out = Arc::new(Mutex::new(Vec::new()));
    let width_in = |room: usize| plan(inp, Some(room), 0, &out).1;
    // `high`'s width is at least 32, `low`'s below.
    let (mut low, mut high) = (0, 1 << 30);
    assert!(width_in(high) >= 32);
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if width_in(mid) >= 32 {
            high = mid;
        } else {
            low = mid;
        }
    }
    assert_eq!(width_in(low), 16);
    low
}

/// The passes in a decision loop of three iterations, as the SCF holds
/// them: the last iteration's J and K bitwise the top level's, and at the
/// bar. Catches a batch's range taken from the decision loop's iteration
/// rather than its trip, and a pass whose accumulators carry one
/// iteration's sums into the next. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_pass_in_a_decision_loop_is_the_same_every_iteration() {
    let map = map(true);
    let inp = inputs(map.sphe());
    let top = run(&inp, None, 0);
    let looped = run(&inp, None, 3);
    check(&looped, inp.packing.nao(), true);
    for (i, (a, b)) in top.results.iter().zip(&looped.results).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "result {i} differs in the loop"
        );
    }
}
