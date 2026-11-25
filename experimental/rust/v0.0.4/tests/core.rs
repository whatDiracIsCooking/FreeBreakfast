//! The core, device-free: the tests the design lists for item 1 of its
//! order of work, each as the closure model checks it, plus the lease.
//!
//! Five fake steps over a small bus of their own, on the plain heap
//! ([`Backing::Host`]): `a`; `b` and `c`, which read `a`; `d`, which reads
//! `b` and `c`; and `e`, which reads nothing and so is ordered against no
//! one. Their sizes come from the bus, so one graph serves every case. The
//! packer is also held on seeded random DAGs, directly.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fb_core_error_handling::{ErrorKind, Result, Status};
use fb_v04::budget::{
    Backing, Budget, Cap, Caps, DeviceBuf, HeapSlab, Kind, Layout, Lease, Ledger,
};
use fb_v04::flow::{Bus, Dag, Placement, Plan, Slot, Step, View};
use fb_v04::pack::{self, Region};
use rustflow::Executor;

const MIB: u64 = 1 << 20;
const N: usize = 5;
const NAMES: [&str; N] = ["a", "b", "c", "d", "e"];

/// What the fake steps are told to do.
#[derive(Clone)]
struct Spec {
    /// Each step's device `Output` bytes, device scratch bytes and heap
    /// bytes.
    device: [u64; N],
    scratch: [u64; N],
    heap: [u64; N],
    /// The step whose `run` fails, if any.
    fails: Option<usize>,
    /// The step whose `run` carves more scratch than it declared, if any.
    overreaches: Option<usize>,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            device: [MIB; N],
            scratch: [4 * MIB, 3 * MIB, 2 * MIB, 6 * MIB, MIB],
            heap: [0; N],
            fails: None,
            overreaches: None,
        }
    }
}

struct Out {
    device: DeviceBuf<f64>,
    heap: HeapSlab<f64>,
    /// How many times `run` ran.
    runs: usize,
    /// The first double of the step's scratch as each `run` found it.
    found: Vec<f64>,
    /// The address of the step's scratch on each `run`.
    address: Vec<usize>,
}

struct TestBus {
    spec: Spec,
    ledger: Ledger,
    slots: [Slot<Out, ()>; N],
    /// Steps inside `run` right now, and the most there ever were.
    running: AtomicUsize,
    most: AtomicUsize,
}

impl Bus for TestBus {
    type Shared = TestBus;
    type Lane = ();
    type Hold<'a> = ();

    fn shared(&self) -> &TestBus {
        self
    }
    fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    fn make_lane(&self, _: &'static str, _: Placement, _: usize) -> Result<()> {
        Ok(())
    }
    fn acquire<'a>(
        &'a self,
        _: &'static str,
        _: Placement,
        _: &mut (),
        lease: &Lease<'_>,
    ) -> Result<()> {
        if self.ledger.nan_fill() {
            lease.fill_host();
        }
        Ok(())
    }
    fn release(&self, _: &'static str, _: &(), _: ()) -> Status {
        Ok(())
    }
}

/// The one layout function of a fake step: its scratch as doubles.
fn layout(bytes: u64) -> (Layout, fb_v04::budget::Span<f64>) {
    let mut l = Layout::new();
    let span = l.span::<f64>(bytes as usize / 8);
    (l, span)
}

fn budget_of(i: usize, bus: &TestBus, budget: &Budget<'_>) -> Result<Out> {
    let spec = &bus.spec;
    budget.scratch(0, Kind::Device, layout(spec.scratch[i]).0.bytes());
    Ok(Out {
        device: budget.device(0, spec.device[i] as usize / 8),
        heap: budget.heap(spec.heap[i] as usize / 8),
        runs: 0,
        found: Vec::new(),
        address: Vec::new(),
    })
}

fn run_of(i: usize, out: &mut Out, bus: &TestBus, scratch: &Lease<'_>) -> Status {
    let now = bus.running.fetch_add(1, Ordering::AcqRel) + 1;
    bus.most.fetch_max(now, Ordering::AcqRel);
    let spec = &bus.spec;
    let grow = if spec.overreaches == Some(i) { 64 } else { 0 };
    let mut mine = scratch.device(0, layout(spec.scratch[i] + grow).1);
    let bytes = mine.host().expect("a host-backed plan");
    out.address.push(bytes.as_ptr() as usize);
    out.found.push(bytes.first().copied().unwrap_or(0.0));
    // Fill the region with the step's own value, give an unordered step
    // time to stomp on it, and look again.
    let value = (i + 1) as f64;
    bytes.fill(value);
    std::thread::sleep(std::time::Duration::from_millis(2));
    let intact = bytes.iter().all(|&x| x == value);
    out.device.host().expect("host-backed").fill(value);
    out.runs += 1;
    bus.running.fetch_sub(1, Ordering::AcqRel);
    if spec.fails == Some(i) {
        return Err(fb_core_error_handling::Error::numerical("told to fail"));
    }
    assert!(
        intact,
        "step '{}' had its scratch written under it",
        NAMES[i]
    );
    Ok(())
}

macro_rules! fake {
    ($name:ident, $index:expr, $reads:ty) => {
        #[derive(Clone, Copy)]
        struct $name;
        impl Step<TestBus> for $name {
            const NAME: &'static str = NAMES[$index];
            type Output = Out;
            type Reads = $reads;
            fn slot(bus: &TestBus) -> &Slot<Out, ()> {
                &bus.slots[$index]
            }
            fn mem_budget(bus: &TestBus, budget: &Budget<'_>) -> Result<Out> {
                budget_of($index, bus, budget)
            }
            fn run(
                out: &mut Out,
                view: &View<'_, TestBus, Self>,
                _: &mut (),
                _: &mut (),
                scratch: &Lease<'_>,
            ) -> Status {
                run_of($index, out, view.shared(), scratch)
            }
        }
    };
}

fake!(A, 0, ());
fake!(B, 1, (A,));
fake!(C, 2, (A,));
fake!(D, 3, (B, C));
fake!(E, 4, ());

fn bus(spec: Spec, device_cap: u64, host_cap: u64, nan_fill: bool) -> Arc<TestBus> {
    let cap = |used| Cap {
        written: Some(used),
        used,
    };
    Arc::new(TestBus {
        spec,
        ledger: Ledger::new(
            Caps {
                device: vec![cap(device_cap)],
                host: cap(host_cap),
            },
            Backing::Host,
            vec![0],
            nan_fill,
        ),
        slots: std::array::from_fn(|_| Slot::new()),
        running: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    })
}

/// The plan with the steps added in `order`.
fn plan_in(bus: Arc<TestBus>, order: [usize; N]) -> Plan<TestBus> {
    let mut dag = Dag::new();
    for i in order {
        match i {
            0 => dag.add(A),
            1 => dag.add(B),
            2 => dag.add(C),
            3 => dag.add(D),
            _ => dag.add(E),
        };
    }
    dag.build(bus).expect("the steps close")
}

fn plan(bus: Arc<TestBus>) -> Plan<TestBus> {
    plan_in(bus, [0, 1, 2, 3, 4])
}

const ROOMY: u64 = 1 << 40;

/// A small generator, seeded: the tests are the same every run.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % below
    }
}

/// A random DAG of `n` steps, each reading some earlier ones, the steps
/// then relabelled so index order is not a topological order.
fn random_dag(rng: &mut Lcg, n: usize) -> Vec<Vec<usize>> {
    let mut label: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        label.swap(i, rng.next(i as u64 + 1) as usize);
    }
    let mut preds = vec![Vec::new(); n];
    for i in 1..n {
        for j in 0..i {
            if rng.next(100) < 25 {
                preds[label[i]].push(label[j]);
            }
        }
    }
    preds
}

/// Reachability by a plain depth-first search.
fn reaches(preds: &[Vec<usize>], from: usize, to: usize) -> bool {
    let mut stack = vec![to];
    let mut seen = vec![false; preds.len()];
    while let Some(at) = stack.pop() {
        for &p in &preds[at] {
            if p == from {
                return true;
            }
            if !std::mem::replace(&mut seen[p], true) {
                stack.push(p);
            }
        }
    }
    false
}

/// Step `i`'s name, `s00` to `s59`.
fn step_name(i: usize) -> &'static str {
    static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| {
        (0..60)
            .map(|i| &*Box::leak(format!("s{i:02}").into_boxed_str()))
            .collect()
    })[i]
}

fn random_regions(rng: &mut Lcg, n: usize) -> Vec<Region> {
    (0..n)
        .filter_map(|step| {
            (rng.next(100) < 70).then(|| Region {
                step,
                name: step_name(step),
                bytes: 256 * (1 + rng.next(40)),
            })
        })
        .collect()
}

#[test]
fn the_closure_equals_reachability_by_a_plain_search() {
    let mut rng = Lcg(1);
    for case in 0..1000 {
        let n = 1 + (case % 60);
        let preds = random_dag(&mut rng, n);
        let reach = pack::closure(&preds);
        for a in 0..n {
            for b in 0..n {
                assert_eq!(
                    reach[a].has(b),
                    a != b && reaches(&preds, a, b),
                    "case {case}: {a} -> {b}"
                );
            }
        }
    }
}

#[test]
fn every_pair_of_overlapping_regions_is_ordered_by_the_dag() {
    let mut rng = Lcg(2);
    let (mut at_bound, mut cases, mut worst) = (0, 0, 1.0f64);
    for case in 0..3000 {
        let n = 2 + (case % 15);
        let preds = random_dag(&mut rng, n);
        let regions = random_regions(&mut rng, n);
        if regions.is_empty() {
            continue;
        }
        let reach = pack::closure(&preds);
        let packed = pack::pack(&reach, &regions);
        for (i, a) in regions.iter().enumerate() {
            for (j, b) in regions.iter().enumerate().skip(i + 1) {
                let (a0, b0) = (packed.offsets[i], packed.offsets[j]);
                let overlap = a0 < b0 + b.bytes && b0 < a0 + a.bytes;
                assert!(
                    !overlap || reaches(&preds, a.step, b.step) || reaches(&preds, b.step, a.step),
                    "case {case}: unordered steps {} and {} share bytes",
                    a.step,
                    b.step
                );
            }
        }
        // largest single <= bound <= packed <= sum.
        let largest = regions.iter().map(|r| r.bytes).max().unwrap();
        assert!(largest <= packed.bound && packed.bound <= packed.peak);
        assert!(packed.peak <= packed.sum, "case {case}");
        // The bound's set is an antichain that sums to it.
        let set = &packed.bound_set;
        assert_eq!(
            set.iter().map(|&i| regions[i].bytes).sum::<u64>(),
            packed.bound
        );
        for (x, &i) in set.iter().enumerate() {
            for &j in &set[x + 1..] {
                assert!(!pack::ordered(&reach, regions[i].step, regions[j].step));
            }
        }
        cases += 1;
        at_bound += usize::from(packed.peak == packed.bound);
        worst = worst.max(packed.peak as f64 / packed.bound as f64);
    }
    println!(
        "packing: {at_bound} of {cases} random DAGs at the lower bound, worst {worst:.3}x the bound"
    );
}

#[test]
fn the_lower_bound_is_the_heaviest_antichain_by_brute_force() {
    let mut rng = Lcg(3);
    for _ in 0..500 {
        let n = 2 + rng.next(9) as usize;
        let preds = random_dag(&mut rng, n);
        let regions = random_regions(&mut rng, n);
        let reach = pack::closure(&preds);
        let mut best = 0;
        for mask in 0u32..1 << regions.len() {
            let chosen: Vec<&Region> = (0..regions.len())
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| &regions[i])
                .collect();
            let antichain = chosen.iter().enumerate().all(|(x, a)| {
                chosen[x + 1..]
                    .iter()
                    .all(|b| !pack::ordered(&reach, a.step, b.step))
            });
            if antichain {
                best = best.max(chosen.iter().map(|r| r.bytes).sum());
            }
        }
        assert_eq!(pack::lower_bound(&reach, &regions).0, best);
    }
}

#[test]
fn offsets_do_not_depend_on_declaration_order() {
    // The packer alone: the regions given in every order of a shuffle.
    let mut rng = Lcg(4);
    for _ in 0..200 {
        let n = 2 + rng.next(14) as usize;
        let preds = random_dag(&mut rng, n);
        let regions = random_regions(&mut rng, n);
        let reach = pack::closure(&preds);
        let first = pack::pack(&reach, &regions);
        let mut order: Vec<usize> = (0..regions.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.next(i as u64 + 1) as usize);
        }
        let shuffled: Vec<Region> = order.iter().map(|&i| regions[i].clone()).collect();
        let second = pack::pack(&reach, &shuffled);
        for (at, &i) in order.iter().enumerate() {
            assert_eq!(second.offsets[at], first.offsets[i]);
        }
        assert_eq!((second.peak, second.bound), (first.peak, first.bound));
    }
    // And through the ledger: the steps added to the DAG in other orders,
    // budgeted on several workers.
    let reference = {
        let plan = plan(bus(Spec::default(), ROOMY, ROOMY, false));
        plan.budget(&Executor::new(1)).unwrap();
        (plan.bus().ledger.regions(), plan.bus().ledger.table())
    };
    for order in [
        [4, 3, 2, 1, 0],
        [2, 4, 0, 3, 1],
        [3, 0, 4, 1, 2],
        [1, 2, 3, 4, 0],
    ] {
        let plan = plan_in(bus(Spec::default(), ROOMY, ROOMY, false), order);
        plan.budget(&Executor::new(4)).unwrap();
        assert_eq!(plan.bus().ledger.regions(), reference.0, "order {order:?}");
        assert_eq!(plan.bus().ledger.table(), reference.1, "order {order:?}");
    }
}

#[test]
fn the_fake_graph_packs_to_its_lower_bound() {
    let plan = plan(bus(Spec::default(), ROOMY, ROOMY, false));
    plan.budget(&Executor::new(2)).unwrap();
    let ledger = &plan.bus().ledger;
    println!("{}", ledger.table());
    let slab = ledger
        .slabs()
        .into_iter()
        .find(|s| s.kind == Kind::Device)
        .unwrap();
    // a, then d, lie on one path and share from offset 0; b and c are
    // unordered with each other; e is unordered with everyone. The
    // heaviest antichain is {d, e} = 7 MiB, and {b, c, e} = 6.
    assert_eq!(slab.sum, 16 * MIB);
    assert_eq!(slab.bound, 7 * MIB);
    assert_eq!(slab.bound_steps, ["d", "e"]);
    assert_eq!(slab.peak, slab.bound);
    let at = |step: &str| {
        let r = ledger
            .regions()
            .into_iter()
            .find(|r| r.step == step)
            .unwrap();
        (r.offset, r.bytes)
    };
    assert_eq!(at("d"), (0, 6 * MIB));
    assert_eq!(at("a"), (0, 4 * MIB));
    assert_eq!(at("e"), (6 * MIB, MIB));
    // The outputs are a sum, the scratch the packed peak, and the fixed
    // BLAS-handle workspace is on top.
    assert_eq!(
        ledger.device_total(0),
        5 * MIB + slab.peak + fb_v04::budget::BLAS_WORKSPACE
    );
}

#[test]
fn accept_and_refuse_are_deterministic_and_the_table_is_complete() {
    // Outputs 5 MiB, packed scratch 7, workspace 128: 140 MiB in all.
    let need = 140 * MIB;
    let mut tables = Vec::new();
    for workers in [1, 2, 4, 8, 8, 8] {
        let accepted = plan(bus(Spec::default(), need, ROOMY, false));
        accepted
            .budget(&Executor::new(workers))
            .expect("a plan at its cap is accepted");
        assert_eq!(accepted.bus().ledger.device_total(0), need);

        let refused = plan(bus(Spec::default(), need - 1, ROOMY, false));
        let error = refused
            .budget(&Executor::new(workers))
            .expect_err("a plan one byte over is refused");
        assert_eq!(error.kind, ErrorKind::Allocation);
        // The whole table, whichever step ran first: every step's row, the
        // packed scratch, its lower bound, the workspace, the totals.
        for line in [
            "a ",
            "b ",
            "c ",
            "d ",
            "e ",
            "scratch (packed)",
            "lower bound",
        ] {
            assert!(
                error
                    .message
                    .lines()
                    .any(|l| l.starts_with(line) || l.trim_start().starts_with(line)),
                "the refusal lacks '{line}':\n{error}"
            );
        }
        assert!(error.message.contains("memory budget: REFUSED"));
        assert!(error.message.contains("blas workspace"));
        tables.push(error.message.clone());
        // Refused at the final node: every step fit, the packed scratch
        // and the workspace did not. Nothing can be run.
        let run = refused.execute(&Executor::new(1)).expect_err("no run");
        assert_eq!(run.kind, ErrorKind::InvalidConfig);
    }
    assert!(
        tables.windows(2).all(|w| w[0] == w[1]),
        "one table, every run"
    );
    println!("{}", tables[0]);
}

#[test]
fn a_heap_slab_cannot_grow_and_an_over_cap_one_is_never_touched() {
    let spec = Spec {
        heap: [MIB, 0, 0, 0, 0],
        ..Spec::default()
    };
    let plan = plan(bus(spec.clone(), ROOMY, ROOMY, false));
    plan.budget(&Executor::new(2)).unwrap();
    {
        let a = plan.output(A);
        // A boxed slice: a length and no capacity, no push, no reserve.
        let slab: &[f64] = &a.heap;
        assert_eq!(slab.len(), MIB as usize / 8);
        assert_eq!(plan.bus().ledger.heap_total(), MIB);
    }
    // The host cap is checked at the call, before the slab exists: a heap
    // past the cap is charged, flagged, and never allocated.
    let refused = self::plan(bus(spec, ROOMY, MIB - 1, false));
    let error = refused
        .budget(&Executor::new(2))
        .expect_err("over the host");
    assert!(error.message.contains("REFUSED"), "{error}");
    assert_eq!(refused.output(A).heap.len(), 0, "never made, never touched");
    assert_eq!(refused.bus().ledger.heap_total(), MIB, "and still counted");
}

#[test]
fn a_config_that_does_not_close_is_refused() {
    // d reads b and c; the config asks for neither c nor a.
    let mut dag = Dag::new();
    dag.add(D).add(B);
    let error = dag
        .build(bus(Spec::default(), ROOMY, ROOMY, false))
        .err()
        .expect("the steps do not close");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(
        error
            .message
            .contains("step 'd' reads step 'c', which the config does not ask for")
            || error
                .message
                .contains("step 'b' reads step 'a', which the config does not ask for"),
        "{error}"
    );
}

#[test]
fn unordered_steps_overlap_in_time_and_never_in_bytes() {
    // Every step checks its scratch was not written under it; b and c, and
    // e with everyone, are free to run at once on eight workers.
    let mut overlapped = 0;
    for _ in 0..20 {
        let plan = plan(bus(Spec::default(), ROOMY, ROOMY, false));
        let executor = Executor::new(8);
        plan.budget(&executor).unwrap();
        plan.execute(&executor)
            .expect("no step's scratch is stomped");
        overlapped += usize::from(plan.bus().most.load(Ordering::Acquire) > 1);
    }
    assert!(
        overlapped > 0,
        "no run had two steps at once: nothing was tested"
    );
}

#[test]
fn a_replay_allocates_nothing_and_keeps_its_addresses_and_nan_fill_shows_stale_reads() {
    let plan = plan(bus(Spec::default(), ROOMY, ROOMY, true));
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    let allocations = plan.bus().ledger.allocations();
    for _ in 0..3 {
        plan.execute(&executor).unwrap();
    }
    assert_eq!(plan.bus().ledger.allocations(), allocations);
    for (i, name) in NAMES.iter().enumerate() {
        let out = plan.bus().slots[i].is_filled();
        assert!(out, "{name}");
    }
    let a = plan.output(A);
    assert_eq!(a.runs, 3);
    // The same region at the same address on every run.
    assert!(a.address.windows(2).all(|w| w[0] == w[1]));
    // In the test mode what a step finds on entry is NaN, every run, even
    // though it wrote its own value there the run before: a step that
    // relied on what was left would fail here.
    assert!(a.found.iter().all(|x| x.is_nan()), "{:?}", a.found);

    // Without the fill, a step whose region nobody shares finds its old
    // data: the contents are undefined, not cleared.
    let plain = self::plan(bus(Spec::default(), ROOMY, ROOMY, false));
    plain.budget(&executor).unwrap();
    plain.execute(&executor).unwrap();
    plain.execute(&executor).unwrap();
    assert_eq!(plain.output(E).found, [0.0, 5.0]);
}

#[test]
fn a_run_that_carves_past_its_declaration_is_stopped() {
    let spec = Spec {
        overreaches: Some(2),
        ..Spec::default()
    };
    let plan = plan(bus(spec, ROOMY, ROOMY, false));
    let executor = Executor::new(2);
    plan.budget(&executor).unwrap();
    let error = plan.execute(&executor).expect_err("c carves too much");
    assert!(
        error.message.contains("but its mem_budget declared"),
        "{error}"
    );
}

#[test]
fn a_failed_step_fails_its_readers_only() {
    let spec = Spec {
        fails: Some(1),
        ..Spec::default()
    };
    let plan = plan(bus(spec, ROOMY, ROOMY, false));
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    let error = plan.execute(&executor).expect_err("b fails");
    assert!(error.to_string().contains("step 'b'"), "{error}");
    let runs = |i: usize| match i {
        0 => plan.output(A).runs,
        2 => plan.output(C).runs,
        3 => plan.output(D).runs,
        _ => plan.output(E).runs,
    };
    assert_eq!((runs(0), runs(2), runs(4)), (1, 1, 1), "a, c and e ran");
    assert_eq!(runs(3), 0, "d reads b and was skipped");
}

#[test]
fn a_dry_budget_counts_and_cannot_run() {
    let cap = |used| Cap {
        written: None,
        used,
    };
    let bus = Arc::new(TestBus {
        spec: Spec::default(),
        ledger: Ledger::new(
            Caps {
                device: vec![cap(ROOMY)],
                host: cap(ROOMY),
            },
            Backing::Dry,
            vec![0],
            false,
        ),
        slots: std::array::from_fn(|_| Slot::new()),
        running: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    });
    let plan = plan(bus);
    let executor = Executor::new(2);
    plan.budget(&executor).unwrap();
    assert_eq!(plan.bus().ledger.allocations(), 0);
    assert_eq!(plan.bus().ledger.device_total(0), 140 * MIB);
    assert!(!plan.output(A).device.is_allocated());
    assert!(plan.execute(&executor).is_err());
}
