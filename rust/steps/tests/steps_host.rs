//! The MainBus and the steps' budget with no device (tier 3): setup's
//! counts on gly10 in cc-pVDZ / cc-pVDZ-JKFIT against the v0.0.4 page's,
//! the dry budget's exact bytes from them, the orthogonalizer's scratch
//! apart from every step unordered with it, the config's refusals, the
//! allocation record failing a step that allocates on the plain heap in its
//! `run`.

mod common;

use std::sync::Arc;

use fb_core_budget::{Backing, Budget, Cap, Caps, Kind, Layout, Lease, Ledger, RECORD_THRESHOLD};
use fb_core_device::HANDLE_WORKSPACE_BYTES;
use fb_core_error_handling::{ErrorKind, Result, Status};
use fb_core_flow::{Bus, Dag, Executor, Held, Lane, Locks, Slot, Step, View};
use fb_steps::{DEFAULT_MEM_PER_GPU, EriStep, MainBus, plan};

const ALIGN: u64 = 256;

fn aligned(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(ALIGN) * ALIGN
}

/// The metric on four GPUs, dry: one pinned staging, gpu0's, of `n_aux²`
/// doubles, uploaded to every GPU under one token, and none on the
/// others; (P|Q) on each GPU's device side.
#[test]
fn the_metric_stages_once_for_every_gpu() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
                [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[resources]\ndevices = [0, 1, 2, 3]\n";
    let config = fb_steps::Config::parse(text, &dir).unwrap();
    let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
    plan.budget(&Executor::new(2)).unwrap();
    let q = plan.bus().shared().dims.n_aux;
    let ledger = plan.bus().ledger();
    let pinned: Vec<(usize, u64)> = ledger
        .regions()
        .iter()
        .filter(|r| r.step == "metric" && r.kind == Kind::Pinned)
        .map(|r| (r.gpu, r.bytes))
        .collect();
    assert_eq!(pinned, [(0, aligned(q * q * 8))]);
    for gpu in 1..4 {
        assert!(ledger.device_total(gpu) >= (q * q * 8) as u64, "gpu {gpu}");
    }
}

/// `--budget-only` on gly10: every length a count, exact with no device;
/// syevd's work buffer zero.
#[test]
fn a_dry_budget_of_gly10_is_exact() {
    let config = common::config("gly10.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
    let dims = plan.bus().shared().dims.clone();
    // The v0.0.4 page's gly10 ("What it showed", "Batched only").
    assert_eq!((dims.n_ao, dims.n_aux), (734, 3606));
    assert_eq!(dims.n_col, 270_497);
    assert_eq!(dims.n_shell_pair, dims.n_shell * (dims.n_shell + 1) / 2);
    assert_eq!(
        dims.classes.iter().map(|c| c.n_pair).sum::<usize>(),
        dims.n_shell_pair
    );
    plan.budget(&Executor::new(2)).unwrap();
    let ledger = plan.bus().ledger();
    assert_eq!(ledger.allocations(), 0, "a dry budget allocates nothing");
    let (n, q) = (dims.n_ao, dims.n_aux);
    let square = |m: usize| aligned(m * m * 8);
    // The orthogonalizer's device scratch at a work buffer of zero.
    let mut l = Layout::new();
    let _ = (
        l.span::<f64>(n * n),
        l.span::<f64>(n),
        l.span::<f64>(n * n),
        l.span::<f64>(0),
        l.span::<i32>(1),
    );
    let regions = ledger.regions();
    let region = |step: &str, kind: Kind| {
        regions
            .iter()
            .find(|r| r.gpu == 0 && r.kind == kind && r.step == step)
            .map(|r| r.bytes)
    };
    assert_eq!(
        region("orthogonalizer", Kind::Device),
        Some(aligned(l.bytes()))
    );
    assert_eq!(region("orthogonalizer", Kind::Pinned), Some(ALIGN));
    for step in ["overlap", "guess"] {
        assert_eq!(region(step, Kind::Pinned), Some(square(n)), "{step}");
    }
    // T's packed triangle and V's side by side, n past H's square.
    assert_eq!(
        region("core_hamiltonian", Kind::Pinned),
        Some(aligned((n * n + n) * 8))
    );
    assert_eq!(region("metric", Kind::Pinned), Some(square(q)));
    // Four n × n outputs, the metric's q × q and L's, the orthogonalizer's
    // and the ERI's device scratch (unordered, so side by side), the
    // handle workspace, and the ERI buffer: within its default limit and
    // at least the least that holds a batch.
    let eri_scratch = region("eri", Kind::Device).expect("the ERI's scratch");
    let rest = (4 * n * n * 8 + 2 * q * q * 8) as u64
        + aligned(l.bytes())
        + eri_scratch
        + HANDLE_WORKSPACE_BYTES as u64;
    let buffer = ledger.device_total(0) - rest;
    assert!(
        EriStep::least_mem_per_gpu(&dims, 2) <= buffer && buffer <= DEFAULT_MEM_PER_GPU,
        "the ERI buffer: {buffer} bytes"
    );
    // B on the host heap, at the unscreened bound.
    assert_eq!(ledger.heap_total(), (dims.n_col * q * 8) as u64);
    assert!(ledger.table().starts_with("memory budget: accepted"));
    // Pinned: the four steps unordered with it. Device: the ERI's.
    assert_eq!(common::orthogonalizer_scratch_is_its_own(ledger), 5);
}

#[test]
fn a_run_of_no_gpu_or_no_host_thread_and_a_table_of_no_step_here_are_refused() {
    let mut config = common::config("h2o.xyz", "cc-pvdz", None);
    config.resources.devices.clear();
    let Err(error) = MainBus::setup(config, Backing::Dry) else {
        panic!("a run of no GPU");
    };
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    let mut config = common::config("h2o.xyz", "cc-pvdz", None);
    config.resources.host_threads = Some(0);
    let Err(error) = MainBus::setup(config, Backing::Dry) else {
        panic!("a host pool of no thread");
    };
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n[scf]\n";
    let error = fb_steps::Config::parse(text, std::path::Path::new(".")).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
}

/// A bus of one step that allocates `BYTES` on the plain heap in its `run`.
struct HeapBus {
    ledger: Ledger,
    locks: Locks,
    out: Slot<usize>,
}

impl Bus for HeapBus {
    type Shared = ();
    fn shared(&self) -> &() {
        &()
    }
    fn ledger(&self) -> &Ledger {
        &self.ledger
    }
    fn locks(&self) -> &Locks {
        &self.locks
    }
}

#[derive(Clone, Copy)]
struct Heap<const BYTES: usize>;

impl<const BYTES: usize> Step<HeapBus> for Heap<BYTES> {
    const NAME: &'static str = "heap_in_run";
    type Output = usize;
    type Reads = ();
    fn slot(bus: &HeapBus) -> &Slot<usize> {
        &bus.out
    }
    fn mem_budget(_: &(), _: &Budget<'_>) -> Result<usize> {
        Ok(0)
    }
    #[allow(
        clippy::disallowed_macros,
        reason = "the record's own test: a run that allocates on the plain heap"
    )]
    fn run(
        out: &mut usize,
        _: &View<'_, HeapBus, Self>,
        _: &mut Held<'_>,
        _: &Lane,
        _: &Lease<'_>,
    ) -> Status {
        *out = std::hint::black_box(vec![1u8; BYTES]).len();
        Ok(())
    }
}

/// `Heap<BYTES>` budgeted and executed once on a host-backed plan.
#[allow(
    clippy::disallowed_macros,
    reason = "a test's ledger, one cap and one device, outside any step's run"
)]
fn execute_heap<const BYTES: usize>() -> (Result<()>, Arc<HeapBus>) {
    let cap = Cap {
        written: None,
        used: 1 << 30,
    };
    let bus = Arc::new(HeapBus {
        ledger: Ledger::new(
            Caps {
                device: vec![cap],
                host: cap,
            },
            Backing::Host,
            vec![0],
            false,
        ),
        locks: Locks::new(&[]),
        out: Slot::new(),
    });
    let mut dag = Dag::new();
    dag.add(Heap::<BYTES>);
    let plan = dag.build(Arc::clone(&bus)).unwrap();
    let executor = Executor::new(2);
    plan.budget(&executor).unwrap();
    (plan.execute(&executor).map(drop), bus)
}

/// A step whose `run` makes one 2 MiB allocation on the plain heap fails
/// the execute, named; one at the threshold itself, 1 MiB, runs, and the
/// table's unbudgeted row then has the execute's peak heap growth, at
/// least that allocation.
#[test]
fn a_step_allocating_2_mib_on_the_heap_in_run_fails_named() {
    let (ran, _) = execute_heap::<{ 2 << 20 }>();
    let error = ran.expect_err("a 2 MiB allocation in run");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(
        error.message.starts_with(
            "step 'heap_in_run' made one allocation of 2.00 MiB on the plain heap in its run, \
             past the allocation record's 1.00 MiB"
        ),
        "{}",
        error.message
    );

    let (ran, bus) = execute_heap::<RECORD_THRESHOLD>();
    ran.unwrap();
    let grown = bus.ledger().heap_growth().expect("the record is installed");
    assert!(grown >= RECORD_THRESHOLD as u64, "{grown}");
    assert!(
        bus.ledger()
            .table()
            .contains("heap, unbudgeted: the last execute's peak growth"),
        "{}",
        bus.ledger().table()
    );
}
