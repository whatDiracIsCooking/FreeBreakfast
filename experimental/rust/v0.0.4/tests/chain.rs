//! The DF-RHF chain as steps: what `README.md` ("What it is held
//! against", "What it runs that no prototype has") lists, as far as this
//! prototype builds it. `README.md`, "What is not built", has the rows that are not here.
//!
//! The device cases are `#[ignore]`: run them with
//!
//! ```sh
//! cargo test --test chain -- --include-ignored --test-threads=1
//! ```
//!
//! One run each, the bars fixed by analysis: 1e-9 Eh on an energy, 1e-10
//! of the largest element on B.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fb_v04::budget::{Backing, LockId};
use fb_v04::bus::{Config, MainBus};
use fb_v04::flow::Bus;
use fb_v04::steps::eri::{EriStep, least_mem_per_gpu};
use fb_v04::steps::scf::ScfStep;
use rustflow::Executor;

/// The bar on an energy, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// DF-RHF, cc-pVDZ over cc-pVDZ-JKFIT, as `../v0.0.2/README.md` records
/// them: each within 1.2e-12 of PySCF's DF-RHF and of the old Rust path.
const WATER: f64 = -76.026_751_140_543;
const BENZENE: f64 = -230.721_729_270_921;
const ETHYLENE: f64 = -77.867_545_824_336;

/// One case at a time: they share the devices.
static TURN: Mutex<()> = Mutex::new(());

fn rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../rust")
}

fn water() -> PathBuf {
    rust_dir().join("integrals/one_electron/overlap/tests/data/h2o.xyz")
}

fn data(file: &str) -> PathBuf {
    rust_dir().join("pipeline/scf/tests/data").join(file)
}

/// A config of `xyz` with `eri` and `resources` as the bodies of their
/// tables, and `[scf]` unless `scf` is false.
fn config(xyz: &Path, eri: &str, scf: bool, resources: &str) -> Config {
    let text = format!(
        r#"
[molecule]
xyz = "{}"
[basis]
name = "cc-pvdz"
[eri]
aux_basis = "cc-pvdz-jkfit"
{eri}
{}
[resources]
{resources}
"#,
        xyz.display(),
        if scf { "[scf]" } else { "" }
    );
    Config::parse(&text, Path::new(".")).expect("the config parses")
}

/// The least `mem_per_gpu` of `xyz`'s plan, as the config writes it.
fn least_mib(xyz: &Path) -> f64 {
    let bus = MainBus::setup(config(xyz, "", true, ""), Backing::Dry).unwrap();
    least_mem_per_gpu(&bus.shared().dims, 2) as f64 / (1u64 << 20) as f64
}

fn energy(xyz: &Path, eri: &str, resources: &str) -> (f64, usize) {
    let run = fb_v04::run(config(xyz, eri, true, resources)).expect("the chain runs");
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged());
    (scf.energy(), scf.iterations())
}

fn devices_seen() -> i32 {
    fb_v04::device::device_count().unwrap()
}

#[test]
#[ignore = "needs a GPU"]
fn the_energies_match_the_recorded_ones() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    for (name, xyz, reference) in [
        ("water", water(), WATER),
        ("benzene", data("benzene.xyz"), BENZENE),
        ("ethylene", data("ethylene_meci856.xyz"), ETHYLENE),
    ] {
        let (e, iterations) = energy(&xyz, "", "");
        println!(
            "{name}: E = {e:.12} in {iterations} iterations, |E - recorded| = {:.1e}",
            (e - reference).abs()
        );
        assert!(
            (e - reference).abs() < ENERGY_BAR,
            "{name}: {e} against {reference}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn forced_batches_give_the_default_energy() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let xyz = water();
    // The default: 2 GiB holds water's whole share as one aux batch.
    let run = fb_v04::run(config(&xyz, "", true, "")).unwrap();
    let default = run.plan.output(ScfStep).energy();
    {
        let eri = run.plan.output(EriStep);
        assert_eq!(eri.jk_width(), eri.share(0).len(), "one aux batch");
        assert_eq!(eri.passes(), run.plan.output(ScfStep).iterations());
        assert_eq!(eri.batches(), run.plan.bus().shared().dims.classes.len());
    }
    assert!((default - WATER).abs() < ENERGY_BAR);
    drop(run);

    // At the least limit: many pair batches and narrow aux batches, so
    // every buffer of a pass's ring is reused many times.
    let least = least_mib(&xyz);
    let run = fb_v04::run(config(
        &xyz,
        &format!("mem_per_gpu_mib = {least}"),
        true,
        "nan_fill = true",
    ))
    .unwrap();
    let forced = run.plan.output(ScfStep).energy();
    {
        let eri = run.plan.output(EriStep);
        println!(
            "forced: {} pair batches of {} columns, aux batches of {} of {} rows; |E - default| = \
         {:.1e}",
            eri.batches(),
            eri.width(),
            eri.jk_width(),
            eri.share(0).len(),
            (forced - default).abs()
        );
        assert!(eri.batches() > 6, "the build is batched");
        assert!(
            eri.jk_width() * 4 < eri.share(0).len(),
            "the passes are batched, past the ring's depth"
        );
    }
    assert!((forced - default).abs() < ENERGY_BAR);
    drop(run);

    // The pass's ring at depth 1: one aux batch in flight.
    let (one, _) = energy(
        &xyz,
        &format!("mem_per_gpu_mib = {least}\nring = 1"),
        "nan_fill = true",
    );
    println!("ring 1: |E - default| = {:.1e}", (one - default).abs());
    assert!((one - default).abs() < ENERGY_BAR);
}

#[test]
#[ignore = "needs a GPU"]
fn b_from_the_ring_of_two_is_b_from_the_ring_of_one() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let xyz = water();
    // Small batches, so every batch meets its neighbour in the ring.
    let limit = format!("mem_per_gpu_mib = {}", least_mib(&xyz));
    let b_at = |ring: usize| {
        let run = fb_v04::run(config(
            &xyz,
            &format!("{limit}\nring = {ring}"),
            false,
            "nan_fill = true",
        ))
        .unwrap();
        let eri = run.plan.output(EriStep);
        assert!(eri.batches() > 6);
        eri.b().to_vec()
    };
    let (two, one) = (b_at(2), b_at(1));
    let largest = one.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let worst = two
        .iter()
        .zip(&one)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    println!("B: largest element {largest:.3e}, ring 2 against ring 1 {worst:.1e}");
    assert!(largest > 0.1 && worst <= 1e-10 * largest);
}

#[test]
#[ignore = "needs a GPU"]
fn a_replay_gives_the_energy_again_and_allocates_nothing() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut run = fb_v04::run(config(&water(), "", true, "")).unwrap();
    let first = run.plan.output(ScfStep).energy();
    let allocations = run.plan.bus().ledger().allocations();
    run.replay().expect("the replay runs");
    let scf = run.plan.output(ScfStep);
    println!(
        "replay: E = {:.12} in {} iterations, |E - first| = {:.1e}",
        scf.energy(),
        scf.iterations(),
        (scf.energy() - first).abs()
    );
    assert!((scf.energy() - first).abs() < ENERGY_BAR);
    assert_eq!(run.plan.bus().ledger().allocations(), allocations);
    // B was rebuilt, and the passes counted again.
    assert_eq!(run.plan.output(EriStep).passes(), scf.iterations());
}

#[test]
#[ignore = "needs a GPU"]
fn several_gpus_give_the_energy_of_one() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let xyz = data("benzene.xyz");
    let (one, _) = energy(&xyz, "", "");
    assert!((one - BENZENE).abs() < ENERGY_BAR);
    // Two logical GPUs on device 0, then every count of real ones.
    let mut cases = vec!["devices = [0, 0]".to_owned()];
    for n in 2..=devices_seen().min(4) {
        let list: Vec<String> = (0..n).map(|d| d.to_string()).collect();
        cases.push(format!("devices = [{}]", list.join(", ")));
    }
    let least = least_mib(&xyz);
    for devices in cases {
        // Each share one aux batch, and forced batches; both NaN-filled,
        // on two workers and on eight.
        for (eri, threads) in [
            (String::new(), 8),
            (format!("mem_per_gpu_mib = {}", 1.5 * least), 2),
        ] {
            let (e, iterations) = energy(
                &xyz,
                &eri,
                &format!("{devices}\nnan_fill = true\nthreads = {threads}"),
            );
            println!(
                "{devices}, [eri] {eri:?}, {threads} threads: E = {e:.12} in {iterations} \
                 iterations, |E - one GPU| = {:.1e}",
                (e - one).abs()
            );
            assert!((e - one).abs() < ENERGY_BAR, "{devices}: {e} against {one}");
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn locks_are_taken_in_rank_order_and_released_after_the_sync() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let n = devices_seen().min(2);
    let list: Vec<String> = (0..n).map(|d| d.to_string()).collect();
    let run = fb_v04::run(config(
        &water(),
        "",
        true,
        &format!("devices = [{}]\nthreads = 8", list.join(", ")),
    ))
    .unwrap();
    let all = run.plan.bus().shared().acquisitions();
    for step in ["orthogonalizer", "eri", "scf"] {
        let of: Vec<_> = all.iter().filter(|a| a.step == step).collect();
        assert!(!of.is_empty(), "{step} takes a lock");
        // In the order taken: rank ascending, each had before the next was
        // asked for, and all released together, after the step-end sync.
        let mut taken = of.clone();
        taken.sort_by_key(|a| a.locked);
        for pair in taken.windows(2) {
            assert!(pair[0].lock < pair[1].lock, "{step}: {:?}", pair);
            assert!(pair[0].locked <= pair[1].asked);
        }
        assert!(
            of.iter()
                .all(|a| a.released == of[0].released && a.locked <= a.released)
        );
    }
    // gpu0 has one holder at a time: the three holders' spans do not meet.
    let mut gpu0: Vec<_> = all.iter().filter(|a| a.lock == LockId::Gpu(0)).collect();
    gpu0.sort_by_key(|a| a.locked);
    assert_eq!(gpu0.len(), 3);
    for pair in gpu0.windows(2) {
        assert!(pair[0].released <= pair[1].locked, "{pair:?}");
    }
    // The orthogonalizer and eri are unordered and both want gpu0: the
    // second to ask is recorded as having waited, unless the first was
    // already done.
    let waited = gpu0.iter().filter(|a| a.waited).count();
    println!("gpu0: {waited} of its 3 holders waited for it");
    // Unordered lock holders keep disjoint scratch.
    let regions = run.plan.bus().ledger().regions();
    let on_gpu0 = |step: &str| {
        regions
            .iter()
            .find(|r| r.step == step && r.gpu == 0 && r.kind == fb_v04::budget::Kind::Device)
            .map(|r| r.offset..r.offset + r.bytes)
            .unwrap()
    };
    let (o, e) = (on_gpu0("orthogonalizer"), on_gpu0("eri"));
    assert!(
        o.end <= e.start || e.end <= o.start,
        "{o:?} and {e:?} overlap"
    );
}

#[test]
#[ignore = "needs a GPU"]
#[should_panic(expected = "short of the size query")]
fn a_short_work_buffer_is_refused_before_the_solver_is_called() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let n = 16;
    let mut gpu = fb_v04::device::GpuResources::new(0, 0).unwrap();
    // The binding repeats the size query and panics on the work buffer's
    // length before any pointer is used: no memory stands behind these.
    let (mut a, mut w, mut work) = (Nowhere::<f64>::of(n * n), Nowhere::of(n), Nowhere::of(1));
    let mut info = Nowhere::<i32>::of(1);
    let _ = gpu.syevd(n, &mut a, &mut w, &mut work, &mut info);
}

/// A length with no memory behind it, for a binding that must refuse it
/// before it touches a pointer.
struct Nowhere<T> {
    len: usize,
    of: std::marker::PhantomData<T>,
}

impl<T> Nowhere<T> {
    fn of(len: usize) -> Self {
        Nowhere {
            len,
            of: std::marker::PhantomData,
        }
    }
}

// SAFETY: never dereferenced: the binding panics first.
unsafe impl<T> fb_v04::device::DeviceSlice<T> for Nowhere<T> {
    fn as_ptr(&self) -> *const T {
        std::ptr::null()
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above.
unsafe impl<T> fb_v04::device::DeviceSliceMut<T> for Nowhere<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        std::ptr::null_mut()
    }
}

#[test]
fn the_eri_buffer_is_refused_one_byte_below_the_ring() {
    let xyz = water();
    let executor = Executor::new(2);
    let budget = |bytes: u64| {
        let eri = format!("mem_per_gpu_mib = {}", bytes as f64 / (1u64 << 20) as f64);
        let bus = MainBus::setup(config(&xyz, &eri, true, ""), Backing::Dry).unwrap();
        fb_v04::plan(bus).unwrap().budget(&executor)
    };
    let dims = MainBus::setup(config(&xyz, "", true, ""), Backing::Dry)
        .unwrap()
        .shared()
        .dims
        .clone();
    let least = least_mem_per_gpu(&dims, 2);
    budget(least).expect("the least limit holds a batch");
    let refused = budget(least - 1)
        .expect_err("a byte short holds none")
        .to_string();
    println!("{refused}");
    assert!(refused.contains("[eri] mem_per_gpu") && refused.contains(&format!("{least} B")));
}

#[test]
fn a_dry_budget_touches_no_device_and_prints_every_row() {
    let executor = Executor::new(2);
    let bus = MainBus::setup(config(&water(), "", true, "devices = [0, 1]"), Backing::Dry).unwrap();
    let plan = fb_v04::plan(bus).unwrap();
    plan.budget(&executor).expect("the plan is accepted");
    let table = plan.bus().ledger().table();
    for row in [
        "overlap",
        "core_hamiltonian",
        "metric",
        "guess",
        "orthogonalizer",
        "eri",
        "scf",
        "blas workspace",
        "lower bound",
        "device 1",
    ] {
        assert!(table.contains(row), "the table has no '{row}':\n{table}");
    }
    assert!(table.contains("eri: Manager(Eri)") && table.contains("scf: Gpu(1)"));
    // The edges are the declared reads, and nothing else.
    let edges = plan.edges();
    let reads = |step: &str| edges.iter().find(|(s, _)| *s == step).unwrap().1.clone();
    assert_eq!(reads("eri"), ["metric"]);
    assert_eq!(reads("orthogonalizer"), ["overlap"]);
    let mut scf = reads("scf");
    scf.sort_unstable();
    assert_eq!(
        scf,
        [
            "core_hamiltonian",
            "eri",
            "guess",
            "orthogonalizer",
            "overlap"
        ]
    );
}

#[test]
fn a_step_that_reads_a_step_the_config_leaves_out_is_refused() {
    let text = format!(
        "[molecule]\nxyz = \"{}\"\n[basis]\nname = \"cc-pvdz\"\n[scf]\n",
        water().display()
    );
    let config = Config::parse(&text, Path::new(".")).unwrap();
    let bus = MainBus::setup(config, Backing::Dry).unwrap();
    let refused = fb_v04::plan(bus).err().expect("scf reads eri").to_string();
    assert!(
        refused.contains("'scf'") && refused.contains("'eri'"),
        "{refused}"
    );
}

#[test]
fn the_layer_names_nothing_of_the_old_wrapper_and_task_layers() {
    // The four crates of rust/ above the C ABI: no path into any of them
    // from src/, and none among the dependencies.
    const OLD: [&str; 4] = [
        "fb_core_wwr",
        "fb_core_arena",
        "fb_core_planner",
        "fb_core_thread_pool",
    ];
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![root.join("build.rs")];
    let mut dirs = vec![root.join("src")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    assert!(files.len() > 10);
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        for old in OLD {
            assert!(!text.contains(old), "{} names {old}", file.display());
        }
    }
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    for old in OLD {
        assert!(
            !manifest.contains(&old.replace('_', "-")),
            "Cargo.toml names {old}"
        );
    }
}
