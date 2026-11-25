//! The batched steps on a card: B built in batches of pairs and kept on
//! the host, J and K over batches of aux functions, on one card and on two
//! logical cards aliased onto device 0, in cc-pVDZ over cc-pVDZ-JKFIT.
//!
//! **The bars, by analysis, one run each.**
//!
//! - B against the old Rust path's B (`add_packed_b`, the same kernels,
//!   the same Cholesky and the same `fb_recipes_sys::trsm`, at the old
//!   planner's batch width): `1e-10` of B's largest element. The two differ
//!   in the order of the linear algebra around the kernels, a few ulps
//!   that the solve against L amplifies by at most the metric's condition
//!   number's root.
//! - J and K against a host f64 contraction of the same B and D: `1e-11`
//!   of the largest element. The same sums in another order.
//! - Another batch width, or two cards, against one batch on one card: the
//!   same two bars. Not bitwise: a batch's gemm sums in a width-dependent
//!   order.
//!
//! **Real multi-card was not run.** This box has one CUDA card; "two
//! cards" here is `devices = [0, 0]`.
//!
//! Needs a GPU, and both card locks held:
//! `cargo test --test gpu -- --include-ignored --nocapture --test-threads=1`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{BLAS_WORKSPACE_BYTES, CpuTask, Dag, Room};
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::add_packed_b;
use fb_recipes_cholesky::add_cholesky;
use fb_v03::budget::{Backing, Kind};
use fb_v03::bus::{Config, MainBus, Shared};
use fb_v03::flow::Bus;
use fb_v03::steps::eri::EriStep;
use fb_v03::steps::guess::sad_density;
use fb_v03::steps::jk::{JkStep, write_gather_tables};
use rustflow::Executor;

const B_BAR: f64 = 1e-10;
const JK_BAR: f64 = 1e-11;

/// One case at a time: all share device 0.
static TURN: Mutex<()> = Mutex::new(());

fn rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../rust")
}

fn water() -> PathBuf {
    rust_dir().join("integrals/one_electron/overlap/tests/data/h2o.xyz")
}

fn benzene() -> PathBuf {
    rust_dir().join("pipeline/scf/tests/data/benzene.xyz")
}

/// A config: `eri` and `jk` are their tables' bodies, `resources` that
/// table's.
fn config(xyz: &Path, eri: &str, jk: &str, resources: &str) -> Config {
    let text = format!(
        r#"
[molecule]
xyz = "{}"

[basis]
name = "cc-pvdz"

[eri]
aux_basis = "cc-pvdz-jkfit"
{eri}

[guess]

[jk]
{jk}

[resources]
{resources}
"#,
        xyz.display()
    );
    Config::parse(&text, Path::new("/")).expect("the config parses")
}

struct Printer;
impl log::Log for Printer {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        println!("{}", record.args());
    }
    fn flush(&self) {}
}

fn logging() {
    let _ = log::set_logger(&Printer);
    log::set_max_level(log::LevelFilter::Warn);
}

/// CPU: the old path's heap B copied out.
struct Keep(RowId<Heap, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Keep {
    type Rows = ();
    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.0);
    }
    fn run(&self, _: &(), _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap() = heap.get(self.0).to_vec();
        Ok(())
    }
}

/// The old Rust path's B: the planner, the arena and `add_packed_b`, every
/// pair kept, as `experimental/rust/df_batched/single_gpu` drives it.
fn old_b(basis: &BasisSet, aux: &BasisSet) -> Vec<f64> {
    let map = PairMap::new(&fb_integrals_mmd::record_tables(basis), None, 0.0);
    let naux = aux.nao_sphe() as usize;
    let room = Room {
        device: (2 << 30) + BLAS_WORKSPACE_BYTES,
        pinned: usize::MAX,
    };
    let (planned, kept) = Dag::plan_within(
        |_| Ok(room),
        |dag| {
            dag.blas_workspace(0);
            let classes = fb_integrals_mmd::add_records(dag, 0, basis)?;
            let (mut u, mut u_aux) = (None, None);
            let metric = add_metric(dag, 0, aux, &mut u_aux)?;
            let factor = add_cholesky(dag, 0, "L", naux, metric)?;
            let b = add_packed_b(
                dag, 0, basis, &classes, aux, &map, factor, &mut u, &mut u_aux,
            )?;
            let out = Arc::new(Mutex::new(Vec::new()));
            dag.add_cpu("keep B", 0, Keep(b, Arc::clone(&out)));
            Ok(out)
        },
    )
    .expect("the old path plans");
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    record.result.as_ref().expect("the old path runs");
    std::mem::take(&mut *kept.lock().unwrap())
}

/// The largest `|x - y|` and the largest `|y|`.
fn against(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).fold((0.0, 0.0), |(diff, big), (a, b)| {
        (f64::max(diff, (a - b).abs()), f64::max(big, b.abs()))
    })
}

/// J and K of `d` over the packed `b`, on the host, in doubles: each row
/// of B unpacked into its symmetric square, then the two sums.
fn host_jk(shared: &Shared, b: &[f64], d: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (n, n_col, n_aux) = (shared.dims.n_ao, shared.dims.n_col, shared.dims.n_aux);
    let (mut rows, mut cols, mut weights) = (vec![0; n_col], vec![0; n_col], vec![0.0; n_col]);
    write_gather_tables(shared, &mut rows, &mut cols, &mut weights);
    let (mut j, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    let (mut square, mut bd) = (vec![0.0; n * n], vec![0.0; n * n]);
    for q in 0..n_aux {
        square.fill(0.0);
        for c in 0..n_col {
            let (m, l) = (rows[c] as usize, cols[c] as usize);
            if weights[c] == 1.0 && m < l {
                continue;
            }
            square[m + n * l] = b[c + n_col * q];
            square[l + n * m] = b[c + n_col * q];
        }
        let gamma: f64 = square.iter().zip(d).map(|(x, y)| x * y).sum();
        for (out, x) in j.iter_mut().zip(&square) {
            *out += gamma * x;
        }
        // bd = B_Q D, then K += bd B_Qᵀ, column-major.
        bd.fill(0.0);
        for c in 0..n {
            for l in 0..n {
                let dlc = d[l + n * c];
                for r in 0..n {
                    bd[r + n * c] += square[r + n * l] * dlc;
                }
            }
        }
        for c in 0..n {
            for l in 0..n {
                let blc = square[c + n * l];
                for r in 0..n {
                    k[r + n * c] += bd[r + n * l] * blc;
                }
            }
        }
    }
    (j, k)
}

/// `step`'s batches at `limit` MiB, on a dry budget (the limit only picks
/// the width); `None` when no batch fits.
fn batches_at(xyz: &Path, step: &str, limit: f64) -> Option<usize> {
    let table = format!("limit_mib = {limit}");
    let config = match step {
        "eri" => config(xyz, &table, "", ""),
        _ => config(xyz, "", &table, ""),
    };
    let plan = fb_v03::plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
    plan.budget(&Executor::new(1)).ok()?;
    Some(match step {
        "eri" => plan.output(EriStep).batches(),
        _ => plan.output(JkStep).batches(),
    })
}

/// A `limit_mib` that forces `step` into several batches. For `eri`, the
/// smallest limit that holds a batch: its narrowest width, one pair block
/// of the widest class. For `jk`, the largest limit with four batches or
/// more.
fn limit_for(xyz: &Path, step: &str) -> f64 {
    if step == "jk" {
        let mut limit = 256.0;
        while batches_at(xyz, step, limit).is_some_and(|b| b < 4) {
            limit *= 0.9;
        }
        return limit;
    }
    let (mut refused, mut held) = (0.0, 256.0);
    for _ in 0..40 {
        let mid = 0.5 * (refused + held);
        match batches_at(xyz, step, mid) {
            Some(_) => held = mid,
            None => refused = mid,
        }
    }
    held
}

struct Result3 {
    b: Vec<f64>,
    j: Vec<f64>,
    k: Vec<f64>,
}

fn results(run: &fb_v03::Run) -> Result3 {
    let (eri, jk) = (run.plan.output(EriStep), run.plan.output(JkStep));
    Result3 {
        b: eri.b().to_vec(),
        j: jk.j_host().to_vec(),
        k: jk.k_host().to_vec(),
    }
}

fn hold(what: &str, stem: &str, new: &Result3, reference: &Result3) {
    for (name, x, y, bar) in [
        ("B", &new.b, &reference.b, B_BAR),
        ("J", &new.j, &reference.j, JK_BAR),
        ("K", &new.k, &reference.k, JK_BAR),
    ] {
        let (diff, big) = against(x, y);
        println!("{stem}: {what}: {name} max |diff| {diff:.3e}, max |{name}| {big:.3e}");
        assert!(
            diff <= bar * big,
            "{stem}: {what}: {name} is off by {diff:e}"
        );
    }
}

fn print_run(stem: &str, what: &str, run: &fb_v03::Run) {
    let (eri, jk) = (run.plan.output(EriStep), run.plan.output(JkStep));
    println!(
        "{stem}: {what}: eri width {} columns in {} batches, jk width {} aux in {} batches; \
         setup {:.3?}, budget mode {:.3?}, execute mode {:.3?}",
        eri.width(),
        eri.batches(),
        jk.width(),
        jk.batches(),
        run.setup,
        run.budget,
        run.execute
    );
    for step in &run.steps {
        println!(
            "  {:<8} began {:>10.3?}  ended {:>10.3?}  run {:>10.3?}  wrapper sync {:>10.3?}",
            step.name, step.began, step.ended, step.run, step.sync
        );
    }
    println!("{}", run.table);
}

fn case(stem: &str, xyz: &Path) {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    logging();

    // One card, the default limits.
    let one = fb_v03::run(config(xyz, "", "", "")).expect("one card runs");
    let shared = one.plan.bus().shared();
    println!("{stem}: {:?}", shared.dims);
    print_run(stem, "one card, default limits", &one);
    let reference = results(&one);

    // B against the old Rust path's.
    let old = old_b(&shared.basis, shared.aux.as_ref().unwrap());
    let (diff, big) = against(&reference.b, &old);
    println!("{stem}: B against the old path: max |diff| {diff:.3e}, max |B| {big:.3e}");
    assert!(
        diff <= B_BAR * big,
        "{stem}: B is off the old path's by {diff:e}"
    );

    // J and K against a host contraction of the same B and D.
    let d = sad_density(shared).unwrap();
    let (j, k) = host_jk(shared, &reference.b, &d);
    for (name, x, y) in [("J", &reference.j, &j), ("K", &reference.k, &k)] {
        let (diff, big) = against(x, y);
        println!("{stem}: {name} against the host: max |diff| {diff:.3e}, max |{name}| {big:.3e}");
        assert!(
            diff <= JK_BAR * big,
            "{stem}: {name} is off the host's by {diff:e}"
        );
    }

    // Limits that force several batches, on one card.
    let eri = format!("limit_mib = {}", limit_for(xyz, "eri"));
    let jk = format!("limit_mib = {}", limit_for(xyz, "jk"));
    println!("{stem}: forcing batches with [eri] {eri}, [jk] {jk}");
    let batched = fb_v03::run(config(xyz, &eri, &jk, "")).expect("the batched run");
    print_run(stem, "one card, forced batches", &batched);
    assert!(batched.plan.output(JkStep).batches() >= 4);
    assert!(batched.plan.output(EriStep).batches() > one.plan.output(EriStep).batches());
    hold(
        "forced batches against one batch",
        stem,
        &results(&batched),
        &reference,
    );

    // Two logical cards on device 0, the same limits, NaN-filled leases.
    let two = fb_v03::run(config(xyz, &eri, &jk, "devices = [0, 0]\nnan_fill = true"))
        .expect("two logical cards run");
    print_run(stem, "two logical cards, forced batches, NaN fill", &two);
    hold(
        "two logical cards against one card",
        stem,
        &results(&two),
        &reference,
    );
    let ledger = two.plan.bus().ledger();
    // Each card has its own scratch rows, and each all-GPU step a region
    // on both.
    for step in ["records", "eri", "jk"] {
        for card in 0..2 {
            assert!(
                ledger
                    .regions()
                    .iter()
                    .any(|r| r.step == step && r.card == card && r.kind == Kind::Device),
                "{step} has no device scratch on card {card}"
            );
        }
    }
    // The locks: each library-using step took both cards', in card order,
    // and held them until after its sync; the two steps never held a card
    // at once.
    let held = two.plan.bus().shared().held();
    for h in &held {
        println!(
            "  lock: {:<4} card {} from {:>10.3?} to {:>10.3?}",
            h.step, h.card, h.locked, h.released
        );
    }
    for step in ["eri", "jk"] {
        let of: Vec<_> = held.iter().filter(|h| h.step == step).collect();
        assert_eq!(of.len(), 2, "{step} took two locks");
        assert!(of[0].card == 0 && of[1].card == 1 && of[0].locked <= of[1].locked);
    }
    let (eri_held, jk_held) = (
        held.iter().find(|h| h.step == "eri").unwrap(),
        held.iter().find(|h| h.step == "jk").unwrap(),
    );
    assert!(eri_held.released <= jk_held.locked, "jk reads eri");

    // A replay: nothing allocated, the same result, NaN fill still on.
    let executor = Executor::new(4);
    let allocations = ledger.allocations();
    let regions = ledger.regions();
    two.plan.execute(&executor).expect("the replay runs");
    assert_eq!(
        ledger.allocations(),
        allocations,
        "a replay allocates nothing"
    );
    assert_eq!(ledger.regions(), regions, "and runs in the same regions");
    let (eri_out, jk_out) = (two.plan.output(EriStep), two.plan.output(JkStep));
    let replayed = Result3 {
        b: eri_out.b().to_vec(),
        j: jk_out.j_host().to_vec(),
        k: jk_out.k_host().to_vec(),
    };
    hold("the replay against one card", stem, &replayed, &reference);
    println!("{stem}: the replay allocated nothing ({allocations} allocations before and after)");
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_b_j_and_k_hold_in_batches_and_on_two_logical_cards() {
    case("water", &water());
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzene_b_j_and_k_hold_in_batches_and_on_two_logical_cards() {
    case("benzene", &benzene());
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn an_over_cap_plan_is_refused_with_the_whole_table() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // 129 MiB: the 128 MiB workspace fits, the 1.72 MiB around it does not.
    let error = fb_v03::run(config(&water(), "", "", "device_cap_mib = 129"))
        .err()
        .expect("over its device cap");
    assert_eq!(error.kind, ErrorKind::Allocation);
    println!("{error}");
    for line in [
        "memory budget: REFUSED",
        "records",
        "guess",
        "eri",
        "jk",
        "scratch (packed)",
        "lower bound",
        "blas workspace",
        "cap as written",
    ] {
        assert!(error.message.contains(line), "the refusal lacks '{line}'");
    }
    // A cap past the card is clamped to what the card has: the table says
    // both.
    let run = fb_v03::run(config(&water(), "", "", "device_cap_mib = 1000000"))
        .expect("an explicit cap past the card is clamped, not an abort");
    println!("{}", run.table);
    assert!(run.table.contains("976.56 GiB"), "the cap as written");
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_limit_too_small_for_one_batch_is_refused_naming_the_step() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    for (eri, jk, step) in [
        ("limit_mib = 0.01", "", "eri"),
        ("", "limit_mib = 0.01", "jk"),
    ] {
        let error = fb_v03::run(config(&water(), eri, jk, ""))
            .err()
            .expect("no batch fits");
        println!("{error}");
        assert_eq!(error.kind, ErrorKind::Allocation);
        let text = error.to_string();
        assert!(text.contains(&format!(
            "step '{step}': its limit of 10.24 KiB holds no batch"
        )));
        assert!(text.contains("the narrowest, of width"));
        assert!(text.contains("of device scratch per card"));
    }
}
