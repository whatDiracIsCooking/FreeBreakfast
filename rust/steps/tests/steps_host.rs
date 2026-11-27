//! The MainBus and the steps' budget with no device (tier 3): setup's
//! counts on gly10 in cc-pVDZ / cc-pVDZ-JKFIT against the prototype's,
//! the dry budget's exact bytes from them, the orthogonalizer's scratch
//! apart from every step unordered with it, the config's refusals (an
//! unknown key and an impossible multiplicity naming the file and the
//! line), a relative basis path read against the config's directory, an
//! open shell set up and its SCF's budget exact, a closed-shell
//! ion's budget the neutral's, `[guess] from` budgeted as the SAD guess
//! and its refusals naming the file and the line, a localization's budget exact in the
//! occupied block, the allocation record failing a step that allocates on the plain heap in its
//! `run`, and the four host steps passing it: each builds its triangle in
//! its pinned staging, so no `run` allocates one of its own
//! (`docs/memory.md`, "The budget"), past 1 MiB from about 360 AOs; and
//! the parse of amdgpu's fdinfo behind `chain_vram.rs`'s readings.

mod common;

use std::sync::Arc;

use fb_core_budget::{Backing, Budget, Cap, Caps, Kind, Layout, Lease, Ledger, RECORD_THRESHOLD};
use fb_core_device::HANDLE_WORKSPACE_BYTES;
use fb_core_error_handling::{ErrorKind, Result, Status};
use fb_core_flow::{Bus, Dag, Executor, Held, Lane, Locks, Slot, Step, View};

use common::fast_tests;
use fb_steps::{
    CoreHamiltonianStep, DEFAULT_MEM_PER_GPU, EriStep, GuessStep, MainBus, MetricStep, OverlapStep,
    ScfStep, plan,
};

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
    // gly10 as measured (`docs/programming_model.md`, "What the chain
    // measured"; `docs/memory.md`, "Batched only").
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
    let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n[gradient]\n";
    let error = fb_steps::Config::parse(text, std::path::Path::new(".")).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
}

/// `text` written to `name` in the target's scratch directory, `@data`
/// replaced by `tests/data`, and read back as the file it is.
fn read_config(
    name: &str,
    text: &str,
) -> (
    std::path::PathBuf,
    fb_core_error_handling::Result<fb_steps::Config>,
) {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, text.replace("@data", &data.display().to_string())).unwrap();
    let config = fb_steps::Config::read(&path);
    (path, config)
}

/// The water cation's doublet counts five alpha and four beta electrons
/// and is set up, an open shell the SCF runs as ROHF.
#[test]
fn the_water_cations_doublet_is_five_alpha_and_four_beta() {
    let text = "[molecule]\nxyz = \"@data/h2o.xyz\"\ncharge = 1\nmultiplicity = 2\n\
                [basis]\nname = \"cc-pvdz\"\n";
    let (_, config) = read_config("h2o_cation.toml", text);
    let config = config.unwrap();
    let spin = &config.molecule;
    let molecule = fb_io_xyz::read_xyz(&spin.xyz, spin.charge, spin.multiplicity).unwrap();
    assert_eq!(spin.occupations(molecule.electron_count()).unwrap(), (5, 4));
    let bus = MainBus::setup(config, Backing::Dry).expect("an open shell is set up");
    let dims = &bus.shared().dims;
    assert_eq!((dims.n_alpha, dims.n_beta), (5, 4));
}

/// A relative basis and aux-basis path is read against the config's
/// directory, as `xyz` is, not the working directory: a config in the
/// target's scratch directory names its BSE JSON copies of sto-3g and
/// cc-pVDZ-JKFIT by relative path, and its dry setup counts what the
/// carried names count. A carried name is not joined: beside a garbage file
/// called `sto-3g` in the same directory, `sto-3g` still resolves from the
/// binary.
#[test]
fn a_relative_basis_path_is_read_against_the_configs_directory() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("relative_basis");
    std::fs::create_dir_all(dir.join("sets")).unwrap();
    std::fs::copy(common::data("h2o.xyz"), dir.join("h2o.xyz")).unwrap();
    std::fs::copy(
        root.join("basis/sto-3g.json"),
        dir.join("sets/orbital.json"),
    )
    .unwrap();
    std::fs::copy(
        root.join("aux_basis/cc-pVDZ-JKFIT.json"),
        dir.join("sets/fitting.json"),
    )
    .unwrap();
    std::fs::write(dir.join("sto-3g"), "not a basis").unwrap();
    std::fs::write(dir.join("cc-pvdz-jkfit"), "not a basis").unwrap();
    assert!(
        !std::path::Path::new("sets/orbital.json").exists(),
        "the working directory is not the config's"
    );
    let dims = |name: &str, basis: &str, aux: &str| {
        let path = dir.join(name);
        let text = format!(
            "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"{basis}\"\n\
             [eri]\naux_basis = \"{aux}\"\n"
        );
        std::fs::write(&path, text).unwrap();
        let config = fb_steps::Config::read(&path).unwrap();
        let bus = MainBus::setup(config, Backing::Dry).unwrap_or_else(|e| panic!("{name}: {e}"));
        let dims = &bus.shared().dims;
        (dims.n_ao, dims.n_aux)
    };
    let carried = dims("carried.toml", "sto-3g", "cc-pvdz-jkfit");
    assert_eq!(carried.0, 7, "water in sto-3g");
    let paths = dims("paths.toml", "sets/orbital.json", "sets/fitting.json");
    assert_eq!(paths, carried);
}

/// The SCF's scratch, dry, as its layout places it: the closed shell's
/// tenants, then an open shell's after them at `m` singly occupied
/// orbitals; syevd's work buffer zero.
fn scf_scratch(n: usize, cap: usize, m: Option<usize>) -> (u64, u64) {
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    for _ in 0..8 {
        let _ = dev.span::<f64>(n * n);
    }
    let _ = (dev.span::<f64>(2), dev.span::<i32>(1));
    let _ = (dev.span::<f64>(n * n * cap), dev.span::<f64>(n * n * cap));
    let _ = (dev.span::<f64>(cap * cap), dev.span::<f64>(cap));
    let _ = (dev.span::<i32>(1), dev.span::<f64>(0));
    let _ = (pin.span::<f64>(2), pin.span::<i32>(1), pin.span::<i32>(1));
    let _ = (pin.span::<f64>(n * n), pin.span::<f64>(n));
    let _ = (pin.span::<f64>(n * n), pin.span::<f64>(n));
    if let Some(m) = m {
        // Fa, Fb, Da, Db, the second density's J and K, the identity.
        for _ in 0..7 {
            let _ = dev.span::<f64>(n * n);
        }
        let _ = (dev.span::<f64>(n * m), dev.span::<f64>(n * m));
        let _ = (
            dev.span::<f64>(m * m),
            dev.span::<f64>(m),
            dev.span::<i32>(1),
        );
        let _ = (pin.span::<f64>(n * n), pin.span::<i32>(1));
    }
    (aligned(dev.bytes()), aligned(pin.bytes()))
}

/// The dry budget of an open shell is exact: the SCF scratch of the
/// water cation's doublet and of its dication's triplet is the closed
/// shell's layout with the open shell's tenants after it, and its outputs
/// add the effective Fock matrix; the neutral water's is the closed
/// shell's alone.
#[test]
fn an_open_shells_dry_budget_is_exact() {
    for (charge, multiplicity, m) in [(0, 1, None), (1, 2, Some(1)), (2, 3, Some(2))] {
        let text = format!(
            "[molecule]\nxyz = \"h2o.xyz\"\ncharge = {charge}\nmultiplicity = {multiplicity}\n\
             [basis]\nname = \"cc-pvdz\"\n[eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n"
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let config = fb_steps::Config::parse(&text, &dir).unwrap();
        let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
        plan.budget(&Executor::new(2)).unwrap();
        let n = plan.bus().shared().dims.n_ao;
        let ledger = plan.bus().ledger();
        let region = |kind: Kind| {
            ledger
                .regions()
                .iter()
                .find(|r| r.gpu == 0 && r.kind == kind && r.step == "scf")
                .map(|r| r.bytes)
        };
        let (device, pinned) = scf_scratch(n, 8, m);
        assert_eq!(region(Kind::Device), Some(device), "charge {charge}");
        assert_eq!(region(Kind::Pinned), Some(pinned), "charge {charge}");
        assert_eq!(
            plan.output(ScfStep).fock().is_some(),
            m.is_some(),
            "charge {charge}: the effective Fock matrix"
        );
        assert!(ledger.table().starts_with("memory budget: accepted"));
    }
}

/// A multiplicity the electron count cannot have, and an unknown key, are
/// each refused `InvalidConfig` naming the file and the line.
#[test]
fn an_impossible_multiplicity_and_an_unknown_key_name_the_file_and_line() {
    for (name, text, line) in [
        (
            "h2o_triplet_cation.toml",
            "[molecule]\nxyz = \"@data/h2o.xyz\"\ncharge = 1\nmultiplicity = 3\n\
             [basis]\nname = \"cc-pvdz\"\n",
            4,
        ),
        (
            "h2o_cation_singlet.toml",
            "[basis]\nname = \"cc-pvdz\"\n[molecule]\nxyz = \"@data/h2o.xyz\"\ncharge = 1\n",
            5,
        ),
        (
            "h2o_neutral_doublet.toml",
            "[basis]\nname = \"cc-pvdz\"\n\n[molecule]\nxyz = \"@data/h2o.xyz\"\n\
             multiplicity = 2\n",
            6,
        ),
    ] {
        let (path, config) = read_config(name, text);
        let Err(error) = MainBus::setup(config.unwrap(), Backing::Dry) else {
            panic!("{name}: an impossible multiplicity");
        };
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
        let at = format!("{}:{line}: [molecule] multiplicity = ", path.display());
        assert!(error.message.starts_with(&at), "{name}: {}", error.message);
        assert!(
            error.message.contains("is impossible for"),
            "{}",
            error.message
        );
    }
    let text = "[molecule]\nxyz = \"@data/h2o.xyz\"\nspin = 2\n[basis]\nname = \"cc-pvdz\"\n";
    let (path, config) = read_config("h2o_spin.toml", text);
    let error = config.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    let at = format!("{}:3: unknown field `spin`", path.display());
    assert!(error.message.starts_with(&at), "{}", error.message);
}

/// A density file of `shape` written beside the configs `read_config`
/// writes, named `name`.
fn density_file(name: &str, shape: &[usize]) {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let data: Vec<f64> = (0..shape.iter().product::<usize>())
        .map(|i| i as f64)
        .collect();
    fb_io_npy::write_npy(&path, shape, &data).unwrap();
}

/// `[guess] from` budgets as the SAD guess does: the same staging, the
/// same bytes, so the dry table is the same text but for the host's
/// available memory read at setup.
#[test]
fn a_guess_from_a_file_is_budgeted_as_the_sad_guess() {
    density_file("guess_from_24.npy", &[24, 24]);
    let table = |guess: &str| {
        let text = format!(
            "[molecule]\nxyz = \"@data/h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
             [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n{guess}"
        );
        let (_, config) = read_config("guess_budget.toml", &text);
        let plan = plan(MainBus::setup(config.unwrap(), Backing::Dry).unwrap()).unwrap();
        plan.budget(&Executor::new(2)).unwrap();
        assert_eq!(plan.bus().shared().dims.n_ao, 24);
        let lines: Vec<String> = plan
            .bus()
            .ledger()
            .table()
            .lines()
            .filter(|l| !l.starts_with("host: "))
            .map(str::to_owned)
            .collect();
        lines
    };
    let sad = table("");
    assert!(sad.iter().any(|l| l.contains("guess")), "{sad:?}");
    assert_eq!(table("[guess]\nfrom = \"guess_from_24.npy\"\n"), sad);
    assert_eq!(table("[guess]\ntype = \"sad\"\n"), sad);
}

/// Setup refuses `[guess] from` beside `type`, of a shape other than
/// `n_ao × n_ao`, missing, or with an open shell, each naming the key, the
/// file and `from`'s line.
#[test]
fn a_guess_from_refusals_name_the_key_the_file_and_the_line() {
    density_file("guess_from_23.npy", &[23, 23]);
    density_file("guess_from_24.npy", &[24, 24]);
    let water = "[molecule]\nxyz = \"@data/h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n";
    let cation = "[molecule]\nxyz = \"@data/h2o.xyz\"\nmultiplicity = 2\ncharge = 1\n\
                  [basis]\nname = \"cc-pvdz\"\n";
    for (name, molecule, guess, line, kind, needle) in [
        (
            "guess_both.toml",
            water,
            "[guess]\ntype = \"sad\"\nfrom = \"guess_from_24.npy\"\n",
            7,
            ErrorKind::InvalidConfig,
            "[guess] type and from are both set",
        ),
        (
            "guess_shape.toml",
            water,
            "[guess]\n\nfrom = \"guess_from_23.npy\"\n",
            7,
            ErrorKind::InvalidConfig,
            "the array is [23, 23], not [24, 24]",
        ),
        (
            "guess_missing.toml",
            water,
            "[guess]\nfrom = \"no_such.npy\"\n",
            6,
            ErrorKind::Io,
            "[guess] from: cannot open",
        ),
        (
            "guess_open.toml",
            cation,
            "[guess]\nfrom = \"guess_from_24.npy\"\n",
            8,
            ErrorKind::Unsupported,
            "with an open shell (multiplicity 2)",
        ),
    ] {
        let (path, config) = read_config(name, &format!("{molecule}{guess}"));
        let Err(error) = MainBus::setup(config.unwrap(), Backing::Dry) else {
            panic!("{name}: refused");
        };
        assert_eq!(error.kind, kind, "{name}: {}", error.message);
        let at = format!("{}:{line}: [guess] ", path.display());
        assert!(error.message.starts_with(&at), "{name}: {}", error.message);
        assert!(error.message.contains(needle), "{name}: {}", error.message);
    }
}

/// A closed-shell ion, water's dication, is budgeted exactly as the
/// neutral: its dimensions differ in the occupations alone, and no step's
/// budget reads them but for whether the shell is open, so the table is
/// the same text, but for the host's available memory read at setup.
#[test]
fn a_closed_shell_ions_dry_budget_is_the_neutrals() {
    let budget = |charge: i32| {
        let text = format!(
            "[molecule]\nxyz = \"h2o.xyz\"\ncharge = {charge}\n[basis]\nname = \"cc-pvdz\"\n\
             [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n"
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let config = fb_steps::Config::parse(&text, &dir).unwrap();
        let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
        plan.budget(&Executor::new(2)).unwrap();
        let dims = plan.bus().shared().dims.clone();
        // The host's line reads its available memory at setup, which
        // moves between the two runs on a shared machine.
        let table: Vec<String> = plan
            .bus()
            .ledger()
            .table()
            .lines()
            .filter(|l| !l.starts_with("host: "))
            .map(str::to_owned)
            .collect();
        (dims, table)
    };
    let (neutral, neutral_table) = budget(0);
    let (ion, ion_table) = budget(2);
    assert_eq!((neutral.n_alpha, neutral.n_beta), (5, 5));
    assert_eq!((ion.n_alpha, ion.n_beta), (4, 4));
    let occupied_as_the_neutral = fb_steps::Dims {
        n_alpha: 5,
        n_beta: 5,
        ..ion
    };
    assert_eq!(occupied_as_the_neutral, neutral);
    assert_eq!(ion_table, neutral_table);
}

/// The localization's scratch at `n` AOs, `n_atom` atoms and blocks of
/// `widths` orbitals, `(device, pinned)` bytes: each block's in turn, the
/// input, `S C`, the population tensor and its pong, U and its pong, the
/// angles, the offsets, the schedule, `diag(ε) U`, `[L, G]` and `C U` on
/// the device; the input, the offsets, the schedule, `[L, G]`, `C U` and
/// the order pinned.
fn localization_scratch(n: usize, n_atom: usize, widths: &[usize]) -> (u64, u64) {
    let (mut dev, mut pin) = (Layout::new(), Layout::new());
    for &m in widths {
        block_scratch(&mut dev, &mut pin, n, n_atom, m);
    }
    (aligned(dev.bytes()), aligned(pin.bytes()))
}

/// One block's tenants of [`localization_scratch`].
fn block_scratch(dev: &mut Layout, pin: &mut Layout, n: usize, n_atom: usize, m: usize) {
    let (nm, m2, tensor) = (n * m, m * m, n_atom * m * m);
    let pad = m + m % 2;
    let schedule = (pad - 1) * pad;
    let _ = (dev.span::<f64>(nm + m), pin.span::<f64>(nm + m));
    let _ = (
        dev.span::<f64>(nm),
        dev.span::<f64>(tensor),
        dev.span::<f64>(tensor),
    );
    let _ = (dev.span::<f64>(m2), dev.span::<f64>(m2));
    let _ = (dev.span::<f64>(m), dev.span::<f64>(m), dev.span::<i32>(m));
    let _ = (dev.span::<i32>(n_atom + 1), dev.span::<i32>(schedule));
    let _ = (
        dev.span::<f64>(m2),
        dev.span::<f64>(1 + m2),
        dev.span::<f64>(nm),
    );
    let _ = (pin.span::<i32>(n_atom + 1), pin.span::<i32>(schedule));
    let _ = (
        pin.span::<f64>(1 + m2),
        pin.span::<f64>(nm),
        pin.span::<i32>(m),
    );
}

/// A `"localization"` run's dry budget is exact, its arrays sized by the
/// occupied blocks: water's five doubly occupied orbitals and its
/// dication's four, each step's scratch the formula's, `C_local`
/// `n_ao × n_beta` on the host; and every other step's region
/// the same for the two, since only the localization's budget reads more
/// of the occupations than whether the shell is open
/// (`docs/programming_model.md`, "The MainBus and its steps"). The water
/// cation's doublet is budgeted as its two blocks, four doubly occupied
/// columns then one singly, one after the other, and `C_local` as
/// `n_ao × n_alpha`.
#[test]
fn a_localizations_dry_budget_is_exact_in_the_occupied_block() {
    let budget = |charge: i32, multiplicity: i32| {
        let text = format!(
            "[molecule]\nxyz = \"h2o.xyz\"\ncharge = {charge}\nmultiplicity = {multiplicity}\n\
             [basis]\nname = \"cc-pvdz\"\n[eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n\
             [localization]\n"
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let config = fb_steps::Config::parse(&text, &dir).unwrap();
        let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
        let budgeted = plan.budget(&Executor::new(2));
        (plan, budgeted)
    };
    let mut regions = Vec::new();
    let mut heap = Vec::new();
    for (charge, m) in [(0, 5), (2, 4)] {
        let (plan, budgeted) = budget(charge, 1);
        budgeted.unwrap();
        let dims = plan.bus().shared().dims.clone();
        assert_eq!((dims.n_atom, dims.n_beta), (3, m), "charge {charge}");
        let ledger = plan.bus().ledger();
        let region = |kind: Kind| {
            ledger
                .regions()
                .iter()
                .find(|r| r.gpu == 0 && r.kind == kind && r.step == "localization")
                .map(|r| r.bytes)
        };
        let (device, pinned) = localization_scratch(dims.n_ao, dims.n_atom, &[m]);
        assert_eq!(region(Kind::Device), Some(device), "charge {charge}");
        assert_eq!(region(Kind::Pinned), Some(pinned), "charge {charge}");
        assert_eq!(ledger.allocations(), 0, "a dry budget allocates nothing");
        assert!(ledger.table().starts_with("memory budget: accepted"));
        let mut others: Vec<(&str, Kind, u64)> = ledger
            .regions()
            .iter()
            .filter(|r| r.step != "localization")
            .map(|r| (r.step, r.kind, r.bytes))
            .collect();
        others.sort_by_key(|r| (r.0, r.2));
        regions.push(others);
        heap.push((ledger.heap_total(), dims.n_ao));
    }
    assert_eq!(regions[0], regions[1], "the other steps' scratch");
    // C_local's host copy, n_ao × n_beta, the one heap difference.
    assert_eq!(heap[0].0 - heap[1].0, (heap[0].1 * 8) as u64);

    let (plan, budgeted) = budget(1, 2);
    budgeted.unwrap();
    let dims = plan.bus().shared().dims.clone();
    assert_eq!((dims.n_alpha, dims.n_beta), (5, 4));
    let ledger = plan.bus().ledger();
    let region = |kind: Kind| {
        ledger
            .regions()
            .iter()
            .find(|r| r.gpu == 0 && r.kind == kind && r.step == "localization")
            .map(|r| r.bytes)
    };
    let (device, pinned) = localization_scratch(dims.n_ao, dims.n_atom, &[4, 1]);
    assert_eq!(region(Kind::Device), Some(device), "the cation");
    assert_eq!(region(Kind::Pinned), Some(pinned), "the cation");
    let local = plan.output(fb_steps::LocalizationStep);
    assert_eq!(local.n_cols(), 5);
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

/// `xyz`'s four host steps in cc-pVDZ / cc-pVDZ-JKFIT, budgeted and
/// executed once on a host-backed plan under the allocation record: no
/// step's `run` makes an allocation past its threshold on the thread that
/// runs it. The record does not watch the pool's threads, so each host
/// integral is then run again in the pool into a buffer of the test's own,
/// watched on the thread that `install` runs it on, where its unshared
/// work is: none past the threshold either.
#[allow(
    clippy::disallowed_macros,
    reason = "a test's buffer for the host integrals, outside any step's run"
)]
fn host_steps_pass_the_record(xyz: &str) {
    let config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    let mut dag = Dag::new();
    dag.add(OverlapStep)
        .add(CoreHamiltonianStep)
        .add(GuessStep)
        .add(MetricStep);
    let plan = dag
        .build(MainBus::setup(config, Backing::Host).unwrap())
        .unwrap();
    let executor = Executor::new(plan.bus().shared().config.resources.threads);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();

    let shared = plan.bus().shared();
    let (n, q) = (shared.dims.n_ao, shared.dims.n_aux);
    let aux = shared.aux.as_ref().unwrap();
    let charges = fb_integrals_v_potential_en_host::point_charges(&shared.molecule);
    let mut s = vec![0.0; n * (n + 1) / 2];
    let mut m = vec![0.0; q * (q + 1) / 2];
    let ((), largest) = shared.host_pool().install(|| {
        fb_core_budget::AllocationRecord::watch(|| {
            fb_integrals_s_overlap_host::overlap_into(&shared.basis, &mut s);
            fb_integrals_t_kinetic_host::kinetic_into(&shared.basis, &mut s);
            fb_integrals_v_potential_en_host::potential_en_into(&shared.basis, &charges, &mut s);
            fb_integrals_df_metric::metric_into(aux, &mut m);
        })
    });
    assert_eq!(largest, None, "{xyz}: a host integral's largest allocation");
    println!(
        "{xyz}: n_ao {n}, n_aux {q}: the host steps and their integrals pass the record (packed S {:.2} MiB, (P|Q) {:.2} MiB)",
        (s.len() * 8) as f64 / f64::from(1 << 20),
        (m.len() * 8) as f64 / f64::from(1 << 20)
    );
}

/// Benzene, 114 AOs: the family's representative, its triangles under the
/// record's threshold even had they been returned.
#[test]
fn benzenes_host_steps_pass_the_allocation_record() {
    host_steps_pass_the_record("benzene.xyz");
}

/// gly10, 734 AOs and 3606 fitting functions: every triangle past the
/// threshold (S's 2.06 MiB, (P|Q)'s 49.6 MiB), so a step that returned
/// one from its library would fail. Slow: the metric's host build.
#[test]
fn gly10s_host_steps_pass_the_allocation_record() {
    if fast_tests() {
        return;
    }
    host_steps_pass_the_record("gly10.xyz");
}

/// amdgpu's lines, as kernel 6.8 writes them for one render node.
fn fdinfo(id: u32, vram_kib: u64) -> String {
    format!(
        "pos:\t0\nflags:\t02100002\nmnt_id:\t26\nino:\t1033\ndrm-driver:\tamdgpu\n\
         drm-pdev:\t0000:03:00.0\ndrm-client-id:\t{id}\npasid:\t32790\n\
         drm-memory-vram:\t{vram_kib} KiB\ndrm-memory-gtt: \t2048 KiB\n\
         drm-memory-cpu: \t0 KiB\n"
    )
}

/// Two clients summed, a client's second descriptor counted once, and
/// every other file passed over.
#[test]
fn sums_each_amdgpu_client_once() {
    let other = "pos:\t0\nflags:\t02\ndrm-driver:\tnvidia\ndrm-client-id:\t3\n".to_owned();
    let files = [
        fdinfo(7, 100),
        fdinfo(7, 100),
        fdinfo(9, 28),
        other,
        "pos:\t0\n".to_owned(),
    ];
    assert_eq!(common::amdgpu_vram(files), Some(128 << 10));
}

/// No amdgpu file: no reading, not zero.
#[test]
fn no_amdgpu_file_is_none() {
    assert_eq!(common::amdgpu_vram(["pos:\t0\n".to_owned()]), None);
}

/// Each unit, and a bare count.
#[test]
fn drm_memory_units() {
    assert_eq!(common::bytes_of("5"), 5);
    assert_eq!(common::bytes_of("5 KiB"), 5 << 10);
    assert_eq!(common::bytes_of("5 MiB"), 5 << 20);
    assert_eq!(common::bytes_of("5 GiB"), 5 << 30);
}
