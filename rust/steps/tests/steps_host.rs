//! The MainBus and the steps' budget with no device (tier 3): setup's
//! counts on gly10 in cc-pVDZ / cc-pVDZ-JKFIT against the v0.0.4 page's,
//! the dry budget's exact bytes from them, the orthogonalizer's scratch
//! apart from every step unordered with it, and the config's refusals.

mod common;

use fb_core_budget::{Backing, Kind, Layout};
use fb_core_device::HANDLE_WORKSPACE_BYTES;
use fb_core_error_handling::ErrorKind;
use fb_core_flow::{Bus, Executor};
use fb_steps::{MainBus, plan};

const ALIGN: u64 = 256;

fn aligned(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(ALIGN) * ALIGN
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
    for step in ["overlap", "core_hamiltonian", "guess"] {
        assert_eq!(region(step, Kind::Pinned), Some(square(n)), "{step}");
    }
    assert_eq!(region("metric", Kind::Pinned), Some(square(q)));
    // Four n × n outputs and the metric's q × q, the one device scratch
    // region, and the handle workspace.
    assert_eq!(
        ledger.device_total(0),
        (4 * n * n * 8 + q * q * 8) as u64 + aligned(l.bytes()) + HANDLE_WORKSPACE_BYTES as u64
    );
    assert!(ledger.table().starts_with("memory budget: accepted"));
    // Pinned: the three host steps unordered with it. Device: no other.
    assert_eq!(common::orthogonalizer_scratch_is_its_own(ledger), 3);
}

#[test]
fn a_run_of_no_gpu_and_a_table_of_no_step_here_are_refused() {
    let mut config = common::config("h2o.xyz", "cc-pvdz", None);
    config.resources.devices.clear();
    let Err(error) = MainBus::setup(config, Backing::Dry) else {
        panic!("a run of no GPU");
    };
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n[scf]\n";
    let error = fb_steps::Config::parse(text, std::path::Path::new(".")).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
}
