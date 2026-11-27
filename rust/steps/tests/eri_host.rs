//! The ERI step's budget with no device (tier 3): `[eri] mem_per_gpu` one
//! byte under the least that holds a batch is refused, naming the key, its
//! value and the bytes needed, and the least itself is accepted; a ring
//! of neither 1 nor 2 is refused; the budget at the default `[eri]
//! schwarz_threshold` is the budget at 0, and a threshold under 0 or not a
//! number is refused.

mod common;

use fb_core_budget::Backing;
use fb_core_error_handling::ErrorKind;
use fb_core_flow::{Bus, Executor};
use fb_steps::{EriStep, MainBus, plan};

/// Water's plan with `[eri]` given `mem_per_gpu` and `ring`, budgeted dry.
fn budget(mem_per_gpu: u64, ring: usize) -> fb_core_error_handling::Status {
    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let eri = config.eri.as_mut().unwrap();
    (eri.mem_per_gpu, eri.ring) = (Some(mem_per_gpu), ring);
    plan(MainBus::setup(config, Backing::Dry).unwrap())
        .unwrap()
        .budget(&Executor::new(2))
}

#[test]
fn a_mem_per_gpu_a_byte_under_the_least_is_refused_and_the_least_accepted() {
    let config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let dims = MainBus::setup(config, Backing::Dry)
        .unwrap()
        .shared()
        .dims
        .clone();
    for ring in [1, 2] {
        let least = EriStep::least_mem_per_gpu(&dims, ring);
        let error = budget(least - 1, ring).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
        let named = format!("[eri] mem_per_gpu = {} holds no batch", least - 1);
        assert!(error.message.contains(&named), "{}", error.message);
        assert!(
            error.message.contains(&format!("need {least} bytes")),
            "{}",
            error.message
        );
        budget(least, ring).unwrap();
    }
}

#[test]
fn a_ring_of_neither_one_nor_two_is_refused() {
    let error = budget(1 << 30, 3).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(
        error.message.contains("[eri] ring = 3"),
        "{}",
        error.message
    );
}

/// Benzene's dry budget at the default `[eri] schwarz_threshold` is its
/// budget at 0, row for row of the table (its caps' rows aside): B and the build's ring stay at the
/// unscreened bound `Dims` gives (`docs/memory.md`, "Batched only").
#[test]
fn the_dry_budget_is_the_same_at_the_default_threshold_and_at_0() {
    let table = |threshold: Option<f64>| {
        let mut config = common::config("benzene.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
        let eri = config.eri.as_mut().unwrap();
        assert_eq!(eri.schwarz_threshold, 1e-12, "the default");
        if let Some(t) = threshold {
            eri.schwarz_threshold = t;
        }
        let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
        plan.budget(&Executor::new(2)).unwrap();
        let ledger = plan.bus().ledger();
        // Every row of the table but its caps', which read the machine's
        // free memory as it is now.
        let rows: Vec<String> = ledger
            .table()
            .lines()
            .filter(|l| !l.contains(" against "))
            .map(str::to_owned)
            .collect();
        (rows, ledger.device_total(0), ledger.heap_total())
    };
    assert_eq!(table(None), table(Some(0.0)));
}

/// A threshold under 0, or not a number, is refused, naming the key.
#[test]
fn a_negative_or_nan_threshold_is_refused() {
    for t in [-1e-12, f64::NAN] {
        let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
        config.eri.as_mut().unwrap().schwarz_threshold = t;
        let error = plan(MainBus::setup(config, Backing::Dry).unwrap())
            .unwrap()
            .budget(&Executor::new(2))
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
        assert!(
            error.message.contains("[eri] schwarz_threshold"),
            "{}",
            error.message
        );
    }
}
