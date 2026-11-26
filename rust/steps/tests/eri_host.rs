//! The ERI step's budget with no device (tier 3): `[eri] mem_per_gpu` one
//! byte under the least that holds a batch is refused, naming the key, its
//! value and the bytes needed, and the least itself is accepted; a ring
//! of neither 1 nor 2 is refused.

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
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
