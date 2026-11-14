//! Admission on a real card (device 0): a device cap past the card is
//! bounded by its free memory at planning, read by
//! [`fb_core_arena::device_free`].
//!
//! The card's free memory moves with its other tenants, so nothing here
//! asserts it equals anything: a block of a PiB is past every card, and
//! one of a MiB within any card the run can use at all.

use fb_core_error_handling::ErrorKind;
use fb_core_planner::{Dag, Planned};
use fb_pipeline_config::{MemoryCaps, admit_plan};

/// A plan of one device worker row of `bytes`.
fn device_row(bytes: usize) -> Planned<'static> {
    let mut dag = Dag::new();
    let _ = dag.device_worker_row::<u8>(0, "device row", bytes);
    dag.plan()
}

/// Under a cap of 2^62 bytes, past every card, a block of a PiB is refused
/// by the card's free memory, `Allocation`, naming it; one of a MiB is
/// admitted. Without the bound, the cap alone would admit both. Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_cap_past_the_card_is_bounded_by_its_free_memory() {
    let free = fb_core_arena::device_free(0).unwrap();
    println!("device 0: {free} bytes free");
    assert!(free > 0, "the card has free memory");
    let caps = MemoryCaps {
        max_device_mem_per_gpu: 1 << 62,
        ..MemoryCaps::default()
    };
    let card = |_| fb_core_arena::device_free(0);
    let error = admit_plan(&device_row(1 << 50), &caps, card, usize::MAX).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    assert!(error.message.contains("bytes free at planning"), "{error}");
    assert!(
        error.message.contains("over the cap alone: device row"),
        "{error}"
    );
    admit_plan(&device_row(1 << 20), &caps, card, usize::MAX).unwrap();
}
