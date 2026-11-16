//! Admission on a real card (device 0), as the binary makes it: the card
//! read by [`fb_core_arena::device_memory`] with the worker's stream and
//! handles made, and the kernels' stack by
//! [`fb_core_wwr::runtime::stack_growth`].
//!
//! The card's free memory moves with its other tenants, so nothing here
//! asserts it equals anything: a block of a PiB is past every card, one of
//! a MiB within any card the run can use at all, and the measured reserve
//! is held to its floor from one side.

use fb_core_error_handling::ErrorKind;
use fb_core_planner::{Dag, Plan};
use fb_core_thread_pool::DeviceStream;
use fb_pipeline_config::{DEVICE_RESERVE, KERNEL_STACK_BYTES, MemoryCaps, admit_plan};

/// A plan of one device worker row of `bytes`.
fn device_row(bytes: usize) -> Plan<'static> {
    let mut dag = Dag::new();
    let _ = dag.device_worker_row::<u8>(0, "device row", bytes);
    dag.plan()
}

/// The bytes the refusal `message` names right after `term`.
fn bytes_after(message: &str, term: &str) -> usize {
    let at = message
        .find(term)
        .unwrap_or_else(|| panic!("{term}: {message}"))
        + term.len();
    let digits: String = message[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("{term}: {message}"))
}

/// Under a cap of 2^62 bytes, past every card, a block of a PiB is refused
/// by the card, `Allocation`, naming its free memory, the measured reserve
/// and the stack; one of a MiB is admitted. Without the bound, the cap
/// alone would admit both. The reserve is at least [`DEVICE_RESERVE`] and
/// short of the card. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_cap_past_the_card_is_bounded_by_it() {
    let card = |_| {
        let _worker = DeviceStream::new(0);
        fb_core_arena::device_memory(0)
    };
    let stack = |_| fb_core_wwr::runtime::stack_growth(0, KERNEL_STACK_BYTES);
    let (free, total) = card(0).unwrap();
    println!("device 0: {free} bytes free of {total}");
    assert!(
        free > 0 && free < total,
        "the card has memory free and held"
    );
    let caps = MemoryCaps {
        max_device_mem_per_gpu: 1 << 62,
        ..MemoryCaps::default()
    };
    let error = admit_plan(&device_row(1 << 50), &caps, card, stack, usize::MAX).unwrap_err();
    println!("{error}");
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    assert!(error.message.contains("bytes free at planning"), "{error}");
    assert!(
        error.message.contains("over the cap alone: device row"),
        "{error}"
    );
    let reserve = bytes_after(&error.message, "the reserve of ");
    assert!(reserve >= DEVICE_RESERVE, "{error}");
    assert!(
        reserve < bytes_after(&error.message, "the card's "),
        "{error}"
    );
    let _ = bytes_after(&error.message, "the kernels' stack of ");
    admit_plan(&device_row(1 << 20), &caps, card, stack, usize::MAX).unwrap();
}
