//! The arena's blocks on a real card. Every case needs a GPU (device 0).
//!
//! A binary of its own, its cases serialized by a lock: two of them measure
//! the card's free memory, which another case allocating beside them would
//! move. The abort case re-runs this binary as a child (selected by
//! `FB_ARENA_CHILD`), as `tests/log.rs` does, because what it checks is
//! what survives the process ending.

use std::process::Command;
use std::sync::Mutex;

use fb_core_arena::{Block, Device, Pinned, TableBuilder};
use fb_core_wwr::common::ScopedDeviceIndex;
use wwr_sys::{wwr_memcpy_kind, wwr_status};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

const MIB: usize = 1 << 20;

#[track_caller]
fn ok(status: wwr_status) {
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
}

/// Device 0's free memory, with it made current.
fn free_bytes() -> usize {
    let _device = ScopedDeviceIndex::new(0);
    let (mut free, mut total) = (0, 0);
    // SAFETY: a plain runtime call writing two locals.
    ok(unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) });
    free
}

/// Make the context exist, so the measurements below see the block alone.
fn warm_up() {
    let mut b = TableBuilder::<Device>::new();
    let _ = b.output::<u8>(0, "warm", 1);
    drop(Block::new(0, b.build()));
}

/// The card gives back the bytes it was asked for, within the driver's
/// rounding: declared equals allocated, and the drop frees it all. The one
/// check that the table's total is the whole of the block's cost.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn declared_equals_allocated() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    warm_up();
    let bytes = 256 * MIB;
    let slack = 4 * MIB;
    let mut b = TableBuilder::<Device>::new();
    let _ = b.output::<u8>(0, "big", bytes);
    let table = b.build();
    assert_eq!(table.total(), bytes);

    let before = free_bytes();
    let block = Block::new(0, table);
    let during = free_bytes();
    drop(block);
    let after = free_bytes();
    let taken = before - during;
    assert!(
        (bytes..=bytes + slack).contains(&taken),
        "the block took {taken} bytes for a table of {bytes}"
    );
    assert!(
        after + slack >= before,
        "the drop gave back {} of {taken} bytes",
        after.saturating_sub(during)
    );
}

/// The fill reaches every byte before anyone writes: a pinned f64 row is
/// all NaN in every profile, and a device row reads back all `0xFF` in a
/// debug build. The check that a row read before it is written fails.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn blocks_are_filled() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let n = 1000;
    let mut p = TableBuilder::<Pinned>::new();
    let pr = p.output::<f64>(0, "P", n);
    let pinned = Block::new(0, p.build());
    assert!(pinned.partition(0).get(pr).iter().all(|x| x.is_nan()));

    if !cfg!(debug_assertions) {
        return;
    }
    let mut d = TableBuilder::<Device>::new();
    let dr = d.output::<u64>(0, "D", n);
    let device = Block::new(0, d.build());
    let v = device.partition(0);
    let row = v.get(dr);
    let mut host = vec![0u64; n];
    let _device = ScopedDeviceIndex::new(0);
    // SAFETY: n u64s from the device row into a local of the same size.
    ok(unsafe {
        wwr_sys::wwr_memcpy(
            host.as_mut_ptr().cast(),
            row.as_ptr().cast(),
            n * 8,
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        )
    });
    assert!(host.iter().all(|&x| x == u64::MAX));
}

/// What the child does, by `FB_ARENA_CHILD`; nothing when unset.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn child_entry() {
    if std::env::var("FB_ARENA_CHILD").as_deref() == Ok("too_big") {
        let mut b = TableBuilder::<Device>::new();
        let _ = b.output::<u8>(0, "huge", 1 << 50);
        let _block = Block::new(0, b.build()); // MARK too_big
    }
}

/// A block the card cannot hold aborts at the caller, as every create
/// does, with a critical line naming the kind, the bytes and the memory
/// free.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn an_allocation_the_card_cannot_hold_aborts() {
    use std::os::unix::process::ExitStatusExt;
    let out = Command::new(std::env::current_exe().unwrap())
        // The child's case is a device case too: --include-ignored runs it.
        .args([
            "--exact",
            "child_entry",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("FB_ARENA_CHILD", "too_big")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.signal(), Some(libc::SIGABRT), "{:?}", out.status);
    let line = include_str!("arena_block.rs")
        .lines()
        .position(|l| l.contains("MARK too_big") && !l.contains("contains("))
        .unwrap()
        + 1;
    for needle in [
        format!("[critical] [arena_block.rs:{line}]"),
        "the device block of 1125899906842624 bytes on device 0".to_owned(),
        "bytes free of".to_owned(),
    ] {
        assert!(stdout.contains(&needle), "{needle:?} not in {stdout}");
    }
}
