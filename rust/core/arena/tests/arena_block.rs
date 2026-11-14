//! The arena's blocks on a real card. Every case needs a GPU (device 0).
//!
//! A binary of its own, its cases serialized by a lock: two of them measure
//! the card's free memory, which another case allocating beside them would
//! move. The abort case re-runs this binary as a child (selected by
//! `FB_ARENA_CHILD`), as `tests/log.rs` does, because what it checks is
//! what survives the process ending.

use std::process::Command;
use std::sync::Mutex;

use fb_core_arena::{
    Block, Device, Pinned, TableBuilder, memcpy_d2d, memcpy_d2p, memcpy_h2p, memcpy_p2d, memcpy_p2h,
};
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::runtime::Stream;
use wwr_sys::wwr_status;

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
    // A pinned row to read it into, zeroed first, since its own fill is the
    // same bytes: only the copy can put `u64::MAX` back.
    let mut h = TableBuilder::<Pinned>::new();
    let hr = h.output::<u64>(0, "H", n);
    let host = Block::new(0, h.build());
    memcpy_h2p(&host.partition(0), hr, &vec![0; n]);
    let stream = Stream::new(0);
    memcpy_d2p(&host.partition(0), hr, v.get(dr), &stream).unwrap();
    ok(stream.sync());
    let got = memcpy_p2h(&host.partition(0), hr);
    assert!(got.iter().all(|&x| x == u64::MAX));
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

/// A value goes up and comes back through every copy: in from a `Vec`
/// (h2p), up on a stream (p2d), across the card (d2d), down (d2p) and out
/// (p2h), each node's rows its own. And a stream copy whose rows' lengths
/// differ is refused before anything is enqueued.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn copies_round_trip_through_every_kind() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let n = 4097;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 7.0).collect();

    // Node 0 stages the input, node 1 moves it, node 2 reads the result.
    let mut p = TableBuilder::<Pinned>::new();
    let input = p.output::<f64>(0, "in", n);
    p.read(1, input);
    let output = p.output::<f64>(1, "out", n);
    p.read(2, output);
    let short = p.output::<f64>(1, "short", n - 1);
    let mut d = TableBuilder::<Device>::new();
    let first = d.output::<f64>(1, "first", n);
    let second = d.output::<f64>(1, "second", n);
    let pinned = Block::new(0, p.build());
    let device = Block::new(0, d.build());
    let stream = Stream::new(0);

    memcpy_h2p(&pinned.partition(0), input, &data);
    {
        let (h, dv) = (pinned.partition(1), device.partition(1));
        let (a, b) = (dv.take_mut(first), dv.take_mut(second));
        memcpy_p2d(&a, &h, input, &stream).unwrap();
        memcpy_d2d(&b, &a, &stream).unwrap();
        memcpy_d2p(&h, output, &b, &stream).unwrap();
        ok(stream.sync());

        let m = fb_test_support::panic_message(|| {
            let _ = memcpy_d2p(&h, short, &b, &stream);
        });
        assert!(m.contains("the rows' lengths differ"), "{m}");
    }
    assert_eq!(memcpy_p2h(&pinned.partition(2), output), data);
}

/// Needs a GPU. A held worker row is the row's bytes and length, keeps its
/// block alive past the block's last other owner, and is its only user: a
/// view asking for it, or a second hold, panics until it drops. A node's
/// row cannot be held at all.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_held_worker_row_is_alone_and_keeps_its_block() {
    use fb_core_wwr::device_slice::DeviceSlice;
    use fb_test_support::panic_message;
    use std::sync::Arc;

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut d = TableBuilder::<Device>::new();
    let out = d.output::<f64>(0, "out", 64);
    let ws = d.worker::<u8>("ws", MIB);
    let block = Arc::new(Block::new(0, d.build()));
    assert!(panic_message(|| drop(block.hold(out))).contains("out is not a worker row"));

    let held = block.hold(ws);
    assert_eq!(held.len(), MIB);
    assert_eq!(held.dev_idx(), 0);
    assert!(panic_message(|| drop(block.hold(ws))).contains("ws is already held"));
    assert!(
        panic_message(|| {
            block.partition(0).take_mut(ws);
        })
        .contains("ws is held by another node's view")
    );
    let at = held.as_ptr().addr();
    drop(held);
    assert_eq!(block.partition(0).take_mut(ws).as_ptr().addr(), at);

    let held = block.hold(ws);
    let weak = Arc::downgrade(&block);
    drop(block);
    assert!(weak.upgrade().is_some(), "the held row let its block go");
    drop(held);
    assert!(weak.upgrade().is_none());
}
