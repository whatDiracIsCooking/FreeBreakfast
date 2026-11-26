//! The device-free tests: the ledger on the plain heap ([`Backing::Host`]),
//! from v0.0.4's `tests/core.rs` with the steps' budget calls made
//! directly, since the step DAG is not in this crate.
//!
//! Five fake steps: `a`; `b` and `c`, which read `a`; `d`, which reads
//! `b` and `c`; and `e`, which reads nothing and so is ordered against no
//! one. Each owns 1 MiB of device `Output` and declares device scratch of
//! 4, 3, 2, 6 and 1 MiB.

use super::*;
use fb_core_error_handling::{ErrorKind, Status};

const MIB: u64 = 1 << 20;
const N: usize = 5;
const NAMES: [&str; N] = ["a", "b", "c", "d", "e"];
const PREDS: [&[usize]; N] = [&[], &[0], &[0], &[1, 2], &[]];
const SCRATCH: [u64; N] = [4 * MIB, 3 * MIB, 2 * MIB, 6 * MIB, MIB];
const ROOMY: u64 = 1 << 40;

fn cap(used: u64) -> Cap {
    Cap {
        written: Some(used),
        used,
    }
}

fn ledger(device_cap: u64, host_cap: u64) -> Ledger {
    Ledger::new(
        Caps {
            device: vec![cap(device_cap)],
            host: cap(host_cap),
        },
        Backing::Host,
        vec![0],
        false,
    )
}

/// Step `i`'s `mem_budget`: its device `Output`, its heap and its scratch.
fn budget_step(ledger: &Ledger, i: usize, heap: u64) -> (DeviceBuf<f64>, HeapSlab<f64>) {
    let budget = ledger.budget(NAMES[i]);
    let mut layout = Layout::new();
    let _ = layout.span::<f64>(SCRATCH[i] as usize / 8);
    budget.scratch(0, Kind::Device, layout.bytes());
    budget.locks([LockId::Gpu(0)]);
    (
        budget.device(0, MIB as usize / 8),
        budget.heap(heap as usize / 8),
    )
}

/// Budget every step in `order`, then the final node over the steps named
/// in `names_order`: the DAG's own order, which the offsets must not
/// depend on either.
fn run(ledger: &Ledger, order: &[usize], names_order: &[usize]) -> Status {
    let outputs: Vec<_> = order.iter().map(|&i| budget_step(ledger, i, 0)).collect();
    let names: Vec<&'static str> = names_order.iter().map(|&i| NAMES[i]).collect();
    let preds: Vec<Vec<usize>> = names_order
        .iter()
        .map(|&i| {
            PREDS[i]
                .iter()
                .map(|p| names_order.iter().position(|q| q == p).unwrap())
                .collect()
        })
        .collect();
    let status = ledger.finish(&names, &preds);
    drop(outputs);
    status
}

/// Every permutation of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for p in permutations(n - 1) {
        for at in 0..=p.len() {
            let mut q = p.clone();
            q.insert(at, n - 1);
            out.push(q);
        }
    }
    out
}

#[test]
fn bytes_are_printed_in_binary_units() {
    assert_eq!(bytes(512), "512 B");
    assert_eq!(bytes(1536), "1.50 KiB");
    assert_eq!(bytes(3 << 30), "3.00 GiB");
    assert_eq!(bytes(u64::MAX), "none");
}

#[test]
fn a_cap_is_clamped_to_the_reading() {
    let gib = 1u64 << 30;
    // Defaulted: the reading less the margin.
    assert_eq!(Cap::resolve(None, Some(10 * gib), gib).used, 9 * gib);
    // Explicit and within the machine: as written.
    assert_eq!(
        Cap::resolve(Some(4 * gib), Some(10 * gib), gib).used,
        4 * gib
    );
    // Explicit and past it: the reading itself, no margin taken.
    assert_eq!(
        Cap::resolve(Some(64 * gib), Some(10 * gib), gib).used,
        10 * gib
    );
    // No reading: as written, or none.
    assert_eq!(Cap::resolve(Some(gib), None, gib).used, gib);
    assert_eq!(Cap::resolve(None, None, gib).used, u64::MAX);
    assert_eq!(device_margin(10 * gib), 2 * gib);
    assert_eq!(device_margin(40 * gib), 2 * gib);
    assert_eq!(device_margin(48 * gib), 48 * gib / 20);
    assert_eq!(host_margin(8 * gib), gib);
    assert_eq!(host_margin(64 * gib), 64 * gib / 20);
}

#[test]
fn a_host_cap_past_the_machine_is_clamped_and_the_table_prints_both() {
    let written = 1u64 << 50;
    let caps = Caps::read(&[], &[], Some(written), Backing::Host).unwrap();
    let reading = host_available().unwrap();
    // The host moves between two readings; the clamp is to the first.
    assert!(caps.host.used < written && caps.host.used > 0);
    assert!(
        caps.host.used.abs_diff(reading) < reading / 10,
        "{} against {reading}",
        caps.host.used
    );
    assert_eq!(caps.host.written, Some(written));
    let ledger = Ledger::new(caps.clone(), Backing::Host, vec![], false);
    ledger.finish(&[], &[]).unwrap();
    let line = format!(
        "against {} used ({} as written)",
        bytes(caps.host.used),
        bytes(written)
    );
    assert!(ledger.table().contains(&line), "{}", ledger.table());
}

#[test]
fn a_cgroup_limit_bounds_the_host_reading() {
    let dir = std::env::temp_dir().join(format!("fb-budget-cgroup-{}", std::process::id()));
    let leaf = dir.join("a/b");
    std::fs::create_dir_all(&leaf).unwrap();
    std::fs::write(dir.join("a/memory.max"), "1048576\n").unwrap();
    std::fs::write(dir.join("a/memory.current"), "262144\n").unwrap();
    std::fs::write(leaf.join("memory.max"), "max\n").unwrap();
    std::fs::write(leaf.join("memory.current"), "4096\n").unwrap();
    let meminfo = "MemTotal: 100 kB\nMemAvailable:    8192 kB\n";
    assert_eq!(
        host::available_under(meminfo, &dir, Some("0::/a/b\n")).unwrap(),
        1_048_576 - 262_144
    );
    assert_eq!(
        host::available_under(meminfo, &dir, None).unwrap(),
        8192 * 1024
    );
    assert!(host::available_under("MemTotal: 1 kB\n", &dir, None).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_fake_graph_packs_to_its_lower_bound() {
    let ledger = ledger(ROOMY, ROOMY);
    run(&ledger, &[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    let slab = ledger
        .slabs()
        .into_iter()
        .find(|s| s.kind == Kind::Device)
        .unwrap();
    // a, then d, lie on one path and share from offset 0; b and c are
    // unordered with each other; e is unordered with everyone. The
    // heaviest antichain is {d, e} = 7 MiB, and {b, c, e} = 6.
    assert_eq!(slab.sum, 16 * MIB);
    assert_eq!(slab.bound, 7 * MIB);
    assert_eq!(slab.bound_steps, ["d", "e"]);
    assert_eq!(slab.peak, slab.bound);
    let at = |step: &str| {
        let r = ledger
            .regions()
            .into_iter()
            .find(|r| r.step == step)
            .unwrap();
        (r.offset, r.bytes)
    };
    assert_eq!(at("d"), (0, 6 * MIB));
    assert_eq!(at("a"), (0, 4 * MIB));
    assert_eq!(at("e"), (6 * MIB, MIB));
    // The outputs are a sum, the scratch the packed peak, and the fixed
    // BLAS-handle workspace is on top.
    assert_eq!(
        ledger.device_total(0),
        5 * MIB + slab.peak + fb_core_device::HANDLE_WORKSPACE_BYTES as u64
    );
    assert_eq!(
        ledger.locks_of("a"),
        [LockId::Gpu(0)],
        "a lock declared once is held once"
    );
}

#[test]
fn accept_refuse_and_the_offsets_are_the_same_under_any_budget_order() {
    // Outputs 5 MiB, packed scratch 7, workspace 128: 140 MiB in all.
    let need = 140 * MIB;
    let reference = {
        let ledger = ledger(need, ROOMY);
        run(&ledger, &[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
        (ledger.regions(), ledger.table())
    };
    let refusal = {
        let ledger = ledger(need - 1, ROOMY);
        run(&ledger, &[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4])
            .unwrap_err()
            .message
    };
    let orders = permutations(N);
    assert_eq!(orders.len(), 120);
    for (k, order) in orders.iter().enumerate() {
        // The DAG's own order moves too, a different permutation each time.
        let names = &orders[(k * 7 + 3) % orders.len()];
        let accepted = ledger(need, ROOMY);
        run(&accepted, order, names).expect("a plan at its cap is accepted");
        assert_eq!(accepted.device_total(0), need);
        assert_eq!(accepted.regions(), reference.0, "order {order:?}");
        assert_eq!(accepted.table(), reference.1, "order {order:?}");

        let refused = ledger(need - 1, ROOMY);
        let error = run(&refused, order, names).expect_err("a plan one byte over is refused");
        assert_eq!(error.kind, ErrorKind::Allocation);
        assert_eq!(error.message, refusal, "order {order:?}");
    }
    // The whole table, not whichever step crossed the line: every step's
    // row, the packed scratch, its lower bound, the workspace, the totals.
    for line in [
        "a ",
        "b ",
        "c ",
        "d ",
        "e ",
        "scratch (packed)",
        "lower bound",
        "blas workspace",
        "total",
        "cap used",
    ] {
        assert!(
            refusal
                .lines()
                .any(|l| l.starts_with(line) || l.trim_start().starts_with(line)),
            "the refusal lacks '{line}':\n{refusal}"
        );
    }
    assert!(refusal.contains("memory budget: REFUSED"), "{refusal}");
}

#[test]
fn over_the_cap_is_a_refusal_with_the_table_not_a_panic() {
    // Over at the first step's charge: every later charge is counted and
    // allocates nothing, and the final node refuses.
    let ledger = ledger(MIB / 2, ROOMY);
    let (device, _) = budget_step(&ledger, 0, 0);
    assert!(ledger.over());
    assert!(!device.is_allocated());
    let error = ledger.finish(&["a"], &[vec![]]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation);
    assert!(error.message.contains("REFUSED"), "{error}");
    assert!(error.message.contains("\na "), "{error}");
    assert_eq!(ledger.allocations(), 0);
}

#[test]
fn a_heap_slab_cannot_grow_and_an_over_cap_one_is_never_touched() {
    let ledger = ledger(ROOMY, ROOMY);
    let (_, slab) = budget_step(&ledger, 0, MIB);
    // A boxed slice: a length and no capacity, no push, no reserve.
    let slab: &[f64] = &slab;
    assert_eq!(slab.len(), MIB as usize / 8);
    assert_eq!(ledger.heap_total(), MIB);

    // The host cap is checked at the call, before the slab exists: a heap
    // past the cap is charged, flagged, and never allocated.
    let refused = self::ledger(ROOMY, MIB - 1);
    let (_, slab) = budget_step(&refused, 0, MIB);
    assert_eq!(slab.len(), 0, "never made, never touched");
    assert_eq!(refused.heap_total(), MIB, "and still counted");
    assert!(refused.over());
}

#[test]
fn a_lease_carves_its_region_and_keeps_tenants_apart() {
    let ledger = ledger(ROOMY, ROOMY);
    let mut layout = Layout::new();
    let first = layout.span::<f64>(3);
    let second = layout.span::<i32>(5);
    assert_eq!(layout.bytes(), 64 + 20, "each tenant on a 64-byte boundary");
    ledger.budget("a").scratch(0, Kind::Pinned, layout.bytes());
    ledger.budget("a").scratch(0, Kind::Device, 8);
    ledger.finish(&["a"], &[vec![]]).unwrap();

    let lease = ledger.lease("a");
    assert_eq!(
        lease.bytes(0, Kind::Pinned),
        256,
        "rounded to the alignment"
    );
    let mut f = lease.pinned(0, first);
    let mut i = lease.pinned(0, second);
    f.fill(1.5);
    i.fill(-2);
    assert_eq!((&f[..], &i[..]), (&[1.5; 3][..], &[-2; 5][..]));
    let mut device = lease.device(0, Layout::new().span::<f64>(1));
    device.host().expect("host-backed")[0] = 4.0;
    assert_eq!(fb_core_device::DeviceSlice::len(&device), 1);
}

#[test]
#[should_panic(expected = "which meet the tenant already carved")]
fn two_layouts_cannot_carve_one_tenant_twice() {
    let ledger = ledger(ROOMY, ROOMY);
    ledger.budget("a").scratch(0, Kind::Pinned, 64);
    ledger.finish(&["a"], &[vec![]]).unwrap();
    let lease = ledger.lease("a");
    let _first = lease.pinned(0, Layout::new().span::<f64>(4));
    let _again = lease.pinned(0, Layout::new().span::<f64>(4));
}

#[test]
#[should_panic(expected = "already holds a lease")]
fn a_step_holds_one_lease_at_a_time() {
    let ledger = ledger(ROOMY, ROOMY);
    ledger.budget("a").scratch(0, Kind::Pinned, 64);
    ledger.finish(&["a"], &[vec![]]).unwrap();
    let first = ledger.lease("a");
    drop(ledger.lease("b"));
    drop(first);
    let _again = ledger.lease("a");
    let _twice = ledger.lease("a");
}

#[test]
#[should_panic(expected = "a run carves what its mem_budget declared")]
fn a_run_that_carves_past_its_declaration_is_stopped() {
    let ledger = ledger(ROOMY, ROOMY);
    ledger.budget("a").scratch(0, Kind::Pinned, 256);
    ledger.finish(&["a"], &[vec![]]).unwrap();
    let lease = ledger.lease("a");
    let _ = lease.pinned(0, Layout::new().span::<f64>(33));
}

#[test]
fn the_test_fill_is_nan_and_without_it_contents_stay() {
    let ledger = ledger(ROOMY, ROOMY);
    ledger.budget("a").scratch(0, Kind::Pinned, 64);
    ledger.finish(&["a"], &[vec![]]).unwrap();
    {
        let lease = ledger.lease("a");
        lease.pinned(0, Layout::new().span::<f64>(8)).fill(5.0);
    }
    {
        // No fill: what the run before left, the contents undefined, not
        // cleared.
        let lease = ledger.lease("a");
        assert!(
            lease
                .pinned(0, Layout::new().span::<f64>(8))
                .iter()
                .all(|&x| x == 5.0)
        );
    }
    let mut lease = ledger.lease("a");
    lease.fill_host();
    assert!(
        lease
            .pinned(0, Layout::new().span::<f64>(8))
            .iter()
            .all(|x| x.is_nan())
    );
}

#[test]
#[should_panic(expected = "budgets after the budget DAG's final node")]
fn nothing_is_budgeted_after_the_final_node() {
    let ledger = ledger(ROOMY, ROOMY);
    ledger.finish(&[], &[]).unwrap();
    ledger.budget("late").scratch(0, Kind::Device, 8);
}

#[test]
fn a_dry_budget_counts_and_allocates_nothing() {
    let ledger = Ledger::new(
        Caps {
            device: vec![Cap {
                written: None,
                used: ROOMY,
            }],
            host: cap(ROOMY),
        },
        Backing::Dry,
        vec![0],
        false,
    );
    let (device, _) = budget_step(&ledger, 0, MIB);
    ledger.finish(&["a"], &[vec![]]).unwrap();
    assert_eq!(ledger.allocations(), 0);
    assert!(!device.is_allocated());
    assert_eq!(fb_core_device::DeviceSlice::len(&device), 0);
    assert_eq!(
        ledger.device_total(0),
        MIB + SCRATCH[0] + fb_core_device::HANDLE_WORKSPACE_BYTES as u64,
        "counted as if allocated"
    );
    assert!(ledger.table().contains("default"), "{}", ledger.table());
}

/// The message and the line of the panic `f` raises on this thread.
fn panic_at(f: impl FnOnce()) -> (String, u32) {
    use std::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    thread_local!(static LINE: Cell<u32> = const { Cell::new(0) });
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            LINE.set(info.location().map_or(0, std::panic::Location::line));
            previous(info);
        }));
    });
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("the call panics");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default();
    (message, LINE.get())
}

/// One `Output` of 256 pinned bytes and 256 device bytes, host-backed.
fn tenanted(ledger: &Ledger) -> (PinnedBuf<u8>, DeviceBuf<u8>) {
    let budget = ledger.budget("a");
    (budget.pinned(0, 256), budget.device(0, 256))
}

#[test]
fn two_live_tenants_over_one_byte_panic_at_the_callers_line() {
    let ledger = ledger(ROOMY, ROOMY);
    let (mut pinned, mut device) = tenanted(&ledger);
    let (mut first, mut second) = (Layout::new(), Layout::new());
    let (a, b) = (first.span::<f64>(3), second.span::<i32>(2));
    let _a = pinned.tenant(&a);
    let line = line!() + 1;
    let (message, at) = panic_at(|| drop(pinned.tenant(&b)));
    assert_eq!(at, line, "the panic is at the caller's line: {message}");
    assert!(
        message.contains(
            "a tenant of a pinned buffer carves bytes 0..8, which meet the tenant carved at 0..24"
        ),
        "{message}"
    );
    let _a = device.tenant(&a);
    let line = line!() + 1;
    let (message, at) = panic_at(|| drop(device.tenant(&b)));
    assert_eq!(at, line, "the panic is at the caller's line: {message}");
    assert!(
        message.contains("a tenant of a device buffer carves bytes 0..8"),
        "{message}"
    );
}

#[test]
fn a_tenant_dropped_is_carved_again_and_apart_ones_live_together() {
    let ledger = ledger(ROOMY, ROOMY);
    let (mut pinned, mut device) = tenanted(&ledger);
    let mut layout = Layout::new();
    let (f, i) = (layout.span::<f64>(3), layout.span::<i32>(5));
    for turn in 0..2 {
        let mut a = pinned.tenant(&f);
        let mut b = pinned.tenant(&i);
        a.fill(f64::from(turn));
        b.fill(-turn);
        assert_eq!(
            (&a[..], &b[..]),
            (&[f64::from(turn); 3][..], &[-turn; 5][..])
        );
        let (x, y) = (device.tenant(&f), device.tenant(&i));
        assert_eq!(
            (
                fb_core_device::DeviceSlice::len(&x),
                fb_core_device::DeviceSlice::len(&y)
            ),
            (3, 5)
        );
    }
    // Every tenant dropped: the whole is reached again, holding the bytes
    // the tenants wrote.
    assert_eq!(pinned[0..8], 1.0f64.to_ne_bytes());
    assert!(device.host().is_some());
}

#[test]
fn a_leaked_tenant_keeps_its_record_and_the_whole_is_not_reached() {
    let ledger = ledger(ROOMY, ROOMY);
    let (mut pinned, _) = tenanted(&ledger);
    let mut layout = Layout::new();
    let span = layout.span::<f64>(4);
    // What a copy's token dropped unsynced does to its host end.
    std::mem::forget(pinned.tenant(&span));
    let (message, _) = panic_at(|| drop(pinned.tenant(&span)));
    assert!(
        message.contains(
            "a tenant of a pinned buffer carves bytes 0..32, which meet the tenant carved at 0..32"
        ),
        "{message}"
    );
    let (message, _) = panic_at(|| {
        let _ = pinned.len();
    });
    assert!(
        message.contains("a pinned buffer is reached whole while its bytes 0..32 are a tenant's"),
        "{message}"
    );
}

#[test]
#[should_panic(expected = "a device buffer is reached whole while its bytes 0..8 are a tenant's")]
fn a_device_buffer_with_a_live_tenant_is_not_reached_whole() {
    let ledger = ledger(ROOMY, ROOMY);
    let (_, mut device) = tenanted(&ledger);
    let _tenant = device.tenant(&Layout::new().span::<f64>(1));
    let _ = fb_core_device::DeviceSlice::as_ptr(&device);
}
