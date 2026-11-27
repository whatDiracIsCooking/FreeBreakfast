//! The budget on a card (tier 4): caps read from device 0, an `Output`'s
//! device and pinned buffers, a step's scratch of both kinds and the
//! handle workspace allocated by the final node, and a round trip through
//! them on a stream; and tenants of an `Output`'s own buffers in a ring of
//! two, and one leaked with its copy's token.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use fb_core_budget::{Backing, Caps, DeviceTenant, Kind, Layout, Ledger, PinnedTenant, Span};
use fb_core_device::{DeviceSlice, HANDLE_WORKSPACE_BYTES, Pending, Stream};

#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_real_plan_allocates_and_its_buffers_round_trip() {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    assert!(caps.device[0].used > 0 && caps.device[0].written.is_none());
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);

    let n = 1000;
    let mut layout = Layout::new();
    let staged = layout.span::<f64>(n);
    let budget = ledger.budget("step");
    budget.scratch(0, Kind::Pinned, layout.bytes());
    budget.scratch(0, Kind::Device, n * size_of::<f64>());
    let out = budget.pinned::<f64>(0, n);
    let device = budget.device::<f64>(0, n);
    assert!(out.iter().all(|x| x.is_nan()), "pinned memory is filled");
    ledger.finish(&["step"], &[vec![]]).unwrap();
    let workspace = ledger
        .take_workspace(0)
        .expect("an accepted real plan has one");
    assert_eq!(workspace.len(), HANDLE_WORKSPACE_BYTES);
    assert!(ledger.take_workspace(0).is_none(), "taken once");
    // Two buffers, two slabs and the workspace.
    assert_eq!(ledger.allocations(), 5);

    let lease = ledger.lease("step");
    let mut host = lease.pinned(0, staged);
    for (i, x) in host.iter_mut().enumerate() {
        *x = i as f64 * 0.5 - 3.0;
    }
    let mut scratch = lease.device(0, Layout::new().span::<f64>(n));
    let stream = Stream::new(0);
    let host = stream.upload(&mut scratch, host).unwrap();
    let out = stream.download(out, &scratch).unwrap();
    let (host, out) = (host.sync().unwrap(), out.sync().unwrap());
    assert_eq!(&out[..], &host[..]);
    drop(device);
}

/// A real ledger on device 0.
fn real() -> Ledger {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    Ledger::new(caps, Backing::Real, vec![0], false)
}

/// The panic message of `f`.
fn refusal(f: impl FnOnce()) -> String {
    let payload =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("the call panics");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_default()
}

/// A ring of two buffers over one `Output`'s pinned staging and one device
/// buffer, as a manager's: each turn carves its slot's pinned and device
/// tenants, writes the host end, uploads it on the slot's own stream and
/// downloads it again into a second pinned tenant, and the slot's turn
/// before is handed back by its tokens' sync. Four turns, every slot used
/// twice; every byte comes back as it went.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_ring_of_two_round_trips_bytes_through_tenants() {
    const N: usize = 1000;
    let ledger = real();
    let budget = ledger.budget("ring");
    let (mut pin, mut dev) = (Layout::new(), Layout::new());
    let up: Vec<Span<f64>> = (0..2).map(|_| pin.span(N)).collect();
    let down: Vec<Span<f64>> = (0..2).map(|_| pin.span(N)).collect();
    let on_device: Vec<Span<f64>> = (0..2).map(|_| dev.span(N)).collect();
    let mut staging = budget.pinned::<u8>(0, pin.bytes());
    let mut buffer = budget.device::<u8>(0, dev.bytes());
    let streams = [Stream::new(0), Stream::new(0)];
    type Slot<'s> = (
        Pending<'s, PinnedTenant<f64>>,
        Pending<'s, PinnedTenant<f64>, false>,
        DeviceTenant<f64>,
    );
    let mut slots: [Option<Slot<'_>>; 2] = [None, None];
    for turn in 0..4 {
        let s = turn % 2;
        // The slot's turn before: its tokens' sync hands its tenants back,
        // and dropping them clears their record.
        if let Some((upload, download, _device)) = slots[s].take() {
            let (_up, back) = (upload.sync().unwrap(), download.sync().unwrap());
            let want: Vec<f64> = (0..N).map(|i| (turn - 2) as f64 * 1e3 + i as f64).collect();
            assert!(
                back.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "turn {} did not round-trip",
                turn - 2
            );
        }
        let mut host = staging.tenant(&up[s]);
        for (i, x) in host.iter_mut().enumerate() {
            *x = turn as f64 * 1e3 + i as f64;
        }
        let mut device = buffer.tenant(&on_device[s]);
        let upload = streams[s].upload(&mut device, host).unwrap();
        let download = streams[s]
            .download(staging.tenant(&down[s]), &device)
            .unwrap();
        slots[s] = Some((upload, download, device));
    }
    for (s, slot) in slots.into_iter().enumerate() {
        let (upload, download, _device) = slot.unwrap();
        drop(upload.sync().unwrap());
        let back = download.sync().unwrap();
        let turn = 2 + s;
        assert!(
            back.iter()
                .enumerate()
                .all(|(i, x)| x.to_bits() == (turn as f64 * 1e3 + i as f64).to_bits()),
            "turn {turn} did not round-trip"
        );
    }
    // Every tenant handed back: the whole is reached again.
    assert_eq!(staging.len(), pin.bytes());
}

/// A pinned tenant moved into an upload whose token is dropped unsynced is
/// leaked with it: its bytes stay carved, and a carve that meets them
/// panics, so the host cannot write bytes the copy may still read.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_tenant_leaked_with_its_token_keeps_its_record() {
    let ledger = real();
    let budget = ledger.budget("leak");
    let mut layout = Layout::new();
    let span = layout.span::<f64>(64);
    let mut staging = budget.pinned::<u8>(0, layout.bytes());
    let mut device = budget.device::<f64>(0, 64);
    let stream = Stream::new(0);
    drop(stream.upload(&mut device, staging.tenant(&span)).unwrap());
    stream.sync().unwrap();
    let message = refusal(|| drop(staging.tenant(&span)));
    assert!(
        message.contains("a tenant of a pinned buffer carves bytes 0..512, which meet the tenant carved at 0..512"),
        "{message}"
    );
}
