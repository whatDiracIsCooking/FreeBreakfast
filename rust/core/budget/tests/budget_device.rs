//! The budget on a card (tier 4): caps read from device 0, an `Output`'s
//! device and pinned buffers, a step's scratch of both kinds and the
//! handle workspace allocated by the final node, and a round trip through
//! them on a stream.

use fb_core_budget::{Backing, Caps, Kind, Layout, Ledger};
use fb_core_device::{DeviceSlice, HANDLE_WORKSPACE_BYTES, Stream};

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
    assert!(out.is_allocated() && device.is_allocated());
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
