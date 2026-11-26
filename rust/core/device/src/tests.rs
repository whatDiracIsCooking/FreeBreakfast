//! The crate's tests: the copies and the event on a card (tier 4), and the
//! cuts' contracts on the host.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use std::sync::atomic::Ordering;

use super::*;

/// Device memory from `wwr_malloc`, freed on drop: a test's buffer, since
/// allocating is the budget's and this crate has none.
pub(crate) struct DeviceBuf<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuf<T> {
    #[allow(
        clippy::disallowed_methods,
        reason = "a test's own buffer, outside any plan: allocating is the budget's, and the test checks what is under it"
    )]
    pub(crate) fn new(len: usize) -> DeviceBuf<T> {
        let _current = ScopedDevice::new(0);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer through a valid local.
        let status = unsafe { wwr_sys::wwr_malloc(&mut ptr, len.max(1) * size_of::<T>()) };
        runtime_status(status).unwrap();
        DeviceBuf {
            ptr: ptr.cast(),
            len,
        }
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        // SAFETY: allocated by `wwr_malloc` in `new`, freed once; every test
        // syncs its streams before its buffers drop.
        let status = unsafe { wwr_sys::wwr_free(self.ptr.cast()) };
        runtime_status(status).unwrap();
    }
}

// SAFETY: `len` elements of device memory, live until the drop, which each
// test reaches only after syncing the streams that used it.
unsafe impl<T> DeviceSlice<T> for DeviceBuf<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above; a test hands it to one call at a time.
unsafe impl<T> DeviceSliceMut<T> for DeviceBuf<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Pinned host memory from `wwr_host_alloc`, so a copy of it is truly
/// asynchronous, freed on drop.
pub(crate) struct Pinned<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> Pinned<T> {
    #[allow(
        clippy::disallowed_methods,
        reason = "a test's own buffer, outside any plan: allocating is the budget's, and the test checks what is under it"
    )]
    pub(crate) fn from(values: &[T]) -> Pinned<T> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let bytes = values.len().max(1) * size_of::<T>();
        // SAFETY: writes one pointer through a valid local.
        let status = unsafe { wwr_sys::wwr_host_alloc(&mut ptr, bytes, 0) };
        runtime_status(status).unwrap();
        let ptr = ptr.cast::<T>();
        // SAFETY: `ptr` holds `values.len()` elements, just allocated.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
        Pinned {
            ptr,
            len: values.len(),
        }
    }

    /// The last element, read once the copies into it have run.
    pub(crate) fn last(&self) -> T {
        // SAFETY: `len` initialized elements; the test synced first.
        let all = unsafe { std::slice::from_raw_parts(self.ptr, self.len) };
        all[self.len - 1]
    }

    /// The elements, read once the copies into them have run.
    pub(crate) fn read(&self) -> Vec<T> {
        // SAFETY: `len` initialized elements; the test synced first.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }.to_vec()
    }
}

impl<T> Drop for Pinned<T> {
    fn drop(&mut self) {
        // SAFETY: allocated by `wwr_host_alloc`, freed once, after the
        // test's syncs.
        let status = unsafe { wwr_sys::wwr_free_host(self.ptr.cast()) };
        runtime_status(status).unwrap();
    }
}

// SAFETY: `len` elements of host memory, live until the drop and reached
// only through the value (`read`); a leaked one is never freed.
unsafe impl<T> HostSlice<T> for Pinned<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above.
unsafe impl<T> HostSliceMut<T> for Pinned<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Doubles whose bits a conversion could change: NaNs with payloads, both
/// zeros, a subnormal, the infinities, and a ramp.
fn awkward(n: usize) -> Vec<f64> {
    let mut v = vec![
        f64::from_bits(0x7FF8_0000_DEAD_BEEF),
        f64::from_bits(0xFFF0_0000_0000_0001),
        -0.0,
        0.0,
        f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    v.extend((0..n).map(|i| (i as f64).sqrt() - 1e3));
    v.truncate(n);
    v
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Needs a GPU. Host to device to device to host is the identity on the
/// bits (exact: a copy). The upload reads the head of a longer host end;
/// the download writes the head of a longer one and leaves its tail; the
/// device copy reads a part; one sync is counted, and it covers every
/// token of the copies before it.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_round_trip_through_each_copy_is_bitwise() {
    let n = 4099;
    let values = awkward(n);
    let stream = Stream::new(0);
    let mut longer = values.clone();
    longer.extend([1.0, 2.0, 3.0]);
    let src = Pinned::from(&longer);
    let mut a = DeviceBuf::<f64>::new(n);
    let mut b = DeviceBuf::<f64>::new(n);
    let whole = Pinned::from(&vec![7.0; n + 3]);
    let back = Pinned::from(&vec![0.0; n]);

    let src = stream.upload(&mut a, src).unwrap();
    let whole = stream.download(whole, &a).unwrap();
    stream.copy_device(&mut b, &part(&a, 0, n)).unwrap();
    let back = stream.download(back, &b).unwrap();
    let (syncs, _) = sync_meter();
    stream.sync().unwrap();
    assert!(sync_meter().0 > syncs);
    let (src, whole, back) = (
        src.sync().unwrap(),
        whole.sync().unwrap(),
        back.sync().unwrap(),
    );
    // The stream's own count: the meter is the process's.
    assert_eq!(
        stream.tickets.load(Ordering::SeqCst),
        1,
        "one sync covers the three tokens"
    );

    let whole = whole.read();
    assert_eq!(bits(&whole[..n]), bits(&values));
    assert_eq!(whole[n..], [7.0; 3]);
    assert_eq!(bits(&back.read()), bits(&values));
    assert_eq!(
        bits(&src.read()),
        bits(&longer),
        "the host end comes back as it went"
    );

    // Nothing is enqueued for an empty pair, and its token syncs.
    let mut empty = DeviceBuf::<f64>::new(0);
    let empty = stream.upload(&mut empty, Pinned::<f64>::from(&[])).unwrap();
    assert!(empty.sync().unwrap().read().is_empty());
}

/// Needs a GPU. A token's sync waits for its copy: a 64 MiB download,
/// queued behind 200 device copies of the same size, synced through its
/// token alone with no sync of the stream, hands back every element
/// (exact), its last read first. Without the token's sync the host reads
/// the `-1.0` the buffer was made with.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_tokens_sync_waits_for_its_copy() {
    let n = 8 << 20;
    let stream = Stream::new(0);
    let mut a = DeviceBuf::<f64>::new(n);
    let mut b = DeviceBuf::<f64>::new(n);
    // Made first: a pinned allocation may wait for the device.
    let back = Pinned::from(&vec![-1.0; n]);
    stream.fill(&mut a, 0x3F).unwrap();
    for _ in 0..200 {
        stream.copy_device(&mut b, &a).unwrap();
    }
    let back = stream.download(back, &b).unwrap().sync().unwrap();
    let filled = 0x3F3F_3F3F_3F3F_3F3F;
    assert_eq!(back.last().to_bits(), filled, "the copy had not run");
    assert!(back.read().iter().all(|x| x.to_bits() == filled));
}

/// Needs a GPU. An event recorded after an upload on one stream orders a
/// second stream's read of it: the second copies and downloads what the
/// first uploaded, all of it (exact). The upload is 64 MiB from pinned
/// memory, so without the wait the second stream's copy would start while
/// it is in flight, over a buffer zeroed first.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn an_event_orders_a_second_streams_read() {
    let n = 8 << 20;
    let values: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
    let (first, second) = (Stream::new(0), Stream::new(0));
    let uploaded = Event::new(0);
    let src = Pinned::from(&values);
    let mut a = DeviceBuf::<f64>::new(n);
    let mut b = DeviceBuf::<f64>::new(n);
    let back = Pinned::from(&vec![-1.0; n]);
    first.zero(&mut a).unwrap();
    first.sync().unwrap();

    let src = first.upload(&mut a, src).unwrap();
    uploaded.record(&first).unwrap();
    second.wait(&uploaded).unwrap();
    second.copy_device(&mut b, &a).unwrap();
    let back = second.download(back, &b).unwrap().sync().unwrap();
    uploaded.sync().unwrap();
    drop(src.sync().unwrap());

    assert!(
        back.read() == values,
        "the second stream read before the upload ran"
    );
}

/// Needs a GPU. The runtime sees the card, and its free memory is under
/// its total.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_card_and_its_memory_are_read() {
    assert!(device_count().unwrap() >= 1);
    let (free, total) = memory(0).unwrap();
    assert!(0 < free && free <= total, "{free} of {total}");
}

/// Needs a GPU. An upload past its host end panics before anything is
/// enqueued.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "Pre-condition of upload violated: the host end holds the device end")]
fn an_upload_past_the_host_end_panics() {
    let stream = Stream::new(0);
    let mut a = DeviceBuf::<f64>::new(4);
    let _ = stream.upload(&mut a, Pinned::from(&[0.0; 3]));
}

/// Host memory standing in for device memory: only the pointers and
/// lengths are looked at, and nothing is dereferenced.
struct Fake(Vec<f64>);

// SAFETY: never handed to a call that dereferences it.
unsafe impl DeviceSlice<f64> for Fake {
    fn as_ptr(&self) -> *const f64 {
        self.0.as_ptr()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: as above.
unsafe impl DeviceSliceMut<f64> for Fake {
    fn as_mut_ptr(&mut self) -> *mut f64 {
        self.0.as_mut_ptr()
    }
}

/// The cuts land where they say: a part at its offset, the two halves of
/// a split tiling the slice, a split at either end leaving one empty.
#[test]
fn the_cuts_land_where_they_say() {
    let mut slice = Fake(vec![0.0; 10]);
    let base = slice.as_ptr() as usize;
    let at = |p: *const f64| (p as usize - base) / 8;
    let p = part(&slice, 3, 7);
    assert_eq!((at(p.as_ptr()), p.len()), (3, 7));
    let q = part_mut(&mut slice, 10, 0);
    assert_eq!((at(q.as_ptr()), q.len()), (10, 0));
    let (mut head, tail) = split_at_mut(&mut slice, 4);
    assert_eq!((at(tail.as_ptr()), tail.len()), (4, 6));
    let (x, y) = split_at_mut(&mut head, 1);
    assert_eq!(
        (at(x.as_ptr()), x.len(), at(y.as_ptr()), y.len()),
        (0, 1, 1, 3)
    );
    let (all, empty) = split_at_mut(&mut slice, 10);
    assert_eq!((all.len(), empty.len()), (10, 0));
}

/// Quiet the logger when run as [`each_cut_past_the_end_panics_at_its_caller`]'s
/// child, so the panic hook writes its line to stderr.
pub(crate) fn as_child() {
    if std::env::var_os("FB_CONTRACT_CHILD").is_some() {
        fb_core_log::set_level(fb_core_log::LogLevel::Off);
    }
}

/// A part past the end panics: it would claim memory the slice never
/// promised.
#[test]
#[should_panic(expected = "Pre-condition of part violated: the run is inside the slice")]
fn a_part_past_the_end_panics() {
    as_child();
    let slice = Fake(vec![0.0; 10]);
    let _ = part(&slice, 4, 7); // MARK part
}

/// A writable part past the end panics, as a part does.
#[test]
#[should_panic(expected = "Pre-condition of part_mut violated: the run is inside the slice")]
fn a_writable_part_past_the_end_panics() {
    as_child();
    let mut slice = Fake(vec![0.0; 10]);
    let _ = part_mut(&mut slice, 11, 0); // MARK part_mut
}

/// A split past the end panics, as a part does.
#[test]
#[should_panic(expected = "Pre-condition of split_at_mut violated: mid is within the slice")]
fn a_split_past_the_end_panics() {
    as_child();
    let mut slice = Fake(vec![0.0; 10]);
    let _ = split_at_mut(&mut slice, 11); // MARK split_at_mut
}

/// Each cut's panic is located at the caller's line, `#[track_caller]`
/// reaching through `#[requires]`, and carries its operands' values
/// through `fb_core_log`'s panic hook. The hook's line is not the panic's
/// message, which `#[should_panic]` matches, so each case above is re-run
/// as a child and its stderr read.
#[test]
fn each_cut_past_the_end_panics_at_its_caller() {
    let source = include_str!("tests.rs");
    for (case, call, values) in [
        (
            "a_part_past_the_end_panics",
            "part",
            "the run is inside the slice: le! (offset, of.len()) && le! (len, of.len() - offset) \
             [len = 7, of.len() - offset = 6]",
        ),
        (
            "a_writable_part_past_the_end_panics",
            "part_mut",
            "the run is inside the slice: le! (offset, of.len()) && le! (len, of.len() - offset) \
             [offset = 11, of.len() = 10]",
        ),
        (
            "a_split_past_the_end_panics",
            "split_at_mut",
            "mid is within the slice: le! (mid, of.len()) [mid = 11, of.len() = 10]",
        ),
    ] {
        let name = format!("tests::{case}");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env("FB_CONTRACT_CHILD", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{case}: {:?}\n{stderr}", out.status);
        let (line, text) = source
            .lines()
            .enumerate()
            .find(|(_, l)| l.contains(&format!("// MARK {call}")))
            .unwrap();
        let column = text.find(&format!("{call}(")).unwrap() + 1;
        let want = format!(
            "thread '{name}' panicked at {}:{}:{column}: Pre-condition of {call} violated: {values}",
            file!(),
            line + 1
        );
        assert!(
            stderr.lines().any(|l| l == want),
            "{stderr}\n  wanted {want}"
        );
    }
}
