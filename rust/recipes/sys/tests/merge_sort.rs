//! The forked merge sort on a card, against the host's stable sort (tier 4).
//!
//! Sizes one below, at and one above every launch-shape boundary the
//! wrapper has, for either warp width (the test does not know which it was
//! built for): a warp (32 or 64), the merge-pass block (256), the one-tile
//! block sort's block of each warp count (352, 704, 1408 keys at a 32-wide
//! warp), a tile (2816), and two and four tiles; then one key past 8 and 16
//! tiles and 100 000 keys, so the merge passes alternate the halves from one
//! to six times, an odd and an even number each. Keys repeat, so
//! stability shows in the permutation, which must equal the host's exactly:
//! it is integers. Each sort gets exactly the workspace its query asked for.

use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_sys::{
    FB_DEVICE_ALIGNMENT, FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, fb_merge_sort,
    fb_merge_sort_workspace_bytes,
};

const SIZES: &[u32] = &[
    1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 351, 352, 353, 703, 704, 705, 1407, 1408, 1409,
    2815, 2816, 2817, 5631, 5632, 5633, 11263, 11264, 11265, 22529, 45057, 100_000,
];

/// `n` keys drawn from 13 values, the two zeros and both infinities among
/// them, so most keys repeat.
fn keys(n: u32) -> Vec<f64> {
    const POOL: [f64; 13] = [
        -0.0,
        0.0,
        1.5,
        -1.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2.0,
        -7.25,
        1e300,
        -1e-300,
        3.0,
        0.5,
        -2.0,
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ u64::from(n);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            POOL[(state >> 33) as usize % POOL.len()]
        })
        .collect()
}

fn workspace(n: u32, with_values: bool) -> (Option<DeviceBytes>, usize) {
    // SAFETY: a host function of its arguments alone.
    let bytes = unsafe { fb_merge_sort_workspace_bytes(n, with_values) };
    let buffer = (bytes > 0).then(|| DeviceBytes::new(bytes));
    if let Some(b) = &buffer {
        assert_eq!(b.0 as usize % FB_DEVICE_ALIGNMENT, 0);
    }
    (buffer, bytes)
}

/// Needs a GPU. Keys and their permutation, for every size.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn sorts_keys_and_values_as_the_host_stable_sort() {
    let stream = Stream::new(0);
    for &n in SIZES {
        let host = keys(n);
        let mut order: Vec<u32> = (0..n).collect();
        order.sort_by(|&a, &b| host[a as usize].total_cmp(&host[b as usize]));
        let expected: Vec<u64> = order.iter().map(|&i| host[i as usize].to_bits()).collect();

        let d_keys = DeviceBytes::upload(&host, &stream);
        let d_values = DeviceBytes::upload(&(0..n).collect::<Vec<u32>>(), &stream);
        let (scratch, bytes) = workspace(n, true);
        // SAFETY: device arrays of n keys and n values, and exactly the
        // workspace the query asked for, none overlapping.
        let status = unsafe {
            fb_merge_sort(
                stream.stream(),
                n,
                d_keys.0.cast(),
                d_values.0.cast(),
                scratch.as_ref().map_or(std::ptr::null_mut(), |s| s.0),
                bytes,
            )
        };
        assert_eq!(status, FB_LAUNCH_SUCCESS, "n = {n}");
        let got_keys: Vec<u64> = d_keys
            .read(&stream)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_ne_bytes(*c))
            .collect();
        let got_order: Vec<u32> = d_values
            .read(&stream)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c))
            .collect();
        assert!(got_order == order, "n = {n}: the permutation differs");
        assert!(got_keys == expected, "n = {n}: the keys differ");
    }
}

/// Needs a GPU. Without values, the keys alone, for every size.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn sorts_keys_alone_as_the_host_sort() {
    let stream = Stream::new(0);
    for &n in SIZES {
        let host = keys(n);
        let mut expected = host.clone();
        expected.sort_by(f64::total_cmp);
        let expected: Vec<u64> = expected.iter().map(|k| k.to_bits()).collect();

        let d_keys = DeviceBytes::upload(&host, &stream);
        let (scratch, bytes) = workspace(n, false);
        // SAFETY: as above, with no values.
        let status = unsafe {
            fb_merge_sort(
                stream.stream(),
                n,
                d_keys.0.cast(),
                std::ptr::null_mut(),
                scratch.as_ref().map_or(std::ptr::null_mut(), |s| s.0),
                bytes,
            )
        };
        assert_eq!(status, FB_LAUNCH_SUCCESS, "n = {n}");
        let got: Vec<u64> = d_keys
            .read(&stream)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| u64::from_ne_bytes(*c))
            .collect();
        assert!(got == expected, "n = {n}: the keys differ");
    }
}

/// The workspace query: nothing up to a tile, then the keys' half padded
/// to the alignment and the values' half after it. Host only.
#[test]
fn workspace_query_is_the_two_halves() {
    // SAFETY: a host function of its arguments alone.
    let bytes = |n, with_values| unsafe { fb_merge_sort_workspace_bytes(n, with_values) };
    assert_eq!(bytes(0, true), 0);
    assert_eq!(bytes(2816, true), 0);
    // 2817 keys are 22536 bytes, 22784 at 256; the values 11268 more.
    assert_eq!(bytes(2817, false), 22784);
    assert_eq!(bytes(2817, true), 22784 + 11268);
}

/// A null key array, or a workspace one byte short, is refused before any
/// launch: no device is touched.
#[test]
fn refuses_null_keys_and_a_short_workspace() {
    let null = std::ptr::null_mut();
    // SAFETY: every call below is refused, or enqueues nothing, before it
    // reads a pointer.
    let sort =
        |n: u32,
         keys: *mut f64,
         values: *mut u32,
         workspace: *mut core::ffi::c_void,
         bytes: usize| unsafe { fb_merge_sort(null, n, keys, values, workspace, bytes) };
    assert_eq!(
        sort(5, null.cast(), null.cast(), null, 0),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    // SAFETY: a host function of its arguments alone.
    let bytes = unsafe { fb_merge_sort_workspace_bytes(5000, true) };
    let dangling = std::ptr::NonNull::<f64>::dangling().as_ptr();
    assert_eq!(
        sort(5000, dangling, dangling.cast(), dangling.cast(), bytes - 1),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    // n = 0 enqueues nothing and succeeds, whatever the pointers.
    assert_eq!(
        sort(0, null.cast(), null.cast(), null, 0),
        FB_LAUNCH_SUCCESS
    );
}

/// The library this binary loaded is the one the bindings describe: the
/// header's ABI version, built for the backend of the CMake build linked.
/// Host functions alone, so it needs no device.
#[test]
fn the_library_passes_the_load_check() {
    fb_recipes_sys::check_library().unwrap();
}
