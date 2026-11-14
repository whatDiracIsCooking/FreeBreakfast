//! The forked inverse square root on a card, against the host's (tier 4).
//!
//! The bar is bitwise, by analysis: the kernel's `1 / sqrt(x)` is a
//! correctly rounded `sqrt` and a correctly rounded division in IEEE-754
//! double on both vendors, as is the host's `1.0 / x.sqrt()`, so the two are
//! the same bits. The elements cover each side of the floor (below it, at
//! it, one ulp above it, a negative value, both zeros) and ordinary values
//! over the double range, repeated past three of the launch's blocks of
//! four warps (128 or 256 threads), so a block boundary falls inside them;
//! elements past `count` must be untouched.

use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_sys::{FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, fb_inverse_sqrt};
use wwr_sys::{wwr_memcpy_kind, wwr_status};

/// The orthogonalizer's default floor, the C++ `kOrthogonalizeEigenvalueFloor`.
const FLOOR: f64 = 1e-10;

/// Elements the kernel maps; then `TAIL` more it must not touch.
const COUNT: usize = 777;
const TAIL: usize = 5;

/// `COUNT + TAIL` elements, the pool repeated.
fn elements() -> Vec<f64> {
    let pool = [
        -1.0,
        -0.0,
        0.0,
        1e-300,
        FLOOR / 2.0,
        FLOOR,
        f64::from_bits(FLOOR.to_bits() + 1),
        1e-8,
        0.25,
        0.5,
        1.0,
        2.0,
        3.0,
        7.123_456_789,
        1e10,
        1e300,
        f64::MAX,
    ];
    (0..COUNT + TAIL).map(|i| pool[i % pool.len()]).collect()
}

/// Needs a GPU. Every element as the host maps it, bit for bit; the tail
/// as it was.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn maps_each_element_as_the_host_does() {
    let stream = Stream::new(0);
    let host = elements();
    let bytes = size_of_val(host.as_slice());
    let device = DeviceBytes::new(bytes);
    // SAFETY: both ranges are `bytes` long; the copy is drained before the
    // launch is queued, and `host` outlives it.
    let status = unsafe {
        let copy = wwr_sys::wwr_memcpy_async(
            device.0,
            host.as_ptr().cast(),
            bytes,
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            stream.stream(),
        );
        assert_eq!(copy, wwr_status::WWR_STATUS_SUCCESS);
        wwr_sys::wwr_stream_synchronize(stream.stream())
    };
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);

    // SAFETY: a device array of COUNT + TAIL doubles, on `stream`.
    let status = unsafe { fb_inverse_sqrt(stream.stream(), COUNT as u32, device.0.cast(), FLOOR) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    let got: Vec<u64> = device
        .read(stream.stream())
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_ne_bytes(*c))
        .collect();

    let mut misses = Vec::new();
    for (i, (&x, &bits)) in host.iter().zip(&got).enumerate() {
        let expected = if i >= COUNT {
            x
        } else if x <= FLOOR {
            0.0
        } else {
            1.0 / x.sqrt()
        };
        if bits != expected.to_bits() {
            misses.push(format!(
                "[{i}] {x:e}: {:e}, expected {expected:e}",
                f64::from_bits(bits)
            ));
        }
    }
    assert!(misses.is_empty(), "differ:\n{}", misses.join("\n"));
}

/// A null array is refused before any launch, and a count of 0 enqueues
/// nothing: no device is touched.
#[test]
fn refuses_a_null_array() {
    let null = std::ptr::null_mut();
    // SAFETY: each call returns before it reads a pointer.
    unsafe {
        assert_eq!(
            fb_inverse_sqrt(null, 5, null.cast(), FLOOR),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_inverse_sqrt(null, 0, null.cast(), FLOOR),
            FB_LAUNCH_SUCCESS
        );
    }
}
