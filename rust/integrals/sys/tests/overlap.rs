//! The forked overlap and mmd wrappers through `fb-integrals-sys` alone
//! (tier 4 and its refusals).
//!
//! One hand-written (s|s) record through `fb_overlap_class` against the
//! closed-form primitive overlap
//!
//! `S = c_a c_b (pi / zeta)^(3/2) exp(-alpha beta / zeta |A - B|^2)`.
//!
//! The record puts B on the x axis from A, at distances whose P - A, P - B
//! and `zeta X_PA X_PB` are exact in binary, so the kernel's exponent is the
//! closed form's to the bit and its y and z factors are exp(0) = 1 exactly.
//! What is left is one `exp` on each side, the vendor's and the host's, each
//! within an ulp of the true value, and the one product by `s` on each side,
//! half an ulp each: three ulp at worst, and the bar is four.

use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_integrals_sys::{
    FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, FB_LAUNCH_UNSUPPORTED, FB_OVERLAP_MAX_MOMENTUM,
    FB_PAIR_BOUND_MAX_MOMENTUM, FB_RECORD_BOUND_MAX_MOMENTUM, fb_mmd_bound, fb_mmd_order,
    fb_mmd_order_workspace_bytes, fb_mmd_records, fb_ordered_pair_bound, fb_overlap_class,
    fb_primitive_pair_class_view, fb_primitive_pair_records, fb_primitive_pair_source,
};
use std::ptr::{null, null_mut};
use wwr_sys::{wwr_memcpy_kind, wwr_status};

/// Copies `host` to a new device buffer on `stream` and waits for it: on the
/// stream, not with the synchronous copy, which runs on the legacy stream a
/// non-blocking `Stream` does not wait for (README.md, "Build and test").
fn upload<T: Copy>(host: &[T], stream: &Stream) -> DeviceBytes {
    let bytes = std::mem::size_of_val(host);
    let device = DeviceBytes::new(bytes);
    // SAFETY: both ranges are `bytes` long, and `host` outlives the copy:
    // the stream is drained before it returns.
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
    device
}

/// A view of no arrays: every pointer null, for the fields a test sets.
fn empty_view(la: i32, lb: i32, n_rec: i32) -> fb_primitive_pair_class_view {
    fb_primitive_pair_class_view {
        la,
        lb,
        n_rec,
        zeta: null(),
        p_x: null(),
        p_y: null(),
        p_z: null(),
        pa_x: null(),
        pa_y: null(),
        pa_z: null(),
        pb_x: null(),
        pb_y: null(),
        pb_z: null(),
        s: null(),
        ea: null(),
        eb: null(),
        bound: null(),
        bound_ab: null(),
        bound_ba: null(),
        off_a: null(),
        off_b: null(),
        rank_a: null(),
        rank_b: null(),
    }
}

/// How many representable doubles lie between two positive finite values.
fn ulps(a: f64, b: f64) -> u64 {
    assert!(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite());
    a.to_bits().abs_diff(b.to_bits())
}

/// Needs a GPU. One (s|s) record, bra AO 0 and ket AO 1 of a 2 x 2 matrix:
/// both mirror positions get the closed form within 4 ulp, the diagonal
/// stays 0.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_record_is_the_closed_form_overlap() {
    let (alpha, beta) = (1.0_f64, 3.0_f64);
    let (c_a, c_b) = (0.7_f64, 1.3_f64);
    // A at the origin, B at (1.5, 0, 0): P - A = 1.125 and P - B = -0.375 on
    // x, zeta X_PA X_PB = -1.6875 = -mu |A - B|^2, every one exact.
    let r_ab = 1.5_f64;
    let zeta = alpha + beta;
    let pa_x = beta / zeta * r_ab;
    let pb_x = pa_x - r_ab;
    let mu_r2 = alpha * beta / zeta * r_ab * r_ab;
    assert_eq!(zeta * pa_x * pb_x, -mu_r2, "the record is not exact");
    let s = c_a * c_b * (std::f64::consts::PI / zeta).powf(1.5);
    let expected = c_a * c_b * (std::f64::consts::PI / zeta).powf(1.5) * (-mu_r2).exp();

    let stream = Stream::new(0);
    let d_zeta = upload(&[zeta], &stream);
    let d_pa = [
        upload(&[pa_x], &stream),
        upload(&[0.0_f64], &stream),
        upload(&[0.0_f64], &stream),
    ];
    let d_pb = [
        upload(&[pb_x], &stream),
        upload(&[0.0_f64], &stream),
        upload(&[0.0_f64], &stream),
    ];
    let d_s = upload(&[s], &stream);
    let d_off_a = upload(&[0_i32], &stream);
    let d_off_b = upload(&[1_i32], &stream);
    let n_ao = 2;
    let d_matrix = upload(&[0.0_f64; 4], &stream);
    let view = fb_primitive_pair_class_view {
        zeta: d_zeta.0.cast(),
        pa_x: d_pa[0].0.cast(),
        pa_y: d_pa[1].0.cast(),
        pa_z: d_pa[2].0.cast(),
        pb_x: d_pb[0].0.cast(),
        pb_y: d_pb[1].0.cast(),
        pb_z: d_pb[2].0.cast(),
        s: d_s.0.cast(),
        off_a: d_off_a.0.cast(),
        off_b: d_off_b.0.cast(),
        ..empty_view(0, 0, 1)
    };
    // SAFETY: every array the overlap reads is one record long on the
    // device, the matrix is n_ao x n_ao and zeroed, and none overlap.
    let status = unsafe { fb_overlap_class(stream.stream(), &view, n_ao, d_matrix.0.cast()) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    let got: Vec<f64> = d_matrix
        .read(stream.stream())
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect();
    assert_eq!(got[0].to_bits(), 0, "S_00 was written");
    assert_eq!(got[3].to_bits(), 0, "S_11 was written");
    for (at, value) in [("S_01", got[1]), ("S_10", got[2])] {
        let off = ulps(value, expected);
        assert!(
            off <= 4,
            "{at} = {value:e}, the closed form {expected:e}: {off} ulp apart"
        );
    }
}

/// Every class outside the overlap's table is FB_LAUNCH_UNSUPPORTED, and a
/// view with no records, a null view or matrix, or no AOs is
/// FB_LAUNCH_INVALID_ARGUMENT, each before any launch: no device is touched.
#[test]
fn overlap_refuses_a_class_with_no_kernel() {
    let dangling = std::ptr::NonNull::<f64>::dangling().as_ptr();
    let max = FB_OVERLAP_MAX_MOMENTUM;
    // SAFETY: each call is refused before it reads a device pointer.
    unsafe {
        for (la, lb) in [
            (1, 0),
            (max, max - 1),
            (0, max + 1),
            (max + 1, max + 1),
            (-1, 0),
        ] {
            let view = empty_view(la, lb, 1);
            assert_eq!(
                fb_overlap_class(null_mut(), &view, 4, dangling),
                FB_LAUNCH_UNSUPPORTED,
                "({la}, {lb})"
            );
        }
        assert_eq!(
            fb_overlap_class(null_mut(), &empty_view(0, 0, 0), 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_overlap_class(null_mut(), &empty_view(0, 0, 1), 0, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_overlap_class(null_mut(), null(), 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_overlap_class(null_mut(), &empty_view(0, 0, 1), 4, null_mut()),
            FB_LAUNCH_INVALID_ARGUMENT
        );
    }
}

/// The bounds' tables refuse as the overlap's does: the undressed bound
/// (la <= lb only) and the dressed one (every ordered pair) past their
/// limits are FB_LAUNCH_UNSUPPORTED, and the order and the records refuse a
/// count, a null pointer or a short workspace. No device is touched.
#[test]
fn mmd_wrappers_refuse_before_launching() {
    let dangling = std::ptr::NonNull::<f64>::dangling().as_ptr();
    let pb = FB_PAIR_BOUND_MAX_MOMENTUM;
    let rb = FB_RECORD_BOUND_MAX_MOMENTUM;
    // SAFETY: each call is refused before it reads a device pointer.
    unsafe {
        for (la, lb) in [(1, 0), (0, pb + 1), (-1, 0)] {
            assert_eq!(
                fb_mmd_bound(null_mut(), la, lb, 1, dangling, dangling),
                FB_LAUNCH_UNSUPPORTED,
                "({la}, {lb})"
            );
        }
        assert_eq!(
            fb_mmd_bound(null_mut(), 0, 0, 0, dangling, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        for (la, lb) in [(rb + 1, 0), (0, rb + 1), (-1, 0)] {
            assert_eq!(
                fb_ordered_pair_bound(null_mut(), &empty_view(la, lb, 1), dangling),
                FB_LAUNCH_UNSUPPORTED,
                "({la}, {lb})"
            );
        }
        assert_eq!(
            fb_ordered_pair_bound(null_mut(), &empty_view(0, 0, 0), dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );

        let src = fb_primitive_pair_source {
            n_pairs: 1,
            bra: null(),
            ket: null(),
            rec_begin: null(),
            prim_begin: null(),
            exponent: null(),
            coefficient: null(),
            c_x: null(),
            c_y: null(),
            c_z: null(),
            ao_offset: null(),
            rank: null(),
        };
        let perm = std::ptr::NonNull::<u32>::dangling().as_ptr();
        assert_eq!(
            fb_mmd_order(null_mut(), 0, &src, dangling, perm, null_mut(), 0),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_mmd_order(null_mut(), 5, null(), dangling, perm, null_mut(), 0),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        let bytes = fb_mmd_order_workspace_bytes(5000);
        assert_eq!(
            fb_mmd_order(
                null_mut(),
                5000,
                &src,
                dangling,
                perm,
                dangling.cast(),
                bytes - 1
            ),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        let no_pairs = fb_primitive_pair_source { n_pairs: 0, ..src };
        assert_eq!(
            fb_mmd_records(
                null_mut(),
                5,
                &no_pairs,
                perm,
                null::<fb_primitive_pair_records>()
            ),
            FB_LAUNCH_INVALID_ARGUMENT
        );
    }
}

/// The order's workspace is the merge sort's with values: nothing up to a
/// tile (2816 records) or for no records, then the keys' half padded to the
/// alignment and the values' half after it. Host only.
#[test]
fn mmd_order_workspace_is_the_sort_with_values() {
    // SAFETY: a host function of its argument alone.
    unsafe {
        assert_eq!(fb_mmd_order_workspace_bytes(-1), 0);
        assert_eq!(fb_mmd_order_workspace_bytes(0), 0);
        assert_eq!(fb_mmd_order_workspace_bytes(2816), 0);
        // 2817 keys are 22536 bytes, 22784 at 256; the values 11268 more.
        assert_eq!(fb_mmd_order_workspace_bytes(2817), 22784 + 11268);
    }
}
