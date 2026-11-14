//! The forked kinetic and nuclear-attraction wrappers, the core
//! Hamiltonian's two, through `fb-integrals-sys` alone (tier 4 and their
//! refusals).
//!
//! One hand-written (s|s) record through each, against the closed-form
//! primitive integral, with `mu = alpha beta / zeta` and `R = |A - B|`:
//!
//! `T = s mu (3 - 2 mu R^2) exp(-mu R^2)`,
//! `V = -Z_C s (2 / sqrt(pi)) sqrt(zeta) F0(zeta |P - C|^2) exp(-mu R^2)`,
//! `F0(x) = (1/2) sqrt(pi / x) erf(sqrt(x))`,
//!
//! where `s = c_a c_b (pi / zeta)^(3/2)` is the record's. The record puts B
//! on the x axis from A at distances whose P - A, P - B and
//! `zeta X_PA X_PB` are exact in binary, as `overlap.rs`'s does, and keeps
//! `2 mu R^2 < 1`, so `3 - 2 mu R^2` does not cancel. The kinetic bar,
//! `1e-14` relative, is about twenty flops' rounding with a decade of margin;
//! the nuclear attraction's, `1e-12`, covers the kernel's Boys evaluation and
//! the reference's `erf` (libm's), with two decades.

use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_integrals_sys::{
    FB_KINETIC_MAX_MOMENTUM, FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, FB_LAUNCH_UNSUPPORTED,
    FB_POTENTIAL_EN_MAX_MOMENTUM, fb_kinetic_class, fb_nuclei_view, fb_potential_en_class,
    fb_primitive_pair_class_view,
};
use std::f64::consts::PI;
use std::ptr::{null, null_mut};

// libm's erf, which std links on every target this tree builds for: f64's
// own is not stable yet.
unsafe extern "C" {
    safe fn erf(x: f64) -> f64;
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

/// The (s|s) record both tests launch: alpha 1 and beta 3, A at the origin
/// and B at (0.5, 0, 0), so on x P = P - A = 0.375 and P - B = -0.125, and
/// `zeta X_PA X_PB = -mu R^2 = -0.1875`, every one exact; `2 mu R^2 = 0.375`.
struct Record {
    alpha: f64,
    beta: f64,
    zeta: f64,
    mu_r2: f64,
    p_x: f64,
    pa_x: f64,
    pb_x: f64,
    s: f64,
}

fn record() -> Record {
    let (alpha, beta) = (1.0_f64, 3.0_f64);
    let (c_a, c_b) = (0.7_f64, 1.3_f64);
    let r_ab = 0.5_f64;
    let zeta = alpha + beta;
    let pa_x = beta / zeta * r_ab;
    let pb_x = pa_x - r_ab;
    let mu_r2 = alpha * beta / zeta * r_ab * r_ab;
    assert_eq!(zeta * pa_x * pb_x, -mu_r2, "the record is not exact");
    assert!(2.0 * mu_r2 < 1.0, "3 - 2 mu R^2 cancels");
    Record {
        alpha,
        beta,
        zeta,
        mu_r2,
        p_x: pa_x,
        pa_x,
        pb_x,
        s: c_a * c_b * (PI / zeta).powf(1.5),
    }
}

/// The 2 x 2 matrix a launch wrote: the record is bra AO 0 and ket AO 1, so
/// both mirror positions are within `bar` of `expected`, relative, and the
/// diagonal stays 0.
fn check_mirrors(matrix: &DeviceBytes, stream: &Stream, expected: f64, bar: f64) {
    let got: Vec<f64> = matrix
        .read(stream)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect();
    assert_eq!(got[0].to_bits(), 0, "M_00 was written");
    assert_eq!(got[3].to_bits(), 0, "M_11 was written");
    for (at, value) in [("M_01", got[1]), ("M_10", got[2])] {
        let relative = ((value - expected) / expected).abs();
        assert!(
            relative <= bar,
            "{at} = {value:e}, the closed form {expected:e}: {relative:e} relative, the bar {bar:e}"
        );
    }
}

/// Needs a GPU. One (s|s) record through `fb_kinetic_class`: both mirror
/// positions the closed form within 1e-14 relative.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_record_is_the_closed_form_kinetic_energy() {
    let r = record();
    let mu = r.alpha * r.beta / r.zeta;
    let expected = r.s * mu * (3.0 - 2.0 * r.mu_r2) * (-r.mu_r2).exp();

    let stream = Stream::new(0);
    let d_zeta = DeviceBytes::upload(&[r.zeta], &stream);
    let d_pa = [
        DeviceBytes::upload(&[r.pa_x], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
    ];
    let d_pb = [
        DeviceBytes::upload(&[r.pb_x], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
    ];
    let d_s = DeviceBytes::upload(&[r.s], &stream);
    let d_eb = DeviceBytes::upload(&[r.beta], &stream);
    let d_off_a = DeviceBytes::upload(&[0_i32], &stream);
    let d_off_b = DeviceBytes::upload(&[1_i32], &stream);
    let n_ao = 2;
    let d_matrix = DeviceBytes::upload(&[0.0_f64; 4], &stream);
    let view = fb_primitive_pair_class_view {
        zeta: d_zeta.0.cast(),
        pa_x: d_pa[0].0.cast(),
        pa_y: d_pa[1].0.cast(),
        pa_z: d_pa[2].0.cast(),
        pb_x: d_pb[0].0.cast(),
        pb_y: d_pb[1].0.cast(),
        pb_z: d_pb[2].0.cast(),
        s: d_s.0.cast(),
        eb: d_eb.0.cast(),
        off_a: d_off_a.0.cast(),
        off_b: d_off_b.0.cast(),
        ..empty_view(0, 0, 1)
    };
    // SAFETY: every array the kinetic energy reads is one record long on the
    // device, the matrix is n_ao x n_ao and zeroed, and none overlap.
    let status = unsafe { fb_kinetic_class(stream.stream(), &view, n_ao, d_matrix.0.cast()) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_mirrors(&d_matrix, &stream, expected, 1e-14);
}

/// Needs a GPU. One (s|s) record and one nucleus, Z = 8 at (0.375, 0.5, 0),
/// so `P - C = (0, -0.5, 0)` exactly and `zeta |P - C|^2 = 1`, through
/// `fb_potential_en_class`: both mirror positions the closed form within
/// 1e-12 relative.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_record_and_nucleus_is_the_closed_form_attraction() {
    let r = record();
    let (z_c, c) = (8.0_f64, [0.375_f64, 0.5, 0.0]);
    let pc2 = (r.p_x - c[0]).powi(2) + c[1].powi(2) + c[2].powi(2);
    let x = r.zeta * pc2;
    let f0 = 0.5 * (PI / x).sqrt() * erf(x.sqrt());
    let expected = -z_c * r.s * (2.0 / PI.sqrt()) * r.zeta.sqrt() * f0 * (-r.mu_r2).exp();

    let stream = Stream::new(0);
    let d_zeta = DeviceBytes::upload(&[r.zeta], &stream);
    let d_p = [
        DeviceBytes::upload(&[r.p_x], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
    ];
    let d_pa = [
        DeviceBytes::upload(&[r.pa_x], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
    ];
    let d_pb = [
        DeviceBytes::upload(&[r.pb_x], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
        DeviceBytes::upload(&[0.0_f64], &stream),
    ];
    let d_s = DeviceBytes::upload(&[r.s], &stream);
    let d_off_a = DeviceBytes::upload(&[0_i32], &stream);
    let d_off_b = DeviceBytes::upload(&[1_i32], &stream);
    let d_z = DeviceBytes::upload(&[z_c], &stream);
    let d_c = DeviceBytes::upload(&c, &stream);
    let n_ao = 2;
    let d_matrix = DeviceBytes::upload(&[0.0_f64; 4], &stream);
    let view = fb_primitive_pair_class_view {
        zeta: d_zeta.0.cast(),
        p_x: d_p[0].0.cast(),
        p_y: d_p[1].0.cast(),
        p_z: d_p[2].0.cast(),
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
    let nuclei = fb_nuclei_view {
        n_nuc: 1,
        z: d_z.0.cast(),
        c: d_c.0.cast(),
    };
    // SAFETY: every array the attraction reads is one record (or one
    // nucleus) long on the device, the matrix is n_ao x n_ao and zeroed, and
    // none overlap.
    let status =
        unsafe { fb_potential_en_class(stream.stream(), &view, &nuclei, n_ao, d_matrix.0.cast()) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_mirrors(&d_matrix, &stream, expected, 1e-12);
}

/// The classes outside a table: a bra above the ket, one past the limit on
/// either side, and a negative momentum.
fn outside(max: i32) -> [(i32, i32); 5] {
    [
        (1, 0),
        (max, max - 1),
        (0, max + 1),
        (max + 1, max + 1),
        (-1, 0),
    ]
}

/// Every class outside the kinetic table is FB_LAUNCH_UNSUPPORTED, and a
/// view with no records, a null view or matrix, or no AOs is
/// FB_LAUNCH_INVALID_ARGUMENT, each before any launch: no device is touched.
#[test]
fn kinetic_refuses_a_class_with_no_kernel() {
    let dangling = std::ptr::NonNull::<f64>::dangling().as_ptr();
    // SAFETY: each call is refused before it reads a device pointer.
    unsafe {
        for (la, lb) in outside(FB_KINETIC_MAX_MOMENTUM) {
            assert_eq!(
                fb_kinetic_class(null_mut(), &empty_view(la, lb, 1), 4, dangling),
                FB_LAUNCH_UNSUPPORTED,
                "({la}, {lb})"
            );
        }
        assert_eq!(
            fb_kinetic_class(null_mut(), &empty_view(0, 0, 0), 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_kinetic_class(null_mut(), &empty_view(0, 0, 1), 0, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_kinetic_class(null_mut(), null(), 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_kinetic_class(null_mut(), &empty_view(0, 0, 1), 4, null_mut()),
            FB_LAUNCH_INVALID_ARGUMENT
        );
    }
}

/// Every class outside the nuclear attraction's table is
/// FB_LAUNCH_UNSUPPORTED, and a view with no records, no nuclei, no AOs, or
/// a null view, nuclei or matrix is FB_LAUNCH_INVALID_ARGUMENT, each before
/// any launch: no device is touched.
#[test]
fn potential_en_refuses_a_class_with_no_kernel() {
    let dangling = std::ptr::NonNull::<f64>::dangling().as_ptr();
    let nuclei = fb_nuclei_view {
        n_nuc: 1,
        z: dangling,
        c: dangling,
    };
    let none = fb_nuclei_view { n_nuc: 0, ..nuclei };
    // SAFETY: each call is refused before it reads a device pointer.
    unsafe {
        for (la, lb) in outside(FB_POTENTIAL_EN_MAX_MOMENTUM) {
            assert_eq!(
                fb_potential_en_class(null_mut(), &empty_view(la, lb, 1), &nuclei, 4, dangling),
                FB_LAUNCH_UNSUPPORTED,
                "({la}, {lb})"
            );
        }
        let view = empty_view(0, 0, 1);
        assert_eq!(
            fb_potential_en_class(null_mut(), &empty_view(0, 0, 0), &nuclei, 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_potential_en_class(null_mut(), &view, &none, 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_potential_en_class(null_mut(), &view, &nuclei, 0, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_potential_en_class(null_mut(), null(), &nuclei, 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_potential_en_class(null_mut(), &view, null(), 4, dangling),
            FB_LAUNCH_INVALID_ARGUMENT
        );
        assert_eq!(
            fb_potential_en_class(null_mut(), &view, &nuclei, 4, null_mut()),
            FB_LAUNCH_INVALID_ARGUMENT
        );
    }
}
