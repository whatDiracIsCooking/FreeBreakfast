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

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use fb_core_budget::{Backing, Budget, Caps, DeviceBuf, Ledger, PinnedBuf, Plain};
use fb_core_device::{DeviceSlice, DeviceSliceMut, Stream};
use fb_integrals_sys::{
    FB_KINETIC_MAX_MOMENTUM, FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, FB_LAUNCH_UNSUPPORTED,
    FB_POTENTIAL_EN_MAX_MOMENTUM, fb_kinetic_class, fb_nuclei_view, fb_potential_en_class,
    fb_primitive_pair_class_view,
};
use std::f64::consts::PI;
use std::ptr::{null, null_mut};

/// A real ledger on device 0, for the buffers.
fn ledger() -> Ledger {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    Ledger::new(caps, Backing::Real, vec![0], false)
}

/// `data` in pinned memory, the host end of a copy.
fn pinned<T: Plain>(budget: &Budget<'_>, data: &[T]) -> PinnedBuf<T> {
    let mut p = budget.pinned::<T>(0, data.len());
    p.copy_from_slice(data);
    p
}

/// `data` uploaded into a device buffer.
fn device<T: Plain>(budget: &Budget<'_>, stream: &Stream, data: &[T]) -> DeviceBuf<T> {
    let mut d = budget.device::<T>(0, data.len());
    drop(
        stream
            .upload(&mut d, pinned(budget, data))
            .unwrap()
            .sync()
            .unwrap(),
    );
    d
}

/// A device buffer read back to the host.
fn read<T: Plain>(budget: &Budget<'_>, stream: &Stream, d: &DeviceBuf<T>) -> Vec<T> {
    let host = budget.pinned::<T>(0, d.len());
    stream.download(host, d).unwrap().sync().unwrap().to_vec()
}

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

/// The 2 x 2 row-major matrix a launch wrote: the record is bra AO 0 and
/// ket AO 1, and the kernels add into the lower triangle alone, so `M_10`
/// is within `bar` of `expected`, relative, and `M_01` and the diagonal
/// stay 0 (the caller mirrors the lower triangle onto the upper).
fn check_lower(got: &[f64], expected: f64, bar: f64) {
    assert_eq!(got[0].to_bits(), 0, "M_00 was written");
    assert_eq!(got[3].to_bits(), 0, "M_11 was written");
    assert_eq!(got[1].to_bits(), 0, "M_01, above the diagonal, was written");
    let relative = ((got[2] - expected) / expected).abs();
    assert!(
        relative <= bar,
        "M_10 = {:e}, the closed form {expected:e}: {relative:e} relative, the bar {bar:e}",
        got[2]
    );
}

/// Needs a GPU. One (s|s) record through `fb_kinetic_class`: the lower
/// position the closed form within 1e-14 relative, the upper untouched.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_record_is_the_closed_form_kinetic_energy() {
    let r = record();
    let mu = r.alpha * r.beta / r.zeta;
    let expected = r.s * mu * (3.0 - 2.0 * r.mu_r2) * (-r.mu_r2).exp();

    let ledger = ledger();
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let d_zeta = device(&budget, &stream, &[r.zeta]);
    let d_pa = [
        device(&budget, &stream, &[r.pa_x]),
        device(&budget, &stream, &[0.0_f64]),
        device(&budget, &stream, &[0.0_f64]),
    ];
    let d_pb = [
        device(&budget, &stream, &[r.pb_x]),
        device(&budget, &stream, &[0.0_f64]),
        device(&budget, &stream, &[0.0_f64]),
    ];
    let d_s = device(&budget, &stream, &[r.s]);
    let d_eb = device(&budget, &stream, &[r.beta]);
    let d_off_a = device(&budget, &stream, &[0_i32]);
    let d_off_b = device(&budget, &stream, &[1_i32]);
    let n_ao = 2;
    let mut d_matrix = device(&budget, &stream, &[0.0_f64; 4]);
    let view = fb_primitive_pair_class_view {
        zeta: d_zeta.as_ptr().cast(),
        pa_x: d_pa[0].as_ptr().cast(),
        pa_y: d_pa[1].as_ptr().cast(),
        pa_z: d_pa[2].as_ptr().cast(),
        pb_x: d_pb[0].as_ptr().cast(),
        pb_y: d_pb[1].as_ptr().cast(),
        pb_z: d_pb[2].as_ptr().cast(),
        s: d_s.as_ptr().cast(),
        eb: d_eb.as_ptr().cast(),
        off_a: d_off_a.as_ptr().cast(),
        off_b: d_off_b.as_ptr().cast(),
        ..empty_view(0, 0, 1)
    };
    // SAFETY: every array the kinetic energy reads is one record long on the
    // device, the matrix is n_ao x n_ao and zeroed, and none overlap.
    let status = unsafe { fb_kinetic_class(stream.raw(), &view, n_ao, d_matrix.as_mut_ptr()) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_lower(&read(&budget, &stream, &d_matrix), expected, 1e-14);
}

/// Needs a GPU. One (s|s) record and one nucleus, Z = 8 at (0.375, 0.5, 0),
/// so `P - C = (0, -0.5, 0)` exactly and `zeta |P - C|^2 = 1`, through
/// `fb_potential_en_class`: the lower position the closed form within
/// 1e-12 relative, the upper untouched.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_record_and_nucleus_is_the_closed_form_attraction() {
    let r = record();
    let (z_c, c) = (8.0_f64, [0.375_f64, 0.5, 0.0]);
    let pc2 = (r.p_x - c[0]).powi(2) + c[1].powi(2) + c[2].powi(2);
    let x = r.zeta * pc2;
    let f0 = 0.5 * (PI / x).sqrt() * erf(x.sqrt());
    let expected = -z_c * r.s * (2.0 / PI.sqrt()) * r.zeta.sqrt() * f0 * (-r.mu_r2).exp();

    let ledger = ledger();
    let budget = ledger.budget("test");
    let stream = Stream::new(0);
    let d_zeta = device(&budget, &stream, &[r.zeta]);
    let d_p = [
        device(&budget, &stream, &[r.p_x]),
        device(&budget, &stream, &[0.0_f64]),
        device(&budget, &stream, &[0.0_f64]),
    ];
    let d_pa = [
        device(&budget, &stream, &[r.pa_x]),
        device(&budget, &stream, &[0.0_f64]),
        device(&budget, &stream, &[0.0_f64]),
    ];
    let d_pb = [
        device(&budget, &stream, &[r.pb_x]),
        device(&budget, &stream, &[0.0_f64]),
        device(&budget, &stream, &[0.0_f64]),
    ];
    let d_s = device(&budget, &stream, &[r.s]);
    let d_off_a = device(&budget, &stream, &[0_i32]);
    let d_off_b = device(&budget, &stream, &[1_i32]);
    let d_z = device(&budget, &stream, &[z_c]);
    let d_c = device(&budget, &stream, &c);
    let n_ao = 2;
    let mut d_matrix = device(&budget, &stream, &[0.0_f64; 4]);
    let view = fb_primitive_pair_class_view {
        zeta: d_zeta.as_ptr().cast(),
        p_x: d_p[0].as_ptr().cast(),
        p_y: d_p[1].as_ptr().cast(),
        p_z: d_p[2].as_ptr().cast(),
        pa_x: d_pa[0].as_ptr().cast(),
        pa_y: d_pa[1].as_ptr().cast(),
        pa_z: d_pa[2].as_ptr().cast(),
        pb_x: d_pb[0].as_ptr().cast(),
        pb_y: d_pb[1].as_ptr().cast(),
        pb_z: d_pb[2].as_ptr().cast(),
        s: d_s.as_ptr().cast(),
        off_a: d_off_a.as_ptr().cast(),
        off_b: d_off_b.as_ptr().cast(),
        ..empty_view(0, 0, 1)
    };
    let nuclei = fb_nuclei_view {
        n_nuc: 1,
        z: d_z.as_ptr().cast(),
        c: d_c.as_ptr().cast(),
    };
    // SAFETY: every array the attraction reads is one record (or one
    // nucleus) long on the device, the matrix is n_ao x n_ao and zeroed, and
    // none overlap.
    let status =
        unsafe { fb_potential_en_class(stream.raw(), &view, &nuclei, n_ao, d_matrix.as_mut_ptr()) };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_lower(&read(&budget, &stream, &d_matrix), expected, 1e-12);
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
    // SAFETY: every call below is refused before it reads a device pointer.
    let kinetic = |cls: *const fb_primitive_pair_class_view, n_ao: i32, t: *mut f64| unsafe {
        fb_kinetic_class(null_mut(), cls, n_ao, t)
    };
    for (la, lb) in outside(FB_KINETIC_MAX_MOMENTUM) {
        assert_eq!(
            kinetic(&empty_view(la, lb, 1), 4, dangling),
            FB_LAUNCH_UNSUPPORTED,
            "({la}, {lb})"
        );
    }
    assert_eq!(
        kinetic(&empty_view(0, 0, 0), 4, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        kinetic(&empty_view(0, 0, 1), 0, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(kinetic(null(), 4, dangling), FB_LAUNCH_INVALID_ARGUMENT);
    assert_eq!(
        kinetic(&empty_view(0, 0, 1), 4, null_mut()),
        FB_LAUNCH_INVALID_ARGUMENT
    );
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
    // SAFETY: every call below is refused before it reads a device pointer.
    let potential =
        |cls: *const fb_primitive_pair_class_view,
         nuclei: *const fb_nuclei_view,
         n_ao: i32,
         v: *mut f64| unsafe { fb_potential_en_class(null_mut(), cls, nuclei, n_ao, v) };
    for (la, lb) in outside(FB_POTENTIAL_EN_MAX_MOMENTUM) {
        assert_eq!(
            potential(&empty_view(la, lb, 1), &nuclei, 4, dangling),
            FB_LAUNCH_UNSUPPORTED,
            "({la}, {lb})"
        );
    }
    let view = empty_view(0, 0, 1);
    assert_eq!(
        potential(&empty_view(0, 0, 0), &nuclei, 4, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        potential(&view, &none, 4, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        potential(&view, &nuclei, 0, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        potential(null(), &nuclei, 4, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        potential(&view, null(), 4, dangling),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        potential(&view, &nuclei, 4, null_mut()),
        FB_LAUNCH_INVALID_ARGUMENT
    );
}
