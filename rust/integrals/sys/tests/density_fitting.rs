//! The forked density-fitting wrappers, the metric's and the three-center
//! integrals', through `fb-integrals-sys` alone (tier 4 and their refusals).
//!
//! One hand-written s class through each, against its closed form. With
//! `F0(x) = (1/2) sqrt(pi / x) erf(sqrt(x))`, the aux records' weights
//! `s = c (pi / alpha)^(3/2)` and the bra record's
//! `s = c_a c_b (pi / zeta)^(3/2)`:
//!
//! `(P|Q) = c_P c_Q 2 pi^(5/2) / (alpha beta sqrt(alpha + beta)) F0(alpha beta / (alpha + beta) R^2)`,
//! `(ab|C) = c_a c_b c_C 2 pi^(5/2) / (zeta gamma sqrt(zeta + gamma)) exp(-mu R_AB^2) F0(zeta gamma / (zeta + gamma) |P - C|^2)`.
//!
//! The bar, `1e-12` relative, covers the kernel's Boys evaluation and the
//! reference's `erf` (libm's), each near an ulp, and the tens of flops on
//! each side, with two decades of margin, as `core_hamiltonian.rs`'s nuclear
//! attraction does. The symmetrization is exact: a sum and an exact halving,
//! so it is compared bit for bit.

use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_integrals_sys::{
    FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, FB_LAUNCH_UNSUPPORTED, FB_METRIC_MAX_MOMENTUM,
    FB_METRIC_PAIR_FIELDS, FB_METRIC_RECORD_FIELDS, FB_THREE_CENTER_AUX_FIELDS,
    FB_THREE_CENTER_AUX_MAX_MOMENTUM, FB_THREE_CENTER_BRA_FIELDS, FB_THREE_CENTER_CELL_FIELDS,
    FB_THREE_CENTER_MAX_MOMENTUM, fb_metric_class, fb_three_center_class,
    fb_three_center_symmetrize,
};
use std::f64::consts::PI;
use std::ptr::{NonNull, null, null_mut};

// libm's erf, which std links on every target this tree builds for: f64's
// own is not stable yet.
unsafe extern "C" {
    safe fn erf(x: f64) -> f64;
}

/// The Boys function of order 0, through libm's `erf`.
fn boys_f0(x: f64) -> f64 {
    0.5 * (PI / x).sqrt() * erf(x.sqrt())
}

/// A device array of doubles read back.
fn read_f64(d: &DeviceBytes, stream: &Stream) -> Vec<f64> {
    d.read(stream)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect()
}

/// Both mirror positions of a 2 x 2 block, flat indices 1 and 2, within
/// `bar` of `expected` relative; the diagonal, 0 and 3, never written.
fn check_mirrors(got: &[f64], expected: f64, bar: f64) {
    assert_eq!(got[0].to_bits(), 0, "element 0 was written");
    assert_eq!(got[3].to_bits(), 0, "element 3 was written");
    for at in [1, 2] {
        let rel = ((got[at] - expected) / expected).abs();
        assert!(
            rel <= bar,
            "element {at} = {:e}, the closed form {expected:e}: {rel:e} relative",
            got[at]
        );
    }
}

/// Needs a GPU. One (s|s) aux pair, P at the origin and Q at (1, 0.5, 0),
/// AOs 0 and 1 of a 2 x 2 metric, through `fb_metric_class`: both mirror
/// positions the closed form within 1e-12 relative, the diagonal untouched.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_aux_pair_is_the_closed_form_metric() {
    let (alpha, beta) = (0.5_f64, 1.5_f64);
    let (c_p, c_q) = (0.9_f64, 1.2_f64);
    let q = [1.0_f64, 0.5, 0.0];
    let r2 = q.iter().map(|x| x * x).sum::<f64>();
    let rho = alpha * beta / (alpha + beta);
    let expected =
        c_p * c_q * 2.0 * PI.powf(2.5) / (alpha * beta * (alpha + beta).sqrt()) * boys_f0(rho * r2);

    // Two records, P's and Q's, field by field.
    let s = |c: f64, a: f64| c * (PI / a).powf(1.5);
    let records = [
        0.0,
        q[0],
        0.0,
        q[1],
        0.0,
        q[2],
        alpha,
        beta,
        s(c_p, alpha),
        s(c_q, beta),
    ];
    assert_eq!(records.len(), 2 * FB_METRIC_RECORD_FIELDS as usize);
    // One pair: P's records [0, 1), Q's [1, 2), at AOs 0 and 1.
    let pairs = [0_i32, 1, 1, 2, 0, 1];
    assert_eq!(pairs.len(), FB_METRIC_PAIR_FIELDS as usize);

    let stream = Stream::new(0);
    let d_records = DeviceBytes::upload(&records, &stream);
    let d_pairs = DeviceBytes::upload(&pairs, &stream);
    let n_aux = 2;
    let d_metric = DeviceBytes::upload(&[0.0_f64; 4], &stream);
    // SAFETY: the records are FB_METRIC_RECORD_FIELDS segments of two
    // doubles, the pair FB_METRIC_PAIR_FIELDS segments of one int32 naming
    // records 0 and 1 and AOs within n_aux, the metric n_aux x n_aux; none
    // overlap.
    let status = unsafe {
        fb_metric_class(
            stream.stream(),
            0,
            0,
            2,
            d_records.0.cast(),
            1,
            d_pairs.0.cast(),
            n_aux,
            d_metric.0.cast(),
        )
    };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_mirrors(&read_f64(&d_metric, &stream), expected, 1e-12);
}

/// Needs a GPU. One (ss|s) cell, A at the origin, B at (1.5, 0, 0) and the
/// aux C at (1.125, 0.5, 0.25), AOs 0 and 1 against aux function 0 of a
/// 2 x 2 x 1 tensor, through `fb_three_center_class`: both (m, n) mirror
/// positions the closed form within 1e-12 relative, the diagonal untouched.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_ss_s_cell_is_the_closed_form_three_center() {
    let (alpha, beta, gamma) = (1.0_f64, 3.0_f64, 0.8_f64);
    let (c_a, c_b, c_c) = (0.7_f64, 1.3_f64, 1.1_f64);
    let r_ab = 1.5_f64;
    let zeta = alpha + beta;
    let p_x = beta / zeta * r_ab;
    let (pa_x, pb_x) = (p_x, p_x - r_ab);
    let mu_r2 = alpha * beta / zeta * r_ab * r_ab;
    let c = [1.125_f64, 0.5, 0.25];
    let pc2 = (p_x - c[0]).powi(2) + c[1].powi(2) + c[2].powi(2);
    let theta2 = zeta * gamma / (zeta + gamma);
    let expected = c_a * c_b * c_c * 2.0 * PI.powf(2.5) / (zeta * gamma * (zeta + gamma).sqrt())
        * (-mu_r2).exp()
        * boys_f0(theta2 * pc2);

    // One bra record: zeta, P, P - A, P - B, s.
    let bra = [
        zeta,
        p_x,
        0.0,
        0.0,
        pa_x,
        0.0,
        0.0,
        pb_x,
        0.0,
        0.0,
        c_a * c_b * (PI / zeta).powf(1.5),
    ];
    assert_eq!(bra.len(), FB_THREE_CENTER_BRA_FIELDS as usize);
    let aux = [c[0], c[1], c[2], gamma, c_c * (PI / gamma).powf(1.5)];
    assert_eq!(aux.len(), FB_THREE_CENTER_AUX_FIELDS as usize);
    // One cell: bra records [0, 1), aux records [0, 1), at AOs 0 and 1 and
    // aux function 0.
    let cells = [0_i32, 1, 0, 1, 0, 1, 0];
    assert_eq!(cells.len(), FB_THREE_CENTER_CELL_FIELDS as usize);

    let stream = Stream::new(0);
    let d_bra = DeviceBytes::upload(&bra, &stream);
    let d_aux = DeviceBytes::upload(&aux, &stream);
    let d_cells = DeviceBytes::upload(&cells, &stream);
    let (n_ao, n_aux) = (2, 1);
    let d_eri3 = DeviceBytes::upload(&[0.0_f64; 4], &stream);
    // SAFETY: each array is its field count of one-element segments, the
    // cell names record 0 of each and offsets within n_ao and n_aux, the
    // tensor n_ao x n_ao x n_aux; none overlap.
    let status = unsafe {
        fb_three_center_class(
            stream.stream(),
            0,
            0,
            0,
            1,
            d_bra.0.cast(),
            1,
            d_aux.0.cast(),
            1,
            d_cells.0.cast(),
            n_ao,
            n_aux,
            d_eri3.0.cast(),
        )
    };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    check_mirrors(&read_f64(&d_eri3, &stream), expected, 1e-12);
}

/// Needs a GPU. A 3 x 3 x 2 tensor of distinct values through
/// `fb_three_center_symmetrize`: every element half of itself plus half of
/// its (m, n) mirror, bit for bit.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn symmetrize_averages_each_element_with_its_mirror() {
    let (n_ao, n_aux) = (3_usize, 2_usize);
    let input: Vec<f64> = (0..n_ao * n_ao * n_aux)
        .map(|i| 1.0 + 0.375 * i as f64)
        .collect();
    let at = |m: usize, n: usize, p: usize| (m * n_ao + n) * n_aux + p;
    let stream = Stream::new(0);
    let d_in = DeviceBytes::upload(&input, &stream);
    let d_out = DeviceBytes::upload(&vec![0.0_f64; input.len()], &stream);
    // SAFETY: both arrays are n_ao * n_ao * n_aux doubles on the device, and
    // they do not overlap.
    let status = unsafe {
        fb_three_center_symmetrize(
            stream.stream(),
            n_ao as i32,
            n_aux as i32,
            d_in.0.cast(),
            d_out.0.cast(),
        )
    };
    assert_eq!(status, FB_LAUNCH_SUCCESS);
    let got = read_f64(&d_out, &stream);
    for m in 0..n_ao {
        for n in 0..n_ao {
            for p in 0..n_aux {
                let want = 0.5 * (input[at(m, n, p)] + input[at(n, m, p)]);
                assert_eq!(
                    got[at(m, n, p)].to_bits(),
                    want.to_bits(),
                    "({m}, {n}, {p})"
                );
            }
        }
    }
}

/// Every class outside the metric's table is FB_LAUNCH_UNSUPPORTED, and a
/// null array, a count not positive or an n_aux whose square overflows an
/// int32 is FB_LAUNCH_INVALID_ARGUMENT, each before any launch: no device is
/// touched.
#[test]
fn metric_refuses_before_launching() {
    let d = NonNull::<f64>::dangling().as_ptr();
    let i = NonNull::<i32>::dangling().as_ptr();
    let max = FB_METRIC_MAX_MOMENTUM;
    // SAFETY: every call below is refused before it reads a device pointer.
    let metric = |lp, lq, n_rec, records, n_pair, pairs, n_aux, out| unsafe {
        fb_metric_class(
            null_mut(),
            lp,
            lq,
            n_rec,
            records,
            n_pair,
            pairs,
            n_aux,
            out,
        )
    };
    for (lp, lq) in [
        (1, 0),
        (0, max + 1),
        (max + 1, max + 1),
        (-1, 0),
        (max, max - 1),
    ] {
        assert_eq!(
            metric(lp, lq, 1, d, 1, i, 1, d),
            FB_LAUNCH_UNSUPPORTED,
            "({lp}, {lq})"
        );
    }
    for (n_rec, n_pair, n_aux) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 46_341)] {
        assert_eq!(
            metric(0, 0, n_rec, d, n_pair, i, n_aux, d),
            FB_LAUNCH_INVALID_ARGUMENT,
            "({n_rec}, {n_pair}, {n_aux})"
        );
    }
    assert_eq!(
        metric(0, 0, 1, null(), 1, i, 1, d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        metric(0, 0, 1, d, 1, null(), 1, d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        metric(0, 0, 1, d, 1, i, 1, null_mut()),
        FB_LAUNCH_INVALID_ARGUMENT
    );
}

/// Every class outside the three-center table is FB_LAUNCH_UNSUPPORTED, and
/// a null array, a count not positive or a tensor past an int32's index is
/// FB_LAUNCH_INVALID_ARGUMENT; the symmetrization refuses the same counts
/// and arrays, and an input and output that overlap. No device is touched.
#[test]
fn three_center_wrappers_refuse_before_launching() {
    let d = NonNull::<f64>::dangling().as_ptr();
    let i = NonNull::<i32>::dangling().as_ptr();
    let (max, aux_max) = (
        FB_THREE_CENTER_MAX_MOMENTUM,
        FB_THREE_CENTER_AUX_MAX_MOMENTUM,
    );
    // SAFETY: every call below is refused before it reads a device pointer.
    let eri3 = |l: [i32; 3], counts: [i32; 5], bra, aux, cells, out| unsafe {
        let [n_bra, n_aux_rec, n_cell, n_ao, n_aux] = counts;
        fb_three_center_class(
            null_mut(),
            l[0],
            l[1],
            l[2],
            n_bra,
            bra,
            n_aux_rec,
            aux,
            n_cell,
            cells,
            n_ao,
            n_aux,
            out,
        )
    };
    let ones = [1; 5];
    for l in [
        [1, 0, 0],
        [0, max + 1, 0],
        [max + 1, max + 1, 0],
        [0, 0, aux_max + 1],
        [0, 0, -1],
        [-1, 0, 0],
    ] {
        assert_eq!(eri3(l, ones, d, d, i, d), FB_LAUNCH_UNSUPPORTED, "{l:?}");
    }
    for slot in 0..5 {
        let mut counts = ones;
        counts[slot] = 0;
        assert_eq!(
            eri3([0; 3], counts, d, d, i, d),
            FB_LAUNCH_INVALID_ARGUMENT,
            "{counts:?}"
        );
    }
    // 1024 * 1024 * 2048 = 2^31, one past an int32.
    assert_eq!(
        eri3([0; 3], [1, 1, 1, 1024, 2048], d, d, i, d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        eri3([0; 3], ones, null(), d, i, d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        eri3([0; 3], ones, d, null(), i, d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        eri3([0; 3], ones, d, d, null(), d),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    assert_eq!(
        eri3([0; 3], ones, d, d, i, null_mut()),
        FB_LAUNCH_INVALID_ARGUMENT
    );

    // SAFETY: as above.
    let symmetrize = |n_ao, n_aux, input, output| unsafe {
        fb_three_center_symmetrize(null_mut(), n_ao, n_aux, input, output)
    };
    // Two 2 x 2 x 1 tensors in one host array, never read: the refusals look
    // at the addresses alone.
    let mut buf = [0.0_f64; 8];
    let base = buf.as_mut_ptr();
    for (n_ao, n_aux) in [(0, 1), (1, 0), (1024, 2048)] {
        assert_eq!(
            symmetrize(n_ao, n_aux, base.cast_const(), base.wrapping_add(4)),
            FB_LAUNCH_INVALID_ARGUMENT,
            "({n_ao}, {n_aux})"
        );
    }
    assert_eq!(symmetrize(2, 1, null(), base), FB_LAUNCH_INVALID_ARGUMENT);
    assert_eq!(
        symmetrize(2, 1, base.cast_const(), null_mut()),
        FB_LAUNCH_INVALID_ARGUMENT
    );
    for shift in [0, 1, 3] {
        assert_eq!(
            symmetrize(2, 1, base.cast_const(), base.wrapping_add(shift)),
            FB_LAUNCH_INVALID_ARGUMENT,
            "output {shift} past input"
        );
        assert_eq!(
            symmetrize(2, 1, base.wrapping_add(shift).cast_const(), base),
            FB_LAUNCH_INVALID_ARGUMENT,
            "input {shift} past output"
        );
    }
}
