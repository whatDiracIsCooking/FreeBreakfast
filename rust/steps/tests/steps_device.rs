//! The steps on a card (tier 4, device 0): water (with the metric over
//! cc-pVDZ-JKFIT) and benzene in cc-pVDZ set up, budgeted and run
//! once, X = S^(-1/2) brought back and held against the host's from the
//! same S, and the orthogonalizer's scratch apart from every step
//! unordered with it in the real plan's table.
//!
//! The reference is cyclic Jacobi on the host's S, then `V diag(w^(-1/2))
//! Vᵀ`. The bar is fixed by analysis, one run: X moves by about
//! `½ λ_min^(-3/2)` per unit of backward error in S, so its error relative
//! to its largest element, `λ_min^(-1/2)`, is the eigensolve's backward
//! error relative to `‖S‖₂` times `κ₂(S) / 2`. A Householder eigensolve's
//! backward error is a modest multiple of `ε ‖S‖₂` (the worst-case bound
//! carries a factor `n` that is not seen in practice), and `κ₂(S)` is
//! 2.5e2 for water and 2.4e4 for benzene in cc-pVDZ (printed), so the
//! error is about `ε κ₂ / 2 ≈ 3e-12` at benzene, and 1e-10 of the largest
//! element, the issue's bar, holds over a decade of margin, Jacobi's own
//! error, `ε κ₂`-sized too, inside it.

mod common;

use fb_core_budget::{Backing, Caps, Ledger};
use fb_core_device::Stream;
use fb_core_flow::{Bus, Executor};
use fb_steps::{EIGENVALUE_FLOOR, MainBus, OrthogonalizerStep, plan};

/// The bar: X's error relative to its largest element.
const BAR: f64 = 1e-10;

/// The eigenvalues (ascending) and eigenvectors (column-major) of the
/// symmetric `n × n` `a`, by cyclic Jacobi to round-off.
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i + n * i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|j| (0..j).map(move |i| (i, j)))
            .map(|(i, j)| a[i + n * j] * a[i + n * j])
            .sum();
        if off < 1e-32 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p + n * q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q + n * q] - a[p + n * p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k + n * p], a[k + n * q]);
                    a[k + n * p] = c * akp - s * akq;
                    a[k + n * q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p + n * k], a[q + n * k]);
                    a[p + n * k] = c * apk - s * aqk;
                    a[q + n * k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k + n * p], v[k + n * q]);
                    v[k + n * p] = c * vkp - s * vkq;
                    v[k + n * q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i + n * i].total_cmp(&a[j + n * j]));
    let values = order.iter().map(|&i| a[i + n * i]).collect();
    let vectors = order
        .iter()
        .flat_map(|&j| v[n * j..n * (j + 1)].to_vec())
        .collect();
    (values, vectors)
}

/// The host's S^(-1/2) of the packed `s`, and κ₂(S).
fn host_x(packed: &[f64], n: usize) -> (Vec<f64>, f64) {
    let mut s = vec![0.0; n * n];
    for p in 0..n {
        for q in 0..=p {
            let v = packed[p * (p + 1) / 2 + q];
            s[p + n * q] = v;
            s[q + n * p] = v;
        }
    }
    let (w, v) = jacobi(s, n);
    let scale: Vec<f64> = w
        .iter()
        .map(|&w| {
            if w <= EIGENVALUE_FLOOR {
                0.0
            } else {
                1.0 / w.sqrt()
            }
        })
        .collect();
    let mut x = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            x[i + n * j] = (0..n).map(|k| v[i + n * k] * scale[k] * v[j + n * k]).sum();
        }
    }
    (x, w[n - 1] / w[0])
}

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0` (`docs/testing.md`, "Slow cases").
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// `xyz` in cc-pVDZ, with `[eri]` over `aux` when given, set up on device
/// 0, budgeted and run once; X against the host's, and the orthogonalizer's
/// scratch apart from the `unordered` steps' in the plan's table.
fn x_against_the_host(xyz: &str, aux: Option<&str>, unordered: usize) {
    let executor = Executor::new(4);
    let config = common::config(xyz, "cc-pvdz", aux);
    let plan = plan(MainBus::setup(config, Backing::Real).unwrap()).unwrap();
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();
    assert_eq!(
        common::orthogonalizer_scratch_is_its_own(plan.bus().ledger()),
        unordered
    );

    let shared = plan.bus().shared();
    let n = shared.dims.n_ao;
    // A pinned buffer of a ledger of the test's own to bring X back.
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let host = ledger.budget("test").pinned::<f64>(0, n * n);
    let stream = Stream::new(0);
    let x = plan.output(OrthogonalizerStep);
    let got = stream.download(host, x.x()).unwrap().sync().unwrap();

    let (want, kappa) = host_x(&fb_integrals_s_overlap_host::overlap(&shared.basis), n);
    let largest = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let error = got
        .iter()
        .zip(&want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()))
        / largest;
    println!(
        "{xyz}: n {n}, κ₂(S) {kappa:.3e}, ε κ₂ / 2 = {:.1e}, max |X - X_host| / max |X_host| = {error:.2e} (bar {BAR:e})",
        f64::EPSILON * kappa / 2.0
    );
    assert!(
        !error.is_nan() && error <= BAR,
        "{xyz}: {error:e} > {BAR:e}"
    );
}

/// Water, with the metric: the family's representative. Most of its time
/// is the solver library loading its kernels at the first syevd.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_x_matches_the_hosts_inverse_square_root_of_s() {
    // Pinned: the guess, H and the metric are unordered with it.
    x_against_the_host("h2o.xyz", Some("cc-pvdz-jkfit"), 3);
}

/// Benzene, κ₂(S) a hundred times water's: a slow case, about 2 s on the
/// gfx1200, most of it the solver library's load in a fresh plan.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzenes_x_matches_the_hosts_inverse_square_root_of_s() {
    if fast_tests() {
        return;
    }
    x_against_the_host("benzene.xyz", None, 2);
}
