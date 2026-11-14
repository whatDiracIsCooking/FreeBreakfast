//! The packed gemm on a card, against the product on the host: small
//! integers, so every partial sum is exact in any order and the comparison
//! is bitwise.

use std::rc::Rc;

use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::device_slice::DeviceSlice;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
use fb_recipes_blas::{BlasOp, C, N, T, gemm};
use wwr_sys::wwr_blas_operation;

/// `rows` × `cols`, column-major, packed.
fn matrix(rows: usize, cols: usize, seed: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

/// `alpha op_a(A) op_b(B) + beta C` on the host, every matrix packed.
#[allow(clippy::too_many_arguments)]
fn host_gemm(
    t_a: bool,
    t_b: bool,
    (m, n, k): (usize, usize, usize),
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &[f64],
) -> Vec<f64> {
    let mut out = c.to_vec();
    for j in 0..n {
        for i in 0..m {
            let dot: f64 = (0..k)
                .map(|l| {
                    let x = if t_a { a[l + i * k] } else { a[i + l * m] };
                    let y = if t_b { b[j + l * n] } else { b[l + j * k] };
                    x * y
                })
                .sum();
            out[i + j * m] = alpha * dot + beta * out[i + j * m];
        }
    }
    out
}

fn download(device: &DeviceBytes, owner: &Stream, n: usize) -> Vec<f64> {
    device
        .read(owner)
        .as_chunks::<8>()
        .0
        .iter()
        .take(n)
        .map(|c| f64::from_ne_bytes(*c))
        .collect()
}

/// One packed product on a card, checked bitwise against the host's.
fn check<OpA: BlasOp, OpB: BlasOp>((m, n, k): (usize, usize, usize)) {
    let t = |op| op != wwr_blas_operation::WWR_BLAS_OP_N;
    let (t_a, t_b) = (t(OpA::RAW), t(OpB::RAW));
    let (a, b, c) = (matrix(m, k, 1), matrix(k, n, 2), matrix(m, n, 3));
    let want = host_gemm(t_a, t_b, (m, n, k), -2.0, &a, &b, 3.0, &c);
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    let (da, db, mut dc) = (
        DeviceBytes::upload(&a, &owner),
        DeviceBytes::upload(&b, &owner),
        DeviceBytes::upload(&c, &owner),
    );
    let i = |x: usize| x as i32;
    gemm::<OpA, OpB, f64>(&blas, i(m), i(n), i(k), -2.0, &da, &db, 3.0, &mut dc).unwrap();
    assert_eq!(download(&dc, &owner, m * n), want);
}

/// Needs a GPU. Every pairing of N and T on a rectangular shape, so each
/// derived leading dimension is a different one of m, n and k.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn every_op_pairing_derives_its_leading_dimensions() {
    let shape = (5, 3, 4);
    check::<N, N>(shape);
    check::<T, N>(shape);
    check::<N, T>(shape);
    check::<T, T>(shape);
}

/// Needs a GPU. C on a real type is T.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn c_on_a_real_type_is_the_transpose() {
    check::<C, C>((4, 2, 3));
}

/// Needs a GPU. An empty product touches nothing and succeeds, its leading
/// dimensions raised to 1.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
fn an_empty_product_is_a_no_op() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    let a = DeviceBytes::new(8);
    let mut c = DeviceBytes::new(8);
    assert_eq!(DeviceSlice::<f64>::len(&c), 1);
    gemm::<N, N, f64>(&blas, 0, 0, 0, 1.0, &a, &a, 0.0, &mut c).unwrap();
    ok(owner.sync());
}
