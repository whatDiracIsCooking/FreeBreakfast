//! The diagonal sandwich `C = OpX(X)ᴴ diag(d) OpX(X)` on a card, against
//! the product on the host at the C++ `test/recipes/gmmwich_gpu.cpp` bars,
//! with a rectangular, non-symmetric X, so the diagonal on the wrong side
//! of X, or an op on the wrong factor, cannot pass.
//!
//! As for `gemmwich`, the C++ square overload's cases are these at `m = n`,
//! and its raw handle and pointer-mode cases have nothing to test.

mod common;

use std::rc::Rc;

use common::{
    M, Mat, N, Scalar, adjoint, assert_matches, download, is_n, product, random, room, upload,
};
use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::ok;
use fb_recipes_blas::{C, N as OpN, SandwichOp, T, gmmwich};
use wwr_sys::{wwr_complex64, wwr_complex128};

/// `diag(d)` as a dense n × n matrix, for the reference only.
pub fn diagonal(d: &Mat) -> Mat {
    let n = d.v.len();
    let mut v = vec![(0.0, 0.0); n * n];
    for (i, &z) in d.v.iter().enumerate() {
        v[i + i * n] = z;
    }
    Mat {
        rows: n,
        cols: n,
        v,
    }
}

/// One diagonal sandwich on a card at (m, n) = (`M`, `N`), C and `work`
/// each checked against the host's: `work` is `diag(d) X` under N and
/// `X diag(d)` under the adjoint, X's stored shape either way.
fn check<OpX: SandwichOp<S>, S: Scalar>(blas: &BlasHandle, owner: &Stream) {
    let x_is_n = is_n::<OpX>();
    let what = format!(
        "{} with OpX {}, seeds 4 and 5",
        std::any::type_name::<S>(),
        if x_is_n { "N" } else { "adjoint" },
    );
    let d = random::<S>(N, 1, 4);
    let x = if x_is_n {
        random::<S>(N, M, 5)
    } else {
        random::<S>(M, N, 5)
    };
    let opx = if x_is_n { x.clone() } else { adjoint(&x) };
    let dense = diagonal(&d);
    let want_work = if x_is_n {
        product(&dense, &x)
    } else {
        product(&x, &dense)
    };
    let want_c = product(&adjoint(&opx), &product(&dense, &opx));

    let (dx, dd) = (upload::<S>(&x, owner), upload::<S>(&d, owner));
    let (mut work, mut c) = (room::<S>(N * M), room::<S>(M * M));
    gmmwich::<OpX, S>(blas, M as i32, N as i32, &dx, &dd, &mut work, &mut c).unwrap();
    assert_matches::<S>(&download::<S>(&c, owner), &want_c, &format!("C, {what}"));
    assert_matches::<S>(
        &download::<S>(&work, owner),
        &want_work,
        &format!("work, {what}"),
    );
}

/// Needs a GPU. N and T on f32 and f64: the dgmm's side or shape, or the
/// gemm's ops, operand order or leading dimensions, swapped.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn real_each_op_matches_the_host() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    check::<OpN, f32>(&blas, &owner);
    check::<T, f32>(&blas, &owner);
    check::<OpN, f64>(&blas, &owner);
    check::<T, f64>(&blas, &owner);
}

/// Needs a GPU. N and C on both complex types, X and d with non-zero
/// imaginary parts: a transpose where the conjugate transpose belongs.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn complex_each_op_matches_the_host() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    check::<OpN, wwr_complex64>(&blas, &owner);
    check::<C, wwr_complex64>(&blas, &owner);
    check::<OpN, wwr_complex128>(&blas, &owner);
    check::<C, wwr_complex128>(&blas, &owner);
}

/// Needs a GPU. A C one element short panics before the dgmm is enqueued,
/// so `work` is left as it was.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
fn a_short_c_panics_before_anything_is_enqueued() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    let (x, d) = (random::<f64>(M, N, 1), random::<f64>(N, 1, 2));
    let before = random::<f64>(M, N, 3);
    let (dx, dd) = (upload::<f64>(&x, &owner), upload::<f64>(&d, &owner));
    let mut work = upload::<f64>(&before, &owner);
    let mut c = room::<f64>(M * M - 1);
    let message = fb_test_support::panic_message(|| {
        let _ = gmmwich::<T, f64>(&blas, M as i32, N as i32, &dx, &dd, &mut work, &mut c);
    });
    assert_eq!(
        message,
        "gmmwich: C is 23 x 23, 529 elements, but its slice holds 528"
    );
    ok(owner.sync());
    assert_eq!(download::<f64>(&work, &owner), before.v);
}
