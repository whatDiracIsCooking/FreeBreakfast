//! The sandwich `C = OpX(X)ᴴ OpF(F) OpX(X)` on a card, against the product
//! on the host at the C++ `test/recipes/gemmwich_gpu.cpp` bars, with a
//! rectangular X and non-symmetric X and F, so the two orders, or an op on
//! the wrong side, cannot pass.
//!
//! The C++ square overload's cases (`DefaultIsXTransposeFX`,
//! `OpSelectsTheSide`) are these at `m = n`, the same call here; its raw
//! handle and pointer-mode cases have nothing to test, since every call
//! takes a `BlasHandle` and no safe call sets the pointer mode; its
//! `static_assert`s on the ops are `SandwichOp`'s `compile_fail` examples.

mod common;

use std::rc::Rc;

use common::{
    M, N, Scalar, adjoint, assert_matches, download, is_n, product, random, room, upload,
};
use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::ok;
use fb_recipes_blas::{C, N as OpN, SandwichOp, T, gemmwich};
use wwr_sys::{wwr_complex64, wwr_complex128};

/// One sandwich on a card at (m, n) = (`M`, `N`), C and `work` each checked
/// against the host's.
fn check<OpX: SandwichOp<S>, OpF: SandwichOp<S>, S: Scalar>(blas: &BlasHandle, owner: &Stream) {
    let (x_is_n, f_is_n) = (is_n::<OpX>(), is_n::<OpF>());
    let what = format!(
        "{} with OpX {}, OpF {}, seeds 8 and 9",
        std::any::type_name::<S>(),
        if x_is_n { "N" } else { "adjoint" },
        if f_is_n { "N" } else { "adjoint" },
    );
    let f = random::<S>(N, N, 8);
    // x is X as stored; opx is OpX(X), always n × m.
    let x = if x_is_n {
        random::<S>(N, M, 9)
    } else {
        random::<S>(M, N, 9)
    };
    let opx = if x_is_n { x.clone() } else { adjoint(&x) };
    let opf = if f_is_n { f.clone() } else { adjoint(&f) };
    let want_work = product(&opf, &opx);
    let want_c = product(&adjoint(&opx), &want_work);

    let (dx, df) = (upload::<S>(&x, owner), upload::<S>(&f, owner));
    let (mut work, mut c) = (room::<S>(N * M), room::<S>(M * M));
    gemmwich::<OpX, OpF, S>(blas, M as i32, N as i32, &dx, &df, &mut work, &mut c).unwrap();
    assert_matches::<S>(&download::<S>(&c, owner), &want_c, &format!("C, {what}"));
    assert_matches::<S>(
        &download::<S>(&work, owner),
        &want_work,
        &format!("work, {what}"),
    );
}

/// Needs a GPU. Every pairing of N and T on f32 and f64: a swapped
/// dimension, leading dimension or op on either gemm, or work left holding
/// anything but `OpF(F) OpX(X)`.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn real_every_op_pairing_matches_the_host() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    check::<OpN, OpN, f32>(&blas, &owner);
    check::<OpN, T, f32>(&blas, &owner);
    check::<T, OpN, f32>(&blas, &owner);
    check::<T, T, f32>(&blas, &owner);
    check::<OpN, OpN, f64>(&blas, &owner);
    check::<OpN, T, f64>(&blas, &owner);
    check::<T, OpN, f64>(&blas, &owner);
    check::<T, T, f64>(&blas, &owner);
}

/// Needs a GPU. Every pairing of N and C on both complex types, X and F
/// with non-zero imaginary parts: a transpose where the conjugate
/// transpose belongs, or the reverse.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn complex_every_op_pairing_matches_the_host() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    check::<OpN, OpN, wwr_complex64>(&blas, &owner);
    check::<OpN, C, wwr_complex64>(&blas, &owner);
    check::<C, OpN, wwr_complex64>(&blas, &owner);
    check::<C, C, wwr_complex64>(&blas, &owner);
    check::<OpN, OpN, wwr_complex128>(&blas, &owner);
    check::<OpN, C, wwr_complex128>(&blas, &owner);
    check::<C, OpN, wwr_complex128>(&blas, &owner);
    check::<C, C, wwr_complex128>(&blas, &owner);
}

/// Needs a GPU. A C one element short panics before the first gemm is
/// enqueued, so `work` is left as it was: the second gemm's operand checked
/// only after the first had run.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_short_c_panics_before_anything_is_enqueued() {
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    let (x, f) = (random::<f64>(N, M, 1), random::<f64>(N, N, 2));
    let before = random::<f64>(N, M, 3);
    let (dx, df) = (upload::<f64>(&x, &owner), upload::<f64>(&f, &owner));
    let mut work = upload::<f64>(&before, &owner);
    let mut c = room::<f64>(M * M - 1);
    let message = fb_test_support::panic_message(|| {
        let _ = gemmwich::<OpN, OpN, f64>(&blas, M as i32, N as i32, &dx, &df, &mut work, &mut c);
    });
    assert_eq!(
        message,
        "gemmwich: C is 23 x 23, 529 elements, but its slice holds 528"
    );
    ok(owner.sync());
    assert_eq!(download::<f64>(&work, &owner), before.v);
}
