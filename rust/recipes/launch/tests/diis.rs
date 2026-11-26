//! The DIIS solve wrapper's own refusals, which the safe call's contracts
//! keep a debug build from reaching, called raw: each returns before it
//! reads a pointer or touches a device. Its launches on a card are
//! `fb-recipes-diis`'s tests.

use fb_recipes_sys::{FB_DIIS_SOLVE_MAX_DIM, FB_LAUNCH_INVALID_ARGUMENT, fb_diis_solve};

/// A subspace below one pair, a leading dimension under it or reaching the
/// shared array's order, a newest pair outside the subspace and each null
/// pointer are refused, enqueueing nothing.
#[test]
fn refuses_the_cpp_bounds_and_null_pointers() {
    let stream = std::ptr::null_mut();
    // A dangling pointer stands for each non-null operand: none is read.
    let gram = std::ptr::NonNull::<f64>::dangling().as_ptr().cast_const();
    let coeff = std::ptr::NonNull::<f64>::dangling().as_ptr();
    let flag = std::ptr::NonNull::<i32>::dangling().as_ptr();
    let max = FB_DIIS_SOLVE_MAX_DIM;
    let cases = [
        (0, 4, 0, gram, coeff, flag),
        (3, 2, 0, gram, coeff, flag),
        (3, max, 0, gram, coeff, flag),
        (max, max, 0, gram, coeff, flag),
        (3, 4, -1, gram, coeff, flag),
        (3, 4, 3, gram, coeff, flag),
        (3, 4, 0, std::ptr::null(), coeff, flag),
        (3, 4, 0, gram, std::ptr::null_mut(), flag),
        (3, 4, 0, gram, coeff, std::ptr::null_mut()),
    ];
    for (m, ld, newest, gram, coeff, flag) in cases {
        // SAFETY: every call returns before it reads a pointer or launches.
        let status = unsafe { fb_diis_solve(stream, m, ld, newest, gram, coeff, flag) };
        assert_eq!(
            status, FB_LAUNCH_INVALID_ARGUMENT,
            "m {m}, ld {ld}, newest {newest}"
        );
    }
}
