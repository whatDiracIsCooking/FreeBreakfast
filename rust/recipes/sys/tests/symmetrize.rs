//! The symmetrize wrappers' own refusals, which the safe calls cannot reach,
//! called raw: each returns before it reads a pointer or touches a device.
//! Their launches on a card are `fb-recipes-symmetrize`'s tests.

use fb_recipes_sys::{
    FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS, fb_launch_status, fb_mirror_triangle,
    fb_symmetrize,
};

/// Each wrapper's status for `n` and `lda` over a null matrix.
fn both(n: i32, lda: i32) -> [fb_launch_status; 3] {
    let null = std::ptr::null_mut();
    // SAFETY: every call below returns before it reads a pointer.
    let average = unsafe { fb_symmetrize(null, n, null.cast(), lda) };
    // SAFETY: as above.
    let lower = unsafe { fb_mirror_triangle(null, true, n, null.cast(), lda) };
    // SAFETY: as above.
    let upper = unsafe { fb_mirror_triangle(null, false, n, null.cast(), lda) };
    [average, lower, upper]
}

/// An `n` of 0 enqueues nothing and succeeds, whatever `lda`; a negative
/// `n`, an `lda` below `n`, a null matrix and an order past the launch grid
/// are refused.
#[test]
fn refuses_what_the_cpp_module_refused() {
    assert_eq!(both(0, 0), [FB_LAUNCH_SUCCESS; 3]);
    assert_eq!(both(0, -1), [FB_LAUNCH_SUCCESS; 3]);
    assert_eq!(both(-1, 4), [FB_LAUNCH_INVALID_ARGUMENT; 3]);
    assert_eq!(both(5, 4), [FB_LAUNCH_INVALID_ARGUMENT; 3]);
    assert_eq!(both(5, 5), [FB_LAUNCH_INVALID_ARGUMENT; 3]);
    assert_eq!(both(2_097_121, 2_097_121), [FB_LAUNCH_INVALID_ARGUMENT; 3]);
}
