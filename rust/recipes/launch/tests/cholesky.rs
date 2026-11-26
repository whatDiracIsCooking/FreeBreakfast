//! The Cholesky tile wrapper's own refusals, which the safe call's
//! contracts keep a debug build from reaching, called raw: each returns
//! before it reads a pointer or touches a device. Its launches on a card
//! are `fb-recipes-cholesky`'s tests.

use fb_recipes_sys::{FB_CHOLESKY_TILE, FB_LAUNCH_INVALID_ARGUMENT, fb_cholesky_diagonal};

/// A tile order outside 1 to the tile, a negative first column, a leading
/// dimension under the order and each null pointer are refused, enqueueing
/// nothing.
#[test]
fn refuses_what_the_cpp_launcher_refused_and_null_pointers() {
    let stream = std::ptr::null_mut();
    let null = std::ptr::null_mut::<f64>();
    // A dangling pointer stands for each non-null operand: none is read.
    let (a, l11, info) = (
        std::ptr::NonNull::<f64>::dangling().as_ptr(),
        std::ptr::NonNull::<f64>::dangling().as_ptr(),
        std::ptr::NonNull::<i32>::dangling().as_ptr(),
    );
    let t = FB_CHOLESKY_TILE;
    let cases = [
        (0, 0, a, t, l11, info),
        (t + 1, 0, a, t + 1, l11, info),
        (4, -1, a, 4, l11, info),
        (4, 0, a, 3, l11, info),
        (4, 0, null, 4, l11, info),
        (4, 0, a, 4, null, info),
        (4, 0, a, 4, l11, std::ptr::null_mut()),
    ];
    for (jb, first, a, lda, l11, info) in cases {
        // SAFETY: every call returns before it reads a pointer or launches.
        let status = unsafe { fb_cholesky_diagonal(stream, jb, first, a, lda, l11, info) };
        assert_eq!(
            status, FB_LAUNCH_INVALID_ARGUMENT,
            "jb {jb}, first {first}, lda {lda}"
        );
    }
}
