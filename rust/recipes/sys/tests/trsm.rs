//! The triangular solve's diagonal-block wrappers' own refusals, which the
//! safe call's contracts keep a debug build from reaching, called raw: each
//! returns before it reads a pointer or touches a device. Their launches on
//! a card are `tests/trsm_solve.rs`.

use fb_recipes_sys::{
    FB_LAUNCH_INVALID_ARGUMENT, FB_TRSM_TILE, fb_trsm_diagonal_f32, fb_trsm_diagonal_f64,
};

/// A dangling pointer stands for each non-null operand: none is read.
fn pointers<T>(a_null: bool, b_null: bool) -> (*const T, *mut T) {
    let at = |null| {
        if null {
            std::ptr::null_mut::<T>()
        } else {
            std::ptr::NonNull::dangling().as_ptr()
        }
    };
    (at(a_null).cast_const(), at(b_null))
}

/// A block order outside 1 to the block, no right-hand side, a leading
/// dimension under its column (B's the right-hand sides' count when
/// transposed) and each null pointer are refused, enqueueing nothing, in
/// both element types.
#[test]
fn refuses_what_the_cpp_launcher_refused_short_leading_dimensions_and_null_pointers() {
    let stream = std::ptr::null_mut();
    let t = FB_TRSM_TILE;
    // (jb, ncols, lda, ldb, b_trans, a null, b null)
    let cases = [
        (0, 3, 4, 4, false, false, false),
        (t + 1, 3, t + 1, t + 1, false, false, false),
        (4, 0, 4, 4, false, false, false),
        (4, 3, 3, 4, false, false, false),
        (4, 3, 4, 3, false, false, false),
        (2, 3, 2, 2, true, false, false),
        (4, 3, 4, 4, false, true, false),
        (4, 3, 4, 4, false, false, true),
    ];
    for (jb, ncols, lda, ldb, b_trans, a_null, b_null) in cases {
        let (a, b) = pointers(a_null, b_null);
        // SAFETY: the call returns before it reads a pointer or launches.
        let status = unsafe {
            fb_trsm_diagonal_f64(
                stream, jb, ncols, 1.0, a, lda, false, false, false, b, ldb, b_trans,
            )
        };
        assert_eq!(
            status, FB_LAUNCH_INVALID_ARGUMENT,
            "f64: jb {jb}, ncols {ncols}, lda {lda}, ldb {ldb}"
        );
        let (a, b) = pointers(a_null, b_null);
        // SAFETY: as above.
        let status = unsafe {
            fb_trsm_diagonal_f32(
                stream, jb, ncols, 1.0, a, lda, false, false, false, b, ldb, b_trans,
            )
        };
        assert_eq!(
            status, FB_LAUNCH_INVALID_ARGUMENT,
            "f32: jb {jb}, ncols {ncols}, lda {lda}, ldb {ldb}"
        );
    }
}
