//! [`trsm`](crate::trsm), BLAS's triangular solve on the GPU, in place: `B ← alpha op(A)⁻¹ B` on
//! the left or `B ← alpha B op(A)⁻¹` on the right, the tree's own blocked
//! solve, never the BLAS library's `trsm`. The C++ `src/recipes/trsm` is the
//! reference design, and its README has why not the library's: on ROCm,
//! rocBLAS's trsm takes workspace that grows with the right-hand sides, so
//! no fixed handle workspace covers it and a call past it allocates outside
//! any budget. This one needs nothing but the gemm the handle's fixed
//! workspace already serves. One code path for both backends, so CUDA runs
//! it too.
//!
//! **The algorithm.** One right-hand side (`n = 1` on the left, `m = 1` on
//! the right) with a nonzero `alpha` is a vector: a [`scal`] when `alpha` is
//! not 1, then the library's [`trsv`], which takes no workspace on either
//! backend. The rest is the blocked solve. Every case is put in one frame,
//! `M Y = alpha C` with M of order k: on the left M is `op(A)` and C is B,
//! on the right M is `op(A)ᵀ` and C is `Bᵀ`. A lower M is solved from the
//! top block down, an upper one from the bottom up, [`FB_TRSM_TILE`] (64)
//! rows at a time:
//!
//! 1. the diagonal block, by `libfb_recipes_kernels.so`'s shared-memory
//!    kernel ([`trsm_diagonal`]);
//! 2. the rows of C still to come lose M's off-diagonal block times the rows
//!    just solved: one gemm, written in terms of A and B as they are stored,
//!    so neither is ever transposed or copied.
//!
//! `alpha` is applied once: the first diagonal block scales by it, and the
//! first gemm scales every other row by its beta. With `alpha` zero B is set
//! to zero block by block and neither A nor B is read, as the reference BLAS
//! does. Off-diagonal blocks lie wholly in A's stored triangle, and the
//! kernel reads only the stored triangle (and not the diagonal under
//! `WWR_BLAS_DIAG_UNIT`), so whatever the other triangle holds never reaches
//! the result.
//!
//! **The blocks of one matrix.** On the right the rows of C are columns of
//! B, so the block solved and the columns still to come are the two sides of
//! one cut ([`split_at_mut`]) and the update is a plain [`gemm`]. On the left
//! they are rows of the same columns, which no cut parts: the update is
//! [`gemm_rows`], which takes B whole and checks the two blocks of rows
//! apart.
//!
//! This module holds no `unsafe`: the launch is [`trsm_diagonal`]'s, in the
//! crate root, and the BLAS calls are `fb_core_wwr::blas`'s checked ones.
//!
//! [`gemm`]: fb_core_wwr::blas::gemm
//! [`gemm_rows`]: fb_core_wwr::blas::gemm_rows
//! [`scal`]: fb_core_wwr::blas::scal
//! [`trsv`]: fb_core_wwr::blas::trsv
//! [`split_at_mut`]: fb_core_wwr::device_slice::split_at_mut
//! [`FB_TRSM_TILE`]: crate::FB_TRSM_TILE
//! [`trsm_diagonal`]: crate::trsm_diagonal

#![forbid(unsafe_code)]

use contracts::requires;
use fb_core_error_handling::Status;
use fb_core_log::le;
use fb_core_wwr::blas::{BlasHandle, gemm, gemm_rows, scal, trsv};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at, split_at_mut};
use fb_core_wwr::runtime::Stream;

use crate::{FB_TRSM_TILE, TrsmScalar, trsm_diagonal};
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode};

/// The elements a column-major `rows × cols` matrix at leading dimension
/// `ld` spans, 0 for an empty one: what [`trsm`]'s contracts hold each slice
/// to. Widened first, so the product of two `i32`s cannot overflow.
fn extent(rows: i32, cols: i32, ld: i32) -> i64 {
    if rows <= 0 || cols <= 0 {
        return 0;
    }
    i64::from(ld) * (i64::from(cols) - 1) + i64::from(rows)
}

/// The order of A: m on the left, n on the right.
fn order(side: wwr_blas_side_mode, m: i32, n: i32) -> i32 {
    match side {
        wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT => m,
        wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT => n,
    }
}

/// BLAS's trsm on `blas`'s stream: `B ← alpha op(A)⁻¹ B` (`side`
/// `WWR_BLAS_SIDE_LEFT`) or `B ← alpha B op(A)⁻¹` (`_RIGHT`), in place, for
/// a triangular A.
///
/// B is m × n at leading dimension `ldb`. A is triangular of order k, m on
/// the left and n on the right, at leading dimension `lda`; `fill` says
/// which triangle holds it, and only that triangle is read. `op` is
/// `WWR_BLAS_OP_N`, `_T` or `_C` (on a real `T`, `_C` is `_T`). With
/// `WWR_BLAS_DIAG_UNIT` the diagonal is taken as 1 and not read. Every
/// matrix is column-major. With `alpha` zero, B becomes zero and neither A
/// nor B's old values are read, as in the reference BLAS. With m or n zero
/// nothing is enqueued.
///
/// A singular A is not detected: as in BLAS, a zero pivot gives Inf or NaN.
///
/// Everything is enqueued on the stream `blas` is bound to, with `blas` in
/// host pointer mode (its default), and nothing is synchronized. One
/// right-hand side goes to the library's `trsv` (after a `scal` for an
/// `alpha` that is not 1), which takes no workspace; the library's trsm is
/// never called, and no workspace is taken, so the call has no row in any
/// layout. Otherwise the solve is blocked, [`FB_TRSM_TILE`] rows at a time:
/// each diagonal block by the shared-memory kernel ([`fb_trsm_diagonal_f64`]
/// or [`fb_trsm_diagonal_f32`]), then one gemm that takes that block's
/// solution out of the rows still to come. The caller has the handle's
/// device current, and has run [`check_library`] before its first launch.
/// The C++ `src/recipes/trsm` is the reference design; its README has why
/// not the library's, and the timings against it.
///
/// [`FB_TRSM_TILE`]: crate::FB_TRSM_TILE
/// [`fb_trsm_diagonal_f64`]: crate::fb_trsm_diagonal_f64
/// [`fb_trsm_diagonal_f32`]: crate::fb_trsm_diagonal_f32
/// [`check_library`]: crate::check_library
///
/// ```no_run
/// use std::sync::Arc;
///
/// use fb_core_wwr::blas::BlasHandle;
/// use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
/// use fb_core_wwr::runtime::Stream;
/// use fb_recipes_sys::trsm;
/// use wwr_sys::{
///     wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode,
/// };
///
/// /// `B ← L⁻¹ B` for a lower-triangular `naux × naux` L and `cols`
/// /// right-hand sides: a density fit.
/// fn fit(
///     l: &impl DeviceSlice<f64>,
///     b: &mut impl DeviceSliceMut<f64>,
///     naux: i32,
///     cols: i32,
/// ) {
///     fb_recipes_sys::check_library().unwrap();
///     let stream = Arc::new(Stream::new(0));
///     let blas = BlasHandle::new(stream.clone());
///     trsm(
///         &blas,
///         wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
///         wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
///         wwr_blas_operation::WWR_BLAS_OP_N,
///         wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
///         naux,
///         cols,
///         1.0,
///         l,
///         naux,
///         b,
///         naux,
///     )
///     .unwrap();
/// }
/// ```
///
/// # Errors
/// The first launch's or BLAS call's failure, as a `GpuRuntime` error; what
/// was enqueued before it stays enqueued.
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[requires(le!(0, m) && le!(0, n), "m and n are not negative")]
#[requires(
    le!(order(side, m, n).max(1), lda),
    "A's leading dimension is at least max(1, k)"
)]
#[requires(le!(m.max(1), ldb), "B's leading dimension is at least max(1, m)")]
#[requires(
    le!(extent(order(side, m, n), order(side, m, n), lda), a.len() as i64),
    "a holds the k × k triangle at lda"
)]
#[requires(
    le!(extent(m, n, ldb), b.len() as i64),
    "b holds the m × n matrix at ldb"
)]
#[track_caller]
pub fn trsm<T: TrsmScalar>(
    blas: &BlasHandle,
    side: wwr_blas_side_mode,
    fill: wwr_blas_fill_mode,
    op: wwr_blas_operation,
    diag: wwr_blas_diag_type,
    m: i32,
    n: i32,
    alpha: T,
    a: &impl DeviceSlice<T>,
    lda: i32,
    b: &mut impl DeviceSliceMut<T>,
    ldb: i32,
) -> Status {
    if m == 0 || n == 0 {
        return Ok(());
    }
    let left = side == wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
    let trans = op != wwr_blas_operation::WWR_BLAS_OP_N;
    let k = order(side, m, n);
    let zero = alpha == T::ZERO;
    // One right-hand side is a vector: the library's trsv, which takes no
    // workspace on either backend, in place of a launch and a gemm per
    // block. On the right, x op(A) = alpha b is op(A)ᵀ xᵀ = alpha bᵀ, over
    // b's stride. alpha = 0 stays below, where A and B are not read.
    if (if left { n } else { m }) == 1 && !zero {
        let inc = if left { 1 } else { ldb };
        if alpha != T::ONE {
            scal(blas, k, alpha, b, inc)?;
        }
        let vector_op = if left == trans {
            wwr_blas_operation::WWR_BLAS_OP_T
        } else {
            wwr_blas_operation::WWR_BLAS_OP_N
        };
        return trsv(blas, fill, vector_op, diag, k, a, lda, b, inc);
    }

    let stream: &Stream = blas.owner();
    let lower_fill = fill == wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
    // op(A) is lower for (lower, N) and (upper, T); M is op(A) on the left
    // and its transpose on the right.
    let lower = if left {
        lower_fill != trans
    } else {
        lower_fill == trans
    };
    // M(p, q) is A(q, p) where M is op(A)ᵀ on the right or Aᵀ on the left.
    let a_trans = if left { trans } else { !trans };
    let unit = diag == wwr_blas_diag_type::WWR_BLAS_DIAG_UNIT;
    let a_op = if trans {
        wwr_blas_operation::WWR_BLAS_OP_T
    } else {
        wwr_blas_operation::WWR_BLAS_OP_N
    };
    // The contracts hold every count below non-negative, so each is its own
    // usize.
    let (block, order, la, lb) = (
        FB_TRSM_TILE as usize,
        k as usize,
        lda as usize,
        ldb as usize,
    );
    let ncols = if left { n } else { m } as usize;
    let blocks = order.div_ceil(block);
    for step in 0..blocks {
        let k0 = block * if lower { step } else { blocks - 1 - step };
        let jb = block.min(order - k0);
        // alpha is applied once per row of C: the first diagonal block takes
        // it, and the first gemm scales every other row by its beta.
        // alpha = 0 keeps s = 0 on every block, which zeroes it unread.
        let first = step == 0;
        let scale = if first || zero { alpha } else { T::ONE };
        {
            let (_, triangle) = split_at(a, k0 + k0 * la);
            // Row k0 of C: row k0 of B on the left, column k0 on the right.
            let (_, mut tile) = split_at_mut(b, if left { k0 } else { k0 * lb });
            trsm_diagonal(
                stream, jb, ncols, scale, &triangle, la, a_trans, !lower, unit, &mut tile, lb,
                !left,
            )?;
        }
        // alpha = 0 zeroed this block without reading it, and zeroes the
        // rest block by block the same way: no gemm, which would read A.
        if zero {
            continue;
        }
        // The rows still to come: below the block for a lower M, above it
        // for an upper one.
        let r0 = if lower { k0 + jb } else { 0 };
        let rows = if lower { order - r0 } else { k0 };
        if rows == 0 {
            continue;
        }
        // M(rest, block) in A's storage. On the left it is
        // op(A)(rest, block): A at (rest, block) under N, A at (block, rest)
        // transposed under T. On the right it is op(A)(block, rest)ᵀ, so the
        // gemm multiplies by op(A)(block, rest): A at (block, rest) under N,
        // A at (rest, block) transposed under T.
        let (_, off) = split_at(
            a,
            if left != trans {
                r0 + k0 * la
            } else {
                k0 + r0 * la
            },
        );
        let beta = if first { alpha } else { T::ONE };
        // Each count is at most k, an i32.
        let (jb, rows, k0, r0) = (jb as i32, rows as i32, k0 as i32, r0 as i32);
        if left {
            // B(rest, :) = beta B(rest, :) - op(A)(rest, block) B(block, :)
            gemm_rows(
                blas,
                a_op,
                rows,
                n,
                jb,
                -T::ONE,
                &off,
                lda,
                b,
                ldb,
                k0,
                beta,
                r0,
            )?;
            continue;
        }
        // B(:, rest) = beta B(:, rest) - B(:, block) op(A)(block, rest):
        // the block's columns and the rest's are the two sides of a cut at
        // the later one's first column.
        let (head, tail) = split_at_mut(b, lb * k0.max(r0) as usize);
        let (solved, mut rest) = if lower {
            // The head ends with the block's columns.
            let (_, solved) = split_at(&head, lb * k0 as usize);
            (solved, tail)
        } else {
            let (solved, _) = split_at(&tail, tail.len());
            (solved, head)
        };
        gemm(
            blas,
            wwr_blas_operation::WWR_BLAS_OP_N,
            a_op,
            m,
            rows,
            jb,
            -T::ONE,
            &solved,
            ldb,
            &off,
            lda,
            beta,
            &mut rest,
            ldb,
        )?;
    }
    Ok(())
}
