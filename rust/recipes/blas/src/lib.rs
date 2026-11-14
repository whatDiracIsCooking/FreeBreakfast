//! BLAS conveniences over [`fb_core_wwr::blas`]'s checked calls: the C++
//! `src/recipes/blas`, whose README is the reasoning.
//!
//! The general calls stay [`fb_core_wwr::blas`]'s ([`blas::gemm`] and
//! [`blas::dgmm`], every leading dimension, op and side passed in); this crate is the shorthand for the
//! common case, and holds no `unsafe`. Every matrix is a
//! [`DeviceSlice`], checked against its extent by the call underneath.
//!
//! | C++ | Here |
//! |---|---|
//! | `template <wwrblasOperation_t OpA, OpB>`, each `WWRBLAS_OP_N` when not given | type parameters, the markers [`N`], [`T`] and [`C`] behind [`BlasOp`]: `gemm::<T, N, _>`, both always given (a function has no default type parameter) |
//! | `concept BlasOp<Op>` | the trait [`BlasOp`] |
//! | `gemm<OpA, OpB>(handle, m, n, k, alpha, a, b, beta, c)`, packed | [`gemm`], packed |
//! | `gemm<OpA, OpB>(handle, n, alpha, a, b, beta, c)`, every matrix n × n | [`gemm`] at `m = n = k`: one name, not two |
//! | `alpha`, `beta` by pointer, as the handle's pointer mode says | by value, read in the default (host) mode |
//! | `handle`, a `BlasHandle` or a raw `wwrblasHandle_t` | a [`BlasHandle`] |
//! | a raw `wwrblasStatus_t`, the caller picking the tier | a [`Status`] |
//! | `wwr::dgmm<T>(handle, side, m, n, a, m, x, 1, c, m)`, as `gmmwich` and `orthogonalize` call it, packed | [`dgmm`], the side a type parameter, the marker [`Left`] or [`Right`] behind [`BlasSide`]: `dgmm::<Right, _>` |
//! | `gemmwich<OpX, OpF>(handle, m, n, x, f, work, c)`, `C = OpX(X)ᴴ OpF(F) OpX(X)` | [`gemmwich`], two [`gemm`]s: `gemmwich::<N, N, _>` |
//! | `gemmwich<OpX, OpF>(handle, n, x, f, work, c)`, every matrix n × n | [`gemmwich`] at `m = n`: one name, not two |
//! | `gmmwich<OpX>(handle, m, n, x, d, work, c)`, `C = OpX(X)ᴴ diag(d) OpX(X)` | [`gmmwich`], a [`dgmm`] and a [`gemm`]: `gmmwich::<T, _>` |
//! | `gmmwich<OpX>(handle, n, x, d, work, c)` | [`gmmwich`] at `m = n` |
//! | `concept SandwichOp<Op, T>`, `kAdjointOp<T>` | the trait [`SandwichOp`], implemented for each scalar by [`N`] and the one adjoint marker, its associated type the op that gives `OpX(X)ᴴ` from X as stored |
//! | `raw_blas_handle`, a `BlasHandle` or a raw handle unwrapped | not ported: every call takes a [`BlasHandle`] |
//! | `ScopedPointerMode`, the handle held in host pointer mode for a sandwich | not ported: no safe call sets a handle's pointer mode, so [`blas::gemm`]'s host scalars are always read in the default (host) mode, and there is nothing to restore |
//! | "`work` must not alias X, F or C, and C must not alias X or F" | the borrow checker: `work` and C are each a [`DeviceSliceMut`], borrowed exclusively |
//! | the scalars `kOne<T>`, `kZero<T>` | [`BlasScalar::ONE`] and [`BlasScalar::ZERO`] |
//!
//! [`blas::gemm`]: fb_core_wwr::blas::gemm
//! [`blas::dgmm`]: fb_core_wwr::blas::dgmm
//!
//! **`ᴴ` is the scalar's adjoint**: the conjugate transpose on a complex
//! type, the plain transpose on a real one, where the sandwiches are
//! `Xᵀ F X` and `Xᵀ diag(d) X`. On a complex type the sandwich is the one
//! that keeps a Hermitian F Hermitian; the complex-symmetric `Xᵀ F X` is
//! not offered.

#![forbid(unsafe_code)]

use fb_core_error_handling::Status;
use fb_core_wwr::blas::{self, BlasHandle, BlasScalar};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use wwr_sys::{wwr_blas_operation, wwr_blas_side_mode, wwr_complex64, wwr_complex128};

/// A BLAS operation fixed at compile time: [`N`], [`T`] or [`C`]. The C++
/// `BlasOp` concept, over a non-type template parameter Rust's stable
/// const generics cannot take (an enum).
pub trait BlasOp {
    /// The operation, as the library takes it.
    const RAW: wwr_blas_operation;
}

/// `op(X) = X`.
pub struct N;
/// `op(X) = Xᵀ`.
pub struct T;
/// `op(X) = Xᴴ`, the conjugate transpose: the same as [`T`] on a real type.
pub struct C;

impl BlasOp for N {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
}
impl BlasOp for T {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
}
impl BlasOp for C {
    const RAW: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_C;
}

/// `C = alpha OpA(A) OpB(B) + beta C` on `handle`'s stream, for
/// `OpA(A)` m × k, `OpB(B)` k × n and C m × n, every matrix column-major
/// and packed: A is stored m × k under [`N`] and k × m otherwise (its
/// leading dimension m or k), B k × n or n × k (k or n), and C's leading
/// dimension is m. [`blas::gemm`] with those leading dimensions; for a
/// padded one (a block of a larger matrix) or an op known only at run
/// time, call that.
///
/// `alpha` and `beta` are host values. With `beta` zero, C need not be
/// initialised. A leading dimension is at least 1, as the library asks, so
/// an empty product is a call that touches nothing.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// As [`blas::gemm`]: a negative dimension, or a slice shorter than its
/// matrix; before anything is enqueued.
///
/// [`blas::gemm`]: fb_core_wwr::blas::gemm
#[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
#[track_caller]
pub fn gemm<OpA: BlasOp, OpB: BlasOp, S: BlasScalar>(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    k: i32,
    alpha: S,
    a: &impl DeviceSlice<S>,
    b: &impl DeviceSlice<S>,
    beta: S,
    c: &mut impl DeviceSliceMut<S>,
) -> Status {
    let is_n = |op| op == wwr_blas_operation::WWR_BLAS_OP_N;
    let lda = if is_n(OpA::RAW) { m } else { k };
    let ldb = if is_n(OpB::RAW) { k } else { n };
    blas::gemm(
        handle,
        OpA::RAW,
        OpB::RAW,
        m,
        n,
        k,
        alpha,
        a,
        lda.max(1),
        b,
        ldb.max(1),
        beta,
        c,
        m.max(1),
    )
}

/// The side a [`dgmm`]'s diagonal stands on, fixed at compile time:
/// [`Left`] or [`Right`]. As [`BlasOp`], a trait over markers where the
/// C++ would take the enum as a template parameter.
pub trait BlasSide {
    /// The side, as the library takes it.
    const RAW: wwr_blas_side_mode;
}

/// `diag(x) A`: each row of A scaled.
pub struct Left;
/// `A diag(x)`: each column of A scaled.
pub struct Right;

impl BlasSide for Left {
    const RAW: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
}
impl BlasSide for Right {
    const RAW: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;
}

/// `C = diag(x) A` under [`Left`] or `C = A diag(x)` under [`Right`] on
/// `handle`'s stream, A and C m × n, column-major and packed (leading
/// dimension m), x the diagonal, m entries on the left and n on the right,
/// contiguous. [`blas::dgmm`] with those leading dimensions and stride 1;
/// for a padded matrix, a strided x or a side known only at run time, call
/// that. C is memory of its own: there is no in-place call.
///
/// # Errors
/// The library's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// As [`blas::dgmm`]: a negative dimension, or a slice shorter than its
/// operand; before anything is enqueued.
///
/// [`blas::dgmm`]: fb_core_wwr::blas::dgmm
#[track_caller]
pub fn dgmm<Side: BlasSide, S: BlasScalar>(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    a: &impl DeviceSlice<S>,
    x: &impl DeviceSlice<S>,
    c: &mut impl DeviceSliceMut<S>,
) -> Status {
    blas::dgmm(handle, Side::RAW, m, n, a, m.max(1), x, 1, c, m.max(1))
}

/// An op a sandwich ([`gemmwich`], [`gmmwich`]) takes on a scalar `S`:
/// [`N`], or `S`'s adjoint in its one spelling, [`T`] on a real `S` and
/// [`C`] on a complex one. The C++ `SandwichOp<Op, T>`.
///
/// A real type leaves out [`C`], which on it is [`T`] under another name:
///
/// ```compile_fail,E0277
/// use fb_recipes_blas::{C, SandwichOp};
/// fn takes<Op: SandwichOp<f64>>() {}
/// takes::<C>();
/// ```
///
/// A complex type leaves out [`T`]: the sandwich's other factor would be
/// `(Xᵀ)ᴴ = conj(X)`, which no BLAS op spells:
///
/// ```compile_fail,E0277
/// use fb_recipes_blas::{SandwichOp, T};
/// use wwr_sys::wwr_complex128;
/// fn takes<Op: SandwichOp<wwr_complex128>>() {}
/// takes::<T>();
/// ```
pub trait SandwichOp<S: BlasScalar>: BlasOp {
    /// The op that, applied to X as stored, gives `Op(X)ᴴ`: `S`'s adjoint
    /// for [`N`], and [`N`] for the adjoint.
    type Adjoint: BlasOp;
}

macro_rules! sandwich_ops {
    ($($s:ty => $adjoint:ty);* $(;)?) => {$(
        impl SandwichOp<$s> for N {
            type Adjoint = $adjoint;
        }
        impl SandwichOp<$s> for $adjoint {
            type Adjoint = N;
        }
    )*};
}

sandwich_ops! {
    f32 => T;
    f64 => T;
    wwr_complex64 => C;
    wwr_complex128 => C;
}

/// Panic unless C, m × m and packed, fits in its slice: checked before the
/// first of a sandwich's two calls, so a short C enqueues nothing. The
/// other operands are each checked by the first call, which reads or
/// writes all of them.
#[track_caller]
fn check_c(routine: &str, len: usize, m: i32) {
    // m's sign is the first call's to check; a negative one needs nothing.
    let need = u64::try_from(m).map_or(0, |m| m * m);
    assert!(
        need <= len as u64,
        "{routine}: C is {m} x {m}, {need} elements, but its slice holds {len}"
    );
}

/// `C = OpX(X)ᴴ OpF(F) OpX(X)` on `handle`'s stream, for `OpX(X)` n × m, F
/// n × n and C m × m, through an n × m `work`: the sandwich product, as two
/// [`gemm`]s with no host synchronisation between them,
///
/// ```text
/// work = OpF(F) OpX(X)
/// C    = OpX(X)ᴴ work
/// ```
///
/// `OpX` and `OpF` are each [`N`] or `S`'s adjoint ([`SandwichOp`]). Every
/// matrix is column-major and packed, so its leading dimension is its row
/// count as stored:
///
/// | Matrix | Stored as | Leading dimension |
/// |---|---|---|
/// | X, `OpX` = [`N`] | n × m | n |
/// | X, `OpX` = the adjoint | m × n | m |
/// | F | n × n | n |
/// | work | n × m | n |
/// | C | m × m | m |
///
/// So `OpX` = [`N`] is `Xᴴ OpF(F) X`, the projection of F onto X's m
/// columns, and at `m = n` the adjoint is `X F Xᴴ`. F and C need not be
/// symmetric (or Hermitian). `work` is scratch: ignored on entry and left
/// holding `OpF(F) OpX(X)`. C is overwritten, not accumulated into.
///
/// ```no_run
/// use fb_core_error_handling::Status;
/// use fb_core_wwr::blas::BlasHandle;
/// use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
/// use fb_recipes_blas::{N, gemmwich};
///
/// /// C (m × m) = Xᵀ F X, X stored n × m.
/// fn project(
///     blas: &BlasHandle,
///     (m, n): (i32, i32),
///     x: &impl DeviceSlice<f64>,
///     f: &impl DeviceSlice<f64>,
///     work: &mut impl DeviceSliceMut<f64>,
///     c: &mut impl DeviceSliceMut<f64>,
/// ) -> Status {
///     gemmwich::<N, N, f64>(blas, m, n, x, f, work, c)
/// }
/// ```
///
/// # Errors
/// The first failing gemm's refusal, as a `GpuRuntime` error naming the
/// status; C, and possibly `work`, are then unspecified.
///
/// # Panics
/// If m or n is negative, or a slice is shorter than its matrix; before
/// anything is enqueued.
#[track_caller]
pub fn gemmwich<OpX: SandwichOp<S>, OpF: SandwichOp<S>, S: BlasScalar>(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    x: &impl DeviceSlice<S>,
    f: &impl DeviceSlice<S>,
    work: &mut impl DeviceSliceMut<S>,
    c: &mut impl DeviceSliceMut<S>,
) -> Status {
    check_c("gemmwich", c.len(), m);
    // work (n × m) = OpF(F) (n × n) OpX(X) (n × m)
    gemm::<OpF, OpX, S>(handle, n, m, n, S::ONE, f, x, S::ZERO, work)?;
    // C (m × m) = OpX(X)ᴴ (m × n) work (n × m)
    gemm::<OpX::Adjoint, N, S>(handle, m, m, n, S::ONE, x, &*work, S::ZERO, c)
}

/// `C = OpX(X)ᴴ diag(d) OpX(X)` on `handle`'s stream, for `OpX(X)` n × m, d
/// of n entries and C m × m, through a `work` of X's stored shape:
/// [`gemmwich`] with F = diag(d), the diagonal never formed. A [`dgmm`]
/// scales X in O(nm), where a gemm against a dense diagonal would be
/// O(n²m), and one [`gemm`] finishes, with no host synchronisation between
/// them. There is no `OpF`: a diagonal is its own transpose.
///
/// `OpX` is [`N`] or `S`'s adjoint ([`SandwichOp`]). Every matrix is
/// column-major and packed, and d is contiguous and an `S` too, so on a
/// complex `S` a real diagonal (eigenvalues) is stored with zero imaginary
/// parts. The diagonal goes on whichever side of X needs no adjoint:
///
/// | `OpX` | X stored as | `work =` | `C =` |
/// |---|---|---|---|
/// | [`N`] | n × m, leading dimension n | `diag(d) X` | `Xᴴ work` |
/// | the adjoint | m × n, leading dimension m | `X diag(d)` | `work Xᴴ` |
///
/// The adjoint with an eigenvector matrix U at `m = n` is the spectral
/// recombination `U f(Λ) Uᴴ`. `work` is scratch: ignored on entry and left
/// as in the table. C is overwritten, and Hermitian (symmetric, on a real
/// `S`) up to rounding when d is real.
///
/// ```no_run
/// use fb_core_error_handling::Status;
/// use fb_core_wwr::blas::BlasHandle;
/// use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
/// use fb_recipes_blas::{T, gmmwich};
///
/// /// X = U diag(λ^(-1/2)) Uᵀ, U n × n; scratch is left holding
/// /// U diag(λ^(-1/2)).
/// fn inverse_sqrt(
///     blas: &BlasHandle,
///     n: i32,
///     u: &impl DeviceSlice<f64>,
///     inv_sqrt_lambda: &impl DeviceSlice<f64>,
///     scratch: &mut impl DeviceSliceMut<f64>,
///     x: &mut impl DeviceSliceMut<f64>,
/// ) -> Status {
///     gmmwich::<T, f64>(blas, n, n, u, inv_sqrt_lambda, scratch, x)
/// }
/// ```
///
/// # Errors
/// The first failing call's refusal, as a `GpuRuntime` error naming the
/// status; C, and possibly `work`, are then unspecified.
///
/// # Panics
/// If m or n is negative, or a slice is shorter than its operand; before
/// anything is enqueued.
#[track_caller]
pub fn gmmwich<OpX: SandwichOp<S>, S: BlasScalar>(
    handle: &BlasHandle,
    m: i32,
    n: i32,
    x: &impl DeviceSlice<S>,
    d: &impl DeviceSlice<S>,
    work: &mut impl DeviceSliceMut<S>,
    c: &mut impl DeviceSliceMut<S>,
) -> Status {
    check_c("gmmwich", c.len(), m);
    if OpX::RAW == wwr_blas_operation::WWR_BLAS_OP_N {
        // work (n × m) = diag(d) X; C (m × m) = Xᴴ work
        dgmm::<Left, S>(handle, n, m, x, d, work)?;
        gemm::<OpX::Adjoint, N, S>(handle, m, m, n, S::ONE, x, &*work, S::ZERO, c)
    } else {
        // work (m × n) = X diag(d); C (m × m) = work Xᴴ, Xᴴ = OpX(X)
        dgmm::<Right, S>(handle, m, n, x, d, work)?;
        gemm::<N, OpX, S>(handle, m, m, n, S::ONE, &*work, x, S::ZERO, c)
    }
}
