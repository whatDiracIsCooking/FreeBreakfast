//! `src/core/primitives` (module `free_breakfast.core.primitives`): the small
//! exact-in-double combinatorics a Gaussian basis is normalized with.
//!
//! Only what `atom`, `molecule`, `basis`, `io/xyz` and `io/basis` name is
//! ported; the rest waits for the module that first needs it.
//!
//! | C++ (`free_breakfast::math`) | Here |
//! |---|---|
//! | `double_factorial_odd` | [`double_factorial_odd`] |
//! | `factorial` | [`factorial`] |
//! | `binomial` | [`binomial`] |
//! | `align_up` | nothing: `usize::next_multiple_of` |
//! | `idivup` | nothing: `usize::div_ceil` |
//! | `sqrt`, `exp`, `log`, `pow`, ..., `isnan`, `fma`, the element-wise rest | nothing: `f64` / `f32` methods |
//! | `rsqrt` | not ported yet: nothing above calls it |
//!
//! There is no `math` module: the C++ namespace exists only because
//! `free_breakfast::log` is the logging facade's and a function cannot share
//! its name. Here the crate is the namespace.
//!
//! **The element-wise functions have no counterpart.** They exist so one
//! body compiles for host and device, and to refuse an `int` argument the
//! C library would promote. The host half is `f64::sqrt` and its siblings,
//! which take exactly their own type; the device half stays in the kernels.
//!
//! **`idivup` and `align_up` have no counterpart.** The standard library has
//! both, on every unsigned integer type, keeping the argument's type as the
//! C++ templates do: `a.div_ceil(b)` and `a.next_multiple_of(b)`. They panic
//! on `b == 0`, where the C++ is undefined.
//!
//! The three functions here are `const fn`, as the C++ ones are
//! `constexpr`, and run the same loops, so each result is the same double.

#![forbid(unsafe_code)]

/// `n!!` over odd `n`: `n * (n - 2) * ... * 1`; the empty product, 1.0, for
/// `n <= 1`: `double_factorial_odd`.
///
/// The `(2l - 1)!!` of a Gaussian primitive's normalization and of the
/// per-shell spherical scale `sqrt((2l - 1)!!)`. Exact while the product
/// fits a double's 53-bit significand, far beyond the angular momenta a
/// basis set reaches.
#[must_use]
pub const fn double_factorial_odd(n: i32) -> f64 {
    let mut result = 1.0;
    let mut k = n;
    while k > 1 {
        result *= k as f64;
        k -= 2;
    }
    result
}

/// `n!` as a double; 1.0 for `n <= 1`: `factorial`.
#[must_use]
pub const fn factorial(n: i32) -> f64 {
    let mut result = 1.0;
    let mut k = 2;
    while k <= n {
        result *= k as f64;
        k += 1;
    }
    result
}

/// The binomial coefficient `C(n, k)` as a double; zero outside
/// `0 <= k <= n`: `binomial`.
///
/// Multiplies then divides at each step, in the C++ order, so every
/// intermediate is an exact integer and the result is bitwise the C++ one.
#[must_use]
pub const fn binomial(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    let mut result = 1.0;
    let mut i = 0;
    while i < k {
        result = result * (n - i) as f64 / (i + 1) as f64;
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{binomial, double_factorial_odd, factorial};

    // `core_vocabulary_host.cpp`'s static_asserts, still checked at compile
    // time: the functions are `const fn`.
    const _: () = {
        assert!(double_factorial_odd(-1) == 1.0);
        assert!(double_factorial_odd(1) == 1.0);
        assert!(double_factorial_odd(7) == 105.0);
        assert!(factorial(0) == 1.0);
        assert!(factorial(5) == 120.0);
        assert!(binomial(5, 2) == 10.0);
        assert!(binomial(4, 0) == 1.0);
        assert!(binomial(4, 5) == 0.0);
        assert!(binomial(4, -1) == 0.0);
    };

    /// `CorePrimitives.CombinatoricsAgree`: `(2n - 1)!! = (2n)! / (2^n n!)`,
    /// and the binomial row sums to `2^n`, exactly.
    #[test]
    fn combinatorics_agree() {
        for n in 0..=10 {
            let two_n = f64::from(1u32 << n);
            assert_eq!(
                double_factorial_odd(2 * n - 1),
                factorial(2 * n) / (two_n * factorial(n)),
                "n = {n}"
            );
            let row: f64 = (0..=n).map(|k| binomial(n, k)).sum();
            assert_eq!(row, two_n, "n = {n}");
        }
    }

    /// `CorePrimitives.IntegerRoundingCoversEveryRemainder`, over what
    /// replaces `idivup` and `align_up`: `div_ceil` is the smallest count of
    /// `b`-chunks covering `a`, and `next_multiple_of` is that many `b`s. The
    /// C++ static_asserts' cases are the first four.
    #[test]
    fn integer_rounding_covers_every_remainder() {
        assert_eq!(0usize.div_ceil(4), 0);
        assert_eq!(1usize.div_ceil(4), 1);
        assert_eq!(8usize.div_ceil(4), 2);
        assert_eq!(9usize.div_ceil(4), 3);
        assert_eq!(0usize.next_multiple_of(4), 0);
        assert_eq!(9usize.next_multiple_of(4), 12);
        assert_eq!(12usize.next_multiple_of(4), 12);
        for a in 0u32..=100 {
            for b in 1u32..=9 {
                let q = a.div_ceil(b);
                let r = a.next_multiple_of(b);
                assert!(q * b >= a, "{a} / {b}");
                assert!(q * b - a < b, "{a} / {b}");
                assert_eq!(r, q * b, "{a} / {b}");
            }
        }
    }
}
