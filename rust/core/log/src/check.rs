//! The value checks: [`le!`](crate::le), [`lt!`](crate::lt),
//! [`eq!`](crate::eq) and [`fits_i32!`](crate::fits_i32), comparisons a
//! contract's condition is written with so that a tripped contract's panic
//! carries its operands' values (`docs/testing.md`, "Contracts are part of
//! the spec").
//!
//! Each is a `bool`, true when the comparison holds. When it does not, it
//! stores each operand's source text and `Debug` value in a slot of this
//! thread's, then is false; the contract that wrote it panics, and the panic
//! hook takes the slot (clearing it) and ends its one line with them:
//!
//! ```text
//! Pre-condition of inverse_sqrt violated: count is within data: le! (count, data.len()) [count = 5, data.len() = 4]
//! ```
//!
//! (The `contracts` crate writes the condition as its tokens print, so the
//! macro's name is followed by a space.)
//!
//! A false check that did not end in a panic would leave its values for this
//! thread's next one, which is why a check is written only as a whole `&&`
//! term of a contract's condition: never negated, never under `||`, never
//! anywhere else (`doc_check/tests/contracts.rs`).
//!
//! The macros are exported at the crate root, `fb_core_log::le`, and
//! imported from there. What they expand to is public only so that they can
//! reach it, and hidden; it is not for calling.

use std::cell::RefCell;
use std::fmt::Debug;

thread_local! {
    /// The last failed check's operands, `text = value, ...`, until the
    /// panic hook takes them.
    static FAILED: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Store `operands`, each its source text and value, as this thread's
/// failed check, and return false: the tail of every check that does not
/// hold.
#[doc(hidden)]
pub fn failed(operands: &[(&str, &dyn Debug)]) -> bool {
    let values = operands
        .iter()
        .map(|(text, value)| format!("{text} = {value:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    // A thread being torn down has no slot left; its panic loses the values.
    let _ = FAILED.try_with(|f| f.replace(Some(values)));
    false
}

/// This thread's failed check's operands, clearing the slot.
pub(crate) fn take() -> Option<String> {
    FAILED.try_with(RefCell::take).ok().flatten()
}

/// An integer, or an `Option` of one (a `checked_mul`'s), that may fit an
/// `i32`: what [`fits_i32!`](crate::fits_i32) asks of its operand.
#[doc(hidden)]
pub trait FitsI32 {
    /// Whether the value is in `i32::MIN..=i32::MAX`; `None` never is.
    fn fits_i32(&self) -> bool;
}

/// [`FitsI32`] for each integer type, by `i32::try_from`.
macro_rules! fits_i32_by_try_from {
    ($($t:ty),+) => {
        $(impl FitsI32 for $t {
            fn fits_i32(&self) -> bool {
                i32::try_from(*self).is_ok()
            }
        })+
    };
}

fits_i32_by_try_from!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

impl<T: FitsI32> FitsI32 for Option<T> {
    fn fits_i32(&self) -> bool {
        self.as_ref().is_some_and(T::fits_i32)
    }
}

/// `a <= b`, its operands stored for the panic hook when false
/// ([`check`](crate::check)). A `>=` is `le!(b, a)`. Each operand is
/// evaluated once.
///
/// ```
/// use contracts::requires;
/// use fb_core_log::le;
///
/// #[requires(le!(count, data.len()), "count is within data")]
/// fn head(count: usize, data: &[f64]) -> &[f64] {
///     &data[..count]
/// }
///
/// assert_eq!(head(2, &[1.0, 2.0, 3.0]), [1.0, 2.0]);
/// ```
#[macro_export]
macro_rules! le {
    ($a:expr, $b:expr $(,)?) => {
        match (&$a, &$b) {
            (a, b) => {
                *a <= *b
                    || $crate::check::failed(&[
                        (::std::stringify!($a), a as &dyn ::std::fmt::Debug),
                        (::std::stringify!($b), b as &dyn ::std::fmt::Debug),
                    ])
            }
        }
    };
}

/// `a < b`, its operands stored for the panic hook when false
/// ([`check`](crate::check)). A `>` is `lt!(b, a)`. Each operand is
/// evaluated once.
///
/// ```should_panic
/// use contracts::requires;
/// use fb_core_log::lt;
///
/// #[requires(lt!(i, data.len()), "i indexes data")]
/// fn at(i: usize, data: &[f64]) -> f64 {
///     data[i]
/// }
///
/// fb_core_log::init();
/// // The hook's line ends "i indexes data: lt! (i, data.len()) [i = 3, data.len() = 3]".
/// at(3, &[1.0, 2.0, 3.0]);
/// ```
#[macro_export]
macro_rules! lt {
    ($a:expr, $b:expr $(,)?) => {
        match (&$a, &$b) {
            (a, b) => {
                *a < *b
                    || $crate::check::failed(&[
                        (::std::stringify!($a), a as &dyn ::std::fmt::Debug),
                        (::std::stringify!($b), b as &dyn ::std::fmt::Debug),
                    ])
            }
        }
    };
}

/// `a == b`, its operands stored for the panic hook when false
/// ([`check`](crate::check)). Each operand is evaluated once.
///
/// ```
/// use contracts::requires;
/// use fb_core_log::eq;
///
/// #[requires(eq!(dst.len(), src.len()), "the two slices are one length")]
/// fn copy(dst: &mut [f64], src: &[f64]) {
///     dst.copy_from_slice(src);
/// }
///
/// let mut dst = [0.0; 2];
/// copy(&mut dst, &[1.0, 2.0]);
/// assert_eq!(dst, [1.0, 2.0]);
/// ```
#[macro_export]
macro_rules! eq {
    ($a:expr, $b:expr $(,)?) => {
        match (&$a, &$b) {
            (a, b) => {
                *a == *b
                    || $crate::check::failed(&[
                        (::std::stringify!($a), a as &dyn ::std::fmt::Debug),
                        (::std::stringify!($b), b as &dyn ::std::fmt::Debug),
                    ])
            }
        }
    };
}

/// Whether an integer, or an `Option` of one, fits an `i32`, its operand
/// stored for the panic hook when it does not ([`check`](crate::check)).
/// `None` does not fit, so a recomputed extent is written with checked
/// arithmetic, `fits_i32!(n.checked_mul(n))`, and never wraps to a value
/// that does. Evaluated once.
///
/// ```
/// use contracts::requires;
/// use fb_core_log::fits_i32;
///
/// #[requires(fits_i32!(n.checked_mul(n)), "an n x n matrix fits a 32-bit index")]
/// fn square(n: usize) -> usize {
///     n * n
/// }
///
/// assert_eq!(square(46_340), 2_147_395_600);
/// ```
#[macro_export]
macro_rules! fits_i32 {
    ($x:expr $(,)?) => {
        match &$x {
            x => {
                $crate::check::FitsI32::fits_i32(x)
                    || $crate::check::failed(&[(
                        ::std::stringify!($x),
                        x as &dyn ::std::fmt::Debug,
                    )])
            }
        }
    };
}
