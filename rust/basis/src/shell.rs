//! `basis.h`, `angular_momentum.h` and `:shell`: the counts, the primitive,
//! the ceiling and one centered shell.

use fb_core_error_handling::{Error, check_or_abort};
use fb_core_types::Double3;

/// The largest angular momentum the GPU kernels support, f:
/// `kMaxAngularMomentum`.
///
/// The kernels cover `l` in `0 ..= MAX_ANGULAR_MOMENTUM`, and each engine
/// refuses a basis with a higher shell. [`Shell`] imposes no such cap, so this
/// is the kernels' limit, not a basis-wide invariant.
pub const MAX_ANGULAR_MOMENTUM: i32 = 3;

/// The number of Cartesian Gaussians in a shell of momentum `l`,
/// `(l+1)(l+2)/2`: 1 (s), 3 (p), 6 (d), 10 (f): `function_count_cart`.
///
/// Where a signed angular momentum becomes an unsigned count. `l` must not
/// be negative; [`Shell`] and [`crate::ShellStamp`] abort on one.
#[must_use]
pub const fn function_count_cart(l: i32) -> u32 {
    let n = l as u32;
    ((n + 1) * (n + 2)) / 2
}

/// The number of real solid harmonics in a shell of momentum `l`, `2l+1`,
/// the count a shell presents to a basis: `function_count_sphe`.
#[must_use]
pub const fn function_count_sphe(l: i32) -> u32 {
    let n = l as u32;
    (2 * n) + 1
}

/// One primitive Gaussian in a contraction: its exponent, in inverse bohr²,
/// and the coefficient that multiplies it, stored as supplied: `Primitive`.
///
/// It belongs to one shell and shares that shell's momentum and center, so
/// neither is repeated here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Primitive {
    /// The exponent, in inverse bohr².
    pub exponent: f64,
    /// The contraction coefficient, with no normalization applied.
    pub coefficient: f64,
}

/// One contracted Gaussian shell: an angular momentum, its primitives and a
/// center in bohr: `Shell`.
///
/// **Coefficients are stored as supplied.** A caller that wants normalized
/// primitives bakes them in with [`crate::BasisStamp::primitive_normalized`].
#[derive(Clone, Debug, PartialEq)]
pub struct Shell {
    l: i32,
    primitives: Vec<Primitive>,
    center: Double3,
}

impl Shell {
    /// A shell of momentum `l` (0 = s, ..., 3 = f; no cap) over
    /// `primitives`, in contraction order, at `center`, in bohr.
    ///
    /// A negative `l` or no primitives aborts at the caller with a critical
    /// line, as the C++ constructor does: the one production caller, the
    /// basis-set reader, screens both first.
    #[track_caller]
    pub fn new(l: i32, primitives: Vec<Primitive>, center: Double3) -> Shell {
        Shell {
            l,
            primitives: checked("Shell", l, primitives),
            center,
        }
    }

    /// [`Shell::new`] over two parallel lists, as basis-set files give them.
    /// Lists of different lengths abort too.
    #[track_caller]
    pub fn zipped(l: i32, exponents: &[f64], coefficients: &[f64], center: Double3) -> Shell {
        Shell::new(l, zip("Shell", exponents, coefficients), center)
    }

    /// The angular momentum `l`.
    pub fn angular_momentum(&self) -> i32 {
        self.l
    }

    /// The primitives, in contraction order.
    pub fn primitives(&self) -> &[Primitive] {
        &self.primitives
    }

    /// The number of primitives.
    pub fn nprimitives(&self) -> u32 {
        self.primitives.len() as u32
    }

    /// The center, in bohr.
    pub fn center(&self) -> Double3 {
        self.center
    }

    /// The number of Cartesian functions the kernels evaluate,
    /// [`function_count_cart`] of its momentum.
    pub fn function_count_cart(&self) -> u32 {
        function_count_cart(self.l)
    }

    /// The number of real solid harmonics the shell presents, `2l+1`.
    pub fn function_count_sphe(&self) -> u32 {
        function_count_sphe(self.l)
    }
}

/// `primitives`, or an abort naming `what` if `l` is negative or there are
/// none: the shape contract [`Shell`] and [`crate::ShellStamp`] share.
#[track_caller]
pub(crate) fn checked(what: &str, l: i32, primitives: Vec<Primitive>) -> Vec<Primitive> {
    if l < 0 {
        check_or_abort::<()>(Err(Error::config(format!(
            "{what}: angular momentum must be non-negative"
        ))));
    }
    if primitives.is_empty() {
        check_or_abort::<()>(Err(Error::config(format!(
            "{what}: a shell needs at least one primitive"
        ))));
    }
    primitives
}

/// Two equal-length lists paired into primitives, or an abort naming
/// `what`: the one place the two lengths must agree.
#[track_caller]
pub(crate) fn zip(what: &str, exponents: &[f64], coefficients: &[f64]) -> Vec<Primitive> {
    if exponents.len() != coefficients.len() {
        check_or_abort::<()>(Err(Error::config(format!(
            "{what}: exponent and coefficient counts must match"
        ))));
    }
    exponents
        .iter()
        .zip(coefficients)
        .map(|(&exponent, &coefficient)| Primitive {
            exponent,
            coefficient,
        })
        .collect()
}

/// `test/basis/shell_host.cpp` and `ao_host.cpp`'s counts, each named after
/// its C++ case. The abort case is `tests/basis_abort.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    /// `BasisShell.ShellZipsParallelArrays`.
    #[test]
    fn shell_zips_parallel_arrays() {
        let shell = Shell::zipped(
            2,
            &[3.0, 0.5],
            &[0.25, 0.75],
            Double3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
        );
        assert_eq!(shell.angular_momentum(), 2);
        assert_eq!(shell.nprimitives(), 2);
        assert_eq!(shell.primitives()[1].exponent, 0.5);
        assert_eq!(shell.primitives()[1].coefficient, 0.75);
        assert_eq!(shell.center().z, 3.0);
        assert_eq!(shell.function_count_cart(), 6); // Cartesian, what the kernels evaluate
        assert_eq!(shell.function_count_sphe(), 5); // 2l+1, what the shell means
    }

    /// `BasisAo.FunctionCounts`.
    #[test]
    fn function_counts() {
        assert_eq!(MAX_ANGULAR_MOMENTUM, 3);
        let cart = [1, 3, 6, 10, 15];
        for l in 0..5 {
            assert_eq!(function_count_cart(l), cart[l as usize], "l = {l}");
            assert_eq!(function_count_sphe(l), (2 * l + 1) as u32, "l = {l}");
        }
    }

    /// No C++ counterpart: the ceiling is the value the C++ header
    /// `src/basis/angular_momentum.h` defines, read from the header itself,
    /// so the two cannot drift apart unnoticed.
    #[test]
    fn max_angular_momentum_is_the_cpp_header_value() {
        let header = include_str!("../../../src/basis/angular_momentum.h");
        let needle = "inline constexpr int32_t kMaxAngularMomentum = ";
        let start = header.find(needle).expect("the definition") + needle.len();
        let value: i32 = header[start..]
            .split(';')
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(MAX_ANGULAR_MOMENTUM, value);
    }
}
