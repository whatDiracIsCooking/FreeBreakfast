//! `:shell_stamp`: the geometry-free shell, a [`Shell`] with the center
//! removed.

use std::f64::consts::PI;

use fb_core_primitives::double_factorial_odd;
use fb_core_types::Double3;

use crate::shell::{checked, zip};
use crate::{Primitive, Shell, function_count_cart, function_count_sphe};

/// One contracted Gaussian shell without a center, the part of a [`Shell`] a
/// basis-set file gives per element rather than per atom: `ShellStamp`.
///
/// [`ShellStamp::place`] supplies a center; [`ShellStamp::primitive_normalized`]
/// folds in the per-primitive normalization, which depends on no center, so
/// it is done once per stamp rather than once per geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellStamp {
    l: i32,
    primitives: Vec<Primitive>,
}

impl ShellStamp {
    /// A stamp of momentum `l` (no cap) over `primitives`, in contraction
    /// order. A negative `l` or no primitives aborts at the caller, the same
    /// shape contract as [`Shell::new`].
    #[track_caller]
    pub fn new(l: i32, primitives: Vec<Primitive>) -> ShellStamp {
        ShellStamp {
            l,
            primitives: checked("ShellStamp", l, primitives),
        }
    }

    /// [`ShellStamp::new`] over two parallel lists, as basis-set files give
    /// them. Lists of different lengths abort too.
    #[track_caller]
    pub fn zipped(l: i32, exponents: &[f64], coefficients: &[f64]) -> ShellStamp {
        ShellStamp::new(l, zip("ShellStamp", exponents, coefficients))
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

    /// The number of Cartesian functions, [`function_count_cart`] of its
    /// momentum.
    pub fn function_count_cart(&self) -> u32 {
        function_count_cart(self.l)
    }

    /// The number of real solid harmonics, `2l+1`.
    pub fn function_count_sphe(&self) -> u32 {
        function_count_sphe(self.l)
    }

    /// A copy with each primitive's normalization folded into its
    /// coefficient: `primitive_normalized`.
    ///
    /// Basis-set coefficients are given for normalized primitives, and the
    /// engines take coefficients as given. The scale is the `(l,0,0)`
    /// primitive's, `N(α, l) = (2α/π)^{3/4} (4α)^{l/2} / sqrt((2l-1)!!)`. The
    /// contraction is not renormalized.
    #[must_use]
    pub fn primitive_normalized(&self) -> ShellStamp {
        let dfac = double_factorial_odd(2 * self.l - 1);
        let primitives = self
            .primitives
            .iter()
            .map(|prim| {
                let a = prim.exponent;
                let norm = (2.0 * a / PI).powf(0.75) * (4.0 * a).powf(f64::from(self.l) / 2.0)
                    / dfac.sqrt();
                Primitive {
                    exponent: a,
                    coefficient: prim.coefficient * norm,
                }
            })
            .collect();
        ShellStamp {
            l: self.l,
            primitives,
        }
    }

    /// This stamp at `center`, in bohr: a [`Shell`] with the same momentum
    /// and primitives: `place(stamp, center)`.
    pub fn place(&self, center: Double3) -> Shell {
        Shell::new(self.l, self.primitives.clone(), center)
    }
}

/// `test/basis/shell_stamp_host.cpp`, one test per case, each named after
/// it. The abort case is `tests/basis_abort.rs`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moments::double_factorial;

    /// `BasisShellStamp.ZipsParallelArrays`.
    #[test]
    fn zips_parallel_arrays() {
        let stamp = ShellStamp::zipped(1, &[3.0, 0.5], &[0.25, 0.75]);
        assert_eq!(stamp.angular_momentum(), 1);
        assert_eq!(stamp.nprimitives(), 2);
        assert_eq!(stamp.primitives()[0].exponent, 3.0);
        assert_eq!(stamp.primitives()[1].coefficient, 0.75);
        assert_eq!(stamp.function_count_cart(), 3);
        assert_eq!(stamp.function_count_sphe(), 3);
    }

    /// `BasisShellStamp.NormalizesEachPrimitive`, at its 1e-12.
    #[test]
    fn normalizes_each_primitive() {
        for l in 0..=3 {
            let stamp = ShellStamp::zipped(l, &[2.0, 0.3], &[0.4, 0.6]).primitive_normalized();
            for (i, alpha, c) in [(0, 2.0, 0.4), (1, 0.3, 0.6)] {
                let norm = (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).powf(f64::from(l) / 2.0)
                    / double_factorial(2 * l - 1).sqrt();
                let got = stamp.primitives()[i].coefficient;
                assert!((got - c * norm).abs() <= 1e-12, "l = {l}: {got}");
                assert_eq!(stamp.primitives()[i].exponent, alpha);
            }
        }
    }

    /// `BasisShellStamp.PlaceCopiesMomentumAndPrimitivesAtTheCenter`.
    #[test]
    fn place_copies_momentum_and_primitives_at_the_center() {
        let stamp = ShellStamp::zipped(2, &[1.5, 0.2], &[0.3, 0.7]);
        let shell = stamp.place(Double3 {
            x: 0.5,
            y: -1.0,
            z: 2.5,
        });
        assert_eq!(shell.angular_momentum(), 2);
        assert_eq!(shell.primitives(), stamp.primitives());
        assert_eq!(shell.center().x, 0.5);
        assert_eq!(shell.center().y, -1.0);
        assert_eq!(shell.center().z, 2.5);
    }
}
