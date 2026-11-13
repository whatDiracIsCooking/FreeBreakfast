//! `:hermite`: the one-center McMurchie–Davidson building blocks both
//! integral families share, the same-center Hermite expansion and the
//! zero-distance Hermite Coulomb integrals.

use fb_core_primitives::double_factorial_odd;

/// The Hermite expansion coefficients of `x^n exp(−p x²)` about its own
/// center, for every `n ≤ nmax`: `x^n exp(−p x²) = Σ_t e[n][t] (d/dx)^t
/// exp(−p x²)`: `one_center_hermite`.
///
/// With every McMurchie–Davidson distance zero, the two-center recurrence
/// reduces to `e[n+1][t] = e[n][t−1] / (2p) + (t+1) e[n][t+1]`,
/// `e[0][0] = 1`, and `e[n][t]` vanishes unless t and n have the same parity.
/// Each element is the C++ expression, grouped the same way.
pub(super) fn one_center_hermite(nmax: i32, p: f64) -> Vec<Vec<f64>> {
    let size = nmax as usize + 1;
    // One spare column, so e[n][t + 1] is in range at t = n + 1.
    let mut e = vec![vec![0.0; size + 1]; size];
    e[0][0] = 1.0;
    for n in 0..size - 1 {
        for t in 0..=n + 1 {
            let down = if t > 0 { e[n][t - 1] / (2.0 * p) } else { 0.0 };
            let up = (t + 1) as f64 * e[n][t + 1];
            e[n + 1][t] = down + up;
        }
    }
    e
}

/// `R_{tuv}(α)` at zero distance, the (t, u, v)-th derivative of
/// `F_0(α R²)` at `R = 0`: `one_center_hermite_coulomb`.
///
/// Only even t, u, v survive, and then it is
/// `(t−1)!! (u−1)!! (v−1)!! (−2α)^{N/2} / (N + 1)`, `N = t + u + v`, the
/// recurrence `R^n_{t+1} = t R^{n+1}_{t−1}` unrolled down to
/// `R^n_{000} = (−2α)^n F_n(0)` with `F_n(0) = 1 / (2n + 1)`. The power is
/// the C library's `pow` over a double exponent, as the C++ `std::pow(double,
/// int)` is, not `powi`, so the value is the C++ one bitwise.
pub(super) fn one_center_hermite_coulomb(t: i32, u: i32, v: i32, alpha: f64) -> f64 {
    if t % 2 != 0 || u % 2 != 0 || v % 2 != 0 {
        return 0.0;
    }
    let half = (t + u + v) / 2;
    double_factorial_odd(t - 1)
        * double_factorial_odd(u - 1)
        * double_factorial_odd(v - 1)
        * (-2.0 * alpha).powf(f64::from(half))
        / f64::from((2 * half) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `x² exp(−p x²) = 1/(2p) + 1/(4p²) (d/dx)² …`: the recurrence's first
    /// rows, by hand.
    #[test]
    fn expansion_matches_its_first_rows() {
        let p = 1.5;
        let e = one_center_hermite(2, p);
        assert_eq!(e[1][1], 1.0 / (2.0 * p));
        assert_eq!(e[1][0], 0.0);
        assert_eq!(e[2][0], 1.0 / (2.0 * p));
        assert_eq!(e[2][2], (1.0 / (2.0 * p)) / (2.0 * p));
    }

    /// `R_000 = F_0(0) = 1`, `R_200 = −2α/3`, and any odd index is zero.
    #[test]
    fn coulomb_at_zero_distance() {
        assert_eq!(one_center_hermite_coulomb(0, 0, 0, 0.7), 1.0);
        assert_eq!(one_center_hermite_coulomb(2, 0, 0, 0.75), -1.5 / 3.0);
        assert_eq!(one_center_hermite_coulomb(1, 0, 2, 0.7), 0.0);
    }
}
