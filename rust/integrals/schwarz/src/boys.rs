//! The Boys function `F_n(T) = ∫₀¹ t²ⁿ exp(-T t²) dt`, every order
//! `0..=n_max` at one `T`, on the host.
//!
//! Below [`T_SWITCH`], the series for the top order (Helgaker, Jørgensen
//! and Olsen, "Molecular Electronic-Structure Theory", Eq. 9.8.12)
//!
//! ```text
//! F_n(T) = exp(-T) Σ_k (2T)^k / ((2n+1)(2n+3)···(2n+2k+1))
//! ```
//!
//! whose terms are all positive, so its rounding is about one ulp per
//! term, then the downward recursion (Eq. 9.8.13)
//! `F_{n-1} = (2T F_n + exp(-T)) / (2n-1)`, a sum of positive terms too.
//! From [`T_SWITCH`] up, `F_0 = √(π/T) / 2` (erf(√T) is 1 there to below
//! 1e-18) and the upward recursion `F_n = ((2n-1) F_{n-1} - exp(-T)) / (2T)`,
//! whose subtraction loses nothing there: `exp(-T)` is under 1e-17, and
//! `F_{n-1}` at least `1e-12` for `n <= 12`.

/// Where the series hands over to the asymptotic form.
const T_SWITCH: f64 = 40.0;

/// `F_n(t)` for `n` in `0..out.len()`, into `out`; `t >= 0`.
pub(crate) fn boys(t: f64, out: &mut [f64]) {
    let Some(n_max) = out.len().checked_sub(1) else {
        return;
    };
    let e = (-t).exp();
    if t < T_SWITCH {
        let mut denom = (2 * n_max + 1) as f64;
        let mut term = 1.0 / denom;
        let mut sum = term;
        // The terms grow while 2T > 2n+2k+1, then fall geometrically; stop
        // once one no longer moves the sum.
        loop {
            denom += 2.0;
            term *= 2.0 * t / denom;
            sum += term;
            if term < sum * 1e-17 {
                break;
            }
        }
        out[n_max] = e * sum;
        for n in (1..=n_max).rev() {
            out[n - 1] = (2.0 * t * out[n] + e) / (2 * n - 1) as f64;
        }
    } else {
        out[0] = 0.5 * (std::f64::consts::PI / t).sqrt();
        for n in 1..=n_max {
            out[n] = ((2 * n - 1) as f64 * out[n - 1] - e) / (2.0 * t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two branches meet at the switch: catches a wrong recursion or
    /// a wrong asymptote in either, which the libcint comparison would
    /// only show as a quartet somewhere off. Through order 12, at 1e-13.
    #[test]
    fn series_and_asymptote_agree_at_the_switch() {
        let mut below = [0.0; 13];
        let mut above = [0.0; 13];
        boys(T_SWITCH * (1.0 - 1e-15), &mut below);
        boys(T_SWITCH, &mut above);
        for (n, (b, a)) in below.iter().zip(&above).enumerate() {
            assert!(
                (b - a).abs() <= 1e-13 * a,
                "F_{n}: {b:e} below, {a:e} above"
            );
        }
    }

    /// `F_n(0) = 1 / (2n+1)` exactly: catches the series' first term or
    /// the downward recursion's divisor.
    #[test]
    fn at_zero_is_one_over_2n_plus_1() {
        let mut f = [0.0; 13];
        boys(0.0, &mut f);
        for (n, v) in f.iter().enumerate() {
            assert!(
                (v - 1.0 / (2 * n + 1) as f64).abs() <= 1e-16,
                "F_{n}(0) = {v}"
            );
        }
    }
}
