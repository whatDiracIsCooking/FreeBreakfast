//! `src/core/constants` (module `free_breakfast.core.constants`): the
//! physical constants the host modules convert with.
//!
//! Only what `atom`, `molecule`, `basis`, `io/xyz` and `io/basis` name is
//! ported; the rest waits for the module that first needs it. Each ported
//! constant is the C++ literal, written the same way, so it is the same
//! double bitwise; the tests assert each against its bit pattern.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `kBohrPerAngstrom<double>` | [`BOHR_PER_ANGSTROM`] |
//! | `kBohrPerAngstrom<T>` for another `T` | not ported yet: only `double` is named |
//! | `kAngstromPerBohr`, the other physical constants, `kPi`, `kSqrt2`, `kInvSqrt2`, the SI prefixes | not ported yet: nothing above names them |
//! | `kOne`, `kZero` | `fb_core_wwr::blas::BlasScalar`'s `ONE` and `ZERO`: they exist for BLAS's scalars, which Rust passes by value |
//! | `kTwo` ... `kNegHalf`, `kImaginaryUnit` | not ported yet: they exist for BLAS's by-address scalars |
//!
//! **A constant is an `f64`, not a template.** The C++ variable templates
//! give each constant every precision through one `static_cast`; a Rust
//! `const` has one type, and an `f32` one is added beside it when a caller
//! first needs it.

#![forbid(unsafe_code)]

/// Bohr per Angstrom, the reciprocal of the Bohr radius in Angstrom (CODATA
/// 2018, 0.529177210903): `kBohrPerAngstrom<double>`. Multiply a length in
/// Angstrom by this to get bohr, as the XYZ reader does.
///
/// The C++ literal, `1.0 / 0.529177210903`, a division the compiler rounds
/// once in both languages.
pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529177210903;

#[cfg(test)]
mod tests {
    use super::BOHR_PER_ANGSTROM;

    /// The bit pattern of the C++ `kBohrPerAngstrom<double>`, printed by
    /// `std::bit_cast<std::uint64_t>` from a program over the same literal.
    #[test]
    fn bohr_per_angstrom_is_the_cpp_double() {
        assert_eq!(BOHR_PER_ANGSTROM.to_bits(), 0x3ffe_3c51_75fa_a1ce);
    }

    /// `CoreConstants.BohrConversionsAreReciprocal`, for the half ported:
    /// times the Bohr radius in Angstrom it is one, to `EXPECT_DOUBLE_EQ`'s
    /// four ulps.
    #[test]
    fn bohr_conversion_is_the_reciprocal() {
        let product = 0.529177210903 * BOHR_PER_ANGSTROM;
        assert!((product - 1.0).abs() <= 4.0 * f64::EPSILON, "{product}");
    }
}
