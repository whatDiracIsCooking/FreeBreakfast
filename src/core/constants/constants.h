/// @file
/// @brief Physical constants and SI prefix multipliers (header for global-module-fragment
/// inclusion)

#pragma once

namespace free_breakfast {

// Numeric literals
//
// Typed 0/1/2/-1 for use where a scalar must be passed by address -- cuBLAS and
// cuSOLVER take alpha/beta through a host pointer, so a named constant replaces
// the per-call `const T one{1}` local while keeping the call site readable.

/// @brief Typed zero (0)
template <typename T> constexpr T zero = static_cast<T>(0);

/// @brief Typed one (1)
template <typename T> constexpr T one = static_cast<T>(1);

/// @brief Typed two (2)
template <typename T> constexpr T two = static_cast<T>(2);

/// @brief Typed negative one (-1)
template <typename T> constexpr T neg_one = static_cast<T>(-1);

// Physical Constants

/// @brief Boltzmann constant k_B in J/K (1.380649e-23)
template <typename T> constexpr T boltzmann = static_cast<T>(1.380649e-23);

/// @brief Vacuum permittivity epsilon_0 in F/m (8.8541878188e-12)
template <typename T> constexpr T epsilon_0 = static_cast<T>(8.8541878188e-12);

/// @brief Elementary charge e in C (1.602176634e-19)
template <typename T> constexpr T elementary_charge = static_cast<T>(1.602176634e-19);

/// @brief Avogadro constant N_A in mol^-1 (6.02214076e+23)
template <typename T> constexpr T avogadro = static_cast<T>(6.02214076e+23);

/// @brief Faraday constant F in C/mol (96485.33212)
template <typename T> constexpr T faraday = static_cast<T>(96485.33212331001);

/// @brief Molar gas constant R in J/(mol*K) (8.31446261815324)
template <typename T> constexpr T gas_constant = static_cast<T>(8.31446261815324);

/// @brief Mathematical constant pi (3.141592653589793)
template <typename T> constexpr T pi = static_cast<T>(3.141592653589793);

// Length conversions (atomic units)

/// @brief Bohr radius a_0 in Angstrom (CODATA 0.529177210903)
///
/// One bohr expressed in Angstrom: multiply a length in bohr by this to get
/// Angstrom, divide to go the other way.
template <typename T> constexpr T angstrom_per_bohr = static_cast<T>(0.529177210903);

/// @brief Bohr per Angstrom, the reciprocal of @c angstrom_per_bohr
/// (1.8897259886...); multiply an Angstrom length by this to get bohr.
template <typename T> constexpr T bohr_per_angstrom = static_cast<T>(1.0 / 0.529177210903);

// SI Unit Prefixes

/// @brief SI prefix giga (1e9)
template <typename T> constexpr T giga = static_cast<T>(1e9);

/// @brief SI prefix mega (1e6)
template <typename T> constexpr T mega = static_cast<T>(1e6);

/// @brief SI prefix kilo (1e3)
template <typename T> constexpr T kilo = static_cast<T>(1e3);

/// @brief SI prefix milli (1e-3)
template <typename T> constexpr T milli = static_cast<T>(1e-3);

/// @brief SI prefix micro (1e-6)
template <typename T> constexpr T micro = static_cast<T>(1e-6);

/// @brief SI prefix nano (1e-9)
template <typename T> constexpr T nano = static_cast<T>(1e-9);

/// @brief SI prefix pico (1e-12)
template <typename T> constexpr T pico = static_cast<T>(1e-12);

/// @brief SI prefix femto (1e-15)
template <typename T> constexpr T femto = static_cast<T>(1e-15);

} // namespace free_breakfast
