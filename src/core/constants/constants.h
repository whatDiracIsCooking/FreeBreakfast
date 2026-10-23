/// @file
/// @brief Physical constants and SI prefix multipliers (header for global-module-fragment
/// inclusion)

#pragma once

// wwrFloatComplex / wwrDoubleComplex, for the numeric literals' complex
// specializations. Reached through wwr's neutral src/-root header, which picks
// cuComplex.h or hip_complex.h; never include either vendor header here.
#include "complex.h"

namespace free_breakfast {

// Numeric literals
//
// Typed 0/1/2/-1 for use where a scalar must be passed by address -- cuBLAS and
// cuSOLVER take alpha/beta through a host pointer, so a named constant replaces
// the per-call `const T one{1}` local while keeping the call site readable.
//
// Each also holds for the two GPU complex types, as value + 0i. Those need the
// explicit specializations after each primary: a complex type is a vector type,
// not an arithmetic one, so static_cast<T>(1) does not compile on CUDA (a plain
// float2 struct) and on HIP would call HIP_vector_type's one-scalar constructor,
// which broadcasts -- 1 + 1i. Brace-init {re, im} is constexpr on both. The
// specializations are `inline`: unlike the template, a const non-template
// variable has internal linkage, which an importer of the module cannot reach.

/// @brief Typed zero (0)
template <typename T>
constexpr T kZero = static_cast<T>(0);
template <>
inline constexpr wwr::wwrFloatComplex kZero<wwr::wwrFloatComplex>{0, 0};
template <>
inline constexpr wwr::wwrDoubleComplex kZero<wwr::wwrDoubleComplex>{0, 0};

/// @brief Typed one (1)
template <typename T>
constexpr T kOne = static_cast<T>(1);
template <>
inline constexpr wwr::wwrFloatComplex kOne<wwr::wwrFloatComplex>{1, 0};
template <>
inline constexpr wwr::wwrDoubleComplex kOne<wwr::wwrDoubleComplex>{1, 0};

/// @brief Typed two (2)
template <typename T>
constexpr T kTwo = static_cast<T>(2);
template <>
inline constexpr wwr::wwrFloatComplex kTwo<wwr::wwrFloatComplex>{2, 0};
template <>
inline constexpr wwr::wwrDoubleComplex kTwo<wwr::wwrDoubleComplex>{2, 0};

/// @brief Typed negative one (-1)
template <typename T>
constexpr T kNegOne = static_cast<T>(-1);
template <>
inline constexpr wwr::wwrFloatComplex kNegOne<wwr::wwrFloatComplex>{-1, 0};
template <>
inline constexpr wwr::wwrDoubleComplex kNegOne<wwr::wwrDoubleComplex>{-1, 0};

// Physical Constants

/// @brief Boltzmann constant k_B in J/K (1.380649e-23)
template <typename T>
constexpr T kBoltzmann = static_cast<T>(1.380649e-23);

/// @brief Vacuum permittivity epsilon_0 in F/m (8.8541878188e-12)
template <typename T>
constexpr T kEpsilon0 = static_cast<T>(8.8541878188e-12);

/// @brief Elementary charge e in C (1.602176634e-19)
template <typename T>
constexpr T kElementaryCharge = static_cast<T>(1.602176634e-19);

/// @brief Avogadro constant N_A in mol^-1 (6.02214076e+23)
template <typename T>
constexpr T kAvogadro = static_cast<T>(6.02214076e+23);

/// @brief Faraday constant F in C/mol (96485.33212)
template <typename T>
constexpr T kFaraday = static_cast<T>(96485.33212331001);

/// @brief Molar gas constant R in J/(mol*K) (8.31446261815324)
template <typename T>
constexpr T kGasConstant = static_cast<T>(8.31446261815324);

/// @brief Mathematical constant pi (3.141592653589793)
template <typename T>
constexpr T kPi = static_cast<T>(3.141592653589793);

// Length conversions (atomic units)

/// @brief Bohr radius a_0 in Angstrom (CODATA 0.529177210903)
///
/// One bohr expressed in Angstrom: multiply a length in bohr by this to get
/// Angstrom, divide to go the other way.
template <typename T>
constexpr T kAngstromPerBohr = static_cast<T>(0.529177210903);

/// @brief Bohr per Angstrom, the reciprocal of @c angstrom_per_bohr
/// (1.8897259886...); multiply an Angstrom length by this to get bohr.
template <typename T>
constexpr T kBohrPerAngstrom = static_cast<T>(1.0 / 0.529177210903);

// SI Unit Prefixes

/// @brief SI prefix giga (1e9)
template <typename T>
constexpr T kGiga = static_cast<T>(1e9);

/// @brief SI prefix mega (1e6)
template <typename T>
constexpr T kMega = static_cast<T>(1e6);

/// @brief SI prefix kilo (1e3)
template <typename T>
constexpr T kKilo = static_cast<T>(1e3);

/// @brief SI prefix milli (1e-3)
template <typename T>
constexpr T kMilli = static_cast<T>(1e-3);

/// @brief SI prefix micro (1e-6)
template <typename T>
constexpr T kMicro = static_cast<T>(1e-6);

/// @brief SI prefix nano (1e-9)
template <typename T>
constexpr T kNano = static_cast<T>(1e-9);

/// @brief SI prefix pico (1e-12)
template <typename T>
constexpr T kPico = static_cast<T>(1e-12);

/// @brief SI prefix femto (1e-15)
template <typename T>
constexpr T kFemto = static_cast<T>(1e-15);

} // namespace free_breakfast
