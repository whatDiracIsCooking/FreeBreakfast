/// @file
/// @brief Physical constants and SI prefix multipliers

module;

#include "constants.h"

export module free_breakfast.core.constants;

namespace free_breakfast {

/// @brief Typed zero (0)
export using free_breakfast::kZero;
/// @brief Typed one (1)
export using free_breakfast::kOne;
/// @brief Typed two (2)
export using free_breakfast::kTwo;
/// @brief Typed negative one (-1)
export using free_breakfast::kNegOne;

/// @brief Boltzmann constant k_B in J/K (1.380649e-23)
export using free_breakfast::kBoltzmann;
/// @brief Vacuum permittivity epsilon_0 in F/m (8.8541878188e-12)
export using free_breakfast::kEpsilon0;
/// @brief Elementary charge e in C (1.602176634e-19)
export using free_breakfast::kElementaryCharge;
/// @brief Avogadro constant N_A in mol^-1 (6.02214076e+23)
export using free_breakfast::kAvogadro;
/// @brief Faraday constant F in C/mol (96485.33212)
export using free_breakfast::kFaraday;
/// @brief Molar gas constant R in J/(mol*K) (8.31446261815324)
export using free_breakfast::kGasConstant;
/// @brief Mathematical constant pi (3.141592653589793)
export using free_breakfast::kPi;

/// @brief Bohr radius a_0 in Angstrom (CODATA 0.529177210903)
export using free_breakfast::kAngstromPerBohr;
/// @brief Bohr per Angstrom, the reciprocal (1.8897259886...)
export using free_breakfast::kBohrPerAngstrom;

/// @brief SI prefix giga (1e9)
export using free_breakfast::kGiga;
/// @brief SI prefix mega (1e6)
export using free_breakfast::kMega;
/// @brief SI prefix kilo (1e3)
export using free_breakfast::kKilo;
/// @brief SI prefix milli (1e-3)
export using free_breakfast::kMilli;
/// @brief SI prefix micro (1e-6)
export using free_breakfast::kMicro;
/// @brief SI prefix nano (1e-9)
export using free_breakfast::kNano;
/// @brief SI prefix pico (1e-12)
export using free_breakfast::kPico;
/// @brief SI prefix femto (1e-15)
export using free_breakfast::kFemto;

} // namespace free_breakfast
