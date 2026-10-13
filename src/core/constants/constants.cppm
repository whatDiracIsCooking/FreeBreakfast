/// @file
/// @brief Physical constants and SI prefix multipliers

module;

#include "constants.h"

export module free_breakfast.core.constants;

namespace free_breakfast {

/// @brief Typed zero (0)
export using free_breakfast::zero;
/// @brief Typed one (1)
export using free_breakfast::one;
/// @brief Typed two (2)
export using free_breakfast::two;
/// @brief Typed negative one (-1)
export using free_breakfast::neg_one;

/// @brief Boltzmann constant k_B in J/K (1.380649e-23)
export using free_breakfast::boltzmann;
/// @brief Vacuum permittivity epsilon_0 in F/m (8.8541878188e-12)
export using free_breakfast::epsilon_0;
/// @brief Elementary charge e in C (1.602176634e-19)
export using free_breakfast::elementary_charge;
/// @brief Avogadro constant N_A in mol^-1 (6.02214076e+23)
export using free_breakfast::avogadro;
/// @brief Faraday constant F in C/mol (96485.33212)
export using free_breakfast::faraday;
/// @brief Molar gas constant R in J/(mol*K) (8.31446261815324)
export using free_breakfast::gas_constant;
/// @brief Mathematical constant pi (3.141592653589793)
export using free_breakfast::pi;

/// @brief Bohr radius a_0 in Angstrom (CODATA 0.529177210903)
export using free_breakfast::angstrom_per_bohr;
/// @brief Bohr per Angstrom, the reciprocal (1.8897259886...)
export using free_breakfast::bohr_per_angstrom;

/// @brief SI prefix giga (1e9)
export using free_breakfast::giga;
/// @brief SI prefix mega (1e6)
export using free_breakfast::mega;
/// @brief SI prefix kilo (1e3)
export using free_breakfast::kilo;
/// @brief SI prefix milli (1e-3)
export using free_breakfast::milli;
/// @brief SI prefix micro (1e-6)
export using free_breakfast::micro;
/// @brief SI prefix nano (1e-9)
export using free_breakfast::nano;
/// @brief SI prefix pico (1e-12)
export using free_breakfast::pico;
/// @brief SI prefix femto (1e-15)
export using free_breakfast::femto;

} // namespace free_breakfast
