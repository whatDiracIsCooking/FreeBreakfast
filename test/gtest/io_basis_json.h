/// @file
/// @brief The BSE JSON both io/basis test files read, io_basis_host.cpp and
/// io_basis_gpu.cpp. Include it after `import std;`.

#pragma once

namespace free_breakfast::test {

// STO-3G hydrogen; oxygen with a plain s, an sp block (one column per
// momentum), a two-column general contraction, and a tagged spherical d.
inline constexpr std::string_view kBasis = R"({
  "elements": {
    "1": {"electron_shells": [
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["3.42525091", "0.62391373", "0.16885540"],
       "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]}
    ]},
    "8": {"electron_shells": [
      {"angular_momentum": [0],
       "exponents": ["130.7093200"], "coefficients": [["1.0"]]},
      {"function_type": "gto", "angular_momentum": [0, 1],
       "exponents": ["5.0331513", "1.1695961"],
       "coefficients": [["-0.09996723", "0.39951283"], ["0.15591627", "0.60768372"]]},
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["0.38", "0.12"],
       "coefficients": [["1.0", "0.0"], ["0.0", "1.0"]]},
      {"function_type": "gto_spherical", "angular_momentum": [2],
       "exponents": ["1.185"], "coefficients": [["1.0E+00"]]}
    ]}
  }
})";

} // namespace free_breakfast::test
