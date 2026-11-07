/// @file
/// @brief A NEVPT2 refusal message with its computed numbers masked, for the
/// cases that compare two entry points' refusals of one state (#755, #1010).
///
/// Include it AFTER `import std;`, like scratch_dir.h: it includes nothing
/// itself. nevpt2_energy_gpu.cpp and nevpt2_gradient_gpu.cpp compare with
/// it; stable_part_host.cpp holds it to what it masks.

#pragma once

namespace free_breakfast::test {

/// @brief @p message with each computed number ({:.3e}, a gap or a
/// denominator) masked as `#`. Two entry points that refuse the same state
/// agree on the words, the states named and the floor, but the number is
/// round-off and need not: an intruder's message carries the mode's
/// amplitude y / (λ + Δ), and y is the slab walk's, so the energy and its
/// adjoint, which reassociate its einsums, differ in its third digit at
/// CAS(8,8), whose Sr keeps modes at 1e-10 of its largest
/// (src/nevpt2/energy/README.md "Conditioning").
inline std::string stable_part(const std::string &message) {
  // std::format prints a non-finite value as inf, -inf or nan: an amplitude
  // y / (λ + Δ) whose denominator rounds to exactly 0 on one side only (#1010).
  static const std::regex computed(R"(-?(?:\d\.\d+e[+-]\d+|\binf\b|\bnan\b))");
  return std::regex_replace(message, computed, "#");
}

} // namespace free_breakfast::test
