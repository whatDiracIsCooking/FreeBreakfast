/// @file
/// @brief The setup SCF the mcscf tests (fock, hessian, solve) start from.
///
/// Include it AFTER `import free_breakfast.pipeline.step;` (or any import that
/// makes `ScfStep` visible), like scratch_dir.h: it includes nothing itself.

#pragma once

namespace free_breakfast::test {

/// @brief Unscreened J, every Fock build full, and converged past the step's
/// defaults, since the gradient and Hessian move to first order in the orbitals.
///
/// The bars sit above the GPU's run-to-run noise floor, not at it (#281): on
/// H2CO/cc-pVDZ, |[F,D]| levels off at ~1.08e-12 with dE at +-1e-13, so the
/// 1e-12/1e-13 these once were missed in 100 iterations about one run in
/// several. At |[F,D]| = 1e-10 the orbitals are off by ~1e-10 / gap, which
/// moves the CASCI energy by ~1e-11 (bar 1e-9), the gradient and Hessian by
/// ~1e-10 (bars 1e-7 and up), and the SCF energy at second order only.
inline const ScfStep kExactScf{.energy_tolerance = 1e-11,
                               .commutator_tolerance = 1e-10,
                               .fock_rebuild_interval = 1,
                               .tau_j = 0.0};

} // namespace free_breakfast::test
