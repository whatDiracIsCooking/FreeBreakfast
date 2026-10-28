/// @file
/// @brief The `Casscf` step: the state-averaged CASSCF of the lowest singlets
/// of an active space of the @ref ScfOutput's orbitals, named by index or
/// picked by AVAS, its core and virtual orbitals canonicalized.
///
/// Device-shaped. The selection is `pipeline.active_space`'s, which
/// `CasscfStep` derives from, as `CasciStep` does. `mcscf.solve`'s
/// `mcscf::solve` does the optimization; the canonicalization, the Fock it
/// reads and the Molden writer are private to `run`. See README.md.

export module free_breakfast.pipeline.casscf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.mcscf.solve; // mcscf::Iteration, in the product
export import free_breakfast.pipeline.scf;
export import free_breakfast.pipeline.active_space; // CasscfStep's base

export namespace free_breakfast {

/// @brief A converged state-averaged CASSCF over a set of RHF orbitals.
struct CasscfOutput {
  /// @brief The doubly occupied core orbitals: columns [0, ncore) of @ref c.
  int32_t ncore;
  /// @brief The active orbitals: columns [ncore, ncore + ncas) of @ref c.
  int32_t ncas;
  /// @brief The active electrons, nelecas / 2 of each spin.
  int32_t nelecas;
  /// @brief The optimized orbitals, AO by MO: core, active, virtual. The core
  /// and the virtuals each diagonalize the state-averaged Fock within their
  /// space, by ascending energy; the active orbitals are as optimized, the
  /// ones @ref vectors and @ref rdm1 are over.
  SphericalMatrixAO c;
  /// @brief Each column of @ref c's diagonal element of the state-averaged
  /// Fock F^I + F^A (hartree): within the core and the virtuals its
  /// eigenvalues.
  std::vector<double> mo_energies;
  /// @brief The total energy per root, core and nuclear repulsion included
  /// (hartree), ascending.
  std::vector<double> energies;
  /// @brief <S^2> per root.
  std::vector<double> spin_squares;
  /// @brief The weights w_r the energy averages, one per root, summing to 1.
  std::vector<double> weights;
  /// @brief sum_r w_r E_r (hartree).
  double energy;
  /// @brief ||g|| at @ref c.
  double gradient_norm;
  /// @brief The unit CI vectors, root r at [r dim, (r + 1) dim), in
  /// @ref CasciOutput::vectors' layout.
  std::vector<double> vectors;
  /// @brief The spin-summed 1-RDM per root over the active orbitals, root r
  /// at [r ncas^2, (r + 1) ncas^2), ncas x ncas (symmetric).
  std::vector<double> rdm1;
  /// @brief One entry per macro iteration run, rejected ones included.
  std::vector<mcscf::Iteration> iterations;
};

/// @brief State-averaged CASSCF from the @ref ScfOutput's orbitals, on
/// @p res's device. The starting active space is the
/// @ref ActiveSpaceSelection's: `ncas`, `nelecas`, `active`, or `avas` with
/// `avas_threshold`.
struct CasscfStep : ActiveSpaceSelection {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "casscf";

  /// @brief The singlet roots averaged, at least 1.
  int32_t nroots = 1;
  /// @brief Their weights, @ref nroots of them, non-negative and summing to
  /// 1; empty for equal ones.
  std::vector<double> weights{};
  /// @brief @ref mcscf::Options::conv_tol, |Delta E| at convergence (hartree).
  double conv_tol = mcscf::Options{}.conv_tol;
  /// @brief @ref mcscf::Options::conv_tol_grad, ||g|| at convergence.
  double conv_tol_grad = mcscf::Options{}.conv_tol_grad;
  /// @brief @ref mcscf::Options::max_cycle, macro iterations before
  /// MaxIterationsExceeded.
  int32_t max_cycle = mcscf::Options{}.max_cycle;
  /// @brief The density-fitting aux basis, a compiled-in fitting set's name
  /// (`cc-pvdz-jkfit`) or the path to a BSE JSON file, as
  /// ScfStep::aux_basis. Empty: every J and K integral-direct. Otherwise the
  /// step builds its own B over it, by DensityFittingStep, and runs
  /// `mcscf::solve` in mcscf::DensityFitting::Full, so the result is the
  /// DF-CASSCF; the Fock at the solution is fitted too.
  std::string aux_basis{};
  /// @brief Where the state-averaged natural orbitals are written as a
  /// Molden file, if not empty: the canonical core at occupation 2, the
  /// active space's natural orbitals by descending weighted occupation, the
  /// canonical virtuals at 0, each with its energy in the state-averaged
  /// Fock.
  std::filesystem::path molden{};

  /// @brief The optimized orbitals and roots. The start is the orbitals
  /// @ref ActiveSpaceSelection::select gives; `mcscf::solve` optimizes them
  /// at @ref conv_tol, @ref conv_tol_grad and @ref max_cycle, its other
  /// options at their defaults, over J and K classes built per run, or over
  /// the B the run builds under @ref aux_basis; the state-averaged Fock at
  /// the solution is `response::fock`'s F^I + F^A over one more
  /// `casscf_integrals`, both fitted under @ref aux_basis. Logs each macro
  /// iteration's energy and ||g|| at info. Synchronizes res.stream() many
  /// times.
  ///
  /// @return The CASSCF; @ref ActiveSpaceSelection::select's error;
  ///         under @ref aux_basis DensityFittingStep::run's (B past the
  ///         device's free memory among them), and Unsupported when the
  ///         fitted integrals' scratch, L and J_AO, is past what B leaves
  ///         free; `mcscf::solve`'s (InvalidConfig for @ref nroots above the
  ///         determinant count or bad @ref weights, MaxIterationsExceeded);
  ///         or the error of the first module call that failed:
  ///         `casscf_integrals`', `response::fock`'s, `diagonalize`'s or
  ///         `write_molden`'s, and GpuRuntime for a failed copy or BLAS call.
  [[nodiscard]] Result<CasscfOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
           CoreHamiltonianOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
