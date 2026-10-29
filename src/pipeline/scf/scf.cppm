/// @file
/// @brief The `Scf` step: restricted Hartree-Fock, closed-shell or (density
/// fitted) open-shell ROHF, its loop (the
/// module-private `:loop`), its Fock build and its DIIS history all private to
/// one `run`.
///
/// Device-shaped. The two-electron part is the @ref free_breakfast::EriOutput's:
/// integral-direct or density-fitted. The step's first device
/// consumer of the host `GuessOutput`, which it uploads and rescales to the
/// molecule's electron count. See README.md.

export module free_breakfast.pipeline.scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.pipeline.orthogonalizer;
export import free_breakfast.pipeline.core_hamiltonian;
export import free_breakfast.pipeline.guess;
export import free_breakfast.pipeline.eri;
export import :output;

export namespace free_breakfast {

/// @brief Iterates RHF to self-consistency from the @ref GuessOutput, with Pulay
/// DIIS on the Fock matrix, its two-electron part the @ref EriOutput's: either
/// integral-direct, with incremental Fock builds, or density-fitted. A molecule
/// at a multiplicity above 1 is ROHF, over the Roothaan effective Fock matrix
/// in PySCF's convention, and density-fitted only.
struct ScfStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "scf";

  /// @brief The most Fock builds before giving up. At least 1.
  int32_t max_iterations = 100;
  /// @brief Converged when the energy moved by less than this between two
  /// Fock builds (hartree)...
  double energy_tolerance = 1e-10;
  /// @brief ...and the Frobenius norm of the orthogonal-basis commutator
  /// X^T (F D S - S D F) X is under this.
  double commutator_tolerance = 1e-7;
  /// @brief The most (F, error) pairs DIIS keeps, 1..kDiisMaxHistoryLimit.
  int32_t diis_history = 8;
  /// @brief Every this many Fock builds, one is full: G of the density itself.
  /// The rest are incremental, F plus G of the density's change since the last
  /// build, which J's and K's screens skip more of. Convergence is only taken
  /// on a full build. 1 makes every build full. At least 1. The
  /// integral-direct build's only: under a density-fitted @ref EriOutput every
  /// build is full, and this and @ref tau_j are not read.
  int32_t fock_rebuild_interval = 20;
  /// @brief J's screening threshold, @c coulomb's tau: 0 or above, 0 skipping
  /// nothing. The default is @c coulomb's own. Its screening error is in the
  /// converged energy and orbitals (2.6e-10 Eh on N2/cc-pVDZ at the default),
  /// so a comparison tighter than that runs at 0. Not a [scf] TOML key.
  double tau_j = 1e-10;
  /// @brief Where the converged orbitals are written as a Molden file, if not
  /// empty: every MO, occupation 2 for the doubly occupied ones and 1 for an
  /// open shell's singly occupied ones. Arrives resolved
  /// (pipeline/config sets it to scf.molden in the run's output directory).
  std::filesystem::path molden{};

  /// @brief The converged wavefunction. An open shell under an integral-direct
  /// @ref EriOutput is ErrorKind::Unsupported; an option out of range (a
  /// negative or NaN @ref tau_j among them), an electron count whose parity
  /// the multiplicity cannot have, more alpha electrons than orbitals, or a
  /// guess of the wrong size is ErrorKind::InvalidConfig; a class above f is J's and K's
  /// InvalidConfig; no convergence within @ref max_iterations is
  /// ErrorKind::MaxIterationsExceeded; a non-converged eigensolve is
  /// ErrorKind::Numerical; a failed runtime, BLAS, solver or launch call is
  /// ErrorKind::GpuRuntime; a @ref molden file that cannot be written is
  /// @c write_molden's error.
  [[nodiscard]] Result<ScfOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, EriOutput, OverlapOutput,
           OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
