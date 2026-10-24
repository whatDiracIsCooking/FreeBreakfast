/// @file
/// @brief The `Scf` step: closed-shell restricted Hartree-Fock, its loop, its
/// Fock build and its DIIS history all private to one `run`.
///
/// Device-shaped. The step's first device consumer of the host `GuessOutput`, which
/// it uploads and rescales to the molecule's electron count. See README.md.

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

export namespace free_breakfast {

/// @brief A converged closed-shell RHF wavefunction.
///
/// The matrices are over the spherical AOs, @c nao_sphe square, column-major,
/// on the device the step ran on.
struct ScfOutput {
  /// @brief The total energy, electronic plus nuclear repulsion (hartree), of
  /// the density the last Fock matrix was built from.
  double energy;
  /// @brief The Fock builds it took.
  int32_t iterations;
  /// @brief The orbital energies, ascending, one per column of @ref c
  /// (hartree), on the host.
  std::vector<double> mo_energies;
  /// @brief The MO coefficients C, AO by MO: F C = S C eps.
  SphericalMatrixAO c;
  /// @brief The spin-summed density D = 2 C_occ C_occ^T, of the canonical
  /// orbitals of the last Fock matrix.
  SphericalMatrixAO d;
};

/// @brief Iterates RHF to self-consistency from the @ref GuessOutput, with Pulay
/// DIIS on the Fock matrix and incremental Fock builds.
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
  /// on a full build. 1 makes every build full. At least 1.
  int32_t fock_rebuild_interval = 20;
  /// @brief J's screening threshold, @c coulomb's tau: 0 or above, 0 skipping
  /// nothing. The default is @c coulomb's own. Its screening error is in the
  /// converged energy and orbitals (2.6e-10 Eh on N2/cc-pVDZ at the default),
  /// so a comparison tighter than that runs at 0. Not a [scf] TOML key.
  double tau_j = 1e-10;
  /// @brief Where the converged orbitals are written as a Molden file, if not
  /// empty: every MO, occupation 2 for the occupied ones. Arrives resolved
  /// (pipeline/config anchors a relative one to the config file).
  std::filesystem::path molden{};

  /// @brief The converged wavefunction. An open shell (an odd electron count,
  /// or a multiplicity other than 1) is ErrorKind::Unsupported; an option out
  /// of range (a negative or NaN @ref tau_j among them), more electron pairs than orbitals, or a guess of the wrong size
  /// is ErrorKind::InvalidConfig; a class above d is J's and K's
  /// InvalidConfig; no convergence within @ref max_iterations is
  /// ErrorKind::MaxIterationsExceeded; a non-converged eigensolve is
  /// ErrorKind::Numerical; a failed runtime, BLAS, solver or launch call is
  /// ErrorKind::GpuRuntime; a @ref molden file that cannot be written is
  /// @c write_molden's error.
  [[nodiscard]] Result<ScfOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
           OverlapOutput, OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
