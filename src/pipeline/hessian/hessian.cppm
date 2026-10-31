/// @file
/// @brief The `Hessian` step: the closed-shell RHF Cartesian Hessian at the
/// pipeline's molecule, by central differences of the analytic gradient of
/// the chain from the molecule to `[gradient]`, rerun as a nested Pipeline at
/// each of the 6N displaced geometries.
///
/// Node-shaped, because each displacement runs a Pipeline of its own, as
/// `[optimize]`'s evaluations do. It reads the MoleculeOutput, so beside an
/// `[optimize]` it runs at the optimum. The numerical Hessian is the
/// reference the analytic one (`free_breakfast.hessian`) is tested against,
/// and depends on none of it. See README.md.

export module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
// Every step the chain reruns, and MoleculeOutput among the products it
// re-exports.
export import free_breakfast.pipeline.gradient;

export namespace free_breakfast {

/// @brief How `[hessian]` computes the Hessian.
enum class HessianMethod : uint8_t {
  /// Central differences of the analytic RHF gradient, 6N chain reruns.
  Numerical,
};

/// @brief The Cartesian Hessian of the SCF energy at the MoleculeOutput's
/// geometry, and how far it is from the two identities it should keep.
struct HessianOutput {
  /// @brief d²E/dx dy (hartree / bohr²), 3 n_atom square, row-major, x =
  /// 3 A + k, atoms in the Molecule's order: the central differences
  /// symmetrized, (H + Hᵀ) / 2.
  std::vector<double> hessian;
  /// @brief max |H - Hᵀ| of the central differences before symmetrizing.
  double asymmetry{0.0};
  /// @brief max over x and l of |Σ_B H[x][3 B + l]| of @ref hessian: zero
  /// for a Hessian invariant under translation.
  double translational_sum{0.0};
};

/// @brief The RHF Hessian of the MoleculeOutput's geometry, on @p node's
/// device 0.
struct HessianStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "hessian";

  /// @brief How the Hessian is computed; the only method so far.
  HessianMethod method{HessianMethod::Numerical};
  /// @brief The central difference's step h (bohr): row x is
  /// (g(R + h e_x) - g(R - h e_x)) / 2h. Its error is O(h²) in the gradient's
  /// third derivative plus the SCF's noise over 2h, so the SCF in @ref scf
  /// must be converged well past the step.
  double displacement{1e-3};

  /// @brief The chain rerun at every displaced geometry, configured as the
  /// config's own tables configure it (pipeline/config copies them in). The
  /// steps with no options -- the spherical transform, the pair classes, the
  /// overlap, the core Hamiltonian and the gradient -- are run at their
  /// defaults, and no step of the chain writes a file.
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};

  /// @brief Where the Hessian is written as JSON, if not empty: the step,
  /// the Hessian as 3 n_atom rows of 3 n_atom, the asymmetry and the
  /// translational sum. Arrives resolved (pipeline/config sets it to
  /// hessian.json in the run's output directory).
  std::filesystem::path json_path{};

  /// @brief The 6N gradients at the MoleculeOutput's geometry displaced by
  /// ±@ref displacement along each coordinate, each one Pipeline run of the
  /// chain ending in GradientStep, their central differences, and the
  /// symmetrized Hessian.
  ///
  /// @return The Hessian; with no device, ErrorKind::GpuRuntime; otherwise
  ///         the first error of a displacement's Pipeline (`[gradient]`'s
  ///         Unsupported for an ROHF SCF among them), or the error writing
  ///         @ref json_path.
  [[nodiscard]] Result<HessianOutput> run(View<MoleculeOutput> in, NodeResources &node) const;
};

} // namespace free_breakfast
