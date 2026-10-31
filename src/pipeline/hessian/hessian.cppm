/// @file
/// @brief The `Hessian` step: the closed-shell RHF Cartesian Hessian at the
/// pipeline's molecule, and its harmonic frequencies and normal modes. By
/// default analytic, @c hessian::df_rhf over one nested Pipeline run of the
/// chain from the molecule to `[scf]`; or by central differences of the
/// analytic gradient of the chain to `[gradient]`, rerun at each of the 6N
/// displaced geometries.
///
/// Node-shaped, because each method runs Pipelines of its own, as
/// `[optimize]`'s evaluations do. It reads the MoleculeOutput, so beside an
/// `[optimize]` it runs at the optimum. See README.md.

export module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.geometry; // geometry::Vibrations, in the product
// Every step the chain reruns, and MoleculeOutput among the products it
// re-exports.
export import free_breakfast.pipeline.gradient;

export namespace free_breakfast {

/// @brief How `[hessian]` computes the Hessian.
enum class HessianMethod : uint8_t {
  /// @c hessian::df_rhf at the SCF of one chain run: density fitting only.
  Analytic,
  /// Central differences of the analytic RHF gradient, 6N chain reruns.
  Numerical,
};

/// @brief The Cartesian Hessian of the SCF energy at the MoleculeOutput's
/// geometry, how far it is from the two identities it should keep, and the
/// harmonic vibrations it gives.
struct HessianOutput {
  /// @brief d²E/dx dy (hartree / bohr²), 3 n_atom square, row-major, x =
  /// 3 A + k, atoms in the Molecule's order, symmetrized, (H + Hᵀ) / 2.
  std::vector<double> hessian;
  /// @brief max |H - Hᵀ| before symmetrizing: the central differences', or
  /// the analytic terms' sum's (their rounding).
  double asymmetry{0.0};
  /// @brief max over x and l of |Σ_B H[x][3 B + l]| of @ref hessian: zero
  /// for a Hessian invariant under translation.
  double translational_sum{0.0};
  /// @brief @c geometry::harmonic_analysis of @ref hessian at the
  /// MoleculeOutput's atoms: the frequencies (cm⁻¹, ascending, an imaginary
  /// one negative) and the Cartesian normal modes.
  geometry::Vibrations vibrations;
  /// @brief dmu_c/dx (e), 3 n_atom x 3 row-major, element (x, c) at 3 x + c:
  /// @c hessian::rhf_dipole_derivative over the analytic Hessian's CPHF U^x,
  /// the origin at 0. Empty under HessianMethod::Numerical, and for a basis
  /// with a shell above d (the dipole integrals' @c kDipoleMaxL).
  std::vector<double> dipole_derivative;
  /// @brief The IR intensities (km/mol), one per vibrations.frequencies:
  /// @c hessian::infrared_intensities of @ref dipole_derivative over the
  /// normal modes. Empty when @ref dipole_derivative is.
  std::vector<double> intensities;
};

/// @brief The RHF Hessian of the MoleculeOutput's geometry and its harmonic
/// vibrations, on @p node's device 0.
struct HessianStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "hessian";

  /// @brief How the Hessian is computed.
  HessianMethod method{HessianMethod::Analytic};
  /// @brief HessianMethod::Numerical's step h (bohr): row x is
  /// (g(R + h e_x) - g(R - h e_x)) / 2h. Its error is O(h²) in the gradient's
  /// third derivative plus the SCF's noise over 2h, so the SCF in @ref scf
  /// must be converged well past the step. Not read by the analytic method.
  double displacement{1e-3};

  /// @brief The chain each method runs, configured as the config's own
  /// tables configure it (pipeline/config copies them in). The steps with no
  /// options -- the spherical transform, the pair classes, the overlap, the
  /// core Hamiltonian and the gradient -- are run at their defaults, and no
  /// step of the chain writes a file.
  BasisSetStep basis_set{};
  EriStep eri{};
  OrthogonalizerStep orthogonalizer{};
  GuessStep guess{};
  ScfStep scf{};

  /// @brief Where the Hessian is written as JSON, if not empty: the method,
  /// the Hessian as 3 n_atom rows of 3 n_atom, the asymmetry, the
  /// translational sum, the frequencies and the normal modes, and when the
  /// product has them the dipole derivative and the IR intensities. Arrives
  /// resolved (pipeline/config sets it to hessian.json in the run's output
  /// directory).
  std::filesystem::path json_path{};

  /// @brief The Hessian at the MoleculeOutput's geometry: under
  /// HessianMethod::Analytic one Pipeline run of the chain ending in ScfStep
  /// and @c hessian::df_rhf over its products; under
  /// HessianMethod::Numerical the 6N gradients at the geometry displaced by
  /// ±@ref displacement along each coordinate, each one Pipeline run of the
  /// chain ending in GradientStep, and their central differences. Then
  /// symmetrized, checked, and its harmonic vibrations; under the analytic
  /// method, the dipole derivative from the same CPHF and the IR intensities,
  /// unless a shell is above d (logged as a warning).
  ///
  /// @return The Hessian; with no device, ErrorKind::GpuRuntime; under the
  ///         analytic method, Unsupported, before anything runs, for an
  ///         integral-direct @ref eri, and after the SCF for an ROHF one;
  ///         otherwise the first error of a Pipeline (`[gradient]`'s
  ///         Unsupported for an ROHF SCF among them), of @c hessian::df_rhf,
  ///         @c hessian::rhf_dipole_derivative or
  ///         @c geometry::harmonic_analysis, or the error writing
  ///         @ref json_path.
  [[nodiscard]] Result<HessianOutput> run(View<MoleculeOutput> in, NodeResources &node) const;
};

} // namespace free_breakfast
