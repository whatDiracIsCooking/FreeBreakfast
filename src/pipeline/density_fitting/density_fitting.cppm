/// @file
/// @brief The `DensityFitting` step: the fitted three-center factor
///        B = L^-1 (mn|P) over an auxiliary basis the step places itself.
///
/// A thin wrapper over integrals.eri.density_fitting's metric, three_center
/// and density_fit. Device-shaped; see README.md for why the aux basis is the
/// step's own and not a second product.

export module free_breakfast.pipeline.density_fitting;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.core.wwr; // Buffers, in the product
export import free_breakfast.pipeline.basis_set;
export import free_breakfast.pipeline.primitive_pair_classes;

export namespace free_breakfast {

/// @brief The density-fitting factor of the @ref BasisSet over one auxiliary
/// basis, on the device the step ran on: (mn|ls) ~ sum_Q B_Q,mn B_Q,ls.
struct DensityFittingOutput {
  /// @brief B, @ref naux x nao_sphe^2 column-major: element (Q, n + nao m) at
  /// Q + naux (n + nao m), Q fastest ([Q][n][m]), what df_coulomb and
  /// df_exchange read. Symmetric in m and n.
  Buffers::device<double> b;
  /// @brief The aux basis's spherical function count: the rows of @ref b.
  int32_t naux;
};

/// @brief Resolves and places @ref aux_basis over the @ref Molecule, then
/// builds the metric, the three-center tensor and their fit on @p res's
/// device.
struct DensityFittingStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "density_fitting";

  /// @brief The aux basis: either a compiled-in fitting set's name
  /// (`cc-pvdz-jkfit`) or the path to a BSE JSON file. @ref resolve_aux_basis
  /// decides which, preferring the name. A path is resolved by the front end,
  /// so one written here is relative to the working directory.
  std::string aux_basis{};

  /// @brief B for the basis and @ref aux_basis. Every error is the call's that
  /// failed: an unreadable aux file is ErrorKind::IO; an unknown name, a
  /// malformed set or one lacking an element the molecule uses is
  /// ErrorKind::InvalidConfig; an orbital shell above f, an aux shell above g,
  /// or a B or three-center scratch past the device's free memory is
  /// ErrorKind::Unsupported, naming the shell or the bytes; a metric that is
  /// not positive definite is ErrorKind::Numerical; a failed copy, launch or
  /// BLAS call is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<DensityFittingOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
