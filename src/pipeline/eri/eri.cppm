/// @file
/// @brief The `Eri` step: the two-electron integrals of the whole pipeline,
///        integral-direct (J's and K's pair classes) or density-fitted (the
///        factor B = L^-1 (mn|P) over an auxiliary basis the step places
///        itself, with that basis and L kept beside it), built once for every
///        reader.
///
/// The density-fitted mode is a thin wrapper over
/// integrals.eri.density_fitting's metric, three_center and density_fit; the
/// integral-direct one over coulomb_pair_classes and ordered_pair_classes.
/// Device-shaped; see README.md for why the aux basis is held in the product
/// and not a second one.

export module free_breakfast.pipeline.eri;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.core.wwr;                               // Buffers, in the product
export import free_breakfast.integrals.eri.integral_direct.coulomb;  // CoulombPairClass
export import free_breakfast.integrals.eri.integral_direct.exchange; // OrderedPairClass
export import free_breakfast.pipeline.basis_set;                     // BasisSet, in the product too
export import free_breakfast.pipeline.primitive_pair_classes;

export namespace free_breakfast {

/// @brief Which two-electron integrals @ref EriStep builds.
enum class EriMode : uint8_t {
  /// @brief J and K from the integrals themselves, over pair classes.
  IntegralDirect,
  /// @brief J and K from B over an auxiliary basis.
  DensityFitting,
};

/// @brief The integral-direct J's and K's pair classes of the @ref BasisSet,
/// made from the shared PrimitivePairClassesOutput.
struct IntegralDirect {
  /// @brief J's classes, for @c coulomb. Mutable: each @c coulomb call writes
  /// their scratch, one call at a time, which the device chain gives every
  /// reader of this product (pipeline/README.md).
  mutable std::vector<CoulombPairClass> j_classes;
  /// @brief K's classes, for @c exchange; mutable as @ref j_classes is.
  mutable std::vector<OrderedPairClass> k_classes;
};

/// @brief The density-fitting factor of the @ref BasisSet over one auxiliary
/// basis, on the device the step ran on: (mn|ls) ~ sum_Q B_Q,mn B_Q,ls.
struct DensityFitted {
  /// @brief B, @ref naux x nao_sphe^2 column-major: element (Q, n + nao m) at
  /// Q + naux (n + nao m), Q fastest ([Q][n][m]), what df_coulomb and
  /// df_exchange read. Symmetric in m and n.
  Buffers::device<double> b;
  /// @brief The aux basis's spherical function count: the rows of @ref b.
  int32_t naux;
  /// @brief The aux basis, placed over the Molecule: what the gradient's
  /// derivative integrals are over.
  BasisSet aux;
  /// @brief L, the metric's Cholesky factor M = L L^T over @ref aux, @ref naux
  /// square column-major: the lower triangle, its strict upper one not
  /// meaningful. B is L^-1 (mn|P).
  Buffers::device<double> l;
};

/// @brief The pipeline's two-electron integrals, whichever @ref EriMode
/// chose. Every step that needs them reads this one product and branches on
/// the alternative.
struct EriOutput {
  /// @brief The integral-direct classes or the density-fitted B.
  std::variant<IntegralDirect, DensityFitted> eri;
};

/// @brief Builds the @ref BasisSet's two-electron integrals on @p res's
/// device: under EriMode::IntegralDirect J's and K's pair classes, under
/// EriMode::DensityFitting @ref aux_basis resolved and placed over the
/// @ref Molecule, then the metric, the three-center tensor and their fit,
/// the placed aux basis and L kept with B.
struct EriStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "eri";

  /// @brief Which integrals.
  EriMode mode = EriMode::IntegralDirect;
  /// @brief Under EriMode::DensityFitting, the aux basis: either a
  /// compiled-in fitting set's name (`cc-pvdz-jkfit`) or the path to a BSE
  /// JSON file. @ref resolve_aux_basis decides which, preferring the name. A
  /// path is resolved by the front end, so one written here is relative to
  /// the working directory. Empty: `<BasisSet::name()>-acCD-6-6`, when the
  /// basis is a compiled-in set and that fitting set is carried. Under
  /// EriMode::IntegralDirect it must be empty.
  std::string aux_basis{};

  /// @brief The integrals. An @ref aux_basis under EriMode::IntegralDirect is
  /// ErrorKind::InvalidConfig, and so is an empty one under
  /// EriMode::DensityFitting whose basis has no carried `-acCD-6-6` set (one
  /// read from a path among them). Every other error is the call's that
  /// failed: J's or K's class builders' under EriMode::IntegralDirect; under
  /// EriMode::DensityFitting, an unreadable aux file is ErrorKind::IO; an
  /// unknown name, a malformed set or one lacking an element the molecule
  /// uses is ErrorKind::InvalidConfig; an orbital shell above f, an aux shell
  /// above i, or a B or three-center scratch past the device's free memory is
  /// ErrorKind::Unsupported, naming the shell or the bytes; a metric that is
  /// not positive definite is ErrorKind::Numerical; a failed copy, launch or
  /// BLAS call is ErrorKind::GpuRuntime.
  [[nodiscard]] Result<EriOutput>
  run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
