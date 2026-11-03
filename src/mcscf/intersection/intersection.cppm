/// @file
/// @brief What a minimum-energy conical intersection search takes at one
/// geometry of a converged DF-SA-CASSCF, from one call: the nuclear gradients
/// g_I and g_J of two roots of equal weight and their coupling h_IJ, over one
/// CI-relaxed Hessian and one block Z-vector solve. See README.md.

export module free_breakfast.mcscf.intersection;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;

export namespace free_breakfast::mcscf {

/// @brief How @ref df_intersection solves its multipliers.
struct IntersectionOptions {
  /// @brief Each Z-vector column's tolerance on ||H~ z_o + g||, the one
  /// @c df_gradient's own solve takes.
  double tolerance = 1e-10;
};

/// @brief The gradients of two roots and their coupling, each flat and
/// row-major as @c (n_atom, 3), as @c GradientTerms::total: element 3 A + k
/// is the derivative along R_{A,k} (hartree / bohr), atoms in the molecule's
/// order.
struct Intersection {
  /// @brief dE_I/dR, @c df_gradient's total for root I.
  std::vector<double> g_i;
  /// @brief dE_J/dR, @c df_gradient's total for root J.
  std::vector<double> g_j;
  /// @brief h_IJ = <I|dH/dR|J> with the orbitals and CI relaxed,
  /// @c Nac::numerator: symmetric in I and J, and translation-invariant.
  std::vector<double> h_ij;
  /// @brief The block solve's products (@c zvectors' largest column's).
  int32_t iterations;
  /// @brief The largest of the three columns' residuals.
  double residual;
};

/// @brief g_I, g_J and h_IJ of roots @p i and @p j of @p solution, a
/// converged DF-SA-CASSCF in which the two carry equal weights, from one
/// call: what an MECI search takes at each geometry.
///
/// Each is what its own call gives: g_I and g_J @c df_gradient's for the
/// root, h_IJ @c df_nac's @c Nac::numerator, to their Z-vector solves'
/// convergence (each is solved here, not there). What the three share is
/// built once: the CI objects and the CI-relaxed Hessian at the solution;
/// and the three Z-vector right-hand sides (root I's and root J's orbital
/// gradients and the transition orbital gradient) are solved together, one
/// @c zvectors over that Hessian. Each set of multipliers then goes to its
/// own derivative call (@c df_gradient for a root, @c df_transition_gradient
/// at the symmetrized transition RDMs). README.md has the derivation and
/// the timing against the three calls.
///
/// Every argument from @p res to @p ncas is @c df_gradient's, and so are
/// the memory and the stream; the Hessian, the CI objects and the three
/// columns of multipliers are held beside them.
///
/// @return The three vectors; Unsupported if the two roots' weights differ
///         (by more than @c kRotationWeightTolerance of the larger: the
///         rotation between them would divide by E_I - E_J, singular at an
///         intersection), or as @c sa_gradient_weights; InvalidConfig if the
///         solution has fewer than two roots, @p i or @p j is not one of
///         them or they are equal, the tolerance is not positive, or as
///         @c df_nac's checks of the solution, the orbitals and the fit;
///         otherwise the first error of @c casscf_integrals,
///         @c response::fock, the CI objects, the Hessian, @c zvectors,
///         @c df_gradient or @c df_transition_gradient, or the GpuRuntime
///         error of the first copy that failed.
[[nodiscard]] Result<Intersection>
df_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const Solution &solution, int32_t ncore, int32_t ncas, int32_t i, int32_t j,
                const IntersectionOptions &options = {});

} // namespace free_breakfast::mcscf
