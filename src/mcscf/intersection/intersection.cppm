/// @file
/// @brief What a minimum-energy conical intersection search takes at one
/// geometry of a converged DF-SA-CASSCF, from one call: the nuclear gradients
/// g_I and g_J of two roots of equal weight and their coupling h_IJ, from the
/// derivative of the pair's 2 x 2 Hamiltonian under the state average's
/// response alone, which nothing divides by the gap. See README.md.

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
  /// @c df_gradient's own solve takes. A column whose true residual, from
  /// one more product, is above it is an error. The CI response inside each
  /// H~ product is converged to @c response_tolerance_for of it, relative to
  /// its right-hand side, so that the products' error stays under it
  /// (README.md "The floor under the true residual").
  double tolerance = 1e-10;
  /// @brief The H~ products the block solve may take, @c zvectors' default.
  int32_t max_iterations = 200;
};

/// @brief The gradients of two roots and their coupling, each flat and
/// row-major as @c (n_atom, 3), as @c GradientTerms::total: element 3 A + k
/// is the derivative along R_{A,k} (hartree / bohr), atoms in the molecule's
/// order. Each is the adiabatic one, the derivative of the pair's 2 x 2
/// Hamiltonian turned to its eigenvectors at the geometry (README.md): away
/// from a crossing, @c df_gradient's and @c df_nac's numerator; near one,
/// a rotation within the branching plane, which leaves (g_I + g_J) / 2 and
/// span{g_I - g_J, h_IJ} as they are.
struct Intersection {
  /// @brief dE_I/dR.
  std::vector<double> g_i;
  /// @brief dE_J/dR.
  std::vector<double> g_j;
  /// @brief h_IJ = <I|dH/dR|J> with the orbitals and CI relaxed, as
  /// @c Nac::numerator: symmetric in I and J, and translation-invariant.
  std::vector<double> h_ij;
  /// @brief The block solve's products (@c zvectors' largest column's).
  int32_t iterations;
  /// @brief The largest of the three columns' true residuals, each at most
  /// the tolerance.
  double residual;
};

/// @brief g_I, g_J and h_IJ of roots @p i and @p j of @p solution, a
/// converged DF-SA-CASSCF in which the two carry equal weights, from one
/// call: what an MECI search takes at each geometry.
///
/// The three are the derivatives of the pair's 2 x 2 Hamiltonian
/// H_KL = <c_K|H|c_L> at the solution's CI, with the orbitals and the CI
/// relaxed by the state-averaged Lagrangian's response alone (the I-J
/// rotation is redundant at equal weights, so nothing divides by
/// E_I - E_J): its mean, half difference and coupling, three right-hand
/// sides of one @c zvectors over one CI-relaxed Hessian, each to its own
/// derivative call (@c df_gradient at the mixes (1/2, 1/2) and
/// (1/2, -1/2), @c df_transition_gradient at the symmetrized transition
/// RDMs). The 2 x 2 is then diagonalized at the solution and the three
/// turned to its eigenvectors. Away from a crossing that turn is roundoff,
/// and each equals its own call (@c df_gradient, @c df_nac's numerator) to
/// the Z-vector solves' convergence. README.md has the derivation.
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
///         them or they are equal, the tolerance is not positive or
///         max_iterations below 1, or as @c df_nac's checks of the solution,
///         the orbitals and the fit; MaxIterationsExceeded when a column is
///         still above the tolerance after max_iterations products, and
///         Numerical when a column's true residual is (the conjugate
///         gradients' recurrence having passed it); otherwise the first
///         error of @c casscf_integrals, @c response::fock, the CI objects,
///         the Hessian, @c zvectors, @c df_gradient or
///         @c df_transition_gradient, or the GpuRuntime error of the first
///         copy that failed.
[[nodiscard]] Result<Intersection>
df_intersection(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
                const Solution &solution, int32_t ncore, int32_t ncas, int32_t i, int32_t j,
                const IntersectionOptions &options = {});

} // namespace free_breakfast::mcscf
