/// @file
/// @brief The nonadiabatic (derivative) coupling between two roots of a
/// converged DF-SA-CASSCF at any weights on the GPU,
///
///     d_IJ = <I|dJ/dR> = h_IJ / (E_J - E_I) + d^CSF_IJ,
///
/// h_IJ the Hellmann-Feynman term at the symmetrized transition RDMs plus its
/// Lagrangian's, over the Z-vector of the transition orbital gradient, and
/// d^CSF the antisymmetrized transition density against the overlap's
/// half derivative; and a combination of couplings Sum_MN T_MN d_MN plus an
/// orbital density's term, in one Z-vector. See README.md for the derivation
/// and the conventions.

export module free_breakfast.mcscf.nac;

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

/// @brief What @ref df_nac returns in @ref Nac::total, PySCF's
/// NonAdiabaticCouplings switches of the same names.
struct NacOptions {
  /// @brief Return h_IJ + (E_J - E_I) d^CSF instead of h_IJ / (E_J - E_I) +
  /// d^CSF: the coupling times the gap, finite at a degeneracy.
  bool mult_ediff = false;
  /// @brief Leave the CSF term out of @ref Nac::total (electron-translation
  /// factors), so it is h_IJ / (E_J - E_I), or h_IJ under @ref mult_ediff.
  bool use_etfs = false;
};

/// @brief A coupling between two roots and its pieces, each flat and
/// row-major as @c (n_atom, 3): element 3 A + k is the derivative along
/// R_{A,k}, atoms in the molecule's order.
struct Nac {
  /// @brief h_IJ = <I|dH/dR|J> with the orbitals and CI relaxed, unscaled
  /// (hartree / bohr): symmetric in I and J, and translation-invariant.
  std::vector<double> numerator;
  /// @brief d^CSF_IJ (1 / bohr), unscaled, whatever the options:
  /// antisymmetric in I and J, and not translation-invariant.
  std::vector<double> csf;
  /// @brief The coupling as @ref NacOptions asks: by default d_IJ =
  /// numerator / (E_J - E_I) + csf.
  std::vector<double> total;
  /// @brief The Z-vector's conjugate-gradient products.
  int32_t iterations;
  /// @brief The Z-vector's residual.
  double residual;
};

/// @brief The nonadiabatic coupling d_IJ = <I|dJ/dR> between roots @p bra
/// (I) and @p ket (J) of @p solution, a converged DF-SA-CASSCF at any
/// weights, with its numerator and CSF term.
///
/// h_IJ is the derivative of <I|H|J> at the symmetrized transition RDMs
/// (@c ci::Rdm::trans_rdm12) with the core's own energy off, plus the
/// Lagrangian's terms over (z_o, z_c), the Z-vector over the CI-relaxed
/// Hessian at the solution whose right-hand side is the transition orbital
/// gradient (@c response::fock's at the transition RDMs less its at zero
/// RDMs; its CI and root-rotation parts are zero at any weights): one
/// @c df_transition_gradient with those multipliers, the rotations among the
/// roots of unequal weight among them. The weights are exactly 1 / nroots
/// when the weighted gamma is the roots' plain average to 1e-10 (and then
/// the call runs the same operations it did before unequal weights were
/// taken), else recovered by @c sa_gradient_weights. d^CSF =
/// 1/2 G[C_a (gamma^T - gamma) C_a^T], gamma = @c ci::Rdm::trans_rdm1(I, J)
/// and G @c overlap_half_gradient. README.md has the derivation, the signs
/// and the energies' order.
///
/// Every argument from @p res to @p ncas is @c df_gradient's, and so are the
/// memory and the stream; the Hessian, the CI objects and the Z-vector are
/// held beside them and not counted. The coupling changes sign with either
/// root's CI phase, as the roots in @p solution carry it.
///
/// @return The coupling; Unsupported for roots whose energies differ by less
///         than 1e-10 without @c mult_ediff, or as @c df_transition_gradient's;
///         InvalidConfig if the solution has fewer than two roots, @p bra or
///         @p ket is not one of them or they are equal, solution.c is not
///         basis.nao_sphe() square or not on @p res's device, ncore or ncas is
///         out of range, the RDMs are not ncas^2, ncas^4 and nroots ncas^2
///         long, the CI vectors are not the roots', or trace(gamma) is no
///         electron count of the molecule's multiplicity; otherwise the first
///         error of @c sa_gradient_weights, @c casscf_integrals,
///         @c response::fock, the CI objects, the Hessian, @c zvector,
///         @c df_transition_gradient or @c overlap_half_gradient, or the
///         GpuRuntime error of the first copy or BLAS call that failed.
[[nodiscard]] Result<Nac> df_nac(DeviceResources &res, const Molecule &molecule,
                                 const BasisSet &basis, std::span<const PrimitivePairClass> classes,
                                 const BasisSet &aux, const SphericalMatrixAO &h,
                                 const Buffers::device<double> &b, const double *l,
                                 const Solution &solution, int32_t ncore, int32_t ncas, int32_t bra,
                                 int32_t ket, const NacOptions &options = {});

/// @brief A combination of couplings and its pieces, each flat and row-major
/// as @c (n_atom, 3), as @ref Nac's.
struct NacCombination {
  /// @brief Sum_MN T_MN h_MN / (E_N - E_M) plus D^X's orbital response: one
  /// @c df_transition_gradient at the combined transition RDMs over the
  /// Z-vector of both right-hand sides (1 / bohr).
  std::vector<double> response;
  /// @brief Sum_MN T_MN d^CSF_MN = 1/2 G[C_a (gamma^T^T - gamma^T) C_a^T],
  /// gamma^T = Sum_MN T_MN gamma^MN.
  std::vector<double> csf;
  /// @brief D^X's basis-motion term, 1/2 G[C (D^X^T - D^X) C^T] over every
  /// orbital; zero without D^X.
  std::vector<double> basis_motion;
  /// @brief response + csf + basis_motion: Sum_MN T_MN d_MN + d[D^X].
  std::vector<double> total;
  /// @brief The Z-vector's orbital right-hand side on the host, over the
  /// packed pairs of @c response::pack_orbital_pairs: the transition orbital
  /// gradient at the combined RDMs plus D^X_pq - D^X_qp.
  std::vector<double> rhs;
  /// @brief The smallest |E_N - E_M| over the pairs with T_MN != 0
  /// (hartree); infinity when T is zero.
  double gap;
  /// @brief The Z-vector's conjugate-gradient products.
  int32_t iterations;
  /// @brief The Z-vector's residual.
  double residual;
};

/// @brief A combination of couplings between the roots of @p solution plus
/// an orbital density's term, in one Z-vector:
///
///     d^T = Sum_MN T_MN d_MN + d[D^X],
///     d[D^X] = Sum_xy D^X_xy <phi_x|d phi_y / dR>,
///
/// d_MN = <M|dN/dR> as the pair overload's, and d[D^X] with the orbitals
/// relaxed: the basis functions' motion and the SA orbitals' response.
///
/// @p t is T, nroots x nroots row-major (T_MN at M nroots + N), exactly
/// antisymmetric (T_MN == -T_NM). @p d_x is D^X, nmo x nmo row-major
/// (D^X_xy at x nmo + y, nmo = nao, indices in @p solution's orbitals), or
/// empty for none; it must be zero within the core, the active and the
/// virtual blocks, whose rotations are redundant, so it lives on the pairs
/// (t, i), (a, t), (a, i) and their transposes. Its index order is the
/// transition density's, gamma^MN_xy = <M|E_xy|N>: D^X in gamma^MN's place
/// gives Sum_xy gamma^MN_xy <phi_x|d phi_y>, the pair's CSF term.
///
/// By linearity: one right-hand side, the transition orbital gradient at
/// Sum_MN T_MN / (E_N - E_M) (gamma^MN, Gamma^MN) symmetrized, plus
/// D^X_pq - D^X_qp over the packed pairs; one @c zvector and one
/// @c df_transition_gradient at those RDMs; the CSF term at
/// Sum_MN T_MN gamma^MN; and D^X's basis motion in its own
/// @c overlap_half_gradient. README.md "A combination of couplings".
///
/// Every other argument, the memory and the stream are the pair overload's.
///
/// @return The combination; InvalidConfig if T is not nroots^2 long, not
///         finite or not antisymmetric, D^X is neither empty nor nmo^2 long,
///         not finite or not zero on the three diagonal blocks, or as the pair
///         overload's (but for its root checks); Numerical, naming the pair,
///         for two roots closer than @c kRootGapFloor (1e-8 hartree) with
///         T_MN != 0; otherwise the pair overload's errors.
[[nodiscard]] Result<NacCombination>
df_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
       std::span<const PrimitivePairClass> classes, const BasisSet &aux, const SphericalMatrixAO &h,
       const Buffers::device<double> &b, const double *l, const Solution &solution, int32_t ncore,
       int32_t ncas, std::span<const double> t, std::span<const double> d_x = {});

} // namespace free_breakfast::mcscf
