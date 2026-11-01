/// @file
/// @brief The nonadiabatic (derivative) coupling between two roots of a
/// converged equal-weight DF-SA-CASSCF on the GPU,
///
///     d_IJ = <I|dJ/dR> = h_IJ / (E_J - E_I) + d^CSF_IJ,
///
/// h_IJ the Hellmann-Feynman term at the symmetrized transition RDMs plus its
/// Lagrangian's, over the Z-vector of the transition orbital gradient, and
/// d^CSF the antisymmetrized transition density against the overlap's
/// half derivative. See README.md for the derivation and the conventions.

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
/// (I) and @p ket (J) of @p solution, a converged DF-SA-CASSCF at equal
/// weights, with its numerator and CSF term.
///
/// h_IJ is the derivative of <I|H|J> at the symmetrized transition RDMs
/// (@c ci::Rdm::trans_rdm12) with the core's own energy off, plus the
/// Lagrangian's terms over (z_o, z_c), the Z-vector over the CI-relaxed
/// Hessian at the solution whose right-hand side is the transition orbital
/// gradient (@c response::fock's at the transition RDMs less its at zero
/// RDMs; its CI part is zero at equal weights): one
/// @c df_transition_gradient with those multipliers. d^CSF =
/// 1/2 G[C_a (gamma^T - gamma) C_a^T], gamma = @c ci::Rdm::trans_rdm1(I, J)
/// and G @c overlap_half_gradient. README.md has the derivation, the signs
/// and the energies' order.
///
/// Every argument from @p res to @p ncas is @c df_gradient's, and so are the
/// memory and the stream; the Hessian, the CI objects and the Z-vector are
/// held beside them and not counted. The coupling changes sign with either
/// root's CI phase, as the roots in @p solution carry it.
///
/// @return The coupling; Unsupported for a solution at unequal weights (its
///         weighted gamma not the plain average of root_gammas to 1e-10),
///         for roots whose energies differ by less than 1e-10 without
///         @c mult_ediff, or as @c df_transition_gradient's; InvalidConfig if
///         the solution has fewer than two roots, @p bra or @p ket is not one
///         of them or they are equal, solution.c is not basis.nao_sphe()
///         square or not on @p res's device, ncore or ncas is out of range,
///         the RDMs are not ncas^2, ncas^4 and nroots ncas^2 long, the CI
///         vectors are not the roots', or trace(gamma) is no electron count
///         of the molecule's multiplicity; otherwise the first error of
///         @c casscf_integrals, @c response::fock, the CI objects, the
///         Hessian, @c zvector, @c df_transition_gradient or
///         @c overlap_half_gradient, or the GpuRuntime error of the first
///         copy or BLAS call that failed.
[[nodiscard]] Result<Nac> df_nac(DeviceResources &res, const Molecule &molecule,
                                 const BasisSet &basis, std::span<const PrimitivePairClass> classes,
                                 const BasisSet &aux, const SphericalMatrixAO &h,
                                 const Buffers::device<double> &b, const double *l,
                                 const Solution &solution, int32_t ncore, int32_t ncas, int32_t bra,
                                 int32_t ket, const NacOptions &options = {});

} // namespace free_breakfast::mcscf
