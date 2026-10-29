/// @file
/// @brief @ref free_breakfast::ScfOutput, the converged restricted (RHF or
/// ROHF) wavefunction.
///
/// An interface partition so that the module-private `:loop` can name the
/// product it returns. The primary interface re-exports it.

export module free_breakfast.pipeline.scf:output;

import std;
import free_breakfast.core.types;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief A converged restricted wavefunction: closed-shell RHF, or ROHF, one
/// set of orbitals whose n_beta lowest are doubly occupied and the next
/// n_alpha - n_beta singly.
///
/// The matrices are over the spherical AOs, @c nao_sphe square, column-major,
/// on the device the step ran on.
struct ScfOutput {
  /// @brief The total energy, electronic plus nuclear repulsion (hartree), of
  /// the density the last Fock matrix was built from.
  double energy;
  /// @brief The Fock builds it took.
  int32_t iterations;
  /// @brief The alpha electrons: the first n_alpha columns of @ref c are
  /// occupied in alpha. Equal to @ref n_beta for RHF.
  int32_t n_alpha;
  /// @brief The beta electrons: the first n_beta columns of @ref c are
  /// occupied in beta, n_beta <= @ref n_alpha. A step that handles only a
  /// closed shell refuses n_alpha != n_beta as Unsupported.
  int32_t n_beta;
  /// @brief The orbital energies, ascending, one per column of @ref c
  /// (hartree), on the host. For ROHF, the eigenvalues of the Roothaan
  /// effective Fock matrix in PySCF's convention (its @c get_roothaan_fock).
  std::vector<double> mo_energies;
  /// @brief The MO coefficients C, AO by MO: F C = S C eps, F the effective
  /// Fock matrix for ROHF.
  SphericalMatrixAO c;
  /// @brief The spin-summed density D = 2 C_occ C_occ^T, of the canonical
  /// orbitals of the last Fock matrix; for ROHF, Da + Db, with
  /// Da = C[:, :n_alpha] C[:, :n_alpha]^T and Db likewise over n_beta.
  SphericalMatrixAO d;
};

} // namespace free_breakfast
