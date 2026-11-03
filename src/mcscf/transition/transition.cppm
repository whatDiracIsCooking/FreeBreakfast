/// @file
/// @brief The orbital gradient of a transition element <c_s|H|c_r> between
/// two roots of a DF-CASSCF, at fixed orbitals and CI vectors: the
/// right-hand side of a coupling's Z-vector (@c df_nac), and of a
/// multiplier that holds the roots apart. See README.md.

export module free_breakfast.mcscf.transition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast::mcscf {

/// @brief g^(sr), the orbital gradient of <c_s|H|c_r> = E_act(gamma,
/// Gamma) with the CI vectors held: @c response::fock's gradient at the
/// symmetrized transition RDMs @p gamma and @p gamma2 less its gradient at
/// zero RDMs (the core's own energy multiplies <c_s|c_r> = 0, so it drops).
///
/// Every argument is the density-fitted @c response::fock's, and so are the
/// layouts: @p ppaa and @p papa @c casscf_integrals' over @p c, @p gamma
/// and @p gamma2 @c ci::Rdm::trans_rdm12(s, r)'s read back to the host
/// (ncas^2 and ncas^4, row-major), symmetric in s and r. The result is
/// @c Fock::gradient's: the non-redundant pairs p > q in
/// @c response::pack_orbital_pairs' order, 2 (F - F^T), on @p res's device.
/// Only the pairs (p, i), i core and p not, differ from the gradient at the
/// RDMs alone.
///
/// Two @c response::fock calls and one axpy, queued on res.stream() with
/// res.blas(), and waited for. The zero-RDM Fock is freed on return.
///
/// @return The gradient; the first error of either @c response::fock, or
///         the GpuRuntime error of the axpy or the sync.
[[nodiscard]] Result<Buffers::device<double>> transition_orbital_gradient(
    DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, int32_t naux, const SphericalMatrixAO &c, int32_t ncore,
    int32_t ncas, const Buffers::device<double> &ppaa, const Buffers::device<double> &papa,
    std::span<const double> gamma, std::span<const double> gamma2);

} // namespace free_breakfast::mcscf
