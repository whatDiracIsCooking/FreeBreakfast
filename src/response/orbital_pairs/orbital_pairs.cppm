/// @file
/// @brief The packed order of a CASSCF's non-redundant orbital pairs, on the
/// device: a full nmo x nmo matrix's pairs into the packed vector, and a
/// packed vector out to the antisymmetric kappa that C exp(kappa) rotates by.
/// The order the gradient, the Hessian and the optimizer share. See README.md.

export module free_breakfast.response.orbital_pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast::response {

/// @brief The non-redundant pairs of the nmo x nmo column-major device matrix
/// @p full into the device vector @p packed, in the packed order: element
/// (p, q), p > q, for q core and p active or virtual, then q active and p
/// virtual, column by column (q outer), p ascending within a column. So
/// @p packed is two column-major blocks back to back: rows [ncore, nmo) of
/// columns [0, ncore), then rows [nocc, nmo) of columns [ncore, nocc),
/// ncore (nmo - ncore) + ncas (nmo - nocc) doubles in all.
///
/// Two 2D copies on res.stream(), not synchronized.
///
/// @return InvalidConfig if ncore or ncas is negative or ncore + ncas
///         exceeds nmo; or a copy's GpuRuntime error.
[[nodiscard]] Status pack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                        int32_t ncas, const double *full, double *packed);

/// @brief The antisymmetric kappa of the packed @p packed into the nmo x nmo
/// column-major device matrix @p kappa: kappa_pq = x_pq and kappa_qp = -x_pq
/// for each packed pair p > q, every other element 0. C exp(kappa) is then
/// the rotation along x that @c Fock::gradient differentiates.
/// @p kappa and @p packed must not overlap.
///
/// A memset, two 2D copies and two transposing geams on res.stream() with
/// res.blas() (host pointer mode), not synchronized.
///
/// @return InvalidConfig as @ref pack_orbital_pairs; or the GpuRuntime error
///         of a copy or a geam.
[[nodiscard]] Status unpack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                          int32_t ncas, const double *packed, double *kappa);

} // namespace free_breakfast::response
