/// @file
/// @brief The density-fitting Coulomb metric (P|Q) of an auxiliary basis on the
/// GPU, by McMurchie-Davidson over the metric<lP, lQ> kernels FelizJueves
/// generates, in both of the basis's layouts.
///
/// The aux basis is an ordinary @c BasisSet, placed as the orbital one is:
/// `place(resolve_aux_basis(name)->primitive_normalized(), molecule)`. The
/// README has how its coefficients become the kernels' record weights, and why
/// no further normalisation enters.

export module free_breakfast.integrals.eri.density_fitting.metric;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief The Coulomb metric (P|Q) = (P|1/r12|Q) of the auxiliary basis
/// @p aux into @p m: the Cartesian metric into m.cartesian(), then
/// m.spherical() = U^T m.cartesian() U, the metric over @p aux's spherical
/// fitting functions, which is what the fit (@c density_fit) takes.
///
/// @p m must be made for @p aux: `MatrixAOPair(res, aux,
/// upload_spherical_transform_matrix(res, aux))`. Its orders are checked, its
/// U is not (@ref MatrixAOPair).
///
/// Stages one record per aux primitive (its centre, exponent and weight
/// s = c (pi / alpha)^{3/2}, c its coefficient as @p aux holds it) and one
/// entry per aux shell pair P <= Q, by shell index, in its class (lP, lQ),
/// lP <= lQ: a pair whose first shell has the higher momentum is staged
/// swapped, as FelizJueves's staging does. Then one metric<lP, lQ> launch per
/// class that has a pair, every one into m.cartesian(), the transform, and
/// m.spherical() replaced by half of itself plus half of its transpose: the
/// transform's two gemms round (a, b) and (b, a) differently, and the average
/// makes the spherical metric exactly symmetric, as the Cartesian one is by
/// construction. Every element of the Cartesian metric belongs to one pair,
/// so nothing is zeroed first, and the result is bit-reproducible run to run.
///
/// Queued on res.stream(). The staging comes from res.device_workspace(), and
/// the call synchronizes res.stream() once, after its upload, so the host
/// staging can be released; it does not wait for the launches, the transform
/// or the average. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return Success (and nothing written for an empty @p aux); InvalidConfig if
///         @p res is not on @p m's device, if @p m's orders are not @p aux's,
///         if an aux shell is above g, or if aux.nao_cart() is too large for
///         the kernels' 32-bit index (its square above INT32_MAX); or the
///         GpuRuntime error of the first copy, synchronization, launch or BLAS
///         call that failed, which leaves @p m unspecified.
[[nodiscard]] Status metric(DeviceResources &res, const BasisSet &aux, MatrixAOPair &m);

} // namespace free_breakfast
