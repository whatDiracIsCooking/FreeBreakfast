/// @file
/// @brief The second derivatives of the density-fitting integrals on the GPU,
/// contracted, never stored: the Coulomb metric's, SUM_PQ W_PQ
/// d2(P|Q)/dR dR, and the three-center integrals', SUM_mnP Gam_mnP
/// d2(mn|P)/dR dR, by McMurchie-Davidson over the d2metric<lP, lQ> and
/// d2eri3<la, lb, lP> kernels FelizJueves generates.
///
/// Both take their matrix in the spherical layout the energy builds write
/// (metric's, three_center's), convert it to the Cartesian one the kernels
/// contract against, and return the 3 n_atom x 3 n_atom Hessian on the host.
/// The README has the staging and the transforms.

export module free_breakfast.integrals.eri.density_fitting.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

/// @brief SUM_PQ W_PQ d2(P|Q)/dR_{A,k} dR_{B,l} of the auxiliary basis
/// @p aux, for any spherical naux x naux @p w on the device (column-major; no
/// symmetry assumed): H[A][k][B][l] at (3 A + k) 3 n_atom + 3 B + l, row-major
/// over the atoms of @p aux.
///
/// @p aux must be made by @c place, so it knows which atom each shell sits on.
/// W is made Cartesian as U_aux W U_aux^T, then the metric's records and aux
/// shell pairs are staged as @c metric_gradient stages them, and one
/// d2metric<lP, lQ> launch per class that has a pair adds into a zeroed
/// device H, which is downloaded. The kernels' atomic adds make a run
/// reproducible to round-off, not bitwise.
///
/// Queued on res.stream(), and synchronized: the call returns when H is on the
/// host. The scratch comes from res.device_workspace(), so @p w must not point
/// into it. The device res.dev_idx() names is made current for the call and
/// the previous one restored.
///
/// @return The Hessian (all zeros for an empty @p aux); InvalidConfig if
///         @p aux was not made by @c place, if an aux shell is above i, or if
///         aux.nao_cart() is too large for the kernels' 32-bit index; or the
///         GpuRuntime error of the first copy, launch or BLAS call that
///         failed.
[[nodiscard]] Result<std::vector<double>> metric_hessian(DeviceResources &res, const BasisSet &aux,
                                                         const double *w);

/// @brief SUM_mnP Gam_mnP d2(mn|P)/dR_{A,k} dR_{B,l} of the orbital basis
/// @p basis and the auxiliary basis @p aux, for any spherical @p gamma on the
/// device in @c three_center's layout, element (m, n, P) at
/// P + naux (n + nao m), P fastest (no symmetry assumed): H[A][k][B][l] at
/// (3 A + k) 3 n_atom + 3 B + l.
///
/// @p classes must be @c primitive_pair_classes(res, basis), finished, and
/// both bases made by @c place over the same molecule. Gam is made Cartesian
/// and staged as @c three_center_gradient does it; then one
/// d2eri3<la, lb, lP> launch per class that has a cell adds into a zeroed
/// device H, which is downloaded. The atomic adds make a run reproducible to
/// round-off.
///
/// **Refused up front, not split.** The Cartesian Gam and its transform's
/// intermediate are scratch from res.device_workspace(); if they and the
/// staging need more bytes than the device has free (beyond what the
/// workspace already holds), the call is Unsupported, naming the bytes,
/// before anything is allocated.
///
/// Queued on res.stream(), and synchronized: the call returns when H is on the
/// host. @p gamma must not point into res.device_workspace(). The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return The Hessian (all zeros if either basis is empty); Unsupported if
///         an orbital shell is above f or an aux shell above i (naming the
///         shell), if the scratch does not fit (naming the bytes), or if
///         nao_cart^2 naux_cart is past the kernels' 32-bit index;
///         InvalidConfig if the bases were not made by @c place over one
///         molecule, or if @p classes do not hold @p basis's shell pairs; or
///         the GpuRuntime error of the first copy, launch or BLAS call that
///         failed.
[[nodiscard]] Result<std::vector<double>>
three_center_hessian(DeviceResources &res, const BasisSet &basis,
                     std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                     const double *gamma);

} // namespace free_breakfast
