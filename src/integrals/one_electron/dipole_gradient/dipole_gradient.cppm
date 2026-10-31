/// @file
/// @brief The dipole integrals' nuclear derivative contracted with an AO
/// matrix on the GPU, per component and atom, over the gmu_{la}{lb} kernels
/// FelizJueves generates.
///
/// G[c, A, k] = Sum_mn M_mn d mu^c_mn / dR_{A,k}, mu^c_mn = <m| r_c |n> about
/// the origin (the @c dipole matrices'), c = x, y, z: the derivative integrals
/// themselves are never formed. The origin is fixed, so only the basis
/// functions move. The sign of an electronic dipole is the caller's: its
/// derivative at fixed density is -G at M = D.

export module free_breakfast.integrals.one_electron.dipole_gradient;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief G[c, A, k] = Sum_mn M_mn d mu^c_mn / dR_{A,k}, the dipole integrals'
/// derivative contracted with @p m, for every component c, atom A of
/// @p basis and axis k.
///
/// mu^c_mn = <m| r_c |n> about the origin, as @c dipole builds it. The origin
/// does not move with the atoms, so there is no operator term: both centres of
/// every pair are differentiated, and the per-axis sum over atoms is minus the
/// origin's derivative, Sum_A G[c, A, k] = delta_ck Sum_mn M_mn S_mn, a check
/// rather than a construction.
///
/// @p m is a spherical AO matrix of @p basis, carried onto the Cartesian layout
/// internally (U m U^T, as the Fock build carries the density); it need not be
/// symmetric. @p classes must be @c primitive_pair_classes of @p basis, which
/// must carry its per-atom shell map (made by @c place). Queued on
/// res.stream(), each class on its own stream forked from it and joined back
/// into it; the call waits for the stream, to return the gradient on the host.
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The gradient, flat and row-major as @c (3, basis.natoms(), 3),
///         component first: G[c, A, k] at 3 (natoms c + A) + k; or
///         InvalidConfig if @p m is not basis.nao_sphe() square or on
///         @p res's device, if @p basis has no per-atom map, or for any input
///         @c check_one_electron_build refuses (a class above f); or the
///         GpuRuntime error of the first copy, launch or wait that failed.
[[nodiscard]] Result<std::vector<double>>
dipole_gradient(DeviceResources &res, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &m);

} // namespace free_breakfast
