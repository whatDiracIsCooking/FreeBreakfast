/// @file
/// @brief The one-electron nuclear-gradient contractions on the GPU: the
/// overlap, kinetic-energy and nuclear-attraction derivative integrals
/// contracted with an AO matrix into a per-atom gradient, over the gs_, gt_ and
/// gv_{la}{lb} kernels FelizJueves generates.
///
/// Each returns G[A, k] = Sum_mn M_mn dX_mn/dR_{A,k}: the derivative integrals
/// themselves (an [n_atom, 3, n_ao, n_ao] tensor) are never formed. The signs
/// and halves of an energy are the caller's: an RHF gradient takes M = D for T
/// and V, and -W, the energy-weighted density negated, for S.
///
/// Beside them, the derivative integrals themselves, for a Hessian's CPHF
/// right-hand sides: overlap_derivative, kinetic_derivative and
/// potential_derivative form dX_mn/dR_{A,k} for every atom and axis as
/// spherical AO matrices on the device, over the ip1 kernels (ips_, ipt_,
/// ipv_) and V's per-nucleus operator kernel (iprv_).

export module free_breakfast.integrals.one_electron.gradient;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief G[A, k] = Sum_mn M_mn dS_mn/dR_{A,k}, the overlap's derivative
/// integrals contracted with @p m, for every atom A of @p basis and axis k.
///
/// @p m is a spherical AO matrix of @p basis, carried onto the Cartesian layout
/// internally (U m U^T, as the Fock build carries the density); it need not be
/// symmetric. Each atom's derivative takes both of its pair's centres,
/// differentiated, so the per-axis sum over atoms vanishes to rounding rather
/// than by construction.
///
/// @p classes must be @c primitive_pair_classes of @p basis, which must carry
/// its per-atom shell map (made by @c place): the gradient's rows are its
/// atoms. Queued on res.stream(), each class on its own stream forked from it
/// and joined back into it; the call waits for the stream, to return the
/// gradient on the host. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return The gradient, flat and row-major as @c (basis.natoms(), 3) (as
///         @c nuclear_repulsion_gradient); or InvalidConfig if @p m is not
///         basis.nao_sphe() square or on @p res's device, if @p basis has no
///         per-atom map, or for any input @c check_one_electron_build refuses;
///         or the GpuRuntime error of the first copy, launch or wait that
///         failed.
[[nodiscard]] Result<std::vector<double>>
overlap_gradient(DeviceResources &res, const BasisSet &basis,
                 std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &m);

/// @brief G[A, k] = Sum_mn M_mn dT_mn/dR_{A,k}, the kinetic energy's derivative
/// integrals contracted with @p m. As @ref overlap_gradient in every other
/// respect.
[[nodiscard]] Result<std::vector<double>>
kinetic_gradient(DeviceResources &res, const BasisSet &basis,
                 std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &m);

/// @brief G[A, k] = Sum_mn M_mn dV_mn/dR_{A,k}, the nuclear attraction's
/// derivative integrals contracted with @p m: the functions' derivative on
/// their two centres and the operator's (Hellmann-Feynman) derivative on every
/// nucleus C of @p molecule, V = -Sum_C Z_C / |r - C| over point nuclei.
///
/// @p molecule must be the molecule @p basis was placed on, so its atoms are
/// the basis's, in the same order. As @ref overlap_gradient in every other
/// respect.
///
/// @return As @ref overlap_gradient; InvalidConfig too if @p molecule's atom
///         count is not basis.natoms(), or its nuclei are on another device.
[[nodiscard]] Result<std::vector<double>>
potential_gradient(DeviceResources &res, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes, const Molecule &molecule,
                   const SphericalMatrixAO &m);

/// @brief S^x: dS_mn/dR_{A,k} for every atom A of @p basis and axis k, as
/// spherical AO matrices of @p basis on @p res's device.
///
/// The result is basis.natoms() * 3 matrices of order n = basis.nao_sphe(),
/// matrix x = 3 A + k at x n^2, each symmetric (to rounding), so either
/// layout reads it: the [n_atom, 3, n, n] tensor. Both centres of every pair
/// are differentiated, so the sum over atoms vanishes to rounding rather than
/// by construction. Contracting matrix (A, k) with M gives
/// @ref overlap_gradient's G[A, k].
///
/// Built from the first-index derivative X1 (the ip1 kernels, one launch per
/// class, all atoms at once) carried onto the spherical layout, atom A's matrix
/// being X1's rows on A plus their transpose. Its scratch is its own: one
/// Cartesian [3, n_cart, n_cart], one spherical [3, n, n] and a
/// @c MatrixAOPair. @p classes must be @c primitive_pair_classes of @p basis,
/// which must carry its per-atom shell map (made by @c place). Queued on
/// res.stream(), each class on its own stream forked from it and joined back
/// into it; the call waits for the stream, so its scratch can be released.
/// The device res.dev_idx() names is made current for the call and the
/// previous one restored.
///
/// @return The matrices; or InvalidConfig if @p basis has no per-atom map or
///         more than 21845 atoms, if 3 n_cart^2 does not fit an int32_t, or for
///         any input @c check_one_electron_build refuses; or the GpuRuntime
///         error of the first copy, launch, transform or wait that failed. A
///         failed allocation aborts (CreatePolicy).
[[nodiscard]] Result<Buffers::device<double>>
overlap_derivative(DeviceResources &res, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes);

/// @brief T^x: dT_mn/dR_{A,k}, the kinetic energy's. As
/// @ref overlap_derivative in every other respect.
[[nodiscard]] Result<Buffers::device<double>>
kinetic_derivative(DeviceResources &res, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes);

/// @brief V^x: dV_mn/dR_{A,k}, the nuclear attraction's, V = -Sum_C Z_C /
/// |r - C| over the point nuclei of @p molecule: the functions' derivative on
/// their centres and the operator's on every nucleus.
///
/// Atom C's matrix also holds dV^C/dC_k, V^C its own nucleus's term,
/// built per nucleus (the iprv kernels, one launch per class and nucleus) into
/// the same scratch. @p molecule must be the molecule @p basis was placed on.
/// As @ref overlap_derivative in every other respect.
///
/// @return As @ref overlap_derivative; InvalidConfig too if @p molecule's atom
///         count is not basis.natoms(), or its nuclei are on another device.
[[nodiscard]] Result<Buffers::device<double>>
potential_derivative(DeviceResources &res, const BasisSet &basis,
                     std::span<const PrimitivePairClass> classes, const Molecule &molecule);

} // namespace free_breakfast
