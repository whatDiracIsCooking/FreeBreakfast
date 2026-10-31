/// @file
/// @brief The one-electron nuclear-Hessian contractions on the GPU: the
/// overlap, kinetic-energy and nuclear-attraction second-derivative integrals
/// contracted with an AO matrix into the 3N x 3N Hessian, over the d2s_, d2t_
/// and d2v_{la}{lb} kernels FelizJueves generates.
///
/// Each returns H[A, k, B, l] = Sum_mn M_mn d2X_mn/dR_{A,k}dR_{B,l}: the
/// second-derivative integrals themselves (an [n_atom, 3, n_atom, 3, n_ao,
/// n_ao] tensor) are never formed. The signs and halves of an energy are the
/// caller's: an RHF Hessian takes M = D for T and V, and -W, the
/// energy-weighted density negated, for S.

export module free_breakfast.integrals.one_electron.hessian;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief H[A, k, B, l] = Sum_mn M_mn d2S_mn/dR_{A,k}dR_{B,l}, the overlap's
/// second-derivative integrals contracted with @p m, for every pair of atoms
/// A, B of @p basis and axes k, l.
///
/// @p m is a spherical AO matrix of @p basis, carried onto the Cartesian layout
/// internally (U m U^T, as the Fock build carries the density); it need not be
/// symmetric. The off-diagonal blocks H[A, ., B, .], A != B, are the kernels'
/// own; each diagonal block is formed from its column by translational
/// invariance, H[A, k, A, l] = -Sum_{B != A} H[B, k, A, l], which keeps the
/// rounding of the one-centre pairs' cancelling terms out of it. So the sum over
/// the row atom vanishes by construction, and the sum over the column atom to
/// the rounding of the off-diagonal blocks' symmetry.
///
/// @p classes must be @c primitive_pair_classes of @p basis, which must carry
/// its per-atom shell map (made by @c place): the Hessian's rows and columns are
/// its atoms. Queued on res.stream(), each class on its own stream forked from
/// it and joined back into it; the call waits for the stream, to return the
/// Hessian on the host. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return The Hessian, flat and row-major as @c (3 natoms, 3 natoms), row
///         3 A + k and column 3 B + l (as @c nuclear_repulsion_hessian); or
///         InvalidConfig if @p m is not basis.nao_sphe() square or on @p res's
///         device, if @p basis has no per-atom map, or for any input
///         @c check_one_electron_build refuses; or the GpuRuntime error of the
///         first copy, launch or wait that failed.
[[nodiscard]] Result<std::vector<double>>
overlap_hessian(DeviceResources &res, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &m);

/// @brief H[A, k, B, l] = Sum_mn M_mn d2T_mn/dR_{A,k}dR_{B,l}, the kinetic
/// energy's second-derivative integrals contracted with @p m. As
/// @ref overlap_hessian in every other respect.
[[nodiscard]] Result<std::vector<double>>
kinetic_hessian(DeviceResources &res, const BasisSet &basis,
                std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &m);

/// @brief H[A, k, B, l] = Sum_mn M_mn d2V_mn/dR_{A,k}dR_{B,l}, the nuclear
/// attraction's second-derivative integrals contracted with @p m, V = -Sum_C
/// Z_C / |r - C| over point nuclei: the functions' second derivative on their
/// two centres, the function-operator terms on a centre and a nucleus C, and
/// the operator's own second derivative on C.
///
/// @p molecule must be the molecule @p basis was placed on, so its atoms are
/// the basis's, in the same order. As @ref overlap_hessian in every other
/// respect.
///
/// @return As @ref overlap_hessian; InvalidConfig too if @p molecule's atom
///         count is not basis.natoms(), or its nuclei are on another device.
[[nodiscard]] Result<std::vector<double>>
potential_hessian(DeviceResources &res, const BasisSet &basis,
                  std::span<const PrimitivePairClass> classes, const Molecule &molecule,
                  const SphericalMatrixAO &m);

} // namespace free_breakfast
