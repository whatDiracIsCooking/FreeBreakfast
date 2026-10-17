/// @file
/// @brief One-electron integral matrices on the GPU, over the kernels
/// FelizJueves generates: the overlap S, the kinetic energy T, the nuclear
/// attraction V, and the core Hamiltonian H = T + V.
///
/// The host builds the primitive-pair records (free_breakfast.integrals.mmd),
/// @ref upload_primitive_pairs puts them on the device once per basis (and
/// @ref upload_nuclei the nuclei once per geometry; both in
/// free_breakfast.integrals.one_electron.upload), and each matrix is then one
/// kernel launch per shell-pair class into a @c CartesianMatrixAO. The
/// spherical matrix is @c MatrixAOPair's @c cartesian_to_spherical of that.
///
/// S is free_breakfast.integrals.one_electron.overlap, re-exported here; T,
/// V and H are still defined in this module.

export module free_breakfast.integrals.one_electron;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.linalg.matrix_ao;

// The device inputs and S, each its own module; re-exported so this import
// still gives every one-electron build.
export import free_breakfast.integrals.one_electron.upload;
export import free_breakfast.integrals.one_electron.overlap;

export namespace free_breakfast {

/// @brief The Cartesian kinetic energy T_mn = <m| -1/2 grad^2 |n> into @p t.
///
/// @ref overlap's contract, for the generated t_{la}{lb} kernels, which also
/// read each record's ket exponent.
///
/// @return As @ref overlap.
[[nodiscard]] Status kinetic(DeviceResources &res,
                             std::span<const DevicePrimitivePairClass> classes,
                             CartesianMatrixAO &t);

/// @brief The Cartesian nuclear attraction
/// V_mn = -Sum_C Z_C <m| 1/|r - C| |n> into @p v.
///
/// @ref overlap's contract, for the generated v_{la}{lb} kernels, which loop
/// over @p nuclei serially inside each record. Point nuclei.
///
/// @return As @ref overlap; also InvalidConfig if @p nuclei are on another
///         device than @p res.
[[nodiscard]] Status nuclear_attraction(DeviceResources &res,
                                        std::span<const DevicePrimitivePairClass> classes,
                                        const DeviceNuclei &nuclei, CartesianMatrixAO &v);

/// @brief The Cartesian core Hamiltonian H = T + V into @p h.
///
/// Zeroes @p h once, then launches the kinetic and the nuclear-attraction
/// kernels of every class into it: both only add, so their sum is formed in
/// place, with no T or V of its own and no separate addition. Otherwise
/// @ref kinetic's and @ref nuclear_attraction's contract.
///
/// @return As @ref nuclear_attraction.
[[nodiscard]] Status core_hamiltonian(DeviceResources &res,
                                      std::span<const DevicePrimitivePairClass> classes,
                                      const DeviceNuclei &nuclei, CartesianMatrixAO &h);

} // namespace free_breakfast
