/// @file
/// @brief The Cartesian core Hamiltonian H = T + V on the GPU, formed in place
/// from free_breakfast.integrals.one_electron.kinetic and .potential_en.

export module free_breakfast.integrals.one_electron.core_hamiltonian;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief The Cartesian core Hamiltonian H = T + V into @p h.
///
/// Checks every input first, then @c kinetic zeroes @p h and writes T, and
/// @c add_nuclear_attraction adds V on top: both kernels only add, so the sum
/// is formed in place, with no T or V of its own and no separate addition.
/// Otherwise @c kinetic's and @c nuclear_attraction's contract.
///
/// @return Success; InvalidConfig, before anything is launched, if @p res is
///         not on @p h's or @p nuclei's device, if a class is above d (or not
///         la <= lb), or if h.n() is too large for the kernels' 32-bit index;
///         or the GpuRuntime error of the first memset or launch that failed,
///         which leaves @p h unspecified.
[[nodiscard]] Status core_hamiltonian(DeviceResources &res, const DevicePrimitivePairs &pairs,
                                      const DeviceNuclei &nuclei, CartesianMatrixAO &h);

} // namespace free_breakfast
