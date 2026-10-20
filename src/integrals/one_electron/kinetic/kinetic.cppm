/// @file
/// @brief The Cartesian kinetic-energy matrix T on the GPU, over the
/// t_{la}{lb} kernels FelizJueves generates.
///
/// The records come from free_breakfast.integrals.one_electron.upload; T is one
/// kernel launch per shell-pair class into a @c CartesianMatrixAO.

export module free_breakfast.integrals.one_electron.kinetic;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief The Cartesian kinetic energy T_mn = <m| -1/2 grad^2 |n> into @p t.
///
/// Zeroes @p t, then launches the generated kernel of each class, which adds
/// every record's contributions at (m, n) and (n, m); the kernels also read each
/// record's ket exponent. @p pairs must come from the basis @p t was made for:
/// a record's AO offsets are not checked against t.n().
///
/// Queued on res.stream(), each class on its own stream, forked from it after
/// the zeroing and joined back into it (@c DevicePrimitivePairs::launch_each),
/// with no host synchronization. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @return Success; InvalidConfig if @p res is not on @p t's or @p pairs's
///         device, if a class is above d (or not la <= lb), or if t.n() is too
///         large for the kernels' 32-bit index (t.n()^2 above INT32_MAX, about
///         46k AOs); or the GpuRuntime error of the first memset, fork, launch
///         or join that failed, which leaves @p t unspecified.
[[nodiscard]] Status kinetic(DeviceResources &res, const DevicePrimitivePairs &pairs,
                             CartesianMatrixAO &t);

} // namespace free_breakfast
