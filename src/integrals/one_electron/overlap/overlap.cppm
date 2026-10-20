/// @file
/// @brief The Cartesian overlap matrix S on the GPU, over the s_{la}{lb}
/// kernels FelizJueves generates.
///
/// The records come from free_breakfast.integrals.one_electron.upload; S is one
/// kernel launch per shell-pair class into a @c CartesianMatrixAO.

export module free_breakfast.integrals.one_electron.overlap;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief The Cartesian overlap S_mn = <m|n> into @p s.
///
/// Zeroes @p s, then launches the generated kernel of each class, on that
/// class's own stream, which adds every record's contributions at (m, n) and
/// (n, m). @p classes must come from the basis @p s was made for: a record's AO
/// offsets are not checked against s.n().
///
/// Queued on res.stream(), the classes forked from it after the zeroing and
/// joined back into it (@c launch_each), with no host
/// synchronization. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return Success; InvalidConfig if @p res is not on @p s's or @p classes'
///         device, if a class is above d (or not la <= lb), or if s.n() is too
///         large for the kernels' 32-bit index (s.n()^2 above INT32_MAX, about
///         46k AOs); or the GpuRuntime error of the first memset, fork, launch
///         or join that failed, which leaves @p s unspecified.
[[nodiscard]] Status overlap(DeviceResources &res, std::span<const PrimitivePairClass> classes,
                             CartesianMatrixAO &s);

} // namespace free_breakfast
