/// @file
/// @brief The Cartesian dipole matrices mu_x, mu_y, mu_z on the GPU, origin at
/// 0, over the mu_{la}{lb} kernels FelizJueves generates.
///
/// The records come from free_breakfast.integrals.mmd; the three matrices are
/// one kernel launch per shell-pair class, every component at once, into three
/// @c CartesianMatrixAO.

export module free_breakfast.integrals.one_electron.dipole;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief The largest momentum the dipole kernels cover: d.
///
/// Below @ref kOneElectronMaxL: FelizJueves validates its dipole kernel
/// against libcint through (dd) only, and its emitter refuses f.
/// tools/emit/one_electron_kernels.py's `MU_MAX_L`, written by hand because a
/// module unit cannot include the emitted table; the launcher bounds itself by
/// that table's extent as well.
inline constexpr int32_t kDipoleMaxL = 2;

/// @brief The Cartesian dipole matrices mu_d,mn = <m| r_d |n>, d = x, y, z,
/// origin at 0, into @p mu_x, @p mu_y and @p mu_z.
///
/// One launch of the generated kernel per class, on that class's own stream,
/// adds every record's contributions to all three components at (m, n) and
/// (n, m), into a 3 n^2 region of res.device_workspace() zeroed first; each
/// component is then copied into its matrix. @p classes must come from the
/// basis the matrices were made for: a record's AO offsets are not checked
/// against their order. Each matrix is symmetric to rounding, not bitwise.
///
/// Queued on res.stream(), the classes forked from it after the zeroing and
/// joined back into it (@c launch_each), with no host synchronization. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored.
///
/// @return Success; InvalidConfig if @p res is not on the matrices' or
///         @p classes' device, if the three matrices differ in order, if a
///         class is above d (@ref kDipoleMaxL: a basis with an f shell is
///         refused), or if the order is too large for the kernels' 32-bit
///         index (3 n^2 above INT32_MAX, about 26k AOs); or the GpuRuntime
///         error of the first memset, fork, launch, join or copy that failed,
///         which leaves the matrices unspecified.
[[nodiscard]] Status dipole(DeviceResources &res, std::span<const PrimitivePairClass> classes,
                            CartesianMatrixAO &mu_x, CartesianMatrixAO &mu_y,
                            CartesianMatrixAO &mu_z);

} // namespace free_breakfast
