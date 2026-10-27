/// @file
/// @brief A chosen norm (ell_1, ell_2 or ell_inf) of a real device vector, as
/// one BLAS reduction: asum, nrm2, or iamax plus a read of the element it names.
///
/// The README has why only real types, and how ell_inf works in each pointer
/// mode.

export module free_breakfast.recipes.norm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief The vector norm norm() reports.
enum class Norm : std::uint8_t {
  l1,  ///< sum_i |x_i|, asum
  l2,  ///< sqrt(sum_i x_i^2), nrm2
  inf, ///< max_i |x_i|, iamax and a read of that element
};

/// @brief Write the @p which norm of the @p n elements of @p x (stride @p incx)
/// to @p result.
///
/// @p result follows res.blas()'s pointer mode, as a BLAS reduction's does: a
/// host pointer in host mode (the default), where the call has waited for the
/// value; a device pointer in device mode, where nothing synchronizes and
/// @p result is ready once res.stream() is. The handle's mode is read, never
/// changed.
///
/// ell_inf in device mode draws one int32_t for iamax's index from
/// res.device()'s pool, not from res.device_workspace(), so @p x may point into
/// the workspace. Everything is queued on res.stream(). The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @tparam which The norm; not deducible, so spell norm<Norm::inf>(...).
/// @param res    Supplies the stream and the BLAS handle.
/// @param n      Length of @p x. n <= 0 is empty: @p result is set to 0.
/// @param x      Device vector, stride @p incx.
/// @param incx   Stride between elements of @p x; must be positive.
/// @param result The norm: host or device memory, per the pointer mode.
/// @return Success; or a GpuRuntime error: the first runtime, BLAS or launch
///         call that failed.
template <Norm which, real_fp T>
[[nodiscard]] Status norm(DeviceResources &res, int32_t n, const T *x, int32_t incx, T *result);

} // namespace free_breakfast
