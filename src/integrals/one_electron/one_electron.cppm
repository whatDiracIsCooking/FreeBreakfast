/// @file
/// @brief One-electron integral matrices on the GPU, over the kernels
/// FelizJueves generates: today the overlap S.
///
/// The host builds the primitive-pair records (free_breakfast.integrals.mmd),
/// @ref upload_primitive_pairs puts them on the device once per basis, and each
/// matrix is then one kernel launch per shell-pair class into a
/// @c CartesianMatrixAO. The spherical matrix is @c MatrixAOPair's
/// @c cartesian_to_spherical of that.

export module free_breakfast.integrals.one_electron;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief One @ref PrimitivePairClass on the device: the same fields, the same
/// layout, @ref n_rec records.
struct DevicePrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive.
  int32_t n_rec;
  /// @brief zeta = alpha + beta.
  Buffers::device<double> zeta;
  /// @brief P (3 per record).
  Buffers::device<double> p;
  /// @brief P - A (3 per record).
  Buffers::device<double> xpa;
  /// @brief P - B (3 per record).
  Buffers::device<double> xpb;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  Buffers::device<double> s;
  /// @brief beta, the ket primitive's exponent.
  Buffers::device<double> eb;
  /// @brief The bra shell's first Cartesian AO.
  Buffers::device<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  Buffers::device<int32_t> off_b;
};

/// @brief Copy @p classes to @p res's device.
///
/// Synchronizes res.stream() once, after the copies, so @p classes may be
/// released when this returns. The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The device classes, in the same order; InvalidConfig for an empty
///         class or one with more records than an int32_t counts; or the
///         GpuRuntime error of the first copy that failed.
[[nodiscard]] Result<std::vector<DevicePrimitivePairClass>>
upload_primitive_pairs(DeviceResources &res, std::span<const PrimitivePairClass> classes);

/// @brief The Cartesian overlap S_mn = <m|n> into @p s.
///
/// Zeroes @p s, then launches the generated kernel of each class, which adds
/// every record's contributions at (m, n) and (n, m). @p classes must come from
/// the basis @p s was made for: a record's AO offsets are not checked against
/// s.n().
///
/// Queued on res.stream(), with no host synchronization. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p res is not on @p s's device, if a class
///         is above d (or not la <= lb), or if s.n() is too large for the
///         kernels' 32-bit index (s.n()^2 above INT32_MAX, about 46k AOs); or
///         the GpuRuntime error of the first memset or launch that failed,
///         which leaves @p s unspecified.
[[nodiscard]] Status overlap(DeviceResources &res,
                             std::span<const DevicePrimitivePairClass> classes,
                             CartesianMatrixAO &s);

} // namespace free_breakfast
