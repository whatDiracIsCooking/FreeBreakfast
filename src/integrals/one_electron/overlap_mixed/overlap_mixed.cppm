/// @file
/// @brief The mixed Cartesian overlap S_{rows,cols} between two bases on the
/// GPU, over the overlap_mixed<la, lb> kernels FelizJueves generates.
///
/// The two bases may differ in basis set, in geometry, in both, or in neither:
/// a basis against itself is the square S that overlap builds. The records
/// come from free_breakfast.integrals.mmd's mixed_primitive_pair_classes; S is
/// one kernel launch per ordered shell-pair class into a plain device buffer.

export module free_breakfast.integrals.one_electron.overlap_mixed;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The Cartesian overlap S_mn = <m|n>, m an AO of @p rows and n of
/// @p cols, into @p s: column-major rows.nao_cart() x cols.nao_cart(), leading
/// dimension rows.nao_cart() (element (m, n) at m + n * rows.nao_cart()), as
/// MatrixAO lays a matrix out.
///
/// Forms the records between the two bases (mixed_primitive_pair_classes,
/// which waits for its own streams), zeroes @p s on res.stream(), then
/// launches each ordered class's generated kernel on that class's own stream,
/// forked from res.stream() and joined back into it (@c launch_each), with no
/// host synchronization after the records. Every element is added by exactly
/// one ordered shell pair, nothing mirrored, so with @p rows and @p cols the
/// same basis S is symmetric to rounding, not bitwise.
///
/// @p s must hold exactly rows.nao_cart() * cols.nao_cart() doubles, on
/// res.device(); its device is not checked (a buffer does not say). The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p s is the wrong size, if a shell of
///         either basis is above d, or if the matrix is too large for the
///         kernels' 32-bit index (rows.nao_cart() * cols.nao_cart() above
///         INT32_MAX); or the GpuRuntime error of the record build, or of the
///         first memset, fork, launch or join that failed, which leaves @p s
///         unspecified.
[[nodiscard]] Status overlap_mixed(DeviceResources &res, const BasisSet &rows, const BasisSet &cols,
                                   Buffers::device<double> &s);

/// @brief The spherical overlap S_mn = <m|n>, m a spherical AO of @p rows and
/// n of @p cols, into @p s: column-major rows.nao_sphe() x cols.nao_sphe(),
/// leading dimension rows.nao_sphe().
///
/// overlap_mixed's Cartesian block carried over as U_rows^T S_cart U_cols,
/// each U the basis's spherical_transform_matrix: two gemms. The Cartesian
/// block, both U's and the intermediate are pool buffers on res.device().
/// Queued on res.stream(), which is synchronized once at the end, so the
/// host copies of the U's outlive their upload.
///
/// @p s must hold exactly rows.nao_sphe() * cols.nao_sphe() doubles, on
/// res.device().
///
/// @return Success; InvalidConfig if @p s is the wrong size, or overlap_mixed's
///         errors; or the GpuRuntime error of a failed upload or gemm, which
///         leaves @p s unspecified.
[[nodiscard]] Status overlap_mixed_spherical(DeviceResources &res, const BasisSet &rows,
                                             const BasisSet &cols, Buffers::device<double> &s);

} // namespace free_breakfast
