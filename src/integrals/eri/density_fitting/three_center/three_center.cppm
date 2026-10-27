/// @file
/// @brief The density-fitting three-center integrals (mn|P) on the GPU, by
/// McMurchie-Davidson over the eri3<la, lb, lP> kernels FelizJueves generates,
/// spherical on all three indices, in the layout density_fit takes.
///
/// The bra records are mmd's PrimitivePairClass ones, the aux records the
/// metric's. The README has the build, the layout and why the coefficients
/// need no normalisation beyond the bases' own.

export module free_breakfast.integrals.eri.density_fitting.three_center;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

/// @brief The spherical three-center integrals (mn|P) = (mn|1/r12|P) of the
/// orbital basis @p basis and the auxiliary basis @p aux into @p b: the
/// naux x nao^2 matrix (P|mn), column-major, element (P, n + nao m) at
/// P + naux (n + nao m), so P is fastest ([m][n][P]), nao and naux the two
/// bases' spherical counts. That is the layout @c density_fit takes. Exactly
/// symmetric in m and n.
///
/// @p classes must be @c primitive_pair_classes(res, basis), finished (as it
/// returns them). Their records are regrouped by shell pair, mmd's order kept
/// within each, and staged with one aux record per aux primitive and one cell
/// per (AO shell pair i <= j, aux shell) in its class (la, lb, lP). Then one
/// eri3<la, lb, lP> launch per class that has a cell, every one into the
/// Cartesian tensor, nao_cart^2 naux_cart doubles in one allocation; then the
/// aux index made spherical by one gemm with U_aux^T, the two AO indices by a
/// strided-batched gemm and a gemm with U, and the result written to @p b as
/// half of itself plus half of its (m, n) mirror, since the last two gemms
/// round (m, n) and (n, m) differently. Every Cartesian element belongs to one
/// cell, so nothing is zeroed and a run is bit-reproducible.
///
/// **Refused up front, not split.** The Cartesian tensor and its transform's
/// intermediate are scratch from res.device_workspace(); if they and the
/// staging need more bytes than @p max_bytes, or than the device has free
/// (beyond what the workspace already holds), the call is Unsupported, naming
/// the bytes, before anything is allocated. Splitting the tensor over m would
/// need FelizJueves's mirrored store changed. @p max_bytes is the cap a test
/// forces; leave it at its default to be limited by the device alone.
///
/// Queued on res.stream(). The call synchronizes it twice: once after the
/// records' download, once after the staging's upload, so the host copies can
/// go; it does not wait for the launches or the transform. @p b must not point
/// into res.device_workspace(). The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return Success (and nothing written if either basis is empty);
///         Unsupported if an orbital shell is above f or an aux shell above g
///         (naming the shell), or if the tensor does not fit (naming the
///         bytes), or if nao_cart^2 naux_cart is past the kernels' 32-bit
///         index; InvalidConfig if @p classes do not hold @p basis's shell
///         pairs; or the GpuRuntime error of the first copy, synchronization,
///         launch or BLAS call that failed, which leaves @p b unspecified.
[[nodiscard]] Status three_center(DeviceResources &res, const BasisSet &basis,
                                  std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                                  double *b, size_t max_bytes = std::numeric_limits<size_t>::max());

} // namespace free_breakfast
