/// @file
/// @brief The three n_act⁶ tensors SC-NEVPT2 reads of its reference state,
/// dm3, f3ca and f3ac, built on the device from one CI vector, its string
/// sets and the active (tu|vw), tile by tile over the determinants.

export module free_breakfast.nevpt2.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;

export namespace free_breakfast::nevpt2 {

/// @brief The bytes of res.device_workspace() a call to @ref rdms over
/// @p n_orbitals active orbitals carves at tile width @p width: the two
/// copies of (tu|vw) and fdm2, n⁴ doubles each, and the tile's R, L2 and W,
/// (2 n² + n⁴) @p width doubles, each region rounded up to
/// DeviceResources::kWorkspaceAlignment.
[[nodiscard]] size_t rdms_workspace_bytes(int32_t n_orbitals, int64_t width) noexcept;

/// @brief dm3, f3ca and f3ac of the CI vector @p vector on the device, in
/// PySCF's layout: each n⁶ row-major, (p, q, r, s, t, u) at
/// ((((p n + q) n + r) n + s) n + t) n + u.
///
/// - dm3[p, q, r, s, t, u] = ⟨c| E_pq E_rs E_tu |c⟩, PySCF's
///   `fci.rdm.make_dm123('FCI3pdm_kern_sf', …)` before `reorder_dm123`;
/// - f3ca and f3ac are `mrpt.nevpt2._contract4pdm`'s 'NEVPTkern_cedf_aedf'
///   and 'NEVPTkern_aedf_ecdf' of (tu|vw) and c: the 4-RDM contracted with
///   the integrals, which never exists.
///
/// Per tile of determinants K, R[tu, K] = ⟨K| E_tu |c⟩ and
/// L2[pqrs, K] = ⟨c| E_pq E_rs |K⟩ are gathered over K's own replacement
/// lists; dm3 += L2 Rᵀ is a gemm; W = (tu|vw) · L2, per order, is a strided
/// batched gemm per contracted index, and f3 += L2 Wᵀ a gemm. fdm2 and the
/// wedge run once after the last tile. (tu|vw) is converted at entry, once,
/// from @p eri's layout to the two row-major copies the kernels and the
/// consume read.
///
/// **The tile width is the widest whose rdms_workspace_bytes fits the
/// budget**, at most every determinant: the budget is @p max_bytes, or less
/// if the device has less: the larger of the workspace already held and the
/// free memory less 512 MiB of headroom, for the BLAS libraries' own
/// workspaces and the pool. The tiles are that wide but for the last, which
/// takes the remainder. There is no knob for the count.
///
/// Every scratch region is res.device_workspace()'s. Queued on res.stream()
/// with res.blas(), whose pointer mode must be host (its default); it does
/// not synchronize.
///
/// @param alpha, beta The two spins' strings, on res's device (one set may be
///        both); the vector is alpha-major, (K_α, K_β) at
///        K_α · beta.size() + K_β, PySCF's (na, nb) CI array.
/// @param vector Device, alpha.size() · beta.size() doubles.
/// @param eri Device, (tu|vw) at t + n (u + n (v + n w)), as
///        ActiveSpaceHamiltonian::eri holds it.
/// @param dm3, f3ca, f3ac Device, n⁶ doubles each, overwritten.
/// @param max_bytes The most of the device workspace the build may hold.
/// @return The number of tiles; InvalidConfig, with nothing queued, for a set
///         on another device than @p res, sets over different orbital counts
///         or none, or more determinants than an int32_t counts; Unsupported,
///         with nothing queued or allocated, for a budget below
///         rdms_workspace_bytes(n, 1); or the GpuRuntime error of a launch or
///         a BLAS call.
[[nodiscard]] Result<int64_t> rdms(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                   const ci::DeviceStringSet &beta, const double *vector,
                                   const double *eri, double *dm3, double *f3ca, double *f3ac,
                                   size_t max_bytes = std::numeric_limits<size_t>::max());

} // namespace free_breakfast::nevpt2
