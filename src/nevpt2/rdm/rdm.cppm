/// @file
/// @brief The three n_act⁶ tensors SC-NEVPT2 reads of its reference state,
/// dm3, f3ca and f3ac, built on the device from one CI vector, its string
/// sets and the active (tu|vw), tile by tile over the determinants, or
/// between a bra and a ket vector, with the transition dm1 and dm2 beside
/// them, for QD-NEVPT2; and the one-vector build's adjoint, from bars on the
/// three back onto the vector and (tu|vw).

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
/// DeviceResources::kWorkspaceAlignment. A @p transition call (a bra other
/// than its ket) carves the ket's own L2 too, n⁴ @p width doubles more.
[[nodiscard]] size_t rdms_workspace_bytes(int32_t n_orbitals, int64_t width,
                                          bool transition = false) noexcept;

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

/// @brief The same three tensors between the bra @p bra, ⟨I|, and the ket
/// @p ket, |J⟩, with the transition dm1 and dm2 beside them; all five
/// non-symmetric, row-major, in PySCF's layout:
///
/// - dm1[p, q] = ⟨I| E_pq |J⟩ (the transpose of `make_dm123(bra, ket)`'s
///   dm1, which is ⟨I| E_qp |J⟩), dm2[p, q, r, s] = ⟨I| E_pq E_rs |J⟩ and
///   dm3[p, q, r, s, t, u] = ⟨I| E_pq E_rs E_tu |J⟩, `make_dm123(bra, ket)`'s
///   before `reorder_dm123`;
/// - f3ca and f3ac the contractions of the transition 4-RDM ⟨I| E E E E |J⟩
///   with (tu|vw) that 'NEVPTkern_cedf_aedf' and 'NEVPTkern_aedf_ecdf' are of
///   one vector's: f3ca[p, q, r, s, t, u] = Σ_ijk (ki|uj) ⟨I| E_pq E_rs E_tj
///   E_ki |J⟩ and f3ac[p, q, r, s, t, u] = Σ_ijk (ik|jt) ⟨I| E_pq E_rs E_ju
///   E_ik |J⟩.
///
/// Per tile, L2 is gathered over the bra and R over the ket, and the ket's own
/// L2, which the consume reads, by a second gather; dm1 += R c_I and dm2 +=
/// L2 c_J are a gemv each. fdm2 and the wedge are those of the transition
/// tensors. I ← J and J ← I are two calls.
///
/// **bra == ket (one pointer) is the one-vector build**: dm3, f3ca and f3ac
/// are the other overload's bit for bit, planned against the same
/// rdms_workspace_bytes(n, width), with one L2 per tile. A bra and a ket that
/// are different pointers plan against rdms_workspace_bytes(n, width, true),
/// even if they hold the same vector.
///
/// @param bra, ket Device, alpha.size() · beta.size() doubles each, laid out
///        as @ref rdms' vector.
/// @param dm1, dm2 Device, n² and n⁴ doubles, overwritten; either may be
///        null, and is then not built.
/// @param alpha, beta, eri, dm3, f3ca, f3ac, max_bytes As for @ref rdms.
/// @return As for @ref rdms, against rdms_workspace_bytes(n, 1, bra != ket)
///         for the budget.
[[nodiscard]] Result<int64_t> rdms(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                   const ci::DeviceStringSet &beta, const double *bra,
                                   const double *ket, const double *eri, double *dm1, double *dm2,
                                   double *dm3, double *f3ca, double *f3ac,
                                   size_t max_bytes = std::numeric_limits<size_t>::max());

/// @brief The bytes of res.device_workspace() a call to @ref rdms_adjoint
/// over @p n_orbitals active orbitals carves at tile width @p width: the two
/// copies of (tu|vw), the two fdm2 bars and the consume's bar, n⁴ doubles
/// each; the two pre-wedge f3 bars, dm3's total bar and dm3 itself, n⁶ each;
/// and the tile's R, L2, W and their bars, (4 n² + 2 n⁴) @p width doubles;
/// each region rounded up to DeviceResources::kWorkspaceAlignment.
[[nodiscard]] size_t rdms_adjoint_workspace_bytes(int32_t n_orbitals, int64_t width) noexcept;

/// @brief The adjoint of @ref rdms: given bars @p dm3_bar, @p f3ca_bar and
/// @p f3ac_bar on its three outputs (same layout), write @p vector_bar =
/// ∂E/∂c and @p eri_bar = ∂E/∂(tu|vw) of E = ⟨dm3̄, dm3⟩ + ⟨f3cā, f3ca⟩ +
/// ⟨f3ac̄, f3ac⟩, at @p vector and @p eri.
///
/// rdms is quadratic in c and linear in (tu|vw), so these are exact:
/// ⟨vector_bar, u⟩ is the bars against (rdms(c + u) − rdms(c − u)) / 2, and
/// ⟨eri_bar, δ⟩ the bars against (rdms(eri + δ) − rdms(eri − δ)) / 2. @p
/// eri_bar is the derivative by each stored element, in @p eri's layout, with
/// no symmetry imposed: rdms reads (tu|vw) as given, so the caller folds it.
///
/// The wedge and fdm2 are transposed once, before the tiles, onto the bars
/// of f3 before the wedge and of dm3. Per tile of determinants K, R and L2
/// are rebuilt as rdms builds them; each digest's transpose is two gemms (L2̄
/// += f3̄ W, W̄ = f3̄ᵀ L2, and dm3's L2̄ += dm3̄ R, R̄ = dm3̄ᵀ L2), each
/// consume's two strided batched gemm loops (onto L2̄ and onto the consume's
/// bar, summed over the tiles), and the produce's transpose gathers R̄ and L2̄
/// onto every determinant J over J's own replacement lists. dm3 is summed
/// over the tiles as rdms sums it, for the fdm2s' (tu|vw) dependence after
/// the last. The 4-RDM is never formed, and nothing is atomic.
///
/// The tile width is planned as rdms plans it, from
/// rdms_adjoint_workspace_bytes against the same budget; the tiles are that
/// wide but for the last.
///
/// @param alpha, beta, vector, eri As for @ref rdms.
/// @param dm3_bar, f3ca_bar, f3ac_bar Device, n⁶ doubles each.
/// @param vector_bar Device, alpha.size() · beta.size() doubles, overwritten.
/// @param eri_bar Device, n⁴ doubles in @p eri's layout, overwritten.
/// @param max_bytes The most of the device workspace the call may hold.
/// @return The number of tiles; the refusals of @ref rdms, against
///         rdms_adjoint_workspace_bytes(n, 1) for the budget; or the
///         GpuRuntime error of a launch or a BLAS call.
[[nodiscard]] Result<int64_t> rdms_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                           const ci::DeviceStringSet &beta, const double *vector,
                                           const double *eri, const double *dm3_bar,
                                           const double *f3ca_bar, const double *f3ac_bar,
                                           double *vector_bar, double *eri_bar,
                                           size_t max_bytes = std::numeric_limits<size_t>::max());

} // namespace free_breakfast::nevpt2
