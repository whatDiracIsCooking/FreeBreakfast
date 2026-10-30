/// @file
/// @brief SC-NEVPT2's integrals over density fitting: the MO blocks of the
/// fitted factor B that its classes read, the active h2e from them, and the
/// external four-index blocks built slab by slab, so that no whole external
/// block need ever exist on the device. And the adjoints of all three, onto
/// Lambda = dE/dB in the MO basis.
///
/// (pq|rs) = sum_Q B_Q,pq B_Q,rs with the metric already folded into B. No
/// (Q|vv) block: no SC class has two virtuals in one charge distribution,
/// which is what keeps the density-fitted variant cheap. The README has the
/// layouts and the (X, Y, axes) table every slab is built from.

export module free_breakfast.nevpt2.density_fitting;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast::nevpt2 {

/// @brief The external two-electron blocks of the eight SC classes, each in
/// the physicist layout PySCF's `Sxxx` takes from its `eris`.
///
/// c core, a (and p, t, u) active, v (and r, s, x, y) virtual, i, j core.
/// Every block is column-major, its first index fastest.
enum class ExtBlock {
  Sr,    ///< [r, a, b, c] = (rb|ac), v a a a, batched over r
  Si,    ///< [t, u, i, v] = (it|uv), a a c a, batched over i
  Sijrs, ///< [i, a, j, b] = (ia|jb), c v c v, batched over i
  Sijr,  ///< [r, p, j, i] = (rj|ip), v a c c, batched over r
  Srsi,  ///< [x, y, i, t] = (xi|yt), v v c a, batched over x
  SrsiT, ///< Srsi again, batched over y: [y, x, i, t] = (xt|yi)
  Srs,   ///< [r, s, t, u] = (rt|su), v v a a, batched over r
  Sij,   ///< [t, u, i, j] = (it|ju), a a c c, batched over i
  Sir1,  ///< [r, t, i, u] = (ri|tu), v a c a, batched over r
  Sir2,  ///< [r, u, t, i] = (rt|iu), v a a c, batched over r
};

/// @brief B in the MO basis over the blocks SC-NEVPT2 reads, each [Q][p][q]:
/// element (Q, p, q) at Q + naux (p + n_p q), as @ref b_mo writes it.
struct MoBlocks {
  /// @brief Fitting functions: the fast index of every block.
  int32_t naux = 0;
  /// @brief Core orbitals.
  int32_t ncore = 0;
  /// @brief Active orbitals.
  int32_t ncas = 0;
  /// @brief Virtual orbitals.
  int32_t nvirt = 0;
  /// @brief B_aa, naux x ncas x ncas.
  Buffers::device<double> aa;
  /// @brief B_ca, naux x ncore x ncas.
  Buffers::device<double> ca;
  /// @brief B_va, naux x nvirt x ncas.
  Buffers::device<double> va;
  /// @brief B_cv, naux x ncore x nvirt.
  Buffers::device<double> cv;
  /// @brief B_vc, naux x nvirt x ncore: B_cv with its orbital pair swapped,
  /// because a slab batches over the first orbital of its batched pair and
  /// several blocks batch over a virtual paired with a core.
  Buffers::device<double> vc;
};

/// @brief The MO blocks from the fitted factor @p b and the orbitals @p c:
/// four @ref b_mo calls (aa, ca, va, cv) and a permute of cv into vc, all on
/// res.stream(), with no host synchronization.
///
/// Columns [0, ncore) of @p c are the core, [ncore, ncore + ncas) the active
/// space and the rest, nmo - ncore - ncas, the virtuals. @p b is B as
/// @ref density_fit leaves it, naux x nao^2; @p c is nao x nmo, column-major
/// and packed. The blocks are allocated from res.device().
///
/// @return The blocks; InvalidConfig for a negative count or more core and
///         active orbitals than @p nmo; or @ref b_mo's error, which leaves the
///         blocks unspecified.
[[nodiscard]] Result<MoBlocks> mo_blocks(DeviceResources &res, int32_t naux, int32_t nao,
                                         int32_t nmo, const double *b, const double *c,
                                         int32_t ncore, int32_t ncas);

/// @brief The active h2e[t, u, v, w] = (tv|uw), physicist, ncas^4 doubles at
/// t + ncas (u + ncas (v + ncas w)), from B_aa: one gemm and a permute.
///
/// Queued on res.stream() with no host synchronization; the gemm's product is
/// scratch from res.device_workspace(), which @p out must not point into.
///
/// @return Success; InvalidConfig past the int32_t BLAS dispatch; or the
///         GpuRuntime error of the first call that failed.
[[nodiscard]] Status h2e(DeviceResources &res, const MoBlocks &blocks, double *out);

/// @brief The axis of @p block that a slab restricts: 0, except Si and Sij
/// (2) and SrsiT (1). Each class output element depends on exactly one index
/// along it.
[[nodiscard]] int32_t batch_axis(ExtBlock block);

/// @brief The whole of @p block's dimensions, in its physicist order; a slab's
/// are the same with its @ref batch_axis shortened to b1 - b0.
[[nodiscard]] std::array<int32_t, 4> block_shape(const MoBlocks &blocks, ExtBlock block);

/// @brief @p block with its @ref batch_axis restricted to [@p b0, @p b1), into
/// @p out: C[x, y, z, w] = sum_Q X_Q,xy Y_Q,zw over x in the batch, one
/// strided-batched gemm, then a permute of C into the block's physicist
/// layout (none for Sijrs, whose layout is C's).
///
/// @p out is the slab, column-major with @ref block_shape's dimensions and
/// the batch axis b1 - b0 long, overwritten. It must not overlap the blocks
/// or point into res.device_workspace(), which holds C when a permute
/// follows. Queued on res.stream() with no host synchronization.
///
/// @return Success, writing nothing for an empty slab; InvalidConfig for a
///         range outside [0, extent] or b1 < b0, or past the int32_t BLAS
///         dispatch; or the GpuRuntime error of the first call that failed,
///         which leaves @p out unspecified.
[[nodiscard]] Status slab(DeviceResources &res, const MoBlocks &blocks, ExtBlock block, int32_t b0,
                          int32_t b1, double *out);

// ---------------------------------------------------------------------------
// The adjoints: bars on the blocks, h2e and the slabs pulled back onto
// Lambda = dE/dB in the MO basis.
//
// Lambda is naux x nmo^2, nmo = ncore + ncas + nvirt, in the layout of B
// itself as b_mo reads it with C the identity: element (Q, p, q) at
// Q + naux (p + nmo q), p and q absolute MO indices. Every adjoint ADDS onto
// it, so a caller zeroes it once and walks the slabs one at a time, and no
// slab's bar need outlive its call. Lambda is the adjoint of the maps as
// written, which read B_cv's pair (and so B_vc's) from Lambda's (core,
// virtual) corner only: its symmetric part, for a symmetric B, is the
// caller's (Lambda + Lambda^T) / 2 per Q.
// ---------------------------------------------------------------------------

/// @brief The adjoint of @ref h2e at @p blocks' B: @p lambda +=
/// d<h2e_bar, h2e(B)>/dB, onto Lambda's active-active corner.
///
/// @p h2e_bar is ncas^4 doubles in @ref h2e's layout. Queued on res.stream()
/// with no host synchronization; the bar permuted back is scratch from
/// res.device_workspace(), which neither @p h2e_bar nor @p lambda may point
/// into.
///
/// @return Success; InvalidConfig past the int32_t BLAS dispatch; or the
///         GpuRuntime error of the first call that failed, which leaves
///         @p lambda unspecified.
[[nodiscard]] Status h2e_adjoint(DeviceResources &res, const MoBlocks &blocks,
                                 const double *h2e_bar, double *lambda);

/// @brief The adjoint of @ref slab at @p blocks' B: @p lambda +=
/// d<slab_bar, slab(B)>/dB, for @p block's slab over [@p b0, @p b1).
///
/// The transposed gemms of @ref slab's one, each strided-batched: with C-bar
/// the bar permuted back, X-bar_y += Y C-bar_y^T over the batch's columns of
/// X, and Y-bar_w += X_b C-bar_w, both added at X's and Y's places in Lambda.
/// @p slab_bar is the slab's own layout and size. Queued on res.stream()
/// with no host synchronization; C-bar (when a permute was taken) and the
/// batch's columns of X (when the batch is not the whole extent) are scratch
/// from res.device_workspace(), which neither @p slab_bar nor @p lambda may
/// point into.
///
/// @return Success, adding nothing for an empty slab or naux = 0;
///         InvalidConfig for a range outside [0, extent] or b1 < b0, or past
///         the int32_t BLAS dispatch (Lambda's leading dimension naux nmo
///         included); or the GpuRuntime error of the first call that failed,
///         which leaves @p lambda unspecified.
[[nodiscard]] Status slab_adjoint(DeviceResources &res, const MoBlocks &blocks, ExtBlock block,
                                  int32_t b0, int32_t b1, const double *slab_bar, double *lambda);

/// @brief The adjoint of @ref mo_blocks in the MO basis: each bar in @p bar
/// added at its block's place in @p lambda (B_vc's swapped back onto B_cv's),
/// and the bars on B_cc's and B_vv's diagonals, B_Q,ii and B_Q,aa, onto
/// Lambda's.
///
/// Those diagonals are what the orbital energies and h1e_v_sr read beyond
/// the five blocks ((ii|jj), (aa|tu), ...), which @ref mo_blocks never
/// forms. @p bar's counts give the shapes; any of its five buffers may be
/// empty, adding nothing, and so may either diagonal bar be null.
/// @p core_diagonal_bar is naux x ncore, element (Q, i) at Q + naux i, and
/// @p virt_diagonal_bar naux x nvirt the same. Queued on res.stream() with
/// no host synchronization; B_vc's bar swapped is scratch from
/// res.device_workspace(), which no argument may point into.
///
/// @return Success; InvalidConfig for a negative count, a buffer neither
///         empty nor its block's size, or Lambda's diagonal stride
///         naux (nmo + 1) past int32_t; or the GpuRuntime error of the first
///         call that failed, which leaves @p lambda unspecified.
[[nodiscard]] Status mo_blocks_adjoint(DeviceResources &res, const MoBlocks &bar,
                                       const double *core_diagonal_bar,
                                       const double *virt_diagonal_bar, double *lambda);

} // namespace free_breakfast::nevpt2
