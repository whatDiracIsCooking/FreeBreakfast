/// @file
/// @brief The hand-written device steps the generated kb kernels leave to
/// their launcher: max|D|, the Dcd gather, the block table and its factors,
/// the shell starts and mirror that complete the sweep's K, and the E tables
/// the rolled kernel (kbr) reads, declared once for the module that calls them
/// and the device TU that defines them.
///
/// Included by the exchange module's implementation unit in its GLOBAL MODULE
/// FRAGMENT, and by each of this directory's .cu files directly, for the reason
/// overlap_bridge.h gives: a declaration in a module's purview gets module
/// linkage and can never bind to a definition compiled in a plain TU.
/// wwrStream_t and wwrError_t come from a different wwr header in each pass, as
/// there. test/gtest/exchange_helpers_gpu.cpp includes it too, and so imports
/// nothing that carries wwr.runtime_api (its header says why).
///
/// Every pointer is device memory, every launch is asynchronous on @p stream,
/// and every matrix is n_ao x n_ao Cartesian, indexed `m * n_ao + n`. The
/// launchers return launch errors only, as overlap_class.
///
/// A shell's RANK is its index among the basis's shells of the same angular
/// momentum, in basis order: the p shells of a basis are ranks 0, 1, ... in the
/// order they appear, whatever s or d shells lie between them. It depends only
/// on the basis, so the shell offsets by momentum the gather takes are built
/// once per basis, and a record's `dc` / `dd` (kb's row of Dcd) must be
/// formed from the same ranks.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief K's L limit: the largest momentum of a shell in any class the K
/// build takes, f. Its kernels are emitted through it
/// (tools/emit_exchange_kernels.py's MAX_L), and exchange_launch.cu checks it
/// against their tables, which a host module unit cannot include; the tree-wide
/// kMaxAngularMomentum is not K's.
inline constexpr int32_t kExchangeMaxMomentum = 3;

/// @brief The largest momentum K's unrolled kernel, kb, is emitted at, d: a
/// class with any shell above it is launched as the rolled kbr, which reads
/// the records' E tables (@ref exchange_e_table), so the ordered classes make
/// those only for a basis that reaches past it. exchange_launch.cu checks it
/// against kb's table. FelizJueves's rule, through d kb, at f kbr (#580): kb at
/// f is unbuildable (#569).
inline constexpr int32_t kExchangeKbMaxMomentum = 2;

/// @brief The int32s per block of a kb launch's table: bra0, bra1, ket0, ket1,
/// ko_a, ko_b, each a segment of n_blk.
inline constexpr int32_t kExchangeBlockFields = 6;

/// @brief The doubles of the DENSITY DECAY (FelizJueves's ao.k.density_decay,
/// #484): max|D|, then Λ and H_off, in that order. One per call: max|D| from
/// @ref exchange_density_max, the other two from @ref exchange_density_decay.
inline constexpr int32_t kExchangeDecayFields = 3;

/// @brief The doubles per group of a class's GEOMETRY: its first shell's
/// centre x, y and z, then its REACH R, each a segment of n_groups. R is the
/// largest |AC| over the group's records (a, c) a kb screen at tau can visit,
/// or negative when it visits none (exchange_group_geometry,
/// exchange_bridge.h).
inline constexpr int32_t kExchangeGroupFields = 4;

/// @brief Rounding room in the density-decay factor, only ever loosening it:
/// it widens the reach's cutoff and raises the factor, against the few-ulp
/// rounding of the products, distances and exponentials. FelizJueves's
/// DECAY_MARGIN.
inline constexpr double kExchangeDecayMargin = 0x1p-40;

/// @brief Two AO centres closer than this, in bohr, are ONE centre to the
/// density decay: a centre is recomputed from a record as P - (P - A), which
/// may miss A by an ulp, so equality would split an atom's AOs into
/// "different" centres a few 1e-16 apart. No two atoms are this close.
inline constexpr double kExchangeSameCentre = 1e-6;

/// @brief Write max |D| over the n_ao x n_ao @p density into the one double
/// @p dmax, on @p stream: the decay's first field, the density factor kb's
/// screen `Ip * Iq * dmax[blk] < tau` falls back to.
///
/// One block folds the whole matrix, so the result is exact and needs no
/// zeroing first; a NaN anywhere is the result.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_ao is not
///         positive).
[[nodiscard]] wwr::wwrError_t exchange_density_max(wwr::wwrStream_t stream, int32_t n_ao,
                                                   const double *density, double *dmax);

/// @brief Fold the largest of @p n_rec record bounds into @p g_max, on
/// @p stream: `*g_max = max(*g_max, max_r bound[r])`, atomically, so every
/// class of a basis can fold into one @p g_max from its own stream. Zero
/// @p g_max first. The bounds must be non-negative and not NaN, as the ordered
/// classes' record bounds are: the atomic max compares their bits.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_rec is not
///         positive).
[[nodiscard]] wwr::wwrError_t exchange_bound_max(wwr::wwrStream_t stream, int32_t n_rec,
                                                 const double *bound, double *g_max);

/// @brief Write the centre of every AO of a class's first shells into
/// @p centre (three segments of @p n_ao, x, y and z), on @p stream: for each
/// of its @p n_groups groups (@p group_start, OrderedPairClassView's), its
/// first record's P - (P - A), at that record's @p off_a and the @p nx - 1
/// Cartesians after it.
///
/// Given a class (l, l), which holds every shell of momentum l as a first
/// shell, one launch per momentum writes @p centre whole. One thread per
/// group, each reading one record, so the centres are the same every call.
/// @p p and @p pa are the records' P and P - A, three segments of their count
/// each (x, y, z).
///
/// @return The launch status (wwrErrorInvalidValue if @p n_groups, @p nx or
///         @p n_ao is not positive).
[[nodiscard]] wwr::wwrError_t exchange_ao_centres(wwr::wwrStream_t stream, int32_t n_groups,
                                                  const int32_t *group_start, int32_t nx,
                                                  const int32_t *off_a, const double *const p[3],
                                                  const double *const pa[3], int32_t n_ao,
                                                  double *centre);

/// @brief Write a class's group GEOMETRY (@ref kExchangeGroupFields) into
/// @p geom, on @p stream: per group of its @p n_groups (@p group_start), its
/// first record's P - (P - A) as the shell's centre, and its REACH
///
///     R = max |AC| over the group's records with bound * (g_max * max|D|) >= tau (1 - margin)
///
/// or -1 when none passes, |AC| the length of (P - C) - (P - A), @p pa and
/// @p pb the records' P - A and P - C as @p p is their P (three segments of
/// the record count each). A record under that cutoff is never visited by a
/// kb screen at @p tau under any block factor at most max|D|: its every
/// product is under tau, so its group's sorted `break` fires first. So the
/// cutoff is the screen's own, not a pair list's: mmd screens no pairs, and
/// over all of a group's records R would be the farthest shell in the
/// molecule (FelizJueves #484, decision 4). @p g_max is @ref
/// exchange_bound_max's over every class, @p decay the call's (its max|D|
/// read).
///
/// @return The launch status (wwrErrorInvalidValue if @p n_groups is not
///         positive).
[[nodiscard]] wwr::wwrError_t
exchange_group_geometry(wwr::wwrStream_t stream, int32_t n_groups, const int32_t *group_start,
                        const double *bound, const double *const p[3], const double *const pa[3],
                        const double *const pb[3], double tau, const double *g_max,
                        const double *decay, double *geom);

/// @brief Fit the density's decay with distance, on @p stream: write Λ and
/// H_off into @p decay[1] and @p decay[2] (@ref kExchangeDecayFields).
///
/// Over the OFF-CENTRE elements D_mn, those whose AO centres are at least
/// @ref kExchangeSameCentre apart (@p centre is three segments of @p n_ao, x,
/// y and z, the centre of each AO), at r = |MN|:
///
///     Λ      the least-squares slope of log(max |D_mn| per 1-bohr bin of r),
///            negated and clamped to >= 0; 0 under two non-empty bins
///     H_off  max |D_mn| e^{Λ r}, an exact reduction
///
/// so `|D_mn| <= H_off e^{-Λ r}` holds for every off-centre element whatever
/// the fit, which moves only how tight it is. Bins past the last (1024 bohr)
/// fold into it. One block folds the whole matrix twice, so the result is
/// deterministic and needs no zeroing first.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_ao is not
///         positive).
[[nodiscard]] wwr::wwrError_t exchange_density_decay(wwr::wwrStream_t stream, int32_t n_ao,
                                                     const double *density, const double *centre,
                                                     double *decay);

/// @brief Gather the (lc, ld) shell blocks of the n_ao x n_ao @p density into
/// kb's @p dcd, on @p stream.
///
/// @p off_c holds the AO offset of each of the @p n_c shells of momentum @p lc,
/// by rank, and @p off_d those of the @p n_d shells of momentum @p ld. With
/// n_x = (l_x + 1)(l_x + 2) / 2 Cartesians per shell, @p dcd is
/// (n_c n_d) x (n_x(lc) n_x(ld)), row-major and written whole:
///
///     dcd[rank_c * n_d + rank_d][i * n_x(ld) + j] = D[off_c[rank_c] + i][off_d[rank_d] + j]
///
/// so the kernel's `dc[bra] + dd[ket]` names the (c, d) block when
/// dc = rank_c * n_d and dd = rank_d. Nothing is mirrored: (lc, ld) and
/// (ld, lc) are two gathers.
///
/// @return The launch status (wwrErrorInvalidValue if @p lc or @p ld is
///         outside [0, @ref kExchangeMaxMomentum], @p n_ao, @p n_c or @p n_d is not
///         positive, or @p dcd has more than INT32_MAX elements).
[[nodiscard]] wwr::wwrError_t exchange_gather_dcd(wwr::wwrStream_t stream, int32_t lc, int32_t ld,
                                                  int32_t n_ao, const double *density, int32_t n_c,
                                                  const int32_t *off_c, int32_t n_d,
                                                  const int32_t *off_d, double *dcd);

/// @brief Write the block table of one kb launch, a bra class against a ket
/// class, into @p blocks, on @p stream: one block per (bra group, ket group)
/// whose bra shell is at most the ket shell, the a <= b blocks K's sweep
/// computes (exchange.cppm). It depends only on the basis, so the ordered
/// classes write each bra's tables once (ordered_pair_classes).
///
/// Each side is given by its groups by first shell: @p n_a bra groups, group g
/// the bra records [@p a_start[g], @p a_start[g + 1]), and @p a_off the bra
/// records' first-shell AO offset (OrderedPairClass::off_a); likewise @p n_b,
/// @p b_start and @p b_off for the kets. The groups of each side must be in
/// basis order, so AO offsets rise with them and `a <= b` is
/// `a_off[a_start[ga]] <= b_off[b_start[gb]]`.
///
/// @p blocks is @ref kExchangeBlockFields segments of @p n_blk int32s, kb's
/// bra0, bra1, ket0, ket1, ko_a and ko_b in that order, written whole: block
/// blk is bra group ga against ket group gb, bra-major, the ket groups of one
/// bra group ascending:
///
///     bra0 = a_start[ga], bra1 = a_start[ga + 1], ko_a = a_off[bra0]
///     ket0 = b_start[gb], ket1 = b_start[gb + 1], ko_b = b_off[ket0]
///
/// @p n_blk must be the number of blocks so chosen (the host counts it from
/// the basis; nothing here checks it, and a smaller one is written out of
/// bounds).
///
/// @return The launch status (wwrErrorInvalidValue if @p n_a, @p n_b or
///         @p n_blk is not positive, or the table has more than INT32_MAX
///         elements).
[[nodiscard]] wwr::wwrError_t exchange_blocks(wwr::wwrStream_t stream, int32_t n_a,
                                              const int32_t *a_start, const int32_t *a_off,
                                              int32_t n_b, const int32_t *b_start,
                                              const int32_t *b_off, int32_t n_blk, int32_t *blocks);

/// @brief Write the density factor of every block of @ref exchange_blocks's
/// table for the same groups into @p factor (@p n_blk doubles, in the table's
/// order, written whole), on @p stream: kb's `dmax`, each block's F_ab, a
/// bound on every |D_cd| the block can visit (arXiv:2407.21445 §4.3.1;
/// FelizJueves #484). It depends on the density, so the K build writes it
/// per call.
///
/// With @p a_geom and @p b_geom the two classes' group geometries
/// (@ref kExchangeGroupFields), A and B the two groups' centres and R_a and
/// R_b their reaches, and @p decay the call's (@ref kExchangeDecayFields),
/// d = |AB| - R_a - R_b and
///
///     F_ab = max|D|                                  if d <= 2 kExchangeSameCentre
///                                                    or either R is negative
///     F_ab = min(max|D|, H_off e^{-Λ d} (1 + kExchangeDecayMargin))   otherwise
///
/// A visited (c, d) has |CD| >= d, so past twice the same-centre distance
/// every one is off-centre, and H_off's bound covers it. FelizJueves's cut is
/// d <= 0; the margin is what its exact centres do not need.
///
/// @return The launch status, as @ref exchange_blocks.
[[nodiscard]] wwr::wwrError_t
exchange_factors(wwr::wwrStream_t stream, int32_t n_a, const int32_t *a_start, const int32_t *a_off,
                 const double *a_geom, int32_t n_b, const int32_t *b_start, const int32_t *b_off,
                 const double *b_geom, const double *decay, int32_t n_blk, double *factor);

/// @brief Write the first AO of its shell for every AO of the @p n_s shells of
/// momentum @p l, `start[off[r] + i] = off[r]` for i < n_x(l), on @p stream.
///
/// @p off is the momentum's AO offsets by rank (the gather's `off_c`); one
/// launch per momentum the basis reaches writes the n_ao entries of @p start
/// whole, for @ref exchange_mirror.
///
/// @return The launch status (wwrErrorInvalidValue if @p l is outside
///         [0, @ref kExchangeMaxMomentum] or @p n_s is not positive).
[[nodiscard]] wwr::wwrError_t exchange_shell_starts(wwr::wwrStream_t stream, int32_t l, int32_t n_s,
                                                    const int32_t *off, int32_t *start);

/// @brief Complete the sweep's @p k from its a <= b shell blocks, on
/// @p stream: every element of a lower shell block, whose column is in an
/// earlier shell than its row, is set to its transpose times @p sign,
///
///     k[m][q] = sign k[q][m]   for q < start[m]
///
/// @p sign is 1 for a symmetric density's K, which is symmetric, and -1 for an
/// antisymmetric one's, which is antisymmetric (exchange.cppm).
/// @p start the first AO of each AO's shell (@ref exchange_shell_starts). The
/// a == b blocks, computed whole, are left as they are. It sets, never adds:
/// the lower blocks' contents are overwritten, and every kb launch into the
/// upper ones must be ordered before it.
///
/// @return The launch status (wwrErrorInvalidValue if @p n_ao is not
///         positive or @p sign is neither 1 nor -1).
[[nodiscard]] wwr::wwrError_t exchange_mirror(wwr::wwrStream_t stream, int32_t n_ao,
                                              const int32_t *start, double sign, double *k);

/// @brief The doubles per record of @ref exchange_e_table's table for a class
/// (@p la, @p lb): the nonzero entries of one record's E^{la lb} block,
///
///     SUM_{a in cartesians(la), b in cartesians(lb)} PRODUCT_k (a_k + b_k + 1)
///
/// since E^{ab}_t is structurally zero exactly when some t_k > a_k + b_k, and
/// every other t has |t| <= la + lb. FelizJueves's
/// `len(launch_kblock.e_table(la, lb, ...))`, kbr's n_eac (bra) or n_ebd
/// (ket). Zero for a negative momentum.
inline constexpr int32_t exchange_e_table_width(const int32_t la, const int32_t lb) {
  int32_t width = 0;
  for (int32_t ax = la; ax >= 0; --ax) {
    for (int32_t ay = la - ax; ay >= 0; --ay) {
      for (int32_t bx = lb; bx >= 0; --bx) {
        for (int32_t by = lb - bx; by >= 0; --by) {
          width += (ax + bx + 1) * (ay + by + 1) * (la - ax - ay + lb - bx - by + 1);
        }
      }
    }
  }
  return width;
}

/// @brief Write the UNDRESSED Hermite expansion E of each of @p n_rec records
/// of a class (@p la, @p lb) into @p e, on @p stream: kbr's `Eac` (a bra
/// class) or `Ebd` (a ket class, @p ket), FelizJueves's
/// `launch_kblock.e_table(la, lb, zeta, P, A, B, sign=ket)` of every record.
///
/// @p e is n_rec x @ref exchange_e_table_width(la, lb), record-major (record r
/// at `r * width`, in the records' own order) and written whole. Within a
/// record the rows are the Cartesian pairs (a, b), a in cartesians(la) outer
/// (x descending, then y descending), b in cartesians(lb) inner; within a row
/// the Hermite t in |t|-ascending, then t_x, then t_y descending order,
/// structural zeros (some t_k > a_k + b_k) dropped. Each entry is
///
///     E^{ab}_t = E^{a_x b_x}_{t_x} E^{a_y b_y}_{t_y} E^{a_z b_z}_{t_z}
///
/// per axis by McMurchie and Davidson's recurrence from
/// E^{00}_0 = exp(zeta X_PA X_PB), FelizJueves's mmd.hermite.e_axis, operation
/// for operation (the device may fuse a product into an add). No contraction
/// coefficient, no (pi / zeta)^{3/2}: those are the record's s, which kbr
/// applies. With @p ket, every entry of odd |t| is negated, the (-1)^|q| of
/// [p|q] = (-1)^|q| [p + q]: exact, so a ket's table is a bra's with those
/// entries' signs flipped.
///
/// @p zeta is the records' zeta, and @p pa and @p pb their P - A and P - B,
/// "a" the first shell (OrderedPairClassView's fields), three segments each
/// (x, y, z). It depends only on the records, so it is made once per basis.
///
/// @return The launch status (wwrErrorInvalidValue if @p la or @p lb is
///         outside [0, 3], through f, @p n_rec is not positive, or @p e has
///         more than INT32_MAX elements).
[[nodiscard]] wwr::wwrError_t exchange_e_table(wwr::wwrStream_t stream, int32_t la, int32_t lb,
                                               int32_t n_rec, const double *zeta,
                                               const double *const pa[3], const double *const pb[3],
                                               bool ket, double *e);

} // namespace free_breakfast::device
