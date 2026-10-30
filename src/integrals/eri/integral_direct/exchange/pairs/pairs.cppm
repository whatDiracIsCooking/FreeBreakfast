/// @file
/// @brief Ordered primitive-pair classes: mmd's la <= lb records in both
/// orientations, one class per ordered (la, lb), grouped by first shell, for
/// the exchange build.
///
/// K_ab = sum_cd D_cd (ac|bd) has no symmetry inside a pair: kb's bra
/// (a, c) and ket (b, d) are ordered shell pairs whose first shell is the
/// output index. So each stored off-diagonal mmd record contributes both
/// orientations, (a, b) and (b, a), and a diagonal one (a shell against
/// itself) once.
///
/// The launchers' view of a class is ordered_pair_class_view.h's, included in
/// the global module fragment and re-exported (atom.cppm has the why).

module;

#include "ordered_pair_class_view.h"

export module free_breakfast.integrals.eri.integral_direct.exchange.pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

using free_breakfast::OrderedPairClassView;

/// @brief Every ordered primitive-pair record of one class (la, lb), in either
/// order of la and lb, one @ref n_rec long device buffer per field, all on the
/// class's own work stream; its groups by first shell, as their bounds on the
/// device; and what the K build reads of it as a bra, and writes per build.
///
/// The fields are mmd's, named as there, with "a" the record's first shell and
/// "b" its second: a record taken from mmd's (b, a) is stored with its P - A
/// and P - B, its two exponents and its two AO offsets swapped; zeta, P and s
/// are orientation-free and copied as they are. The exponents @ref ea and
/// @ref eb are read by the gradient's gk alone (kb and kbr read neither).
/// mmd's undressed `bound` is not carried: K does not read it. @ref record_bound is mmd's dressed bound of the orientation
/// (bound_ab as stored, bound_ba swapped), and @ref rank_b the second shell's
/// rank (mmd's rank_b as stored, rank_a swapped).
///
/// **The records are grouped by first shell**, the groups in basis order, and
/// each group sorted by @ref record_bound, non-increasing: the order kb's
/// screen `break`s need of its bras and kets.
///
/// **The classes are a set.** A bra's block tables are made against every
/// class of its @ref ordered_pair_classes call, so the K build takes the whole
/// set, in its order, and refuses anything else (@ref n_blk is how it tells).
///
/// **The K build's scratch is the class's, not the build's** (as J's is
/// @c CoulombPairClass's). What depends only on the basis is written once, by
/// @ref ordered_pair_classes: as a bra, its Dcd rows per ld (@ref dc) and its
/// kb block tables (@ref blocks). Every @c exchange call rewrites the rest:
/// what depends on the density, @ref dcd, @ref geom and @ref factor, and the
/// tables the build reads across classes, a class (l, l)'s @ref rank_offset
/// and the set's own scratch, which its first class holds (@ref centre,
/// @ref shell_start, @ref decay and @ref g_max). So an SCF loop allocates
/// none of it per build, and the K build allows one @c exchange or
/// @c exchange_block_factors at a time per set of classes: two builds over
/// the same classes must be queued on one stream, which orders them.
///
/// **Device only.** There is no host copy of a record: a reader copies a
/// field back itself, on @ref device_stream()'s stream or after it.
///
/// Movable, not copyable.
struct OrderedPairClass {
  /// @brief The first shell's angular momentum.
  int32_t la;
  /// @brief The second shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive: the length of every record field.
  int32_t n_rec;
  /// @brief The number of groups by first shell, positive: one per shell of
  /// momentum @ref la, in basis order.
  int32_t n_groups;
  /// @brief The basis's Cartesian AO count: the order of the density and the
  /// K the class serves.
  int32_t n_ao;
  /// @brief The basis's shell count per angular momentum, ascending from s, at
  /// least kExchangeMaxMomentum + 1 entries (0 for a momentum it does not
  /// reach): the extents of @ref dc and @ref dcd.
  std::vector<int32_t> n_shells;
  /// @brief The blocks of this class's block table as a bra against each
  /// class of its set as a ket, in the set's order: one per (a, b), a a first
  /// shell of this class and b one of the ket's, with a <= b in basis order.
  /// So its size is the set's class count. Each is at most INT32_MAX / 6.
  std::vector<int32_t> n_blk;
  /// @brief The class's own work stream: everything was allocated and written
  /// on it. Read through @ref device_stream().
  std::shared_ptr<DeviceStream> work_stream;
  /// @brief zeta = alpha + beta.
  StreamBuffers::device<double> zeta;
  /// @brief P, the product centre: x.
  StreamBuffers::device<double> p_x;
  /// @brief P: y.
  StreamBuffers::device<double> p_y;
  /// @brief P: z.
  StreamBuffers::device<double> p_z;
  /// @brief P - A, A the first shell's centre: x.
  StreamBuffers::device<double> pa_x;
  /// @brief P - A: y.
  StreamBuffers::device<double> pa_y;
  /// @brief P - A: z.
  StreamBuffers::device<double> pa_z;
  /// @brief P - B, B the second shell's centre: x.
  StreamBuffers::device<double> pb_x;
  /// @brief P - B: y.
  StreamBuffers::device<double> pb_y;
  /// @brief P - B: z.
  StreamBuffers::device<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  StreamBuffers::device<double> s;
  /// @brief alpha, the first shell's primitive exponent (mmd's ea as stored,
  /// eb swapped).
  StreamBuffers::device<double> ea;
  /// @brief beta, the second shell's primitive exponent.
  StreamBuffers::device<double> eb;
  /// @brief The record's DRESSED Schwarz bound, FelizJueves's
  /// ao.k.record_bounds: `max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p|
  /// I_p(zeta)`, the max over the first shell's Cartesians, so it depends on
  /// the orientation: mmd's @c PrimitivePairClass::bound_ab as stored,
  /// @c PrimitivePairClass::bound_ba swapped. kb's Ip as a bra and Iq as a
  /// ket, and what the K build's reach and bound maximum read. Non-increasing
  /// within each group; exactly 0.0 for a pair whose pair factor underflows.
  /// Zero in a class above f, which has no kernel and which the K build
  /// refuses.
  StreamBuffers::device<double> record_bound;
  /// @brief The first shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  StreamBuffers::device<int32_t> off_a;
  /// @brief The second shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_b;
  /// @brief The second shell's rank: its index among the basis's shells of
  /// momentum @ref lb, in basis order. kb's Dcd row is
  /// `rank_c * n_shells(ld) + rank_d` (exchange_helpers_bridge.h), so a bra's
  /// `dc` and a ket's `dd` are formed from it.
  StreamBuffers::device<int32_t> rank_b;
  /// @brief The groups' bounds, @ref n_groups + 1 entries, covering the
  /// records exactly: group g is records [group_start[g], group_start[g + 1]),
  /// its first shell the one of rank g among the basis's shells of momentum
  /// @ref la, and the last entry n_rec. The block tables are made from it.
  StreamBuffers::device<int32_t> group_start;
  /// @brief As a bra (a, c), its Dcd rows for each ld, n_shells.size() x
  /// n_rec, row ld at `ld * n_rec`: `rank_b[r] * n_shells[ld]`, kb's `dc`
  /// (ordered_pair_bra_rows). Written once; a row of a momentum the basis
  /// does not reach is never written or read.
  StreamBuffers::device<int32_t> dc;
  /// @brief As a bra, its kb block tables (exchange_blocks), one per ket class
  /// of the set in its order, each kExchangeBlockFields x @ref n_blk segments
  /// (bra0, bra1, ket0, ket1, ko_a, ko_b), the tables back to back. Written
  /// once.
  StreamBuffers::device<int32_t> blocks;
  /// @brief Scratch: as the density class (lc, ld) = (la, lb), Dcd, the
  /// (la, lb) shell blocks of the density, (n_shells[la] n_shells[lb]) x
  /// (n_x(la) n_x(lb)) (exchange_gather_dcd).
  StreamBuffers::device<double> dcd;
  /// @brief Scratch: the group geometry, kExchangeGroupFields x @ref n_groups:
  /// each first shell's centre and its reach at the build's tau and density
  /// (exchange_group_geometry).
  StreamBuffers::device<double> geom;
  /// @brief The records' undressed E table as a bra (exchange_e_table): kbr's
  /// Eac, @ref n_rec x exchange_e_table_width(la, lb), record-major. Written
  /// once, and only when the set reaches past kb, a basis with a shell above
  /// kExchangeKbMaxMomentum, since then every class pairs with a kbr class;
  /// otherwise, or in a class above kExchangeMaxMomentum (which the K build
  /// refuses), empty, so the view's pointer is null.
  StreamBuffers::device<double> e_bra;
  /// @brief As @ref e_bra, with every entry of odd |t| negated: kbr's Ebd as a
  /// ket. Written and empty with it.
  StreamBuffers::device<double> e_ket;
  /// @brief Scratch: as a bra, each block's density factor F_ab
  /// (exchange_factors), beside @ref blocks: one per block, the tables'
  /// factors back to back in the same order.
  StreamBuffers::device<double> factor;
  /// @brief Scratch of a class (l, l), which names every shell of momentum l:
  /// the first Cartesian AO of each of those shells, by rank, n_shells[l]
  /// entries (exchange_rank_offsets). Empty in a class with la != lb.
  StreamBuffers::device<int32_t> rank_offset;
  /// @brief Scratch of the set's first class, the set's own: every AO's
  /// centre, 3 x n_ao, x then y then z (exchange_ao_centres). Empty in every
  /// other class.
  StreamBuffers::device<double> centre;
  /// @brief Scratch of the set's first class: the first AO of each AO's
  /// shell, n_ao entries (exchange_shell_starts), for the mirror. Empty in
  /// every other class.
  StreamBuffers::device<int32_t> shell_start;
  /// @brief Scratch of the set's first class: the density's decay,
  /// kExchangeDecayFields doubles (max|D|, Λ, H_off). Empty in every other
  /// class.
  StreamBuffers::device<double> decay;
  /// @brief Scratch of the set's first class: G_max, the largest record bound
  /// over the set, one double. Empty in every other class.
  StreamBuffers::device<double> g_max;

  /// @brief The class's own work stream (@ref work_stream), as mmd's classes
  /// name theirs, for the builds' fork and join.
  const std::shared_ptr<DeviceStream> &device_stream() const noexcept { return work_stream; }

  /// @brief The records on the device by pointer, for a launcher. Valid while
  /// this is alive.
  [[nodiscard]] OrderedPairClassView view() const {
    return {
        .la = la,
        .lb = lb,
        .n_rec = n_rec,
        .zeta = zeta.data(),
        .p_x = p_x.data(),
        .p_y = p_y.data(),
        .p_z = p_z.data(),
        .pa_x = pa_x.data(),
        .pa_y = pa_y.data(),
        .pa_z = pa_z.data(),
        .pb_x = pb_x.data(),
        .pb_y = pb_y.data(),
        .pb_z = pb_z.data(),
        .s = s.data(),
        .ea = ea.data(),
        .eb = eb.data(),
        .record_bound = record_bound.data(),
        .off_a = off_a.data(),
        .off_b = off_b.data(),
        .rank_b = rank_b.data(),
        .n_groups = n_groups,
        .group_start = group_start.data(),
        .e_bra = e_bra.data(),
        .e_ket = e_ket.data(),
    };
  }
};

/// @brief The ordered classes of @p classes, which are
/// @c primitive_pair_classes(res, basis): one per ordered (la, lb) reached, in
/// ascending (la, lb) order, each on a new @c DeviceStream of res.device().
///
/// Each record of mmd's class (la, lb) goes to the ordered (la, lb) as stored
/// and, unless it is diagonal (off_a == off_b), to the ordered (lb, la)
/// swapped; for la == lb both land in the one class. So a class holds one
/// record per primitive pair of every ordered shell pair (A, B) of its
/// momenta, a shell against itself once. Only reached classes appear, so none
/// is empty.
///
/// The groups are @p basis's: one per shell of momentum la, in basis order,
/// of nprim(a) times the primitives of every lb shell; so are
/// @ref OrderedPairClass::n_shells, @ref OrderedPairClass::n_ao and every
/// table's block count, @ref OrderedPairClass::n_blk. The rest is on the
/// device, on each class's stream, once per basis: the order that groups the
/// mmd class's orientations by first shell and sorts each group by its
/// dressed bound, non-increasing, equal bounds in mmd's record order (two
/// stable device sorts); @ref OrderedPairClass::group_start from it; every
/// field written in that order straight from the mmd class, and from those
/// its E tables when the basis has a shell above kb's momenta
/// (@ref OrderedPairClass::e_bra); then, once every class is gathered, each class's Dcd rows and its block tables against
/// every class. **kb's screen `break`s depend on that order**: an unsorted
/// group is a wrong K, not a slow one. The K build's per-build scratch is
/// allocated here too, and left unwritten, the set's own on its first class.
///
/// No host synchronization: each class's stream waits on an event recorded on
/// its mmd class's stream, so it reads mmd's final records. A bra's tables
/// read every class's groups, so every class's stream is joined into
/// res.stream() after its gather and forked back from it for its tables, and
/// joined and forked once more after them, so a class's frees follow every
/// table that read it. res.stream() then waits on every gather and table, and
/// each mmd class's stream waits on res.stream(), so a free queued there after
/// the return (the mmd class's destruction) does not overtake a gather. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored. A class of more than INT32_MAX records logs and aborts.
///
/// @return The ordered classes (none for no @p classes), the GpuRuntime error
///         of the first event or launch that failed, or an InvalidConfig error
///         if @p classes are not those of @p basis (a class missing, repeated,
///         not la <= lb, or of a record count not the basis's) or a block
///         table would pass the kernels' 32-bit index (6 int32s per block
///         above INT32_MAX).
[[nodiscard]] Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes);

} // namespace free_breakfast
