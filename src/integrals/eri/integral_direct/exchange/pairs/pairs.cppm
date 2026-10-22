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

/// @brief The records of one ordered class whose first shell is one shell: a
/// contiguous run of an @ref OrderedPairClass, which a K block takes whole
/// (its bra or its ket range).
struct OrderedPairGroup {
  /// @brief The first shell, as an index into @c BasisSet::shells().
  int32_t shell;
  /// @brief The first shell's first Cartesian AO (@c BasisSet::ao_offset_cart),
  /// every record's @ref OrderedPairClass::off_a.
  int32_t ao_offset;
  /// @brief The group's first record.
  int32_t start;
  /// @brief The group's record count, positive.
  int32_t count;
};

/// @brief Every ordered primitive-pair record of one class (la, lb), in either
/// order of la and lb, one @ref n_rec long device buffer per field, all on the
/// class's own work stream, and its groups by first shell on the host and,
/// as their bounds, on the device.
///
/// The fields are mmd's, named as there, with "a" the record's first shell and
/// "b" its second: a record taken from mmd's (b, a) is stored with its P - A
/// and P - B, and its two AO offsets, swapped; zeta, P and s are
/// orientation-free and copied as they are. mmd's exponents `ea` and `eb` are
/// not carried (kb reads neither), nor is mmd's undressed `bound`, which K
/// does not read. @ref record_bound is mmd's dressed bound of the orientation
/// (bound_ab as stored, bound_ba swapped), and @ref rank_b the second shell's
/// rank (mmd's rank_b as stored, rank_a swapped).
///
/// **The records are grouped by first shell**, the groups in basis order, and
/// each group sorted by @ref record_bound, non-increasing: the order kb's
/// screen `break`s need of its bras and kets.
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
  /// @brief The number of records, positive: the length of every field.
  int32_t n_rec;
  /// @brief The class's own work stream: the records were allocated and
  /// written on it. Read through @ref device_stream().
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
  /// @brief The record's DRESSED Schwarz bound, FelizJueves's
  /// ao.k.record_bounds: `max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p|
  /// I_p(zeta)`, the max over the first shell's Cartesians, so it depends on
  /// the orientation: mmd's @c PrimitivePairClass::bound_ab as stored,
  /// @c PrimitivePairClass::bound_ba swapped. kb's Ip as a bra and Iq as a
  /// ket, and what the K build's reach and bound maximum read. Non-increasing
  /// within each group; exactly 0.0 for a pair whose pair factor underflows.
  /// Zero in a class above d, which has no kernel and which the K build
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
  /// @brief The groups by first shell, in basis order, covering the records
  /// exactly: group g is records [start, start + count), its first shell the
  /// one of rank g among the basis's shells of momentum @ref la.
  std::vector<OrderedPairGroup> groups;
  /// @brief The groups' bounds on the device, groups.size() + 1 entries: group
  /// g is records [group_start[g], group_start[g + 1]), the last entry n_rec.
  /// The K build makes its block table from it on the device.
  StreamBuffers::device<int32_t> group_start;

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
        .record_bound = record_bound.data(),
        .off_a = off_a.data(),
        .off_b = off_b.data(),
        .rank_b = rank_b.data(),
        .n_groups = static_cast<int32_t>(groups.size()),
        .group_start = group_start.data(),
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
/// The groups are the host's, from @p basis alone: one per shell of momentum
/// la, in basis order, of nprim(a) times the primitives of every lb shell.
/// The rest is on the device, on each class's stream, once per basis: the
/// order that groups the mmd class's orientations by first shell and sorts
/// each group by its dressed bound, non-increasing, equal bounds in mmd's
/// record order (two stable device sorts); @ref OrderedPairClass::group_start
/// from it; and every field written in that order straight from the mmd
/// class. **kb's screen `break`s depend on that order**: an unsorted group is
/// a wrong K, not a slow one.
///
/// No host synchronization: each class's stream waits on an event recorded on
/// its mmd class's stream, so it reads mmd's final records, and the mmd
/// class's stream then waits on the class's gather, so a free queued there
/// after the return (the mmd class's destruction) does not overtake it. The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored. A class of more than INT32_MAX records logs and aborts.
///
/// @return The ordered classes (none for no @p classes), the GpuRuntime error
///         of the first event or launch that failed, or an InvalidConfig error
///         if @p classes are not those of @p basis: a class missing, repeated,
///         not la <= lb, or of a record count not the basis's.
[[nodiscard]] Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes);

} // namespace free_breakfast
