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

export module free_breakfast.integrals.exchange.pairs;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
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
  /// @brief The group's first record.
  int32_t start;
  /// @brief The group's record count, positive.
  int32_t count;
};

/// @brief Every ordered primitive-pair record of one class (la, lb), in either
/// order of la and lb, one @ref n_rec long @c MirroredBuffer per field, all on
/// the class's own work stream, and its groups by first shell on the host and,
/// as their bounds, on the device.
///
/// The fields are mmd's, named as there, with "a" the record's first shell and
/// "b" its second: a record taken from mmd's (b, a) is stored with its P - A
/// and P - B, and its two AO offsets, swapped; zeta, P and s are
/// orientation-free and copied as they are. mmd's ket exponent `eb` is not
/// carried (kb does not read it, and the swapped orientation would want
/// the other exponent), nor is mmd's undressed `bound`, which K does not read.
/// Two fields are new: @ref record_bound, K's dressed Schwarz bound, which
/// depends on the orientation, and @ref rank_b.
///
/// **The records are grouped by first shell**, the groups in basis order, and
/// each group sorted by @ref record_bound, non-increasing: the order kb's
/// screen `break`s need of its bras and kets.
///
/// Movable, not copyable.
struct OrderedPairClass {
  /// @brief The first shell's angular momentum.
  int32_t la;
  /// @brief The second shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive: the length of every field.
  int32_t n_rec;
  /// @brief zeta = alpha + beta.
  MirroredBuffer<double> zeta;
  /// @brief P, the product centre: x.
  MirroredBuffer<double> p_x;
  /// @brief P: y.
  MirroredBuffer<double> p_y;
  /// @brief P: z.
  MirroredBuffer<double> p_z;
  /// @brief P - A, A the first shell's centre: x.
  MirroredBuffer<double> pa_x;
  /// @brief P - A: y.
  MirroredBuffer<double> pa_y;
  /// @brief P - A: z.
  MirroredBuffer<double> pa_z;
  /// @brief P - B, B the second shell's centre: x.
  MirroredBuffer<double> pb_x;
  /// @brief P - B: y.
  MirroredBuffer<double> pb_y;
  /// @brief P - B: z.
  MirroredBuffer<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  MirroredBuffer<double> s;
  /// @brief The record's DRESSED Schwarz bound, FelizJueves's
  /// ao.k.record_bounds: `max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p|
  /// I_p(zeta)`, the max over the first shell's Cartesians, so it depends on
  /// the orientation. kb's Ip as a bra and Iq as a ket, and what the K build's
  /// reach and bound maximum read. Non-increasing within each group; exactly
  /// 0.0 for a pair whose pair factor underflows. Zero in a class above d,
  /// which has no kernel and which the K build refuses.
  MirroredBuffer<double> record_bound;
  /// @brief The first shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  MirroredBuffer<int32_t> off_a;
  /// @brief The second shell's first Cartesian AO.
  MirroredBuffer<int32_t> off_b;
  /// @brief The second shell's rank: its index among the basis's shells of
  /// momentum @ref lb, in basis order. kb's Dcd row is
  /// `rank_c * n_shells(ld) + rank_d` (exchange_helpers_bridge.h), so a bra's
  /// `dc` and a ket's `dd` are formed from it.
  MirroredBuffer<int32_t> rank_b;
  /// @brief The groups by first shell, in basis order, covering the records
  /// exactly: group g is records [start, start + count).
  std::vector<OrderedPairGroup> groups;
  /// @brief The groups' bounds on the device, groups.size() + 1 entries: group
  /// g is records [group_start[g], group_start[g + 1]), the last entry n_rec.
  /// The K build makes its block table from it on the device.
  MirroredBuffer<int32_t> group_start;

  /// @brief The class's own work stream: the records were allocated and copied
  /// on it.
  const std::shared_ptr<DeviceStream> &device_stream() const noexcept { return zeta.handle(); }

  /// @brief The records on the device by pointer, for a launcher. Valid while
  /// this is alive.
  [[nodiscard]] OrderedPairClassView view() const {
    return {
        .la = la,
        .lb = lb,
        .n_rec = n_rec,
        .zeta = zeta.device().data(),
        .p_x = p_x.device().data(),
        .p_y = p_y.device().data(),
        .p_z = p_z.device().data(),
        .pa_x = pa_x.device().data(),
        .pa_y = pa_y.device().data(),
        .pa_z = pa_z.device().data(),
        .pb_x = pb_x.device().data(),
        .pb_y = pb_y.device().data(),
        .pb_z = pb_z.device().data(),
        .s = s.device().data(),
        .record_bound = record_bound.device().data(),
        .off_a = off_a.device().data(),
        .off_b = off_b.device().data(),
        .rank_b = rank_b.device().data(),
        .n_groups = static_cast<int32_t>(groups.size()),
        .group_start = group_start.device().data(),
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
/// The gather is on the host, once per basis: each mmd class's stream is
/// synchronized first (the pinned side holds mmd's final, zeta-ordered records
/// only after), then each record is appended to its first shell's group. The
/// ordered records are written into each class's pinned side and copied to the
/// device on its stream, as is each class's @ref OrderedPairClass::group_start,
/// written from its groups. Then, on that stream, each record's
/// @ref OrderedPairClass::record_bound (the generated record_bound<la, lb>),
/// the order that sorts each group by it, non-increasing, equal bounds in
/// gather order (two stable device sorts), every field put in that order on
/// the device, and the pinned side refreshed from it. **kb's screen `break`s
/// depend on that order**: an unsorted group is a wrong K, not a slow one.
/// Every class's stream is synchronized before the return, so the pinned side
/// is final. The device res.dev_idx() names is made current for the call and
/// the previous one restored. A class of more than INT32_MAX records logs and
/// aborts.
///
/// @return The ordered classes (none for no @p classes), the GpuRuntime error
///         of the first synchronization, copy or launch that failed, or an
///         InvalidConfig error if a record's AO offset names no shell of
///         @p basis (the classes are of another basis).
[[nodiscard]] Result<std::vector<OrderedPairClass>>
ordered_pair_classes(DeviceResources &res, const BasisSet &basis,
                     const std::vector<PrimitivePairClass> &classes);

} // namespace free_breakfast
