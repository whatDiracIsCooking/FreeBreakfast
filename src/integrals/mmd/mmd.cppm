/// @file
/// @brief McMurchie-Davidson primitive-pair records: a BasisSet turned into the
/// per-class inputs the generated one-electron kernels read, on the device.
///
/// For a primitive pair (alpha on A, beta on B) the Gaussian product theorem
/// gives one Gaussian of exponent zeta = alpha + beta centred at
/// P = (alpha A + beta B) / zeta. A record carries what a kernel needs of that
/// pair and nothing it can form itself: zeta, P, P - A, P - B, the scalar
/// s = c_a c_b (pi / zeta)^{3/2}, the ket exponent beta and the two shells' AO
/// offsets. The kernels form the Hermite expansion E from those, including its
/// exp(-alpha beta / zeta |A - B|^2) prefactor (as exp(zeta X_PA X_PB) per axis),
/// so the host forms neither. The record also carries what the two-electron
/// builds derive from: the bra exponent alpha and each shell's rank among the
/// basis's shells of its momentum. Three fields are formed on the device, once
/// per basis: the record's Schwarz bound I = pair_bound(zeta, la + lb), and its
/// dressed bound (FelizJueves's ao.k.record_bounds) as stored and swapped.
///
/// The launchers' view of a class is primitive_pair_class_view.h's, included
/// in the global module fragment and re-exported (atom.cppm has the why).

module;

#include "primitive_pair_class_view.h"

export module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast {

using free_breakfast::PrimitivePairClassView;

/// @brief Every primitive-pair record of one shell-pair class (la, lb), with
/// la <= lb, one @ref n_rec long @c MirroredBuffer per field, all on the
/// class's own work stream.
///
/// A vector is three fields, one per axis (P - A is @ref pa_x, @ref pa_y,
/// @ref pa_z), named and laid out as the generated kernels take them. The
/// pinned side is what the records were written into; it is kept for the
/// class's lifetime, so the copy to the device never needs a host
/// synchronization to protect it.
///
/// **The class is always la <= lb.** A shell pair whose bra has the higher
/// angular momentum is stored swapped, B as the bra and A as the ket, so the
/// kernels need one class per unordered pair of momenta: (s|p) and no (p|s).
/// The overlap, kinetic and nuclear-attraction matrices are symmetric, and the
/// kernels write each value at (m, n) and at (n, m), so the swap changes
/// nothing about the matrix built.
///
/// Movable, not copyable.
struct PrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la;
  /// @brief The ket shell's angular momentum.
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
  /// @brief P - A: x.
  MirroredBuffer<double> pa_x;
  /// @brief P - A: y.
  MirroredBuffer<double> pa_y;
  /// @brief P - A: z.
  MirroredBuffer<double> pa_z;
  /// @brief P - B: x.
  MirroredBuffer<double> pb_x;
  /// @brief P - B: y.
  MirroredBuffer<double> pb_y;
  /// @brief P - B: z.
  MirroredBuffer<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  MirroredBuffer<double> s;
  /// @brief alpha, the bra primitive's exponent.
  MirroredBuffer<double> ea;
  /// @brief beta, the ket primitive's exponent.
  MirroredBuffer<double> eb;
  /// @brief The record's Schwarz bound I = pair_bound(zeta, la + lb), formed
  /// on the device by the generated pair_bound<la, lb> after the zeta order:
  /// a J or K screen's Ip as a bra and Iq as a ket. It depends on zeta and
  /// la + lb only, and rises with zeta, so it is non-increasing in record order.
  /// Zero in a class above kMaxAngularMomentum, which has no kernel and which
  /// every build refuses.
  MirroredBuffer<double> bound;
  /// @brief The record's DRESSED Schwarz bound as stored, FelizJueves's
  /// ao.k.record_bounds: max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p|
  /// I_p(zeta), the max over the bra shell's Cartesians, formed on the device
  /// by the generated record_bound<la, lb> after the zeta order. Zero in a
  /// class above kMaxAngularMomentum, as @ref bound.
  MirroredBuffer<double> bound_ab;
  /// @brief The same bound of the record swapped, the ket shell first: the
  /// max over the ket shell's Cartesians, by record_bound<lb, la> with P - A
  /// and P - B exchanged. Equal to @ref bound_ab on a diagonal record.
  MirroredBuffer<double> bound_ba;
  /// @brief The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  MirroredBuffer<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  MirroredBuffer<int32_t> off_b;
  /// @brief The bra shell's rank: its index among the basis's shells of
  /// momentum @ref la, in basis order.
  MirroredBuffer<int32_t> rank_a;
  /// @brief The ket shell's rank, among the basis's shells of momentum
  /// @ref lb.
  MirroredBuffer<int32_t> rank_b;

  /// @brief The class's own work stream: the records were allocated and copied
  /// on it, and every build launches the class on it.
  const std::shared_ptr<DeviceStream> &device_stream() const noexcept { return zeta.handle(); }

  /// @brief The records on the device by pointer, for a launcher. Valid while
  /// this is alive.
  [[nodiscard]] PrimitivePairClassView view() const {
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
        .ea = ea.device().data(),
        .eb = eb.device().data(),
        .bound = bound.device().data(),
        .bound_ab = bound_ab.device().data(),
        .bound_ba = bound_ba.device().data(),
        .off_a = off_a.device().data(),
        .off_b = off_b.device().data(),
        .rank_a = rank_a.device().data(),
        .rank_b = rank_b.device().data(),
    };
  }
};

/// @brief The primitive-pair records of @p basis on @p res's device, one class
/// per (la, lb) reached, in ascending (la, lb) order, each on a new
/// @c DeviceStream of res.device().
///
/// Every shell pair i <= j (in basis order) contributes one record for each of
/// its primitive pairs, to the class of its momenta in la <= lb order; a
/// diagonal pair (i == j) is the shell against itself, its two AO offsets
/// equal, which is how the kernels recognise it. Only reached classes appear,
/// so no class is empty. No pair is screened out.
///
/// **Each class's records are in descending zeta order**, equal zetas in
/// shell-pair, then alpha-major, order. Within a class the record bound
/// @ref PrimitivePairClass::bound rises with zeta, so it is non-increasing in
/// record order and a J build's bras with the longest ket loops launch first.
/// No kernel depends on the order: each accumulates.
///
/// The records are written into each class's pinned side, copied to the
/// device, ordered there (a stable merge sort on -zeta, every field permuted
/// in place), each record's three bounds formed from its ordered fields, and
/// the pinned side refreshed from the device, all queued on the class's stream, ahead of
/// anything a build launches there; no host synchronization. So the pinned side holds the final records only once that
/// stream has been synchronized. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// Coefficients are taken from the basis as given (see @ref Shell), so a basis
/// that wants normalized primitives is placed from
/// @c BasisStamp::primitive_normalized. The AO offsets are int32_t;
/// @ref BasisSet already guarantees its Cartesian AO count fits one. A class of
/// more than INT32_MAX records logs and aborts.
///
/// @return The classes (none for an empty basis), or the GpuRuntime error of
///         the first copy that failed.
[[nodiscard]] Result<std::vector<PrimitivePairClass>> primitive_pair_classes(DeviceResources &res,
                                                                             const BasisSet &basis);

} // namespace free_breakfast
