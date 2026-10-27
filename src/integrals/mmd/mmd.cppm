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
/// so this module never forms it. The record also carries what the
/// two-electron builds derive from: the bra exponent alpha and each shell's
/// rank among the basis's shells of its momentum. Every field is formed on the
/// device, once per basis, from the basis's shells and each class's shell
/// pairs: the host uploads O(shells) and O(shell pairs), never a record. Three
/// of them are bounds: the record's Schwarz bound I = pair_bound(zeta, la + lb),
/// and its dressed bound (FelizJueves's ao.k.record_bounds) as stored and
/// swapped.
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
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast {

using free_breakfast::PrimitivePairClassView;

/// @brief Every primitive-pair record of one shell-pair class (la, lb), with
/// la <= lb, one @ref n_rec long device buffer per field, all on the class's
/// own work stream.
///
/// A vector is three fields, one per axis (P - A is @ref pa_x, @ref pa_y,
/// @ref pa_z), named and laid out as the generated kernels take them.
///
/// **The records exist on the device only.** They are formed there from the
/// basis's shells and the class's shell pairs, already in their final order,
/// then bounded; every reader (the builds, J's prune, K's gather) reads them
/// there. A reader that wants them on the host copies them back itself.
///
/// **A basis's class is always la <= lb.** (Between two bases,
/// @ref mixed_primitive_pair_classes', it is any order, never swapped.) Within
/// one basis a shell pair whose bra has the higher
/// angular momentum is stored swapped, B as the bra and A as the ket, so the
/// kernels need one class per unordered pair of momenta: (s|p) and no (p|s).
/// The overlap, kinetic and nuclear-attraction matrices are symmetric, and the
/// kernels write each value at (m, n) and at (n, m), so the swap changes
/// nothing about the matrix built.
///
/// Movable, not copyable.
struct PrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb within one basis.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive: the length of every field.
  int32_t n_rec;
  /// @brief The class's own work stream: every field was allocated and written
  /// on it, and every build launches the class on it. Declared before the
  /// fields, so it outlives their stream-ordered frees.
  std::shared_ptr<DeviceStream> handle;
  /// @brief zeta = alpha + beta.
  StreamBuffers::device<double> zeta;
  /// @brief P, the product centre: x.
  StreamBuffers::device<double> p_x;
  /// @brief P: y.
  StreamBuffers::device<double> p_y;
  /// @brief P: z.
  StreamBuffers::device<double> p_z;
  /// @brief P - A: x.
  StreamBuffers::device<double> pa_x;
  /// @brief P - A: y.
  StreamBuffers::device<double> pa_y;
  /// @brief P - A: z.
  StreamBuffers::device<double> pa_z;
  /// @brief P - B: x.
  StreamBuffers::device<double> pb_x;
  /// @brief P - B: y.
  StreamBuffers::device<double> pb_y;
  /// @brief P - B: z.
  StreamBuffers::device<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  StreamBuffers::device<double> s;
  /// @brief alpha, the bra primitive's exponent.
  StreamBuffers::device<double> ea;
  /// @brief beta, the ket primitive's exponent.
  StreamBuffers::device<double> eb;
  /// @brief The record's Schwarz bound I = pair_bound(zeta, la + lb), formed
  /// on the device by the generated pair_bound<la, lb> after the zeta order:
  /// a J or K screen's Ip as a bra and Iq as a ket. It depends on zeta and
  /// la + lb only, and rises with zeta, so it is non-increasing in record order.
  /// Real through f, its kernels' limit; zero in a class above, which has no
  /// kernel and which every build refuses.
  StreamBuffers::device<double> bound;
  /// @brief The record's DRESSED Schwarz bound as stored, FelizJueves's
  /// ao.k.record_bounds: max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p|
  /// I_p(zeta), the max over the bra shell's Cartesians, formed on the device
  /// by the generated record_bound<la, lb> after the zeta order: a K screen's
  /// Ip and Iq. Real through f, its kernels' limit (@ref bound's too, though
  /// another emitter writes it); zero in a class above f, which has no kernel
  /// and which the K build refuses.
  StreamBuffers::device<double> bound_ab;
  /// @brief The same bound of the record swapped, the ket shell first: the
  /// max over the ket shell's Cartesians, by record_bound<lb, la> with P - A
  /// and P - B exchanged. Equal to @ref bound_ab on a diagonal record.
  StreamBuffers::device<double> bound_ba;
  /// @brief The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  StreamBuffers::device<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_b;
  /// @brief The bra shell's rank: its index among the basis's shells of
  /// momentum @ref la, in basis order.
  StreamBuffers::device<int32_t> rank_a;
  /// @brief The ket shell's rank, among the basis's shells of momentum
  /// @ref lb.
  StreamBuffers::device<int32_t> rank_b;

  /// @brief @ref handle, as launch_each takes a class's stream.
  const std::shared_ptr<DeviceStream> &device_stream() const noexcept { return handle; }

  /// @brief The records on the device by pointer, for a launcher. Valid while
  /// this is alive.
  [[nodiscard]] PrimitivePairClassView view() const {
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
        .bound = bound.data(),
        .bound_ab = bound_ab.data(),
        .bound_ba = bound_ba.data(),
        .off_a = off_a.data(),
        .off_b = off_b.data(),
        .rank_a = rank_a.data(),
        .rank_b = rank_b.data(),
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
/// The host uploads the basis's shells and primitives, and each class's shell
/// pairs with each one's first record; the device does the rest, on the
/// class's stream: a stable merge sort of the records' -zeta, formed from
/// those tables, then every field written once, already in that order, then
/// each record's three bounds. The tables are the call's own, so it waits for
/// every class's stream before returning: the records are then final, and a
/// consumer on any stream may read them. The device res.dev_idx() names is
/// made current for the call and the previous one restored.
///
/// Coefficients are taken from the basis as given (see @ref Shell), so a basis
/// that wants normalized primitives is placed from
/// @c BasisStamp::primitive_normalized. The AO offsets are int32_t;
/// @ref BasisSet already guarantees its Cartesian AO count fits one. A class of
/// more than INT32_MAX records, or a basis of more than INT32_MAX primitives,
/// logs and aborts.
///
/// @return The classes (none for an empty basis), or the GpuRuntime error of
///         the first copy, launch or wait that failed.
[[nodiscard]] Result<std::vector<PrimitivePairClass>> primitive_pair_classes(DeviceResources &res,
                                                                             const BasisSet &basis);

/// @brief The primitive-pair records between two bases, @p rows and @p cols, on
/// @p res's device: what overlap_mixed reads.
///
/// Every ORDERED shell pair (i of @p rows, j of @p cols) contributes one record
/// for each of its primitive pairs, i always the bra, to class (la of i, lb of
/// j). Nothing is swapped, so la > lb classes appear, and @ref
/// PrimitivePairClass::off_a is a Cartesian AO of @p rows while @ref
/// PrimitivePairClass::off_b is one of @p cols: two index sets, so equal
/// offsets mean nothing and no record is diagonal. The bases may differ in
/// basis set, in geometry, in both or in neither; @p rows and @p cols may be
/// the same object.
///
/// These are not @ref primitive_pair_classes' records: the symmetric builds
/// (S, T, V, J, K) read a basis's i <= j pairs and mirror each value, so a
/// mixed class handed to one builds the wrong matrix. Every field is formed as
/// there, in the same order (descending zeta per class) and with the same
/// bounds, each rank among its own basis's shells; ascending class order, none
/// empty, each on a new @c DeviceStream, finished when it returns. Empty if
/// either basis is. The same limits, checked the same way.
///
/// @return The classes, or the GpuRuntime error of the first copy, launch or
///         wait that failed.
[[nodiscard]] Result<std::vector<PrimitivePairClass>>
mixed_primitive_pair_classes(DeviceResources &res, const BasisSet &rows, const BasisSet &cols);

} // namespace free_breakfast
