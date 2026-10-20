/// @file
/// @brief McMurchie-Davidson primitive-pair records: a BasisSet turned into the
/// per-class inputs the generated one-electron kernels read.
///
/// For a primitive pair (alpha on A, beta on B) the Gaussian product theorem
/// gives one Gaussian of exponent zeta = alpha + beta centred at
/// P = (alpha A + beta B) / zeta. A record carries what a kernel needs of that
/// pair and nothing it can form itself: zeta, P, P - A, P - B, the scalar
/// s = c_a c_b (pi / zeta)^{3/2}, the ket exponent beta and the two shells' AO
/// offsets. The kernels form the Hermite expansion E from those, including its
/// exp(-alpha beta / zeta |A - B|^2) prefactor (as exp(zeta X_PA X_PB) per axis),
/// so the host forms neither.
///
/// Host only, so it builds and is tested on a device-less box. It owns no
/// record storage: the caller sizes it from @ref primitive_pair_classes and
/// @ref write_primitive_pairs fills it, so free_breakfast.integrals.one_electron
/// can hand it pinned memory and copy it to the device as written.
///
/// The fields are named once, by @ref PairFields (pair_fields.h, included in the
/// global module fragment and re-exported, as atom.cppm does).

module;

#include "pair_fields.h"

export module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;

export namespace free_breakfast {

using free_breakfast::PairFields;

/// @brief The number of fields in a @ref PairFields.
constexpr size_t kPairFields = 14;

/// @brief `f(fields.zeta, fields.p_x, ..., fields.off_b)`: every field of
/// @p fields, in declaration order, as one call.
///
/// The one place besides pair_fields.h that lists the fields; everything that
/// walks them goes through here or @ref map_pair_fields.
template <typename F, typename Fields>
constexpr decltype(auto) apply_pair_fields(F &&f, Fields &&fields) {
  static_assert(sizeof(PairFields<char, char>) == kPairFields,
                "a field was added to PairFields but not here");
  return std::forward<F>(f)(fields.zeta, fields.p_x, fields.p_y, fields.p_z, fields.pa_x,
                            fields.pa_y, fields.pa_z, fields.pb_x, fields.pb_y, fields.pb_z,
                            fields.s, fields.eb, fields.off_a, fields.off_b);
}

/// @brief A @ref PairFields of `f(field)` for every field of @p fields: the
/// real fields become `decltype(f(fields.zeta))`, the offsets
/// `decltype(f(fields.off_a))`.
template <typename F, typename Fields>
constexpr auto map_pair_fields(F &&f, Fields &&fields) {
  using Real = decltype(f(fields.zeta));
  using Offset = decltype(f(fields.off_a));
  return apply_pair_fields([&f](auto &&...field) { return PairFields<Real, Offset>{f(field)...}; },
                           fields);
}

/// @brief One shell-pair class (la, lb), with la <= lb, and its record count:
/// what a caller needs to size the class's @ref PrimitivePairRecords.
///
/// **The class is always la <= lb.** A shell pair whose bra has the higher
/// angular momentum is stored swapped, B as the bra and A as the ket, so the
/// kernels need one class per unordered pair of momenta: (s|p) and no (p|s).
/// The overlap, kinetic and nuclear-attraction matrices are symmetric, and the
/// kernels write each value at (m, n) and at (n, m), so the swap changes
/// nothing about the matrix built.
struct PrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la{0};
  /// @brief The ket shell's angular momentum.
  int32_t lb{0};
  /// @brief The number of records, positive: the length of every field.
  int32_t n_rec{0};
};

/// @brief One class's record storage, owned by the caller: one span per field,
/// each @ref PrimitivePairClass::n_rec long, record @c k of a field at index
/// @c k.
using PrimitivePairRecords = PairFields<std::span<double>, std::span<int32_t>>;

/// @brief The primitive-pair classes of @p basis, one per (la, lb) reached, in
/// ascending (la, lb) order, each with its record count.
///
/// Every shell pair i <= j (in basis order) contributes one record per
/// primitive pair to the class of its momenta in la <= lb order. Only reached
/// classes appear, so no class is empty. No pair is screened out.
///
/// The AO offsets are int32_t; @ref BasisSet already guarantees its Cartesian
/// AO count fits one. A class of more than INT32_MAX records, which
/// @ref PrimitivePairClass::n_rec cannot count, logs and aborts.
[[nodiscard]] std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis);

/// @brief Write the records of @p basis into @p records, one entry per class
/// of @p classes.
///
/// @p classes must be @ref primitive_pair_classes of @p basis, and every span
/// of each entry of @p records its class's n_rec long; anything else logs and
/// aborts. Each class's records are in shell-pair order, alpha-major within a
/// pair; a diagonal pair (i == j) is the shell against itself, its two AO
/// offsets equal, which is how the kernels recognise it.
///
/// Coefficients are taken from the basis as given (see @ref Shell), so a basis
/// that wants normalized primitives is placed from
/// @c BasisStamp::primitive_normalized.
void write_primitive_pairs(const BasisSet &basis, std::span<const PrimitivePairClass> classes,
                           std::span<const PrimitivePairRecords> records);

} // namespace free_breakfast
