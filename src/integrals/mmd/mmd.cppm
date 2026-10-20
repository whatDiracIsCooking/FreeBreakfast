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

export module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The real-valued fields of a primitive-pair record, in their order
/// within a class's @ref PrimitivePairRecords::reals block. A vector is three
/// fields, one per axis, as the generated kernels take it.
enum class PairReal : uint8_t {
  Zeta, ///< zeta = alpha + beta.
  PX,   ///< P, the product centre: x.
  PY,   ///< P: y.
  PZ,   ///< P: z.
  PaX,  ///< P - A: x.
  PaY,  ///< P - A: y.
  PaZ,  ///< P - A: z.
  PbX,  ///< P - B: x.
  PbY,  ///< P - B: y.
  PbZ,  ///< P - B: z.
  S,    ///< c_a c_b (pi / zeta)^{3/2}.
  Eb,   ///< beta, the ket primitive's exponent.
};

/// @brief The number of @ref PairReal fields.
constexpr int32_t kPairReals = 12;

/// @brief The integer fields of a primitive-pair record, in their order within
/// a class's @ref PrimitivePairRecords::offsets block.
enum class PairOffset : uint8_t {
  A, ///< The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  B, ///< The ket shell's first Cartesian AO.
};

/// @brief The number of @ref PairOffset fields.
constexpr int32_t kPairOffsets = 2;

/// @brief Where field @p f of a class of @p n_rec records starts within its
/// block: every field is @p n_rec consecutive values, one per record.
[[nodiscard]] constexpr size_t field_offset(const PairReal f, const int32_t n_rec) noexcept {
  return static_cast<size_t>(f) * static_cast<size_t>(n_rec);
}

/// @copydoc field_offset(PairReal, int32_t)
[[nodiscard]] constexpr size_t field_offset(const PairOffset f, const int32_t n_rec) noexcept {
  return static_cast<size_t>(f) * static_cast<size_t>(n_rec);
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
  /// @brief The number of records, positive.
  int32_t n_rec{0};
};

/// @brief One class's record storage, owned by the caller: field @c f of record
/// @c k is at `reals[field_offset(f, n_rec) + k]` (and likewise @ref offsets).
struct PrimitivePairRecords {
  /// @brief kPairReals * n_rec values, laid out by @ref PairReal.
  std::span<double> reals;
  /// @brief kPairOffsets * n_rec values, laid out by @ref PairOffset.
  std::span<int32_t> offsets;
};

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
/// @p classes must be @ref primitive_pair_classes of @p basis, and each entry
/// of @p records sized for its class; anything else logs and aborts. Each
/// class's records are in shell-pair order, alpha-major within a pair; a
/// diagonal pair (i == j) is the shell against itself, its two AO offsets
/// equal, which is how the kernels recognise it.
///
/// Coefficients are taken from the basis as given (see @ref Shell), so a basis
/// that wants normalized primitives is placed from
/// @c BasisStamp::primitive_normalized.
void write_primitive_pairs(const BasisSet &basis, std::span<const PrimitivePairClass> classes,
                           std::span<const PrimitivePairRecords> records);

} // namespace free_breakfast
