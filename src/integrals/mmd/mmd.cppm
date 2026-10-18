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
/// Host only, so it builds and is tested on a device-less box. Uploading the
/// records is free_breakfast.integrals.one_electron's job.

export module free_breakfast.integrals.mmd;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief Every primitive-pair record of one shell-pair class (la, lb), with
/// la <= lb, as parallel arrays (one entry per record; three, x y z, for the
/// vector fields).
///
/// **The class is always la <= lb.** A shell pair whose bra has the higher
/// angular momentum is stored swapped, B as the bra and A as the ket, so the
/// kernels need one class per unordered pair of momenta: (s|p) and no (p|s).
/// The overlap, kinetic and nuclear-attraction matrices are symmetric, and the
/// kernels write each value at (m, n) and at (n, m), so the swap changes
/// nothing about the matrix built.
///
/// Coefficients are taken from the basis as given (see @ref Shell), so a basis
/// that wants normalized primitives is placed from
/// @c BasisStamp::primitive_normalized.
struct PrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la{0};
  /// @brief The ket shell's angular momentum.
  int32_t lb{0};
  /// @brief The number of records: the length of every per-record field.
  int32_t n_rec{0};
  /// @brief zeta = alpha + beta.
  std::vector<double> zeta;
  /// @brief P, the product centre (3 per record).
  std::vector<double> p;
  /// @brief P - A (3 per record).
  std::vector<double> xpa;
  /// @brief P - B (3 per record).
  std::vector<double> xpb;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  std::vector<double> s;
  /// @brief beta, the ket primitive's exponent.
  std::vector<double> eb;
  /// @brief The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  std::vector<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  std::vector<int32_t> off_b;
};

/// @brief The primitive-pair records of @p basis, one class per (la, lb)
/// reached, in ascending (la, lb) order.
///
/// Every shell pair i <= j (in basis order) contributes alpha-major records
/// for each of its primitive pairs, to the class of its momenta in la <= lb
/// order; a diagonal pair (i == j) is the shell against itself, its two AO
/// offsets equal, which is how the kernels recognise it. Only reached classes
/// appear, so no class is empty. No pair is screened out.
///
/// The AO offsets are int32_t; @ref BasisSet already guarantees its Cartesian
/// AO count fits one. A class of more than INT32_MAX records, which
/// @ref PrimitivePairClass::n_rec cannot count, logs and aborts.
[[nodiscard]] std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis);

} // namespace free_breakfast
