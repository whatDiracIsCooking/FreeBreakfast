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
/// la <= lb, as parallel arrays: one entry per record, and each vector field
/// split per axis (xpa, ypa, zpa rather than one interleaved array).
///
/// That is the layout the kernels read: structure of arrays, so with one
/// thread per record a warp's loads are contiguous, and on the device the
/// arrays travel as one FelizJueves @c PrimPairSoA
/// (free_breakfast.integrals.one_electron.upload).
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
  /// @brief zeta = alpha + beta.
  std::vector<double> zeta;
  /// @brief P, the product centre: x.
  std::vector<double> xp;
  /// @brief P: y.
  std::vector<double> yp;
  /// @brief P: z.
  std::vector<double> zp;
  /// @brief P - A: x.
  std::vector<double> xpa;
  /// @brief P - A: y.
  std::vector<double> ypa;
  /// @brief P - A: z.
  std::vector<double> zpa;
  /// @brief P - B: x.
  std::vector<double> xpb;
  /// @brief P - B: y.
  std::vector<double> ypb;
  /// @brief P - B: z.
  std::vector<double> zpb;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  std::vector<double> s;
  /// @brief beta, the ket primitive's exponent.
  std::vector<double> eb;
  /// @brief The bra shell's first Cartesian AO (@c BasisSet::ao_offset_cart).
  std::vector<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  std::vector<int32_t> off_b;

  /// @brief The number of records.
  [[nodiscard]] size_t size() const noexcept { return zeta.size(); }
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
/// AO count fits one, so this cannot fail.
[[nodiscard]] std::vector<PrimitivePairClass> primitive_pair_classes(const BasisSet &basis);

} // namespace free_breakfast
