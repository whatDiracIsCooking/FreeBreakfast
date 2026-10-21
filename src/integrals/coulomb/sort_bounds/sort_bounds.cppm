/// @file
/// @brief One ket class of the J build, sorted descending by its screen bound
/// Iq * Dmax: the order e_hermite's screen breaks on.

export module free_breakfast.integrals.coulomb.sort_bounds;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

/// @brief A ket class's records in descending Iq * Dmax, as e_hermite reads a
/// ket: zeta, P, Iq, Dmax and D_q, each in the sorted order.
///
/// e_hermite's screen is a break: a lane stops at its first ket with
/// `Dmax * Ip * Iq < tau`, which skips only kets under tau when the kets
/// arrive sorted descending by Iq * Dmax. Iq depends only on zeta, but Dmax
/// carries the density and the pair's Gaussian prefactor, so the order is the
/// density's and is made again for each build.
///
/// Built once per class before the build forks (its buffers are drawn from
/// @p device's pool in that pool's stream order, as the build's own scratch),
/// then @ref sort queued on the class's stream once its bound, Dmax and D_q
/// are written. Movable, not copyable. Holds a view of the class's records,
/// so the class must outlive it.
class SortedKets {
public:
  /// @brief Scratch and sorted copies for @p cls, whose D_q rows are
  /// @p n_hermite wide. Allocates; launches nothing.
  SortedKets(const std::shared_ptr<Device> &device, const PrimitivePairClass &cls,
             int32_t n_hermite);

  /// @brief Sort the class's records by Iq * Dmax, descending, and gather
  /// zeta, P, @p bound, @p dmax and @p dq into that order, on @p stream.
  ///
  /// @p bound, @p dmax and @p dq are the class's own, in record order: its
  /// pair_bound, and dq's Dmax and D_q (n_rec x n_hermite). Records of equal
  /// bound keep their record order. Queued only, no synchronization.
  ///
  /// @return The first launch status. Launch errors only.
  [[nodiscard]] wwr::wwrError_t sort(wwr::wwrStream_t stream, const double *bound,
                                     const double *dmax, const double *dq);

  /// @brief The class as e_hermite's ket: its la, lb and n_rec, and zeta and
  /// P in the sorted order. The other fields are null; e_hermite does not
  /// read them of a ket.
  [[nodiscard]] PrimitivePairClassView view() const;
  /// @brief Iq, sorted.
  [[nodiscard]] const double *bound() const noexcept { return bound_.data(); }
  /// @brief Dmax, sorted.
  [[nodiscard]] const double *dmax() const noexcept { return dmax_.data(); }
  /// @brief D_q, its rows sorted: n_rec x n_hermite.
  [[nodiscard]] const double *dq() const noexcept { return dq_.data(); }

private:
  PrimitivePairClassView cls_;
  int32_t n_hermite_;
  Buffers::device<double> key_;
  Buffers::device<double> key_alt_;
  Buffers::device<uint32_t> perm_;
  Buffers::device<uint32_t> perm_alt_;
  Buffers::device<double> zeta_;
  Buffers::device<double> p_x_;
  Buffers::device<double> p_y_;
  Buffers::device<double> p_z_;
  Buffers::device<double> bound_;
  Buffers::device<double> dmax_;
  Buffers::device<double> dq_;
};

} // namespace free_breakfast
