/// @file
/// @brief Pulay DIIS (direct inversion of the iterative subspace)
/// extrapolation, on the GPU, over wwr's BLAS dispatch.
///
/// DIIS accelerates a fixed-point iteration by replacing the current iterate
/// with a linear combination F* = sum_i c_i F_i of recent ones, the
/// coefficients chosen to minimize the norm of the same combination of their
/// residuals e_i. They solve the bordered least-squares system
///
///     [ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
///     [   -1^T      0 ] [ l ] = [ -1 ]
///
/// (sum_i c_i = 1 enforced by the Lagrange row and column). For SCF, F_i is the
/// Fock matrix and e_i the orthogonal-basis commutator X^T [F, D S] X, but the
/// class sees only opaque length-len vectors.
///
/// The README has the ring storage, the incremental Gram and the one host
/// round-trip.

export module free_breakfast.linalg.diis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief The largest history Diis keeps. The coefficient solve holds the
/// (cap+1)-square bordered system in one block's statically sized shared
/// memory, and 64 keeps that under 48 KiB, CUDA's static per-block limit.
inline constexpr size_t kDiisMaxHistoryLimit = 64;

/// @brief Pulay DIIS extrapolation over device-resident iterate and residual
/// histories.
///
/// Holds a bounded ring of (F, e) pairs, each history one column-major
/// (len x cap) device buffer with one pair per column, plus a cap x cap
/// residual Gram updated incrementally. Everything is allocated by the
/// constructor, on the device of the DeviceResources it is given, and reused.
/// Move-only.
///
/// `double` only: an SCF Fock is double, and the solve's singular threshold
/// (1e-14) is a double-precision value.
class Diis {
public:
  /// @brief Allocate the history for length-@p len vectors, keeping at most
  /// @p max_history pairs, on @p res's device.
  ///
  /// @p len is an element count, not a matrix order: n*n for one n x n Fock,
  /// 2*n*n for an alpha/beta pair extrapolated with one shared coefficient set.
  /// @p max_history is clamped to 1..kDiisMaxHistoryLimit; capacity() reports
  /// the result. A failed allocation aborts (CreatePolicy).
  Diis(DeviceResources &res, size_t len, size_t max_history);

  Diis(Diis &&) noexcept = default;
  Diis &operator=(Diis &&) noexcept = default;
  Diis(const Diis &) = delete;
  Diis &operator=(const Diis &) = delete;
  ~Diis() = default;

  /// @brief Push (@p cur_fock, @p cur_residual) onto the history and, once it
  /// holds at least two pairs, write F* = sum_i c_i F_i into @p out_fock.
  ///
  /// The new pair overwrites the oldest once the history is full. With fewer
  /// than two pairs @p out_fock is left untouched. A singular subspace writes
  /// the newest Fock (a copy of @p cur_fock) and sets
  /// last_extrapolation_singular().
  ///
  /// Queued on res.stream(); with two or more pairs the call synchronizes that
  /// stream once, to read the singular flag. The device res.dev_idx() names is
  /// made current for the call and the previous one restored.
  ///
  /// @param res           Supplies the stream and the BLAS handle. Must be on
  ///                      the device the constructor allocated on.
  /// @param cur_fock      Device iterate F (len elements), copied in.
  /// @param cur_residual  Device residual e (len elements), copied in.
  /// @param out_fock      Device output (len elements). May alias @p cur_fock.
  /// @return Success; InvalidConfig if @p res is on another device or len does
  ///         not fit a 32-bit BLAS dimension; or the GpuRuntime error of the
  ///         first copy, BLAS or launch call that failed. An InvalidConfig
  ///         leaves the history unchanged; a GpuRuntime error resets it, since
  ///         a column or Gram entry may be half-written.
  [[nodiscard]] Status push_and_extrapolate(DeviceResources &res, const double *cur_fock,
                                            const double *cur_residual, double *out_fock);

  /// @brief Drop every pair, so the next push starts a fresh subspace. The
  /// buffers are kept.
  void reset() noexcept {
    size_ = 0;
    head_ = 0;
    last_singular_ = false;
  }

  /// @brief Number of pairs currently held, 0..capacity().
  [[nodiscard]] size_t size() const noexcept { return size_; }

  /// @brief The most pairs kept: max_history after clamping.
  [[nodiscard]] size_t capacity() const noexcept { return cap_; }

  /// @brief Elements per iterate or residual vector.
  [[nodiscard]] size_t length() const noexcept { return len_; }

  /// @brief Whether the last push that extrapolated (two or more pairs) found
  /// the subspace singular and fell back to the newest Fock. False before any
  /// such push and after one with fewer than two pairs, which is ordinary
  /// start-up rather than a fallback.
  [[nodiscard]] bool last_extrapolation_singular() const noexcept { return last_singular_; }

private:
  size_t len_;
  size_t cap_;
  int32_t dev_idx_;
  size_t size_ = 0; // pairs held, 0..cap_
  size_t head_ = 0; // the oldest column once full; 0 while growing
  bool last_singular_ = false;
  // (len_ x cap_) column-major, one pair per column in physical (ring) order.
  Buffers::device<double> fock_;
  Buffers::device<double> error_;
  // cap_ x cap_, leading dimension cap_, physical-column indexed. Only the
  // upper triangle over the valid columns is ever written or read.
  Buffers::device<double> gram_;
  Buffers::device<double> coeff_;
  Buffers::device<int32_t> singular_;
};

} // namespace free_breakfast
