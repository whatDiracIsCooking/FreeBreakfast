/// @file
/// @brief Device-resident AO-basis matrices tagged by layout (@ref
/// free_breakfast::MatrixAO), and a pair holding one of each that converts
/// between them through a shared Cartesian-to-spherical transform (@ref
/// free_breakfast::MatrixAOPair).
///
/// Every shell in this tree means its 2l+1 real solid harmonics, but the
/// integral kernels evaluate its (l+1)(l+2)/2 Cartesian Gaussians, so an
/// AO-basis matrix has a Cartesian form (@c BasisSet::nao_cart square) and a
/// spherical one (@c BasisSet::nao_sphe square). The layout is a template
/// parameter, so handing a Cartesian matrix to code that wants a spherical one
/// is a compile error rather than a wrong answer.
///
/// The README has which conversion suits which matrix and why only one
/// direction is lossless.

export module free_breakfast.linalg.matrix_ao;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief Which AO layout of a @ref BasisSet a matrix is over: the Cartesian
/// functions the kernels evaluate, or the real solid harmonics the basis means.
enum class BasisLayout : uint8_t { Cartesian, Spherical };

/// @brief The number of AOs @p basis has in layout @p L: @c nao_cart() for
/// Cartesian, @c nao_sphe() for spherical.
template <BasisLayout L>
[[nodiscard]] uint32_t ao_count(const BasisSet &basis) {
  if constexpr (L == BasisLayout::Cartesian) {
    return basis.nao_cart();
  } else {
    return basis.nao_sphe();
  }
}

/// @brief A square, device-resident AO-basis matrix over one @ref BasisLayout.
///
/// Holds @c n() x @c n() doubles, column-major with packed leading dimension
/// @c n() (element (i, j) at @c i + j*n()), so @ref data goes straight into a
/// BLAS call. Allocated on @p res's device by the constructor and
/// zero-initialised; a failed allocation aborts (CreatePolicy). Move-only.
///
/// It records its order and device and nothing else: not the basis it was
/// sized from, and not what the matrix means (overlap, Fock, density, ...).
template <BasisLayout L>
class MatrixAO : private NonCopyable {
public:
  static constexpr BasisLayout kLayout = L;

  /// @brief A zero matrix of order @c ao_count<L>(basis), on @p res's device.
  MatrixAO(DeviceResources &res, const BasisSet &basis)
      : n_(ao_count<L>(basis)), dev_idx_(res.dev_idx()),
        data_(static_cast<size_t>(n_) * n_, res.device()) {}

  /// @brief The matrix order.
  [[nodiscard]] uint32_t n() const noexcept { return n_; }

  /// @brief The element count, n() * n().
  [[nodiscard]] size_t num_elements() const noexcept { return data_.num_elements(); }

  /// @brief The device the storage was allocated on.
  [[nodiscard]] int32_t dev_idx() const noexcept { return dev_idx_; }

  /// @brief The column-major device storage; @c nullptr when n() is 0.
  [[nodiscard]] double *data() noexcept { return data_.data(); }
  [[nodiscard]] const double *data() const noexcept { return data_.data(); }

private:
  uint32_t n_;
  int32_t dev_idx_;
  Buffers::device<double> data_;
};

using CartesianMatrixAO = MatrixAO<BasisLayout::Cartesian>;
using SphericalMatrixAO = MatrixAO<BasisLayout::Spherical>;

/// @brief Build @p basis's Cartesian-to-spherical transform U on the host and
/// upload it to @p res's device, for every @ref MatrixAOPair over that basis
/// to share.
///
/// U is @c spherical_transform_matrix(basis) stored column-major,
/// @c nao_cart x @c nao_sphe with leading dimension @c nao_cart. It is never
/// written after the upload, so pairs on different streams may share it.
///
/// Uploads nothing, and returns @c nullptr, when no shell is above p (or the
/// basis is empty): U is then the identity, and the pairs copy instead. This
/// differs from the host @c spherical_transform_matrix, which always builds U
/// and returns an explicit identity there.
/// Otherwise synchronizes res.stream() once, so the host staging copy can be
/// released.
///
/// nao_cart always fits a 32-bit BLAS dimension: @ref BasisSet bounds it.
///
/// @return U; or the GpuRuntime error of the upload. A failed allocation aborts
///         (CreatePolicy).
[[nodiscard]] Result<std::shared_ptr<const Buffers::device<double>>>
upload_spherical_transform_matrix(DeviceResources &res, const BasisSet &basis);

/// @brief One AO-basis quantity held in both layouts, a Cartesian and a
/// spherical @ref MatrixAO over the same basis, with the products that carry
/// one into the other.
///
///   - @ref cartesian_to_spherical: spherical() = U^T cartesian() U
///   - @ref spherical_to_cartesian: cartesian() = U spherical() U^T
///
/// Writing one member does not update or invalidate the other; the caller
/// calls @ref cartesian_to_spherical or @ref spherical_to_cartesian to bring them into step.
///
/// **They are not inverses.** Neither round trip returns the matrix it started
/// from once a shell is above p. Cartesian -> spherical -> Cartesian drops each
/// d shell's r^2-like component, which has no spherical counterpart; spherical
/// -> Cartesian -> spherical gives U^T U M U^T U, and U^T U != I (U's columns
/// are orthonormal under the Cartesian overlap, U^T S_cart U = I, not under
/// the plain dot product). What the two do preserve is the pairing
/// tr(U D U^T F) = tr(D U^T F U) between a matrix sent one way and a matrix
/// sent the other.
///
/// The pair co-owns U (@ref upload_spherical_transform_matrix) and owns its own
/// nao_cart x nao_sphe scratch, so two pairs over one U may run on different
/// streams. Calls on one pair must be ordered on one stream. Move-only.
class MatrixAOPair : private NonCopyable {
public:
  /// @brief Zero matrices of both orders for @p basis, on @p res's device,
  /// converting through @p u.
  ///
  /// @p u must be @c upload_spherical_transform_matrix's result for @p basis, made on
  /// the same device: @c nullptr exactly when no shell is above p, and
  /// otherwise nao_cart x nao_sphe. A @p u that does not fit @p basis aborts,
  /// since it is a caller bug; one from another basis of the same orders is not
  /// detected.
  MatrixAOPair(DeviceResources &res, const BasisSet &basis,
               std::shared_ptr<const Buffers::device<double>> u);

  [[nodiscard]] CartesianMatrixAO &cartesian() noexcept { return cartesian_; }
  [[nodiscard]] const CartesianMatrixAO &cartesian() const noexcept { return cartesian_; }
  [[nodiscard]] SphericalMatrixAO &spherical() noexcept { return spherical_; }
  [[nodiscard]] const SphericalMatrixAO &spherical() const noexcept { return spherical_; }

  /// @brief The device both members live on.
  [[nodiscard]] int32_t dev_idx() const noexcept { return cartesian_.dev_idx(); }

  /// @brief spherical() = U^T cartesian() U: the Cartesian member carried onto
  /// the spherical layout.
  ///
  /// Exact for a matrix whose entries are integrals over pairs of AOs (an
  /// overlap, a core Hamiltonian, a Fock matrix): the spherical AOs are fixed
  /// combinations of the Cartesian ones, so their integrals are the same
  /// combinations of the Cartesian integrals.
  ///
  /// Queued on res.stream(), with no host synchronization. The device
  /// res.dev_idx() names is made current for the call and the previous one
  /// restored.
  ///
  /// @return Success; InvalidConfig if @p res is not on the pair's device; or
  ///         the GpuRuntime error of the first BLAS or copy call that failed,
  ///         which leaves spherical() unspecified.
  [[nodiscard]] Status cartesian_to_spherical(DeviceResources &res);

  /// @brief cartesian() = U spherical() U^T: the spherical member carried onto
  /// the Cartesian layout.
  ///
  /// Exact for a matrix contracted against AO pairs rather than made of their
  /// integrals (a density): it gives the Cartesian matrix whose contraction
  /// with any Cartesian integral matrix equals the spherical one's contraction
  /// with that matrix's @ref cartesian_to_spherical.
  ///
  /// Same queueing, device and error contract as @ref cartesian_to_spherical, with
  /// cartesian() the member left unspecified on failure.
  [[nodiscard]] Status spherical_to_cartesian(DeviceResources &res);

private:
  CartesianMatrixAO cartesian_;
  SphericalMatrixAO spherical_;
  std::shared_ptr<const Buffers::device<double>> u_;
  // gemmwich's intermediate: nao_cart x nao_sphe for cartesian_to_spherical, nao_sphe x
  // nao_cart for spherical_to_cartesian, the same element count either way. Empty when U is the
  // identity.
  Buffers::device<double> work_;
};

} // namespace free_breakfast
