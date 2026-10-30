/// @file
/// @brief The orbital Hessian with the CI relaxed, H~ = H + R, over the packed
/// pairs at fixed orbitals and roots: response::Hessian's H plus the CI's
/// response R, and the orbital-CI coupling R is built from.
///
/// Internal to the MCSCF layer: free_breakfast.mcscf does not re-export it.
/// See README.md.

export module free_breakfast.mcscf.relaxed_hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.ci.solver;
import free_breakfast.response.hessian;

export namespace free_breakfast::mcscf {

/// @brief The integrals the CI response contracts, on the host: F^I
/// (nmo x nmo) and ppaa, papa (casscf_integrals' layout), all column-major as
/// the device's.
struct ResponseIntegrals {
  size_t nmo;
  size_t ncore;
  size_t ncas;
  std::vector<double> fock_inactive;
  std::vector<double> ppaa;
  std::vector<double> papa;

  [[nodiscard]] double fi(const size_t p, const size_t q) const {
    return fock_inactive[p + (nmo * q)];
  }
  /// @brief (pq|uv) and (pu|qv), q over every orbital, u and v active.
  [[nodiscard]] double pq_uv(const size_t p, const size_t q, const size_t u, const size_t v) const {
    return ppaa[p + (nmo * (q + (nmo * (u + (ncas * v)))))];
  }
  [[nodiscard]] double pu_qv(const size_t p, const size_t q, const size_t u, const size_t v) const {
    return papa[p + (nmo * (q + (nmo * (u + (ncas * v)))))];
  }
};

/// @brief The first derivative of the active-space Hamiltonian along
/// @p kappa (nmo x nmo, column-major), C <- C (1 + kappa): h_eff' into @p h1
/// (n x n) and (tu|vw)' into @p eri1 (n^4), both row-major as
/// ci::Sigma::set_integrals takes them. With T = ncore + t,
///   (tu|vw)' = X_tuvw + X_utvw + X_vwtu + X_wvtu, X_tuvw = sum_p kappa_pT (pU|vw),
///   h_eff'_tu = sum_p kappa_pT F^I_pU + kappa_pU F^I_Tp
///             + sum_p,i kappa_pi (4 (pi|tu) - (pt|iu) - (pu|it)),
/// the last F^I's change through the core density, i core: no J or K, since
/// every integral it takes has two active indices.
void response_integrals(const ResponseIntegrals &ints, std::span<const double> kappa,
                        std::span<double> h1, std::span<double> eri1);

/// @brief The adjoint of @ref response_integrals: the antisymmetric nmo x nmo
/// (column-major) @p a = G - G^T, G_pq = d/dkappa_pq of
/// sum_tu h_eff'_tu gamma_tu + sum_tuvw (tu|vw)'_tuvw gamma2_tuvw for the
/// row-major @p gamma and @p gamma2, so that A's packed pairs are that sum's
/// gradient in x (kappa_pq = x_pq = -kappa_qp).
void response_gradient(const ResponseIntegrals &ints, std::span<const double> gamma,
                       std::span<const double> gamma2, std::span<double> a);

/// @brief H~ = H + R, the Hessian of the weighted energy over the packed pairs
/// with the CI relaxed, at the orbitals and roots it is created over.
///
/// R x = H_oc delta, delta_r = -(H_0 - E_r)^-1 (1 - c_r c_r^T) H'(x) c_r by
/// conjugate gradients over the sigma (the other roots' part exactly), and
/// H_oc delta the gradient in x of the transition RDMs' h_eff'.gamma_T +
/// 1/2 (tu|vw)'.Gamma_T (README.md has the derivation).
///
/// Move-only. It keeps pointers to the Hessian, the sigma, the Rdm and the
/// roots it is created over, which must outlive it and not change under it;
/// its own scratch is on @p res's device, which every call must pass again.
class RelaxedHessian {
public:
  /// @brief H~ over @p hessian's H at the CI @p roots of @p sigma's
  /// Hamiltonian H_0 (its integrals set), weighted @p weights, from
  /// @p ints at the same orbitals. @p rdm and a second sigma (for H'(x),
  /// created here) are over @p alpha and @p beta, the roots' strings. A failed
  /// allocation aborts (CreatePolicy).
  ///
  /// @return InvalidConfig when @p ints' sizes do not match their nmo, ncore
  ///         and ncas, @p hessian is not over their pairs, the weights'
  ///         count is not the roots', or the roots are not that many
  ///         vectors of the sigma's length; or the error of the sigma's create.
  [[nodiscard]] static Result<RelaxedHessian>
  create(DeviceResources &res, response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm,
         const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const ci::Roots &roots,
         std::span<const double> weights, ResponseIntegrals ints);

  RelaxedHessian(RelaxedHessian &&) noexcept = default;
  RelaxedHessian &operator=(RelaxedHessian &&) noexcept = default;
  RelaxedHessian(const RelaxedHessian &) = delete;
  RelaxedHessian &operator=(const RelaxedHessian &) = delete;
  ~RelaxedHessian() = default;

  /// @brief y = H~ x for one size() column on the device, @p x and @p y not
  /// overlapping: H's product, then R's added. Synchronizes res.stream() to
  /// read kappa(x) and the transition RDMs back, and before returning.
  ///
  /// @return The first error of H's apply, a sigma, an RDM, a BLAS call or a
  ///         copy, which leaves @p y unspecified.
  [[nodiscard]] Status apply(DeviceResources &res, const double *x, double *y);

  /// @brief The CI response to the column @p x alone: delta_r for every root
  /// into delta() (dim x nroots, column-major, each off every root), and its
  /// transition RDMs on the host, @p gamma_t (n x n) and @p gamma2_t (n^4),
  /// row-major as ci::Rdm::rdm12's: gamma_T = sum_r w_r (<c_r|E|delta_r> +
  /// <delta_r|E|c_r>) and Gamma_T half the same sum's two-body part.
  ///
  /// @return InvalidConfig, with nothing queued, when @p gamma_t or
  ///         @p gamma2_t is not n^2 or n^4 long; or the first error of a
  ///         sigma, an RDM, a BLAS call or a copy.
  [[nodiscard]] Status respond(DeviceResources &res, const double *x, std::span<double> gamma_t,
                               std::span<double> gamma2_t);

  /// @brief The last @ref respond's delta, on the device.
  [[nodiscard]] const double *delta() const noexcept { return ci_delta_.data(); }

  /// @brief The order of H~: the packed pairs' count, H's size().
  [[nodiscard]] int32_t size() const noexcept { return n_pairs_; }

private:
  RelaxedHessian(response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm, const ci::Roots &roots,
                 std::vector<double> weights, ResponseIntegrals ints, ci::Sigma response_sigma,
                 size_t dim, DeviceResources &res);

  response::Hessian *hessian_;
  ci::Sigma *sigma_;
  ci::Rdm *rdm_;
  const ci::Roots *roots_;
  std::vector<double> weights_;
  ResponseIntegrals ints_;
  int32_t n_pairs_;
  size_t dim_;
  // H'(x)'s sigma; the CI blocks b = H' c, delta and c +- delta (dim x nroots
  // each); conjugate gradients' three vectors; the roots' overlaps; kappa,
  // then the gradient's antisymmetric matrix; the packed gradient; and the
  // RDMs of c +- delta.
  ci::Sigma response_sigma_;
  Buffers::device<double> ci_b_;
  Buffers::device<double> ci_delta_;
  Buffers::device<double> ci_pm_;
  Buffers::device<double> cg_vectors_;
  Buffers::device<double> ci_overlaps_;
  Buffers::device<double> response_matrix_;
  Buffers::device<double> response_packed_;
  Buffers::device<double> response_gamma_;
  Buffers::device<double> response_gamma2_;
};

} // namespace free_breakfast::mcscf
