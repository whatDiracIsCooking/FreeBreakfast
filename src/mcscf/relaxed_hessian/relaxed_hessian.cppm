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
  /// allocation aborts (CreatePolicy). The CI response's conjugate gradients
  /// stop at @p response_tolerance of their right-hand side's norm (or after
  /// 200 sigmas per root): every product is good to about that, relative.
  ///
  /// @return InvalidConfig when @p ints' sizes do not match their nmo, ncore
  ///         and ncas, @p hessian is not over their pairs, the weights'
  ///         count is not the roots', the roots are not that many vectors of
  ///         the sigma's length, or @p response_tolerance is not positive; or
  ///         the error of the sigma's create.
  [[nodiscard]] static Result<RelaxedHessian>
  create(DeviceResources &res, response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm,
         const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const ci::Roots &roots,
         std::span<const double> weights, ResponseIntegrals ints, double response_tolerance = 1e-8);

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

  /// @brief The response to a CI right-hand side @p ci_rhs (dim x nroots,
  /// column-major, on the device): w_r = -(H_0 - E_r)^-1 P y_r into delta(),
  /// each y_r projected off every root first, by @ref respond's conjugate
  /// gradients (y must be in the roots' spin); and @p y = H_oc w over the packed pairs, overwritten: the
  /// gradient in x of w's transition RDMs' h_eff'.gamma_T +
  /// 1/2 (tu|vw)'.Gamma_T, as R x is of delta(x)'s. Synchronizes
  /// res.stream() to read the transition RDMs back, and before returning.
  ///
  /// @return The first error of a sigma, an RDM, a BLAS call or a copy,
  ///         which leaves delta() and @p y unspecified.
  [[nodiscard]] Status couple(DeviceResources &res, const double *ci_rhs, double *y);

  /// @brief Y = H~ X for the size() x @p k column-major device blocks @p x and
  /// @p y, not overlapping: H's product over the block in one call, H'(x_j) c
  /// per column, then every column's CI response at once, the conjugate
  /// gradients over H_0's sigma in lockstep across the k nroots vectors (one
  /// sigma of the block per step), and R x_j per column. Each column is
  /// @ref apply's to the CI response's tolerance, not to the bit: the inner
  /// steps sum in another order. Leaves the block's delta, k blocks of
  /// dim x nroots, in delta(). Synchronizes res.stream() per column and per
  /// inner step.
  ///
  /// @return InvalidConfig for a negative @p k; or as @ref apply.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *x, double *y);

  /// @brief @ref couple for @p k columns at once: @p ci_rhs is k blocks of
  /// dim x nroots, w their k blocks into delta() by the lockstep conjugate
  /// gradients of the block @ref apply, and @p y (size() x k) H_oc w_j per
  /// column, overwritten.
  ///
  /// @return InvalidConfig for a negative @p k; or as @ref couple.
  [[nodiscard]] Status couple(DeviceResources &res, int32_t k, const double *ci_rhs, double *y);

  /// @brief The unreduced operator over (x, x_c) for @p k columns, x the
  /// size() x k orbital block and x_c k blocks of dim x nroots, each off every
  /// root (not checked):
  ///   y   = H x + H_oc x_c,
  ///   y_c,r = P (H'(x) c_r + (H_0 - E_r) x_c,r),   P = 1 - sum_s c_s c_s^T,
  /// the CI rows unweighted, as zvector's equations (its README.md "The CI
  /// right-hand side"). H's product in one call, the sigma of every x_c,r in
  /// one, H'(x_j) c and H_oc x_c,j per column. No CI response is solved.
  /// With one string set for both spins, y_c is then put exactly in the
  /// roots' sector under the swap of the two strings (README.md "Block
  /// calls"); x_c should be in it too.
  /// None of the four blocks may overlap. Synchronizes res.stream() per column.
  ///
  /// @p theta, when not empty, adds the rotations among the roots (README.md
  /// "The rotations among the roots"): per column the nroots (nroots - 1) / 2
  /// Theta_sr for r < s in (r, s) lexicographic order, on the host, c_r
  /// gaining Theta_sr c_s and c_s losing Theta_sr c_r. They join x_c as its
  /// part along the roots, so y gains H_oc of it, and @p y_theta (as
  /// @p theta) receives their rows, the CI rows' components along the other
  /// roots before the projection:
  ///   y_theta_sr = c_s . H'(x) c_r + (E_s - E_r) Theta_sr,
  /// unscaled as the CI rows are.
  ///
  /// @return InvalidConfig for a negative @p k, or a @p theta or @p y_theta
  ///         that is not empty and not k such columns; or the first error of
  ///         H's apply, a sigma, an RDM, a BLAS call or a copy.
  [[nodiscard]] Status apply_coupled(DeviceResources &res, int32_t k, const double *x,
                                     const double *x_c, double *y, double *y_c,
                                     std::span<const double> theta = {},
                                     std::span<double> y_theta = {});

  /// @brief y = H_oc x_c for @p k CI blocks @p x_c (dim x nroots each, on the
  /// device) into the size() x @p k @p y, overwritten: the gradient in x of
  /// x_c's transition RDMs' h_eff'.gamma_T + 1/2 (tu|vw)'.Gamma_T, as R x is
  /// of delta(x)'s, with no CI solve. x_c may lie along the roots: a part
  /// sum_s Theta_sr c_s of x_c,r is the rotations' coupling to the orbitals,
  /// (w_r - w_s) Theta_sr times the r-s transition orbital gradient.
  /// Synchronizes res.stream() per column.
  ///
  /// @return InvalidConfig for a negative @p k; or the first error of an
  ///         RDM, a BLAS call or a copy.
  [[nodiscard]] Status coupling(DeviceResources &res, int32_t k, const double *x_c, double *y);

  /// @brief diag(H_0) into the dim device doubles @p diagonal: the sigma's
  /// @c ci::Sigma::diagonal, the roots' energies not subtracted.
  [[nodiscard]] Status ci_diagonal(DeviceResources &res, double *diagonal);

  /// @brief The last @ref respond's (or @ref couple's, or a block call's)
  /// delta, on the device: one dim x nroots block per column.
  [[nodiscard]] const double *delta() const noexcept { return ci_delta_.data(); }

  /// @brief The order of H~: the packed pairs' count, H's size().
  [[nodiscard]] int32_t size() const noexcept { return n_pairs_; }

  /// @brief The weights it was created with, one per root.
  [[nodiscard]] std::span<const double> weights() const noexcept { return weights_; }

private:
  RelaxedHessian(response::Hessian &hessian, ci::Sigma &sigma, ci::Rdm &rdm, const ci::Roots &roots,
                 std::vector<double> weights, ResponseIntegrals ints, ci::Sigma response_sigma,
                 size_t dim, double response_tolerance, int32_t swap_parity, DeviceResources &res);

  // v <- v - C (C^T v): off every root.
  [[nodiscard]] Status project(DeviceResources &res, double *v);
  // delta_r = -(H_0 - E_r)^-1 (1 - c_r c_r^T) b_r for b = ci_b_, into
  // ci_delta_, and its transition RDMs (respond's).
  [[nodiscard]] Status solve_response(DeviceResources &res, std::span<double> gamma_t,
                                      std::span<double> gamma2_t);
  // y += the packed gradient in x of h_eff'.gamma_t + (tu|vw)'.gamma2_t.
  [[nodiscard]] Status add_coupling(DeviceResources &res, std::span<const double> gamma_t,
                                    std::span<const double> gamma2_t, double *y);
  // b_r = H'(x) c_r for every root, b dim x nroots.
  [[nodiscard]] Status prime_sigma(DeviceResources &res, const double *x, double *b);
  // gamma_T and Gamma_T of the CI block delta (dim x nroots), on the host.
  [[nodiscard]] Status transition_rdms(DeviceResources &res, const double *delta,
                                       std::span<double> gamma_t, std::span<double> gamma2_t);
  // ci_b_ and ci_delta_ to k columns, and block_ to the lockstep scratch of
  // k columns' vectors.
  void reserve(DeviceResources &res, size_t k);
  // v <- v - C (C^T v) for the count vectors of v (dim x count).
  [[nodiscard]] Status project_block(DeviceResources &res, size_t count, double *v);
  // y_c <- (H_0 - E_r) x_c for k columns' blocks (vector j nroots + r at
  // E_r), not projected: one sigma of the block.
  [[nodiscard]] Status shifted_sigma(DeviceResources &res, size_t k, const double *x_c,
                                     double *y_c);
  // The CI responses of k columns' b = ci_b_ into ci_delta_, lockstep.
  [[nodiscard]] Status solve_responses(DeviceResources &res, size_t k);
  // y_j += H_oc delta_j for k columns' CI blocks @p delta.
  [[nodiscard]] Status add_couplings(DeviceResources &res, size_t k, const double *delta,
                                     double *y);

  response::Hessian *hessian_;
  ci::Sigma *sigma_;
  ci::Rdm *rdm_;
  const ci::Roots *roots_;
  std::vector<double> weights_;
  ResponseIntegrals ints_;
  int32_t n_pairs_;
  size_t dim_;
  double response_tolerance_;
  // The roots' parity under the swap of the alpha and beta strings (one set
  // for both spins): +1 even S, -1 odd, 0 for two sets or no definite one.
  int32_t swap_parity_;
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
  // The block calls' lockstep scratch: r, p, A p and a product (dim x count
  // each, count = k nroots), the count shifts E_r, the 2 count dots and the
  // count step lengths, and the nroots x count overlaps with the roots.
  Buffers::device<double> block_;
};

} // namespace free_breakfast::mcscf
