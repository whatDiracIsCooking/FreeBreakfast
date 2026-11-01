/// @file
/// @brief The third derivative of a state-averaged DF-CASSCF energy in its
/// parameters, contracted once with a fixed vector z: T(., v, z) for a block
/// of vectors v, analytic, on the device. The z . d2 g_SA / dlambda dlambda
/// term of a state-averaged (or NEVPT2) nuclear Hessian. Beside it, the same
/// operator's Hessian action along z differentiated in the integrals (I^x z,
/// L_xlambda's z part) and the MO derivative integrals it takes. See
/// README.md.

export module free_breakfast.mcscf.third_derivative;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;

export namespace free_breakfast::mcscf {

/// @brief T(a, v, z) = d3 E_SA / dlambda_a dlambda_v dlambda_z at lambda = 0,
/// E_SA(lambda) = sum_r w_r <c_r(lambda)|H(lambda)|c_r(lambda)> with the
/// orbitals C exp(K) and the roots exp(A) c_r, over the density-fitted
/// integrals: for a fixed z, the covector a -> T(a, v, z) of each vector v.
///
/// A vector over the parameters, and a covector, is one column of size()
/// doubles: the orbital rotation over the packed pairs (response's order,
/// kappa_pq for p > q), then each root's CI change d_r (dim doubles per
/// root, root-major; the rotation is A = sum_r (d_r c_r^T - c_r d_r^T)),
/// then, at unequal weights only, the rotations among the roots, Theta_sr
/// for r < s in (r, s) lexicographic order (c_r gains Theta_sr c_s). The
/// pairing of a covector with a vector is their dot product, so the
/// third-derivative action of a Hessian is Lambda^T Y over the columns
/// `apply` writes.
class ThirdDerivative {
public:
  /// @brief The action at the orbitals @p c and the roots @p roots, contracted
  /// with @p z.
  ///
  /// @p h is the AO core Hamiltonian, @p b the fitted factor (naux x nao^2,
  /// `DensityFitted::b`'s layout) and @p c the orbitals, core, active and
  /// virtual columns in that order, nmo = nao. @p roots is dim x nroots on
  /// the device, dim = alpha.size() beta.size(), nroots = weights.size():
  /// unit, orthogonal, and eigenvectors of the active-space H over these
  /// integrals (a converged SA-CASSCF's). @p z is one column of size() on the
  /// device, its CI part taken off every root here. The rotations among the
  /// roots are parameters when the weights differ pairwise, and are not when
  /// they are all equal.
  ///
  /// Reads every argument in the call and keeps none of them but @p alpha's
  /// and @p beta's device tables, which its sigmas and RDM read in place and
  /// which must outlive it. Builds the MO factor L_Q = C^T B_Q C
  /// (nmo^2 naux doubles, held) and everything z alone fixes. Synchronizes.
  ///
  /// @return The operator; InvalidConfig for ncas < 1, ncore < 0, ncore +
  ///         ncas > nmo, naux < 1, a b, c or h of the wrong size, strings
  ///         over another active space, weights not positive or not summing
  ///         to 1, or weights equal for some pairs of roots and not others;
  ///         or the first error of a BLAS call, copy, sigma or RDM.
  [[nodiscard]] static Result<ThirdDerivative>
  create(DeviceResources &res, const SphericalMatrixAO &h, const Buffers::device<double> &b,
         int32_t naux, const SphericalMatrixAO &c, int32_t ncore, int32_t ncas,
         const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const double *roots,
         std::span<const double> weights, const double *z);

  ThirdDerivative(ThirdDerivative &&) noexcept = default;
  ThirdDerivative &operator=(ThirdDerivative &&) noexcept = default;
  ThirdDerivative(const ThirdDerivative &) = delete;
  ThirdDerivative &operator=(const ThirdDerivative &) = delete;
  ~ThirdDerivative() = default;

  /// @brief Y = T(., V, z) for the size() x @p k column-major device blocks
  /// @p v and @p y: column j of Y is the covector of column j of V. A
  /// column's CI part is taken off every root first. One column at a time,
  /// each synchronizing (the RDMs and the sigmas' integrals pass through
  /// the host).
  ///
  /// @return InvalidConfig for a negative @p k; or the first error of a
  ///         BLAS call, copy, sigma or RDM.
  [[nodiscard]] Status apply(DeviceResources &res, int32_t k, const double *v, double *y);

  /// @brief Y = d/de E_SA''(., z) at the integrals h + e h1 and L + e W, for
  /// the @p k integral directions in @p h1 and @p w1: column j of Y is the
  /// covector a -> d/de E_lambda lambda(a, z), the Hessian of E_SA along z
  /// differentiated along direction j, in apply's layout. With
  /// @c df_integral_derivatives' directions, column x is I^x z, the
  /// state-averaged Hessian at the derivative integrals applied to z.
  ///
  /// @p h1 is k nmo x nmo symmetric MO matrices (slab j at j nmo^2), the
  /// core Hamiltonian's direction; @p w1 is k MO factors (slab j at
  /// j nmo^2 naux, element (p, q, Q) at p + nmo (q + nmo Q), each Q
  /// symmetric in p and q), the direction of L_Q = C^T B_Q C, so that the
  /// integrals move along (pq|rs)' = sum_Q (W_Q,pq L_Q,rs + L_Q,pq W_Q,rs).
  /// Both on the device, in the MO basis of create's @p c. Nothing requires
  /// the integrals moved to keep the roots stationary: the Hessian is the
  /// exact second derivative in the exponential parametrization, its
  /// gradient term and the couplings among the roots included. One column
  /// at a time, each synchronizing.
  ///
  /// @return InvalidConfig for a negative @p k or a null pointer with k > 0;
  ///         or the first error of a BLAS call, copy or sigma.
  [[nodiscard]] Status apply_integrals(DeviceResources &res, int32_t k, const double *h1,
                                       const double *w1, double *y);

  /// @brief The parameter count: pairs + dim nroots (+ nroots (nroots - 1) / 2
  /// at unequal weights).
  [[nodiscard]] int32_t size() const noexcept { return size_; }

private:
  ThirdDerivative(DeviceResources &res, int32_t nmo, int32_t ncore, int32_t ncas, int32_t naux,
                  int32_t dim, std::span<const double> weights, bool rotations, ci::Sigma sigma0,
                  ci::Sigma sigma_z, ci::Sigma sigma_work, ci::Rdm rdm);

  // One parameter vector unpacked: kappa (nmo^2), d (dim x nroots, off the
  // roots), Theta (nroots^2, antisymmetric, device), and A c (dim x nroots).
  Status unpack(DeviceResources &res, const double *v, double *kappa, double *d, double *theta,
                double *rotated);
  // out = A V for the dim x k block v, A of (d, theta): d (C^T V) - C (d^T V)
  // + C (Theta (C^T V)).
  Status rotate(DeviceResources &res, const double *d, const double *theta, int32_t k,
                const double *v, double *out);
  // P[s, t, Q] (nocc^2 x naux) of X: X^T L + L X over the occupied pair, from
  // L X's columns lx (nmo x nocc x naux).
  Status pair_transform(DeviceResources &res, const double *x, const double *lx, double *p);
  // lx[m, r, Q] = sum_n L[m, n, Q] X[n, r], r occupied.
  Status right_transform(DeviceResources &res, const double *x, double *lx);
  // The symmetric transition RDMs of (a, b) at the weights, sum_r w_r
  // (T(a_r, b_r) + T(b_r, a_r)), over every occupied orbital, added into the
  // host nocc^4 Gamma and nocc^2 D.
  Status transition(DeviceResources &res, const double *a, const double *b, std::span<double> gamma,
                    std::span<double> d);
  // H over the occupied integrals h_oo (host, nocc x nocc, column-major) and
  // g = sum_i P1_i P2_i^T (device pair tensors): e_core, h_eff and (tu|vw)
  // into sigma.
  Status set_hamiltonian(DeviceResources &res, std::span<const double> h_oo,
                         std::span<const std::pair<const double *, const double *>> g,
                         ci::Sigma &sigma);
  // The CI half of a covector from the pairs (M, V) of sum M . A_a V: g_d =
  // Q sum [M (C^T V)^T - V (C^T M)^T] (the d_r part) and x_theta = sum (C^T M)
  // (C^T V)^T (the Theta part is x_theta - x_theta^T).
  Status pull_ci(DeviceResources &res,
                 std::span<const std::pair<const double *, const double *>> mv, double *g_d,
                 double *x_theta);

  int32_t nmo_;
  int32_t ncore_;
  int32_t ncas_;
  int32_t nocc_;
  int32_t naux_;
  int32_t dim_;
  int32_t nroots_;
  int32_t pairs_;
  int32_t n_theta_; // nroots (nroots - 1) / 2 at unequal weights, else 0
  int32_t size_;
  std::vector<double> weights_;
  ci::Sigma sigma0_;     // H_0
  ci::Sigma sigma_z_;    // H'[K_z]
  ci::Sigma sigma_work_; // per column
  ci::Rdm rdm_;
  // Held across the sigmas and RDMs, which take the workspace: every buffer
  // below is the operator's own.
  Buffers::device<double> two_w_;   // 2 w_r
  Buffers::device<double> roots_;   // C, dim x nroots
  Buffers::device<double> h_mo_;    // C^T h C
  Buffers::device<double> l_;       // L[m, n, Q]
  Buffers::device<double> lo_;      // L[m, r, Q], r occupied
  Buffers::device<double> p0_;      // L[s, t, Q], s and t occupied
  Buffers::device<double> gamma0_;  // the averaged Gamma over the occupied orbitals
  Buffers::device<double> gamma_z_; // the transition one of A_z c
  Buffers::device<double> d0_;      // the averaged D over the occupied orbitals
  Buffers::device<double> d_z_;     // the transition one of A_z c
  Buffers::device<double> k_z_;     // K_z
  Buffers::device<double> h_k_z_;   // h K_z
  Buffers::device<double> d_zp_;    // z's d, off the roots
  Buffers::device<double> theta_z_; // Theta_z
  Buffers::device<double> rot_z_;   // A_z c
  Buffers::device<double> l_k_z_;   // L K_z, occupied columns
  Buffers::device<double> p_k_z_;   // P of K_z
  Buffers::device<double> y0_;      // Gamma_0 P_0
  Buffers::device<double> y_b_;     // Gamma_0 P(K_z) + Gamma_z P_0
  Buffers::device<double> m1_;      // 2 F at the averaged densities (nmo^2)
  Buffers::device<double> m_z_;     // its derivative along z (nmo^2)
  Buffers::device<double> c1_;      // 2 w_r H c_r
  Buffers::device<double> c_z_;     // its derivative along z
  // One column's scratch.
  Buffers::device<double> mat_;   // nmo^2 matrices: 10
  Buffers::device<double> lmo_;   // nmo nocc naux: 2
  Buffers::device<double> pair_;  // nocc^2 naux: 4
  Buffers::device<double> occ4_;  // nocc^4: 3
  Buffers::device<double> occ2_;  // nocc^2: 2
  Buffers::device<double> ci_;    // dim nroots blocks: 12
  Buffers::device<double> small_; // nroots^2 blocks: 8
  Buffers::device<double> act_;   // gamma and Gamma at a + s b and a - s b
};

/// @brief I^x, the derivative along every nuclear coordinate x = 3 A + k of
/// the MO integrals a DF-CASSCF energy sees at fixed parameters, as the
/// directions @c ThirdDerivative::apply_integrals takes: the core
/// Hamiltonian's h1^x and the MO factor's W^x, the connection
/// tau^x = -1/2 C^T S^x C included (C -> C (1 + tau^x)).
///
///     h1^x = C^T h^x C + tau^x h_MO + h_MO tau^x
///     W^x  = U^x + tau^x L + L tau^x,   U^x = (C^T T^x C) L_M^-T - 1/2 L L_M^-1 M^x L_M^-T
///
/// over the fitted index, h^x the kinetic and nuclear-attraction derivative
/// (@c kinetic_derivative + @c potential_derivative), T^x the three-center
/// integrals' derivative (mn|P)^x, M^x the metric's, L_M the metric's
/// Cholesky factor and L the MO factor, so that the fitted (pq|rs)^x =
/// U^x L^T + L U^x^T exactly, the metric through its factor (trsms, never an
/// inverse). T^x is built one aux atom at a time (@c three_center_derivative)
/// and M^x from @c metric_derivative.
///
/// @p molecule is the molecule both bases were placed on; @p classes
/// @c primitive_pair_classes of @p basis; @p h the AO core Hamiltonian;
/// @p b and @p l the fit as @c density_fit leaves it (B naux x nao^2, L the
/// naux x naux lower triangle); @p c the orbitals, nmo = nao. @p h1 receives
/// 3 n_atom slabs of nmo^2 (slab x at x nmo^2) and @p w1 3 n_atom slabs of
/// nmo^2 naux (slab x at x nmo^2 naux, element (p, q, Q) at
/// p + nmo (q + nmo Q)), both on the device: 3 n_atom (nmo^2 naux) doubles,
/// every coordinate at once.
///
/// Its own memory beside the outputs is the MO factor (nmo^2 naux), one
/// aux atom's derivative block (3 nao^2 nblk), the metric derivative
/// (3 naux^2) and three 3 n_atom nao^2 stacks; past the device's free memory
/// it is Unsupported before anything is allocated. Queued on res.stream()
/// with res.blas(); the derivative calls synchronize it.
///
/// @return Success; InvalidConfig if the molecule and the bases are not one
///         placement, @p c or @p h is not nao x nao, @p b not naux nao^2, or a
///         pointer is null; Unsupported for memory past the device's free
///         bytes or an index past int32_t; or the first error of a
///         derivative call, copy or BLAS call.
[[nodiscard]] Status df_integral_derivatives(DeviceResources &res, const Molecule &molecule,
                                             const BasisSet &basis,
                                             std::span<const PrimitivePairClass> classes,
                                             const BasisSet &aux, const SphericalMatrixAO &h,
                                             const Buffers::device<double> &b, const double *l,
                                             const SphericalMatrixAO &c, double *h1, double *w1);

} // namespace free_breakfast::mcscf
