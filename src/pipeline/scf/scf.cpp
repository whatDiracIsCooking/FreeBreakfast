/// @file
/// @brief ScfStep::run: the RHF loop, its Fock build, the eigensolve and the
/// density from the occupied orbitals.

module free_breakfast.pipeline.scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.linalg.blas;
import free_breakfast.linalg.diis;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

#include "diagnostics.h"

namespace free_breakfast {

// syevd's eigenvector mode and triangle, shared by the workspace query and the
// call so the two cannot disagree. F' is symmetric, so either triangle does.
constexpr auto kScfEigMode = wwr::WWRSOLVER_EIG_MODE_VECTOR;
constexpr auto kScfFillMode = wwr::WWRBLAS_FILL_MODE_LOWER;

// Everything the loop holds across iterations, allocated once on res's device:
// the step's private state. The J and K classes are the Fock build's, made
// from the shared PrimitivePairClasses and never published.
struct ScfState {
  std::vector<CoulombPairClass> j_classes;
  std::vector<OrderedPairClass> k_classes;
  MatrixAOPair d;        // the density: spherical is the iterate, Cartesian what J and K read
  MatrixAOPair g;        // J - K/2, built Cartesian and carried onto the spherical layout
  CartesianMatrixAO k;   // K, before it is folded into g
  SphericalMatrixAO f;   // the Fock matrix, then DIIS's extrapolation of it
  SphericalMatrixAO e;   // F D S - S D F
  SphericalMatrixAO tmp; // F D S; then X^T e X; then F' and its eigenvectors
  SphericalMatrixAO work;
  SphericalMatrixAO c;
  Buffers::device<double> eps;
  int32_t lwork; // syevd's work, in doubles: taken from res's workspace per call
};

// F = H + J(D) - K(D)/2, J and K built from D's Cartesian form.
[[nodiscard]] Status build_fock(DeviceResources &res, ScfState &st, const SphericalMatrixAO &h) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(st.d.spherical_to_cartesian(res));
  FBF_TRY(coulomb(res, st.j_classes, st.d.cartesian(), st.g.cartesian()));
  FBF_TRY(exchange(res, st.k_classes, st.d.cartesian(), st.k));
  constexpr double kMinusHalf = -0.5;
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(st.k.num_elements()),
                                                &kMinusHalf, st.k.data(), 1,
                                                st.g.cartesian().data(), 1)));
  FBF_TRY(st.g.cartesian_to_spherical(res));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.f.data(), h.data(), h.num_elements() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  return gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(st.f.num_elements()),
                                               &kOne<double>, st.g.spherical().data(), 1,
                                               st.f.data(), 1));
}

// E_elec = tr(D (H + F)) / 2. Two dots into host scalars, each of which
// synchronizes the stream.
[[nodiscard]] Result<double> electronic_energy(DeviceResources &res, const ScfState &st,
                                               const SphericalMatrixAO &h) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto len = static_cast<int32_t>(h.num_elements());
  double dh = 0.0;
  double df = 0.0;
  // D, H and F are symmetric, so tr(D M) is their elementwise dot.
  FBF_TRY(gpu_status(
      wwr::dot<double, int32_t>(blas, len, st.d.spherical().data(), 1, h.data(), 1, &dh)));
  FBF_TRY(gpu_status(
      wwr::dot<double, int32_t>(blas, len, st.d.spherical().data(), 1, st.f.data(), 1, &df)));
  return 0.5 * (dh + df);
}

// The DIIS residual X^T (F D S - S D F) X into st.tmp, and its Frobenius norm.
// S D F = (F D S)^T for symmetric F, D and S, so one product and a geam.
[[nodiscard]] Result<double> commutator(DeviceResources &res, ScfState &st,
                                        const SphericalMatrixAO &s, const SphericalMatrixAO &x) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(s.n());
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, st.f.data(), st.d.spherical().data(),
                          &kZero<double>, st.work.data())));
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, st.work.data(), s.data(), &kZero<double>, st.tmp.data())));
  constexpr double kMinusOne = -1.0;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kOne<double>, st.tmp.data(), n, &kMinusOne,
                                                st.tmp.data(), n, st.e.data(), n)));
  FBF_TRY(gpu_status(gemmwich(blas, n, x.data(), st.e.data(), st.work.data(), st.tmp.data())));
  double norm = 0.0;
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, static_cast<int32_t>(st.tmp.num_elements()),
                                                st.tmp.data(), 1, &norm)));
  return norm;
}

// Solve F C = S C eps through X: F' = X^T F X = C' eps C'^T, C = X C'. Then
// D = 2 C_occ C_occ^T from the nocc lowest columns.
[[nodiscard]] Status diagonalize(DeviceResources &res, ScfState &st, const SphericalMatrixAO &x,
                                 int32_t nocc) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(x.n());
  FBF_TRY(gpu_status(gemmwich(blas, n, x.data(), st.f.data(), st.work.data(), st.tmp.data())));
  // syevd's work is dead once it returns, so it is the shared workspace's, not
  // the loop's. In a pipeline the Orthogonalizer has already asked for more on
  // the same n, so this request does not grow it.
  auto eig_work = wwr::extension::reinterpret_buffer_view<double>(
      res.device_workspace(static_cast<size_t>(st.lwork) * sizeof(double)));
  FBF_TRY(gpu_status(wwr::syevd<double>(res.solver().get(), kScfEigMode, kScfFillMode, n,
                                        st.tmp.data(), n, st.eps.data(), eig_work.data(), st.lwork,
                                        res.info().data())));
  const int32_t info = FBF_TRY(res.fetch_info());
  // info < 0 names a bad argument (a bug here); info > 0 is non-convergence.
  if (info < 0) {
    return err_gpu("scf: syevd rejected argument {}", -info);
  }
  if (info > 0) {
    return err_numerical("scf: syevd did not converge on the Fock matrix (info = {})", info);
  }
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, x.data(), st.tmp.data(), &kZero<double>, st.c.data())));
  return gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                               nocc, &kTwo<double>, st.c.data(), n, st.c.data(), n,
                                               &kZero<double>, st.d.spherical().data(), n));
}

[[nodiscard]] Result<Scf>
ScfStep::run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses, Overlap,
                  Orthogonalizer, CoreHamiltonian, Guess> in,
             DeviceResources &res) const {
  const Molecule &molecule = in.get<Molecule>();
  const BasisSet &basis = in.get<BasisSet>();
  const SphericalMatrixAO &s = in.get<Overlap>().s;
  const SphericalMatrixAO &x = in.get<Orthogonalizer>().x;
  const SphericalMatrixAO &h = in.get<CoreHamiltonian>().h;
  const std::vector<double> &guess = in.get<Guess>().d;

  if (max_iterations < 1) {
    return err_config("[scf] max_iterations is {}, below 1", max_iterations);
  }
  if (!(energy_tolerance >= 0.0) || !(commutator_tolerance >= 0.0)) {
    return err_config("[scf] energy_tolerance ({}) and commutator_tolerance ({}) must be "
                      "non-negative",
                      energy_tolerance, commutator_tolerance);
  }
  if (diis_history < 1 || std::cmp_greater(diis_history, kDiisMaxHistoryLimit)) {
    return err_config("[scf] diis_history is {}, outside 1..{}", diis_history,
                      kDiisMaxHistoryLimit);
  }
  const int32_t electrons = molecule.electron_count();
  if (molecule.multiplicity() != 1 || electrons % 2 != 0) {
    return err_unsupported("scf: only closed-shell RHF is implemented, and the molecule has {} "
                           "electrons at multiplicity {}",
                           electrons, molecule.multiplicity());
  }
  const uint32_t n = basis.nao_sphe();
  const int32_t nocc = electrons / 2;
  if (electrons < 0 || std::cmp_greater(nocc, n)) {
    return err_config("scf: {} electrons do not fit {} spatial orbitals", electrons, n);
  }
  if (guess.size() != std::size_t{n} * n) {
    return err_config("scf: the guess has {} elements, not {} squared", guess.size(), n);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const std::shared_ptr<const Buffers::device<double>> &u = in.get<SphericalTransform>().u;
  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClasses>().classes;
  int32_t lwork = 0;
  FBF_TRY(gpu_status(wwr::syevd_bufferSize<double>(res.solver().get(), kScfEigMode, kScfFillMode,
                                                   static_cast<int32_t>(n), nullptr,
                                                   static_cast<int32_t>(n), nullptr, &lwork)));
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  ScfState st{
      .j_classes = std::move(j_classes),
      .k_classes = std::move(k_classes),
      .d = MatrixAOPair(res, basis, u),
      .g = MatrixAOPair(res, basis, u),
      .k = CartesianMatrixAO(res, basis),
      .f = SphericalMatrixAO(res, basis),
      .e = SphericalMatrixAO(res, basis),
      .tmp = SphericalMatrixAO(res, basis),
      .work = SphericalMatrixAO(res, basis),
      .c = SphericalMatrixAO(res, basis),
      .eps = Buffers::device<double>(n, res.device()),
      .lwork = lwork,
  };
  Diis diis(res, std::size_t{n} * n, static_cast<size_t>(diis_history));

  // The guess is the step's to upload, and to rescale: SAD places sum Z
  // electrons whatever the charge, and the SCF needs the molecule's own count.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.d.spherical().data(), guess.data(),
                                         guess.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                         res.stream())));
  if (const int32_t z = molecule.total_nuclear_charge(); z != electrons && z > 0) {
    const double scale = static_cast<double>(electrons) / static_cast<double>(z);
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(
        res.blas().get(), static_cast<int32_t>(guess.size()), &scale, st.d.spherical().data(), 1)));
  }

  const double nuclear = nuclear_repulsion_energy(molecule);
  double previous = 0.0;
  for (int32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    FBF_TRY(build_fock(res, st, h));
    const double energy = nuclear + FBF_TRY(electronic_energy(res, st, h));
    const double norm = FBF_TRY(commutator(res, st, s, x));
    const double delta = energy - previous;
    previous = energy;
    FBF_LOG_INFO("scf: iteration {:3}  E = {:.12f}  dE = {:+.3e}  |[F,D]| = {:.3e}", iteration,
                 energy, delta, norm);

    const bool converged =
        iteration > 1 && std::abs(delta) < energy_tolerance && norm < commutator_tolerance;
    if (!converged) {
      // Extrapolated in place: with fewer than two pairs F is left as built.
      FBF_TRY(diis.push_and_extrapolate(res, st.f.data(), st.tmp.data(), st.f.data()));
    }
    // Converged: the canonical orbitals of the Fock matrix as built, not of an
    // extrapolation, and the density they give.
    FBF_TRY(diagonalize(res, st, x, nocc));
    if (converged) {
      std::vector<double> mo_energies(n);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(mo_energies.data(), st.eps.data(), n * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, res.stream())));
      FBF_TRY(res.sync());
      FBF_LOG_INFO("scf: converged in {} iterations, E = {:.12f}", iteration, energy);
      return Scf{.energy = energy,
                 .iterations = iteration,
                 .mo_energies = std::move(mo_energies),
                 .c = std::move(st.c),
                 .d = std::move(st.d.spherical())};
    }
  }
  return err_max_iterations_exceeded(
      "scf: not converged in {} iterations (energy_tolerance {:g}, commutator_tolerance {:g})",
      max_iterations, energy_tolerance, commutator_tolerance);
}

} // namespace free_breakfast
