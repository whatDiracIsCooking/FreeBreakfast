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
import free_breakfast.recipes.blas;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.diis;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

#include "diagnostics.h"

namespace free_breakfast {

// Everything the loop holds across iterations, allocated once on res's device:
// the step's private state. The J and K classes are the Fock build's, made
// from the shared PrimitivePairClasses and never published.
struct ScfState {
  std::vector<CoulombPairClass> j_classes;
  std::vector<OrderedPairClass> k_classes;
  SphericalMatrixAO d;       // the density, the iterate
  SphericalMatrixAO d_built; // the density f is the Fock matrix of
  MatrixAOPair dd;           // what J and K read: D on a full build, D - d_built otherwise
  MatrixAOPair g;            // J - K/2 of dd, built Cartesian and carried onto the spherical layout
  CartesianMatrixAO k;       // K, before it is folded into g
  SphericalMatrixAO f;       // the Fock matrix of d_built, as built
  SphericalMatrixAO f_diis;  // DIIS's extrapolation of f
  SphericalMatrixAO e;       // F D S - S D F
  SphericalMatrixAO tmp;     // F D S; then X^T e X; then F' and its eigenvectors
  SphericalMatrixAO work;
  SphericalMatrixAO c;
  Buffers::device<double> eps;
};

// F = H + G(D), G(D) = J(D) - K(D)/2, J and K built from a Cartesian density.
// A full build makes G of D itself; an incremental one adds G(D - d_built) to
// the F of d_built, which G's linearity makes the same matrix. J's and K's
// screens scale with the density they read, and the difference shrinks as the
// SCF converges, so they skip more of it.
[[nodiscard]] Status build_fock(DeviceResources &res, ScfState &st, const SphericalMatrixAO &h,
                                bool full) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(st.d.n());
  const size_t bytes = st.d.num_elements() * sizeof(double);
  if (full) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.dd.spherical().data(), st.d.data(), bytes,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  } else {
    constexpr double kMinusOne = -1.0;
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &kOne<double>, st.d.data(), n, &kMinusOne,
        st.d_built.data(), n, st.dd.spherical().data(), n)));
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.d_built.data(), st.d.data(), bytes,
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(st.dd.spherical_to_cartesian(res));
  FBF_TRY(coulomb(res, st.j_classes, st.dd.cartesian(), st.g.cartesian()));
  FBF_TRY(exchange(res, st.k_classes, st.dd.cartesian(), st.k));
  constexpr double kMinusHalf = -0.5;
  FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(st.k.num_elements()),
                                                &kMinusHalf, st.k.data(), 1,
                                                st.g.cartesian().data(), 1)));
  FBF_TRY(st.g.cartesian_to_spherical(res));
  if (full) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.f.data(), h.data(), bytes,
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
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
  FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, len, st.d.data(), 1, h.data(), 1, &dh)));
  FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, len, st.d.data(), 1, st.f.data(), 1, &df)));
  return 0.5 * (dh + df);
}

// The DIIS residual X^T (F D S - S D F) X into st.tmp, and its Frobenius norm.
// S D F = (F D S)^T for symmetric F, D and S, so one product and a geam.
[[nodiscard]] Result<double> commutator(DeviceResources &res, ScfState &st,
                                        const SphericalMatrixAO &s, const SphericalMatrixAO &x) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(s.n());
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, st.f.data(), st.d.data(), &kZero<double>, st.work.data())));
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

// Solve F C = S C eps through X, @p fock being F: F' = X^T F X = C' eps C'^T,
// C = X C'. Then D = 2 C_occ C_occ^T from the nocc lowest columns.
[[nodiscard]] Status solve_fock(DeviceResources &res, ScfState &st, const SphericalMatrixAO &fock,
                                const SphericalMatrixAO &x, int32_t nocc) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(x.n());
  FBF_TRY(gpu_status(gemmwich(blas, n, x.data(), fock.data(), st.work.data(), st.tmp.data())));
  // syevd's work is dead once it returns, so diagonalize takes it from the
  // shared workspace, not the loop's buffers.
  FBF_TRY(diagonalize<double>(res, n, st.tmp.data(), st.eps.data()));
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, x.data(), st.tmp.data(), &kZero<double>, st.c.data())));
  return gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                               nocc, &kTwo<double>, st.c.data(), n, st.c.data(), n,
                                               &kZero<double>, st.d.data(), n));
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
  if (fock_rebuild_interval < 1) {
    return err_config("[scf] fock_rebuild_interval is {}, below 1", fock_rebuild_interval);
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
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  ScfState st{
      .j_classes = std::move(j_classes),
      .k_classes = std::move(k_classes),
      .d = SphericalMatrixAO(res, basis),
      .d_built = SphericalMatrixAO(res, basis),
      .dd = MatrixAOPair(res, basis, u),
      .g = MatrixAOPair(res, basis, u),
      .k = CartesianMatrixAO(res, basis),
      .f = SphericalMatrixAO(res, basis),
      .f_diis = SphericalMatrixAO(res, basis),
      .e = SphericalMatrixAO(res, basis),
      .tmp = SphericalMatrixAO(res, basis),
      .work = SphericalMatrixAO(res, basis),
      .c = SphericalMatrixAO(res, basis),
      .eps = Buffers::device<double>(n, res.device()),
  };
  Diis diis(res, std::size_t{n} * n, static_cast<size_t>(diis_history));

  // The guess is the step's to upload, and to rescale: SAD places sum Z
  // electrons whatever the charge, and the SCF needs the molecule's own count.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.d.data(), guess.data(), guess.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  if (const int32_t z = molecule.total_nuclear_charge(); z != electrons && z > 0) {
    const double scale = static_cast<double>(electrons) / static_cast<double>(z);
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(
        res.blas().get(), static_cast<int32_t>(guess.size()), &scale, st.d.data(), 1)));
  }

  const double nuclear = nuclear_repulsion_energy(molecule);
  double previous = 0.0;
  int32_t incremental_builds = 0; // since the last full build
  // Set once the commutator is under its tolerance, and every build from then
  // on is full. By then D - d_built is small enough that J's and K's screening
  // error is a good part of its G, and an incremental F gains a fresh error
  // every build: the energy wanders by about that much each iteration and
  // never meets energy_tolerance (H2O/STO-3G: 4.5e-10 Eh a build). A full F's
  // error is a fixed function of D, so the loop keeps a fixed point; and
  // convergence needs a full build anyway.
  bool full_from_now = false;
  for (int32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    const bool full =
        iteration == 1 || full_from_now || incremental_builds + 1 >= fock_rebuild_interval;
    incremental_builds = full ? 0 : incremental_builds + 1;
    FBF_TRY(build_fock(res, st, h, full));
    const double energy = nuclear + FBF_TRY(electronic_energy(res, st, h));
    const double norm = FBF_TRY(commutator(res, st, s, x));
    const double delta = energy - previous;
    previous = energy;
    FBF_LOG_INFO("scf: iteration {:3} {}  E = {:.12f}  dE = {:+.3e}  |[F,D]| = {:.3e}", iteration,
                 full ? "full" : "incr", energy, delta, norm);

    // Only a full build may converge: an incremental F carries the screening
    // error of every build since the last full one.
    const bool converged =
        full && iteration > 1 && std::abs(delta) < energy_tolerance && norm < commutator_tolerance;
    const bool switching = !full_from_now && norm < commutator_tolerance;
    full_from_now = full_from_now || switching;
    if (!converged) {
      // With fewer than two pairs DIIS leaves its output alone: F as built.
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.f_diis.data(), st.f.data(),
                                             st.f.num_elements() * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      FBF_TRY(diis.push_and_extrapolate(res, st.f.data(), st.tmp.data(), st.f_diis.data()));
      // Switching off an incremental build, the history is as stale as its
      // residuals are small: it fits the drifted F's, not the full ones to come.
      // Dropped once, after this step's extrapolation, here and not earlier:
      // a reset further from convergence restarts DIIS where it is still
      // needed (gly5/cc-pVDZ: 3 full builds to finish against 4 kept, 5 and 6
      // reset at 10x and 100x commutator_tolerance).
      if (switching && !full) {
        diis.reset();
      }
    }
    // Converged: the canonical orbitals of the Fock matrix as built, not of an
    // extrapolation, and the density they give.
    FBF_TRY(solve_fock(res, st, converged ? st.f : st.f_diis, x, nocc));
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
                 .d = std::move(st.d)};
    }
  }
  return err_max_iterations_exceeded(
      "scf: not converged in {} iterations (energy_tolerance {:g}, commutator_tolerance {:g})",
      max_iterations, energy_tolerance, commutator_tolerance);
}

} // namespace free_breakfast
