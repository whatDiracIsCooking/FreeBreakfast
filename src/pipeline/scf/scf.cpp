/// @file
/// @brief ScfStep::run: the step's checks and its two Fock builders, the
/// integral-direct one and the density-fitted one, run by the restricted loop
/// of `:loop`.

module free_breakfast.pipeline.scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.log;
import free_breakfast.recipes.diis;
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.eri.density_fitting.jk;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import :loop;

#include "diagnostics.h"

namespace free_breakfast {

// The integral-direct Fock build, F = H + G(D), G(D) = J(D) - K(D)/2, J and K
// built from a Cartesian density, and its policy: which builds are full. A full
// build makes G of D itself; an incremental one adds G(D - d_built) to the F of
// d_built, which G's linearity makes the same matrix. J's and K's screens scale
// with the density they read, and the difference shrinks as the SCF converges,
// so they skip more of it. The J and K classes are the shared EriOutput's.
//
// An open shell's Fa = H + J(D) - K(Da) and Fb = H + J(D) - K(Db), D = Da + Db,
// each spin's K at weight 1 where the closed shell's D = 2 C C^T takes a half:
// one J and two K calls. An incremental build adds J(D - d_built) to both and
// K(Ds - ds_built) to each, under the same policy, so the spins' builds are
// full together. A closed shell allocates neither spin's built density.
struct DirectFockBuilder {
  const IntegralDirect &eri;
  SphericalMatrixAO d_built; // the density f is the Fock matrix of
  MatrixAOPair dd;           // what J and K read: D on a full build, D - d_built otherwise
  MatrixAOPair g;            // J - K/2 of dd, built Cartesian and carried onto the spherical layout
  CartesianMatrixAO k;       // K, before it is folded into g; an open shell's J, for both spins
  // An open shell's Da and Db, that Fa and Fb are of. dd then also reads Ds or
  // Ds - ds_built, and takes each spin's G = J - K once J and K have read it;
  // g holds that K.
  std::optional<SphericalMatrixAO> da_built{};
  std::optional<SphericalMatrixAO> db_built{};
  double tau_j;
  int32_t fock_rebuild_interval;
  double commutator_tolerance;
  bool built = false;             // any build yet
  bool full = false;              // the last build was
  int32_t incremental_builds = 0; // since the last full build
  // Set once the commutator is under its tolerance, and every build from then
  // on is full. By then D - d_built is small enough that J's and K's screening
  // error is a good part of its G, and an incremental F gains a fresh error
  // every build: the energy wanders by about that much each iteration and
  // never meets energy_tolerance (H2O/STO-3G: 4.5e-10 Eh a build). A full F's
  // error is a fixed function of D, so the loop keeps a fixed point; and
  // convergence needs a full build anyway.
  bool full_from_now = false;

  // The orbitals are not read: J and K are both built from the densities.
  [[nodiscard]] Result<bool> build(DeviceResources &res, const SphericalMatrixAO &d,
                                   const SphericalMatrixAO * /*c*/, const SphericalMatrixAO &h,
                                   SphericalMatrixAO &f, OpenShellState *open) {
    // Only a full build may converge: an incremental F carries the screening
    // error of every build since the last full one.
    full = !built || full_from_now || incremental_builds + 1 >= fock_rebuild_interval;
    built = true;
    incremental_builds = full ? 0 : incremental_builds + 1;
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const auto n = static_cast<int32_t>(d.n());
    const size_t bytes = d.num_elements() * sizeof(double);
    // dd to @p now on a full build, or to now - built_from, both layouts; and
    // built_from to now.
    const auto load = [&](const SphericalMatrixAO &now, SphericalMatrixAO &built_from) -> Status {
      if (full) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dd.spherical().data(), now.data(), bytes,
                                               wwr::wwrMemcpyDeviceToDevice, res.stream())));
      } else {
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, &kOne<double>, now.data(), n,
            &kNegOne<double>, built_from.data(), n, dd.spherical().data(), n)));
      }
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(built_from.data(), now.data(), bytes,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      return dd.spherical_to_cartesian(res);
    };
    // fs = G, Cartesian in @p pair, carried onto the spherical layout, plus H
    // on a full build; or fs += G.
    const auto add_g = [&](MatrixAOPair &pair, SphericalMatrixAO &fs) -> Status {
      FBF_TRY(pair.cartesian_to_spherical(res));
      if (full) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fs.data(), h.data(), bytes,
                                               wwr::wwrMemcpyDeviceToDevice, res.stream())));
      }
      return gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(fs.num_elements()),
                                                   &kOne<double>, pair.spherical().data(), 1,
                                                   fs.data(), 1));
    };
    FBF_TRY(load(d, d_built));
    if (open == nullptr) {
      FBF_TRY(coulomb(res, eri.j_classes, dd.cartesian(), g.cartesian(), tau_j));
      FBF_TRY(exchange(res, eri.k_classes, dd.cartesian(), k));
      FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(k.num_elements()),
                                                    &kNegHalf<double>, k.data(), 1,
                                                    g.cartesian().data(), 1)));
      FBF_TRY(add_g(g, f));
      return full;
    }
    // J into k, kept there for both spins; then each spin's K(dd) into g, and
    // G = J - K into dd, read by then.
    FBF_TRY(coulomb(res, eri.j_classes, dd.cartesian(), k, tau_j));
    const auto nc = static_cast<int32_t>(k.n());
    for (const auto &[fs, ds, ds_built] :
         {std::tuple{&f, &open->da, &*da_built}, std::tuple{&open->fb, &open->db, &*db_built}}) {
      FBF_TRY(load(*ds, *ds_built));
      FBF_TRY(exchange(res, eri.k_classes, dd.cartesian(), g.cartesian()));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, nc, &kOne<double>, k.data(), nc,
          &kNegOne<double>, g.cartesian().data(), nc, dd.cartesian().data(), nc)));
      FBF_TRY(add_g(dd, *fs));
    }
    return full;
  }

  // Switching off an incremental build, the history is as stale as its
  // residuals are small: it fits the drifted F's, not the full ones to come.
  // Dropped once, after that step's extrapolation, and not earlier: a reset
  // further from convergence restarts DIIS where it is still needed
  // (gly5/cc-pVDZ: 3 full builds to finish against 4 kept, 5 and 6 reset at
  // 10x and 100x commutator_tolerance).
  [[nodiscard]] bool reset_diis(double norm) {
    const bool switching = !full_from_now && norm < commutator_tolerance;
    full_from_now = full_from_now || switching;
    return switching && !full;
  }
};

// Eigenvalues of the first build's density at or under this fraction of its
// largest are dropped from its factor. The guess is a sum of atomic densities
// with non-negative occupations, so positive semidefinite, and what this drops
// is syevd's round-off (about n * eps of the largest).
constexpr double kDensityFactorFloor = 1e-12;

// The density-fitted Fock build, F = H + J - K/2: J from D, by df_coulomb, and
// K from a factor C of D = 2 C C^T, by df_exchange, over the EriOutput's B.
// The factor is the occupied orbitals D came from; the first build's D, the
// guess, has none, so its own is made, V (Lambda/2)^{1/2} from D = V Lambda V^T
// over the eigenvalues above kDensityFactorFloor of the largest. Every build is
// full: DF's cost is dense BLAS, which a small density difference does not
// shrink, and K from orbitals is no function of the difference anyway. So DIIS
// is never reset either.
//
// An open shell's Fa = H + J - K(Da) and Fb = H + J - K(Db), J of D = Da + Db:
// df_exchange of C[:, :n_alpha] is K(2 Da), so each spin's K is the same call
// over its own columns, halved as a closed shell's is. The guess's Da = Db = D/2,
// so the first build's K(Da) is K(D)/2, the closed shell's term, and its Kb
// is the same matrix: one factor and one K for both spins.
struct DfFockBuilder {
  const DensityFitted &fit;
  SphericalMatrixAO j;
  SphericalMatrixAO k;
  int32_t n_alpha;
  int32_t n_beta;

  [[nodiscard]] Result<bool> build(DeviceResources &res, const SphericalMatrixAO &d,
                                   const SphericalMatrixAO *c, const SphericalMatrixAO &h,
                                   SphericalMatrixAO &f, OpenShellState *open) {
    SphericalMatrixAO *const fb = open != nullptr ? &open->fb : nullptr;
    const wwr::wwrblasHandle_t blas = res.blas().get();
    const auto n = static_cast<int32_t>(d.n());
    FBF_TRY(df_coulomb(res, n, fit.naux, fit.b.data(), d.data(), j.data()));
    if (c != nullptr) {
      // The loop's C is every MO, ascending: the first n_alpha columns are
      // C_occ, or an open shell's alpha occupied.
      FBF_TRY(df_exchange(res, n, fit.naux, n_alpha, fit.b.data(), c->data(), k.data()));
    } else {
      FBF_TRY(guess_exchange(res, d));
    }
    const size_t bytes = d.num_elements() * sizeof(double);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(f.data(), h.data(), bytes, wwr::wwrMemcpyDeviceToDevice,
                                           res.stream())));
    const auto len = static_cast<int32_t>(f.num_elements());
    FBF_TRY(
        gpu_status(wwr::axpy<double, int32_t>(blas, len, &kOne<double>, j.data(), 1, f.data(), 1)));
    if (fb != nullptr) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fb->data(), f.data(), bytes,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
    }
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, len, &kNegHalf<double>, k.data(), 1, f.data(), 1)));
    if (fb != nullptr) {
      if (c != nullptr) {
        FBF_TRY(df_exchange(res, n, fit.naux, n_beta, fit.b.data(), c->data(), k.data()));
      }
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, len, &kNegHalf<double>, k.data(), 1, fb->data(), 1)));
    }
    return true;
  }

  // K of the guess D, from the factor of its eigendecomposition. Once per run,
  // so its scratch is allocated here and freed stream-ordered at return.
  [[nodiscard]] Status guess_exchange(DeviceResources &res, const SphericalMatrixAO &d) {
    const auto n = static_cast<int32_t>(d.n());
    const size_t nn = d.num_elements();
    Buffers::device<double> v(nn, res.device());
    Buffers::device<double> lambda(static_cast<size_t>(n), res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(v.data(), d.data(), nn * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    FBF_TRY(diagonalize<double>(res, n, v.data(), lambda.data()));
    std::vector<double> host(static_cast<size_t>(n));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), lambda.data(), host.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    // Ascending, so the kept eigenvalues are the last columns of V.
    const double largest = host.empty() ? 0.0 : host.back();
    const double floor = kDensityFactorFloor * std::max(largest, 0.0);
    const auto first = static_cast<int32_t>(std::ranges::upper_bound(host, floor) - host.begin());
    const int32_t rank = n - first;
    if (!host.empty() && host.front() < -floor) {
      // Only a density loaded from a file can be: the SAD guess is PSD. Its
      // negative part is left out of this first K, which only moves the first
      // iterate.
      FBF_LOG_WARN("scf: the first density has an eigenvalue {:.3e}, below -{:.1e} of its "
                   "largest; its exchange is of the positive part",
                   host.front(), kDensityFactorFloor);
    }
    for (int32_t i = first; i < n; ++i) {
      host[static_cast<size_t>(i)] = std::sqrt(0.5 * host[static_cast<size_t>(i)]);
    }
    // C = V (Lambda/2)^{1/2} over the kept columns, in place.
    double *factor = v.data() + (static_cast<size_t>(first) * static_cast<size_t>(n));
    if (rank > 0) {
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(lambda.data(), host.data() + first,
                                             static_cast<size_t>(rank) * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, res.stream())));
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(res.blas().get(), wwr::WWRBLAS_SIDE_RIGHT, n,
                                                    rank, factor, n, lambda.data(), 1, factor, n)));
    }
    FBF_TRY(df_exchange(res, n, fit.naux, rank, fit.b.data(), factor, k.data()));
    // host is read by the upload above.
    return res.sync();
  }

  [[nodiscard]] static bool reset_diis(double /*norm*/) { return false; }
};

[[nodiscard]] Result<ScfOutput>
ScfStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, EriOutput,
                  OverlapOutput, OrthogonalizerOutput, CoreHamiltonianOutput, GuessOutput> in,
             DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const std::vector<double> &guess = in.get<GuessOutput>().d;

  if (max_iterations < 1) {
    return err_config("[scf] max_iterations is {}, below 1", max_iterations);
  }
  if (!(energy_tolerance >= 0.0) || !(commutator_tolerance >= 0.0)) {
    return err_config("[scf] energy_tolerance ({}) and commutator_tolerance ({}) must be "
                      "non-negative",
                      energy_tolerance, commutator_tolerance);
  }
  // Written so that NaN fails too: a NaN tau would never screen, silently.
  if (!(tau_j >= 0.0)) {
    return err_config("[scf] tau_j is {}, but must be 0 or above", tau_j);
  }
  if (fock_rebuild_interval < 1) {
    return err_config("[scf] fock_rebuild_interval is {}, below 1", fock_rebuild_interval);
  }
  if (diis_history < 1 || std::cmp_greater(diis_history, kDiisMaxHistoryLimit)) {
    return err_config("[scf] diis_history is {}, outside 1..{}", diis_history,
                      kDiisMaxHistoryLimit);
  }
  // n_alpha - n_beta = 2S = multiplicity - 1, and N has its parity.
  const int32_t electrons = molecule.electron_count();
  const int32_t n_open = molecule.multiplicity() - 1;
  if (n_open < 0 || electrons < n_open || (electrons - n_open) % 2 != 0) {
    return err_config("scf: {} electrons cannot be at multiplicity {}", electrons,
                      molecule.multiplicity());
  }
  const uint32_t n = basis.nao_sphe();
  const int32_t n_alpha = (electrons + n_open) / 2;
  const int32_t n_beta = (electrons - n_open) / 2;
  if (std::cmp_greater(n_alpha, n)) {
    return err_config("scf: {} electrons at multiplicity {} do not fit {} spatial orbitals",
                      electrons, molecule.multiplicity(), n);
  }
  if (guess.size() != std::size_t{n} * n) {
    return err_config("scf: the guess has {} elements, not {} squared", guess.size(), n);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
  const SphericalMatrixAO &x = in.get<OrthogonalizerOutput>().x;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const EriOutput &eri = in.get<EriOutput>();
  if (const auto *fit = std::get_if<DensityFitted>(&eri.eri)) {
    DfFockBuilder builder{
        .fit = *fit,
        .j = SphericalMatrixAO(res, basis),
        .k = SphericalMatrixAO(res, basis),
        .n_alpha = n_alpha,
        .n_beta = n_beta,
    };
    return scf_loop(res, builder, molecule, basis, s, x, h, guess, n_alpha, n_beta, max_iterations,
                    energy_tolerance, commutator_tolerance, diis_history, molden);
  }
  const std::shared_ptr<const Buffers::device<double>> &u = in.get<SphericalTransformOutput>().u;
  DirectFockBuilder builder{
      .eri = std::get<IntegralDirect>(eri.eri),
      .d_built = SphericalMatrixAO(res, basis),
      .dd = MatrixAOPair(res, basis, u),
      .g = MatrixAOPair(res, basis, u),
      .k = CartesianMatrixAO(res, basis),
      .tau_j = tau_j,
      .fock_rebuild_interval = fock_rebuild_interval,
      .commutator_tolerance = commutator_tolerance,
  };
  if (n_open != 0) {
    builder.da_built.emplace(res, basis);
    builder.db_built.emplace(res, basis);
  }
  return scf_loop(res, builder, molecule, basis, s, x, h, guess, n_alpha, n_beta, max_iterations,
                  energy_tolerance, commutator_tolerance, diis_history, molden);
}

} // namespace free_breakfast
