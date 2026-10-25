/// @file
/// @brief FciStep: the device-memory estimate, then CasciStep's path over
/// every orbital and electron: the active-space Hamiltonian, the CI solve and
/// the 1-RDMs.

module free_breakfast.pipeline.fci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.solver;

#include "diagnostics.h"

namespace free_breakfast {

// The device bytes the solve over all n orbitals, n_alpha electrons of each
// spin and nroots roots holds at its peak, inside ci_solve's Davidson. In
// double, as the space it refuses can pass any integer type: C(n, n_alpha)^2
// is 1.4e12 for N2 cc-pVDZ. A singlet solve runs on the symmetric layout's
// packed vectors, s = C(n, n_alpha) strings giving dim = s^2 determinants and
// t = s (s + 1) / 2 packed elements; g = min(t, 2 nroots) guesses,
// m = max(2 nroots, min(t, 16 nroots)) subspace columns (ci_solve's choices),
// P = n (n + 1) / 2 packed pairs:
//
//   ci_solve       g t guesses, nroots t vectors, its t penalized diagonal, and
//                  the step's dim diagonal; after the Davidson, nroots dim
//                  unpacked roots in its place
//   davidson       2 m t for V and A V, 3 nroots t for A X, R and the
//                  corrections, m^2 + m nroots + m for H, V^T C and theta;
//                  its GD+k collapse works in R and V^T C, adding nothing
//   CiSigma apply  t (P + 1) for G, one vector's (apply goes a vector at a
//                  time), and one chunk of D, at most 64 MiB
//   CiSigma tables two sets of strings (8 bytes) and links (12 bytes per
//                  replacement), the P x (P + 1) contraction in double and
//                  again in float (apply_fp32's), n + 2 n^2 terms
//   integrals      (tu|vw) n^4 and h_eff n^2
double fci_device_bytes(int32_t n, int32_t n_alpha, int32_t nroots) {
  double strings = 1.0;
  for (int32_t k = 0; k < n_alpha; ++k) {
    strings = strings * static_cast<double>(n - k) / static_cast<double>(k + 1);
  }
  strings = std::round(strings);
  const double dim = strings * strings;
  const double packed = strings * (strings + 1.0) / 2.0;
  const double roots = nroots;
  const double guesses = std::min(packed, 2.0 * roots);
  const double subspace = std::max(2.0 * roots, std::min(packed, 16.0 * roots));
  const double orbitals = n;
  const double pairs = orbitals * (orbitals + 1.0) / 2.0;
  const double replacements = static_cast<double>(n_alpha) * (orbitals - n_alpha + 1.0);

  // ci_solve's own and the step's diagonal throughout; then the Davidson and
  // the sigma's G, or after it the unpacked roots, whichever is more.
  const double held = ((guesses + roots + 1.0) * packed) + dim;
  const double davidson = (2.0 * subspace * packed) + (3.0 * roots * packed) +
                          (subspace * (subspace + roots + 1.0)) + (packed * (pairs + 1.0));
  const double doubles = held + std::max(davidson, roots * dim) + (pairs * (pairs + 1.0)) +
                         (orbitals + (2.0 * orbitals * orbitals)) + std::pow(orbitals, 4.0) +
                         (orbitals * orbitals);
  const double chunk =
      std::min(static_cast<double>(size_t{64} << 20U), packed * pairs * sizeof(double));
  const double tables =
      2.0 * strings * (8.0 + (12.0 * replacements)) + (pairs * (pairs + 1.0) * sizeof(float));
  return (doubles * sizeof(double)) + chunk + tables;
}

[[nodiscard]] Result<FciOutput>
FciStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                  PrimitivePairClassesOutput, CoreHamiltonianOutput, ScfOutput> in,
             DeviceResources &res) const {
  if (nroots < 1) {
    return err_config("fci: nroots is {}, not at least 1", nroots);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const ScfOutput &scf = in.get<ScfOutput>();
  const int32_t n_electrons = molecule.electron_count();
  const auto nmo = static_cast<int32_t>(scf.mo_energies.size());
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  // Refused before anything is allocated: past the device's memory, the
  // first allocation to fail would abort (CreatePolicy), not return.
  const double need = fci_device_bytes(nmo, n_electrons / 2, nroots);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (need > static_cast<double>(free_bytes)) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    return err_allocation("fci: {} electrons in {} orbitals, {} roots, need an estimated {:.3g} "
                          "GiB of device memory, and device {} has {:.3g} GiB of {:.3g} free",
                          n_electrons, nmo, nroots, need / kGiB, res.dev_idx(),
                          static_cast<double>(free_bytes) / kGiB,
                          static_cast<double>(total_bytes) / kGiB);
  }

  // From here CasciStep::run's path with no core and every orbital active,
  // so the ScfOutput's C is used as it is: its orbital order is the identity.
  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  const ActiveSpaceHamiltonian ham = FBF_TRY(active_space_hamiltonian(
      res, basis, molecule, in.get<CoreHamiltonianOutput>().h, in.get<SphericalTransformOutput>().u,
      j_classes, k_classes, scf.c, 0, nmo));

  // The sigma takes host arrays, row-major: h transposed, and (pq|rs) from
  // the Hamiltonian's p-fastest order to s-fastest.
  const auto n = static_cast<size_t>(nmo);
  std::vector<double> h_col(n * n);
  std::vector<double> eri_col(n * n * n * n);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(h_col.data(), ham.h_eff.data(), h_col.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri_col.data(), ham.eri.data(),
                                         eri_col.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<double> h(n * n);
  std::vector<double> eri(n * n * n * n);
  for (size_t p = 0; p < n; ++p) {
    for (size_t q = 0; q < n; ++q) {
      h[(p * n) + q] = h_col[p + (n * q)];
      for (size_t r = 0; r < n; ++r) {
        for (size_t s = 0; s < n; ++s) {
          eri[(((((p * n) + q) * n) + r) * n) + s] = eri_col[p + (n * (q + (n * (r + (n * s)))))];
        }
      }
    }
  }

  const StringSet alpha = FBF_TRY(StringSet::create(nmo, n_electrons / 2));
  const StringSet beta = FBF_TRY(StringSet::create(nmo, n_electrons / 2));
  CiSigma sigma = FBF_TRY(CiSigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ham.e_core, h, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  // Not ci_solve's defaults: over H2O 6-31G's 1.66M determinants its 1e-12
  // relative residual is past what the Davidson reaches (it stalls near 1.2e-10
  // absolute, 125 iterations in, as Divergence), and its 100 iterations are
  // not enough for 1e-11 (112 were taken). 1e-11 still leaves the energies'
  // error second order in the residual, ~1e-19 hartree.
  CiSolverOptions options;
  options.davidson.residual_tolerance = 1e-11;
  options.davidson.max_iterations = 400;
  CiRoots roots =
      FBF_TRY(ci_solve(res, sigma, alpha, beta, diagonal_device.data(), nroots, options));

  const auto count = static_cast<size_t>(nroots);
  // Each root's 1-RDM on the device, from the roots ci_solve left there.
  std::vector<double> gamma(count * n * n);
  Buffers::device<double> gamma_device(gamma.size(), res.device());
  for (size_t r = 0; r < count; ++r) {
    FBF_TRY(sigma.rdm1(res, roots.vectors.data() + (r * dim), gamma_device.data() + (r * n * n)));
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), gamma_device.data(),
                                         gamma.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         stream)));
  FBF_TRY(res.sync());
  return FciOutput{
      .nmo = nmo,
      .n_electrons = n_electrons,
      .energies = std::move(roots.energies),
      .spin_squares = std::move(roots.spin_squares),
      .rdm1 = std::move(gamma),
      .davidson = roots.davidson,
  };
}

} // namespace free_breakfast
