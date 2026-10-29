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

// C(n, k), the strings of k electrons of one spin, in double, as the space the
// step refuses can pass any integer type: C(n, N/2)^2 is 1.4e12 for N2 cc-pVDZ.
double fci_strings(int32_t n, int32_t electrons) {
  double strings = 1.0;
  for (int32_t k = 0; k < electrons; ++k) {
    strings = strings * static_cast<double>(n - k) / static_cast<double>(k + 1);
  }
  return std::round(strings);
}

// The bytes of ci::Sigma apply's scratch in double over n orbitals with G for
// rows rows: rows (P + 1) for G and one chunk of D, at most 64 MiB.
double fci_sigma_bytes(int32_t n, double rows) {
  const double orbitals = n;
  const double pairs = orbitals * (orbitals + 1.0) / 2.0;
  return (rows * (pairs + 1.0) * sizeof(double)) +
         std::min(static_cast<double>(size_t{64} << 20U), rows * pairs * sizeof(double));
}

// The device bytes the solve over all n orbitals, n_alpha and n_beta
// electrons and nroots roots holds at its peak, besides ci::Sigma apply's
// scratch (fci_sigma_bytes). With s_a = C(n, n_alpha) and s_b = C(n, n_beta)
// strings there are dim = s_a s_b determinants. A singlet solve (n_alpha =
// n_beta, s = s_a) runs on the symmetric layout's packed vectors, of
// t = s (s + 1) / 2 elements; any other on the full layout's, t = dim.
// g = min(t, 2 nroots) guesses, m = max(2 nroots, min(t, 16 nroots)) subspace
// columns, p P-space elements (ci::solve's choices: 0 with the P-space off,
// else max(pspace_size, g), or t once that leaves out fewer than g),
// P = n (n + 1) / 2 packed pairs:
//
//   ci::solve       g t guesses, nroots t vectors, its t penalized diagonal, and
//                  the step's dim diagonal; the P-space's U, p^2, and Λ, p, and
//                  max(p, g) int32 picks; after the Davidson, nroots dim
//                  unpacked roots in its place
//   davidson       2 m t for V and A V, 3 nroots t for A X, R and the
//                  corrections, m^2 + m nroots + m for H, V^T C and theta;
//                  its GD+k collapse works in R and V^T C, adding nothing
//   strings        one DeviceStringSet per spin, or the one both share when
//                  n_alpha = n_beta: s strings (8 bytes) and their links (8
//                  bytes per replacement)
//   ci::Sigma      the P x (P + 1) contraction in double and again in float
//                  (apply_fp32's), n + 2 n^2 terms
//   integrals      (tu|vw) n^4 and h_eff n^2
//
// The scratch is the device workspace, which outlives the Davidson, so it
// is added to whichever of the Davidson and the unpacked roots is more.
double fci_device_bytes(int32_t n, int32_t n_alpha, int32_t n_beta, int32_t nroots) {
  const double orbitals = n;
  const double alpha_strings = fci_strings(n, n_alpha);
  const double beta_strings = fci_strings(n, n_beta);
  const double dim = alpha_strings * beta_strings;
  const double packed = n_alpha == n_beta ? alpha_strings * (alpha_strings + 1.0) / 2.0 : dim;
  const double roots = nroots;
  const double guesses = std::min(packed, 2.0 * roots);
  const double subspace = std::max(2.0 * roots, std::min(packed, 16.0 * roots));
  const double requested = std::max(static_cast<double>(ci::SolverOptions{}.pspace_size), guesses);
  const double pspace = ci::SolverOptions{}.pspace_size <= 0 ? 0.0
                        : packed - requested < guesses       ? packed
                                                             : requested;
  const double pairs = orbitals * (orbitals + 1.0) / 2.0;
  const double alpha_replacements = static_cast<double>(n_alpha) * (orbitals - n_alpha + 1.0);
  const double beta_replacements = static_cast<double>(n_beta) * (orbitals - n_beta + 1.0);

  // ci::solve's own and the step's diagonal throughout; then the Davidson, or
  // after it the unpacked roots, whichever is more.
  const double held = ((guesses + roots + 1.0) * packed) + dim + (pspace * (pspace + 1.0)) +
                      (std::max(pspace, guesses) * sizeof(int32_t) / sizeof(double));
  const double davidson =
      (2.0 * subspace * packed) + (3.0 * roots * packed) + (subspace * (subspace + roots + 1.0));
  const double doubles = held + std::max(davidson, roots * dim) + (pairs * (pairs + 1.0)) +
                         (orbitals + (2.0 * orbitals * orbitals)) + std::pow(orbitals, 4.0) +
                         (orbitals * orbitals);
  const double beta_tables =
      n_alpha == n_beta ? 0.0 : beta_strings * (8.0 + (8.0 * beta_replacements));
  const double tables = (alpha_strings * (8.0 + (8.0 * alpha_replacements))) + beta_tables +
                        (pairs * (pairs + 1.0) * sizeof(float));
  return (doubles * sizeof(double)) + tables;
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
  if (scf.n_alpha != scf.n_beta) {
    return err_unsupported("fci: only a closed-shell reference is implemented, and the "
                           "ScfOutput has {} alpha and {} beta electrons",
                           scf.n_alpha, scf.n_beta);
  }
  const int32_t n_electrons = scf.n_alpha + scf.n_beta;
  const auto nmo = static_cast<int32_t>(scf.mo_energies.size());
  // N_α − N_β = 2S from the molecule's multiplicity 2S + 1, so the strings
  // and ci::solve's spin penalty are the molecule's S.
  const int32_t two_s = molecule.multiplicity() - 1;
  if (two_s < 0 || two_s > n_electrons || (n_electrons - two_s) % 2 != 0 ||
      n_electrons + two_s > 2 * nmo) {
    return err_config("fci: multiplicity {} does not fit {} electrons in {} orbitals: "
                      "N_alpha - N_beta = multiplicity - 1 must have the electron count's parity "
                      "and N_alpha be at most the orbitals",
                      molecule.multiplicity(), n_electrons, nmo);
  }
  const int32_t n_beta = (n_electrons - two_s) / 2;
  const int32_t n_alpha = n_electrons - n_beta;
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  // Refused before anything is allocated: past the device's memory, the
  // first allocation to fail would abort (CreatePolicy), not return. The
  // sigma's scratch is what is left: G for the whole triangle when it fits,
  // else ci::Sigma blocks it, down to one string's rows a block: an alpha
  // string's C(n, n_beta) and a beta string's C(n, n_alpha) in the full layout.
  const double held = fci_device_bytes(nmo, n_alpha, n_beta, nroots);
  const double need =
      held + fci_sigma_bytes(nmo, std::max(fci_strings(nmo, n_alpha), fci_strings(nmo, n_beta)));
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

  // One set serves both spins when N_α = N_β.
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, nmo, n_alpha));
  std::optional<ci::DeviceStringSet> own_beta;
  if (n_alpha != n_beta) {
    own_beta = FBF_TRY(ci::DeviceStringSet::create(res, nmo, n_beta));
  }
  const ci::DeviceStringSet &beta = own_beta ? *own_beta : alpha;
  ci::Sigma sigma = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  // Everything the estimate leaves free, which is G for the whole triangle
  // (and the solve as it was) whenever that fits.
  sigma.set_workspace_limit(static_cast<size_t>(static_cast<double>(free_bytes) - held));
  FBF_TRY(sigma.set_integrals(res, ham.e_core, h, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  // Not ci::solve's defaults: over H2O 6-31G's 1.66M determinants its 1e-12
  // relative residual is past what the Davidson reaches (it stalls near 1.2e-10
  // absolute, 125 iterations in, as Divergence), and its 100 iterations are
  // not enough for 1e-11 (112 were taken). 1e-11 still leaves the energies'
  // error second order in the residual, ~1e-19 hartree.
  ci::SolverOptions options;
  options.davidson.residual_tolerance = 1e-11;
  options.davidson.max_iterations = 400;
  ci::Roots roots =
      FBF_TRY(ci::solve(res, sigma, alpha, beta, diagonal_device.data(), nroots, options));

  const auto count = static_cast<size_t>(nroots);
  // Each root's 1-RDM on the device, from the roots ci::solve left there.
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
