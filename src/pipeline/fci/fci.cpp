/// @file
/// @brief FciStep: its footprint against the memory budget, then CasciStep's path over
/// every orbital and electron: the active-space Hamiltonian, the CI solve and
/// the 1-RDMs.

module free_breakfast.pipeline.fci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives;     // math::binomial
import free_breakfast.recipes.diagonalize; // syevd_lwork
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.pipeline.active_space; // active_space_ci_footprint
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.solver;

#include "diagnostics.h"

namespace free_breakfast {

// C(n, k), the strings of k electrons of one spin, in double, as the space the
// step refuses can pass any integer type: C(n, N/2)^2 is 1.4e12 for N2 cc-pVDZ.
double fci_strings(int32_t n, int32_t electrons) {
  return std::round(math::binomial(n, electrons));
}

// ci::solve's options here, not its defaults: over H2O 6-31G's 1.66M
// determinants its 1e-12 relative residual is past what the Davidson reaches
// (it stalls near 1.2e-10 absolute, 125 iterations in, as Divergence), and its
// 100 iterations are not enough for 1e-11 (112 were taken). 1e-11 still leaves
// the energies' error second order in the residual, ~1e-19 hartree.
ci::SolverOptions fci_solver_options() {
  ci::SolverOptions options;
  options.davidson.residual_tolerance = 1e-11;
  options.davidson.max_iterations = 400;
  return options;
}

// What FciStep::run holds over @p dims, every orbital active and every
// electron, @p nroots roots, ci::Sigma's scratch under @p limit: J's and K's
// classes and the Hamiltonian, freed once it is on the host; then the CI.
MemoryFootprint fci_footprint(const FootprintDimensions &dims, const int32_t nroots,
                              const size_t limit) {
  const BasisSet &basis = *dims.basis;
  const auto nmo = static_cast<int32_t>(dims.nao);
  const auto n_alpha = static_cast<int32_t>(dims.n_alpha);
  const auto n_beta = static_cast<int32_t>(dims.n_beta);
  const MemoryFootprint hamiltonian = drop(then(
      then(coulomb_pair_classes_footprint(dims.mmd_records), ordered_pair_classes_footprint(basis)),
      active_space_hamiltonian_footprint(basis, 0, nmo)));
  return then(hamiltonian,
              drop(active_space_ci_footprint(nmo, n_alpha, n_beta, nroots, fci_solver_options(),
                                             dims.syevd_lwork, limit)));
}

MemoryFootprint FciStep::footprint(const FootprintDimensions &dims) const {
  // ci::Sigma's scratch at the least it runs with, one string's G: at run
  // time it takes up to G for the whole space as the budget allows.
  return fci_footprint(dims, nroots, 0);
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
  // step's footprint, from its own dimensions, with ci::Sigma's scratch the
  // one adaptive part: G for the whole space when that fits the budget, else
  // as large a block of strings as does, down to one string's; refused when
  // even that does not. What the device workspace still has to grow by
  // counts too.
  if (fci_strings(nmo, n_alpha) * fci_strings(nmo, n_beta) >
      static_cast<double>(std::numeric_limits<int32_t>::max())) {
    return err_allocation("fci: {} electrons in {} orbitals span {:.3g} determinants, more than "
                          "the CI addresses",
                          n_electrons, nmo, fci_strings(nmo, n_alpha) * fci_strings(nmo, n_beta));
  }
  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
  FootprintDimensions dims{.basis = &basis,
                           .nao = static_cast<size_t>(nmo),
                           .n_alpha = static_cast<size_t>(n_alpha),
                           .n_beta = static_cast<size_t>(n_beta),
                           .ncas = static_cast<size_t>(nmo),
                           .nelecas = static_cast<size_t>(n_electrons)};
  // mmd's record counts, [la][lb], from the classes themselves: mmd screens
  // nothing, so they are plan_shape's.
  int32_t l_max = 0;
  for (const PrimitivePairClass &cls : pairs) {
    l_max = std::max({l_max, cls.la, cls.lb});
  }
  const auto nl = static_cast<size_t>(l_max) + 1;
  dims.mmd_records.assign(nl, std::vector<size_t>(nl, 0));
  for (const PrimitivePairClass &cls : pairs) {
    dims.mmd_records[static_cast<size_t>(cls.la)][static_cast<size_t>(cls.lb)] =
        static_cast<size_t>(cls.n_rec);
  }
  const auto max_order = std::max(static_cast<size_t>(nmo), 16 * static_cast<size_t>(nroots));
  dims.syevd_lwork.resize(max_order + 1);
  for (size_t order = 0; order <= max_order; ++order) {
    dims.syevd_lwork[order] = FBF_TRY(syevd_lwork<double>(res, static_cast<int32_t>(order)));
  }
  // The budget less the fragmentation headroom, as core/footprint has an
  // adaptive routine leave: the pool reserves in chunks, and what it reserves
  // is what the cap bounds.
  const size_t whole_budget = FBF_TRY(res.memory_budget());
  const size_t budget = whole_budget > kPoolHeadroomBytes ? whole_budget - kPoolHeadroomBytes : 0;
  const size_t workspace = res.device_workspace(0).num_elements();
  const auto need = [&](const size_t limit) {
    const MemoryFootprint fp = fci_footprint(dims, nroots, limit);
    return fp.device.persistent + fp.device.scratch +
           (fp.device.workspace > workspace ? fp.device.workspace - workspace : 0);
  };
  constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();
  std::optional<size_t> limit;
  if (need(kUnbounded) > budget) {
    if (need(0) > budget) {
      constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
      return err_allocation("fci: {} electrons in {} orbitals, {} roots, need {:.3g} GiB of device "
                            "memory at the least, and device {}'s budget is {:.3g} GiB, less "
                            "{} MiB of headroom",
                            n_electrons, nmo, nroots, static_cast<double>(need(0)) / kGiB,
                            res.dev_idx(), static_cast<double>(whole_budget) / kGiB,
                            kPoolHeadroomBytes >> 20U);
    }
    // The largest limit that fits: need grows with it.
    size_t fits = 0;
    size_t past = budget;
    while (past - fits > 1) {
      const size_t mid = fits + ((past - fits) / 2);
      (need(mid) <= budget ? fits : past) = mid;
    }
    limit = fits;
  }

  // From here CasciStep::run's path with no core and every orbital active,
  // so the ScfOutput's C is used as it is: its orbital order is the identity.
  // The J and K classes and the Hamiltonian, freed once it is on the host,
  // before the CI allocates.
  const auto n = static_cast<size_t>(nmo);
  std::vector<double> h_col(n * n);
  std::vector<double> eri_col(n * n * n * n);
  double e_core = 0.0;
  {
    std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
    std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
    const ActiveSpaceHamiltonian ham = FBF_TRY(active_space_hamiltonian(
        res, basis, molecule, in.get<CoreHamiltonianOutput>().h,
        in.get<SphericalTransformOutput>().u, j_classes, k_classes, scf.c, 0, nmo));
    e_core = ham.e_core;
    // The sigma takes host arrays, row-major: h transposed, and (pq|rs) from
    // the Hamiltonian's p-fastest order to s-fastest.
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_col.data(), ham.h_eff.data(),
                                           h_col.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri_col.data(), ham.eri.data(),
                                           eri_col.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
  }
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
  // What the budget leaves, which is G for the whole space (and the solve as
  // it was) whenever that fits.
  if (limit) {
    sigma.set_scratch_limit(*limit);
  }
  FBF_TRY(sigma.set_integrals(res, e_core, h, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  ci::Roots roots = FBF_TRY(
      ci::solve(res, sigma, alpha, beta, diagonal_device.data(), nroots, fci_solver_options()));

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
