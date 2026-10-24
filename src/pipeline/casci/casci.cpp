/// @file
/// @brief CasciStep::run's path from the selected active space through the
/// active-space Hamiltonian and the CI solve to the 1-RDMs and the
/// natural-orbital Molden file.

module free_breakfast.pipeline.casci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.molden;
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.solver;

#include "diagnostics.h"

namespace free_breakfast {

// The eigenvalues of the symmetric n x n @p a (column-major, on the host),
// ascending, by diagonalize on res's device; @p a becomes the eigenvectors.
// The two buffers are this call's own: diagonalize takes the workspace.
[[nodiscard]] Result<std::vector<double>> casci_eigh(DeviceResources &res, size_t n,
                                                     std::vector<double> &a) {
  std::vector<double> w(n);
  if (n == 0) {
    return w;
  }
  const wwr::wwrStream_t stream = res.stream();
  Buffers::device<double> a_device(n * n, res.device());
  Buffers::device<double> w_device(n, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_device.data(), a.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(n), a_device.data(), w_device.data()));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a.data(), a_device.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(w.data(), w_device.data(), n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  return w;
}

// The lowest root's natural orbitals to @p path as Molden: columns core,
// natural, rest of C R', R' = R with its active block rotated to gamma's
// eigenvectors by descending occupation; each energy sum_k R'_kj^2 eps_k.
[[nodiscard]] Status write_casci_molden(DeviceResources &res, const std::filesystem::path &path,
                                        const Molecule &molecule, const BasisSet &basis,
                                        const ScfOutput &scf, const ActiveSpace &space,
                                        std::span<const double> gamma) {
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  const auto ncore = static_cast<size_t>(space.ncore);
  const auto n = static_cast<size_t>(space.ncas);
  std::vector<double> v(gamma.begin(), gamma.end()); // symmetric: row- is column-major
  const std::vector<double> occupation = FBF_TRY(casci_eigh(res, n, v));
  std::vector<double> r = space.r;
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < nmo; ++k) {
      double value = 0.0;
      for (size_t t = 0; t < n; ++t) {
        value += space.r[k + (nmo * (ncore + t))] * v[t + (n * (n - 1 - j))];
      }
      r[k + (nmo * (ncore + j))] = value;
    }
  }
  std::vector<double> c(nao * nmo);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data(), scf.c.data(), c.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> orbitals(nao * nmo, 0.0);
  std::vector<double> energies(nmo, 0.0);
  std::vector<double> occupations(nmo, 0.0);
  for (size_t j = 0; j < nmo; ++j) {
    for (size_t k = 0; k < nmo; ++k) {
      const double rkj = r[k + (nmo * j)];
      energies[j] += rkj * rkj * scf.mo_energies[k];
      for (size_t mu = 0; mu < nao; ++mu) {
        orbitals[mu + (nao * j)] += c[mu + (nao * k)] * rkj;
      }
    }
    if (j < ncore) {
      occupations[j] = 2.0;
    } else if (j < ncore + n) {
      occupations[j] = occupation[n - 1 - (j - ncore)];
    }
  }
  return write_molden(path, molecule.atoms(), basis, orbitals, energies, occupations);
}

[[nodiscard]] Result<CasciOutput>
CasciStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                    PrimitivePairClassesOutput, CoreHamiltonianOutput, ScfOutput> in,
               DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const ScfOutput &scf = in.get<ScfOutput>();
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  // CasciOutput::c = C R, R the selection's.
  ActiveSpace space = FBF_TRY(select(res, molecule, basis, scf));
  const int32_t ncore = space.ncore;
  const int32_t n_active = space.ncas;
  const int32_t n_electrons = space.nelecas;

  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  const ActiveSpaceHamiltonian ham = FBF_TRY(active_space_hamiltonian(
      res, basis, molecule, in.get<CoreHamiltonianOutput>().h, in.get<SphericalTransformOutput>().u,
      j_classes, k_classes, space.c, ncore, n_active));

  // The sigma takes host arrays, row-major: h_eff transposed, and (tu|vw)
  // from the Hamiltonian's t-fastest order to w-fastest.
  const auto n = static_cast<size_t>(n_active);
  std::vector<double> h_col(n * n);
  std::vector<double> eri_col(n * n * n * n);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(h_col.data(), ham.h_eff.data(), h_col.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri_col.data(), ham.eri.data(),
                                         eri_col.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }

  const int32_t n_beta = n_electrons / 2;
  const StringSet alpha = FBF_TRY(StringSet::create(n_active, n_electrons - n_beta));
  const StringSet beta = FBF_TRY(StringSet::create(n_active, n_beta));
  CiSigma sigma = FBF_TRY(CiSigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ham.e_core, h_eff, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  CiRoots roots = FBF_TRY(ci_solve(res, sigma, alpha, beta, diagonal_device.data(), nroots));

  const auto count = static_cast<size_t>(nroots);
  // Each root's 1-RDM on the device, from the roots ci_solve left there; then
  // both come to the host, the vectors for CasciOutput::vectors.
  std::vector<double> gamma(count * n * n);
  std::vector<double> vectors(count * dim);
  Buffers::device<double> gamma_device(gamma.size(), res.device());
  for (size_t r = 0; r < count; ++r) {
    FBF_TRY(sigma.rdm1(res, roots.vectors.data() + (r * dim), gamma_device.data() + (r * n * n)));
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), gamma_device.data(),
                                         gamma.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                         stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), roots.vectors.data(),
                                         vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  if (!molden.empty()) {
    FBF_TRY(write_casci_molden(res, molden, molecule, basis, scf, space,
                               std::span<const double>{gamma.data(), n * n}));
  }
  return CasciOutput{
      .ncore = ncore,
      .ncas = n_active,
      .nelecas = n_electrons,
      .c = std::move(space.c),
      .energies = std::move(roots.energies),
      .spin_squares = std::move(roots.spin_squares),
      .vectors = std::move(vectors),
      .rdm1 = std::move(gamma),
      .davidson = roots.davidson,
  };
}

} // namespace free_breakfast
