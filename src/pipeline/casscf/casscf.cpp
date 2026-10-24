/// @file
/// @brief CasscfStep::run's path from the selected active space through
/// casscf_solve to the canonical core and virtual orbitals and the
/// natural-orbital Molden file.

module free_breakfast.pipeline.casscf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.molden;
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.mcscf.fock;

#include "diagnostics.h"

namespace free_breakfast {

// The eigenvalues of the symmetric n x n @p a (column-major, on the host),
// ascending, by diagonalize on res's device; @p a becomes the eigenvectors.
// The two buffers are this call's own: diagonalize takes the workspace.
[[nodiscard]] Result<std::vector<double>> casscf_eigh(DeviceResources &res, size_t n,
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

// Rotates columns [begin, end) of the nmo x nmo column-major @p r to the
// eigenvectors of @p fock's block over them, ascending, and writes the
// eigenvalues into @p energies there. @p r is the identity on that block.
[[nodiscard]] Status canonicalize_block(DeviceResources &res, size_t nmo, size_t begin, size_t end,
                                        std::span<const double> fock, std::vector<double> &r,
                                        std::vector<double> &energies) {
  const size_t n = end - begin;
  std::vector<double> block(n * n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      block[i + (n * j)] = fock[(begin + i) + (nmo * (begin + j))];
    }
  }
  const std::vector<double> w = FBF_TRY(casscf_eigh(res, n, block));
  for (size_t j = 0; j < n; ++j) {
    energies[begin + j] = w[j];
    for (size_t i = 0; i < n; ++i) {
      r[(begin + i) + (nmo * (begin + j))] = block[i + (n * j)];
    }
  }
  return {};
}

// The state-averaged natural orbitals to @p path as Molden: the canonical
// @p c with its active columns rotated to the weighted gamma's eigenvectors by
// descending occupation. Core and virtual energies are @p mo_energies; an
// active one is its natural orbital's expectation in @p fock.
[[nodiscard]] Status write_casscf_molden(DeviceResources &res, const std::filesystem::path &path,
                                         const Molecule &molecule, const BasisSet &basis,
                                         const SphericalMatrixAO &c, size_t ncore, size_t ncas,
                                         std::span<const double> fock,
                                         std::span<const double> mo_energies,
                                         std::span<const double> gamma) {
  const size_t nao = c.n();
  const size_t nmo = c.n();
  const size_t n = ncas;
  std::vector<double> v(gamma.begin(), gamma.end()); // symmetric: row- is column-major
  const std::vector<double> occupation = FBF_TRY(casscf_eigh(res, n, v));
  std::vector<double> orbitals(nao * nmo);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(orbitals.data(), c.data(), orbitals.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  const std::vector<double> active(orbitals.begin() + static_cast<std::ptrdiff_t>(nao * ncore),
                                   orbitals.begin() +
                                       static_cast<std::ptrdiff_t>(nao * (ncore + n)));
  std::vector<double> energies(mo_energies.begin(), mo_energies.end());
  std::vector<double> occupations(nmo, 0.0);
  for (size_t j = 0; j < ncore; ++j) {
    occupations[j] = 2.0;
  }
  for (size_t j = 0; j < n; ++j) {
    const double *vj = v.data() + (n * (n - 1 - j)); // descending
    occupations[ncore + j] = occupation[n - 1 - j];
    double energy = 0.0;
    for (size_t t = 0; t < n; ++t) {
      for (size_t u = 0; u < n; ++u) {
        energy += vj[t] * fock[(ncore + t) + (nmo * (ncore + u))] * vj[u];
      }
    }
    energies[ncore + j] = energy;
    for (size_t mu = 0; mu < nao; ++mu) {
      double value = 0.0;
      for (size_t t = 0; t < n; ++t) {
        value += active[mu + (nao * t)] * vj[t];
      }
      orbitals[mu + (nao * (ncore + j))] = value;
    }
  }
  return write_molden(path, molecule.atoms(), basis, orbitals, energies, occupations);
}

[[nodiscard]] Result<CasscfOutput>
CasscfStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput,
                     PrimitivePairClassesOutput, CoreHamiltonianOutput, ScfOutput> in,
                DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const std::shared_ptr<const Buffers::device<double>> &u = in.get<SphericalTransformOutput>().u;
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  const ActiveSpace space = FBF_TRY(select(res, molecule, basis, in.get<ScfOutput>()));
  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClassesOutput>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  CasscfOptions options;
  options.conv_tol = conv_tol;
  options.conv_tol_grad = conv_tol_grad;
  options.max_cycle = max_cycle;
  CasscfSolution sol =
      FBF_TRY(casscf_solve(res, basis, molecule, h, u, j_classes, k_classes, space.c, space.ncore,
                           space.ncas, space.nelecas, nroots, weights, options));
  for (size_t i = 0; i < sol.iterations.size(); ++i) {
    const CasscfIteration &it = sol.iterations[i];
    log::info(std::format("CASSCF macro {:2}: E {:.12f} |g| {:.3e}{}", i + 1, it.energy,
                          it.gradient_norm, it.accepted ? "" : " (rejected)"));
  }

  // The state-averaged Fock F^I + F^A at the solution: casscf_solve keeps
  // neither, so one more set of integrals.
  const CasscfIntegrals ints = FBF_TRY(casscf_integrals(res, basis, molecule, h, u, j_classes,
                                                        k_classes, sol.c, space.ncore, space.ncas));
  const CasscfFock fock =
      FBF_TRY(casscf_fock(res, basis, molecule, h, u, j_classes, k_classes, sol.c, space.ncore,
                          space.ncas, ints.ppaa, sol.gamma, sol.gamma2));
  const size_t nmo = sol.c.n();
  std::vector<double> f(nmo * nmo);
  std::vector<double> f_active(nmo * nmo);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(f.data(), fock.fock_inactive.data(), f.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(f_active.data(), fock.fock_active.data(),
                                         f_active.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::ranges::transform(f, f_active, f.begin(), std::plus<>{});

  // Canonicalization, PySCF's default: R the identity but on the core and the
  // virtual blocks, each F's eigenvectors there; C <- C R.
  const auto ncore = static_cast<size_t>(space.ncore);
  const auto nocc = ncore + static_cast<size_t>(space.ncas);
  std::vector<double> r(nmo * nmo, 0.0);
  std::vector<double> mo_energies(nmo);
  for (size_t p = 0; p < nmo; ++p) {
    r[p + (nmo * p)] = 1.0;
    mo_energies[p] = f[p + (nmo * p)];
  }
  FBF_TRY(canonicalize_block(res, nmo, 0, ncore, f, r, mo_energies));
  FBF_TRY(canonicalize_block(res, nmo, nocc, nmo, f, r, mo_energies));
  Buffers::device<double> r_device(r.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_device.data(), r.data(), r.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  SphericalMatrixAO c(res, basis);
  const auto order = static_cast<int32_t>(nmo);
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, order, order, order, &kOne<double>,
      sol.c.data(), order, r_device.data(), order, &kZero<double>, c.data(), order)));

  // The CI vectors to the host, for CasscfOutput::vectors.
  std::vector<double> vectors(sol.roots.vectors.num_elements());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), sol.roots.vectors.data(),
                                         vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  if (!molden.empty()) {
    FBF_TRY(write_casscf_molden(res, molden, molecule, basis, c, ncore,
                                static_cast<size_t>(space.ncas), f, mo_energies, sol.gamma));
  }
  std::vector<double> averaged = weights;
  if (averaged.empty()) {
    averaged.assign(static_cast<size_t>(nroots), 1.0 / nroots);
  }
  return CasscfOutput{
      .ncore = space.ncore,
      .ncas = space.ncas,
      .nelecas = space.nelecas,
      .c = std::move(c),
      .mo_energies = std::move(mo_energies),
      .energies = std::move(sol.roots.energies),
      .spin_squares = std::move(sol.roots.spin_squares),
      .weights = std::move(averaged),
      .energy = sol.energy,
      .gradient_norm = sol.gradient_norm,
      .vectors = std::move(vectors),
      .rdm1 = std::move(sol.root_gammas),
      .iterations = std::move(sol.iterations),
  };
}

} // namespace free_breakfast
