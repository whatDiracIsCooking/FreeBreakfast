/// @file
/// @brief CasscfStep::run's path from the selected active space through
/// mcscf::solve to the canonical core and virtual orbitals, the
/// natural-orbital Molden file and the orbitals' and CI's .npy files.

module free_breakfast.pipeline.casscf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::binomial, math::align_up
import free_breakfast.io.molden;
import free_breakfast.io.npy;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.integrals.active_space;
import free_breakfast.response.fock;

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

Status canonicalize_core_virtual(DeviceResources &res, size_t nmo, size_t ncore, size_t nocc,
                                 std::span<const double> fock, std::vector<double> &r,
                                 std::vector<double> &energies) {
  if (fock.size() != nmo * nmo || ncore > nocc || nocc > nmo) {
    return err_config("canonicalize_core_virtual: a Fock of {} elements over {} orbitals, {} core "
                      "and {} occupied",
                      fock.size(), nmo, ncore, nocc);
  }
  r.assign(nmo * nmo, 0.0);
  energies.resize(nmo);
  for (size_t p = 0; p < nmo; ++p) {
    r[p + (nmo * p)] = 1.0;
    energies[p] = fock[p + (nmo * p)];
  }
  FBF_TRY(canonicalize_block(res, nmo, 0, ncore, fock, r, energies));
  return canonicalize_block(res, nmo, nocc, nmo, fock, r, energies);
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
CasscfStep::run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, EriOutput,
                     CoreHamiltonianOutput, ScfOutput> in,
                DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const std::shared_ptr<const Buffers::device<double>> &u = in.get<SphericalTransformOutput>().u;
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  const ActiveSpace space = FBF_TRY(select(res, molecule, basis, in.get<ScfOutput>()));
  mcscf::Options options;
  options.conv_tol = conv_tol;
  options.conv_tol_grad = conv_tol_grad;
  options.max_cycle = max_cycle;
  // Integral-direct, the EriOutput's J and K classes; or density-fitted, its
  // B, and no classes: DensityFitting::Full reads none.
  const EriOutput &eri = in.get<EriOutput>();
  const auto *const direct = std::get_if<IntegralDirect>(&eri.eri);
  const auto *const fit = std::get_if<DensityFitted>(&eri.eri);
  std::vector<CoulombPairClass> no_j_classes;
  std::vector<OrderedPairClass> no_k_classes;
  std::vector<CoulombPairClass> &j_classes = direct ? direct->j_classes : no_j_classes;
  std::vector<OrderedPairClass> &k_classes = direct ? direct->k_classes : no_k_classes;
  if (fit) {
    // Each macro iteration's fitted casscf_integrals holds, past B, L and the
    // first half transform X (naux ncas nmo each), J_AO in ppaa's buffer, the
    // gemm intermediate of C^T J_AO C and papa (nmo^2 ncas^2 each): refused
    // here, since past the device's memory an allocation aborts.
    const auto nmo = static_cast<size_t>(space.c.n());
    const auto ncas = static_cast<size_t>(space.ncas);
    const auto naux = static_cast<size_t>(fit->naux);
    const size_t scratch_bytes =
        sizeof(double) * ((2 * naux * ncas * nmo) + (3 * nmo * nmo * ncas * ncas));
    if (const size_t budget = FBF_TRY(res.memory_budget()); scratch_bytes > budget) {
      return err_unsupported("casscf: the fitted integrals over {} orbitals, {} active and {} "
                             "fitting functions need {} bytes of device memory past B's {}, and "
                             "device {}'s budget is {}",
                             nmo, ncas, naux, scratch_bytes, fit->b.num_elements() * sizeof(double),
                             res.dev_idx(), budget);
    }
    options.density_fitting = mcscf::DensityFitting::Full;
  }
  const Buffers::device<double> *const b = fit ? &fit->b : nullptr;
  const int32_t naux = fit ? fit->naux : 0;
  // The CI guess, ci_guess or the ci file's (dim x nroots, root r the column
  // r: the layout of ci_guess), on the device for mcscf::solve, which checks
  // its size against the determinant count.
  std::vector<double> start = ci_guess;
  if (start.empty() && !ci.empty()) {
    NpyArray file = FBF_TRY(read_npy(ci));
    const bool vector = file.shape.size() == 1 && nroots == 1;
    if (!vector && (file.shape.size() != 2 || std::cmp_not_equal(file.shape[1], nroots))) {
      return err_config("casscf: ci '{}' has shape {}, not (dim, nroots) with nroots {}",
                        ci.string(), file.shape, nroots);
    }
    start = std::move(file.data);
  }
  std::optional<Buffers::device<double>> guess;
  if (!start.empty()) {
    guess.emplace(start.size(), res.device());
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(guess->data(), start.data(), start.size() * sizeof(double),
                                       wwr::wwrMemcpyHostToDevice, stream)));
  }
  mcscf::Solution sol = FBF_TRY(mcscf::solve(
      res, basis, molecule, h, u, j_classes, k_classes, space.c, space.ncore, space.ncas,
      space.nelecas, nroots, weights, options, b, naux,
      guess ? std::span<const double>{guess->data(), start.size()} : std::span<const double>{}));
  for (size_t i = 0; i < sol.iterations.size(); ++i) {
    const mcscf::Iteration &it = sol.iterations[i];
    log::info(std::format("CASSCF macro {:2}: E {:.12f} |g| {:.3e}{}", i + 1, it.energy,
                          it.gradient_norm, it.accepted ? "" : " (rejected)"));
  }

  // The state-averaged Fock F^I + F^A at the solution: mcscf::solve keeps
  // neither, so one more set of integrals, fitted as the loop's were.
  const CasscfIntegrals ints = FBF_TRY(
      fit ? casscf_integrals(res, basis, molecule, h, *b, naux, sol.c, space.ncore, space.ncas)
          : casscf_integrals(res, basis, molecule, h, u, j_classes, k_classes, sol.c, space.ncore,
                             space.ncas));
  const response::Fock fock = FBF_TRY(
      fit ? response::fock(res, molecule, h, *b, naux, sol.c, space.ncore, space.ncas, ints.ppaa,
                           ints.papa, sol.gamma, sol.gamma2)
          : response::fock(res, basis, molecule, h, u, j_classes, k_classes, sol.c, space.ncore,
                           space.ncas, ints.ppaa, ints.papa, sol.gamma, sol.gamma2));
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

  // Canonicalization, PySCF's default; C <- C R.
  const auto ncore = static_cast<size_t>(space.ncore);
  std::vector<double> r;
  std::vector<double> mo_energies;
  FBF_TRY(canonicalize_core_virtual(res, nmo, ncore, ncore + static_cast<size_t>(space.ncas), f, r,
                                    mo_energies));
  Buffers::device<double> r_device(r.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_device.data(), r.data(), r.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  SphericalMatrixAO c(res, basis);
  const auto order = static_cast<int32_t>(nmo);
  FBF_TRY(gpu_status(gemm(res.blas(), order, &kOne<double>, sol.c.data(), r_device.data(),
                          &kZero<double>, c.data())));

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
  // The start a later run reads back: the orbitals as CasscfOutput::c holds
  // them, and the CI over their active columns.
  if (!orbitals_npy.empty()) {
    NpyArray host{.shape = {c.n(), nmo}, .data = std::vector<double>(c.num_elements())};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data.data(), c.data(),
                                           host.data.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    FBF_TRY(write_npy(orbitals_npy, host));
  }
  if (!ci_npy.empty()) {
    const auto count = static_cast<size_t>(nroots);
    FBF_TRY(write_npy(ci_npy, NpyArray{.shape = {vectors.size() / count, count}, .data = vectors}));
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
      .ci_residual_norm = sol.roots.davidson.max_residual_norm,
      .vectors = std::move(vectors),
      .rdm1 = std::move(sol.root_gammas),
      .rdm2 = std::move(sol.gamma2),
      .iterations = std::move(sol.iterations),
  };
}

MemoryFootprint CasscfStep::footprint(const FootprintDimensions &dims) const {
  const BasisSet &basis = *dims.basis;
  const size_t nmo = dims.nao;
  const size_t nn = nmo * nmo;
  const auto n_electrons = static_cast<int32_t>(dims.n_alpha + dims.n_beta);
  const auto ncas = static_cast<int32_t>(dims.ncas);
  const auto nelecas = static_cast<int32_t>(dims.nelecas);
  const int32_t ncore = (n_electrons - nelecas) / 2;
  // N_alpha - N_beta = 2S, the molecule's, as run takes it.
  const auto two_s = static_cast<int32_t>(dims.n_alpha - dims.n_beta);
  const int32_t n_beta = (nelecas - two_s) / 2;
  const auto held = [](size_t bytes) { return MemoryFootprint{.device = {.persistent = bytes}}; };
  // casscf_eigh at order n: the matrix and eigenvalues beside diagonalize's
  // work; nothing for n = 0.
  const auto eigh = [&](const size_t n) {
    MemoryFootprint out;
    if (n > 0) {
      out = diagonalize_footprint<double>(dims.syevd_lwork[n]);
      out.device.scratch = sizeof(double) * ((n * n) + n);
    }
    return out;
  };
  const auto widest = [](MemoryFootprint a, const MemoryFootprint &b) {
    a.device.scratch = std::max(a.device.scratch, b.device.scratch);
    a.device.workspace = std::max(a.device.workspace, b.device.workspace);
    a.pinned.workspace = std::max(a.pinned.workspace, b.pinned.workspace);
    return a;
  };

  // The CI guess on the device: nroots vectors over the selected space.
  const auto strings = [ncas](const int32_t electrons) {
    return static_cast<size_t>(std::round(math::binomial(ncas, electrons)));
  };
  const size_t dim = strings(nelecas - n_beta) * strings(n_beta);
  const bool guessed = !ci_guess.empty() || !ci.empty();
  const MemoryFootprint guess =
      held(guessed ? sizeof(double) * dim * static_cast<size_t>(nroots) : 0);

  // mcscf::solve over the EriOutput: integral-direct, or every build fitted.
  const bool fitted = dims.aux != nullptr;
  const auto naux = static_cast<int32_t>(dims.naux);
  mcscf::Options options;
  options.density_fitting = fitted ? mcscf::DensityFitting::Full : mcscf::DensityFitting::Direct;
  const MemoryFootprint solve =
      mcscf::solve_footprint(basis, ncore, ncas, nelecas, two_s + 1, nroots, options, naux,
                             dims.syevd_lwork, dims.skew_eig_lwork);

  // The state-averaged Fock at the solution, both held to the end; the core's
  // and the virtuals' eigenproblems; R on the device and C R; the Molden
  // file's natural orbitals.
  const auto nmo32 = static_cast<int32_t>(nmo);
  const MemoryFootprint ints = fitted ? casscf_integrals_footprint(nmo32, naux, ncore, ncas)
                                      : casscf_integrals_footprint(basis, ncore, ncas);
  const MemoryFootprint fock = fitted ? response::fock_footprint(nmo32, naux, ncore, ncas)
                                      : response::fock_footprint(basis, ncore, ncas);
  const auto core = static_cast<size_t>(ncore);
  const size_t nocc = core + static_cast<size_t>(ncas);
  const MemoryFootprint canonical = widest(eigh(core), eigh(nmo - nocc));
  const MemoryFootprint natural = molden.empty() ? MemoryFootprint{} : eigh(dims.ncas);
  const MemoryFootprint tail =
      then(ints, then(fock, then(drop(canonical), then(held(sizeof(double) * nn),
                                                       then(held(sizeof(double) * nn), natural)))));
  const MemoryFootprint whole = then(select_footprint(dims), then(guess, then(solve, tail)));
  // CasscfOutput::c is the product; the rest is freed on return.
  MemoryFootprint out = whole;
  out.device.persistent = sizeof(double) * nn;
  out.device.scratch = whole.device.persistent + whole.device.scratch - out.device.persistent;
  return out;
}

Result<ScfOutput> CarriedOrbitalsStep::run(View<BasisSetOutput, OverlapOutput> in,
                                           DeviceResources &res) const {
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
  if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
    return err_config("the carried orbitals: {} coefficients and {} energies into a basis of {} "
                      "functions",
                      c.size(), mo_energies.size(), s.n());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n = static_cast<int32_t>(s.n());
  const size_t n2 = s.num_elements();
  // C, M = C^T S C and M^-1/2: held across orthogonalize, which takes the
  // workspace, so each is its own buffer.
  SphericalMatrixAO carried(res, basis);
  Buffers::device<double> m(n2, res.device());
  Buffers::device<double> m_inv_sqrt(n2, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(carried.data(), c.data(), n2 * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  {
    const auto [work] = res.device_regions<double>(n2);
    FBF_TRY(gpu_status(gemmwich(res.blas(), n, carried.data(), s.data(), work, m.data())));
  }
  FBF_TRY(orthogonalize<double>(res, n, m.data(), m_inv_sqrt.data()));
  ScfOutput out{
      .energy = std::numeric_limits<double>::quiet_NaN(),
      .iterations = 0,
      .n_alpha = n_alpha,
      .n_beta = n_beta,
      .mo_energies = mo_energies,
      .c = SphericalMatrixAO(res, basis),
      .d = SphericalMatrixAO(res, basis),
  };
  FBF_TRY(gpu_status(gemm(res.blas(), n, &kOne<double>, carried.data(), m_inv_sqrt.data(),
                          &kZero<double>, out.c.data())));
  FBF_TRY(res.sync());
  return out;
}

MemoryFootprint CarriedOrbitalsStep::footprint(const FootprintDimensions &dims) const {
  const size_t matrix = sizeof(double) * dims.nao * dims.nao;
  // C, M and M^-1/2 are allocated first and held until run returns, past the
  // product's C and d: the three are folded as kept, then moved to scratch.
  const MemoryFootprint held{.device = {.persistent = 3 * matrix}};
  const MemoryFootprint gemmwich_work{
      .device = {.workspace = math::align_up(matrix, DeviceResources::kWorkspaceAlignment)}};
  const MemoryFootprint product{.device = {.persistent = 2 * matrix}};
  MemoryFootprint out = then(then(then(held, gemmwich_work),
                                  orthogonalize_footprint<double>(static_cast<int32_t>(dims.nao),
                                                                  dims.syevd_lwork[dims.nao])),
                             product);
  out.device.persistent -= held.device.persistent;
  out.device.scratch += held.device.persistent;
  return out;
}

} // namespace free_breakfast
