/// @file
/// @brief Nevpt2Step::run: per root, the root's canonical orbitals, its RDMs
/// and the inactive Fock's blocks, then the eight SC and the eight PC
/// classes over the fitted B; and with quasi_degenerate, the SC and PC
/// effective Hamiltonians over the state-averaged orbitals, pair by pair.

module free_breakfast.pipeline.nevpt2;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.active_space;
import free_breakfast.response.fock;
import free_breakfast.ci.strings;
import free_breakfast.ci.rdm;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.rdm;

#include "diagnostics.h"

namespace free_breakfast {

// The rows x cols block of the nmo-row column-major @p from at (row, col),
// packed column-major into @p to. Nothing for an empty block, whose pitch a
// copy would refuse.
[[nodiscard]] Status nevpt2_copy_block(DeviceResources &res, const double *from, size_t nmo,
                                       size_t row, size_t col, size_t rows, size_t cols,
                                       double *to) {
  if (rows == 0 || cols == 0) {
    return {};
  }
  return gpu_status(wwr::wwrMemcpy2DAsync(to, rows * sizeof(double), from + row + (nmo * col),
                                          nmo * sizeof(double), rows * sizeof(double), cols,
                                          wwr::wwrMemcpyDeviceToDevice, res.stream()));
}

// Each element's chemical core, PySCF's `pyscf.data.elements.chemcore_atm`
// (2.14), by the first Z of each count: none to Be, then 1 (B), 5 (Al),
// 9 (Ga), 14 (Y), 18 (In), 23 (Lu), 34 (Tl), 50 (Rf) and 55 (Nh). Not a
// noble-gas core: Na and Mg keep their 2s2p, K to Zn their 3s3p.
[[nodiscard]] int32_t nevpt2_chemical_core(int32_t z) {
  constexpr std::array<std::pair<int32_t, int32_t>, 9> kFirst{
      {{113, 55}, {104, 50}, {81, 34}, {71, 23}, {49, 18}, {39, 14}, {31, 9}, {13, 5}, {5, 1}}};
  for (const auto &[first, count] : kFirst) {
    if (z >= first) {
      return count;
    }
  }
  return 0;
}

Result<int32_t> Nevpt2Step::resolve_frozen_core(const std::string_view step,
                                                const int32_t frozen_core, const Molecule &molecule,
                                                const int32_t ncore) {
  // A count, or each atom's chemical core.
  int32_t frozen = frozen_core;
  if (frozen == kFrozenCoreAuto) {
    frozen = 0;
    for (const Atom &atom : molecule.atoms()) {
      frozen += nevpt2_chemical_core(nuclear_charge(atom));
    }
  }
  if (frozen < 0 || frozen > ncore) {
    return err_config("[{}] frozen_core is {}{}, and the CASSCF has {} core orbitals: it must be "
                      "0 to {}, or \"auto\"",
                      step, frozen, frozen_core == kFrozenCoreAuto ? " (auto)" : "", ncore, ncore);
  }
  return frozen;
}

// What the energy layer reads of a set of orbitals but the RDMs and the
// orbital energies, over the orbitals @p c (AO x MO, device) and F^I in them,
// @p fi (nmo x nmo column-major, device): h1e, F^I's active block; the
// h1e_v_* blocks against the correlated core (the core past its lowest
// @p nfrozen) and the virtuals, Sr's less Σ_b (vb|ba); the MO blocks of B
// over the orbitals from @p nfrozen on, returned; and h2e from them.
[[nodiscard]] Result<nevpt2::MoBlocks>
nevpt2_orbital_inputs(DeviceResources &res, const DensityFitted &fit, const CasscfOutput &cas,
                      size_t nfrozen, const double *c, const double *fi, double *h1e,
                      double *h1e_v_sr, double *h1e_v_si, double *h1e_v_sir, double *h2e) {
  const auto nmo = static_cast<size_t>(cas.c.n());
  const auto ncore = static_cast<size_t>(cas.ncore);
  const auto n = static_cast<size_t>(cas.ncas);
  const size_t nocc = ncore + n;
  const size_t nvirt = nmo - nocc;
  const size_t ncorr = ncore - nfrozen;
  const auto order = static_cast<int32_t>(nmo);
  FBF_TRY(nevpt2_copy_block(res, fi, nmo, ncore, ncore, n, n, h1e));
  FBF_TRY(nevpt2_copy_block(res, fi, nmo, nocc, ncore, nvirt, n, h1e_v_sr));
  FBF_TRY(nevpt2_copy_block(res, fi, nmo, ncore, nfrozen, n, ncorr, h1e_v_si));
  FBF_TRY(nevpt2_copy_block(res, fi, nmo, nocc, nfrozen, nvirt, ncorr, h1e_v_sir));

  // B over the orbitals the classes walk, and h2e from it: C's columns from
  // nfrozen on, packed, so the blocks' core is the correlated one.
  nevpt2::MoBlocks blocks = FBF_TRY(
      nevpt2::mo_blocks(res, fit.naux, order, static_cast<int32_t>(nmo - nfrozen), fit.b.data(),
                        c + (nmo * nfrozen), static_cast<int32_t>(ncorr), cas.ncas));
  FBF_TRY(nevpt2::h2e(res, blocks, h2e));

  // Sr's one-electron part, h1eff[v, a] - Σ_b (vb|ba): per active b, one
  // gemm of B_va's b-th column block (naux x nvirt) against B_aa's b-th row
  // (naux x n, every naux n apart).
  const auto naux = static_cast<size_t>(blocks.naux);
  if (nvirt > 0 && naux > 0) {
    for (size_t b = 0; b < n; ++b) {
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, static_cast<int32_t>(nvirt),
          cas.ncas, blocks.naux, &kNegOne<double>, blocks.va.data() + (naux * nvirt * b),
          blocks.naux, blocks.aa.data() + (naux * b), blocks.naux * cas.ncas, &kOne<double>,
          h1e_v_sr, static_cast<int32_t>(nvirt))));
    }
  }
  return blocks;
}

// The n x n symmetrized effective Hamiltonian E_I δ_IJ + ½ (C_IJ + C_JI) of
// the couplings' totals @p totals (row-major, bra first) into @p heff, and
// its eigenvalues, ascending, into @p energies and eigenvectors, column-major,
// each signed so that its largest-magnitude component is positive, into
// @p vectors: by diagonalize on res's device, into buffers of this call's own,
// since diagonalize takes the workspace.
[[nodiscard]] Status nevpt2_qd_eigh(DeviceResources &res, std::span<const double> casscf,
                                    std::span<const double> totals, std::vector<double> &heff,
                                    std::vector<double> &energies, std::vector<double> &vectors) {
  const size_t n = casscf.size();
  heff.assign(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      heff[(i * n) + j] =
          (i == j ? casscf[i] : 0.0) + (0.5 * (totals[(i * n) + j] + totals[(j * n) + i]));
    }
  }
  vectors = heff;
  energies.assign(n, 0.0);
  const wwr::wwrStream_t stream = res.stream();
  Buffers::device<double> a_device(n * n, res.device());
  Buffers::device<double> w_device(n, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_device.data(), vectors.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(n), a_device.data(), w_device.data()));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), a_device.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(energies.data(), w_device.data(), n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  for (size_t k = 0; k < n; ++k) {
    const std::span<double> v{vectors.data() + (k * n), n};
    const auto largest = std::ranges::max_element(v, {}, [](double x) { return std::abs(x); });
    if (*largest < 0.0) {
      std::ranges::transform(v, v.begin(), std::negate<>{});
    }
  }
  return {};
}

[[nodiscard]] Result<Nevpt2Output>
Nevpt2Step::run(View<MoleculeOutput, BasisSetOutput, EriOutput, CoreHamiltonianOutput, ScfOutput,
                     CasscfOutput> in,
                DeviceResources &res) const {
  // The Casscf step takes an open shell, so this refusal is the one that
  // applies: the strings below are nelecas / 2 a spin.
  if (const ScfOutput &scf = in.get<ScfOutput>(); scf.n_alpha != scf.n_beta) {
    return err_unsupported("[{}] needs a closed-shell reference, and the ScfOutput has {} alpha "
                           "and {} beta electrons",
                           kName, scf.n_alpha, scf.n_beta);
  }
  // The CASSCF's roots are of the molecule's spin, and the strings and the
  // NEVPT2 below are a singlet's, nelecas / 2 a spin.
  if (const int32_t multiplicity = in.get<MoleculeOutput>().molecule.multiplicity();
      multiplicity != 1) {
    return err_unsupported("[{}] is implemented for a singlet only, and the molecule's "
                           "multiplicity is {}",
                           kName, multiplicity);
  }
  const auto *const fit = std::get_if<DensityFitted>(&in.get<EriOutput>().eri);
  if (fit == nullptr) {
    return err_unsupported("[{}] runs over density fitting only, and [eri] mode is "
                           "'integral_direct': set [eri] mode = 'density_fitting'",
                           kName);
  }
  // With one root the state-averaged Fock is the root's own: QD would repeat
  // the state-specific result.
  if (quasi_degenerate && in.get<CasscfOutput>().energies.size() < 2) {
    return err_config("[{}] quasi_degenerate couples the roots of a state average, and the "
                      "CASSCF has {} root: set [casscf] nroots to 2 or more, or use plain [{}]",
                      kName, in.get<CasscfOutput>().energies.size(), kName);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const CasscfOutput &cas = in.get<CasscfOutput>();
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  const auto nmo = static_cast<size_t>(cas.c.n());
  const auto ncore = static_cast<size_t>(cas.ncore);
  const auto n = static_cast<size_t>(cas.ncas);
  const size_t nocc = ncore + n;
  const size_t nvirt = nmo - nocc;
  const size_t nroots = cas.energies.size();
  const auto order = static_cast<int32_t>(nmo);

  const int32_t frozen = FBF_TRY(resolve_frozen_core(kName, frozen_core, molecule, cas.ncore));
  const auto nfrozen = static_cast<size_t>(frozen);
  const size_t ncorr = ncore - nfrozen;

  // (pq|uv), (pu|qv) and (tu|vw) over the CASSCF's orbitals, fitted. The
  // per-root rotations leave the active columns alone, so (tu|vw), what
  // nevpt2::rdms contracts, is every root's.
  const CasscfIntegrals ints = FBF_TRY(
      casscf_integrals(res, basis, molecule, h, fit->b, fit->naux, cas.c, cas.ncore, cas.ncas));
  const ci::DeviceStringSet strings =
      FBF_TRY(ci::DeviceStringSet::create(res, cas.ncas, cas.nelecas / 2));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, strings, strings));
  const auto dim = static_cast<size_t>(rdm.size());
  if (cas.vectors.size() != nroots * dim || cas.rdm1.size() != nroots * n * n) {
    return err_config("[{}] the CASSCF holds {} CI coefficients and {} 1-RDM elements, not {} "
                      "roots of {} and {}",
                      kName, cas.vectors.size(), cas.rdm1.size(), nroots, dim, n * n);
  }

  // The adaptive parts size themselves from the memory budget when they run:
  // nevpt2::rdms its tile (all of the determinants when they fit), and each
  // class its slab batch (nevpt2::kBatchFromBudget). Each is scratch of the
  // call's own, freed before the next allocation of this step, so neither
  // leaves less for what follows; at the least, one determinant and a batch
  // of one, they are the footprint's.
  // Every per-root array is the step's own, not the workspace's: each is
  // held across calls that take the workspace (ci::Rdm, response::fock,
  // mo_blocks, rdms, class_energies). With quasi_degenerate each root's RDMs
  // are kept, root r's at slot r, for the effective Hamiltonians' diagonal.
  const size_t slots = quasi_degenerate ? nroots : 1;
  Buffers::device<double> vectors(cas.vectors.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const size_t n2 = n * n;
  const size_t n4 = n2 * n2;
  const size_t n6 = n4 * n2;
  Buffers::device<double> dm1(slots * n2, res.device());
  Buffers::device<double> gamma2(n4, res.device());
  Buffers::device<double> dm2(slots * n4, res.device());
  Buffers::device<double> dm3(slots * n6, res.device());
  Buffers::device<double> f3ca(slots * n6, res.device());
  Buffers::device<double> f3ac(slots * n6, res.device());
  Buffers::device<double> h2e(n4, res.device());
  Buffers::device<double> h1e(n2, res.device());
  Buffers::device<double> h1e_v_sr(nvirt * n, res.device());
  Buffers::device<double> h1e_v_si(n * ncorr, res.device());
  Buffers::device<double> h1e_v_sir(nvirt * ncorr, res.device());
  Buffers::device<double> e_core(ncorr, res.device());
  Buffers::device<double> e_virt(nvirt, res.device());
  Buffers::device<double> r_device(nmo * nmo, res.device());
  Buffers::device<double> fi_r(nmo * nmo, res.device());
  Buffers::device<double> product(nmo * nmo, res.device());
  SphericalMatrixAO c_r(res, basis);

  std::vector<double> gamma(n2);
  std::vector<double> gamma2_host(n4);
  std::vector<double> fock_inactive(nmo * nmo);
  std::vector<double> fock_active(nmo * nmo);
  std::vector<double> dm2_host(n4);
  std::vector<double> r;
  std::vector<double> energies;

  Nevpt2Output out;
  out.frozen_core = frozen;
  if (nfrozen > 0) {
    log::info(std::format("NEVPT2: {} frozen core orbital{}, {} correlated", nfrozen,
                          nfrozen == 1 ? "" : "s", ncorr));
  }
  out.roots.reserve(nroots);
  out.pc_roots.reserve(nroots);
  out.pc_refusals.reserve(nroots);
  const std::array<double, 1> one_root{1.0};
  for (size_t root = 0; root < nroots; ++root) {
    const double *const vector = vectors.data() + (root * dim);
    const size_t slot = quasi_degenerate ? root : 0;
    double *const dm1_r = dm1.data() + (slot * n2);
    double *const dm2_r = dm2.data() + (slot * n4);
    double *const dm3_r = dm3.data() + (slot * n6);
    double *const f3ca_r = f3ca.data() + (slot * n6);
    double *const f3ac_r = f3ac.data() + (slot * n6);

    // γ_r and Γ_r, the root alone; response::fock reads them on the host.
    FBF_TRY(rdm.rdm12(res, one_root, vector, dm1_r, gamma2.data()));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), dm1_r, n2 * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2_host.data(), gamma2.data(), n4 * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());

    // The root's Fock F^I + F^A(γ_r), and its canonical core and virtuals:
    // R, and C_r = C R.
    const response::Fock fock =
        FBF_TRY(response::fock(res, molecule, h, fit->b, fit->naux, cas.c, cas.ncore, cas.ncas,
                               ints.ppaa, ints.papa, gamma, gamma2_host));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fock_inactive.data(), fock.fock_inactive.data(),
                                           fock_inactive.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fock_active.data(), fock.fock_active.data(),
                                           fock_active.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    std::ranges::transform(fock_active, fock_inactive, fock_active.begin(), std::plus<>{});
    FBF_TRY(canonicalize_core_virtual(res, nmo, ncore, nocc, fock_active, r, energies));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_device.data(), r.data(), r.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e_core.data(), energies.data() + nfrozen,
                                           ncorr * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                           stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e_virt.data(), energies.data() + nocc,
                                           nvirt * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                           stream)));
    FBF_TRY(gpu_status(gemm(res.blas(), order, &kOne<double>, cas.c.data(), r_device.data(),
                            &kZero<double>, c_r.data())));

    // F^I in the root's orbitals, Rᵀ F^I R, and what is read of it and of B
    // over C_r.
    FBF_TRY(gpu_status(gemm(res.blas(), order, &kOne<double>, fock.fock_inactive.data(),
                            r_device.data(), &kZero<double>, product.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), order, &kOne<double>, r_device.data(),
                                               product.data(), &kZero<double>, fi_r.data())));
    const nevpt2::MoBlocks blocks = FBF_TRY(
        nevpt2_orbital_inputs(res, *fit, cas, nfrozen, c_r.data(), fi_r.data(), h1e.data(),
                              h1e_v_sr.data(), h1e_v_si.data(), h1e_v_sir.data(), h2e.data()));

    // dm2 in make_dm123's convention, ⟨E_pq E_rs⟩ = Γ_pqrs + δ_qr γ_ps,
    // row-major as Γ is.
    std::ranges::copy(gamma2_host, dm2_host.begin());
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = 0; q < n; ++q) {
        for (size_t s = 0; s < n; ++s) {
          dm2_host[(((((p * n) + q) * n) + q) * n) + s] += gamma[(p * n) + s];
        }
      }
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dm2_r, dm2_host.data(), n4 * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    const int64_t tiles = FBF_TRY(nevpt2::rdms(res, strings, strings, vector,
                                               ints.hamiltonian.eri.data(), dm3_r, f3ca_r, f3ac_r));
    constexpr int32_t batch = nevpt2::kBatchFromBudget;
    const nevpt2::EnergyInputs inputs{.h1e = h1e.data(),
                                      .h2e = h2e.data(),
                                      .e_core = e_core.data(),
                                      .e_virt = e_virt.data(),
                                      .h1e_v_sr = h1e_v_sr.data(),
                                      .h1e_v_si = h1e_v_si.data(),
                                      .h1e_v_sir = h1e_v_sir.data(),
                                      .dm1 = dm1_r,
                                      .dm2 = dm2_r,
                                      .dm3 = dm3_r,
                                      .f3ca = f3ca_r,
                                      .f3ac = f3ac_r};
    out.roots.push_back(FBF_TRY(nevpt2::class_energies(res, blocks, inputs, batch)));
    // A PC class refused as Numerical costs that root its PC, not its SC.
    Result<nevpt2::PcClassEnergies> pc = nevpt2::pc_class_energies(res, blocks, inputs, batch);
    if (!pc && pc.error().kind != ErrorKind::Numerical) {
      return std::unexpected(std::move(pc).error());
    }
    out.pc_refusals.push_back(pc ? std::string{} : std::move(pc.error().message));
    out.pc_roots.push_back(pc ? std::optional{*pc} : std::nullopt);
    if (!pc) {
      log::warn(
          std::format("NEVPT2 root {}: PC refused, SC kept: {}", root, out.pc_refusals.back()));
    }
    log::info(std::format("NEVPT2 root {}: SC E_corr {:.12f}, PC E_corr {} ({} RDM tiles)", root,
                          out.roots.back().total,
                          pc ? std::format("{:.12f}", pc->total) : std::string{"refused"}, tiles));
  }
  if (!quasi_degenerate) {
    return out;
  }

  // QD-NEVPT2. The CASSCF's own orbitals, whose core and virtuals are
  // canonical over the state-averaged Fock: one F^I (the Fock's, over the
  // state average's RDMs, though F^I reads none), one set of MO blocks, and
  // the orbital energies, shared by every root.
  std::vector<double> gamma_sa(n2, 0.0);
  for (size_t root = 0; root < nroots; ++root) {
    for (size_t pq = 0; pq < n2; ++pq) {
      gamma_sa[pq] += cas.weights[root] * cas.rdm1[(root * n2) + pq];
    }
  }
  const response::Fock fock_sa =
      FBF_TRY(response::fock(res, molecule, h, fit->b, fit->naux, cas.c, cas.ncore, cas.ncas,
                             ints.ppaa, ints.papa, gamma_sa, cas.rdm2));
  const nevpt2::MoBlocks blocks = FBF_TRY(nevpt2_orbital_inputs(
      res, *fit, cas, nfrozen, cas.c.data(), fock_sa.fock_inactive.data(), h1e.data(),
      h1e_v_sr.data(), h1e_v_si.data(), h1e_v_sir.data(), h2e.data()));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(e_core.data(), cas.mo_energies.data() + nfrozen,
                                     ncorr * sizeof(double), wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(e_virt.data(), cas.mo_energies.data() + nocc,
                                     nvirt * sizeof(double), wwr::wwrMemcpyHostToDevice, stream)));

  // Per unordered pair I < J, the transition RDMs of (I, J) at slot 0 and of
  // (J, I) at slot 1, consumed by the pair's couplings before the next pair
  // is built: never the n² sets at once. f3ca and f3ac are read only on the
  // diagonal, so the transition builds skip them (null).
  Buffers::device<double> tdm1(2 * n2, res.device());
  Buffers::device<double> tdm2(2 * n4, res.device());
  Buffers::device<double> tdm3(2 * n6, res.device());
  constexpr int32_t batch = nevpt2::kBatchFromBudget;
  // Every entry starts omitted (null RDMs): the layer builds and walks
  // nothing of it.
  const nevpt2::EnergyInputs shared{.h1e = h1e.data(),
                                    .h2e = h2e.data(),
                                    .e_core = e_core.data(),
                                    .e_virt = e_virt.data(),
                                    .h1e_v_sr = h1e_v_sr.data(),
                                    .h1e_v_si = h1e_v_si.data(),
                                    .h1e_v_sir = h1e_v_sir.data()};
  std::vector<nevpt2::EnergyInputs> pairs(nroots * nroots, shared);
  out.qd_couplings.resize(nroots * nroots);
  out.pc_qd_couplings.resize(nroots * nroots);

  // The diagonal, once: every root's own entry and no other, which walks
  // each root's classes (and solves its PC metrics) once and hands back its
  // first-order wave function, the kets every pair's couplings read.
  const auto started = std::chrono::steady_clock::now();
  for (size_t root = 0; root < nroots; ++root) {
    nevpt2::EnergyInputs &in = pairs[(root * nroots) + root];
    in.dm1 = dm1.data() + (root * n2);
    in.dm2 = dm2.data() + (root * n4);
    in.dm3 = dm3.data() + (root * n6);
    in.f3ca = f3ca.data() + (root * n6);
    in.f3ac = f3ac.data() + (root * n6);
  }
  nevpt2::ScAmplitudes sc_kets;
  const std::vector<nevpt2::ClassEnergies> sc_diagonal =
      FBF_TRY(nevpt2::sc_effective_hamiltonian(res, blocks, pairs, batch, &sc_kets));
  // A PC state refused as Numerical costs QD its PC, not its SC.
  nevpt2::PcAmplitudes pc_kets;
  Result<std::vector<nevpt2::PcClassEnergies>> pc_diagonal =
      nevpt2::pc_effective_hamiltonian(res, blocks, pairs, batch, &pc_kets);
  if (!pc_diagonal && pc_diagonal.error().kind != ErrorKind::Numerical) {
    return std::unexpected(std::move(pc_diagonal).error());
  }
  if (!pc_diagonal) {
    // The diagonal call's states are the roots, so its "state J" is root J.
    out.pc_qd_refusal = std::move(pc_diagonal.error().message);
    out.pc_qd_couplings.clear();
    log::warn(std::format("QD-NEVPT2: PC refused, SC kept: {}", out.pc_qd_refusal));
  }
  for (size_t root = 0; root < nroots; ++root) {
    const size_t p = (root * nroots) + root;
    out.qd_couplings[p] = sc_diagonal[p];
    if (pc_diagonal) {
      out.pc_qd_couplings[p] = (*pc_diagonal)[p];
    }
    pairs[p] = shared;
  }
  const auto diagonal_done = std::chrono::steady_clock::now();

  // Then per pair its two off-diagonal entries alone, over the cached kets.
  for (size_t i = 0; i < nroots; ++i) {
    for (size_t j = i + 1; j < nroots; ++j) {
      for (const auto &[slot, bra, ket] :
           {std::tuple{size_t{0}, i, j}, std::tuple{size_t{1}, j, i}}) {
        FBF_TRY(nevpt2::rdms(res, strings, strings, vectors.data() + (bra * dim),
                             vectors.data() + (ket * dim), ints.hamiltonian.eri.data(),
                             tdm1.data() + (slot * n2), tdm2.data() + (slot * n4),
                             tdm3.data() + (slot * n6), nullptr, nullptr));
        nevpt2::EnergyInputs &in = pairs[(bra * nroots) + ket];
        in.dm1 = tdm1.data() + (slot * n2);
        in.dm2 = tdm2.data() + (slot * n4);
        in.dm3 = tdm3.data() + (slot * n6);
      }
      const size_t ij = (i * nroots) + j;
      const size_t ji = (j * nroots) + i;
      const std::vector<nevpt2::ClassEnergies> sc =
          FBF_TRY(nevpt2::sc_effective_hamiltonian(res, blocks, pairs, batch, &sc_kets));
      out.qd_couplings[ij] = sc[ij];
      out.qd_couplings[ji] = sc[ji];
      if (out.pc_qd_refusal.empty()) {
        // No state is solved here, so nothing is refused that the diagonal
        // did not refuse.
        const std::vector<nevpt2::PcClassEnergies> pc =
            FBF_TRY(nevpt2::pc_effective_hamiltonian(res, blocks, pairs, batch, &pc_kets));
        out.pc_qd_couplings[ij] = pc[ij];
        out.pc_qd_couplings[ji] = pc[ji];
      }
      pairs[ij] = shared;
      pairs[ji] = shared;
    }
  }
  const auto pairs_done = std::chrono::steady_clock::now();
  const auto ms = [](const auto from, const auto to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
  };
  log::info(std::format("QD-NEVPT2: one diagonal walk over {} roots {:.1f} ms, {} pair walks "
                        "{:.1f} ms",
                        nroots, ms(started, diagonal_done), nroots * (nroots - 1) / 2,
                        ms(diagonal_done, pairs_done)));

  std::vector<double> totals(nroots * nroots);
  std::ranges::transform(out.qd_couplings, totals.begin(),
                         [](const nevpt2::ClassEnergies &c) { return c.total; });
  FBF_TRY(nevpt2_qd_eigh(res, cas.energies, totals, out.qd_heff, out.qd_energies, out.qd_vectors));
  log::info(std::format("SC-QD-NEVPT2 energies {::.12f}", out.qd_energies));
  if (out.pc_qd_refusal.empty()) {
    std::ranges::transform(out.pc_qd_couplings, totals.begin(),
                           [](const nevpt2::PcClassEnergies &c) { return c.total; });
    FBF_TRY(nevpt2_qd_eigh(res, cas.energies, totals, out.pc_qd_heff, out.pc_qd_energies,
                           out.pc_qd_vectors));
    log::info(std::format("PC-QD-NEVPT2 energies {::.12f}", out.pc_qd_energies));
  }
  return out;
}

} // namespace free_breakfast
