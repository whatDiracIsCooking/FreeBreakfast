/// @file
/// @brief Nevpt2Step::run: per root, the root's canonical orbitals, its RDMs
/// and the inactive Fock's blocks, then the eight SC and the eight PC
/// classes over the fitted B.

module free_breakfast.pipeline.nevpt2;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.blas; // gemm
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

// The slab batch: the most of each class's external index whose slab fits
// @p bytes, at least one. A class holds up to two slabs and nevpt2::slab
// grows the workspace to a third for its gemm product, so the caller gives
// each a share of what is free.
[[nodiscard]] int32_t nevpt2_batch(const nevpt2::MoBlocks &blocks, size_t bytes) {
  size_t widest = 1; // doubles per unit of the batch axis, over every block
  for (const nevpt2::ExtBlock block :
       {nevpt2::ExtBlock::Sr, nevpt2::ExtBlock::Si, nevpt2::ExtBlock::Sijrs, nevpt2::ExtBlock::Sijr,
        nevpt2::ExtBlock::Srsi, nevpt2::ExtBlock::SrsiT, nevpt2::ExtBlock::Srs,
        nevpt2::ExtBlock::Sij, nevpt2::ExtBlock::Sir1, nevpt2::ExtBlock::Sir2}) {
    const std::array<int32_t, 4> shape = nevpt2::block_shape(blocks, block);
    size_t per = 1;
    for (size_t k = 0; k < shape.size(); ++k) {
      if (std::cmp_not_equal(k, nevpt2::batch_axis(block))) {
        per *= static_cast<size_t>(shape[k]);
      }
    }
    widest = std::max(widest, per);
  }
  const size_t batch = bytes / (sizeof(double) * widest);
  return static_cast<int32_t>(std::clamp<size_t>(batch, 1, std::numeric_limits<int32_t>::max()));
}

[[nodiscard]] Result<Nevpt2Output>
Nevpt2Step::run(View<MoleculeOutput, BasisSetOutput, EriOutput, CoreHamiltonianOutput, ScfOutput,
                     CasscfOutput> in,
                DeviceResources &res) const {
  // The CASSCF's own refusal stands in for this one today; this one is for the
  // day it takes an open shell, since the strings below are nelecas / 2 a spin.
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

  // The budgets, from what is free with the CASSCF's products and these
  // integrals held. nevpt2::rdms' default is the free memory less 512 MiB,
  // and the workspace never shrinks: at that, class_energies' own buffers
  // would find nothing left, and a failed allocation aborts. So rdms gets
  // half of what is free (it grows the workspace only to the tile it plans,
  // all of the determinants when they fit), and each slab an eighth.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t rdm_budget = free_bytes / 2;

  // Every per-root array is the step's own, not the workspace's: each is
  // held across calls that take the workspace (ci::Rdm, response::fock,
  // mo_blocks, rdms, class_energies).
  Buffers::device<double> vectors(cas.vectors.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const size_t n2 = n * n;
  const size_t n4 = n2 * n2;
  const size_t n6 = n4 * n2;
  Buffers::device<double> dm1(n2, res.device());
  Buffers::device<double> gamma2(n4, res.device());
  Buffers::device<double> dm2(n4, res.device());
  Buffers::device<double> dm3(n6, res.device());
  Buffers::device<double> f3ca(n6, res.device());
  Buffers::device<double> f3ac(n6, res.device());
  Buffers::device<double> h2e(n4, res.device());
  Buffers::device<double> h1e(n2, res.device());
  Buffers::device<double> h1e_v_sr(nvirt * n, res.device());
  Buffers::device<double> h1e_v_si(n * ncore, res.device());
  Buffers::device<double> h1e_v_sir(nvirt * ncore, res.device());
  Buffers::device<double> e_core(ncore, res.device());
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
  out.roots.reserve(nroots);
  out.pc_roots.reserve(nroots);
  out.pc_refusals.reserve(nroots);
  const std::array<double, 1> one_root{1.0};
  for (size_t root = 0; root < nroots; ++root) {
    const double *const vector = vectors.data() + (root * dim);

    // γ_r and Γ_r, the root alone; response::fock reads them on the host.
    FBF_TRY(rdm.rdm12(res, one_root, vector, dm1.data(), gamma2.data()));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), dm1.data(), n2 * sizeof(double),
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
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e_core.data(), energies.data(), ncore * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e_virt.data(), energies.data() + nocc,
                                           nvirt * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                           stream)));
    FBF_TRY(gpu_status(gemm(res.blas(), order, &kOne<double>, cas.c.data(), r_device.data(),
                            &kZero<double>, c_r.data())));

    // F^I in the root's orbitals, Rᵀ F^I R: h1e is its active block, and the
    // h1e_v_* its blocks against the core and the virtuals.
    FBF_TRY(gpu_status(gemm(res.blas(), order, &kOne<double>, fock.fock_inactive.data(),
                            r_device.data(), &kZero<double>, product.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), order, &kOne<double>, r_device.data(),
                                               product.data(), &kZero<double>, fi_r.data())));
    FBF_TRY(nevpt2_copy_block(res, fi_r.data(), nmo, ncore, ncore, n, n, h1e.data()));
    FBF_TRY(nevpt2_copy_block(res, fi_r.data(), nmo, nocc, ncore, nvirt, n, h1e_v_sr.data()));
    FBF_TRY(nevpt2_copy_block(res, fi_r.data(), nmo, ncore, 0, n, ncore, h1e_v_si.data()));
    FBF_TRY(nevpt2_copy_block(res, fi_r.data(), nmo, nocc, 0, nvirt, ncore, h1e_v_sir.data()));

    // B over the root's orbitals, and h2e from it.
    const nevpt2::MoBlocks blocks = FBF_TRY(nevpt2::mo_blocks(
        res, fit->naux, order, order, fit->b.data(), c_r.data(), cas.ncore, cas.ncas));
    FBF_TRY(nevpt2::h2e(res, blocks, h2e.data()));

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
            h1e_v_sr.data(), static_cast<int32_t>(nvirt))));
      }
    }

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
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dm2.data(), dm2_host.data(), n4 * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    const int64_t tiles =
        FBF_TRY(nevpt2::rdms(res, strings, strings, vector, ints.hamiltonian.eri.data(), dm3.data(),
                             f3ca.data(), f3ac.data(), rdm_budget));

    FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
    const int32_t batch = nevpt2_batch(blocks, free_bytes / 8);
    const nevpt2::EnergyInputs inputs{.h1e = h1e.data(),
                                      .h2e = h2e.data(),
                                      .e_core = e_core.data(),
                                      .e_virt = e_virt.data(),
                                      .h1e_v_sr = h1e_v_sr.data(),
                                      .h1e_v_si = h1e_v_si.data(),
                                      .h1e_v_sir = h1e_v_sir.data(),
                                      .dm1 = dm1.data(),
                                      .dm2 = dm2.data(),
                                      .dm3 = dm3.data(),
                                      .f3ca = f3ca.data(),
                                      .f3ac = f3ac.data()};
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
    log::info(std::format("NEVPT2 root {}: SC E_corr {:.12f}, PC E_corr {} ({} RDM tiles, "
                          "slab batch {})",
                          root, out.roots.back().total,
                          pc ? std::format("{:.12f}", pc->total) : std::string{"refused"}, tiles,
                          batch));
  }
  return out;
}

} // namespace free_breakfast
