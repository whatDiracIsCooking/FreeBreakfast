/// @file
/// @brief ChainSampler, the chain at the reference and at displaced
/// geometries; numerical_heff and its contractions over it.

module free_breakfast.tools.qd_nevpt2_gradient;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.overlap_mixed;
import free_breakfast.mcscf.overlap;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.pipeline.pipeline;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.nevpt2;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// The CASSCF's bars at every point. NEVPT2 is not variational in the
// orbitals, so a residual gradient puts an error linear in ||g|| into E_corr;
// over a step of 1e-3 bohr that is the size of the 1e-6 gate unless ||g|| is
// held near the formaldehyde fixtures' polish (< 1e-10).
constexpr double kChainConvTolGrad = 1e-10;
constexpr double kChainConvTol = 1e-13;
constexpr int32_t kChainMaxCycle = 200;

// The guard's bound on 1 - the active spaces' overlap
// (nevpt2_gradient_reference.py's).
constexpr double kSubspaceTol = 1e-3;

// The polish's most reruns, and each one's macro iterations. The solver's
// ||g|| falls to ~1e-12 on most reruns once it is under 1e-10, but not every
// one.
constexpr size_t kPolishPasses = 10;
constexpr int32_t kPolishCycles = 8;

// The chain's root at one geometry: the molecule from atoms in memory
// (pipeline/optimize's TrialGeometryStep).
struct ChainGeometryStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<Atom> atoms;

  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, atoms, 0, 1)};
  }
};

// Every displaced point's ScfOutput: the reference CASSCF's orbitals,
// Loewdin-orthonormalized in this geometry's overlap, C (C^T S C)^-1/2
// (pipeline/optimize's CarriedOrbitalsStep), so the CasscfStep starts from the
// active space the reference converged. Only the CasscfStep reads it, by index.
struct ChainOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  std::vector<double> c; // AO by MO, column-major
  std::vector<double> mo_energies;
  int32_t n_alpha = 0;
  int32_t n_beta = 0;

  [[nodiscard]] Result<ScfOutput> run(View<BasisSetOutput, OverlapOutput> in,
                                      DeviceResources &res) const {
    const BasisSet &basis = in.get<BasisSetOutput>().basis;
    const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
    if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
      return err_config("carried {} orbital coefficients and {} energies into a basis of {}",
                        c.size(), mo_energies.size(), s.n());
    }
    const ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<int32_t>(s.n());
    const size_t n2 = s.num_elements();
    SphericalMatrixAO previous(res, basis);
    Buffers::device<double> m(n2, res.device());
    Buffers::device<double> m_inv_sqrt(n2, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(previous.data(), c.data(), n2 * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    {
      const auto [work] = res.device_regions<double>(n2);
      FBF_TRY(gpu_status(gemmwich(res.blas(), n, previous.data(), s.data(), work, m.data())));
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
    FBF_TRY(gpu_status(gemm(res.blas(), n, &kOne<double>, previous.data(), m_inv_sqrt.data(),
                            &kZero<double>, out.c.data())));
    FBF_TRY(res.sync());
    return out;
  }
};

using DisplacedChain =
    Pipeline<ChainGeometryStep, BasisSetStep, SphericalTransformStep, PrimitivePairClassesStep,
             EriStep, OverlapStep, CoreHamiltonianStep, ChainOrbitalsStep, CasscfStep, Nevpt2Step>;
using ReferenceChain = Pipeline<ChainGeometryStep, BasisSetStep, SphericalTransformStep,
                                PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                                CoreHamiltonianStep, GuessStep, ScfStep, CasscfStep, Nevpt2Step>;

// The mcscf::Solution a CasscfOutput is (pipeline/casscf_gradient's): its
// orbitals and CI vectors back on the device, the weighted gamma from the
// per-root ones.
Result<mcscf::Solution> chain_solution(DeviceResources &res, const BasisSet &basis,
                                       const CasscfOutput &cas) {
  const auto n2 = static_cast<size_t>(cas.ncas) * static_cast<size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (size_t r = 0; r < cas.weights.size(); ++r) {
    for (size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  mcscf::Solution solution{
      .c = SphericalMatrixAO(res, basis),
      .roots =
          {
              .energies = cas.energies,
              .spin_squares = cas.spin_squares,
              .vectors = Buffers::device<double>(cas.vectors.size(), res.device()),
              .davidson = {},
              .fp32_iterations = 0,
          },
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
  };
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                         cas.c.num_elements() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(res.sync());
  return solution;
}

// @p n device doubles on the host.
Result<std::vector<double>> chain_download(DeviceResources &res, const double *device, size_t n) {
  std::vector<double> host(n);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), device, n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return host;
}

// 1 - the smallest singular value of C_a^T S_x C'_a, the two active spaces'
// overlap through the AO cross overlap of the two geometries.
Result<double> subspace_miss(DeviceResources &res, const BasisSet &bra_basis,
                             const SphericalMatrixAO &bra, const BasisSet &ket_basis,
                             const SphericalMatrixAO &ket, size_t ncore, size_t ncas) {
  const size_t nb = bra.n();
  const size_t nk = ket.n();
  Buffers::device<double> s_dev(nb * nk, res.device());
  FBF_TRY(overlap_mixed_spherical(res, bra_basis, ket_basis, s_dev));
  const std::vector<double> s = FBF_TRY(chain_download(res, s_dev.data(), nb * nk));
  const std::vector<double> cb = FBF_TRY(chain_download(res, bra.data(), bra.num_elements()));
  const std::vector<double> ck = FBF_TRY(chain_download(res, ket.data(), ket.num_elements()));
  // sc = S C'_a (nb x ncas), then M = C_a^T sc.
  std::vector<double> sc(nb * ncas, 0.0);
  for (size_t j = 0; j < ncas; ++j) {
    for (size_t n = 0; n < nk; ++n) {
      const double x = ck[n + (nk * (ncore + j))];
      for (size_t m = 0; m < nb; ++m) {
        sc[m + (nb * j)] += s[m + (nb * n)] * x;
      }
    }
  }
  std::vector<double> mat(ncas * ncas, 0.0);
  for (size_t j = 0; j < ncas; ++j) {
    for (size_t i = 0; i < ncas; ++i) {
      for (size_t m = 0; m < nb; ++m) {
        mat[i + (ncas * j)] += cb[m + (nb * (ncore + i))] * sc[m + (nb * j)];
      }
    }
  }
  // M^T M's least eigenvalue is the least singular value squared.
  std::vector<double> mtm(ncas * ncas, 0.0);
  for (size_t j = 0; j < ncas; ++j) {
    for (size_t i = 0; i < ncas; ++i) {
      for (size_t k = 0; k < ncas; ++k) {
        mtm[i + (ncas * j)] += mat[k + (ncas * i)] * mat[k + (ncas * j)];
      }
    }
  }
  Buffers::device<double> a(mtm.size(), res.device());
  Buffers::device<double> w(ncas, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a.data(), mtm.data(), mtm.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(diagonalize(res, static_cast<int32_t>(ncas), a.data(), w.data()));
  const std::vector<double> values = FBF_TRY(chain_download(res, w.data(), ncas));
  return 1.0 - std::sqrt(std::max(values.front(), 0.0));
}

// The smallest gap between neighbouring values of @p sorted (infinity for
// fewer than two).
double smallest_gap(std::span<const double> sorted) {
  double gap = std::numeric_limits<double>::infinity();
  for (size_t i = 1; i < sorted.size(); ++i) {
    gap = std::min(gap, sorted[i] - sorted[i - 1]);
  }
  return gap;
}

// The PC cut at every coupling (QD) or of the root (one state).
std::vector<std::array<int64_t, 8>> pc_cut(const Nevpt2Output &nev, bool qd) {
  std::vector<std::array<int64_t, 8>> dropped;
  if (qd) {
    for (const PcClassEnergies &c : nev.pc_qd_couplings) {
      dropped.push_back(c.dropped);
    }
  } else if (nev.pc_roots[0]) {
    dropped.push_back(nev.pc_roots[0]->dropped);
  }
  return dropped;
}

struct ChainSampler::State {
  NodeResources *node = nullptr;
  NumericalChain chain;
  bool polish = false;
  size_t n = 0;
  bool qd = false;
  BasisSetStep basis_set;
  EriStep eri;
  CasscfStep casscf;
  // Every point after the reference starts from carried orbitals.
  CasscfStep carried_casscf;
  // A polishing pass asks for kPolishTarget itself within a few macro
  // iterations: at conv_tol_grad it would stop at once.
  CasscfStep polish_casscf;
  Nevpt2Step nevpt2;
  std::optional<ReferenceChain> reference;
  // With polish, the reference's chain polished as every point is.
  std::optional<DisplacedChain> polished;
  const CasscfOutput *cas0 = nullptr;
  const Nevpt2Output *nev0 = nullptr;
  const BasisSet *basis0 = nullptr;
  const DensityFitted *fit0 = nullptr;
  const Molecule *molecule0 = nullptr;
  const std::vector<PrimitivePairClass> *classes0 = nullptr;
  const SphericalMatrixAO *h0 = nullptr;
  std::optional<mcscf::Solution> solution0;
  std::vector<std::array<int64_t, 8>> dropped0;
  ChainOrbitalsStep carried;
  ChainPoint at0;
  size_t polish_passes = 0;

  // The chain at @p atoms from @p start, polished when asked: rerun from its
  // own converged orbitals until ||g|| falls under kPolishTarget, at most
  // kPolishPasses times, the lowest ||g|| kept. At ||g|| ~3e-11, PC's H_eff
  // off the diagonal moves by ~4e-10 hartree on skew H2CO.
  Status converge(const std::vector<Atom> &atoms, const ChainOrbitalsStep &start,
                  std::optional<DisplacedChain> &out) {
    const auto once = [&](const ChainOrbitalsStep &from,
                          const CasscfStep &solve) -> Result<DisplacedChain> {
      auto run = FBF_TRY(DisplacedChain::create({
          ChainGeometryStep{atoms},
          basis_set,
          SphericalTransformStep{},
          PrimitivePairClassesStep{},
          eri,
          OverlapStep{},
          CoreHamiltonianStep{},
          from,
          solve,
          nevpt2,
      }));
      FBF_TRY(run.run(*node));
      return run;
    };
    DeviceResources &res = node->device(0);
    out.emplace(FBF_TRY(once(start, carried_casscf)));
    for (size_t pass = 0;
         polish && pass < kPolishPasses && out->get<CasscfOutput>()->gradient_norm >= kPolishTarget;
         ++pass) {
      const CasscfOutput &now = *out->get<CasscfOutput>();
      const ChainOrbitalsStep from{
          .c = FBF_TRY(chain_download(res, now.c.data(), now.c.num_elements())),
          .mo_energies = now.mo_energies,
          .n_alpha = start.n_alpha,
          .n_beta = start.n_beta,
      };
      ++polish_passes;
      // A pass that does not converge (it can stall above conv_tol_grad)
      // keeps the point it started from.
      Result<DisplacedChain> next = once(from, polish_casscf);
      if (next && next->get<CasscfOutput>()->gradient_norm < now.gradient_norm) {
        out.reset();
        out.emplace(std::move(*next));
      }
    }
    return {};
  }

  Result<ChainPoint> point(std::span<const double> displacement, const ChainOrbitalsStep &start) {
    DeviceResources &res = node->device(0);
    const std::string_view name = "nevpt2::ChainSampler";
    if (displacement.size() != 3 * chain.atoms.size()) {
      return err_config("{}: a displacement of {} for {} atoms", name, displacement.size(),
                        chain.atoms.size());
    }
    std::vector<Atom> atoms = chain.atoms;
    for (size_t a = 0; a < atoms.size(); ++a) {
      atoms[a].position = {atoms[a].position.x + displacement[3 * a],
                           atoms[a].position.y + displacement[(3 * a) + 1],
                           atoms[a].position.z + displacement[(3 * a) + 2]};
    }
    std::optional<DisplacedChain> run;
    FBF_TRY(converge(atoms, start, run));
    const CasscfOutput &cas = *run->get<CasscfOutput>();
    const Nevpt2Output &nev = *run->get<Nevpt2Output>();
    const BasisSet &basis = run->get<BasisSetOutput>()->basis;
    ChainPoint p;
    p.gradient_norm = cas.gradient_norm;
    if (cas.gradient_norm >= casscf.conv_tol_grad) {
      return err_numerical("{}: a displaced CASSCF stopped at ||g|| {:.2e}", name,
                           cas.gradient_norm);
    }
    // The roots, matched by their overlaps with the reference's.
    const mcscf::Solution solution = FBF_TRY(chain_solution(res, basis, cas));
    const std::vector<double> s = FBF_TRY(mcscf::state_overlaps(
        res, *basis0, *solution0, basis, solution, cas0->ncore, cas0->ncas, cas0->nelecas, 1));
    std::vector<size_t> pi(n);
    std::vector<double> signs(n);
    std::vector<bool> taken(n, false);
    for (size_t i = 0; i < n; ++i) {
      size_t best = 0;
      for (size_t j = 1; j < n; ++j) {
        if (std::abs(s[(i * n) + j]) > std::abs(s[(i * n) + best])) {
          best = j;
        }
      }
      if (taken[best]) {
        return err_numerical("{}: reference root {} matches no displaced root alone", name, i);
      }
      taken[best] = true;
      pi[i] = best;
      signs[i] = s[(i * n) + best] < 0.0 ? -1.0 : 1.0;
      p.ci = std::max(p.ci, 1.0 - std::abs(s[(i * n) + best]));
      p.casscf.push_back(cas.energies[best]);
    }
    // The overlaps and H_eff in the reference's root basis.
    p.s.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        p.s[(i * n) + j] = signs[j] * s[(i * n) + pi[j]];
      }
    }
    const auto carried_into = [&](std::span<const double> heff) {
      std::vector<double> out(n * n);
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          out[(i * n) + j] = signs[i] * signs[j] * heff[(pi[i] * n) + pi[j]];
        }
      }
      return out;
    };
    // Eigenvectors over the displaced roots, state k at [k n, (k + 1) n),
    // carried into the reference's root basis.
    const auto vectors_into = [&](std::span<const double> vectors) {
      std::vector<double> out(n * n);
      for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
          out[(k * n) + i] = signs[i] * vectors[(k * n) + pi[i]];
        }
      }
      return out;
    };
    if (qd) {
      p.sc_heff = carried_into(nev.qd_heff);
      if (chain.pc && nev.pc_qd_heff.size() == n * n) {
        p.pc_heff = carried_into(nev.pc_qd_heff);
      }
    }
    p.c = FBF_TRY(chain_download(res, cas.c.data(), cas.c.num_elements()));
    p.mo_energies = cas.mo_energies;
    const auto ncore = static_cast<size_t>(cas0->ncore);
    const auto ncas = static_cast<size_t>(cas0->ncas);
    p.subspace = FBF_TRY(subspace_miss(res, *basis0, cas0->c, basis, cas.c, ncore, ncas));
    if (p.ci >= kCiTol || p.subspace >= kSubspaceTol) {
      return err_numerical("{}: a displaced CASSCF's CI overlap off 1 by {:.2e}, its active "
                           "space by {:.2e}",
                           name, p.ci, p.subspace);
    }
    if (qd) {
      if (nev.qd_energies.size() != n) {
        return err_numerical("{}: no SC-QD-NEVPT2 at a displaced point", name);
      }
      p.sc = nev.qd_energies;
      p.sc_vectors = vectors_into(nev.qd_vectors);
      if (chain.pc) {
        if (nev.pc_qd_energies.size() != n) {
          return err_numerical("{}: PC-QD-NEVPT2 refused at a displaced point: {}", name,
                               nev.pc_qd_refusal);
        }
        p.pc = nev.pc_qd_energies;
        p.pc_vectors = vectors_into(nev.pc_qd_vectors);
      }
    } else {
      if (!nev.pc_roots[0]) {
        return err_numerical("{}: PC refused at a displaced point: {}", name, nev.pc_refusals[0]);
      }
      p.sc = {cas.energies[0] + nev.roots[0].total};
      p.pc = {cas.energies[0] + nev.pc_roots[0]->total};
    }
    // PC's QD couplings are taken over the displaced roots' order: compare the
    // cut as a multiset, which a permutation of the roots leaves alone.
    if (chain.pc) {
      auto dropped = pc_cut(nev, qd);
      auto expected = dropped0;
      std::ranges::sort(dropped);
      std::ranges::sort(expected);
      if (dropped != expected) {
        return err_numerical("{}: PC cuts other modes at a displaced point: a rank change", name);
      }
    }
    const size_t nocc = ncore + ncas;
    p.core_gap = smallest_gap(std::span{cas.mo_energies}.subspan(0, ncore));
    p.virtual_gap = smallest_gap(std::span{cas.mo_energies}.subspan(nocc));
    return p;
  }
};

Result<ChainSampler> ChainSampler::create(NodeResources &node, const NumericalChain &chain,
                                          bool polish) {
  if (chain.weights.empty()) {
    return err_config("nevpt2::ChainSampler: a chain with no root");
  }
  auto state = std::make_unique<State>();
  State &st = *state;
  st.node = &node;
  st.chain = chain;
  st.polish = polish;
  st.n = chain.weights.size();
  st.qd = st.n > 1;
  st.basis_set.basis = chain.basis;
  st.eri.mode = EriMode::DensityFitting;
  st.eri.aux_basis = chain.aux_basis;
  st.casscf.ncas = chain.ncas;
  st.casscf.nelecas = chain.nelecas;
  st.casscf.active = chain.active;
  st.casscf.nroots = static_cast<int32_t>(st.n);
  st.casscf.weights = chain.weights;
  st.casscf.conv_tol = kChainConvTol;
  st.casscf.conv_tol_grad = kChainConvTolGrad;
  st.casscf.max_cycle = kChainMaxCycle;
  st.nevpt2.quasi_degenerate = st.qd;
  st.carried_casscf = st.casscf;
  st.carried_casscf.active.clear();
  st.polish_casscf = st.carried_casscf;
  st.polish_casscf.conv_tol_grad = kPolishTarget;
  st.polish_casscf.max_cycle = kPolishCycles;

  DeviceResources &res = node.device(0);
  st.reference.emplace(FBF_TRY((ReferenceChain::create({
      ChainGeometryStep{chain.atoms},
      st.basis_set,
      SphericalTransformStep{},
      PrimitivePairClassesStep{},
      st.eri,
      OverlapStep{},
      OrthogonalizerStep{},
      CoreHamiltonianStep{},
      GuessStep{},
      ScfStep{},
      st.casscf,
      st.nevpt2,
  }))));
  FBF_TRY(st.reference->run(node));
  const ScfOutput &rhf = *st.reference->get<ScfOutput>();
  if (polish) {
    const CasscfOutput &first = *st.reference->get<CasscfOutput>();
    const ChainOrbitalsStep from{
        .c = FBF_TRY(chain_download(res, first.c.data(), first.c.num_elements())),
        .mo_energies = first.mo_energies,
        .n_alpha = rhf.n_alpha,
        .n_beta = rhf.n_beta,
    };
    FBF_TRY(st.converge(chain.atoms, from, st.polished));
  }
  const auto product = [&]<class T>() -> const T & {
    return st.polished ? *st.polished->get<T>() : *st.reference->get<T>();
  };
  st.cas0 = &product.operator()<CasscfOutput>();
  st.nev0 = &product.operator()<Nevpt2Output>();
  st.basis0 = &product.operator()<BasisSetOutput>().basis;
  st.molecule0 = &product.operator()<MoleculeOutput>().molecule;
  st.classes0 = &product.operator()<PrimitivePairClassesOutput>().classes;
  st.h0 = &product.operator()<CoreHamiltonianOutput>().h;
  st.fit0 = std::get_if<DensityFitted>(&product.operator()<EriOutput>().eri);
  const CasscfOutput &cas0 = *st.cas0;
  const Nevpt2Output &nev0 = *st.nev0;
  const std::string_view name = "nevpt2::ChainSampler";
  if (st.fit0 == nullptr) {
    return err_config("{}: the ERIs are not density-fitted", name);
  }
  if (cas0.gradient_norm >= kChainConvTolGrad) {
    return err_numerical("{}: the reference CASSCF stopped at ||g|| {:.2e}", name,
                         cas0.gradient_norm);
  }
  if (st.qd &&
      (nev0.qd_energies.size() != st.n || (chain.pc && nev0.pc_qd_energies.size() != st.n))) {
    return err_numerical("{}: no QD-NEVPT2 at the reference: {}", name, nev0.pc_qd_refusal);
  }
  if (!st.qd && !nev0.pc_roots[0]) {
    return err_numerical("{}: PC refused at the reference: {}", name, nev0.pc_refusals[0]);
  }
  st.solution0.emplace(FBF_TRY(chain_solution(res, *st.basis0, cas0)));
  st.dropped0 = pc_cut(nev0, st.qd);

  const auto ncore = static_cast<size_t>(cas0.ncore);
  const size_t nocc = ncore + static_cast<size_t>(cas0.ncas);
  ChainPoint &at0 = st.at0;
  at0.casscf = cas0.energies;
  if (st.qd) {
    at0.sc = nev0.qd_energies;
    at0.pc = nev0.pc_qd_energies;
    at0.sc_vectors = nev0.qd_vectors;
    at0.pc_vectors = nev0.pc_qd_vectors;
    at0.sc_heff = nev0.qd_heff;
    at0.pc_heff = nev0.pc_qd_heff;
  } else {
    at0.sc = {cas0.energies[0] + nev0.roots[0].total};
    at0.pc = {cas0.energies[0] + nev0.pc_roots[0]->total};
  }
  at0.s.assign(st.n * st.n, 0.0);
  for (size_t i = 0; i < st.n; ++i) {
    at0.s[(i * st.n) + i] = 1.0;
  }
  at0.gradient_norm = cas0.gradient_norm;
  at0.core_gap = smallest_gap(std::span{cas0.mo_energies}.subspan(0, ncore));
  at0.virtual_gap = smallest_gap(std::span{cas0.mo_energies}.subspan(nocc));
  // What every displaced point starts from.
  st.carried = ChainOrbitalsStep{
      .c = FBF_TRY(chain_download(res, cas0.c.data(), cas0.c.num_elements())),
      .mo_energies = cas0.mo_energies,
      .n_alpha = rhf.n_alpha,
      .n_beta = rhf.n_beta,
  };
  at0.c = st.carried.c;
  at0.mo_energies = cas0.mo_energies;
  return ChainSampler(std::move(state));
}

ChainSampler::ChainSampler(std::unique_ptr<State> state) : state_(std::move(state)) {}
ChainSampler::ChainSampler(ChainSampler &&) noexcept = default;
ChainSampler &ChainSampler::operator=(ChainSampler &&) noexcept = default;
ChainSampler::~ChainSampler() = default;

Result<ChainPoint> ChainSampler::at(std::span<const double> displacement) {
  return state_->point(displacement, state_->carried);
}

Result<ChainPoint> ChainSampler::at(std::span<const double> displacement, const ChainPoint &from) {
  return state_->point(displacement, ChainOrbitalsStep{
                                         .c = from.c,
                                         .mo_energies = from.mo_energies,
                                         .n_alpha = state_->carried.n_alpha,
                                         .n_beta = state_->carried.n_beta,
                                     });
}

const ChainPoint &ChainSampler::reference() const {
  return state_->at0;
}
const CasscfOutput &ChainSampler::casscf() const {
  return *state_->cas0;
}
const Nevpt2Output &ChainSampler::nevpt2() const {
  return *state_->nev0;
}
const Molecule &ChainSampler::molecule() const {
  return *state_->molecule0;
}
const BasisSet &ChainSampler::basis() const {
  return *state_->basis0;
}
const std::vector<PrimitivePairClass> &ChainSampler::classes() const {
  return *state_->classes0;
}
const DensityFitted &ChainSampler::fit() const {
  return *state_->fit0;
}
const SphericalMatrixAO &ChainSampler::core_hamiltonian() const {
  return *state_->h0;
}
const mcscf::Solution &ChainSampler::solution() const {
  return *state_->solution0;
}
size_t ChainSampler::polish_passes() const {
  return state_->polish_passes;
}

Richardson richardson(double coarse, double middle, double fine, double noise, double finest_step) {
  const double value = ((4.0 * fine) - middle) / 3.0;
  return {.value = value,
          .error_bar =
              std::abs(value - (((4.0 * middle) - coarse) / 3.0)) + (noise / (2.0 * finest_step))};
}

Differenced contract(const HeffDerivative &d, std::span<const double> w) {
  Differenced out;
  for (const std::array<std::vector<double>, 3> &per_step : d.central) {
    std::array<double, 3> at{};
    for (size_t s = 0; s < at.size(); ++s) {
      for (size_t ij = 0; ij < w.size(); ++ij) {
        at[s] += w[ij] * per_step[s][ij];
      }
    }
    const Richardson r = richardson(at[0], at[1], at[2], d.noise, d.steps[2]);
    out.value.push_back(r.value);
    out.error_bar.push_back(r.error_bar);
  }
  return out;
}

// u_P u_Q^T over @p vectors (state K at [K n, (K + 1) n)), n x n.
std::vector<double> heff_outer(std::span<const double> vectors, size_t n, size_t p, size_t q) {
  std::vector<double> w(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      w[(i * n) + j] = vectors[(p * n) + i] * vectors[(q * n) + j];
    }
  }
  return w;
}

HeffPair heff_pair(const HeffDerivative &d, std::span<const double> vectors, size_t p, size_t q) {
  const std::vector<double> pp = heff_outer(vectors, d.n, p, p);
  const std::vector<double> qq = heff_outer(vectors, d.n, q, q);
  std::vector<double> mean(pp.size());
  std::vector<double> x(pp.size());
  for (size_t ij = 0; ij < pp.size(); ++ij) {
    mean[ij] = 0.5 * (pp[ij] + qq[ij]);
    x[ij] = qq[ij] - pp[ij];
  }
  return {.mean = contract(d, mean),
          .x = contract(d, x),
          .h = contract(d, heff_outer(vectors, d.n, p, q))};
}

Result<NumericalHeff> numerical_heff(NodeResources &node, const NumericalChain &chain,
                                     std::span<const std::vector<double>> components,
                                     std::array<double, 3> steps, double noise) {
  const std::string_view name = "nevpt2::numerical_heff";
  const size_t n = chain.weights.size();
  const size_t dim = 3 * chain.atoms.size();
  if (n < 2) {
    return err_config("{}: {} roots; H_eff needs two or more", name, n);
  }
  for (const std::vector<double> &u : components) {
    if (u.size() != dim) {
      return err_config("{}: a component of {} for {} atoms", name, u.size(), chain.atoms.size());
    }
  }
  if (std::ranges::any_of(steps, [](double h) { return !(h > 0.0); })) {
    return err_config("{}: steps {} not all positive", name, steps);
  }
  ChainSampler sampler = FBF_TRY(ChainSampler::create(node, chain, true));
  const ChainPoint &at0 = sampler.reference();
  NumericalHeff out{.casscf_energies = at0.casscf, .gradient_norm = at0.gradient_norm};

  const size_t ncomp = components.size();
  // Per scheme, component and step, (H(+h) - H(-h)) / 2h.
  std::array<std::vector<std::array<std::vector<double>, 3>>, 2> central;
  const size_t nschemes = chain.pc ? 2 : 1;
  for (size_t c = 0; c < ncomp; ++c) {
    for (std::vector<std::array<std::vector<double>, 3>> &scheme : central) {
      scheme.emplace_back();
    }
    for (size_t si = 0; si < steps.size(); ++si) {
      std::array<ChainPoint, 2> sides;
      double h2 = 0.0;
      for (size_t side = 0; side < 2; ++side) {
        const double sign = side == 0 ? -1.0 : 1.0;
        std::vector<double> dr(dim);
        for (size_t x = 0; x < dim; ++x) {
          dr[x] = sign * steps[si] * components[c][x];
        }
        h2 = std::inner_product(dr.begin(), dr.end(), dr.begin(), 0.0);
        sides[side] = FBF_TRY(sampler.at(dr));
        out.subspace = std::max(out.subspace, sides[side].subspace);
        out.ci = std::max(out.ci, sides[side].ci);
        ++out.points;
      }
      // The continuity guard's prediction, from energies alone: the even
      // part of E_r(dR) - (E_r0 + g_r . dR) over |dR|^2 needs no g_r.
      for (size_t r = 0; r < n; ++r) {
        const double even =
            std::abs(sides[0].casscf[r] + sides[1].casscf[r] - (2.0 * at0.casscf[r])) / (2.0 * h2);
        out.prediction = std::max(out.prediction, even);
      }
      if (out.prediction >= kPredictionBound) {
        return err_numerical("{}: a displaced CASSCF {:.3f} |dR|^2 off its prediction", name,
                             out.prediction);
      }
      for (size_t scheme = 0; scheme < nschemes; ++scheme) {
        const std::vector<double> &minus = scheme == 0 ? sides[0].sc_heff : sides[0].pc_heff;
        const std::vector<double> &plus = scheme == 0 ? sides[1].sc_heff : sides[1].pc_heff;
        if (minus.size() != n * n || plus.size() != n * n) {
          return err_numerical("{}: no {} H_eff at a displaced point", name,
                               scheme == 0 ? "SC" : "PC");
        }
        std::vector<double> d(n * n);
        for (size_t ij = 0; ij < n * n; ++ij) {
          d[ij] = (plus[ij] - minus[ij]) / (2.0 * steps[si]);
        }
        central[scheme][c][si] = std::move(d);
      }
    }
  }

  const auto derivative = [&](size_t scheme) -> Result<HeffDerivative> {
    HeffDerivative d{
        .n = n,
        .steps = steps,
        .noise = noise,
        .heff = scheme == 0 ? at0.sc_heff : at0.pc_heff,
        .energies = scheme == 0 ? at0.sc : at0.pc,
        .vectors = scheme == 0 ? at0.sc_vectors : at0.pc_vectors,
        .central = std::move(central[scheme]),
    };
    double worst = 0.0;
    const auto bounded = [&](Differenced v) {
      for (const double bar : v.error_bar) {
        worst = std::max(worst, bar);
      }
      return v;
    };
    for (size_t ij = 0; ij < n * n; ++ij) {
      std::vector<double> w(n * n, 0.0);
      w[ij] = 1.0;
      d.elements.push_back(bounded(contract(d, w)));
    }
    for (size_t k = 0; k < n; ++k) {
      d.g.push_back(bounded(contract(d, heff_outer(d.vectors, n, k, k))));
    }
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        d.h.push_back(bounded(contract(d, heff_outer(d.vectors, n, p, q))));
      }
    }
    if (worst > kCouplingErrorBarTol) {
      return err_numerical("{}: {} error bar {:.2e}, above {:.1e}", name, scheme == 0 ? "SC" : "PC",
                           worst, kCouplingErrorBarTol);
    }
    return d;
  };
  out.sc = FBF_TRY(derivative(0));
  if (chain.pc) {
    out.pc = FBF_TRY(derivative(1));
  }
  return out;
}

} // namespace free_breakfast::nevpt2
