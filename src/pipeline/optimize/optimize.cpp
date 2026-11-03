/// @file
/// @brief OptimizeStep: geometry::optimize over a nested Pipeline per
/// evaluation, the RHF chain or a CASSCF chain (to its gradient or its
/// NEVPT2's, a state-averaged root followed by match_roots and a QD state by
/// qd_state_overlap) with its orbitals
/// carried from one evaluation to the next, and the trajectory, final.xyz
/// and history it writes.

module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kAngstromPerBohr, kOne, kZero
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom; // element_symbol
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.xyz;
import free_breakfast.mcscf.overlap; // state_overlaps
import free_breakfast.pipeline.pipeline;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.orthogonalize;

#include "diagnostics.h"

namespace free_breakfast {

// The chain's root at one geometry: the molecule from atoms in memory, where
// MoleculeStep reads a file. Not MoleculeStep itself, whose load reads this
// step's product: a registry holding it must hold OptimizeStep too.
struct TrialGeometryStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<Atom> atoms;
  int32_t charge{0};
  int32_t multiplicity{1};

  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, atoms, charge, multiplicity)};
  }
};

// The CASSCF chain's ScfOutput at every geometry after the first: the last
// evaluation's CASSCF orbitals, core, active, virtual, Loewdin-orthonormalized
// in this geometry's overlap, C (C^T S C)^-1/2, the orthonormal set nearest
// them, so the active space the CasscfStep starts from is the one it
// converged last time and not a fresh pick that could swap orbitals. Only
// the CasscfStep reads it, by index over its first columns: the energy is
// NaN, no SCF ran, and d is left zero.
struct CarriedOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  // AO by MO, column-major, on the host: the last CasscfOutput's c.
  std::vector<double> c;
  // One per column: the last CasscfOutput's mo_energies.
  std::vector<double> mo_energies;
  // The first geometry's ScfOutput's.
  int32_t n_alpha{0};
  int32_t n_beta{0};

  [[nodiscard]] Result<ScfOutput> run(View<BasisSetOutput, OverlapOutput> in,
                                      DeviceResources &res) const {
    const BasisSet &basis = in.get<BasisSetOutput>().basis;
    const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
    if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
      return err_config("[{}] carries {} orbital coefficients and {} energies into a basis of {} "
                        "functions",
                        OptimizeStep::kName, c.size(), mo_energies.size(), s.n());
    }
    const ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<int32_t>(s.n());
    const size_t n2 = s.num_elements();
    // C, M = C^T S C and M^-1/2: held across orthogonalize, which takes the
    // workspace, so each is its own buffer.
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

Result<std::vector<double>> normalized_overlaps(std::span<const double> root_overlaps) {
  const auto n =
      static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(root_overlaps.size()))));
  if (n == 0 || root_overlaps.size() != n * n) {
    return err_config("normalized_overlaps: {} root overlaps; it takes n^2, n at least 1",
                      root_overlaps.size());
  }
  std::vector<double> out(root_overlaps.begin(), root_overlaps.end());
  for (size_t i = 0; i < n; ++i) {
    double norm = 0.0;
    for (size_t j = 0; j < n; ++j) {
      norm += out[(i * n) + j] * out[(i * n) + j];
    }
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) {
      return err_numerical("normalized_overlaps: the last geometry's CASSCF root {} overlaps no "
                           "root of this one",
                           i);
    }
    for (size_t j = 0; j < n; ++j) {
      out[(i * n) + j] /= norm;
    }
  }
  return out;
}

Result<std::vector<RootMatch>> match_roots(std::span<const double> root_overlaps,
                                           std::span<const int32_t> followed) {
  const std::vector<double> s = FBF_TRY(normalized_overlaps(root_overlaps));
  const size_t n = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(s.size()))));
  std::vector<RootMatch> match(n);
  std::vector<bool> taken(n, false);
  for (size_t i = 0; i < n; ++i) {
    size_t best = 0;
    for (size_t j = 1; j < n; ++j) {
      if (std::abs(s[(i * n) + j]) > std::abs(s[(i * n) + best])) {
        best = j;
      }
    }
    if (taken[best]) {
      return err_numerical("match_roots: the last geometry's CASSCF root {} matches no root of "
                           "this one alone: the roots are not followed",
                           i);
    }
    taken[best] = true;
    match[i] = {.root = static_cast<int32_t>(best), .overlap = s[(i * n) + best]};
  }
  for (const int32_t root : followed) {
    if (root < 0 || static_cast<size_t>(root) >= n) {
      return err_config("match_roots: root {} is followed, and there are {}", root, n);
    }
    const RootMatch &m = match[static_cast<size_t>(root)];
    if (std::abs(m.overlap) < kStateOverlapFloor) {
      return err_numerical("match_roots: the last geometry's CASSCF root {} has overlap {:.6f} "
                           "with its match, root {}, below {}: it is lost (a crossing), and is "
                           "not followed onto another surface",
                           root, std::abs(m.overlap), m.root, kStateOverlapFloor);
    }
  }
  return match;
}

Result<double> qd_state_overlap(std::span<const double> root_overlaps,
                                std::span<const double> previous, std::span<const double> current) {
  const size_t n = previous.size();
  if (n == 0 || current.size() != n || root_overlaps.size() != n * n) {
    return err_config("qd_state_overlap: {} and {} eigenvector entries over {} root overlaps; "
                      "it takes n, n and n^2, n at least 1",
                      previous.size(), current.size(), root_overlaps.size());
  }
  const std::vector<RootMatch> match = FBF_TRY(match_roots(root_overlaps));
  double overlap = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double sign = match[i].overlap < 0.0 ? -1.0 : 1.0;
    overlap += previous[i] * sign * current[static_cast<size_t>(match[i].root)];
  }
  overlap = std::abs(overlap);
  if (overlap < kStateOverlapFloor) {
    return err_numerical("qd_state_overlap: the QD state's overlap with the last geometry's is "
                         "{:.6f}, below {}: it is lost (a crossing), and is not followed onto "
                         "another surface",
                         overlap, kStateOverlapFloor);
  }
  return overlap;
}

// The CASSCF in @p cas over @p basis as the mcscf::Solution
// mcscf::state_overlaps reads, its orbitals and CI vectors on the device and
// its energies; it reads nothing else, so nothing else is filled in.
[[nodiscard]] Result<mcscf::Solution> overlap_solution(DeviceResources &res, const BasisSet &basis,
                                                       const CasscfOutput &cas) {
  const ScopedDeviceIndex on{res.dev_idx()};
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
      .gamma = {},
      .gamma2 = {},
      .root_gammas = {},
      .iterations = {},
  };
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                         cas.c.num_elements() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(res.sync());
  return solution;
}

// u_K from the QD Lagrangian's W = u_K u_K^T (n x n row-major): the column of
// its largest diagonal element over that element's square root, so the
// division is by the largest |u_r| of all.
[[nodiscard]] std::vector<double> qd_eigenvector(std::span<const double> w) {
  const auto n = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(w.size()))));
  size_t m = 0;
  for (size_t r = 1; r < n; ++r) {
    if (w[(r * n) + r] > w[(m * n) + m]) {
      m = r;
    }
  }
  const double scale = std::sqrt(w[(m * n) + m]);
  std::vector<double> u(n);
  for (size_t r = 0; r < n; ++r) {
    u[r] = w[(m * n) + r] / scale;
  }
  return u;
}

// The state an OptimizeStep follows, as the last evaluation left it: the
// CASSCF's basis and solution, and the root followed (a state-averaged root)
// or u_K over its roots (a QD state).
struct FollowedState {
  BasisSet basis;
  mcscf::Solution solution;
  int32_t root{0};
  std::vector<double> u;
};

// One evaluation's following, for the history: the matched overlap (NaN at
// the first, which has no last), and the root followed or H_eff's smallest
// gap.
struct FollowRecord {
  double overlap;
  std::optional<int32_t> root;
  std::optional<double> heff_gap;
};

// @p atoms with their positions replaced by @p x (bohr, 3 per atom).
[[nodiscard]] std::vector<Atom> placed(std::vector<Atom> atoms, std::span<const double> x) {
  for (size_t a = 0; a < atoms.size(); ++a) {
    atoms[a].position = {x[(3 * a) + 0], x[(3 * a) + 1], x[(3 * a) + 2]};
  }
  return atoms;
}

// One XYZ frame of @p atoms (Angstrom), @p comment its second line.
[[nodiscard]] std::string xyz_frame(const std::vector<Atom> &atoms, std::string_view comment) {
  std::string out = std::format("{}\n{}\n", atoms.size(), comment);
  for (const Atom &atom : atoms) {
    out += std::format("{:<2} {:18.12f} {:18.12f} {:18.12f}\n", element_symbol(atom.element),
                       atom.position.x * kAngstromPerBohr<double>,
                       atom.position.y * kAngstromPerBohr<double>,
                       atom.position.z * kAngstromPerBohr<double>);
  }
  return out;
}

Result<OptimizeOutput> OptimizeStep::run(View<> /*in*/, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  if (casscf_gradient && nevpt2_gradient) {
    return err_config("[{}] optimizes the energy of [{}] or of [{}], and both are set: keep the "
                      "one to optimize",
                      kName, CasscfGradientStep::kName, Nevpt2GradientStep::kName);
  }
  // The two CASSCF gradients' own refusal, made before the first SCF and
  // CASSCF.
  if ((casscf_gradient || nevpt2_gradient) && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] reruns [{}], which runs over density fitting only, and [{}] mode "
                           "is 'integral_direct': set [{}] mode = 'density_fitting'",
                           kName,
                           casscf_gradient ? CasscfGradientStep::kName : Nevpt2GradientStep::kName,
                           EriStep::kName, EriStep::kName);
  }
  DeviceResources &res = node.device(0);
  const Molecule start =
      FBF_TRY(read_xyz(res, molecule.xyz, molecule.charge, molecule.multiplicity));
  std::vector<double> x0;
  for (const Atom &atom : start.atoms()) {
    x0.insert(x0.end(), {atom.position.x, atom.position.y, atom.position.z});
  }

  // The chain writes nothing: its steps' outputs are the outer pipeline's.
  ScfStep chain_scf = scf;
  chain_scf.molden.clear();
  CasscfStep chain_casscf = casscf;
  chain_casscf.molden.clear();
  CasscfGradientStep chain_casscf_gradient = casscf_gradient.value_or(CasscfGradientStep{});
  chain_casscf_gradient.json_path.clear();
  Nevpt2GradientStep chain_nevpt2_gradient = nevpt2_gradient.value_or(Nevpt2GradientStep{});
  chain_nevpt2_gradient.json_path.clear();
  // After the first CASSCF: its orbitals for the next evaluation's start, and
  // the CasscfStep that takes them, by index, as they come, core, active,
  // virtual.
  std::optional<CarriedOrbitalsStep> carried;
  CasscfStep carried_casscf = chain_casscf;
  // Over a state-averaged root or a QD state: the state as the last
  // evaluation left it, and per evaluation how it was followed.
  std::optional<FollowedState> followed;
  std::vector<FollowRecord> follow_record;
  // The orbitals the next evaluation starts from: @p cas's, and the
  // CasscfStep that takes them.
  const auto carry = [&](const CasscfOutput &cas, int32_t n_alpha, int32_t n_beta) -> Status {
    CarriedOrbitalsStep next{.c = std::vector<double>(cas.c.num_elements()),
                             .mo_energies = cas.mo_energies,
                             .n_alpha = n_alpha,
                             .n_beta = n_beta};
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(next.c.data(), cas.c.data(), next.c.size() * sizeof(double),
                                       wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    carried = std::move(next);
    carried_casscf.active.clear();
    carried_casscf.avas.clear();
    carried_casscf.ncas = cas.ncas;
    carried_casscf.nelecas = cas.nelecas;
    return {};
  };
  // @p cas's roots over @p basis as the Solution state_overlaps reads, and
  // their overlaps with the last evaluation's (empty at the first).
  const auto overlaps_with_last =
      [&](const BasisSet &basis,
          const CasscfOutput &cas) -> Result<std::pair<mcscf::Solution, std::vector<double>>> {
    mcscf::Solution solution = FBF_TRY(overlap_solution(res, basis, cas));
    std::vector<double> s;
    if (followed) {
      s = FBF_TRY(mcscf::state_overlaps(res, followed->basis, followed->solution, basis, solution,
                                        cas.ncore, cas.ncas, cas.nelecas, start.multiplicity()));
    }
    return std::pair{std::move(solution), std::move(s)};
  };
  // The CASSCF chain's root, at the end of @p chain (which stops at the
  // CasscfStep): over a state average, the root the last evaluation's
  // matched, its gradient then taken; one state's, the configured root's.
  const auto root_result = [&](const auto &chain, std::span<double> gradient) -> Result<double> {
    const CasscfOutput &cas = *chain.template get<CasscfOutput>();
    CasscfGradientStep step = chain_casscf_gradient;
    if (cas.energies.size() > 1) {
      const BasisSet &basis = chain.template get<BasisSetOutput>()->basis;
      auto [solution, s] = FBF_TRY(overlaps_with_last(basis, cas));
      double overlap = std::numeric_limits<double>::quiet_NaN();
      if (followed) {
        const std::array last{followed->root};
        const std::vector<RootMatch> match = FBF_TRY(match_roots(s, last));
        const RootMatch &m = match[static_cast<size_t>(followed->root)];
        overlap = std::abs(m.overlap);
        step.root = m.root;
        FBF_LOG_INFO("optimize: CASSCF root {} followed to root {}, matched overlap {:.8f} "
                     "({:.8f} before its row is normalized)",
                     followed->root, m.root, overlap,
                     std::abs(s[(static_cast<size_t>(followed->root) * cas.energies.size()) +
                                static_cast<size_t>(m.root)]));
      } else {
        FBF_LOG_INFO("optimize: CASSCF root {} at the start", step.root);
      }
      follow_record.push_back({.overlap = overlap, .root = step.root, .heff_gap = std::nullopt});
      followed = FollowedState{.basis = basis, .solution = std::move(solution), .root = step.root};
    }
    const CasscfGradientOutput out = FBF_TRY(step.run(
        View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
             CoreHamiltonianOutput, CasscfOutput>{
            *chain.template get<MoleculeOutput>(),
            *chain.template get<BasisSetOutput>(),
            *chain.template get<PrimitivePairClassesOutput>(),
            *chain.template get<EriOutput>(),
            *chain.template get<CoreHamiltonianOutput>(),
            cas,
        },
        res));
    std::ranges::copy(out.gradient.total, gradient.begin());
    return cas.energies[static_cast<size_t>(step.root)];
  };
  // The NEVPT2 chain's energy, at the end of @p chain: SC's or PC's, as the
  // chain's variant picked, of the root: E_cas + E_corr; or QD's, of state
  // K: H_eff's eigenvalue E_K, and u_K, H_eff's eigenvector, recovered from
  // the Lagrangian's W = u_K u_K^T. H_eff's smallest gap is the state's, the
  // root gap the CASSCF's.
  const auto nevpt2_result = [&](const auto &chain, std::span<double> gradient) -> Result<double> {
    const CasscfOutput &cas = *chain.template get<CasscfOutput>();
    double energy = 0.0;
    std::vector<double> u;
    double heff_gap = 0.0;
    double root_gap = 0.0;
    std::visit(
        [&]<class N>(const N &g) {
          if constexpr (requires { g.energies.total; }) {
            std::ranges::copy(g.terms.total, gradient.begin());
            energy =
                cas.energies[static_cast<size_t>(chain_nevpt2_gradient.root)] + g.energies.total;
          } else {
            std::ranges::copy(g.total, gradient.begin());
            energy = g.lagrangian.eigenvalues[static_cast<size_t>(chain_nevpt2_gradient.state)];
            u = qd_eigenvector(g.lagrangian.w);
            heff_gap = g.heff_gap;
            root_gap = g.root_gap;
          }
        },
        chain.template get<Nevpt2GradientOutput>()->gradient);
    if (!chain_nevpt2_gradient.quasi_degenerate) {
      return energy;
    }
    // The state followed: this CASSCF's roots matched to the last's, and
    // u_K to the last u_K in the matched basis.
    const BasisSet &basis = chain.template get<BasisSetOutput>()->basis;
    auto [solution, s] = FBF_TRY(overlaps_with_last(basis, cas));
    double overlap = std::numeric_limits<double>::quiet_NaN();
    if (followed) {
      overlap = FBF_TRY(qd_state_overlap(s, followed->u, u));
      FBF_LOG_INFO("optimize: QD state {} matched overlap {:.8f}, H_eff gap {:.6f}, CASSCF "
                   "root gap {:.6f}",
                   chain_nevpt2_gradient.state, overlap, heff_gap, root_gap);
    } else {
      FBF_LOG_INFO("optimize: QD state {} at the start, H_eff gap {:.6f}, CASSCF root gap "
                   "{:.6f}",
                   chain_nevpt2_gradient.state, heff_gap, root_gap);
    }
    follow_record.push_back({.overlap = overlap, .root = std::nullopt, .heff_gap = heff_gap});
    followed = FollowedState{.basis = basis, .solution = std::move(solution), .u = std::move(u)};
    return energy;
  };
  // The CASSCF chain at @p trial, through the CasscfStep and then @p last
  // (none for the CASSCF root, whose gradient @p finish takes once the root
  // is matched), @p finish's energy and gradient: at the first geometry the
  // SCF, and the active space picked from it; at every later one no SCF,
  // the last orbitals carried over.
  const auto casscf_chain = [&]<class... G>(const TrialGeometryStep &trial, const auto &finish,
                                            std::span<double> gradient,
                                            const G &...last) -> Result<double> {
    if (!carried) {
      auto chain =
          FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                            CoreHamiltonianStep, GuessStep, ScfStep, CasscfStep, G...>::create({
              trial,
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              orthogonalizer,
              CoreHamiltonianStep{},
              guess,
              chain_scf,
              chain_casscf,
              last...,
          })));
      FBF_TRY(chain.run(node));
      const ScfOutput &rhf = *chain.template get<ScfOutput>();
      FBF_TRY(carry(*chain.template get<CasscfOutput>(), rhf.n_alpha, rhf.n_beta));
      return finish(chain, gradient);
    }
    auto chain =
        FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                          PrimitivePairClassesStep, EriStep, OverlapStep, CoreHamiltonianStep,
                          CarriedOrbitalsStep, CasscfStep, G...>::create({
            trial,
            basis_set,
            SphericalTransformStep{},
            PrimitivePairClassesStep{},
            eri,
            OverlapStep{},
            CoreHamiltonianStep{},
            *carried,
            carried_casscf,
            last...,
        })));
    FBF_TRY(chain.run(node));
    FBF_TRY(carry(*chain.template get<CasscfOutput>(), carried->n_alpha, carried->n_beta));
    return finish(chain, gradient);
  };
  // Every evaluated geometry, in order, for the trajectory.
  std::vector<std::vector<Atom>> frames;
  const auto energy_and_gradient = [&](std::span<const double> x,
                                       std::span<double> gradient) -> Result<double> {
    frames.push_back(placed(start.atoms(), x));
    const TrialGeometryStep trial{frames.back(), start.charge(), start.multiplicity()};
    double energy = 0.0;
    if (casscf_gradient) {
      energy = FBF_TRY(casscf_chain(trial, root_result, gradient));
    } else if (nevpt2_gradient) {
      energy = FBF_TRY(casscf_chain(trial, nevpt2_result, gradient, chain_nevpt2_gradient));
    } else {
      auto chain =
          FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                            CoreHamiltonianStep, GuessStep, ScfStep, GradientStep>::create({
              trial,
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              orthogonalizer,
              CoreHamiltonianStep{},
              guess,
              chain_scf,
              GradientStep{},
          })));
      FBF_TRY(chain.run(node));
      std::ranges::copy(chain.get<GradientOutput>()->gradient.total, gradient.begin());
      energy = chain.get<ScfOutput>()->energy;
    }
    FBF_LOG_INFO("optimize: evaluation {:>3} E {:.10f} max |g| {:.3e}", frames.size() - 1, energy,
                 std::ranges::max(gradient, {}, [](double v) { return std::abs(v); }));
    return energy;
  };
  const geometry::Solution solution = FBF_TRY(geometry::optimize(x0, energy_and_gradient, options));

  const std::vector<Atom> optimum = placed(start.atoms(), solution.x);
  if (!trajectory.empty()) {
    std::string text;
    for (size_t i = 0; i < frames.size() && i < solution.iterations.size(); ++i) {
      const geometry::Iteration &it = solution.iterations[i];
      text += xyz_frame(frames[i], std::format("evaluation {} E = {:.12f}{}", i, it.energy,
                                               it.accepted ? "" : " (rejected)"));
    }
    FBF_TRY(write_file(trajectory, text));
  }
  if (!final_xyz.empty()) {
    FBF_TRY(write_file(final_xyz,
                       xyz_frame(optimum, std::format("optimized E = {:.12f}", solution.energy))));
  }
  if (!history.empty()) {
    json iterations = json::array();
    for (size_t i = 0; i < solution.iterations.size(); ++i) {
      const geometry::Iteration &it = solution.iterations[i];
      json entry = {
          {"energy", it.energy},
          {"max_force", it.max_force},
          {"rms_force", it.rms_force},
          {"max_displacement", it.max_displacement},
          {"rms_displacement", it.rms_displacement},
          {"trust_radius", it.trust_radius},
          {"accepted", it.accepted},
      };
      // Over a followed state: its matched overlap, null at the first
      // evaluation, and the root followed or H_eff's smallest gap.
      if (i < follow_record.size()) {
        const FollowRecord &f = follow_record[i];
        entry["state_overlap"] = std::isnan(f.overlap) ? json(nullptr) : json(f.overlap);
        if (f.root) {
          entry["root"] = *f.root;
        }
        if (f.heff_gap) {
          entry["heff_gap"] = *f.heff_gap;
        }
      }
      iterations.push_back(std::move(entry));
    }
    FBF_TRY(write_json(history, json{{"energy", solution.energy}, {"iterations", iterations}}));
  }
  return OptimizeOutput{.molecule = Molecule(res, optimum, start.charge(), start.multiplicity()),
                        .energy = solution.energy};
}

} // namespace free_breakfast
