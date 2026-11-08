/// @file
/// @brief `free-breakfast-qd-nevpt2-hessian`: the analytic SC- and
///        PC-QD-NEVPT2 nuclear gradients G(W) of this tree at displaced
///        geometries, the points `tools/nevpt2_hessian/qd_nevpt2_hessian.py`
///        differences into the QD Hessian references (#953).
///
/// The QD Hessian's oracle is a stencil over the tree's own analytic QD
/// gradient (docs/nevpt2/nevpt2.tex, Table qdhesschecks's semi-numerical
/// row): `qd_*_gradient` is ladder-validated against the tree's energy
/// (`test/gtest/data/qd_nevpt2_gradient/`), and nothing outside this tree
/// has a QD-NEVPT2 gradient. Python cannot call it, so this tool runs the
/// points and the generator judges them.
///
/// It reads a request (JSON, the generator's): the system (atoms in bohr,
/// basis, active space, weights), the schemes ("sc", "pc"), the Z-vectors' tolerance,
/// and a list of points, each a 3N displacement (bohr, atom-major), the index
/// of an earlier point whose orbitals start it or null, and optionally its
/// own Z-vector tolerance (the generator's noise measurement solves some
/// points twice). At every point:
///
/// - **The chain and its guard** are `free_breakfast.tools.qd_nevpt2_gradient`'s
///   `ChainSampler` (polished, every point to ||g|| < kPolishTarget), as
///   `qd_nevpt2_gradient/` was made: the DF-SA-CASSCF re-converged from the
///   reference's orbitals, its roots matched by `mcscf::state_overlaps`, the
///   active space and CI overlaps within their bounds, and PC's cut the
///   reference's. Added here, as that tool's `main.cpp` adds them: each
///   root's energy within kPredictionBound |dR|^2 of its first-order
///   prediction from `mcscf::df_gradient` at the reference, and H_eff's
///   eigenstates matched to the reference's by the overlap of u_K in the
///   matched root basis, never by rank: `pipeline::hessian`'s
///   `tracked_qd_state`, the `[hessian]` step's own match (within 1e-3 of
///   1, no tie).
/// - **The gradients.** The sampler hands out energies and orbitals, not the
///   chain's products, so the point's chain is rerun from its converged
///   orbitals (the same steps and bars; the CASSCF starts converged), its
///   roots matched to the reference's again, and over it one several-W
///   `nevpt2::qd_sc_gradient` (and `qd_pc_gradient`) at, in the reference's
///   root basis carried into the rerun's: each state's W^K = u_K u_K^T
///   (u_K the matched eigenvector, so the state is tracked, Eq. qd-hess's
///   whole state Hessian); each requested pair's W^P + W^Q (Eq. qd-pair); and
///   each requested seam's W = 1/2 (R_X R_Y^T + R_Y R_X^T), R the reference's
///   eigenvectors held fixed (Eq. qd-lowdin's first term).
/// - **At a crossing** (#1059: the request's `crossing`, two states P and Q
///   degenerate at the reference, a MECI): u_P and u_Q are not defined one
///   by one there (any rotation within their eigenspace is as good, and a
///   step of any size turns them by O(1)), so they are not matched by u_K:
///   their two-state eigenspace is (`pipeline::hessian`'s
///   `tracked_qd_pair`, a principal-angle match), and the other states by
///   u_K. No W^P or W^Q is
///   taken, only the pair's W^P + W^Q, the projector onto that eigenspace
///   (Eq. qd-pair), and the seams; a pair holding one of P, Q alone is
///   refused (a cone's vertex).
///
/// It writes, per point, the matched CASSCF and QD energies, H_eff in the
/// reference's root basis, the matched u_K, the guard's values (a crossing
/// pair's subspace miss, `pair_miss`, among them) and every gradient
/// (n_atom x 3 row-major, hartree/bohr), in the order of the output's
/// `states`, `pairs` and `seams`; with `gradients` false, all but the
/// gradients. The generator does every judgement.
///
/// `free-breakfast-qd-nevpt2-hessian <request.json> <points.json>`. A
/// refused point fails the run and writes nothing.

#include "diagnostics.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.atom;    // element_from_symbol
import free_breakfast.io.base; // write_file
import free_breakfast.io.json;
import free_breakfast.mcscf.gradient;
import free_breakfast.mcscf.overlap;
import free_breakfast.nevpt2.gradient;
import free_breakfast.pipeline.pipeline;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.nevpt2;
// The H_eff eigenstates' match, the step's own (tracked_qd_state,
// tracked_qd_pair).
import free_breakfast.pipeline.hessian;
// The chain at every point and its CASSCF guard.
import free_breakfast.tools.qd_nevpt2_gradient;

namespace {

namespace fb = free_breakfast;

using fb::nevpt2::kCiTol;
using fb::nevpt2::kPredictionBound;

// The rerun's CASSCF bars: the sampler's (qd_nevpt2_gradient.cpp's
// kChainConvTolGrad, kChainConvTol, kChainMaxCycle). The rerun starts from
// the sampler's polished orbitals, so it stops at once.
constexpr double kRerunConvTolGrad = 1e-10;
constexpr double kRerunConvTol = 1e-13;
constexpr std::int32_t kRerunMaxCycle = 200;
// The rerun's roots against the sampler's at the same orbitals: the same
// solution up to the CI solve's convergence.
constexpr double kRerunEnergyTol = 1e-9;

// The rerun's root: the molecule from atoms in memory (the library's
// ChainGeometryStep).
struct GeometryStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<fb::Atom> atoms;

  [[nodiscard]] fb::Result<fb::MoleculeOutput> run(fb::View<> /*in*/,
                                                   fb::DeviceResources &res) const {
    return fb::MoleculeOutput{fb::Molecule(res, atoms, 0, 1)};
  }
};

// The rerun's ScfOutput: the sampler's converged orbitals at this geometry,
// already orthonormal in its overlap, as they are. Only the CasscfStep reads
// it.
struct ConvergedOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  std::vector<double> c; // AO by MO, column-major
  std::vector<double> mo_energies;
  std::int32_t n_alpha = 0;
  std::int32_t n_beta = 0;

  [[nodiscard]] fb::Result<fb::ScfOutput> run(fb::View<fb::BasisSetOutput> in,
                                              fb::DeviceResources &res) const {
    const fb::BasisSet &basis = in.get<fb::BasisSetOutput>().basis;
    fb::ScfOutput out{
        .energy = std::numeric_limits<double>::quiet_NaN(),
        .iterations = 0,
        .n_alpha = n_alpha,
        .n_beta = n_beta,
        .mo_energies = mo_energies,
        .c = fb::SphericalMatrixAO(res, basis),
        .d = fb::SphericalMatrixAO(res, basis),
    };
    if (c.size() != out.c.num_elements() || mo_energies.size() != out.c.n()) {
      return fb::err_config("carried {} orbital coefficients and {} energies into a basis of {}",
                            c.size(), mo_energies.size(), out.c.n());
    }
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(out.c.data(), c.data(), c.size() * sizeof(double),
                                               wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(res.sync());
    return out;
  }
};

using RerunChain = fb::Pipeline<GeometryStep, fb::BasisSetStep, fb::SphericalTransformStep,
                                fb::PrimitivePairClassesStep, fb::EriStep, fb::OverlapStep,
                                fb::CoreHamiltonianStep, ConvergedOrbitalsStep, fb::CasscfStep>;

// The mcscf::Solution a CasscfOutput is (pipeline/nevpt2_gradient's): its
// orbitals and CI vectors on the device, gamma the weighted average of the
// roots' 1-RDMs.
fb::Result<fb::mcscf::Solution> solution_of(fb::DeviceResources &res, const fb::BasisSet &basis,
                                            const fb::CasscfOutput &cas) {
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots =
          {
              .energies = cas.energies,
              .spin_squares = cas.spin_squares,
              .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
              .davidson = {},
              .fp32_iterations = 0,
          },
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
      .weights = cas.weights,
  };
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                             cas.vectors.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                             cas.c.num_elements() * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(res.sync());
  return solution;
}

// The request's system.
struct System {
  std::vector<fb::Atom> atoms; // bohr
  std::string basis;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<std::int32_t> active; // empty: the RHF's frontier orbitals
  std::vector<double> weights;
  // The schemes differenced, "sc" and/or "pc"; PC is converged (and
  // guarded) whenever it is one of them.
  std::vector<std::string> schemes;
  // The two states degenerate at the reference, tracked by their
  // eigenspace; none, every state by its own u_K.
  std::optional<std::array<std::size_t, 2>> crossing;
};

// One requested point.
struct Request {
  std::vector<double> displacement;
  // An earlier point whose orbitals start this one; none, the reference's.
  std::optional<std::size_t> from;
  // The Z-vectors' tolerance, the request's unless the point sets its own
  // (the noise measurement's looser solve).
  double zvector_tolerance = 0.0;
};

// The eigenstates of a displaced H_eff in the reference's order: @p vectors
// (state k at [k n, (k + 1) n), in the reference's root basis, so every root
// sign 1) each matched to the reference state K by pipeline::hessian's
// tracked_qd_state (the step's own R_P match, its 1e-3 bar and tie), which
// must make a permutation. @p matched takes u_k at [K n, (K + 1) n), signed
// so that its overlap with u_K is positive; the result is the matched
// energies, and @p miss the worst 1 - |overlap|.
//
// With a @p crossing {P, Q}, its two states are tracked_qd_pair's instead
// (their two-state eigenspace), put at P's and Q's places in ascending
// energy (the lower at the smaller of P and Q) as the displaced solve signs
// them, its miss in @p pair_miss (and in @p miss); every other state by
// tracked_qd_state, never one of the pair's.
fb::Result<std::vector<double>> matched_states(std::span<const double> energies,
                                               std::span<const double> vectors,
                                               std::span<const double> reference,
                                               std::optional<std::array<std::size_t, 2>> crossing,
                                               double &miss, double &pair_miss,
                                               std::vector<double> &matched) {
  const std::size_t n = energies.size();
  const std::vector<double> signs(n, 1.0);
  const auto state = [&](std::size_t k) { return reference.subspan(k * n, n); };
  std::vector<double> out(n);
  std::vector<bool> taken(n, false);
  matched.assign(n * n, 0.0);
  if (crossing) {
    std::vector<double> pair(state((*crossing)[0]).begin(), state((*crossing)[0]).end());
    pair.insert(pair.end(), state((*crossing)[1]).begin(), state((*crossing)[1]).end());
    const auto [states, pair_overlap_miss] = FBF_TRY(fb::tracked_qd_pair(pair, vectors, signs));
    const std::array<std::size_t, 2> slots{std::min((*crossing)[0], (*crossing)[1]),
                                           std::max((*crossing)[0], (*crossing)[1])};
    for (std::size_t x = 0; x < 2; ++x) {
      taken[states[x]] = true;
      out[slots[x]] = energies[states[x]];
      for (std::size_t i = 0; i < n; ++i) {
        matched[(slots[x] * n) + i] = vectors[(states[x] * n) + i];
      }
    }
    pair_miss = pair_overlap_miss;
    miss = std::max(miss, pair_overlap_miss);
  }
  for (std::size_t big_k = 0; big_k < n; ++big_k) {
    if (crossing && ((*crossing)[0] == big_k || (*crossing)[1] == big_k)) {
      continue;
    }
    const auto [best, overlap] = FBF_TRY(fb::tracked_qd_state(state(big_k), vectors, signs));
    if (taken[best]) {
      return fb::err_numerical("H_eff's eigenstate {} matches displaced state {}, already taken",
                               big_k, best);
    }
    taken[best] = true;
    out[big_k] = energies[best];
    const double sign = overlap < 0.0 ? -1.0 : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      matched[(big_k * n) + i] = sign * vectors[(best * n) + i];
    }
    miss = std::max(miss, 1.0 - std::abs(overlap));
  }
  return out;
}

// The weight matrices one scheme's gradient is taken at, n x n row-major in
// the reference's root basis: each of @p states' u_K u_K^T, then each pair's
// u_P u_P^T + u_Q u_Q^T, then each seam's 1/2 (R_X R_Y^T + R_Y R_X^T) at the
// reference's @p r0.
std::vector<std::vector<double>>
weight_matrices(std::span<const double> u, std::span<const double> r0, std::size_t n,
                std::span<const std::size_t> states,
                std::span<const std::array<std::size_t, 2>> pairs,
                std::span<const std::array<std::size_t, 2>> seams) {
  const auto outer = [n](std::span<const double> a, std::span<const double> b) {
    std::vector<double> w(n * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        w[(i * n) + j] = 0.5 * ((a[i] * b[j]) + (b[i] * a[j]));
      }
    }
    return w;
  };
  const auto column = [n](std::span<const double> m, std::size_t k) { return m.subspan(k * n, n); };
  std::vector<std::vector<double>> out;
  for (const std::size_t k : states) {
    out.push_back(outer(column(u, k), column(u, k)));
  }
  for (const auto &[p, q] : pairs) {
    std::vector<double> w = outer(column(u, p), column(u, p));
    const std::vector<double> wq = outer(column(u, q), column(u, q));
    for (std::size_t i = 0; i < n * n; ++i) {
      w[i] += wq[i];
    }
    out.push_back(std::move(w));
  }
  for (const auto &[x, y] : seams) {
    out.push_back(outer(column(r0, x), column(r0, y)));
  }
  return out;
}

// nlohmann's .at() throws on a missing key: the request is read in one place,
// and the exception becomes a Result.
fb::Status read_request(const fb::json &doc, System &sys, std::vector<Request> &points,
                        std::vector<std::array<std::size_t, 2>> &pairs,
                        std::vector<std::array<std::size_t, 2>> &seams, bool &gradients) {
  try {
    const fb::json &s = doc.at("system");
    for (const fb::json &atom : s.at("atoms")) {
      const std::string symbol = atom.at(0).get<std::string>();
      const auto coord = atom.at(1).get<std::vector<double>>();
      const fb::Element element = fb::element_from_symbol(symbol);
      if (element == fb::Element::Sentinel || coord.size() != 3) {
        return fb::err_config("request: atom {} at {} coordinates", symbol, coord.size());
      }
      sys.atoms.push_back(fb::Atom{{coord[0], coord[1], coord[2]}, element});
    }
    sys.basis = s.at("basis").get<std::string>();
    sys.ncas = s.at("ncas").get<std::int32_t>();
    sys.nelecas = s.at("nelecas").get<std::int32_t>();
    if (!s.at("active").is_null()) {
      sys.active = s.at("active").get<std::vector<std::int32_t>>();
    }
    sys.weights = s.at("weights").get<std::vector<double>>();
    sys.schemes = doc.at("schemes").get<std::vector<std::string>>();
    if (doc.contains("crossing") && !doc.at("crossing").is_null()) {
      sys.crossing = doc.at("crossing").get<std::array<std::size_t, 2>>();
    }
    gradients = doc.at("gradients").get<bool>();
    pairs = doc.at("pairs").get<std::vector<std::array<std::size_t, 2>>>();
    seams = doc.at("seams").get<std::vector<std::array<std::size_t, 2>>>();
    for (const fb::json &p : doc.at("points")) {
      Request r{.displacement = p.at("displacement").get<std::vector<double>>()};
      if (!p.at("from").is_null()) {
        r.from = p.at("from").get<std::size_t>();
      }
      r.zvector_tolerance = p.contains("zvector_tolerance")
                                ? p.at("zvector_tolerance").get<double>()
                                : doc.at("zvector_tolerance").get<double>();
      points.push_back(std::move(r));
    }
  } catch (const std::exception &e) {
    return fb::err_io("request: {}", e.what());
  }
  const std::size_t n = sys.weights.size();
  if (n < 2) {
    return fb::err_config("request: QD needs two roots or more, not {}", n);
  }
  if (sys.schemes.empty() || std::ranges::any_of(sys.schemes, [](const std::string &s) {
        return s != "sc" && s != "pc";
      })) {
    return fb::err_config("request: schemes must be \"sc\" and/or \"pc\"");
  }
  for (const auto &list : {pairs, seams}) {
    for (const auto &[p, q] : list) {
      if (p >= n || q >= n || p == q) {
        return fb::err_config("request: a pair ({}, {}) of {} states", p, q, n);
      }
    }
  }
  if (sys.crossing) {
    const auto [p, q] = *sys.crossing;
    if (p >= n || q >= n || p == q) {
      return fb::err_config("request: a crossing ({}, {}) of {} states", p, q, n);
    }
    // E_P + E_S, S outside the crossing pair, holds a cone's vertex.
    for (const auto &pair : pairs) {
      const auto held =
          std::ranges::count_if(pair, [&](std::size_t k) { return k == p || k == q; });
      if (held == 1) {
        return fb::err_config("request: the pair ({}, {}) holds one state of the crossing ({}, {})",
                              pair[0], pair[1], p, q);
      }
    }
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (points[i].displacement.size() != 3 * sys.atoms.size()) {
      return fb::err_config("request: point {} has {} components for {} atoms", i,
                            points[i].displacement.size(), sys.atoms.size());
    }
    if (!(points[i].zvector_tolerance > 0.0)) {
      return fb::err_config("request: point {}'s Z-vector tolerance is {}", i,
                            points[i].zvector_tolerance);
    }
    if (points[i].from && *points[i].from >= i) {
      return fb::err_config("request: point {} starts from point {}, not an earlier one", i,
                            *points[i].from);
    }
  }
  return {};
}

fb::Status run(std::span<char *const> args) {
  if (args.size() != 3) {
    return fb::err_config("usage: {} <request.json> <points.json>",
                          args.empty() ? "free-breakfast-qd-nevpt2-hessian" : args[0]);
  }
  const fb::json request = FBF_TRY(fb::read_json(std::filesystem::path{args[1]}));
  System sys;
  std::vector<Request> requests;
  std::vector<std::array<std::size_t, 2>> pairs;
  std::vector<std::array<std::size_t, 2>> seams;
  bool gradients = true;
  FBF_TRY(read_request(request, sys, requests, pairs, seams, gradients));
  const std::size_t n = sys.weights.size();
  const std::size_t dim = 3 * sys.atoms.size();
  // The states whose own W^K is taken: all but a crossing pair's.
  std::vector<std::size_t> states;
  for (std::size_t k = 0; k < n; ++k) {
    if (!sys.crossing || ((*sys.crossing)[0] != k && (*sys.crossing)[1] != k)) {
      states.push_back(k);
    }
  }

  // The CASSCF logs every macro iteration at info; this tool prints its own
  // progress.
  fb::log::set_level(fb::log::LogLevel::Warn);
  fb::NodeResources node;
  if (node.device_count() == 0) {
    return fb::err_gpu("no GPU device is visible");
  }
  fb::DeviceResources &res = node.device(0);
  const auto started = std::chrono::steady_clock::now();
  fb::nevpt2::ChainSampler sampler = FBF_TRY(fb::nevpt2::ChainSampler::create(
      node,
      fb::nevpt2::NumericalChain{
          .atoms = sys.atoms,
          .basis = sys.basis,
          .ncas = sys.ncas,
          .nelecas = sys.nelecas,
          .active = sys.active,
          .weights = sys.weights,
          .pc = std::ranges::contains(sys.schemes, std::string("pc")),
      },
      true));
  const fb::CasscfOutput &cas0 = sampler.casscf();
  const fb::Nevpt2Output &nev0 = sampler.nevpt2();
  const fb::nevpt2::ChainPoint &ref = sampler.reference();
  const fb::DensityFitted &fit0 = sampler.fit();
  std::vector<std::vector<double>> casscf_gradient;
  for (std::size_t r = 0; r < n; ++r) {
    casscf_gradient.push_back(
        FBF_TRY(fb::mcscf::df_gradient(res, sampler.molecule(), sampler.basis(), sampler.classes(),
                                       fit0.aux, sampler.core_hamiltonian(), fit0.b, fit0.l.data(),
                                       sampler.solution(), cas0.ncore, cas0.ncas,
                                       static_cast<std::int32_t>(r)))
            .total);
  }
  const std::vector<std::string_view> schemes(sys.schemes.begin(), sys.schemes.end());
  const auto reference_vectors = [&](std::string_view scheme) -> const std::vector<double> & {
    return scheme == "sc" ? nev0.qd_vectors : nev0.pc_qd_vectors;
  };

  fb::json reference{
      {"casscf_energies", cas0.energies},   {"gradient_norm", cas0.gradient_norm},
      {"casscf_gradient", casscf_gradient}, {"core_gap", ref.core_gap},
      {"virtual_gap", ref.virtual_gap},     {"polish_passes", sampler.polish_passes()}};
  for (const std::string_view scheme : schemes) {
    const bool sc = scheme == "sc";
    reference[std::string(scheme)] = {{"energies", sc ? nev0.qd_energies : nev0.pc_qd_energies},
                                      {"heff", sc ? nev0.qd_heff : nev0.pc_qd_heff},
                                      {"vectors", reference_vectors(scheme)}};
  }
  {
    fb::json dropped = fb::json::array();
    for (const fb::nevpt2::PcClassEnergies &c : nev0.pc_qd_couplings) {
      dropped.push_back(c.dropped);
    }
    reference["pc_dropped"] = dropped;
  }

  // The rerun's steps, configured as the sampler's are.
  fb::BasisSetStep basis_set;
  basis_set.basis = sys.basis;
  fb::EriStep eri;
  eri.mode = fb::EriMode::DensityFitting;
  eri.aux_basis = fb::nevpt2::NumericalChain{}.aux_basis;
  fb::CasscfStep casscf;
  casscf.ncas = sys.ncas;
  casscf.nelecas = sys.nelecas;
  casscf.nroots = static_cast<std::int32_t>(n);
  casscf.weights = sys.weights;
  casscf.conv_tol = kRerunConvTol;
  casscf.conv_tol_grad = kRerunConvTolGrad;
  casscf.max_cycle = kRerunMaxCycle;
  const std::int32_t electrons = sampler.molecule().electron_count();

  std::vector<fb::nevpt2::ChainPoint> chain_points;
  fb::json out_points = fb::json::array();
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const Request &rq = requests[index];
    fb::nevpt2::ChainPoint p = rq.from
                                   ? FBF_TRY(sampler.at(rq.displacement, chain_points[*rq.from]))
                                   : FBF_TRY(sampler.at(rq.displacement));
    fb::json entry{{"casscf", p.casscf},     {"gradient_norm", p.gradient_norm},
                   {"subspace", p.subspace}, {"ci", p.ci},
                   {"core_gap", p.core_gap}, {"virtual_gap", p.virtual_gap}};
    // The energy prediction, away from the reference.
    const double dr2 = std::inner_product(rq.displacement.begin(), rq.displacement.end(),
                                          rq.displacement.begin(), 0.0);
    double prediction = 0.0;
    if (dr2 > 0.0) {
      for (std::size_t r = 0; r < n; ++r) {
        const double predicted =
            cas0.energies[r] + std::inner_product(casscf_gradient[r].begin(),
                                                  casscf_gradient[r].end(), rq.displacement.begin(),
                                                  0.0);
        prediction = std::max(prediction, std::abs(p.casscf[r] - predicted) / dr2);
      }
      if (prediction >= kPredictionBound) {
        return fb::err_numerical("point {}: a displaced CASSCF {:.3f} |dR|^2 off the first-order "
                                 "prediction",
                                 index, prediction);
      }
    }
    entry["prediction"] = prediction;
    entry["zvector_tolerance"] = rq.zvector_tolerance;
    // H_eff's eigenstates, matched.
    std::vector<std::vector<double>> matched_u;
    for (const std::string_view scheme : schemes) {
      const bool sc = scheme == "sc";
      double miss = 0.0;
      double pair_miss = 0.0;
      std::vector<double> u;
      fb::Result<std::vector<double>> matched =
          matched_states(sc ? p.sc : p.pc, sc ? p.sc_vectors : p.pc_vectors,
                         reference_vectors(scheme), sys.crossing, miss, pair_miss, u);
      if (!matched) {
        fb::Error error = std::move(matched).error();
        error.message = std::format("point {}: {}: {}", index, scheme, error.message);
        return std::unexpected(std::move(error));
      }
      const std::vector<double> energies = *std::move(matched);
      entry[std::string(scheme)] = {{"energies", energies},
                                    {"heff", sc ? p.sc_heff : p.pc_heff},
                                    {"vectors", u},
                                    {"vector_miss", miss},
                                    {"pair_miss", pair_miss}};
      matched_u.push_back(std::move(u));
    }

    if (gradients) {
      std::vector<fb::Atom> atoms = sys.atoms;
      for (std::size_t a = 0; a < atoms.size(); ++a) {
        atoms[a].position = {atoms[a].position.x + rq.displacement[3 * a],
                             atoms[a].position.y + rq.displacement[(3 * a) + 1],
                             atoms[a].position.z + rq.displacement[(3 * a) + 2]};
      }
      auto chain = FBF_TRY(RerunChain::create({
          GeometryStep{atoms},
          basis_set,
          fb::SphericalTransformStep{},
          fb::PrimitivePairClassesStep{},
          eri,
          fb::OverlapStep{},
          fb::CoreHamiltonianStep{},
          ConvergedOrbitalsStep{
              .c = p.c,
              .mo_energies = p.mo_energies,
              .n_alpha = electrons / 2,
              .n_beta = electrons / 2,
          },
          casscf,
      }));
      FBF_TRY(chain.run(node));
      const fb::CasscfOutput &cas = *chain.get<fb::CasscfOutput>();
      const fb::BasisSet &basis = chain.get<fb::BasisSetOutput>()->basis;
      const auto *fit = std::get_if<fb::DensityFitted>(&chain.get<fb::EriOutput>()->eri);
      if (fit == nullptr) {
        return fb::err_config("point {}: the rerun's ERIs are not density-fitted", index);
      }
      if (cas.gradient_norm >= kRerunConvTolGrad) {
        return fb::err_numerical("point {}: the rerun's CASSCF stopped at ||g|| {:.2e}", index,
                                 cas.gradient_norm);
      }
      const fb::mcscf::Solution solution = FBF_TRY(solution_of(res, basis, cas));
      // The rerun's roots in the reference's basis: reference root i is the
      // rerun's pi[i], up to signs[i].
      const std::vector<double> s =
          FBF_TRY(fb::mcscf::state_overlaps(res, sampler.basis(), sampler.solution(), basis,
                                            solution, cas0.ncore, cas0.ncas, cas0.nelecas, 1));
      std::vector<std::size_t> pi(n);
      std::vector<double> signs(n);
      std::vector<bool> taken(n, false);
      double rerun_energy = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < n; ++j) {
          if (std::abs(s[(i * n) + j]) > std::abs(s[(i * n) + best])) {
            best = j;
          }
        }
        if (taken[best] || 1.0 - std::abs(s[(i * n) + best]) >= kCiTol) {
          return fb::err_numerical("point {}: the rerun's roots do not match the reference's",
                                   index);
        }
        taken[best] = true;
        pi[i] = best;
        signs[i] = s[(i * n) + best] < 0.0 ? -1.0 : 1.0;
        rerun_energy = std::max(rerun_energy, std::abs(cas.energies[best] - p.casscf[i]));
      }
      if (rerun_energy >= kRerunEnergyTol) {
        return fb::err_numerical("point {}: the rerun's roots {:.2e} off the sampler's", index,
                                 rerun_energy);
      }
      entry["rerun"] = {{"gradient_norm", cas.gradient_norm}, {"energy", rerun_energy}};
      const fb::Molecule &molecule = chain.get<fb::MoleculeOutput>()->molecule;
      const auto &classes = chain.get<fb::PrimitivePairClassesOutput>()->classes;
      const fb::SphericalMatrixAO &h = chain.get<fb::CoreHamiltonianOutput>()->h;
      for (std::size_t si = 0; si < schemes.size(); ++si) {
        const std::string_view scheme = schemes[si];
        const std::vector<std::vector<double>> ws =
            weight_matrices(matched_u[si], reference_vectors(scheme), n, states, pairs, seams);
        // Each W into the rerun's root basis.
        std::vector<std::vector<double>> carried;
        for (const std::vector<double> &w : ws) {
          std::vector<double> c(n * n);
          for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
              c[(pi[i] * n) + pi[j]] = signs[i] * signs[j] * w[(i * n) + j];
            }
          }
          carried.push_back(std::move(c));
        }
        fb::json grads = fb::json::array();
        fb::json residuals = fb::json::array();
        if (scheme == "sc") {
          for (const fb::nevpt2::QdScGradient &g : FBF_TRY(fb::nevpt2::qd_sc_gradient(
                   res, molecule, basis, classes, fit->aux, h, fit->b, fit->l.data(), solution,
                   cas.ncore, cas.ncas, std::span<const std::vector<double>>{carried}, 0,
                   rq.zvector_tolerance))) {
            grads.push_back(g.total);
            residuals.push_back(g.lagrangian.multipliers.residual);
          }
        } else {
          for (const fb::nevpt2::QdPcGradient &g : FBF_TRY(fb::nevpt2::qd_pc_gradient(
                   res, molecule, basis, classes, fit->aux, h, fit->b, fit->l.data(), solution,
                   cas.ncore, cas.ncas, std::span<const std::vector<double>>{carried}, 0,
                   rq.zvector_tolerance))) {
            grads.push_back(g.total);
            residuals.push_back(g.lagrangian.multipliers.residual);
          }
        }
        if (grads.size() != ws.size() || grads[0].size() != dim) {
          return fb::err_numerical("point {}: {} gradients of {} for {} W", index, grads.size(),
                                   grads.empty() ? 0 : grads[0].size(), ws.size());
        }
        entry[std::string(scheme)]["gradients"] = grads;
        entry[std::string(scheme)]["zvector_residuals"] = residuals;
      }
    }
    out_points.push_back(std::move(entry));
    chain_points.push_back(std::move(p));
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::println("point {}/{}: ||g|| {:.1e}, prediction {:.3f}, {:.0f} s", index + 1,
                 requests.size(), chain_points.back().gradient_norm, prediction, elapsed);
  }
  fb::json crossing = nullptr;
  if (sys.crossing) {
    crossing = *sys.crossing;
  }
  const fb::json doc{{"reference", reference}, {"states", states},     {"pairs", pairs},
                     {"seams", seams},         {"crossing", crossing}, {"points", out_points}};
  FBF_TRY(fb::write_file(std::filesystem::path{args[2]}, fb::json_to_string(doc, 1) + '\n'));
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status) {
    std::println(std::cerr, "error: {}", status.error().message);
    return fb::to_exit_code(status.error().kind);
  }
  return 0;
}
