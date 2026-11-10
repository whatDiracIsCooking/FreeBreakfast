/// @file
/// @brief OptimizeStep: geometry::optimize over a nested Pipeline per
/// evaluation, the RHF chain or a CASSCF chain (to its gradient or its
/// NEVPT2's, a state-averaged root followed by match_roots and a QD state by
/// qd_state_overlap) with its orbitals
/// carried from one evaluation to the next, or geometry::optimize_meci over
/// that CASSCF chain's df_intersection, or its QD-NEVPT2's
/// qd_{sc,pc}_intersection (a pair followed by pair_overlap), and the
/// trajectory, final.xyz and history it writes.

module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kAngstromPerBohr
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom; // element_symbol
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.xyz;
import free_breakfast.mcscf.intersection;  // df_intersection
import free_breakfast.mcscf.overlap;       // state_overlaps
import free_breakfast.nevpt2.intersection; // qd_sc_intersection, qd_pc_intersection
import free_breakfast.pipeline.nevpt2;     // Nevpt2Step::resolve_frozen_core
import free_breakfast.pipeline.pipeline;
import free_breakfast.pipeline.footprint;

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

  // The Molecule, as MoleculeStep's.
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions &dims) const {
    return molecule_footprint(dims.natoms);
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

// The smaller singular value of [[a, b], [c, d]] in closed form: of
// (p + q) / 2 and |p - q| / 2, the latter. The square root of the invariants'
// discriminant would turn a rounding error of 1e-16 into 1e-8 where the two
// are equal, as at a clean swap.
[[nodiscard]] double smaller_singular_value(double a, double b, double c, double d) {
  const double p = std::hypot(a + d, b - c);
  const double q = std::hypot(a - d, b + c);
  return 0.5 * std::abs(p - q);
}

Result<double> pair_overlap(std::span<const double> root_overlaps, int32_t i, int32_t j) {
  const std::vector<double> s = FBF_TRY(normalized_overlaps(root_overlaps));
  const auto n = static_cast<int32_t>(std::lround(std::sqrt(static_cast<double>(s.size()))));
  if (i < 0 || j < 0 || i >= n || j >= n || i == j) {
    return err_config("pair_overlap: roots {} and {} are not two distinct roots of {}", i, j, n);
  }
  const auto at = [&](int32_t r, int32_t c) {
    return s[(static_cast<size_t>(r) * static_cast<size_t>(n)) + static_cast<size_t>(c)];
  };
  const double smaller = smaller_singular_value(at(i, i), at(i, j), at(j, i), at(j, j));
  if (smaller < kStateOverlapFloor) {
    return err_numerical("pair_overlap: the last geometry's CASSCF roots {} and {} keep {:.6f} of "
                         "their space in this one's, below {}: the pair is lost to another root, "
                         "and is not followed onto it",
                         i, j, smaller, kStateOverlapFloor);
  }
  return smaller;
}

Result<double> pair_overlap(std::span<const double> root_overlaps, std::span<const double> previous,
                            std::span<const double> current, int32_t p, int32_t q) {
  const std::vector<double> s = FBF_TRY(normalized_overlaps(root_overlaps));
  const auto n = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(s.size()))));
  if (previous.size() != s.size() || current.size() != s.size()) {
    return err_config("pair_overlap: {} and {} eigenvector entries over {} root overlaps; it "
                      "takes n^2, n^2 and n^2",
                      previous.size(), current.size(), s.size());
  }
  const auto in_range = [&](int32_t k) { return k >= 0 && static_cast<size_t>(k) < n; };
  if (!in_range(p) || !in_range(q) || p == q) {
    return err_config("pair_overlap: states {} and {} are not two distinct states of {}", p, q, n);
  }
  // u_a(last)^T S u_b(this).
  const auto block = [&](int32_t a, int32_t b) {
    const std::span<const double> u = previous.subspan(static_cast<size_t>(a) * n, n);
    const std::span<const double> v = current.subspan(static_cast<size_t>(b) * n, n);
    double sum = 0.0;
    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < n; ++c) {
        sum += u[r] * s[(r * n) + c] * v[c];
      }
    }
    return sum;
  };
  const double smaller = smaller_singular_value(block(p, p), block(p, q), block(q, p), block(q, q));
  if (smaller < kStateOverlapFloor) {
    return err_numerical("pair_overlap: the last geometry's QD states {} and {} keep {:.6f} of "
                         "their space in this one's, below {}: the pair is lost to another "
                         "state, and is not followed onto it",
                         p, q, smaller, kStateOverlapFloor);
  }
  return smaller;
}

Status OptimizeStep::check_meci() const {
  if (!meci) {
    return {};
  }
  const auto [i, j] = *meci;
  if (casscf_gradient) {
    return err_config("[{}] meci optimizes the crossing of two roots, and [{}] the energy of one: "
                      "keep one of them",
                      kName, CasscfGradientStep::kName);
  }
  const bool qd = nevpt2_gradient.has_value();
  if (qd && !nevpt2_gradient->quasi_degenerate) {
    return err_config("[{}] meci crosses two QD-NEVPT2 states, and [{}] differentiates one "
                      "state's NEVPT2, which has none to cross: set [{}] quasi_degenerate = true",
                      kName, Nevpt2GradientStep::kName, Nevpt2GradientStep::kName);
  }
  if (qd && nevpt2_gradient->state != 0) {
    return err_config("[{}] meci = [{}, {}] names the QD states that cross, and [{}] state names "
                      "one ({}): leave [{}] state out",
                      kName, i, j, Nevpt2GradientStep::kName, nevpt2_gradient->state,
                      Nevpt2GradientStep::kName);
  }
  const std::string_view what = qd ? "QD states" : "roots";
  if (casscf.nroots < 2) {
    return err_config("[{}] meci crosses two {} of a state average, and [{}] nroots is {}: set "
                      "[{}] nroots to 2 or more",
                      kName, what, CasscfStep::kName, casscf.nroots, CasscfStep::kName);
  }
  if (i < 0 || j < 0 || i == j || std::max(i, j) >= casscf.nroots) {
    return err_config("[{}] meci = [{}, {}] is not two distinct {} of [{}]'s {}", kName, i, j, what,
                      CasscfStep::kName, casscf.nroots);
  }
  if (!qd && !casscf.weights.empty() &&
      casscf.weights[static_cast<size_t>(i)] != casscf.weights[static_cast<size_t>(j)]) {
    return err_unsupported("[{}] meci = [{}, {}] needs the two roots at equal weights, and [{}] "
                           "weights them {} and {}: the rotation between two roots of unequal "
                           "weight divides by their gap, singular at the crossing",
                           kName, i, j, CasscfStep::kName, casscf.weights[static_cast<size_t>(i)],
                           casscf.weights[static_cast<size_t>(j)]);
  }
  if (qd && molecule.multiplicity != 1) {
    return err_unsupported("[{}] meci over [{}] is implemented for a closed-shell singlet only, "
                           "and [{}] multiplicity is {}",
                           kName, Nevpt2GradientStep::kName, MoleculeStep::kName,
                           molecule.multiplicity);
  }
  if (eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] meci runs over density fitting only, and [{}] mode is "
                           "'integral_direct': set [{}] mode = 'density_fitting'",
                           kName, EriStep::kName, EriStep::kName);
  }
  return {};
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

// The CASSCF in @p cas over @p basis as the mcscf::Solution that
// mcscf::state_overlaps and mcscf::df_intersection read, as
// CasscfGradientStep rebuilds it: the orbitals and CI vectors on the device,
// the energies, and the weighted gamma beside the per-root RDMs.
[[nodiscard]] Result<mcscf::Solution> casscf_solution(DeviceResources &res, const BasisSet &basis,
                                                      const CasscfOutput &cas) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n2 = static_cast<size_t>(cas.ncas) * static_cast<size_t>(cas.ncas);
  if (cas.rdm1.size() != cas.weights.size() * n2) {
    return err_config("[{}] the CASSCF holds {} 1-RDM elements, not {} roots of {}",
                      OptimizeStep::kName, cas.rdm1.size(), cas.weights.size(), n2);
  }
  std::vector<double> gamma(n2, 0.0);
  for (size_t r = 0; r < cas.weights.size(); ++r) {
    for (size_t e = 0; e < n2; ++e) {
      gamma[e] += cas.weights[r] * cas.rdm1[(r * n2) + e];
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
      .weights = cas.weights,
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

// @p start with its positions replaced by @p x (bohr, 3 per atom).
[[nodiscard]] std::vector<Atom> placed(std::span<const Atom> start, std::span<const double> x) {
  std::vector<Atom> atoms(start.begin(), start.end());
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

// What a recorded walk has kept so far, and the one place its files are
// written: after every recorded evaluation, and at the walk's exit.
struct WalkRecord {
  // The step whose trajectory, final_xyz and history these are, each
  // skipped when empty.
  const OptimizeStep &step;
  std::span<const Atom> start;
  // Every geometry handed to the evaluation, in order: one more than the
  // entries while an evaluation runs, or after one fails.
  std::vector<std::vector<Atom>> frames;
  // One trajectory comment and one history entry per recorded evaluation.
  std::vector<std::string> comments;
  json entries = json::array();

  // The trajectory, of the recorded frames, and the history, @p head with
  // the entries as "iterations".
  [[nodiscard]] Status write(json head) const {
    if (!step.trajectory.empty()) {
      std::string text;
      for (size_t k = 0; k < frames.size() && k < comments.size(); ++k) {
        text += xyz_frame(frames[k], comments[k]);
      }
      FBF_TRY(write_file(step.trajectory, text));
    }
    if (!step.history.empty()) {
      head["iterations"] = entries;
      FBF_TRY(write_json(step.history, head));
    }
    return {};
  }

  // One recorded evaluation, and the files rewritten with it.
  [[nodiscard]] Status add(std::string comment, json entry) {
    comments.push_back(std::move(comment));
    entries.push_back(std::move(entry));
    return write(json::object());
  }

  // The walk's exit, every return of it: on @p walk's error, the files with
  // it as "error" and no final_xyz; converged, @p head's, and final_xyz, the
  // walk's optimum with @p final_comment.
  template <class S>
  [[nodiscard]] Result<S> finish(Result<S> walk, json head, std::string_view final_comment) const {
    if (!walk) {
      const Status written = write({{"error", walk.error().message}});
      if (!written) {
        FBF_LOG_ERROR("optimize: the failed walk's files are not written: {}",
                      written.error().message);
      }
      return walk;
    }
    FBF_TRY(write(std::move(head)));
    if (!step.final_xyz.empty()) {
      FBF_TRY(write_file(step.final_xyz, xyz_frame(placed(start, walk->x), final_comment)));
    }
    return walk;
  }
};

// Removes a final_xyz an earlier run left, so that only a converged walk's
// is ever there.
[[nodiscard]] Status remove_stale_final(const OptimizeStep &step) {
  if (step.final_xyz.empty()) {
    return {};
  }
  std::error_code ec;
  std::filesystem::remove(step.final_xyz, ec);
  if (ec) {
    return err_io("[{}] cannot remove the earlier run's {}: {}", OptimizeStep::kName,
                  step.final_xyz.string(), ec.message());
  }
  return {};
}

Result<geometry::Solution> recorded_optimize(
    const OptimizeStep &step, std::span<const Atom> start,
    const std::function<Result<double>(std::span<const double> x, std::span<double> gradient)>
        &energy_and_gradient,
    const std::function<void(size_t k, json &entry)> &annotate) {
  FBF_TRY(remove_stale_final(step));
  WalkRecord record{.step = step, .start = start};
  std::vector<double> x0;
  for (const Atom &atom : start) {
    x0.insert(x0.end(), {atom.position.x, atom.position.y, atom.position.z});
  }
  const auto evaluate = [&](std::span<const double> x, std::span<double> g) -> Result<double> {
    record.frames.push_back(placed(start, x));
    return energy_and_gradient(x, g);
  };
  const auto on_iteration = [&](const geometry::Iteration &it) -> Status {
    const size_t k = record.comments.size();
    json entry = {
        {"energy", it.energy},
        {"max_force", it.max_force},
        {"rms_force", it.rms_force},
        {"max_displacement", it.max_displacement},
        {"rms_displacement", it.rms_displacement},
        {"trust_radius", it.trust_radius},
        {"accepted", it.accepted},
    };
    if (annotate) {
      annotate(k, entry);
    }
    return record.add(
        std::format("evaluation {} E = {:.12f}{}", k, it.energy, it.accepted ? "" : " (rejected)"),
        std::move(entry));
  };
  Result<geometry::Solution> walk = geometry::optimize(x0, evaluate, step.options, on_iteration);
  json head;
  std::string final_comment;
  if (walk) {
    head = {{"energy", walk->energy}};
    final_comment = std::format("optimized E = {:.12f}", walk->energy);
  }
  return record.finish(std::move(walk), std::move(head), final_comment);
}

Result<geometry::MECISolution>
recorded_optimize_meci(const OptimizeStep &step, std::span<const Atom> start,
                       const std::function<Result<std::array<double, 2>>(
                           std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                           std::span<double> h_ij)> &energies_and_gradients,
                       const std::function<void(size_t k, json &entry)> &annotate,
                       const std::function<void(json &head)> &annotate_head) {
  FBF_TRY(remove_stale_final(step));
  WalkRecord record{.step = step, .start = start};
  std::vector<double> x0;
  for (const Atom &atom : start) {
    x0.insert(x0.end(), {atom.position.x, atom.position.y, atom.position.z});
  }
  const auto evaluate = [&](std::span<const double> x, std::span<double> g_i, std::span<double> g_j,
                            std::span<double> h_ij) -> Result<std::array<double, 2>> {
    record.frames.push_back(placed(start, x));
    return energies_and_gradients(x, g_i, g_j, h_ij);
  };
  const auto on_iteration = [&](const geometry::MECISolution::Iteration &it) -> Status {
    const size_t k = record.comments.size();
    const double mean = 0.5 * (it.lower + it.upper);
    json entry = {
        {"energy", mean},
        {"lower", it.lower},
        {"upper", it.upper},
        {"gap", it.gap},
        {"max_force", it.max_force},
        {"rms_force", it.rms_force},
        {"max_displacement", it.max_displacement},
        {"rms_displacement", it.rms_displacement},
        {"trust_radius", it.trust_radius},
        {"branching_dimension", it.branching_dimension},
        {"plane_carried", it.plane_carried},
    };
    if (annotate) {
      annotate(k, entry);
    }
    return record.add(std::format("evaluation {} E = {:.12f} gap = {:.3e}", k, mean, it.gap),
                      std::move(entry));
  };
  Result<geometry::MECISolution> walk =
      geometry::optimize_meci(x0, evaluate, step.options, on_iteration);
  json head;
  std::string final_comment;
  if (walk) {
    const double mean = 0.5 * (walk->lower + walk->upper);
    head = {{"energy", mean},
            {"lower", walk->lower},
            {"upper", walk->upper},
            {"gap", walk->upper - walk->lower}};
    if (annotate_head) {
      annotate_head(head);
    }
    final_comment = std::format("MECI E = {:.12f} gap = {:.3e}", mean, walk->upper - walk->lower);
  }
  return record.finish(std::move(walk), std::move(head), final_comment);
}

// The RHF chain at @p trial, as @p step configures it, with @p scf for its
// [scf]: what each evaluation runs, and what OptimizeStep::footprint folds.
[[nodiscard]] auto rhf_chain(const OptimizeStep &step, TrialGeometryStep trial, ScfStep scf) {
  return Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep, PrimitivePairClassesStep,
                  EriStep, OverlapStep, OrthogonalizerStep, CoreHamiltonianStep, GuessStep, ScfStep,
                  GradientStep>::create({
      std::move(trial),
      step.basis_set,
      SphericalTransformStep{},
      PrimitivePairClassesStep{},
      step.eri,
      OverlapStep{},
      step.orthogonalizer,
      CoreHamiltonianStep{},
      step.guess,
      std::move(scf),
      GradientStep{},
  });
}

std::optional<MemoryFootprint> OptimizeStep::footprint(const FootprintDimensions &dims) const {
  if (meci || casscf_gradient || nevpt2_gradient) {
    return std::nullopt;
  }
  // The chain is fixed, so it always plans; none were it ever not to.
  const auto chain = rhf_chain(*this, TrialGeometryStep{}, scf);
  if (!chain) {
    return std::nullopt;
  }
  std::vector<MemoryFootprint> steps;
  for (const StepFootprint &step : chain->footprints(dims)) {
    if (!step.footprint) {
      return std::nullopt;
    }
    steps.push_back(*step.footprint);
  }
  // The start, then each evaluation's chain freed, then the optimum's
  // Molecule while the start is still held: the chain holds a Molecule of
  // the same atoms, so the two Molecules together are under its peak.
  const MemoryFootprint molecule = molecule_footprint(dims.natoms);
  return then(drop(then(molecule, drop(plan_footprint(steps).footprint))), molecule);
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
  FBF_TRY(check_meci());
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

  // The chain writes nothing: its steps' outputs are the outer pipeline's.
  ScfStep chain_scf = scf;
  chain_scf.molden.clear();
  CasscfStep chain_casscf = casscf;
  chain_casscf.molden.clear();
  chain_casscf.orbitals_npy.clear();
  chain_casscf.ci_npy.clear();
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
    carried_casscf.orbitals.clear(); // a start from files is the first geometry's only
    carried_casscf.ci.clear();
    carried_casscf.ncas = cas.ncas;
    carried_casscf.nelecas = cas.nelecas;
    return {};
  };
  // @p cas's roots over @p basis as the Solution state_overlaps reads, and
  // their overlaps with the last evaluation's (empty at the first).
  const auto overlaps_with_last =
      [&](const BasisSet &basis,
          const CasscfOutput &cas) -> Result<std::pair<mcscf::Solution, std::vector<double>>> {
    mcscf::Solution solution = FBF_TRY(casscf_solution(res, basis, cas));
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
  // The evaluations begun, for the log.
  size_t evaluations = 0;
  const auto energy_and_gradient = [&](std::span<const double> x,
                                       std::span<double> gradient) -> Result<double> {
    const size_t evaluation = evaluations++;
    const TrialGeometryStep trial{placed(start.atoms(), x), start.charge(), start.multiplicity()};
    double energy = 0.0;
    if (casscf_gradient) {
      energy = FBF_TRY(casscf_chain(trial, root_result, gradient));
    } else if (nevpt2_gradient) {
      energy = FBF_TRY(casscf_chain(trial, nevpt2_result, gradient, chain_nevpt2_gradient));
    } else {
      auto chain = FBF_TRY(rhf_chain(*this, trial, chain_scf));
      FBF_TRY(chain.run(node));
      std::ranges::copy(chain.get<GradientOutput>()->gradient.total, gradient.begin());
      energy = chain.get<ScfOutput>()->energy;
    }
    FBF_LOG_INFO("optimize: evaluation {:>3} E {:.10f} max |g| {:.3e}", evaluation, energy,
                 std::ranges::max(gradient, {}, [](double v) { return std::abs(v); }));
    return energy;
  };
  // Under meci: one df_intersection of roots I and J at the end of the
  // CASSCF chain, or over QD states one qd_{sc,pc}_intersection of P and Q,
  // the pair first checked against the last evaluation's by pair_overlap.
  // g_I goes out through the chain's gradient span, g_J and h_IJ through
  // their own, and the two energies through the return; the chain's own
  // energy is their mean. Per evaluation, what the history adds of it.
  std::vector<json> meci_record;
  const bool qd_meci = meci && nevpt2_gradient.has_value();
  // Over QD states, the last evaluation's g_P, g_Q and h_PQ, each n_atom rows
  // of three, and the state average's size: the head's, since optimize_meci's
  // last evaluation is its crossing.
  json qd_branching;
  const auto meci_evaluation = [&](std::span<const double> x, std::span<double> g_i,
                                   std::span<double> g_j,
                                   std::span<double> h_ij) -> Result<std::array<double, 2>> {
    const size_t evaluation = evaluations++;
    const TrialGeometryStep trial{placed(start.atoms(), x), start.charge(), start.multiplicity()};
    const int32_t i = (*meci)[0];
    const int32_t j = (*meci)[1];
    std::array<double, 2> pair{};
    const auto intersection = [&](const auto &chain, std::span<double> gradient) -> Result<double> {
      const CasscfOutput &cas = *chain.template get<CasscfOutput>();
      const BasisSet &basis = chain.template get<BasisSetOutput>()->basis;
      auto [solution, s] = FBF_TRY(overlaps_with_last(basis, cas));
      const auto *const fit = std::get_if<DensityFitted>(&chain.template get<EriOutput>()->eri);
      if (fit == nullptr) {
        return err_unsupported("[{}] meci runs over density fitting only, and the chain's ERIs "
                               "are integral-direct",
                               kName);
      }
      const Molecule &molecule = chain.template get<MoleculeOutput>()->molecule;
      const auto &classes = chain.template get<PrimitivePairClassesOutput>()->classes;
      const SphericalMatrixAO &h = chain.template get<CoreHamiltonianOutput>()->h;
      double overlap = std::numeric_limits<double>::quiet_NaN();
      json record = {{"gradient_norm", cas.gradient_norm},
                     {"ci_residual_norm", cas.ci_residual_norm}};
      std::vector<double> vectors;
      if (qd_meci) {
        const bool pc = chain_nevpt2_gradient.variant == Nevpt2Variant::Pc;
        // [nevpt2_gradient]'s frozen core, resolved at this geometry's CASSCF.
        const nevpt2::QdIntersectionOptions options{
            .nfrozen = FBF_TRY(Nevpt2Step::resolve_frozen_core(Nevpt2GradientStep::kName,
                                                               chain_nevpt2_gradient.frozen_core,
                                                               molecule, cas.ncore))};
        const nevpt2::QdIntersection out =
            FBF_TRY(pc ? nevpt2::qd_pc_intersection(res, molecule, basis, classes, fit->aux, h,
                                                    fit->b, fit->l.data(), solution, cas.ncore,
                                                    cas.ncas, i, j, 0, options)
                       : nevpt2::qd_sc_intersection(res, molecule, basis, classes, fit->aux, h,
                                                    fit->b, fit->l.data(), solution, cas.ncore,
                                                    cas.ncas, i, j, 0, options));
        if (followed) {
          overlap = FBF_TRY(pair_overlap(s, followed->u, out.eigenvectors, i, j));
        }
        if (out.reference_pair_tau) {
          // Inside a degenerate CASSCF pair: the mean gradient alone, which
          // optimize_meci reads as g_I = g_J and h = 0, carrying the last
          // evaluation's plane. The start has none to carry.
          if (evaluation == 0) {
            return err_numerical("[{}] meci: the start's CASSCF roots are {:.3e} hartree apart, "
                                 "under mcscf::kRootGapFloor, the pair under QD states {} and {}: "
                                 "the branching plane has no limit there, and there is none to "
                                 "carry; start off the symmetric point",
                                 kName, out.root_gap, i, j);
          }
          std::ranges::copy(out.g_mean, gradient.begin());
          std::ranges::copy(out.g_mean, g_j.begin());
          std::ranges::fill(h_ij, 0.0);
        } else {
          std::ranges::copy(out.g_p, gradient.begin());
          std::ranges::copy(out.g_q, g_j.begin());
          std::ranges::copy(out.h_pq, h_ij.begin());
        }
        pair = {out.eigenvalues[static_cast<size_t>(i)], out.eigenvalues[static_cast<size_t>(j)]};
        // The state above the pair, E_{Q+1} - E_Q, Q the higher of the two.
        const auto higher = static_cast<size_t>(std::max(i, j));
        const json margin = higher + 1 < out.eigenvalues.size()
                                ? json(out.eigenvalues[higher + 1] - out.eigenvalues[higher])
                                : json(nullptr);
        record["heff_gap"] = out.heff_gap;
        record["third_state_margin"] = margin;
        record["root_gap"] = out.root_gap;
        record["gap_sensitivity_orbital"] = out.gap_sensitivity_orbital;
        record["gap_sensitivity_ci"] = out.gap_sensitivity_ci;
        record["reference_pair_tau"] =
            out.reference_pair_tau ? json(*out.reference_pair_tau) : json(nullptr);
        // Rows of three, or null for a vector inside a degenerate CASSCF pair.
        const auto rows = [](const std::vector<double> &v) {
          if (v.empty()) {
            return json(nullptr);
          }
          json r = json::array();
          for (size_t a = 0; a + 2 < v.size(); a += 3) {
            r.push_back(json::array({v[a], v[a + 1], v[a + 2]}));
          }
          return r;
        };
        qd_branching = {{"g_p", rows(out.g_p)},
                        {"g_q", rows(out.g_q)},
                        {"h_pq", rows(out.h_pq)},
                        {"g_mean", rows(out.g_mean)},
                        {"nroots", cas.energies.size()}};
        FBF_LOG_INFO(
            "optimize: QD-NEVPT2 MECI states {} and {}, pair overlap {}, H_eff gap "
            "{:.3e}, third-state margin {}, CASSCF root gap {:.3e}{}",
            i, j,
            std::isnan(overlap) ? std::string("- (the start)") : std::format("{:.8f}", overlap),
            out.heff_gap, margin.is_null() ? std::string("-") : margin.dump(), out.root_gap,
            out.reference_pair_tau
                ? std::format(", mean gradient alone (tau {:.2e}), plane carried",
                              *out.reference_pair_tau)
                : std::string());
        vectors = out.eigenvectors;
      } else {
        if (followed) {
          overlap = FBF_TRY(pair_overlap(s, i, j));
        }
        const mcscf::Intersection out =
            FBF_TRY(mcscf::df_intersection(res, molecule, basis, classes, fit->aux, h, fit->b,
                                           fit->l.data(), solution, cas.ncore, cas.ncas, i, j));
        std::ranges::copy(out.g_i, gradient.begin());
        std::ranges::copy(out.g_j, g_j.begin());
        std::ranges::copy(out.h_ij, h_ij.begin());
        pair = {cas.energies[static_cast<size_t>(i)], cas.energies[static_cast<size_t>(j)]};
        FBF_LOG_INFO("optimize: MECI roots {} and {}, pair overlap {}, Z-vector {} products, "
                     "residual {:.2e}",
                     i, j,
                     std::isnan(overlap) ? std::string("- (the start)")
                                         : std::format("{:.8f}", overlap),
                     out.iterations, out.residual);
      }
      follow_record.push_back({.overlap = overlap, .root = std::nullopt, .heff_gap = std::nullopt});
      meci_record.push_back(std::move(record));
      followed =
          FollowedState{.basis = basis, .solution = std::move(solution), .u = std::move(vectors)};
      return 0.5 * (pair[0] + pair[1]);
    };
    const double mean = FBF_TRY(casscf_chain(trial, intersection, g_i));
    FBF_LOG_INFO("optimize: evaluation {:>3} E_{} {:.10f} E_{} {:.10f} mean {:.10f} gap {:.3e}",
                 evaluation, i, pair[0], j, pair[1], mean, std::abs(pair[1] - pair[0]));
    return pair;
  };

  // Both walks write the three files through recorded_optimize(_meci): as
  // they go, and on every exit.
  std::vector<double> optimum_x;
  double optimum_energy = 0.0;
  if (meci) {
    // The pair's overlap with the last evaluation's, null at the first; the
    // CASSCF's residuals, and over QD states the pair's gap, the third-state
    // margin, the CASSCF's smallest root gap and the gap sensitivities.
    const auto annotate = [&](size_t k, json &entry) {
      const bool has_overlap = k < follow_record.size() && !std::isnan(follow_record[k].overlap);
      entry["pair_overlap"] = has_overlap ? json(follow_record[k].overlap) : json(nullptr);
      if (k < meci_record.size()) {
        entry.update(meci_record[k]);
      }
    };
    const auto annotate_head = [&](json &head) {
      if (qd_meci) {
        head.update(qd_branching);
      }
    };
    const geometry::MECISolution solution = FBF_TRY(
        recorded_optimize_meci(*this, start.atoms(), meci_evaluation, annotate, annotate_head));
    optimum_x = solution.x;
    optimum_energy = 0.5 * (solution.lower + solution.upper);
  } else {
    // Over a followed state: its matched overlap, null at the first
    // evaluation, and the root followed or H_eff's smallest gap.
    const auto annotate = [&](size_t k, json &entry) {
      if (k < follow_record.size()) {
        const FollowRecord &f = follow_record[k];
        entry["state_overlap"] = std::isnan(f.overlap) ? json(nullptr) : json(f.overlap);
        if (f.root) {
          entry["root"] = *f.root;
        }
        if (f.heff_gap) {
          entry["heff_gap"] = *f.heff_gap;
        }
      }
    };
    const geometry::Solution solution =
        FBF_TRY(recorded_optimize(*this, start.atoms(), energy_and_gradient, annotate));
    optimum_x = solution.x;
    optimum_energy = solution.energy;
  }
  return OptimizeOutput{.molecule = Molecule(res, placed(start.atoms(), optimum_x), start.charge(),
                                             start.multiplicity()),
                        .energy = optimum_energy};
}

} // namespace free_breakfast
