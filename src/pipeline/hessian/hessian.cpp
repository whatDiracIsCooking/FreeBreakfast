/// @file
/// @brief HessianStep: the analytic Hessian, hessian::df_rhf over one nested
/// Pipeline run of the chain to ScfStep, or hessian::df_casscf or
/// df_sacasscf over one to CasscfStep; or the 6N displaced gradients, each
/// a nested Pipeline run of the chain to GradientStep, and their central
/// differences -- the RHF's, or over the CASSCF chain (to CasscfGradientStep
/// or Nevpt2GradientStep), each displaced solve started from the reference's
/// orbitals and CI and checked against it; then the
/// symmetrized Hessian, its two checks, the harmonic vibrations, the analytic
/// RHF's dipole derivative and IR intensities, and hessian.json.

module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kOne, kZero, kInvSqrt2
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.io.json;
import free_breakfast.basis;
import free_breakfast.hessian;
import free_breakfast.integrals.one_electron.dipole; // kDipoleMaxL
import free_breakfast.pipeline.pipeline;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.orthogonalize;

#include "diagnostics.h"

namespace free_breakfast {

// The chain's root at one geometry: the molecule from atoms in memory, where
// MoleculeStep reads a file.
struct DisplacedMoleculeStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<Atom> atoms;
  int32_t charge{0};
  int32_t multiplicity{1};

  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, atoms, charge, multiplicity)};
  }
};

// The CASSCF chain's ScfOutput at a displaced geometry: the reference
// CASSCF's orbitals, core, active, virtual, Loewdin-orthonormalized in this
// geometry's overlap, C (C^T S C)^-1/2, the orthonormal set nearest them, so
// the displaced CASSCF starts from the reference's solution moved and not
// from a fresh pick, and a start that is not orthonormal does not converge
// O(h) off. Only the CasscfStep reads it, by index over its first columns:
// the energy is NaN, no SCF ran, and d is left zero.
struct ReferenceOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  // AO by MO, column-major, on the host: the reference CasscfOutput's c.
  std::vector<double> c;
  // One per column: the reference CasscfOutput's mo_energies.
  std::vector<double> mo_energies;
  // The reference geometry's ScfOutput's.
  int32_t n_alpha{0};
  int32_t n_beta{0};

  [[nodiscard]] Result<ScfOutput> run(View<BasisSetOutput, OverlapOutput> in,
                                      DeviceResources &res) const {
    const BasisSet &basis = in.get<BasisSetOutput>().basis;
    const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
    if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
      return err_config("[{}] carries {} orbital coefficients and {} energies into a basis of {} "
                        "functions",
                        HessianStep::kName, c.size(), mo_energies.size(), s.n());
    }
    const ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<int32_t>(s.n());
    const size_t n2 = s.num_elements();
    // C, M = C^T S C and M^-1/2: held across orthogonalize, which takes the
    // workspace, so each is its own buffer.
    SphericalMatrixAO reference(res, basis);
    Buffers::device<double> m(n2, res.device());
    Buffers::device<double> m_inv_sqrt(n2, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(reference.data(), c.data(), n2 * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    {
      const auto [work] = res.device_regions<double>(n2);
      FBF_TRY(gpu_status(gemmwich(res.blas(), n, reference.data(), s.data(), work, m.data())));
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
    FBF_TRY(gpu_status(gemm(res.blas(), n, &kOne<double>, reference.data(), m_inv_sqrt.data(),
                            &kZero<double>, out.c.data())));
    FBF_TRY(res.sync());
    return out;
  }
};

// The @p count doubles at @p d, on the host.
[[nodiscard]] Result<std::vector<double>> download(DeviceResources &res, const double *d,
                                                   size_t count) {
  std::vector<double> host(count);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return host;
}

// @p a, n x n row-major, as n rows of n.
[[nodiscard]] json hessian_rows(const std::vector<double> &a, size_t n) {
  json rows = json::array();
  for (size_t x = 0; x + n <= a.size(); x += n) {
    rows.push_back(std::vector<double>(a.begin() + static_cast<std::ptrdiff_t>(x),
                                       a.begin() + static_cast<std::ptrdiff_t>(x + n)));
  }
  return rows;
}

// @p status, its error's message prefixed with the step and @p where it was
// met: which of the 6N + 1 geometries.
[[nodiscard]] Status at_geometry(Status status, std::string_view where) {
  if (!status) {
    Error error = std::move(status).error();
    error.message = std::format("[{}] at {}: {}", HessianStep::kName, where, error.message);
    return std::unexpected(std::move(error));
  }
  return status;
}

// The ladder, #803's fixture's procedure (tools/nevpt2_hessian/stencil.py)
// over the analytic gradient. The fourth-order central first derivative,
// f'(0) ~ sum_o w_o f(o s) / s = (f(-2s) - 8 f(-s) + 8 f(s) - f(2s)) / 12s,
// its error -s⁴ f⁽⁵⁾(0) / 30: it falls 2⁴ = 16-fold per halving of s, so
// (D(h) - D(h/2)) / (D(h/2) - D(h/4)) is 16 where truncation dominates.
constexpr std::array<double, 4> kStencilOffsets{-2.0, -1.0, 1.0, 2.0};
constexpr std::array<double, 4> kStencilWeights{1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0};
constexpr double kLadderTarget = 16.0;
// The three rungs, h, h/2 and h/4.
constexpr size_t kRungs = 3;
// Every displacement the three stencils read, in units of h, ascending: the
// finer steps share points with the coarser (h/2's 2s is h's s), so eight.
constexpr std::array<double, 8> kLadderPoints{-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0};
// kLadderPoints' +h, where the noise is measured.
constexpr size_t kLadderPointH = 6;
// A judged ratio lies within 20% of 16: the fixture's window.
constexpr double kRatioLow = 0.8 * kLadderTarget;
constexpr double kRatioHigh = 1.2 * kLadderTarget;
// The noise floor is this many times the finer difference's noise, so a
// one-sigma noise moves a judged ratio by 5% and the window holds four: the
// fixture's.
constexpr double kFloorFactor = 20.0;
// A ratio is judged only where the finer difference is past this many
// floors, the tree's visibility rule: nearer the floor the ratio is noise's.
constexpr double kJudgedFloors = 10.0;
// The noise measurement: at +h along the first kNoisePoints coordinates, the
// gradient from the reference's start and from kNoiseStarts starts each
// rotated off it by up to kNoiseKick radians in every non-redundant pair, a
// fixed seed so a rerun reads the same starts. The fixture's restarts and
// kick, at points the ladder already reads.
constexpr size_t kNoisePoints = 4;
constexpr size_t kNoiseStarts = 4;
constexpr double kNoiseKick = 1e-3;
constexpr uint64_t kNoiseSeed = 812;
// The re-solve that measures what the production tolerances leave: each of
// them, the CASSCF's, its CI's and the Z-vector's, this many times tighter.
constexpr double kTightened = 100.0;
// How far from orthonormal the config's coordinates may be: rounding of
// rows written to 16 digits is ~1e-16.
constexpr double kCoordinatesOrthonormal = 1e-10;

// D(s) at s = h, h/2, h/4 over @p values, values[p] the vector read at
// h kLadderPoints[p].
[[nodiscard]] std::array<std::vector<double>, kRungs>
ladder_rungs(const std::array<std::vector<double>, kLadderPoints.size()> &values, double h) {
  const size_t m = values[0].size();
  std::array<std::vector<double>, kRungs> out;
  for (size_t k = 0; k < kRungs; ++k) {
    const double fraction = 1.0 / static_cast<double>(size_t{1} << k);
    out[k].assign(m, 0.0);
    for (size_t o = 0; o < kStencilOffsets.size(); ++o) {
      // Powers of two: the lookup is exact.
      const auto point = std::ranges::find(kLadderPoints, kStencilOffsets[o] * fraction);
      const std::vector<double> &at = values[static_cast<size_t>(point - kLadderPoints.begin())];
      for (size_t i = 0; i < m; ++i) {
        out[k][i] += kStencilWeights[o] * at[i];
      }
    }
    for (double &d : out[k]) {
      d /= h * fraction;
    }
  }
  return out;
}

// sqrt(sum w²) / s: the factor one stencil at step s multiplies independent
// noise of unit size in each gradient it reads by.
[[nodiscard]] double stencil_noise_gain(double s) {
  double sum = 0.0;
  for (const double w : kStencilWeights) {
    sum += w * w;
  }
  return std::sqrt(sum) / s;
}

// One element's ladder D(h), D(h/2), D(h/4), judged as stencil.judge does
// but for where: its ratio only when the finer difference is past
// kJudgedFloors floors.
struct LadderVerdict {
  double upper{0.0}; // D(h) - D(h/2)
  double lower{0.0}; // D(h/2) - D(h/4)
  // upper / lower; empty when lower is not past kJudgedFloors floors (it is
  // "below the floor": not judged).
  std::optional<double> ratio;
  // A judged ratio inside the window; true when not judged.
  bool settled{true};
  // D(h/4)'s error bar: |lower|, or the floor where that is below it.
  double bar{0.0};
};

[[nodiscard]] LadderVerdict judge_ladder(double coarse, double middle, double fine, double floor) {
  LadderVerdict out{.upper = coarse - middle, .lower = middle - fine};
  out.bar = std::max(std::abs(out.lower), floor);
  if (std::abs(out.lower) <= kJudgedFloors * floor) {
    return out;
  }
  out.ratio = out.upper / out.lower;
  out.settled = *out.ratio > kRatioLow && *out.ratio < kRatioHigh;
  return out;
}

// @p reference's orbitals rotated by a random angle within +-kNoiseKick in
// every non-redundant pair (core-active, core-virtual, active-virtual), to
// first order, C (1 + A) with A antisymmetric; ReferenceOrbitalsStep
// orthonormalizes it. A start the CASSCF reconverges from, so the spread of
// its gradients is the chain's noise.
[[nodiscard]] ReferenceOrbitalsStep kicked_start(const ReferenceOrbitalsStep &reference,
                                                 size_t ncore, size_t ncas, std::mt19937_64 &rng) {
  const size_t nmo = reference.mo_energies.size();
  const size_t nao = nmo == 0 ? 0 : reference.c.size() / nmo;
  std::uniform_real_distribution<double> angle(-kNoiseKick, kNoiseKick);
  const auto space = [&](size_t p) { return p < ncore ? 0 : (p < ncore + ncas ? 1 : 2); };
  ReferenceOrbitalsStep out = reference;
  for (size_t p = 0; p < nmo; ++p) {
    for (size_t r = p + 1; r < nmo; ++r) {
      if (space(p) == space(r)) {
        continue;
      }
      const double a = angle(rng);
      for (size_t mu = 0; mu < nao; ++mu) {
        out.c[(r * nao) + mu] += a * reference.c[(p * nao) + mu];
        out.c[(p * nao) + mu] -= a * reference.c[(r * nao) + mu];
      }
    }
  }
  return out;
}

// @p values, n x n row-major, as n rows of n, an empty ratio null.
[[nodiscard]] json optional_rows(const std::vector<std::optional<double>> &values, size_t n) {
  json rows = json::array();
  for (size_t i = 0; i < n; ++i) {
    json row = json::array();
    for (size_t j = 0; j < n; ++j) {
      const std::optional<double> &v = values[(i * n) + j];
      row.push_back(v ? json(*v) : json(nullptr));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

// check_displaced_casscf's measures, its arguments' sizes checked: each
// converged active orbital t's weight in the start's active space,
// sum_u <start_u|S|active_t>^2, and each root r's squared overlap with the
// reference's root r. One minus each is the sine squared of the angle the
// solve turned that orbital or root by from where it started.
[[nodiscard]] std::pair<std::vector<double>, std::vector<double>>
displaced_weights(size_t nao, std::span<const double> s, std::span<const double> start_active,
                  std::span<const double> active, std::span<const double> reference_ci,
                  std::span<const double> ci, size_t nroots) {
  const size_t ncas = active.size() / nao;
  std::vector<double> weights(ncas, 0.0);
  std::vector<double> s_active(nao);
  for (size_t t = 0; t < ncas; ++t) {
    const std::span<const double> orbital = active.subspan(t * nao, nao);
    for (size_t mu = 0; mu < nao; ++mu) {
      s_active[mu] = std::transform_reduce(orbital.begin(), orbital.end(),
                                           s.begin() + static_cast<std::ptrdiff_t>(mu * nao), 0.0);
    }
    for (size_t u = 0; u < ncas; ++u) {
      const std::span<const double> start = start_active.subspan(u * nao, nao);
      const double overlap =
          std::transform_reduce(start.begin(), start.end(), s_active.begin(), 0.0);
      weights[t] += overlap * overlap;
    }
  }
  std::vector<double> overlaps(nroots);
  const size_t dim = ci.size() / nroots;
  for (size_t r = 0; r < nroots; ++r) {
    const std::span<const double> root = ci.subspan(r * dim, dim);
    const double overlap = std::transform_reduce(
        root.begin(), root.end(), reference_ci.begin() + static_cast<std::ptrdiff_t>(r * dim), 0.0);
    overlaps[r] = overlap * overlap;
  }
  return {std::move(weights), std::move(overlaps)};
}

Status check_displaced_casscf(size_t nao, std::span<const double> s,
                              std::span<const double> start_active, std::span<const double> active,
                              std::span<const double> reference_ci, std::span<const double> ci,
                              size_t nroots) {
  // More in than out: 1 - O(h²) for a solve that followed the reference,
  // about 0 for an orbital or a root that changed places.
  constexpr double kFloor = 0.5;
  const size_t ncas = nao == 0 ? 0 : active.size() / nao;
  if (nao == 0 || s.size() != nao * nao || ncas == 0 || active.size() != nao * ncas ||
      start_active.size() != active.size() || nroots == 0 || ci.size() % nroots != 0 ||
      ci.empty() || reference_ci.size() != ci.size()) {
    return err_config("[{}] check_displaced_casscf: {} AO, an overlap of {}, active orbitals of "
                      "{} and {}, CI vectors of {} and {} over {} roots",
                      HessianStep::kName, nao, s.size(), start_active.size(), active.size(),
                      reference_ci.size(), ci.size(), nroots);
  }
  const auto [weights, overlaps] =
      displaced_weights(nao, s, start_active, active, reference_ci, ci, nroots);
  for (size_t t = 0; t < weights.size(); ++t) {
    if (!(weights[t] > kFloor)) {
      return err_numerical("[{}] active orbital {} of {} left the reference's active space: its "
                           "weight there is {:.3e}, not above {}",
                           HessianStep::kName, t, weights.size(), weights[t], kFloor);
    }
  }
  for (size_t r = 0; r < nroots; ++r) {
    if (!(overlaps[r] > kFloor)) {
      return err_numerical("[{}] root {} of {} is not the reference's root {}: their overlap "
                           "squared is {:.3e}, not above {} (the roots changed order, or one "
                           "from outside the average came in)",
                           HessianStep::kName, r, nroots, r, overlaps[r], kFloor);
    }
  }
  return {};
}

Result<HessianOutput> HessianStep::run(View<MoleculeOutput> in, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  if (casscf_gradient && nevpt2_gradient) {
    return err_config("[{}] is the Hessian of one energy, and is given both [{}] and [{}]: keep "
                      "the one wanted",
                      kName, CasscfGradientStep::kName, Nevpt2GradientStep::kName);
  }
  if ((casscf_gradient || nevpt2_gradient) && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] reruns [{}], which runs over density fitting only, and [{}] mode "
                           "is 'integral_direct': set [{}] mode = 'density_fitting'",
                           kName,
                           casscf_gradient ? CasscfGradientStep::kName : Nevpt2GradientStep::kName,
                           EriStep::kName, EriStep::kName);
  }
  // The NEVPT2 Hessian is numerical, of one root's state-specific energy,
  // all-electron: what milestone 24 covers and the fixture validates.
  if (nevpt2_gradient) {
    constexpr std::string_view kNevpt2 = Nevpt2GradientStep::kName;
    if (method == HessianMethod::Analytic) {
      return err_unsupported("[{}] over [{}] is numerical only: set [{}] method = 'numerical'",
                             kName, kNevpt2, kName);
    }
    if (nevpt2_gradient->quasi_degenerate) {
      return err_unsupported("[{}] differences a state-specific NEVPT2 gradient, and [{}] is "
                             "quasi_degenerate: QD-NEVPT2 has no Hessian here",
                             kName, kNevpt2);
    }
    if (casscf.nroots != 1) {
      return err_unsupported("[{}] differences the NEVPT2 gradient of a single-state CASSCF, and "
                             "[{}] nroots is {}: a state-averaged reference has no NEVPT2 Hessian "
                             "here",
                             kName, CasscfStep::kName, casscf.nroots);
    }
    if (nevpt2_gradient->frozen_core != 0) {
      return err_config("[{}] differences the all-electron NEVPT2 gradient, and [{}] frozen_core "
                        "is {}: set it to 0",
                        kName, kNevpt2,
                        // Below 0 is only "auto", as the config reads it.
                        nevpt2_gradient->frozen_core < 0
                            ? std::string{"'auto'"}
                            : std::to_string(nevpt2_gradient->frozen_core));
    }
  }
  if (method == HessianMethod::Analytic && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] method 'analytic' is density-fitted only, and [{}] mode is "
                           "'integral_direct': set [{}] mode = 'density_fitting', or [{}] method "
                           "= 'numerical'",
                           kName, EriStep::kName, EriStep::kName, kName);
  }
  const bool any_pairs = !pairs || !pairs->empty();
  if (method == HessianMethod::Analytic && (ladder || any_pairs || !coordinates.empty())) {
    return err_config("[{}] ladder, pairs and coordinates are the numerical method's, and method "
                      "is 'analytic'",
                      kName);
  }
  if (ladder && !casscf_gradient && !nevpt2_gradient) {
    return err_unsupported("[{}] ladder measures the gradient's noise by restarting the CASSCF "
                           "from rotated orbitals, and the RHF chain has no such start: the "
                           "ladder needs [{}] or [{}]",
                           kName, CasscfGradientStep::kName, Nevpt2GradientStep::kName);
  }
  if (any_pairs && !ladder) {
    return err_config("[{}] pairs are judged against the ladder's error bars: set ladder = true",
                      kName);
  }
  DeviceResources &res = node.device(0);
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const std::vector<Atom> &atoms = molecule.atoms();
  const size_t n = 3 * atoms.size();
  // Q, row j coordinate j: the identity unless the config gave others,
  // which must be n orthonormal rows.
  std::vector<double> q = coordinates;
  if (q.empty()) {
    q.assign(n * n, 0.0);
    for (size_t x = 0; x < n; ++x) {
      q[(x * n) + x] = 1.0;
    }
  } else if (q.size() != n * n) {
    return err_config("[{}] coordinates hold {} numbers, and the molecule's {} atoms need {} "
                      "rows of {}",
                      kName, q.size(), atoms.size(), n, n);
  } else {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        const double dot =
            std::transform_reduce(q.begin() + static_cast<std::ptrdiff_t>(i * n),
                                  q.begin() + static_cast<std::ptrdiff_t>((i + 1) * n),
                                  q.begin() + static_cast<std::ptrdiff_t>(j * n), 0.0);
        if (!(std::abs(dot - (i == j ? 1.0 : 0.0)) <= kCoordinatesOrthonormal)) {
          return err_config("[{}] coordinates are not orthonormal: rows {} and {} have the dot "
                            "product {:.3e}",
                            kName, i, j, dot);
        }
      }
    }
  }
  // The pairs the bilinearity check is run on, i < j.
  std::vector<std::array<size_t, 2>> checked;
  if (!pairs && n <= kMaxEveryPairCoordinates) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        checked.push_back({i, j});
      }
    }
  } else if (!pairs) {
    FBF_LOG_WARN("hessian: no bilinearity check: every pair of {} coordinates is past the {} "
                 "checked by default; name the pairs in [{}] pairs",
                 n, kMaxEveryPairCoordinates, kName);
  } else {
    for (const auto &[a, b] : *pairs) {
      if (a < 0 || b < 0 || a == b || std::cmp_greater_equal(a, n) ||
          std::cmp_greater_equal(b, n)) {
        return err_config("[{}] pair [{}, {}] is not two distinct coordinates of {}", kName, a, b,
                          n);
      }
      const std::array<size_t, 2> pair{static_cast<size_t>(std::min(a, b)),
                                       static_cast<size_t>(std::max(a, b))};
      if (std::ranges::contains(checked, pair)) {
        return err_config("[{}] pair [{}, {}] is named twice", kName, a, b);
      }
      checked.push_back(pair);
    }
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
  // The molecule at @p atoms moved by @p shift, n Cartesian components
  // (3 A + k); empty, not moved.
  const auto displaced = [&](std::span<const double> shift) {
    DisplacedMoleculeStep out{atoms, molecule.charge(), molecule.multiplicity()};
    for (size_t a = 0; a < out.atoms.size() && !shift.empty(); ++a) {
      Vec<double, 3> &position = out.atoms[a].position;
      position.x += shift[3 * a];
      position.y += shift[(3 * a) + 1];
      position.z += shift[(3 * a) + 2];
    }
    return out;
  };
  // The Pipeline of the chain to [scf], then @p last, at displaced(shift).
  const auto chain_at = [&]<class... Last>(std::span<const double> shift, Last... last) {
    return Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                    PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                    CoreHamiltonianStep, GuessStep, ScfStep, Last...>::create({
        displaced(shift),
        basis_set,
        SphericalTransformStep{},
        PrimitivePairClassesStep{},
        eri,
        OverlapStep{},
        orthogonalizer,
        CoreHamiltonianStep{},
        guess,
        chain_scf,
        std::move(last)...,
    });
  };

  // H before symmetrizing, row-major (the analytic terms' sum is symmetric
  // to rounding, so either order), and its asymmetry.
  std::vector<double> raw(n * n);
  double asymmetry = 0.0;
  std::vector<double> dipole_derivative;
  // Under the ladder: its columns, ratios and bars for hessian.json, each
  // checked pair's verdict, and the pairs that are not bilinear.
  json ladder_doc;
  json bilinearity;
  std::vector<std::array<size_t, 2>> failing;
  if (method == HessianMethod::Analytic && casscf_gradient) {
    // One chain to [casscf] at R, and the root's analytic Hessian at its
    // solution: hessian::df_casscf for a single root, hessian::df_sacasscf
    // for one root of a state average.
    auto chain = FBF_TRY(chain_at({}, chain_casscf));
    FBF_TRY(chain.run(node));
    const CasscfOutput &cas = *chain.get<CasscfOutput>();
    const int32_t root = casscf_gradient->root;
    if (root < 0 || std::cmp_greater_equal(root, cas.energies.size())) {
      return err_config("[{}] root is {}, and the CASSCF has {} root(s)", CasscfGradientStep::kName,
                        root, cas.energies.size());
    }
    const auto n2 = static_cast<size_t>(cas.ncas) * static_cast<size_t>(cas.ncas);
    if (cas.rdm1.size() != cas.weights.size() * n2) {
      return err_config("[{}] the CASSCF holds {} 1-RDM elements, not {} roots of {}", kName,
                        cas.rdm1.size(), cas.weights.size(), n2);
    }
    const BasisSet &basis = chain.get<BasisSetOutput>()->basis;
    const DensityFitted &fit = std::get<DensityFitted>(chain.get<EriOutput>()->eri);
    // The product back into the mcscf::Solution the Hessians take, as
    // [casscf_gradient] builds it for df_gradient: the orbitals and the CI
    // vectors on the device, gamma the weighted average of the roots'.
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
        .weights = cas.weights,
    };
    {
      const ScopedDeviceIndex on{res.dev_idx()};
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                             cas.vectors.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, res.stream())));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                             cas.c.num_elements() * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      FBF_TRY(res.sync());
    }
    const Molecule &chain_molecule = chain.get<MoleculeOutput>()->molecule;
    const auto &classes = chain.get<PrimitivePairClassesOutput>()->classes;
    const SphericalMatrixAO &h = chain.get<CoreHamiltonianOutput>()->h;
    std::vector<double> residuals;
    if (cas.weights.size() == 1) {
      hessian::CasscfTerms terms =
          FBF_TRY(hessian::df_casscf(res, chain_molecule, basis, classes, fit.aux, h, fit.b,
                                     fit.l.data(), solution, cas.ncore, cas.ncas));
      raw = std::move(terms.hessian);
      asymmetry = terms.asymmetry;
      residuals = std::move(terms.residuals);
    } else {
      hessian::SaCasscfTerms terms = FBF_TRY(
          hessian::df_sacasscf(res, chain_molecule, basis, classes, fit.aux, h, fit.b, fit.l.data(),
                               solution, cas.weights, root, cas.ncore, cas.ncas));
      raw = std::move(terms.hessian);
      asymmetry = terms.asymmetry;
      residuals = std::move(terms.residuals);
    }
    FBF_LOG_INFO("hessian: analytic DF-CASSCF({}, {}) root {} of {}, E = {:.10f}, CP residual "
                 "{:.3e}",
                 cas.nelecas, cas.ncas, root, cas.energies.size(),
                 cas.energies[static_cast<size_t>(root)],
                 residuals.empty() ? 0.0 : std::ranges::max(residuals));
  } else if (method == HessianMethod::Analytic) {
    auto chain = FBF_TRY(chain_at({}));
    FBF_TRY(chain.run(node));
    const ScfOutput &rhf = *chain.get<ScfOutput>();
    if (rhf.n_alpha != rhf.n_beta) {
      return err_unsupported("[{}] is the closed-shell RHF Hessian only, and the SCF is ROHF ({} "
                             "alpha, {} beta electrons)",
                             kName, rhf.n_alpha, rhf.n_beta);
    }
    const BasisSet &basis = chain.get<BasisSetOutput>()->basis;
    const DensityFitted &fit = std::get<DensityFitted>(chain.get<EriOutput>()->eri);
    // hessian::df_rhf takes C as a buffer of its nao x nmo columns.
    const size_t nmo_elements = static_cast<size_t>(rhf.c.n()) * rhf.mo_energies.size();
    Buffers::device<double> c(nmo_elements, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data(), rhf.c.data(), nmo_elements * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    // Every term, and the CPHF's U^x, which the dipole derivative reuses.
    const hessian::RhfTerms terms =
        FBF_TRY(hessian::df_rhf(res, chain.get<MoleculeOutput>()->molecule, basis,
                                chain.get<PrimitivePairClassesOutput>()->classes, fit.aux, fit.b,
                                fit.l.data(), c, rhf.mo_energies, rhf.n_alpha));
    FBF_LOG_INFO("hessian: analytic DF-RHF, CPHF in {} sweeps", terms.response.cphf.iterations);
    raw = terms.hessian;
    asymmetry = terms.asymmetry;
    // The dipole derivative from the same U^x, where the dipole integrals reach.
    int32_t max_l = 0;
    for (const Shell &shell : basis.shells()) {
      max_l = std::max(max_l, shell.angular_momentum());
    }
    if (max_l <= kDipoleMaxL) {
      dipole_derivative =
          FBF_TRY(hessian::rhf_dipole_derivative(res, chain.get<MoleculeOutput>()->molecule, basis,
                                                 chain.get<PrimitivePairClassesOutput>()->classes,
                                                 c, rhf.n_alpha, terms.response.cphf.solution));
    } else {
      FBF_LOG_WARN("hessian: no IR intensities: the basis has a shell of l = {}, and the dipole "
                   "integrals stop at l = {}",
                   max_l, kDipoleMaxL);
    }
  } else {
    // Q g: the gradient @p g in the coordinates (exact under the identity).
    const auto in_coordinates = [&](const std::vector<double> &g) {
      std::vector<double> out(n, 0.0);
      for (size_t i = 0; i < n; ++i) {
        for (size_t x = 0; x < n; ++x) {
          out[i] += q[(i * n) + x] * g[x];
        }
      }
      return out;
    };
    // t u, u a Cartesian direction of n: the displacement R moves by.
    const auto scaled = [&](std::span<const double> u, double t) {
      std::vector<double> out(n);
      for (size_t x = 0; x < n; ++x) {
        out[x] = t * u[x];
      }
      return out;
    };
    const auto coordinate = [&](size_t j) { return std::span<const double>(q).subspan(j * n, n); };
    // dE/dR at the geometry moved by @p shift: the RHF's.
    const auto rhf_gradient_at = [&](std::span<const double> shift) -> Result<std::vector<double>> {
      auto chain = FBF_TRY(chain_at(shift, GradientStep{}));
      FBF_TRY(chain.run(node));
      return chain.get<GradientOutput>()->gradient.total;
    };
    // Under the CASSCF chain, the reference at R: the SCF, the configured
    // pick and its CASSCF, whose orbitals every displaced solve starts from
    // and whose roots each is held to. The active space is picked here only.
    std::optional<ReferenceOrbitalsStep> reference;
    std::vector<double> reference_ci;
    size_t reference_ncore = 0;
    CasscfStep displaced_casscf = chain_casscf;
    if (casscf_gradient || nevpt2_gradient) {
      auto chain = FBF_TRY(chain_at({}, chain_casscf));
      FBF_TRY(at_geometry(chain.run(node), "the reference geometry"));
      const ScfOutput &rhf = *chain.get<ScfOutput>();
      const CasscfOutput &cas = *chain.get<CasscfOutput>();
      reference.emplace(ReferenceOrbitalsStep{
          .c = FBF_TRY(download(res, cas.c.data(), cas.c.num_elements())),
          .mo_energies = cas.mo_energies,
          .n_alpha = rhf.n_alpha,
          .n_beta = rhf.n_beta,
      });
      reference_ci = cas.vectors;
      reference_ncore = static_cast<size_t>(cas.ncore);
      // By index, as they come: core, active, virtual; and the CI from the
      // reference's roots, over those active orbitals moved.
      displaced_casscf.active.clear();
      displaced_casscf.avas.clear();
      displaced_casscf.ncas = cas.ncas;
      displaced_casscf.nelecas = cas.nelecas;
      displaced_casscf.ci_guess = cas.vectors;
      FBF_LOG_INFO("hessian: reference DF-CASSCF({}, {}) over {} root(s), E = {:.10f}", cas.nelecas,
                   cas.ncas, cas.energies.size(), cas.energy);
    }
    // The NEVPT2 gradient's hazards, worst over the displacements: the
    // largest |z^c| and Z-vector residual, the smallest inactive gap.
    double worst_zc = 0.0;
    double worst_gap = std::numeric_limits<double>::infinity();
    double worst_residual = 0.0;
    size_t gradients = 0;
    // The solves' own noise, for the ladder's floor: the largest residual a
    // displaced solve left, the CASSCF's ||g|| and CI residual plus the
    // NEVPT2 Z-vector's, and the largest response of the parameters per
    // bohr, the angle a solve from the reference's start turned its active
    // orbitals and root by over the length of its displacement.
    double worst_solve_residual = 0.0;
    double worst_response = 0.0;
    // ... and the gradient of the CASSCF chain @p casscf_step then @p last
    // (CasscfGradientStep or Nevpt2GradientStep), its solve started from
    // @p start (the reference's orbitals, a restart's, or a solve's own) and
    // the step's CI guess, refused unless it converged and kept the
    // reference's active space and root order. With @p solved, the solve's
    // orbitals and CI vectors are left there, for a re-solve to start from.
    const auto casscf_chain_gradient_at =
        [&]<class Last>(std::span<const double> shift, const std::string &where,
                        const ReferenceOrbitalsStep &start, const CasscfStep &casscf_step,
                        const Last &last,
                        std::pair<ReferenceOrbitalsStep, std::vector<double>> *solved =
                            nullptr) -> Result<std::vector<double>> {
      // The production solves' diagnostics, not a tightened re-solve's.
      const bool production = casscf_step.conv_tol_grad == displaced_casscf.conv_tol_grad;
      auto chain =
          FBF_TRY((Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, CoreHamiltonianStep,
                            ReferenceOrbitalsStep, CasscfStep, Last>::create({
              displaced(shift),
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              CoreHamiltonianStep{},
              start,
              casscf_step,
              last,
          })));
      FBF_TRY(at_geometry(chain.run(node), where));
      const CasscfOutput &cas = *chain.template get<CasscfOutput>();
      FBF_LOG_DEBUG("hessian: {}: CASSCF in {} macro iterations", where, cas.iterations.size());
      const size_t nao = cas.c.n();
      const size_t first = static_cast<size_t>(cas.ncore) * nao;
      const size_t count = static_cast<size_t>(cas.ncas) * nao;
      const std::vector<double> s =
          FBF_TRY(download(res, chain.template get<OverlapOutput>()->s.data(), nao * nao));
      const std::vector<double> start_active =
          FBF_TRY(download(res, chain.template get<ScfOutput>()->c.data() + first, count));
      const std::vector<double> active = FBF_TRY(download(res, cas.c.data() + first, count));
      FBF_TRY(at_geometry(check_displaced_casscf(nao, s, start_active, active, reference_ci,
                                                 cas.vectors, cas.energies.size()),
                          where));
      const double length =
          std::sqrt(std::transform_reduce(shift.begin(), shift.end(), shift.begin(), 0.0));
      if (&start == &*reference && length > 0.0) {
        const auto [weights, overlaps] = displaced_weights(
            nao, s, start_active, active, reference_ci, cas.vectors, cas.energies.size());
        double turned = 0.0;
        for (const double w : weights) {
          turned += std::max(0.0, 1.0 - w);
        }
        for (const double o : overlaps) {
          turned += std::max(0.0, 1.0 - o);
        }
        worst_response = std::max(worst_response, std::sqrt(turned) / length);
      }
      if (production) {
        worst_solve_residual =
            std::max(worst_solve_residual, cas.gradient_norm + cas.ci_residual_norm);
      }
      if (solved != nullptr) {
        solved->first = ReferenceOrbitalsStep{
            .c = FBF_TRY(download(res, cas.c.data(), cas.c.num_elements())),
            .mo_energies = cas.mo_energies,
            .n_alpha = start.n_alpha,
            .n_beta = start.n_beta,
        };
        solved->second = cas.vectors;
      }
      if constexpr (std::same_as<Last, CasscfGradientStep>) {
        return chain.template get<CasscfGradientOutput>()->gradient.total;
      } else {
        // run refused QD before anything ran: SC or PC.
        return std::visit(
            [&]<class G>(const G &g) -> Result<std::vector<double>> {
              if constexpr (std::same_as<G, nevpt2::ScGradient> ||
                            std::same_as<G, nevpt2::PcGradient>) {
                worst_zc = std::max(worst_zc, g.zc.max_z);
                worst_gap = std::min(worst_gap, g.zc.min_gap);
                worst_residual = std::max(worst_residual, g.multipliers.residual);
                if (production) {
                  worst_solve_residual =
                      std::max(worst_solve_residual,
                               cas.gradient_norm + cas.ci_residual_norm + g.multipliers.residual);
                }
                FBF_LOG_INFO("hessian: {}: {}-NEVPT2 E = {:.10f}, max |z^c| {:.3e}, smallest "
                             "inactive gap {:.3e}, Z-vector residual {:.3e} in {} iterations",
                             where, std::same_as<G, nevpt2::PcGradient> ? "PC" : "SC",
                             cas.energies[0] + g.energies.total, g.zc.max_z, g.zc.min_gap,
                             g.multipliers.residual, g.multipliers.iterations);
                return g.terms.total;
              } else {
                return err_unsupported("[{}] differences a state-specific NEVPT2 gradient, and "
                                       "got a QD-NEVPT2 one",
                                       kName);
              }
            },
            chain.template get<Nevpt2GradientOutput>()->gradient);
      }
    };
    // The gradient at R + @p shift, from @p start under the CASSCF chain
    // (the reference's orbitals when null).
    const auto gradient_at = [&](std::span<const double> shift, const std::string &where,
                                 const ReferenceOrbitalsStep *start =
                                     nullptr) -> Result<std::vector<double>> {
      ++gradients;
      if (casscf_gradient) {
        return casscf_chain_gradient_at(shift, where, start ? *start : *reference, displaced_casscf,
                                        chain_casscf_gradient);
      }
      if (nevpt2_gradient) {
        return casscf_chain_gradient_at(shift, where, start ? *start : *reference, displaced_casscf,
                                        chain_nevpt2_gradient);
      }
      return rhf_gradient_at(shift);
    };
    // The production gradient at R + @p shift from the reference's start,
    // and the same solve tightened kTightened-fold, the CASSCF's conv_tol
    // and conv_tol_grad (its CI's floor follows conv_tol_grad,
    // mcscf::ci_tolerance_for) and the NEVPT2 Z-vector's, started from the
    // production solve's orbitals and CI: their difference is what the
    // production tolerances leave in the gradient.
    const auto production_and_tightened = [&](std::span<const double> shift,
                                              const std::string &where)
        -> Result<std::pair<std::vector<double>, std::vector<double>>> {
      gradients += 2;
      std::pair<ReferenceOrbitalsStep, std::vector<double>> solved;
      CasscfStep tight = displaced_casscf;
      tight.conv_tol /= kTightened;
      tight.conv_tol_grad /= kTightened;
      const std::string tightened = std::format("{}, tightened {:g}-fold", where, kTightened);
      if (casscf_gradient) {
        std::vector<double> g = FBF_TRY(casscf_chain_gradient_at(
            shift, where, *reference, displaced_casscf, chain_casscf_gradient, &solved));
        tight.ci_guess = std::move(solved.second);
        return std::pair{std::move(g),
                         FBF_TRY(casscf_chain_gradient_at(shift, tightened, solved.first, tight,
                                                          chain_casscf_gradient))};
      }
      std::vector<double> g = FBF_TRY(casscf_chain_gradient_at(
          shift, where, *reference, displaced_casscf, chain_nevpt2_gradient, &solved));
      tight.ci_guess = std::move(solved.second);
      Nevpt2GradientStep tight_nevpt2 = chain_nevpt2_gradient;
      tight_nevpt2.zvector_tolerance /= kTightened;
      return std::pair{std::move(g), FBF_TRY(casscf_chain_gradient_at(
                                         shift, tightened, solved.first, tight, tight_nevpt2))};
    };
    if (!ladder) {
      // Row j the central difference along coordinate j, in the
      // coordinates.
      for (size_t j = 0; j < n; ++j) {
        const std::vector<double> plus = in_coordinates(FBF_TRY(
            gradient_at(scaled(coordinate(j), displacement),
                        std::format("coordinate {} displaced by {:+g} bohr", j, displacement))));
        const std::vector<double> minus = in_coordinates(FBF_TRY(
            gradient_at(scaled(coordinate(j), -displacement),
                        std::format("coordinate {} displaced by {:+g} bohr", j, -displacement))));
        for (size_t i = 0; i < n; ++i) {
          raw[(j * n) + i] = (plus[i] - minus[i]) / (2.0 * displacement);
        }
        FBF_LOG_INFO("hessian: coordinate {:>3} of {}", j + 1, n);
      }
    } else {
      const double h = displacement;
      // points[j][p]: Q g at R + h kLadderPoints[p] q_j.
      std::vector<std::array<std::vector<double>, kLadderPoints.size()>> points(n);
      // At +h along the first noise_points coordinates, the gradient
      // re-solved tightened, in the coordinates.
      const size_t noise_points = std::min(kNoisePoints, n);
      std::vector<std::vector<double>> tightened(noise_points);
      for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < kLadderPoints.size(); ++p) {
          const double t = h * kLadderPoints[p];
          const std::string where = std::format("coordinate {} displaced by {:+g} bohr", j, t);
          if (j < noise_points && p == kLadderPointH) {
            auto [g, tight] = FBF_TRY(production_and_tightened(scaled(coordinate(j), t), where));
            points[j][p] = in_coordinates(g);
            tightened[j] = in_coordinates(tight);
          } else {
            points[j][p] = in_coordinates(FBF_TRY(gradient_at(scaled(coordinate(j), t), where)));
          }
        }
        FBF_LOG_INFO("hessian: ladder of coordinate {:>3} of {}", j + 1, n);
      }
      // rungs[k][(i n) + j]: q_i . D(h / 2^k) along q_j, column j the
      // derivative along coordinate j.
      std::array<std::vector<double>, kRungs> rungs;
      for (std::vector<double> &rung : rungs) {
        rung.assign(n * n, 0.0);
      }
      for (size_t j = 0; j < n; ++j) {
        const std::array<std::vector<double>, kRungs> column = ladder_rungs(points[j], h);
        for (size_t k = 0; k < kRungs; ++k) {
          for (size_t i = 0; i < n; ++i) {
            rungs[k][(i * n) + j] = column[k][i];
          }
        }
      }
      // sigma_g, measured: at +h along each of the first kNoisePoints
      // coordinates, the sample variance of each gradient component over
      // the reference's start and kNoiseStarts restarts, its mean over the
      // components and points.
      std::mt19937_64 rng(kNoiseSeed);
      double variance = 0.0;
      double truncation = 0.0;
      for (size_t j = 0; j < noise_points; ++j) {
        std::vector<std::vector<double>> samples{points[j][kLadderPointH]};
        for (size_t r = 0; r < kNoiseStarts; ++r) {
          const ReferenceOrbitalsStep start = kicked_start(
              *reference, reference_ncore, static_cast<size_t>(displaced_casscf.ncas), rng);
          samples.push_back(in_coordinates(FBF_TRY(gradient_at(
              scaled(coordinate(j), h),
              std::format("coordinate {} displaced by {:+g} bohr, restart {}", j, h, r + 1),
              &start))));
        }
        for (size_t i = 0; i < n; ++i) {
          double mean = 0.0;
          for (const std::vector<double> &g : samples) {
            mean += g[i];
          }
          mean /= static_cast<double>(samples.size());
          double sum = 0.0;
          for (const std::vector<double> &g : samples) {
            sum += (g[i] - mean) * (g[i] - mean);
          }
          variance += sum / static_cast<double>(samples.size() - 1);
        }
      }
      const double sigma_restarts = std::sqrt(variance / static_cast<double>(noise_points * n));
      // What the production tolerances leave in each gradient, which
      // restarts that reconverge to one point do not see, measured: the root
      // mean square over the same points and the components of the
      // production gradient less the tightened re-solve's. The root mean
      // square, as sigma_restarts: the floor's model is an independent error
      // of size sigma in each value a stencil reads.
      for (size_t j = 0; j < noise_points; ++j) {
        for (size_t i = 0; i < n; ++i) {
          const double d = points[j][kLadderPointH][i] - tightened[j][i];
          truncation += d * d;
        }
      }
      const double sigma_solves = std::sqrt(truncation / static_cast<double>(noise_points * n));
      // Its upper bound, a diagnostic only: L stationary in the parameters x
      // (and the multipliers z), a residual r there moves the gradient by
      // r . dx/dR to first order (dz/dR is dx/dR for the Z-vector), so by at
      // most ||r|| ||dx/dR||. Loose by Cauchy-Schwarz: as the floor it hid
      // every ratio and a failing pair (README.md).
      const double solve_bound = worst_solve_residual * worst_response;
      if (!(sigma_solves <= solve_bound)) {
        FBF_LOG_WARN("hessian: the measured solve noise {:.2e} is above its bound {:.2e}",
                     sigma_solves, solve_bound);
      }
      const double sigma = std::max(sigma_restarts, sigma_solves);
      // The finer difference D(h/2) - D(h/4) carries at most sigma (gain(h/2)
      // + gain(h/4)) of it; the floor is kFloorFactor times that.
      const double floor =
          kFloorFactor * sigma * (stencil_noise_gain(h / 2.0) + stencil_noise_gain(h / 4.0));
      // Each element's verdict, and the unsettled ones.
      std::vector<LadderVerdict> verdicts(n * n);
      std::vector<std::array<size_t, 2>> unsettled;
      size_t judged = 0;
      double ratio_low = std::numeric_limits<double>::infinity();
      double ratio_high = -std::numeric_limits<double>::infinity();
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          const size_t ij = (i * n) + j;
          verdicts[ij] = judge_ladder(rungs[0][ij], rungs[1][ij], rungs[2][ij], floor);
          if (!verdicts[ij].settled) {
            unsettled.push_back({i, j});
          } else if (verdicts[ij].ratio) {
            ++judged;
            ratio_low = std::min(ratio_low, *verdicts[ij].ratio);
            ratio_high = std::max(ratio_high, *verdicts[ij].ratio);
          }
        }
      }
      raw = rungs[kRungs - 1];
      // Eq. (bilinear) on each pair: the ladder of the directional
      // derivative u . g along u = (q_a + q_b) / sqrt 2, against
      // (H_aa + H_bb + H_ab + H_ba) / 2 of the finest rung's own
      // (unsymmetrized) elements; a residual above u's bar plus half the
      // four elements' fails.
      bilinearity = json::array();
      for (const auto &[a, b] : checked) {
        std::vector<double> u(n);
        for (size_t x = 0; x < n; ++x) {
          u[x] = (q[(a * n) + x] + q[(b * n) + x]) * kInvSqrt2<double>;
        }
        std::array<std::vector<double>, kLadderPoints.size()> along;
        for (size_t p = 0; p < kLadderPoints.size(); ++p) {
          const double t = h * kLadderPoints[p];
          const std::vector<double> g = FBF_TRY(gradient_at(
              scaled(u, t), std::format("pair ({}, {}) displaced by {:+g} bohr", a, b, t)));
          along[p] = {std::transform_reduce(u.begin(), u.end(), g.begin(), 0.0)};
        }
        const std::array<std::vector<double>, kRungs> uu = ladder_rungs(along, h);
        const LadderVerdict verdict = judge_ladder(uu[0][0], uu[1][0], uu[2][0], floor);
        const size_t aa = (a * n) + a;
        const size_t bb = (b * n) + b;
        const size_t ab = (a * n) + b;
        const size_t ba = (b * n) + a;
        const double predicted = 0.5 * (raw[aa] + raw[bb] + raw[ab] + raw[ba]);
        const double allowed =
            verdict.bar +
            (0.5 * (verdicts[aa].bar + verdicts[bb].bar + verdicts[ab].bar + verdicts[ba].bar));
        const double residual = uu[2][0] - predicted;
        const bool fails = !(std::abs(residual) <= allowed);
        if (fails) {
          failing.push_back({a, b});
        }
        if (!verdict.settled) {
          FBF_LOG_WARN("hessian: the ladder along pair ({}, {}) is not settled", a, b);
        }
        FBF_LOG_INFO("hessian: pair ({}, {}): residual {:+.3e} against {:.3e}{}{}", a, b, residual,
                     allowed, verdict.ratio ? std::format(", ratio {:.3f}", *verdict.ratio) : "",
                     fails ? ": not bilinear" : "");
        bilinearity.push_back({
            {"pair", {a, b}},
            {"ladder", {uu[0][0], uu[1][0], uu[2][0]}},
            {"ratio", verdict.ratio ? json(*verdict.ratio) : json(nullptr)},
            {"settled", verdict.settled},
            {"directional", uu[2][0]},
            {"predicted", predicted},
            {"residual", residual},
            {"allowed", allowed},
            {"fails", fails},
        });
      }
      FBF_LOG_INFO(
          "hessian: ladder h = {}, {}, {}: sigma_g {:.2e} (restarts {:.2e}; solves "
          "{:.2e}, at most residual {:.2e} x response {:.2e} / bohr), floor {:.2e}; {} of {} "
          "ratios judged (finer difference past {:.2e}), in [{:.3f}, {:.3f}]; {} "
          "unsettled ladder(s); {} of {} pair(s) not bilinear",
          h, h / 2.0, h / 4.0, sigma, sigma_restarts, sigma_solves, worst_solve_residual,
          worst_response, floor, judged, n * n, kJudgedFloors * floor, ratio_low, ratio_high,
          unsettled.size(), failing.size(), checked.size());
      for (const auto &[i, j] : unsettled) {
        FBF_LOG_WARN("hessian: the ladder at ({}, {}) is not settled", i, j);
      }
      std::vector<std::optional<double>> ratios(n * n);
      std::vector<double> bars(n * n);
      for (size_t ij = 0; ij < n * n; ++ij) {
        ratios[ij] = verdicts[ij].ratio;
        bars[ij] = verdicts[ij].bar;
      }
      ladder_doc = {
          {"steps", {h, h / 2.0, h / 4.0}},
          {"rungs",
           {hessian_rows(rungs[0], n), hessian_rows(rungs[1], n), hessian_rows(rungs[2], n)}},
          {"ratios", optional_rows(ratios, n)},
          {"bar", hessian_rows(bars, n)},
          {"unsettled", unsettled},
          {"noise",
           {{"sigma_g", sigma},
            {"sigma_restarts", sigma_restarts},
            {"sigma_solves", sigma_solves},
            {"from", sigma_solves > sigma_restarts ? "solves" : "restarts"},
            {"solve_bound", solve_bound},
            {"solve_residual", worst_solve_residual},
            {"response", worst_response},
            {"tightened", kTightened},
            {"floor", floor},
            {"judged_above", kJudgedFloors * floor},
            {"points", noise_points},
            {"starts", kNoiseStarts},
            {"kick", kNoiseKick}}},
      };
    }
    for (size_t x = 0; x < n; ++x) {
      for (size_t y = 0; y < n; ++y) {
        asymmetry = std::max(asymmetry, std::abs(raw[(x * n) + y] - raw[(y * n) + x]));
      }
    }
    // Back to Cartesians, Q^T H Q.
    if (!coordinates.empty()) {
      std::vector<double> hq(n * n, 0.0);
      for (size_t a = 0; a < n; ++a) {
        for (size_t b = 0; b < n; ++b) {
          for (size_t y = 0; y < n; ++y) {
            hq[(a * n) + y] += raw[(a * n) + b] * q[(b * n) + y];
          }
        }
      }
      std::ranges::fill(raw, 0.0);
      for (size_t a = 0; a < n; ++a) {
        for (size_t x = 0; x < n; ++x) {
          for (size_t y = 0; y < n; ++y) {
            raw[(x * n) + y] += q[(a * n) + x] * hq[(a * n) + y];
          }
        }
      }
    }
    if (nevpt2_gradient) {
      FBF_LOG_INFO("hessian: over the {} displaced NEVPT2 gradients, max |z^c| {:.3e}, smallest "
                   "inactive gap {:.3e}, largest Z-vector residual {:.3e}",
                   gradients, worst_zc, worst_gap, worst_residual);
    }
  }

  HessianOutput out{.hessian = std::vector<double>(n * n),
                    .asymmetry = asymmetry,
                    .dipole_derivative = std::move(dipole_derivative)};
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      out.hessian[(x * n) + y] = 0.5 * (raw[(x * n) + y] + raw[(y * n) + x]);
    }
  }
  for (size_t x = 0; x < n; ++x) {
    for (size_t l = 0; l < 3; ++l) {
      double sum = 0.0;
      for (size_t b = l; b < n; b += 3) {
        sum += out.hessian[(x * n) + b];
      }
      out.translational_sum = std::max(out.translational_sum, std::abs(sum));
    }
  }
  FBF_LOG_INFO("hessian: asymmetry {:.3e}, translational sum {:.3e}", out.asymmetry,
               out.translational_sum);

  // A pair that is not bilinear means the Hessian there is no bilinear form,
  // so it has no normal modes: none are computed or published.
  if (failing.empty()) {
    out.vibrations = FBF_TRY(geometry::harmonic_analysis(res, atoms, out.hessian));
  }
  if (!failing.empty()) {
    FBF_LOG_WARN("hessian: {} coordinate pair(s) not bilinear: no frequencies", failing.size());
  } else if (!out.dipole_derivative.empty()) {
    out.intensities =
        FBF_TRY(hessian::infrared_intensities(out.vibrations.modes, out.dipole_derivative));
    FBF_LOG_INFO("hessian: {} harmonic frequencies (cm^-1) and IR intensities (km/mol)",
                 out.vibrations.frequencies.size());
    for (size_t j = 0; j < out.vibrations.frequencies.size(); ++j) {
      FBF_LOG_INFO("hessian: mode {:>3} {:12.2f} {:12.4f}", j + 1, out.vibrations.frequencies[j],
                   out.intensities[j]);
    }
  } else {
    FBF_LOG_INFO("hessian: {} harmonic frequencies (cm^-1)", out.vibrations.frequencies.size());
    for (size_t j = 0; j < out.vibrations.frequencies.size(); ++j) {
      FBF_LOG_INFO("hessian: mode {:>3} {:12.2f}", j + 1, out.vibrations.frequencies[j]);
    }
  }

  if (!json_path.empty()) {
    json doc{
        {"method", method == HessianMethod::Analytic ? "analytic" : "numerical"},
        {"hessian", hessian_rows(out.hessian, n)},
        {"asymmetry", out.asymmetry},
        {"translational_sum", out.translational_sum},
    };
    if (failing.empty()) {
      doc["frequencies"] = out.vibrations.frequencies;
      doc["modes"] = hessian_rows(out.vibrations.modes, n);
    }
    if (method == HessianMethod::Numerical) {
      doc["displacement"] = displacement;
    }
    if (!coordinates.empty()) {
      doc["coordinates"] = hessian_rows(coordinates, n);
    }
    if (ladder) {
      doc["ladder"] = std::move(ladder_doc);
      doc["bilinearity"] = std::move(bilinearity);
      doc["failing_pairs"] = failing;
    }
    if (casscf_gradient) {
      doc["root"] = casscf_gradient->root;
    }
    if (nevpt2_gradient) {
      doc["variant"] = nevpt2_gradient->variant == Nevpt2Variant::Pc ? "pc" : "sc";
    }
    if (!out.dipole_derivative.empty()) {
      doc["dipole_derivative"] = hessian_rows(out.dipole_derivative, 3);
      doc["intensities"] = out.intensities;
    }
    FBF_TRY(write_json(json_path, doc));
  }
  if (!failing.empty()) {
    std::string named;
    for (const auto &[a, b] : failing) {
      named += std::format("{}[{}, {}]", named.empty() ? "" : ", ", a, b);
    }
    return err_numerical("[{}] is not a bilinear form on {} of the {} coordinate pair(s) checked: "
                         "{}; the Hessian has no frequencies there, and none are published{}",
                         kName, failing.size(), checked.size(), named,
                         json_path.empty()
                             ? std::string{}
                             : std::format(" ({} has the ladders)", json_path.string()));
  }
  return out;
}

} // namespace free_breakfast
