/// @file
/// @brief HessianStep: the analytic Hessian, hessian::df_rhf over one nested
/// Pipeline run of the chain to ScfStep, or hessian::df_casscf or
/// df_sacasscf over one to CasscfStep; or the 6N displaced gradients, each
/// a nested Pipeline run of the chain to GradientStep, and their central
/// differences -- the RHF's, or over the CASSCF chain, each displaced solve
/// started from the reference's orbitals and checked against it; then the
/// symmetrized Hessian, its two checks, the harmonic vibrations, the analytic
/// RHF's dipole derivative and IR intensities, and hessian.json.

module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kOne, kZero
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
  // Each converged active orbital's weight in the start's active space.
  std::vector<double> s_active(nao);
  for (size_t t = 0; t < ncas; ++t) {
    const std::span<const double> orbital = active.subspan(t * nao, nao);
    for (size_t mu = 0; mu < nao; ++mu) {
      s_active[mu] = std::transform_reduce(orbital.begin(), orbital.end(),
                                           s.begin() + static_cast<std::ptrdiff_t>(mu * nao), 0.0);
    }
    double weight = 0.0;
    for (size_t u = 0; u < ncas; ++u) {
      const std::span<const double> start = start_active.subspan(u * nao, nao);
      const double overlap =
          std::transform_reduce(start.begin(), start.end(), s_active.begin(), 0.0);
      weight += overlap * overlap;
    }
    if (!(weight > kFloor)) {
      return err_numerical("[{}] active orbital {} of {} left the reference's active space: its "
                           "weight there is {:.3e}, not above {}",
                           HessianStep::kName, t, ncas, weight, kFloor);
    }
  }
  // Each root's overlap with the reference's root of the same index.
  const size_t dim = ci.size() / nroots;
  for (size_t r = 0; r < nroots; ++r) {
    const std::span<const double> root = ci.subspan(r * dim, dim);
    const double overlap = std::transform_reduce(
        root.begin(), root.end(), reference_ci.begin() + static_cast<std::ptrdiff_t>(r * dim), 0.0);
    if (!(overlap * overlap > kFloor)) {
      return err_numerical("[{}] root {} of {} is not the reference's root {}: their overlap "
                           "squared is {:.3e}, not above {} (the roots changed order, or one "
                           "from outside the average came in)",
                           HessianStep::kName, r, nroots, r, overlap * overlap, kFloor);
    }
  }
  return {};
}

Result<HessianOutput> HessianStep::run(View<MoleculeOutput> in, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  if (casscf_gradient && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] reruns [{}], which runs over density fitting only, and [{}] mode "
                           "is 'integral_direct': set [{}] mode = 'density_fitting'",
                           kName, CasscfGradientStep::kName, EriStep::kName, EriStep::kName);
  }
  if (method == HessianMethod::Analytic && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] method 'analytic' is density-fitted only, and [{}] mode is "
                           "'integral_direct': set [{}] mode = 'density_fitting', or [{}] method "
                           "= 'numerical'",
                           kName, EriStep::kName, EriStep::kName, kName);
  }
  DeviceResources &res = node.device(0);
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const std::vector<Atom> &atoms = molecule.atoms();
  const size_t n = 3 * atoms.size();

  // The chain writes nothing: its steps' outputs are the outer pipeline's.
  ScfStep chain_scf = scf;
  chain_scf.molden.clear();
  CasscfStep chain_casscf = casscf;
  chain_casscf.molden.clear();
  CasscfGradientStep chain_casscf_gradient = casscf_gradient.value_or(CasscfGradientStep{});
  chain_casscf_gradient.json_path.clear();
  // The molecule at @p atoms moved by @p shift along coordinate x (3 A + k).
  const auto displaced = [&](size_t x, double shift) {
    DisplacedMoleculeStep out{atoms, molecule.charge(), molecule.multiplicity()};
    if (shift != 0.0) {
      Vec<double, 3> &position = out.atoms[x / 3].position;
      (x % 3 == 0 ? position.x : x % 3 == 1 ? position.y : position.z) += shift;
    }
    return out;
  };
  // The Pipeline of the chain to [scf], then @p last, at displaced(x, shift).
  const auto chain_at = [&]<class... Last>(size_t x, double shift, Last... last) {
    return Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                    PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                    CoreHamiltonianStep, GuessStep, ScfStep, Last...>::create({
        displaced(x, shift),
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
  if (method == HessianMethod::Analytic && casscf_gradient) {
    // One chain to [casscf] at R, and the root's analytic Hessian at its
    // solution: hessian::df_casscf for a single root, hessian::df_sacasscf
    // for one root of a state average.
    auto chain = FBF_TRY(chain_at(0, 0.0, chain_casscf));
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
    auto chain = FBF_TRY(chain_at(0, 0.0));
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
    // dE/dR at the geometry with coordinate x moved by @p shift: the RHF's.
    const auto rhf_gradient_at = [&](size_t x, double shift) -> Result<std::vector<double>> {
      auto chain = FBF_TRY(chain_at(x, shift, GradientStep{}));
      FBF_TRY(chain.run(node));
      return chain.get<GradientOutput>()->gradient.total;
    };
    // Under the CASSCF chain, the reference at R: the SCF, the configured
    // pick and its CASSCF, whose orbitals every displaced solve starts from
    // and whose roots each is held to. The active space is picked here only.
    std::optional<ReferenceOrbitalsStep> reference;
    std::vector<double> reference_ci;
    CasscfStep displaced_casscf = chain_casscf;
    if (casscf_gradient) {
      auto chain = FBF_TRY(chain_at(0, 0.0, chain_casscf));
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
      // By index, as they come: core, active, virtual.
      displaced_casscf.active.clear();
      displaced_casscf.avas.clear();
      displaced_casscf.ncas = cas.ncas;
      displaced_casscf.nelecas = cas.nelecas;
      FBF_LOG_INFO("hessian: reference DF-CASSCF({}, {}) over {} root(s), E = {:.10f}", cas.nelecas,
                   cas.ncas, cas.energies.size(), cas.energy);
    }
    // ... and the CASSCF root's, its solve started from the reference's
    // orbitals, refused unless it converged and kept the reference's active
    // space and root order.
    const auto casscf_gradient_at = [&](size_t x, double shift) -> Result<std::vector<double>> {
      auto chain =
          FBF_TRY((Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, CoreHamiltonianStep,
                            ReferenceOrbitalsStep, CasscfStep, CasscfGradientStep>::create({
              displaced(x, shift),
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              CoreHamiltonianStep{},
              *reference,
              displaced_casscf,
              chain_casscf_gradient,
          })));
      const std::string where = std::format("coordinate {} displaced by {:+g} bohr", x, shift);
      FBF_TRY(at_geometry(chain.run(node), where));
      const CasscfOutput &cas = *chain.get<CasscfOutput>();
      const size_t nao = cas.c.n();
      const size_t first = static_cast<size_t>(cas.ncore) * nao;
      const size_t count = static_cast<size_t>(cas.ncas) * nao;
      const std::vector<double> s =
          FBF_TRY(download(res, chain.get<OverlapOutput>()->s.data(), nao * nao));
      const std::vector<double> start =
          FBF_TRY(download(res, chain.get<ScfOutput>()->c.data() + first, count));
      const std::vector<double> active = FBF_TRY(download(res, cas.c.data() + first, count));
      FBF_TRY(at_geometry(check_displaced_casscf(nao, s, start, active, reference_ci, cas.vectors,
                                                 cas.energies.size()),
                          where));
      return chain.get<CasscfGradientOutput>()->gradient.total;
    };
    const auto gradient_at = [&](size_t x, double shift) {
      return casscf_gradient ? casscf_gradient_at(x, shift) : rhf_gradient_at(x, shift);
    };
    // Row x the central difference along x.
    for (size_t x = 0; x < n; ++x) {
      const std::vector<double> plus = FBF_TRY(gradient_at(x, displacement));
      const std::vector<double> minus = FBF_TRY(gradient_at(x, -displacement));
      for (size_t y = 0; y < n; ++y) {
        raw[(x * n) + y] = (plus[y] - minus[y]) / (2.0 * displacement);
      }
      FBF_LOG_INFO("hessian: coordinate {:>3} of {}", x + 1, n);
    }
    for (size_t x = 0; x < n; ++x) {
      for (size_t y = 0; y < n; ++y) {
        asymmetry = std::max(asymmetry, std::abs(raw[(x * n) + y] - raw[(y * n) + x]));
      }
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

  out.vibrations = FBF_TRY(geometry::harmonic_analysis(res, atoms, out.hessian));
  if (!out.dipole_derivative.empty()) {
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
        {"frequencies", out.vibrations.frequencies},
        {"modes", hessian_rows(out.vibrations.modes, n)},
    };
    if (method == HessianMethod::Numerical) {
      doc["displacement"] = displacement;
    }
    if (casscf_gradient) {
      doc["root"] = casscf_gradient->root;
    }
    if (!out.dipole_derivative.empty()) {
      doc["dipole_derivative"] = hessian_rows(out.dipole_derivative, 3);
      doc["intensities"] = out.intensities;
    }
    FBF_TRY(write_json(json_path, doc));
  }
  return out;
}

} // namespace free_breakfast
