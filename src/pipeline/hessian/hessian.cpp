/// @file
/// @brief HessianStep: the analytic Hessian, hessian::df_rhf over one nested
/// Pipeline run of the chain to ScfStep, or the 6N displaced gradients, each
/// a nested Pipeline run of the chain to GradientStep, and their central
/// differences; then the symmetrized Hessian, its two checks, the harmonic
/// vibrations, the analytic method's dipole derivative and IR intensities,
/// and hessian.json.

module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.core.wwr;
import free_breakfast.io.json;
import free_breakfast.basis;
import free_breakfast.hessian;
import free_breakfast.integrals.one_electron.dipole; // kDipoleMaxL
import free_breakfast.pipeline.pipeline;

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

// @p a, n x n row-major, as n rows of n.
[[nodiscard]] json hessian_rows(const std::vector<double> &a, size_t n) {
  json rows = json::array();
  for (size_t x = 0; x + n <= a.size(); x += n) {
    rows.push_back(std::vector<double>(a.begin() + static_cast<std::ptrdiff_t>(x),
                                       a.begin() + static_cast<std::ptrdiff_t>(x + n)));
  }
  return rows;
}

Result<HessianOutput> HessianStep::run(View<MoleculeOutput> in, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
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
  // The Pipeline of the chain to [scf], then @p last, at @p atoms moved by
  // @p shift along coordinate x (3 A + k).
  const auto chain_at = [&]<class... Last>(size_t x, double shift, Last... last) {
    DisplacedMoleculeStep displaced{atoms, molecule.charge(), molecule.multiplicity()};
    if (shift != 0.0) {
      Vec<double, 3> &position = displaced.atoms[x / 3].position;
      (x % 3 == 0 ? position.x : x % 3 == 1 ? position.y : position.z) += shift;
    }
    return Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                    PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                    CoreHamiltonianStep, GuessStep, ScfStep, Last...>::create({
        std::move(displaced),
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
  if (method == HessianMethod::Analytic) {
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
    // dE/dR at the geometry with coordinate x moved by @p shift.
    const auto gradient_at = [&](size_t x, double shift) -> Result<std::vector<double>> {
      auto chain = FBF_TRY(chain_at(x, shift, GradientStep{}));
      FBF_TRY(chain.run(node));
      return chain.get<GradientOutput>()->gradient.total;
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
    if (!out.dipole_derivative.empty()) {
      doc["dipole_derivative"] = hessian_rows(out.dipole_derivative, 3);
      doc["intensities"] = out.intensities;
    }
    FBF_TRY(write_json(json_path, doc));
  }
  return out;
}

} // namespace free_breakfast
