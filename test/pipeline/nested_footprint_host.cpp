// The nested steps' footprints: an [optimize] over the RHF chain is the
// start's Molecule plus the chain's peak, and a semi-numerical [hessian] the
// larger of the chain's peak and its harmonic analysis's, each held to the
// value worked out by hand here, exactly, from the chain's own steps'
// footprints over water in STO-3G, integral-direct. The analytic DF-RHF
// [hessian], over cc-pVDZ-JKFIT placed the same way, is the chain to [scf]
// held under C's buffer, hessian::df_rhf's and the dipole derivative's, then
// the analysis; over an integral-direct ERI, empty. The analytic DF-CASSCF
// [hessian] is the chain to [casscf] held under the solution's C and CI
// vectors and hessian::df_casscf's (one root) or df_sacasscf's (a state
// average), then the analysis. A chain whose library calls are not all
// footprinted has none. The dimensions are filled by hand,
// the basis placed on the host and syevd's work made up (only its order is
// read), so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.io.basis;
import free_breakfast.pipeline.config;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// Water, bohr.
const std::vector<fb::Atom> kWater{
    fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.2217}, fb::Element::O},
    fb::Atom{fb::Vec<double, 3>{0.0, 1.4309, -0.8867}, fb::Element::H},
    fb::Atom{fb::Vec<double, 3>{0.0, -1.4309, -0.8867}, fb::Element::H},
};

// @p stamp placed on kWater, as the BasisSet and Eri steps place it.
fb::BasisSet place_on_water(const fb::BasisStamp &stamp) {
  std::vector<fb::Shell> shells;
  std::vector<std::uint32_t> offsets{0};
  for (const fb::Atom &atom : kWater) {
    for (const fb::ShellStamp &proto : *stamp.shells_for(fb::nuclear_charge(atom))) {
      shells.push_back(fb::place(proto, atom.position));
    }
    offsets.push_back(static_cast<std::uint32_t>(shells.size()));
  }
  return fb::BasisSet{std::move(shells), std::move(offsets), stamp.name()};
}

// STO-3G placed on kWater.
fb::BasisSet water_basis() {
  return place_on_water(fb::resolve_basis("sto-3g").value().primitive_normalized());
}

// The dimensions plan_shape fills for water in @p basis, with syevd's work at
// order k made up as 1000 + 64 k, through 3 natoms.
fb::FootprintDimensions water_dims(const fb::BasisSet &basis) {
  fb::FootprintDimensions dims;
  dims.basis = &basis;
  dims.atoms = kWater;
  dims.natoms = kWater.size();
  dims.nao = basis.nao_sphe();
  dims.n_alpha = dims.n_beta = dims.nocc = 5;
  std::int32_t l_max = 0;
  for (const fb::Shell &shell : basis.shells()) {
    l_max = std::max(l_max, shell.angular_momentum());
  }
  const auto nl = static_cast<std::size_t>(l_max + 1);
  dims.mmd_records.assign(nl, std::vector<std::size_t>(nl, 0));
  const std::vector<fb::Shell> &shells = basis.shells();
  for (std::size_t i = 0; i < shells.size(); ++i) {
    for (std::size_t j = i; j < shells.size(); ++j) {
      const auto li = static_cast<std::size_t>(shells[i].angular_momentum());
      const auto lj = static_cast<std::size_t>(shells[j].angular_momentum());
      dims.mmd_records[std::min(li, lj)][std::max(li, lj)] +=
          std::size_t{shells[i].nprimitives()} * shells[j].nprimitives();
    }
  }
  for (std::size_t order = 0; order <= std::max(dims.nao, 3 * dims.natoms); ++order) {
    dims.syevd_lwork.push_back(1000 + (64 * order));
  }
  return dims;
}

// The RHF chain's steps' footprints at @p dims, in the order an evaluation
// runs them: each step reads only steps before it.
std::vector<fb::MemoryFootprint> rhf_chain(const fb::FootprintDimensions &dims) {
  return {fb::molecule_footprint(dims.natoms),
          fb::BasisSetStep{}.footprint(dims),
          fb::SphericalTransformStep{}.footprint(dims),
          fb::PrimitivePairClassesStep{}.footprint(dims),
          fb::EriStep{}.footprint(dims),
          fb::OverlapStep{}.footprint(dims),
          fb::OrthogonalizerStep{}.footprint(dims),
          fb::CoreHamiltonianStep{}.footprint(dims),
          fb::GuessStep{}.footprint(dims),
          fb::ScfStep{}.footprint(dims),
          fb::GradientStep{}.footprint(dims)};
}

// One chain's device peak without its workspace, by hand: at step k, every
// product up to k and k's scratch. And its largest workspace.
std::pair<std::size_t, std::size_t> chain_peak(const std::vector<fb::MemoryFootprint> &chain) {
  std::size_t held = 0;
  std::size_t peak = 0;
  std::size_t workspace = 0;
  for (const fb::MemoryFootprint &step : chain) {
    held += step.device.persistent;
    peak = std::max(peak, held + step.device.scratch);
    workspace = std::max(workspace, step.device.workspace);
  }
  return {peak, workspace};
}

// The start's Molecule, held through the walk, under every evaluation's
// chain; the optimum's, the product, kept.
TEST(NestedFootprint, OptimizeIsTheStartPlusTheChain) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  ASSERT_EQ(dims.nao, 7U);
  const auto [chain, workspace] = chain_peak(rhf_chain(dims));
  const fb::MemoryFootprint molecule = fb::molecule_footprint(dims.natoms);
  ASSERT_GT(molecule.device.persistent, 0U);
  ASSERT_GT(chain, molecule.device.persistent);

  const std::optional<fb::MemoryFootprint> fp = fb::OptimizeStep{}.footprint(dims);
  ASSERT_TRUE(fp.has_value());
  EXPECT_EQ(fp->device.persistent, molecule.device.persistent);
  EXPECT_EQ(fp->device.workspace, workspace);
  EXPECT_EQ(fp->device.peak(), molecule.device.persistent + chain + workspace);
}

// No product on the device: the chain's peak at every displacement, then the
// harmonic analysis's, the larger of the two.
TEST(NestedFootprint, NumericalHessianIsTheChainThenTheAnalysis) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  const auto [chain, workspace] = chain_peak(rhf_chain(dims));
  const fb::MemoryFootprint analysis =
      fb::geometry::harmonic_analysis_footprint(dims.atoms, dims.syevd_lwork);
  ASSERT_GT(analysis.device.scratch, 0U);

  const std::optional<fb::MemoryFootprint> fp =
      fb::HessianStep{.method = fb::HessianMethod::Numerical}.footprint(dims);
  ASSERT_TRUE(fp.has_value());
  EXPECT_EQ(fp->device.persistent, 0U);
  EXPECT_EQ(fp->device.workspace, std::max(workspace, analysis.device.workspace));
  EXPECT_EQ(fp->device.peak(), std::max(chain, analysis.device.scratch) +
                                   std::max(workspace, analysis.device.workspace));
}

// The analytic DF-RHF Hessian: the chain to [scf] (water_dims' syevd work
// and no gradient), its products held through hessian::df_rhf, C's buffer
// beside them, and the dipole derivative beside df_rhf's U^x (STO-3G has no
// shell past the dipole integrals' d); all freed before the analysis.
TEST(NestedFootprint, AnalyticRhfHessianIsTheChainThenDfRhf) {
  const fb::BasisSet basis = water_basis();
  const fb::BasisSet aux =
      place_on_water(fb::resolve_aux_basis("cc-pvdz-jkfit").value().primitive_normalized());
  fb::FootprintDimensions dims = water_dims(basis);
  dims.aux = &aux;
  dims.naux = aux.nao_sphe();
  const fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"};
  std::vector<fb::MemoryFootprint> chain = rhf_chain(dims);
  chain.pop_back();
  chain[4] = eri.footprint(dims);
  const auto [chain_scratch, chain_workspace] = chain_peak(chain);
  std::size_t held = 0;
  for (const fb::MemoryFootprint &step : chain) {
    held += step.device.persistent;
  }
  const auto nmo = static_cast<std::int32_t>(dims.nao);
  const fb::MemoryFootprint rhf = fb::hessian::df_rhf_footprint(basis, aux, nmo, 5);
  const fb::MemoryFootprint dipole = fb::hessian::rhf_dipole_derivative_footprint(basis, nmo, 5);
  const fb::MemoryFootprint analysis =
      fb::geometry::harmonic_analysis_footprint(dims.atoms, dims.syevd_lwork);
  ASSERT_GT(rhf.device.persistent, 0U);
  ASSERT_GT(rhf.device.scratch, 0U);
  ASSERT_GT(dipole.device.scratch, 0U);
  EXPECT_EQ(dipole.device.persistent, 0U);

  const std::optional<fb::MemoryFootprint> fp =
      fb::HessianStep{.method = fb::HessianMethod::Analytic, .eri = eri}.footprint(dims);
  ASSERT_TRUE(fp.has_value());
  const std::size_t c = sizeof(double) * dims.nao * dims.nao;
  EXPECT_EQ(fp->device.persistent, 0U);
  EXPECT_EQ(fp->device.workspace, std::max({chain_workspace, rhf.device.workspace,
                                            dipole.device.workspace, analysis.device.workspace}));
  EXPECT_EQ(fp->device.peak() - fp->device.workspace,
            std::max({chain_scratch, held + c + rhf.device.persistent + rhf.device.scratch,
                      held + c + rhf.device.persistent + dipole.device.scratch,
                      analysis.device.scratch}));
}

// Over an integral-direct ERI the analytic method is refused before anything
// runs: nothing held.
TEST(NestedFootprint, AnalyticHessianOverIntegralDirectIsEmpty) {
  const fb::BasisSet basis = water_basis();
  const std::optional<fb::MemoryFootprint> fp = fb::HessianStep{}.footprint(water_dims(basis));
  ASSERT_TRUE(fp.has_value());
  EXPECT_EQ(*fp, fb::MemoryFootprint{});
}

// The analytic DF-CASSCF Hessian of a CAS(4,4) over water in STO-3G (three
// core orbitals), one root and one of two at equal weights and at unequal
// ones: the chain to [casscf], its products held through the branch, the
// solution's C and CI vectors beside them and hessian::df_casscf's or
// df_sacasscf's; all freed before the analysis.
TEST(NestedFootprint, AnalyticCasscfHessianIsTheChainThenDfCasscf) {
  const fb::BasisSet basis = water_basis();
  const fb::BasisSet aux =
      place_on_water(fb::resolve_aux_basis("cc-pvdz-jkfit").value().primitive_normalized());
  fb::FootprintDimensions dims = water_dims(basis);
  dims.aux = &aux;
  dims.naux = aux.nao_sphe();
  dims.ncas = 4;
  dims.nelecas = 4;
  // Through the orders a CASSCF diagonalizes at: its CI subspace and its
  // augmented Hessian's, past nao.
  for (std::size_t order = dims.syevd_lwork.size(); order <= 64; ++order) {
    dims.syevd_lwork.push_back(1000 + (64 * order));
  }
  const fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"};
  const fb::MemoryFootprint analysis =
      fb::geometry::harmonic_analysis_footprint(dims.atoms, dims.syevd_lwork);
  for (const std::vector<double> &weights :
       {std::vector<double>{}, std::vector<double>{0.5, 0.5}, std::vector<double>{0.7, 0.3}}) {
    SCOPED_TRACE(weights.size());
    const auto nroots = static_cast<std::int32_t>(std::max<std::size_t>(weights.size(), 1));
    fb::HessianStep step{.method = fb::HessianMethod::Analytic, .eri = eri};
    step.casscf.nroots = nroots;
    step.casscf.weights = weights;
    step.casscf_gradient.emplace();
    std::vector<fb::MemoryFootprint> chain = rhf_chain(dims);
    chain.back() = step.casscf.footprint(dims);
    chain[4] = eri.footprint(dims);
    const auto [chain_scratch, chain_workspace] = chain_peak(chain);
    std::size_t held = 0;
    for (const fb::MemoryFootprint &one : chain) {
      held += one.device.persistent;
    }
    // CAS(4,4) at two alpha and two beta electrons: 36 determinants.
    const std::size_t solution =
        sizeof(double) * ((dims.nao * dims.nao) + (36 * static_cast<std::size_t>(nroots)));
    const fb::MemoryFootprint terms =
        nroots == 1
            ? fb::hessian::df_casscf_footprint(basis, aux, 3, 4, 2, 2, dims.syevd_lwork[4])
            : fb::hessian::df_sacasscf_footprint(basis, aux, 3, 4, 2, 2, nroots,
                                                 weights[0] != weights[1], dims.syevd_lwork[4]);
    ASSERT_GT(terms.device.persistent, 0U);
    ASSERT_GT(terms.device.scratch, 0U);

    const std::optional<fb::MemoryFootprint> fp = step.footprint(dims);
    ASSERT_TRUE(fp.has_value());
    EXPECT_EQ(fp->device.persistent, 0U);
    EXPECT_EQ(fp->device.workspace,
              std::max({chain_workspace, terms.device.workspace, analysis.device.workspace}));
    EXPECT_EQ(
        fp->device.peak() - fp->device.workspace,
        std::max({chain_scratch, held + solution + terms.device.persistent + terms.device.scratch,
                  analysis.device.scratch}));
  }
}

// The chains not footprinted yet (#1176): the semi-numerical CASSCF
// Hessian, the NEVPT2 Hessians by either method, and the CASSCF and NEVPT2
// chains of the [optimize] step.
TEST(NestedFootprint, UnfootprintedChainsHaveNone) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  fb::HessianStep casscf_hessian{.method = fb::HessianMethod::Numerical};
  casscf_hessian.casscf_gradient.emplace();
  EXPECT_FALSE(casscf_hessian.footprint(dims).has_value());
  for (const fb::HessianMethod method :
       {fb::HessianMethod::Analytic, fb::HessianMethod::Numerical}) {
    fb::HessianStep nevpt2_hessian{.method = method};
    nevpt2_hessian.nevpt2_gradient.emplace();
    EXPECT_FALSE(nevpt2_hessian.footprint(dims).has_value());
  }
  fb::OptimizeStep casscf_optimize;
  casscf_optimize.casscf_gradient.emplace();
  EXPECT_FALSE(casscf_optimize.footprint(dims).has_value());
  fb::OptimizeStep nevpt2_optimize;
  nevpt2_optimize.nevpt2_gradient.emplace();
  EXPECT_FALSE(nevpt2_optimize.footprint(dims).has_value());
  fb::OptimizeStep meci;
  meci.meci = std::array<std::int32_t, 2>{0, 1};
  EXPECT_FALSE(meci.footprint(dims).has_value());
}

// Through the planner: step_footprint passes a nested step's optional
// through, a value or none.
TEST(NestedFootprint, StepFootprintPassesTheOptionalThrough) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  EXPECT_EQ(fb::step_footprint(fb::OptimizeStep{}, dims), fb::OptimizeStep{}.footprint(dims));
  EXPECT_EQ(fb::step_footprint(fb::HessianStep{}, dims), fb::HessianStep{}.footprint(dims));
  fb::HessianStep nevpt2_hessian;
  nevpt2_hessian.nevpt2_gradient.emplace();
  EXPECT_FALSE(fb::step_footprint(nevpt2_hessian, dims).has_value());
}

} // namespace
