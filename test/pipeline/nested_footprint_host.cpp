// The nested steps' footprints: an [optimize] over the RHF chain is the
// start's Molecule plus the chain's peak, and a semi-numerical [hessian] the
// larger of the chain's peak and its harmonic analysis's, each held to the
// value worked out by hand here, exactly, from the chain's own steps'
// footprints over water in STO-3G, integral-direct. A chain whose library
// calls are not all footprinted has none. The dimensions are filled by hand,
// the basis placed on the host and syevd's work made up (only its order is
// read), so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.io.basis;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// Water, bohr.
const std::vector<fb::Atom> kWater{
    fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.2217}, fb::Element::O},
    fb::Atom{fb::Vec<double, 3>{0.0, 1.4309, -0.8867}, fb::Element::H},
    fb::Atom{fb::Vec<double, 3>{0.0, -1.4309, -0.8867}, fb::Element::H},
};

// STO-3G placed on kWater, as the BasisSet step places it.
fb::BasisSet water_basis() {
  const fb::BasisStamp stamp = fb::resolve_basis("sto-3g").value().primitive_normalized();
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

// The chains not footprinted yet: the analytic Hessian (#1053), and the
// CASSCF chains of both steps.
TEST(NestedFootprint, UnfootprintedChainsHaveNone) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  EXPECT_FALSE(fb::HessianStep{}.footprint(dims).has_value());
  fb::HessianStep casscf_hessian{.method = fb::HessianMethod::Numerical};
  casscf_hessian.casscf_gradient.emplace();
  EXPECT_FALSE(casscf_hessian.footprint(dims).has_value());
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
  EXPECT_FALSE(fb::step_footprint(fb::HessianStep{}, dims).has_value());
}

} // namespace
