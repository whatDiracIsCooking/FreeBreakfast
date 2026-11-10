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
// average), then the analysis; the analytic NEVPT2 [hessian] the same over
// the variant's Hessian. The semi-numerical CASSCF and NEVPT2 [hessian] is
// the reference's chain to [casscf], then each point's chain from the
// carried orbitals to the gradient, then the analysis, under QD with the
// state's gradient and the ensemble's edge beside each chain. An [optimize]
// over a CASSCF chain is the start, then each evaluation's chain and what it
// runs past it, a followed state's C and CI held between evaluations. The
// dimensions are filled by hand, the basis placed on the host and syevd's
// work made up (only its order is read), so nothing touches a device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.io.basis;
import free_breakfast.pipeline.config;
import free_breakfast.hessian;
import free_breakfast.mcscf.intersection;  // df_intersection_footprint
import free_breakfast.mcscf.overlap;       // state_overlaps_footprint
import free_breakfast.nevpt2.intersection; // qd_*_intersection_footprint

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

  const fb::MemoryFootprint fp = fb::OptimizeStep{}.footprint(dims);
  EXPECT_EQ(fp.device.persistent, molecule.device.persistent);
  EXPECT_EQ(fp.device.workspace, workspace);
  EXPECT_EQ(fp.device.peak(), molecule.device.persistent + chain + workspace);
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

  const fb::MemoryFootprint fp =
      fb::HessianStep{.method = fb::HessianMethod::Numerical}.footprint(dims);
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_EQ(fp.device.workspace, std::max(workspace, analysis.device.workspace));
  EXPECT_EQ(fp.device.peak(), std::max(chain, analysis.device.scratch) +
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

  const fb::MemoryFootprint fp =
      fb::HessianStep{.method = fb::HessianMethod::Analytic, .eri = eri}.footprint(dims);
  const std::size_t c = sizeof(double) * dims.nao * dims.nao;
  EXPECT_EQ(fp.device.persistent, 0U);
  EXPECT_EQ(fp.device.workspace, std::max({chain_workspace, rhf.device.workspace,
                                           dipole.device.workspace, analysis.device.workspace}));
  EXPECT_EQ(fp.device.peak() - fp.device.workspace,
            std::max({chain_scratch, held + c + rhf.device.persistent + rhf.device.scratch,
                      held + c + rhf.device.persistent + dipole.device.scratch,
                      analysis.device.scratch}));
}

// Over an integral-direct ERI the analytic method is refused before anything
// runs: nothing held.
TEST(NestedFootprint, AnalyticHessianOverIntegralDirectIsEmpty) {
  const fb::BasisSet basis = water_basis();
  const fb::MemoryFootprint fp = fb::HessianStep{}.footprint(water_dims(basis));
  EXPECT_EQ(fp, fb::MemoryFootprint{});
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

    const fb::MemoryFootprint fp = step.footprint(dims);
    EXPECT_EQ(fp.device.persistent, 0U);
    EXPECT_EQ(fp.device.workspace,
              std::max({chain_workspace, terms.device.workspace, analysis.device.workspace}));
    EXPECT_EQ(
        fp.device.peak() - fp.device.workspace,
        std::max({chain_scratch, held + solution + terms.device.persistent + terms.device.scratch,
                  analysis.device.scratch}));
  }
}

// Over an integral-direct ERI the CASSCF and NEVPT2 chains of the nested
// steps are refused before anything runs, so empty (the plan's dimensions
// carry no aux basis).
TEST(NestedFootprint, IntegralDirectMultireferenceChainsAreEmpty) {
  const fb::BasisSet basis = water_basis();
  const fb::FootprintDimensions dims = water_dims(basis);
  fb::HessianStep casscf_hessian{.method = fb::HessianMethod::Numerical};
  casscf_hessian.casscf_gradient.emplace();
  EXPECT_EQ(casscf_hessian.footprint(dims), fb::MemoryFootprint{});
  fb::HessianStep nevpt2_hessian{.method = fb::HessianMethod::Numerical};
  nevpt2_hessian.nevpt2_gradient.emplace();
  EXPECT_EQ(nevpt2_hessian.footprint(dims), fb::MemoryFootprint{});
  fb::OptimizeStep casscf_optimize;
  casscf_optimize.casscf_gradient.emplace();
  EXPECT_EQ(casscf_optimize.footprint(dims), fb::MemoryFootprint{});
  fb::OptimizeStep nevpt2_optimize;
  nevpt2_optimize.nevpt2_gradient.emplace();
  EXPECT_EQ(nevpt2_optimize.footprint(dims), fb::MemoryFootprint{});
  fb::OptimizeStep meci;
  meci.meci = std::array<std::int32_t, 2>{0, 1};
  meci.casscf.nroots = 2;
  EXPECT_EQ(meci.footprint(dims), fb::MemoryFootprint{});
}

// The analytic NEVPT2 [hessian] (#1186): the chain to [casscf] held under the
// solution's C and CI vectors and the variant's Hessian, SC's or PC's for one
// root, or one QD state's of the two-root average, in 3 n_atom coordinates;
// then the analysis. A frozen core, a QD state outside the average and a
// state average without QD are refused before anything runs: empty.
TEST(NestedFootprint, AnalyticNevpt2HessianIsTheChainThenItsTerms) {
  const fb::BasisSet basis = water_basis();
  const fb::BasisSet aux =
      place_on_water(fb::resolve_aux_basis("cc-pvdz-jkfit").value().primitive_normalized());
  fb::FootprintDimensions dims = water_dims(basis);
  dims.aux = &aux;
  dims.naux = aux.nao_sphe();
  dims.ncas = 4;
  dims.nelecas = 4;
  for (std::size_t order = dims.syevd_lwork.size(); order <= 64; ++order) {
    dims.syevd_lwork.push_back(1000 + (64 * order));
  }
  const fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"};
  const fb::MemoryFootprint analysis =
      fb::geometry::harmonic_analysis_footprint(dims.atoms, dims.syevd_lwork);
  constexpr std::int32_t kK = 9;
  for (const bool qd : {false, true}) {
    for (const fb::Nevpt2Variant variant : {fb::Nevpt2Variant::Sc, fb::Nevpt2Variant::Pc}) {
      SCOPED_TRACE(std::format("qd {}, pc {}", qd, variant == fb::Nevpt2Variant::Pc));
      const std::int32_t nroots = qd ? 2 : 1;
      fb::HessianStep step{.method = fb::HessianMethod::Analytic, .eri = eri};
      step.casscf.nroots = nroots;
      step.nevpt2_gradient.emplace();
      step.nevpt2_gradient->variant = variant;
      step.nevpt2_gradient->quasi_degenerate = qd;
      step.nevpt2_gradient->state = qd ? 1 : 0;
      std::vector<fb::MemoryFootprint> chain = rhf_chain(dims);
      chain.back() = step.casscf.footprint(dims);
      chain[4] = eri.footprint(dims);
      const auto [chain_scratch, chain_workspace] = chain_peak(chain);
      std::size_t held = 0;
      for (const fb::MemoryFootprint &one : chain) {
        held += one.device.persistent;
      }
      const std::size_t solution =
          sizeof(double) * ((dims.nao * dims.nao) + (36 * static_cast<std::size_t>(nroots)));
      const bool pc = variant == fb::Nevpt2Variant::Pc;
      const fb::MemoryFootprint terms =
          qd ? (pc ? fb::hessian::df_qd_pc_nevpt2_footprint(basis, aux, 3, 4, 2, 2, 2, false, kK, 0,
                                                            dims.syevd_lwork)
                   : fb::hessian::df_qd_sc_nevpt2_footprint(basis, aux, 3, 4, 2, 2, 2, false, kK, 0,
                                                            dims.syevd_lwork))
             : (pc ? fb::hessian::df_pc_nevpt2_footprint(basis, aux, 3, 4, 2, 2, kK, 0,
                                                         dims.syevd_lwork)
                   : fb::hessian::df_sc_nevpt2_footprint(basis, aux, 3, 4, 2, 2, kK, 0,
                                                         dims.syevd_lwork));
      ASSERT_GT(terms.device.scratch, 0U);

      const fb::MemoryFootprint fp = step.footprint(dims);
      EXPECT_EQ(fp.device.persistent, 0U);
      EXPECT_EQ(fp.device.workspace,
                std::max({chain_workspace, terms.device.workspace, analysis.device.workspace}));
      EXPECT_EQ(
          fp.device.peak() - fp.device.workspace,
          std::max({chain_scratch, held + solution + terms.device.persistent + terms.device.scratch,
                    analysis.device.scratch}));

      // Refused before anything runs.
      fb::HessianStep frozen = step;
      frozen.nevpt2_gradient->frozen_core = 1;
      EXPECT_EQ(frozen.footprint(dims), fb::MemoryFootprint{});
      fb::HessianStep outside = step;
      outside.casscf.nroots = qd ? 1 : 2;
      EXPECT_EQ(outside.footprint(dims), fb::MemoryFootprint{});
    }
  }
}

// The semi-numerical CASSCF and NEVPT2 Hessians (#1204) of a CAS(4,4) over
// water in STO-3G: the reference's chain to [casscf], freed; each point's
// chain from the carried orbitals (and a carried CI guess) to the gradient,
// freed; then the analysis. Under QD, beside a chain's products, the state's
// gradient (P's at the reference, the tracked state's retaken at a point,
// beside the chain's own) and beside that the ensemble's edge: C and d, and
// a CASCI of one more root. The dimensions carry no [casscf]'s roots (none is
// planned at the top level): the step sizes its chain's readers over its
// own CASSCF's.
TEST(NestedFootprint, SemiNumericalMultireferenceHessianIsTheReferenceThenEachPoint) {
  const fb::BasisSet basis = water_basis();
  const fb::BasisSet aux =
      place_on_water(fb::resolve_aux_basis("cc-pvdz-jkfit").value().primitive_normalized());
  fb::FootprintDimensions dims = water_dims(basis);
  dims.aux = &aux;
  dims.naux = aux.nao_sphe();
  dims.ncas = 4;
  dims.nelecas = 4;
  for (std::size_t order = dims.syevd_lwork.size(); order <= 64; ++order) {
    dims.syevd_lwork.push_back(1000 + (64 * order));
  }
  const fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"};
  const fb::MemoryFootprint analysis =
      fb::geometry::harmonic_analysis_footprint(dims.atoms, dims.syevd_lwork);
  struct Case {
    std::string_view name;
    std::vector<double> weights;
    bool nevpt2 = false;
    bool qd = false;
    fb::Nevpt2Variant variant = fb::Nevpt2Variant::Sc;
  };
  const std::vector<Case> cases{
      {.name = "casscf"},
      {.name = "sa-casscf", .weights = {0.7, 0.3}},
      {.name = "sc", .nevpt2 = true},
      {.name = "pc", .nevpt2 = true, .variant = fb::Nevpt2Variant::Pc},
      {.name = "qd-sc", .weights = {0.5, 0.5}, .nevpt2 = true, .qd = true},
      {.name = "qd-pc",
       .weights = {0.5, 0.5},
       .nevpt2 = true,
       .qd = true,
       .variant = fb::Nevpt2Variant::Pc},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    const auto nroots = static_cast<std::int32_t>(std::max<std::size_t>(c.weights.size(), 1));
    fb::HessianStep step{.method = fb::HessianMethod::Numerical, .eri = eri};
    step.casscf.nroots = nroots;
    step.casscf.weights = c.weights;
    if (c.nevpt2) {
      step.nevpt2_gradient.emplace();
      step.nevpt2_gradient->variant = c.variant;
      step.nevpt2_gradient->quasi_degenerate = c.qd;
      step.nevpt2_gradient->state = c.qd ? 1 : 0;
    } else {
      step.casscf_gradient.emplace();
    }
    fb::FootprintDimensions at = dims;
    at.casscf_nroots = static_cast<std::size_t>(nroots);
    at.casscf_unequal_weights = !c.weights.empty() && c.weights[0] != c.weights[1];
    const fb::MemoryFootprint last =
        c.nevpt2 ? step.nevpt2_gradient->footprint(at) : step.casscf_gradient->footprint(at);
    ASSERT_GT(last.device.peak(), 0U);
    // The reference: the RHF chain to [scf], then [casscf].
    std::vector<fb::MemoryFootprint> reference = rhf_chain(at);
    reference.back() = step.casscf.footprint(at);
    reference[4] = eri.footprint(at);
    // A point: no orthogonalizer, guess or SCF; the carried orbitals, the
    // CASSCF over the reference's space from its CI, then the gradient.
    fb::CasscfStep displaced = step.casscf;
    displaced.ncas = 4;
    displaced.nelecas = 4;
    displaced.ci_guess = {0.0};
    const std::vector<fb::MemoryFootprint> point{fb::molecule_footprint(at.natoms),
                                                 fb::BasisSetStep{}.footprint(at),
                                                 fb::SphericalTransformStep{}.footprint(at),
                                                 fb::PrimitivePairClassesStep{}.footprint(at),
                                                 eri.footprint(at),
                                                 fb::OverlapStep{}.footprint(at),
                                                 fb::CoreHamiltonianStep{}.footprint(at),
                                                 fb::CarriedOrbitalsStep{}.footprint(at),
                                                 displaced.footprint(at),
                                                 last};
    // The carried CI guess is counted: one root's 36 determinants or two's.
    EXPECT_EQ(displaced.footprint(at).device.peak() - step.casscf.footprint(at).device.peak(),
              sizeof(double) * 36 * static_cast<std::size_t>(nroots));
    const auto [reference_scratch, reference_workspace] = chain_peak(reference);
    const auto [point_scratch, point_workspace] = chain_peak(point);
    std::size_t reference_held = 0;
    for (const fb::MemoryFootprint &one : reference) {
      reference_held += one.device.persistent;
    }
    std::size_t point_held = 0;
    for (const fb::MemoryFootprint &one : point) {
      point_held += one.device.persistent;
    }
    std::size_t scratch = std::max({reference_scratch, point_scratch, analysis.device.scratch});
    std::size_t workspace =
        std::max({reference_workspace, point_workspace, analysis.device.workspace});
    if (c.qd) {
      fb::CasciStep casci;
      casci.ncas = 4;
      casci.nelecas = 4;
      casci.nroots = 3;
      const fb::MemoryFootprint edge = casci.footprint(at);
      const std::size_t orbitals = 2 * sizeof(double) * dims.nao * dims.nao;
      const std::size_t g = last.device.persistent;
      for (const std::size_t held : {reference_held, point_held}) {
        scratch = std::max({scratch, held + g + last.device.scratch,
                            held + g + orbitals + edge.device.persistent + edge.device.scratch});
      }
      workspace = std::max({workspace, edge.device.workspace});
    }

    const fb::MemoryFootprint fp = step.footprint(dims);
    EXPECT_EQ(fp.device.persistent, 0U);
    EXPECT_EQ(fp.device.workspace, workspace);
    EXPECT_EQ(fp.device.peak() - fp.device.workspace, scratch);

    // Refused before anything runs: both gradients, an integral-direct ERI.
    fb::HessianStep both = step;
    both.casscf_gradient.emplace();
    both.nevpt2_gradient.emplace();
    EXPECT_EQ(both.footprint(dims), fb::MemoryFootprint{});
    fb::HessianStep direct = step;
    direct.eri.mode = fb::EriMode::IntegralDirect;
    EXPECT_EQ(direct.footprint(dims), fb::MemoryFootprint{});
  }
}

// The [optimize] over a CAS(4,4) of water in STO-3G (#1212): the start's
// Molecule; the first evaluation's chain to [casscf] from the SCF, then every
// later one's from the carried orbitals, each to the NEVPT2 gradient under a
// [nevpt2_gradient], freed whole; beside a chain's products, a CASSCF
// root's gradient or under meci the intersection. A followed state (an SA
// root, a QD state, a pair) holds the CASSCF's C and CI vectors beside that,
// kept to the next evaluation, which holds its own beside it and
// state_overlaps'. The dimensions carry no [casscf]'s roots: the step sizes
// its chain's readers over its own CASSCF's.
TEST(NestedFootprint, MultireferenceOptimizeIsTheStartThenEachEvaluation) {
  const fb::BasisSet basis = water_basis();
  const fb::BasisSet aux =
      place_on_water(fb::resolve_aux_basis("cc-pvdz-jkfit").value().primitive_normalized());
  fb::FootprintDimensions dims = water_dims(basis);
  dims.aux = &aux;
  dims.naux = aux.nao_sphe();
  dims.ncas = 4;
  dims.nelecas = 4;
  for (std::size_t order = dims.syevd_lwork.size(); order <= 64; ++order) {
    dims.syevd_lwork.push_back(1000 + (64 * order));
  }
  const fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"};
  const fb::MemoryFootprint molecule = fb::molecule_footprint(dims.natoms);
  enum class Kind : std::uint8_t { Casscf, Nevpt2, Meci };
  struct Case {
    std::string_view name;
    Kind kind = Kind::Casscf;
    std::vector<double> weights;
    bool qd = false;
    fb::Nevpt2Variant variant = fb::Nevpt2Variant::Sc;
  };
  const std::vector<Case> cases{
      {.name = "casscf"},
      {.name = "sa-casscf", .weights = {0.7, 0.3}},
      {.name = "sc", .kind = Kind::Nevpt2},
      {.name = "pc", .kind = Kind::Nevpt2, .variant = fb::Nevpt2Variant::Pc},
      {.name = "qd-sc", .kind = Kind::Nevpt2, .weights = {0.5, 0.5}, .qd = true},
      {.name = "qd-pc",
       .kind = Kind::Nevpt2,
       .weights = {0.5, 0.5},
       .qd = true,
       .variant = fb::Nevpt2Variant::Pc},
      {.name = "meci", .kind = Kind::Meci, .weights = {0.5, 0.5}},
      {.name = "qd-meci-sc", .kind = Kind::Meci, .weights = {0.6, 0.4}, .qd = true},
      {.name = "qd-meci-pc",
       .kind = Kind::Meci,
       .weights = {0.6, 0.4},
       .qd = true,
       .variant = fb::Nevpt2Variant::Pc},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    const auto nroots = static_cast<std::int32_t>(std::max<std::size_t>(c.weights.size(), 1));
    const bool rotating = !c.weights.empty() && c.weights[0] != c.weights[1];
    fb::OptimizeStep step{.eri = eri};
    step.casscf.ncas = 4;
    step.casscf.nelecas = 4;
    step.casscf.nroots = nroots;
    step.casscf.weights = c.weights;
    if (c.kind == Kind::Casscf) {
      step.casscf_gradient.emplace();
    } else if (c.kind == Kind::Nevpt2 || c.qd) {
      step.nevpt2_gradient.emplace();
      step.nevpt2_gradient->variant = c.variant;
      step.nevpt2_gradient->quasi_degenerate = c.qd;
    }
    if (c.kind == Kind::Meci) {
      step.meci = std::array<std::int32_t, 2>{0, 1};
    }
    fb::FootprintDimensions at = dims;
    at.casscf_nroots = static_cast<std::size_t>(nroots);
    at.casscf_unequal_weights = rotating;
    // The first chain: the RHF chain to [scf], then [casscf]; a later one: no
    // orthogonalizer, guess or SCF, the carried orbitals and the CASSCF over
    // them. Each ends in the NEVPT2 gradient under a [nevpt2_gradient]
    // without meci.
    std::vector<fb::MemoryFootprint> first = rhf_chain(at);
    first.back() = step.casscf.footprint(at);
    first[4] = eri.footprint(at);
    std::vector<fb::MemoryFootprint> later{fb::molecule_footprint(at.natoms),
                                           fb::BasisSetStep{}.footprint(at),
                                           fb::SphericalTransformStep{}.footprint(at),
                                           fb::PrimitivePairClassesStep{}.footprint(at),
                                           eri.footprint(at),
                                           fb::OverlapStep{}.footprint(at),
                                           fb::CoreHamiltonianStep{}.footprint(at),
                                           fb::CarriedOrbitalsStep{}.footprint(at),
                                           step.casscf.footprint(at)};
    if (c.kind == Kind::Nevpt2) {
      first.push_back(step.nevpt2_gradient->footprint(at));
      later.push_back(step.nevpt2_gradient->footprint(at));
      ASSERT_GT(first.back().device.peak(), 0U);
    }
    // Past the chain: the root's gradient, or the intersection.
    fb::MemoryFootprint finish;
    if (c.kind == Kind::Casscf) {
      finish = step.casscf_gradient->footprint(at);
    } else if (c.kind == Kind::Meci && c.qd) {
      finish = c.variant == fb::Nevpt2Variant::Pc
                   ? fb::nevpt2::qd_pc_intersection_footprint(basis, aux, 3, 4, 4, 1, 2, rotating,
                                                              0, dims.syevd_lwork)
                   : fb::nevpt2::qd_sc_intersection_footprint(basis, aux, 3, 4, 4, 1, 2, rotating,
                                                              0, dims.syevd_lwork);
    } else if (c.kind == Kind::Meci) {
      finish = fb::mcscf::df_intersection_footprint(basis, aux, 3, 4, 4, 1, 2, rotating);
    }
    if (c.kind != Kind::Nevpt2) {
      ASSERT_GT(finish.device.peak(), 0U);
    }
    const bool followed = c.kind == Kind::Meci || c.qd || nroots > 1;
    // C and two roots' (or one's) 36 determinants; state_overlaps', freed.
    const std::size_t solution =
        followed
            ? sizeof(double) * ((dims.nao * dims.nao) + (36 * static_cast<std::size_t>(nroots)))
            : 0;
    const fb::MemoryFootprint overlaps =
        followed ? fb::mcscf::state_overlaps_footprint(basis, basis, 3, 4, 4, 1, nroots, nroots)
                 : fb::MemoryFootprint{};
    const auto held = [](const std::vector<fb::MemoryFootprint> &chain) {
      std::size_t sum = 0;
      for (const fb::MemoryFootprint &one : chain) {
        sum += one.device.persistent;
      }
      return sum;
    };
    const auto [first_scratch, first_workspace] = chain_peak(first);
    const auto [later_scratch, later_workspace] = chain_peak(later);
    const std::size_t m = molecule.device.persistent;
    const std::size_t past = finish.device.persistent + finish.device.scratch;
    const std::size_t at_first = std::max(first_scratch, held(first) + solution + past);
    // The last state's freed before a root's gradient, which follows the
    // match; kept through the intersection.
    const std::size_t at_later =
        c.kind == Kind::Casscf
            ? std::max(solution + std::max(later_scratch,
                                           held(later) + solution + overlaps.device.scratch),
                       std::max(later_scratch, held(later) + solution + past))
            : solution + std::max(later_scratch,
                                  held(later) + solution + finish.device.persistent +
                                      std::max(overlaps.device.scratch, finish.device.scratch));
    const std::size_t scratch = std::max({m + at_first, m + at_later, m + solution + m});
    const std::size_t workspace = std::max(
        {first_workspace, later_workspace, finish.device.workspace, overlaps.device.workspace});

    const fb::MemoryFootprint fp = step.footprint(dims);
    EXPECT_EQ(fp.device.persistent, m);
    EXPECT_EQ(fp.device.workspace, workspace);
    EXPECT_EQ(fp.device.peak() - fp.device.workspace, scratch);

    // Refused before anything runs: both gradients, an integral-direct ERI,
    // meci beside [casscf_gradient].
    fb::OptimizeStep both = step;
    both.casscf_gradient.emplace();
    both.nevpt2_gradient.emplace();
    EXPECT_EQ(both.footprint(dims), fb::MemoryFootprint{});
    fb::OptimizeStep direct = step;
    direct.eri.mode = fb::EriMode::IntegralDirect;
    EXPECT_EQ(direct.footprint(dims), fb::MemoryFootprint{});
    if (c.kind == Kind::Meci) {
      fb::OptimizeStep beside = step;
      beside.casscf_gradient.emplace();
      beside.nevpt2_gradient.reset();
      EXPECT_EQ(beside.footprint(dims), fb::MemoryFootprint{});
    }
  }
}

} // namespace
