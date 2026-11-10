// plan_shape over parsed configs: water in cc-pVDZ, its dimensions held to the
// values counted by hand beside each case, exactly, and mmd's record counts to
// the classes primitive_pair_classes builds. Nothing is computed on the device
// but those classes, but the Molecule owns its nuclei there and syevd's size
// is the solver handle's to say, so every case needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.core.nontransferrable;
import free_breakfast.io.base;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.lu; // getrf_lwork
import free_breakfast.pipeline.shape;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

constexpr std::string_view kWaterXyz = R"(3
water
O  0.0  0.0     0.1173
H  0.0  0.7572 -0.4692
H  0.0 -0.7572 -0.4692
)";

constexpr std::string_view kWater = R"(
[molecule]
xyz = "water.xyz"
[basis_set]
basis = "cc-pvdz"
)";

constexpr std::string_view kDensityFitted = R"(
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
)";

// The shape of the water config with @p tables after it, the geometry written
// where the config resolves it.
fb::PlanShape water_shape(fb::DeviceResources &res, std::string_view tables) {
  const fb::test::ScratchDir dir("pipeline_shape");
  EXPECT_TRUE(fb::write_file(dir / "water.xyz", kWaterXyz).has_value());
  auto calc = fb::parse_config(std::string(kWater) + std::string(tables), dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  auto shape = fb::plan_shape(*calc, res);
  EXPECT_TRUE(shape.has_value()) << shape.error().message;
  return std::move(*shape);
}

// Water in cc-pVDZ, as data/basis/cc-pVDZ.json carries it (a general
// contraction is one shell per column, zero coefficients kept): O has s shells
// of 9, 9 and 9 primitives, p of 4 and 4, and d of 1; each H s of 4 and 4,
// and p of 1. So nao = 3 + 2 * 3 + 5 + 2 * (2 + 3) = 24. Per momentum, the
// primitive sum S and square sum Q are s 43 and 307, p 10 and 34, d 1 and 1,
// and a class's records are (S^2 + Q) / 2 on the diagonal and S_a S_b off it:
// (s,s) 1078, (s,p) 430, (s,d) 43, (p,p) 67, (p,d) 10, (d,d) 1. 1629 in all.
// cc-pVDZ-JKFIT: O 70 functions, each H 23, so naux = 116. Ten electrons, a
// singlet: five doubly occupied.
TEST(PipelineShape, WaterDensityFittedMatchesHandCount) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape shape = water_shape(res, kDensityFitted);
  const fb::FootprintDimensions &dims = shape.dims;

  ASSERT_NE(dims.basis, nullptr);
  ASSERT_NE(dims.aux, nullptr);
  EXPECT_EQ(dims.basis, shape.basis.get());
  EXPECT_EQ(dims.aux, shape.aux.get());
  EXPECT_EQ(dims.nao, std::size_t{24});
  EXPECT_EQ(dims.basis->nao_sphe(), 24U);
  EXPECT_EQ(dims.naux, std::size_t{116});
  EXPECT_EQ(dims.aux->nao_sphe(), 116U);
  EXPECT_EQ(dims.nocc, std::size_t{5});
  EXPECT_EQ(dims.n_alpha, std::size_t{5});
  EXPECT_EQ(dims.n_beta, std::size_t{5});
  EXPECT_EQ(dims.natoms, std::size_t{3});
  EXPECT_EQ(dims.ncas, std::size_t{0});
  EXPECT_EQ(dims.nelecas, std::size_t{0});

  const std::vector<std::vector<std::size_t>> records{{1078, 430, 43}, {0, 67, 10}, {0, 0, 1}};
  EXPECT_EQ(dims.mmd_records, records);

  // The primitive-pair count, from the primitive totals alone: of P primitives
  // with Q the sum of each shell's square, (P^2 + Q) / 2 pairs within shells
  // i <= j.
  std::size_t p = 0;
  std::size_t q = 0;
  for (const fb::Shell &shell : dims.basis->shells()) {
    p += shell.nprimitives();
    q += std::size_t{shell.nprimitives()} * shell.nprimitives();
  }
  std::size_t total = 0;
  for (const std::vector<std::size_t> &row : dims.mmd_records) {
    total = std::accumulate(row.begin(), row.end(), total);
  }
  EXPECT_EQ(total, ((p * p) + q) / 2);
  EXPECT_EQ(total, std::size_t{1629});

  // mmd's own footprint, sized from the basis, against the hand count: 16
  // doubles and 4 int32_t a record.
  EXPECT_EQ(fb::primitive_pair_classes_footprint(*dims.basis).device.persistent,
            std::size_t{1629} * ((16 * sizeof(double)) + (4 * sizeof(std::int32_t))));

  // syevd's work by order, as diagonalize sizes its workspace from the same
  // query: through nao, the largest order an RHF plan diagonalizes at.
  ASSERT_EQ(dims.syevd_lwork.size(), std::size_t{25});
  for (std::int32_t order = 0; order <= 24; ++order) {
    auto lwork = fb::syevd_lwork<double>(res, order);
    ASSERT_TRUE(lwork.has_value()) << lwork.error().message;
    EXPECT_EQ(dims.syevd_lwork[static_cast<std::size_t>(order)], *lwork) << "order " << order;
  }
  EXPECT_GT(dims.syevd_lwork[24], std::size_t{0});
  EXPECT_EQ(dims.minao, nullptr);
  EXPECT_EQ(dims.atoms.size(), std::size_t{3});
}

// mmd's classes, built on the device from the shape's own basis, hold the
// record counts the shape gives, class for class.
TEST(PipelineShape, RecordCountsMatchThePrimitivePairClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape shape = water_shape(res, "");
  const fb::FootprintDimensions &dims = shape.dims;
  ASSERT_NE(dims.basis, nullptr);
  EXPECT_EQ(dims.aux, nullptr);
  EXPECT_EQ(dims.naux, std::size_t{0});

  auto classes = fb::primitive_pair_classes(res, *dims.basis);
  ASSERT_TRUE(classes.has_value()) << classes.error().message;
  std::size_t nonzero = 0;
  for (const std::vector<std::size_t> &row : dims.mmd_records) {
    nonzero += static_cast<std::size_t>(std::ranges::count_if(row, [](auto n) { return n > 0; }));
  }
  EXPECT_EQ(classes->size(), nonzero);
  for (const fb::PrimitivePairClass &cls : *classes) {
    const auto la = static_cast<std::size_t>(cls.la);
    const auto lb = static_cast<std::size_t>(cls.lb);
    ASSERT_LT(lb, dims.mmd_records.size());
    EXPECT_EQ(static_cast<std::size_t>(cls.n_rec), dims.mmd_records[la][lb])
        << "(" << la << ", " << lb << ")";
  }
}

// MINAO on water: O's shells 1s, 2s, 2p; each H's 1s. "O 2p" labels 3 AOs, so
// AVAS may pick 6; "H 1s" adds 2, so 10. avas_max_ncas below that is the
// bound. nelecas at half filling, ncas.
TEST(PipelineShape, AvasBoundIsTheCapOrTheLabelledAos) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto active = [&](std::string_view casscf) {
    const fb::PlanShape shape =
        water_shape(res, std::string(kDensityFitted) + "[casscf]\n" + std::string(casscf));
    return std::pair{shape.dims.ncas, shape.dims.nelecas};
  };
  using Space = std::pair<std::size_t, std::size_t>;
  EXPECT_EQ(active("avas = [\"O 2p\"]\n"), (Space{6, 6}));
  EXPECT_EQ(active("avas = [\"O 2p\"]\navas_max_ncas = 4\n"), (Space{4, 4}));
  EXPECT_EQ(active("avas = [\"O 2p\", \"H 1s\"]\n"), (Space{10, 10}));
  EXPECT_EQ(active("avas = [\"O 2p\", \"H 1s\"]\navas_max_ncas = 7\n"), (Space{7, 7}));
  EXPECT_EQ(active("ncas = 4\nnelecas = 2\n"), (Space{4, 2}));
}

// The orders syevd's work is queried through: a CI's Davidson subspace, 16
// nroots, past nao; a CASSCF's augmented Hessian, ah_max_subspace + 1; and
// under AVAS MINAO, placed, with its 7 AOs on water.
TEST(PipelineShape, EigensolveOrdersAndMinao) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape casci = water_shape(res, "[casci]\nncas = 4\nnelecas = 4\nnroots = 3\n");
  EXPECT_EQ(casci.dims.syevd_lwork.size(), std::size_t{49});
  EXPECT_EQ(casci.dims.minao, nullptr);
  auto lwork = fb::syevd_lwork<double>(res, 48);
  ASSERT_TRUE(lwork.has_value()) << lwork.error().message;
  EXPECT_EQ(casci.dims.syevd_lwork.back(), *lwork);
  const fb::PlanShape casscf =
      water_shape(res, std::string(kDensityFitted) + "[casscf]\navas = [\"O 2p\"]\n");
  EXPECT_EQ(casscf.dims.syevd_lwork.size(),
            static_cast<std::size_t>(fb::mcscf::Options{}.ah_max_subspace) + 2);
  ASSERT_NE(casscf.dims.minao, nullptr);
  EXPECT_EQ(casscf.dims.minao, casscf.minao.get());
  EXPECT_EQ(casscf.dims.minao->nao_sphe(), 7U);
}

// A CIS step's Davidson subspace, max(2 n_states, min(max_subspace, n_ov)):
// at the default 40 columns past water's nao of 24 (n_ov 95), at 8 under it.
// getrf's work at the RIS basis's order (water's Sp one: six functions) with
// ris, 0 without. And MINAO for a localization that builds IAOs (IBO, or the
// LIVVOs in either mode), not for Pipek-Mezey alone.
TEST(PipelineShape, CisAndLocalization) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape cis = water_shape(res, "[cis]\n");
  EXPECT_EQ(cis.dims.syevd_lwork.size(), std::size_t{41});
  EXPECT_EQ(cis.dims.getrf_lwork, 0U);
  EXPECT_EQ(cis.dims.minao, nullptr);
  const fb::PlanShape narrow = water_shape(res, "[cis]\nn_states = 3\nmax_subspace = 8\n");
  EXPECT_EQ(narrow.dims.syevd_lwork.size(), std::size_t{25});
  const fb::PlanShape ris = water_shape(res, "[cis]\nris = true\n");
  auto lwork = fb::getrf_lwork(res, 6);
  ASSERT_TRUE(lwork.has_value()) << lwork.error().message;
  EXPECT_GT(*lwork, 0U);
  EXPECT_EQ(ris.dims.getrf_lwork, *lwork);

  EXPECT_EQ(water_shape(res, "[scf]\n[localization]\n").dims.minao, nullptr);
  for (const std::string_view table :
       {"[scf]\n[localization]\nmode = \"ibo\"\n", "[scf]\n[localization]\nlivvo = true\n"}) {
    SCOPED_TRACE(table);
    const fb::PlanShape localization = water_shape(res, table);
    ASSERT_NE(localization.dims.minao, nullptr);
    EXPECT_EQ(localization.dims.minao, localization.minao.get());
    EXPECT_EQ(localization.dims.minao->nao_sphe(), 7U);
    EXPECT_EQ(localization.dims.syevd_lwork.size(), std::size_t{25});
  }
}

// The CASSCF step's roots, and whether two of its weights differ: none planned,
// none given (equal, one root and three), given equal, and given unequal. A CASCI's roots are not
// the CASSCF's.
TEST(PipelineShape, CasscfRootsAndWeights) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape none = water_shape(res, "[casci]\nncas = 4\nnelecas = 4\nnroots = 3\n");
  EXPECT_EQ(none.dims.casscf_nroots, 0U);
  EXPECT_FALSE(none.dims.casscf_unequal_weights);
  const std::string casscf = std::string(kDensityFitted) + "[casscf]\nncas = 4\nnelecas = 4\n";
  const fb::PlanShape one = water_shape(res, casscf);
  EXPECT_EQ(one.dims.casscf_nroots, 1U);
  EXPECT_FALSE(one.dims.casscf_unequal_weights);
  const fb::PlanShape equal = water_shape(res, casscf + "nroots = 2\nweights = [0.5, 0.5]\n");
  EXPECT_EQ(equal.dims.casscf_nroots, 2U);
  EXPECT_FALSE(equal.dims.casscf_unequal_weights);
  const fb::PlanShape three = water_shape(res, casscf + "nroots = 3\n");
  EXPECT_EQ(three.dims.casscf_nroots, 3U);
  EXPECT_FALSE(three.dims.casscf_unequal_weights);
  const fb::PlanShape unequal = water_shape(res, casscf + "nroots = 2\nweights = [0.7, 0.3]\n");
  EXPECT_EQ(unequal.dims.casscf_nroots, 2U);
  EXPECT_TRUE(unequal.dims.casscf_unequal_weights);
}

// An FCI beside a CASCI: the FCI's space, every orbital and electron, is the
// larger.
TEST(PipelineShape, FciIsTheLargestActiveSpace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape shape = water_shape(res, "[casci]\nncas = 4\nnelecas = 4\n[fci]\n");
  EXPECT_EQ(shape.dims.ncas, std::size_t{24});
  EXPECT_EQ(shape.dims.nelecas, std::size_t{10});
}

// Under an [optimize] the molecule is loaded from the final.xyz it writes,
// which does not exist yet: the shape reads the chain's start, the same atoms.
TEST(PipelineShape, OptimizeReadsTheChainsStart) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::PlanShape shape = water_shape(res, "[optimize]\n");
  EXPECT_EQ(shape.dims.natoms, std::size_t{3});
  EXPECT_EQ(shape.dims.nao, std::size_t{24});
  EXPECT_EQ(shape.dims.nocc, std::size_t{5});
}

} // namespace
