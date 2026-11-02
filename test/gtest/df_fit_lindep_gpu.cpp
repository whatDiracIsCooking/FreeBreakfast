// GPU cases for density_fit's lindep path (#630): the fit over the metric's
// eigenmodes above a threshold, B = M^-1/2 (P|mn). lindep = 0 is the Cholesky
// path bit for bit; on a well-conditioned metric (H2O, cc-pVDZ /
// cc-pVDZ-JKFIT) B^T B and the DF-RHF energy are the Cholesky path's; on a
// singular one (the same set with three O shells duplicated,
// data/df_lindep/), the default fit refuses and the lindep fit's DF-RHF
// energy is PySCF's eigen-route DF-RHF at the same threshold, and the plain
// set's, whose space it spans.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.molecule;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.spherical_transform;
import free_breakfast.pipeline.primitive_pair_classes;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.scf;

namespace {

namespace fb = free_breakfast;

// The issue's bar for a metric that drops nothing: B^T B, which J and K are
// linear in, against the Cholesky path's to 1e-12 of its largest element.
constexpr double kRelTol = 1e-12;
// The two paths' converged DF-RHF energies: each SCF is converged to 1e-12 Eh
// from the same guess over fits that agree to rounding.
constexpr double kSamePathTolerance = 1e-11;
// Against PySCF's energy, as pipeline_df_scf_gpu.cpp holds it.
constexpr double kEnergyTolerance = 1e-8;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "df_lindep";

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(kDataDir / "df_lindep.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return doc;
}

// The setup steps for the reference's water in its orbital basis: everything
// the Scf step reads but the ERI.
struct Setup {
  fb::MoleculeOutput molecule;
  fb::BasisSetOutput basis;
  fb::SphericalTransformOutput transform;
  fb::PrimitivePairClassesOutput pairs;
  fb::OverlapOutput overlap;
  fb::OrthogonalizerOutput x;
  fb::CoreHamiltonianOutput h;
  fb::GuessOutput guess;
};

std::optional<Setup> set_up(fb::DeviceResources &res) {
  std::vector<fb::Atom> atoms;
  for (const fb::json &atom : reference().at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    atoms.push_back(fb::Atom{{coord[0], coord[1], coord[2]},
                             static_cast<fb::Element>(atom.at("z").get<std::int32_t>())});
  }
  fb::MoleculeOutput molecule{fb::Molecule(res, std::move(atoms), 0)};
  auto basis = fb::BasisSetStep{.basis = reference().at("basis").get<std::string>()}.run(
      fb::View<fb::MoleculeOutput>{molecule});
  if (!basis.has_value()) {
    ADD_FAILURE() << basis.error().message;
    return std::nullopt;
  }
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  if (!transform.has_value() || !pairs.has_value()) {
    ADD_FAILURE() << "spherical transform or pair classes failed";
    return std::nullopt;
  }
  auto overlap = fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          *basis, *transform, *pairs},
      res);
  if (!overlap.has_value()) {
    ADD_FAILURE() << overlap.error().message;
    return std::nullopt;
  }
  auto x = fb::OrthogonalizerStep{}.run(
      fb::View<fb::BasisSetOutput, fb::OverlapOutput>{*basis, *overlap}, res);
  auto h = fb::CoreHamiltonianStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
               fb::PrimitivePairClassesOutput>{molecule, *basis, *transform, *pairs},
      res);
  if (!x.has_value() || !h.has_value()) {
    ADD_FAILURE() << "orthogonalizer or core Hamiltonian failed";
    return std::nullopt;
  }
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::GuessOutput> guess = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) {
    guess = fb::GuessStep{}.run(fb::View<fb::MoleculeOutput, fb::BasisSetOutput>{molecule, *basis},
                                subflow);
  });
  executor.run(flow).wait();
  if (!guess.has_value()) {
    ADD_FAILURE() << guess.error().message;
    return std::nullopt;
  }
  return Setup{std::move(molecule), std::move(*basis), std::move(*transform), std::move(*pairs),
               std::move(*overlap), std::move(*x),     std::move(*h),         std::move(*guess)};
}

// The metric and (P|mn) over @p aux_spec (a carried name or a path), as the
// Eri step builds them, both still to be fitted.
struct Unfitted {
  fb::BasisSet aux;
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  fb::Buffers::device<double> metric;
  fb::Buffers::device<double> b;
};

std::optional<Unfitted> unfitted(fb::DeviceResources &res, const Setup &s,
                                 const std::string &aux_spec) {
  const auto stamp = fb::resolve_aux_basis(aux_spec);
  if (!stamp.has_value()) {
    ADD_FAILURE() << stamp.error().message;
    return std::nullopt;
  }
  auto aux = fb::place(stamp->primitive_normalized(), s.molecule.molecule);
  auto u = fb::upload_spherical_transform_matrix(res, *aux);
  if (!aux.has_value() || !u.has_value()) {
    ADD_FAILURE() << "aux basis placement or its transform failed";
    return std::nullopt;
  }
  const std::size_t nao = s.basis.basis.nao_sphe();
  const std::size_t naux = aux->nao_sphe();
  fb::MatrixAOPair m(res, *aux, std::move(*u));
  fb::Buffers::device<double> b(naux * nao * nao, res.device());
  if (const fb::Status st = fb::metric(res, *aux, m); !st.has_value()) {
    ADD_FAILURE() << st.error().message;
    return std::nullopt;
  }
  if (const fb::Status st = fb::three_center(res, s.basis.basis, s.pairs.classes, *aux, b.data());
      !st.has_value()) {
    ADD_FAILURE() << st.error().message;
    return std::nullopt;
  }
  fb::Buffers::device<double> metric(naux * naux, res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(metric.data(), m.spherical().data(), sizeof(double) * naux * naux,
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return Unfitted{std::move(*aux), static_cast<std::int32_t>(nao), static_cast<std::int32_t>(naux),
                  std::move(metric), std::move(b)};
}

fb::Buffers::device<double> copy_of(fb::DeviceResources &res,
                                    const fb::Buffers::device<double> &d) {
  fb::Buffers::device<double> out(d.num_elements(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), d.data(), sizeof(double) * d.num_elements(),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  return out;
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The converged DF-RHF energy over the fitted B. The Scf step reads only B
// and naux; L is a placeholder, since a lindep fit has none.
std::optional<double> df_rhf_energy(fb::DeviceResources &res, const Setup &s, const Unfitted &u,
                                    fb::Buffers::device<double> b) {
  fb::EriOutput eri{fb::DensityFitted{.b = std::move(b),
                                      .naux = u.naux,
                                      .aux = u.aux,
                                      .l = fb::Buffers::device<double>(1, res.device())}};
  const fb::ScfStep step{.energy_tolerance = 1e-12, .commutator_tolerance = 1e-10};
  auto scf = step.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput, fb::EriOutput,
               fb::OverlapOutput, fb::OrthogonalizerOutput, fb::CoreHamiltonianOutput,
               fb::GuessOutput>{s.molecule, s.basis, s.transform, eri, s.overlap, s.x, s.h,
                                s.guess},
      res);
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return std::nullopt;
  }
  return scf->energy;
}

TEST(DfFitLindep, ZeroIsTheCholeskyPathBitwise) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto s = set_up(res);
  ASSERT_TRUE(s.has_value());
  auto u = unfitted(res, *s, reference().at("plain_aux_basis").get<std::string>());
  ASSERT_TRUE(u.has_value());
  auto metric = copy_of(res, u->metric);
  auto b = copy_of(res, u->b);
  ASSERT_TRUE(fb::density_fit(res, u->nao, u->naux, u->metric.data(), u->b.data()));
  ASSERT_TRUE(fb::density_fit(res, u->nao, u->naux, metric.data(), b.data(), 0.0));
  EXPECT_EQ(download(res, b), download(res, u->b));
  EXPECT_EQ(download(res, metric), download(res, u->metric));
}

TEST(DfFitLindep, WellConditionedMetricFitsAsCholeskyDoes) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto s = set_up(res);
  ASSERT_TRUE(s.has_value());
  auto u = unfitted(res, *s, reference().at("plain_aux_basis").get<std::string>());
  ASSERT_TRUE(u.has_value());
  auto metric = copy_of(res, u->metric);
  auto b = copy_of(res, u->b);
  const double lindep = reference().at("lindep").get<double>();
  ASSERT_TRUE(fb::density_fit(res, u->nao, u->naux, u->metric.data(), u->b.data()));
  const fb::Status fitted = fb::density_fit(res, u->nao, u->naux, metric.data(), b.data(), lindep);
  ASSERT_TRUE(fitted) << fitted.error().message;

  // B^T B over every pair of columns (mn) <= (ls).
  const std::vector<double> chol = download(res, u->b);
  const std::vector<double> eig = download(res, b);
  const auto naux = static_cast<std::size_t>(u->naux);
  const std::size_t pairs = chol.size() / naux;
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t c2 = 0; c2 < pairs; ++c2) {
    for (std::size_t c1 = 0; c1 <= c2; ++c1) {
      double want = 0.0;
      double got = 0.0;
      for (std::size_t q = 0; q < naux; ++q) {
        want += chol[q + (naux * c1)] * chol[q + (naux * c2)];
        got += eig[q + (naux * c1)] * eig[q + (naux * c2)];
      }
      scale = std::max(scale, std::abs(want));
      worst = std::max(worst, std::abs(got - want));
    }
  }
  ASSERT_GT(scale, 0.0);
  EXPECT_LE(worst, kRelTol * scale) << "max|B^T B (lindep) - B^T B (Cholesky)| = " << worst << ", "
                                    << worst / scale << " of max " << scale;

  const auto e_chol = df_rhf_energy(res, *s, *u, std::move(u->b));
  const auto e_eig = df_rhf_energy(res, *s, *u, std::move(b));
  ASSERT_TRUE(e_chol.has_value() && e_eig.has_value());
  EXPECT_NEAR(*e_eig, *e_chol, kSamePathTolerance);
  EXPECT_NEAR(*e_eig, reference().at("plain_energy").get<double>(), kEnergyTolerance);
  std::cout << std::format("[ dflindep ] well conditioned: B^T B {:.2e} rel, energy {:.2e} Eh\n",
                           worst / scale, std::abs(*e_eig - *e_chol));
}

TEST(DfFitLindep, DuplicatedAuxShellsAreRefusedByDefaultAndFittedWithLindep) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto s = set_up(res);
  ASSERT_TRUE(s.has_value());
  const std::string aux = (kDataDir / reference().at("aux_basis").get<std::string>()).string();
  auto u = unfitted(res, *s, aux);
  ASSERT_TRUE(u.has_value());
  ASSERT_EQ(u->naux, reference().at("naux").get<std::int32_t>());
  auto metric = copy_of(res, u->metric);
  auto b = copy_of(res, u->b);

  const fb::Status refused = fb::density_fit(res, u->nao, u->naux, u->metric.data(), u->b.data());
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << refused.error().message;

  const fb::Status fitted = fb::density_fit(res, u->nao, u->naux, metric.data(), b.data(),
                                            reference().at("lindep").get<double>());
  ASSERT_TRUE(fitted) << fitted.error().message;
  const auto energy = df_rhf_energy(res, *s, *u, std::move(b));
  ASSERT_TRUE(energy.has_value());
  EXPECT_NEAR(*energy, reference().at("energy").get<double>(), kEnergyTolerance);
  EXPECT_NEAR(*energy, reference().at("plain_energy").get<double>(), kEnergyTolerance);
  std::cout << std::format("[ dflindep ] duplicated shells: energy {:.2e} Eh from PySCF's\n",
                           std::abs(*energy - reference().at("energy").get<double>()));
}

TEST(DfFitLindep, NegativeOrNanLindepIsInvalid) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const double lindep : {-1e-7, std::numeric_limits<double>::quiet_NaN()}) {
    const fb::Status st = fb::density_fit(res, 5, 5, nullptr, nullptr, lindep);
    ASSERT_FALSE(st);
    EXPECT_EQ(st.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

} // namespace
