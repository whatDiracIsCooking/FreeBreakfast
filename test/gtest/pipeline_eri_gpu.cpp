// The Eri step. Density-fitted: its B for FelizJueves's water in cc-pVDZ /
// cc-pVDZ-JKFIT and cc-pVTZ / cc-pVTZ-JKFIT (f orbitals, g fitting functions)
// against the B density_fit makes of PySCF's int2c2e and int3c2e
// (data/density_fitting/), the default aux basis, and the step's refusals of
// an aux basis it cannot resolve, place or default. Integral-direct: J's and
// K's classes, and an aux basis refused. The step's contract is checked by
// static_assert; the rest builds on the device, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.primitive_pair_classes;
import free_breakfast.pipeline.eri;

namespace {

namespace fb = free_breakfast;

using EriInput = fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>;

static_assert(fb::PipelineStep<fb::EriStep>);
static_assert(fb::kRunShape<fb::EriStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::EriStep>, EriInput>);
static_assert(std::same_as<fb::StepOutput<fb::EriStep>, fb::EriOutput>);
static_assert(!fb::LoadableStep<fb::EriStep>);
static_assert(fb::EriStep{}.mode == fb::EriMode::IntegralDirect);

// The integrals each agree with PySCF to 1e-12 of their own maximum
// (df_metric_gpu, df_three_center_gpu), and so does the fitted (mn|ls) = B^T B
// they make. B itself is not held to it: its last rows, the metric's smallest
// pivots, carry the integrals' differences amplified by L's conditioning
// (5.5e-12 of max|B| in DZ, 9.8e-12 in TZ, at the last aux functions).
constexpr double kRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "density_fitting";

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host,
            fb::Buffers::device<double> &d) {
  ASSERT_EQ(host.size(), d.num_elements());
  ASSERT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_TRUE(res.sync().has_value());
}

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held.
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : shell.primitives()) {
    for (const fb::Primitive &q : shell.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

// Each spherical function's norm N_i = sqrt(<i|i>): function i here is N_i
// times PySCF's unit-normalised one (test/gtest/data/README.md).
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

const fb::json *find_case(const fb::json &doc, const std::string &name) {
  for (const fb::json &entry : doc["integrals"]) {
    if (entry["name"].get<std::string>() == name) {
      return &entry;
    }
  }
  return nullptr;
}

fb::MoleculeOutput molecule_of(fb::DeviceResources &res, const fb::json &c) {
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  return fb::MoleculeOutput{fb::Molecule(res, std::move(atoms))};
}

// B as density_fit makes it of PySCF's int2c2e and int3c2e, so over PySCF's
// unit-normalised orbital and fitting functions: naux x nao^2, [Q][n][m].
std::vector<double> golden_b(fb::DeviceResources &res, const fb::json &c) {
  const auto nao = c["nao"].get<std::size_t>();
  const auto naux = c["naux"].get<std::size_t>();
  const auto metric = fb::read_npy(kDataDir / c["int2c2e"].get<std::string>());
  EXPECT_TRUE(metric) << metric.error().message;
  if (!metric) {
    return {};
  }
  EXPECT_EQ(metric->shape, (std::vector<std::size_t>{naux, naux}));

  // Each piece is (nao, nao, end - begin), Fortran order: (mn|P) at
  // m + nao (n + nao (P - begin)). Gathered into the fit's P + naux (n + nao m).
  std::vector<double> b(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
  std::size_t covered = 0;
  for (const auto &piece : c["int3c2e"]) {
    const auto range = piece["aux"].get<std::vector<std::size_t>>();
    const auto slab = fb::read_npy(kDataDir / piece["file"].get<std::string>());
    EXPECT_TRUE(slab) << slab.error().message;
    if (!slab) {
      return {};
    }
    EXPECT_EQ(slab->shape, (std::vector<std::size_t>{nao, nao, range[1] - range[0]}));
    EXPECT_EQ(range[0], covered);
    for (std::size_t p = range[0]; p < range[1]; ++p) {
      for (std::size_t n = 0; n < nao; ++n) {
        for (std::size_t m = 0; m < nao; ++m) {
          b[p + (naux * (n + (nao * m)))] = slab->data[m + (nao * (n + (nao * (p - range[0]))))];
        }
      }
    }
    covered = range[1];
  }
  EXPECT_EQ(covered, naux);

  fb::Buffers::device<double> d_metric(naux * naux, res.device());
  fb::Buffers::device<double> d_b(b.size(), res.device());
  upload(res, metric->data, d_metric);
  upload(res, b, d_b);
  const fb::Status fit =
      fb::density_fit(res, static_cast<std::int32_t>(nao), static_cast<std::int32_t>(naux),
                      d_metric.data(), d_b.data());
  EXPECT_TRUE(fit) << fit.error().message;
  return download(res, d_b.data(), d_b.num_elements());
}

// The case's molecule and orbital basis through their steps, then the step's
// B against the golden-built one, through the fitted (mn|ls) = B^T B of each.
// The step's functions are N_i times PySCF's. With D the fitting norms, the
// step's metric is D M D, so its L is D L_py and its B = L_py^-1 D^-1 D
// (P|mn)_py N_m N_n = B_py N_m N_n: the fitting norms cancel exactly, and only
// the two orbital ones are divided out before the product is formed.
void check_b(const std::string &name) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "density_fitting.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = find_case(*doc, name);
  ASSERT_NE(c, nullptr) << "no integrals case " << name;

  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = molecule_of(res, *c);
  auto basis = fb::BasisSetStep{.basis = (*c)["basis"].get<std::string>()}.run(
      fb::View<fb::MoleculeOutput>{molecule});
  ASSERT_TRUE(basis) << basis.error().message;
  auto classes = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(classes) << classes.error().message;
  auto eri = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                         .aux_basis = (*c)["aux_basis"].get<std::string>()}
                 .run(EriInput{molecule, *basis, *classes}, res);
  ASSERT_TRUE(eri) << eri.error().message;
  const auto *fit = std::get_if<fb::DensityFitted>(&eri->eri);
  ASSERT_NE(fit, nullptr);

  const auto nao = (*c)["nao"].get<std::size_t>();
  const auto naux = (*c)["naux"].get<std::size_t>();
  ASSERT_EQ(basis->basis.nao_sphe(), nao);
  ASSERT_EQ(fit->naux, static_cast<std::int32_t>(naux));
  ASSERT_EQ(fit->b.num_elements(), naux * nao * nao);
  const std::vector<double> ours = download(res, fit->b.data(), fit->b.num_elements());
  const std::vector<double> golden = golden_b(res, *c);
  ASSERT_EQ(golden.size(), ours.size());

  // Over PySCF's functions, as the golden B is.
  std::vector<double> rescaled = ours;
  const std::vector<double> n_ao = norms(basis->basis);
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t q = 0; q < naux; ++q) {
        rescaled[q + (naux * (n + (nao * m)))] /= n_ao[m] * n_ao[n];
      }
    }
  }
  double b_scale = 0.0;
  double b_worst = 0.0;
  for (std::size_t i = 0; i < golden.size(); ++i) {
    b_scale = std::max(b_scale, std::abs(golden[i]));
    b_worst = std::max(b_worst, std::abs(rescaled[i] - golden[i]));
  }

  // The fitted (mn|ls) = sum_Q B_Q,mn B_Q,ls, over every m <= n and l <= s.
  std::vector<std::size_t> pairs;
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t n = m; n < nao; ++n) {
      pairs.push_back(n + (nao * m));
    }
  }
  const auto fitted = [&](const std::vector<double> &b, const std::size_t mn,
                          const std::size_t ls) {
    double sum = 0.0;
    for (std::size_t q = 0; q < naux; ++q) {
      sum += b[q + (naux * mn)] * b[q + (naux * ls)];
    }
    return sum;
  };
  double scale = 0.0;
  double worst = 0.0;
  std::array<std::size_t, 2> at{};
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    for (std::size_t j = i; j < pairs.size(); ++j) {
      const double g = fitted(golden, pairs[i], pairs[j]);
      scale = std::max(scale, std::abs(g));
      const double d = std::abs(fitted(rescaled, pairs[i], pairs[j]) - g);
      if (!(d <= worst)) {
        worst = d;
        at = {pairs[i], pairs[j]};
      }
    }
  }
  EXPECT_LE(worst, kRelTol * scale)
      << name << ": worst |B^T B - B_golden^T B_golden| " << worst << " at (" << at[0] % nao << " "
      << at[0] / nao << " | " << at[1] % nao << " " << at[1] / nao << "), max " << scale;
  std::cout << std::format("[ df step  ] {}: nao {}, naux {}, max|(mn|ls)| {:.4f}, worst "
                           "|B^T B - golden| / max {:.3e}; max|B| {:.4f}, worst |B - golden| / "
                           "max {:.3e}\n",
                           name, nao, naux, scale, worst / scale, b_scale, b_worst / b_scale);
}

TEST(PipelineEri, DzMatchesTheGoldenBuiltB) {
  check_b("h2o_ccpvdz");
}

TEST(PipelineEri, TzWithFOrbitalsMatchesTheGoldenBuiltB) {
  check_b("h2o_ccpvtz");
}

// The molecule, a compiled-in basis over it and its pair classes, for the
// cases that need only the step's input.
struct Water {
  fb::MoleculeOutput molecule;
  fb::BasisSetOutput basis;
  fb::PrimitivePairClassesOutput classes;

  [[nodiscard]] EriInput view() const { return EriInput{molecule, basis, classes}; }
};

std::optional<Water> water(fb::DeviceResources &res, const std::string &basis_name) {
  fb::MoleculeOutput molecule{
      fb::Molecule(res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.221665}, fb::Element::O},
                         fb::Atom{fb::Vec<double, 3>{0.0, 1.430901, -0.886659}, fb::Element::H},
                         fb::Atom{fb::Vec<double, 3>{0.0, -1.430901, -0.886659}, fb::Element::H}})};
  auto basis = fb::BasisSetStep{.basis = basis_name}.run(fb::View<fb::MoleculeOutput>{molecule});
  if (!basis) {
    ADD_FAILURE() << basis.error().message;
    return std::nullopt;
  }
  auto classes = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  if (!classes) {
    ADD_FAILURE() << classes.error().message;
    return std::nullopt;
  }
  return Water{std::move(molecule), std::move(*basis), std::move(*classes)};
}

// The default mode: J's and K's classes, one of each per class J and K can
// need, and no B.
TEST(PipelineEri, IntegralDirectHoldsJsAndKsClasses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto in = water(res, "cc-pvdz");
  ASSERT_TRUE(in);
  auto eri = fb::EriStep{}.run(in->view(), res);
  ASSERT_TRUE(eri) << eri.error().message;
  const auto *direct = std::get_if<fb::IntegralDirect>(&eri->eri);
  ASSERT_NE(direct, nullptr);
  auto j_classes = fb::coulomb_pair_classes(res, in->classes.classes);
  auto k_classes = fb::ordered_pair_classes(res, in->basis.basis, in->classes.classes);
  ASSERT_TRUE(j_classes && k_classes);
  EXPECT_FALSE(direct->j_classes.empty());
  EXPECT_FALSE(direct->k_classes.empty());
  EXPECT_EQ(direct->j_classes.size(), j_classes->size());
  EXPECT_EQ(direct->k_classes.size(), k_classes->size());
}

TEST(PipelineEri, AnAuxBasisUnderIntegralDirectIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto in = water(res, "cc-pvdz");
  ASSERT_TRUE(in);
  auto eri = fb::EriStep{.aux_basis = "cc-pvdz-jkfit"}.run(in->view(), res);
  ASSERT_FALSE(eri);
  EXPECT_EQ(eri.error().kind, fb::ErrorKind::InvalidConfig) << eri.error().message;
}

// No aux_basis: the orbital set's acCD-6-6, the same B as naming it.
TEST(PipelineEri, DensityFittingDefaultsToTheBasissAcCdSet) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto in = water(res, "cc-pvdz");
  ASSERT_TRUE(in);
  auto by_default = fb::EriStep{.mode = fb::EriMode::DensityFitting}.run(in->view(), res);
  auto named =
      fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pVDZ-acCD-6-6"}.run(
          in->view(), res);
  auto jkfit = fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pVDZ-JKFIT"}.run(
      in->view(), res);
  ASSERT_TRUE(by_default) << by_default.error().message;
  ASSERT_TRUE(named) << named.error().message;
  ASSERT_TRUE(jkfit) << jkfit.error().message;
  const auto &b_default = std::get<fb::DensityFitted>(by_default->eri);
  const auto &b_named = std::get<fb::DensityFitted>(named->eri);
  ASSERT_EQ(b_default.naux, b_named.naux);
  EXPECT_NE(b_default.naux, std::get<fb::DensityFitted>(jkfit->eri).naux);
  const std::vector<double> ours = download(res, b_default.b.data(), b_default.b.num_elements());
  const std::vector<double> theirs = download(res, b_named.b.data(), b_named.b.num_elements());
  ASSERT_EQ(ours.size(), theirs.size());
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < ours.size(); ++i) {
    scale = std::max(scale, std::abs(theirs[i]));
    worst = std::max(worst, std::abs(ours[i] - theirs[i]));
  }
  EXPECT_LE(worst, kRelTol * scale);
}

// A basis with no carried acCD-6-6 set, or none known to be a carried set at
// all, has no default: the step asks for aux_basis.
TEST(PipelineEri, DensityFittingWithNoDefaultAuxBasisIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto in = water(res, "sto-3g");
  ASSERT_TRUE(in);
  auto eri = fb::EriStep{.mode = fb::EriMode::DensityFitting}.run(in->view(), res);
  ASSERT_FALSE(eri);
  EXPECT_EQ(eri.error().kind, fb::ErrorKind::InvalidConfig) << eri.error().message;
  EXPECT_NE(eri.error().message.find("STO-3G"), std::string::npos) << eri.error().message;
  EXPECT_NE(eri.error().message.find("aux_basis"), std::string::npos) << eri.error().message;

  // A path-read basis is unnamed, whatever its document claims: the carried
  // cc-pVDZ, read from its file, gets no default.
  const auto from_file = water(
      res,
      (std::filesystem::path{FBF_TEST_DATA_DIR} / "../../../data/basis/cc-pVDZ.json").string());
  ASSERT_TRUE(from_file);
  ASSERT_TRUE(from_file->basis.basis.name().empty());
  auto unnamed = fb::EriStep{.mode = fb::EriMode::DensityFitting}.run(from_file->view(), res);
  ASSERT_FALSE(unnamed);
  EXPECT_EQ(unnamed.error().kind, fb::ErrorKind::InvalidConfig) << unnamed.error().message;
}

// An aux basis the step cannot resolve, or cannot place over the molecule, is
// the step's error, before anything is built.
TEST(PipelineEri, AnAuxBasisItCannotResolveOrPlaceIsItsError) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // cc-pVDZ carries helium; neither fitting set does.
  const fb::MoleculeOutput helium{
      fb::Molecule(res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::He}})};
  auto basis = fb::BasisSetStep{.basis = "cc-pvdz"}.run(fb::View<fb::MoleculeOutput>{helium});
  ASSERT_TRUE(basis) << basis.error().message;
  auto classes = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{*basis}, res);
  ASSERT_TRUE(classes) << classes.error().message;
  const EriInput in{helium, *basis, *classes};

  auto unplaced =
      fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"}.run(in, res);
  ASSERT_FALSE(unplaced);
  EXPECT_EQ(unplaced.error().kind, fb::ErrorKind::InvalidConfig) << unplaced.error().message;

  // Neither a fitting set's name nor a file: read as a path, and the
  // compiled-in names offered.
  auto unknown =
      fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvqz-jkfit"}.run(in, res);
  ASSERT_FALSE(unknown);
  EXPECT_EQ(unknown.error().kind, fb::ErrorKind::IO) << unknown.error().message;
  EXPECT_NE(unknown.error().message.find("cc-pVDZ-JKFIT"), std::string::npos)
      << unknown.error().message;
}

} // namespace
