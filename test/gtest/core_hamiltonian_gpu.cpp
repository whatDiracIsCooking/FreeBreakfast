// GPU cases for integrals.one_electron's kinetic, potential_en and
// core_hamiltonian modules: the Cartesian T, V and H = T + V of H2O in STO-3G
// (s, p; O's p shell precedes the H s shells, so the la <= lb swap is
// exercised) and cc-pVDZ (s, p, d) against libcint's int1e_kin and int1e_nuc
// (point nuclei) and PySCF's hcore, recorded by PySCF
// (test/gtest/data/one_electron/, copied from FelizJueves's fixtures by
// tools/extract_integral_golden.py), plus, as overlap_gpu.cpp, the zeroing
// and the class-range refusal of each.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.io.json;
import free_breakfast.linalg.matrix_ao;
import free_breakfast.integrals;

namespace {

namespace fb = free_breakfast;

// FelizJueves holds its one-electron kernels to libcint at 1e-12 of scale
// (test/oracle/tolerance.py); the same bar here, against max|M| as
// overlap_gpu.cpp.
constexpr double kRelTol = 1e-12;

struct Golden {
  fb::BasisSet basis;
  std::vector<fb::Atom> atoms;
  std::vector<double> kin; // nao x nao, row-major; each is symmetric
  std::vector<double> nuc;
  std::vector<double> hcore;
};

Golden read_golden(const std::string &stem) {
  const auto doc =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "one_electron" / (stem + ".json"));
  EXPECT_TRUE(doc) << stem;
  if (!doc) {
    return {};
  }
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(),
                        fb::Vec<double, 3>{c[0], c[1], c[2]});
  }
  std::vector<fb::Atom> atoms;
  for (const auto &at : (*doc)["atoms"]) {
    const auto c = at["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{c[0], c[1], c[2]},
                             fb::Element{at["charge"].get<std::uint8_t>()}});
  }
  const auto &arrays = (*doc)["arrays"];
  Golden g{fb::BasisSet(std::move(shells)), std::move(atoms),
           arrays["int1e_kin"]["values"].get<std::vector<double>>(),
           arrays["int1e_nuc"]["values"].get<std::vector<double>>(),
           arrays["hcore"]["values"].get<std::vector<double>>()};
  EXPECT_EQ(g.basis.nao_cart(), (*doc)["nao"].get<std::uint32_t>()) << stem;
  return g;
}

std::vector<double> download(fb::DeviceResources &res, const fb::CartesianMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

// Which matrix a case builds.
enum class Build { Kinetic, Nuclear, Core };

// gtest prints a parameter with this, and gtest_discover_tests names each ctest
// entry by the printed GetParam(), so without it they read "4-byte object".
void PrintTo(const Build which, std::ostream *os) {
  switch (which) {
  case Build::Kinetic:
    *os << "Kinetic";
    return;
  case Build::Nuclear:
    *os << "Nuclear";
    return;
  case Build::Core:
    *os << "Core";
    return;
  }
}

// The matrix on the device for g: records, upload, launch.
std::vector<double> device_build(fb::DeviceResources &res, const Golden &g, const Build which) {
  const auto device = fb::primitive_pair_classes(res, g.basis);
  EXPECT_TRUE(device);
  const fb::Molecule molecule(res, g.atoms);
  fb::CartesianMatrixAO m(res, g.basis);
  fb::Status status;
  switch (which) {
  case Build::Kinetic:
    status = fb::kinetic(res, *device, m);
    break;
  case Build::Nuclear:
    status = fb::nuclear_attraction(res, *device, molecule, m);
    break;
  case Build::Core:
    status = fb::core_hamiltonian(res, *device, molecule, m);
    break;
  }
  EXPECT_TRUE(status) << (status ? "" : status.error().message);
  return download(res, m);
}

// got against want, element by element and against its own mirror, at
// kRelTol * max|want|; the worst absolute error is recorded.
void expect_matches(const std::vector<double> &got, const std::vector<double> &want,
                    const std::size_t n, const std::string_view name) {
  ASSERT_EQ(got.size(), want.size());
  double scale = 0.0;
  for (const double v : want) {
    scale = std::max(scale, std::abs(v));
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double err = std::abs(got[i * n + j] - want[i * n + j]);
      worst = std::max(worst, err);
      EXPECT_LE(err, kRelTol * scale) << name << "(" << i << ", " << j << ")";
      // The atomics add each mirror pair in their own order, so symmetric to
      // rounding, not bitwise.
      EXPECT_LE(std::abs(got[i * n + j] - got[j * n + i]), kRelTol * scale);
    }
  }
  ::testing::Test::RecordProperty("worst_abs_error", std::format("{:.3e}", worst));
}

class OneElectronGolden : public ::testing::TestWithParam<const char *> {};

TEST_P(OneElectronGolden, KineticMatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.kin.empty());
  fb::DeviceResources res{0};
  expect_matches(device_build(res, g, Build::Kinetic), g.kin, g.basis.nao_cart(), "T");
}

TEST_P(OneElectronGolden, NuclearAttractionMatchesLibcint) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.nuc.empty());
  fb::DeviceResources res{0};
  expect_matches(device_build(res, g, Build::Nuclear), g.nuc, g.basis.nao_cart(), "V");
}

TEST_P(OneElectronGolden, CoreHamiltonianMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const Golden g = read_golden(GetParam());
  ASSERT_FALSE(g.hcore.empty());
  fb::DeviceResources res{0};
  expect_matches(device_build(res, g, Build::Core), g.hcore, g.basis.nao_cart(), "H");
}

INSTANTIATE_TEST_SUITE_P(H2O, OneElectronGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));
// Five H2 4 bohr apart in cc-pVDZ, with a converged RHF density.
INSTANTIATE_TEST_SUITE_P(Chain, OneElectronGolden, ::testing::Values("h10_chain_ccpvdz"));

// One shell of momentum l at the origin, and one hydrogen there: the smallest
// basis and molecule a build takes.
fb::BasisSet one_shell(const std::int32_t l) {
  return fb::BasisSet({fb::Shell(l, {{1.0, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 0.0})});
}

fb::Molecule one_hydrogen(fb::DeviceResources &res) {
  return fb::Molecule(res, {fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::H}});
}

class OneElectronBuild : public ::testing::TestWithParam<Build> {};

TEST_P(OneElectronBuild, ZeroesTheMatrixFirst) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = one_shell(0);
  const fb::Molecule molecule = one_hydrogen(res);
  fb::CartesianMatrixAO m(res, basis);
  const double stale = 42.0;
  ASSERT_EQ(wwr::wwrMemcpyAsync(m.data(), &stale, sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  // No classes: the only work left is the zeroing.
  fb::Status status;
  switch (GetParam()) {
  case Build::Kinetic:
    status = fb::kinetic(res, {}, m);
    break;
  case Build::Nuclear:
    status = fb::nuclear_attraction(res, {}, molecule, m);
    break;
  case Build::Core:
    status = fb::core_hamiltonian(res, {}, molecule, m);
    break;
  }
  ASSERT_TRUE(status);
  EXPECT_EQ(download(res, m)[0], 0.0);
}

TEST_P(OneElectronBuild, RefusesAClassAboveD) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::BasisSet basis = one_shell(3);
  const auto device = fb::primitive_pair_classes(res, basis);
  ASSERT_TRUE(device);
  const fb::Molecule molecule = one_hydrogen(res);
  fb::CartesianMatrixAO m(res, basis);
  fb::Status status;
  switch (GetParam()) {
  case Build::Kinetic:
    status = fb::kinetic(res, *device, m);
    break;
  case Build::Nuclear:
    status = fb::nuclear_attraction(res, *device, molecule, m);
    break;
  case Build::Core:
    status = fb::core_hamiltonian(res, *device, molecule, m);
    break;
  }
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
}

INSTANTIATE_TEST_SUITE_P(TVH, OneElectronBuild,
                         ::testing::Values(Build::Kinetic, Build::Nuclear, Build::Core));

} // namespace
