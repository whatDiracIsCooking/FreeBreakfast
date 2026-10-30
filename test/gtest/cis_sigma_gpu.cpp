// GPU cases for cis.sigma: <b1, A b2> = <b2, A b1> for both spins on H2O /
// cc-pVDZ with synthetic orbitals and orbital energies (d shells, so the
// spherical bridge is exercised); a block equal to its columns applied one at
// a time; a second apply that allocates nothing of its own; the same three
// for the density-fitted cis::DfSigma over a cc-pVDZ-JKFIT fit, and its
// refusals (its physics, DF against PySCF's DF, is cis_step_gpu.cpp's); and
// the physics
// over the engine's own RHF, against data/cis/cis.json
// (tools/emit/cis_reference.py) to the issue's 1e-5 hartree: H2O / STO-3G's
// singlets and triplets from the full A, assembled column by column and
// diagonalized, and HF / cc-pVTZ's singlets (f shells, a degenerate Π pair)
// from davidson_solve over the sigma, since its full A is past ctest's
// timeout.
//
// The setup is mcscf_solve_gpu.cpp's: the reference's molecule through the
// real registry, then the Eri and Scf steps by hand (the SCF with J
// unscreened), so this executable links pipeline; cis itself does not.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.recipes;
import free_breakfast.cis.precondition;
import free_breakfast.cis.sigma;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace cis = free_breakfast::cis;

// The issue's bar against cis.json's PySCF roots, which are good to ~1e-14.
constexpr double kEnergyTolerance = 1e-5;
// <b1, A b2> − <b2, A b1>, relative to the largest |σ| element times n_ov,
// and a block's columns against the same columns applied alone, relative to
// the largest |σ| element. Not bitwise: J sums in its atomics' order, and the screen is per density.
constexpr double kOperatorTolerance = 1e-12;
// The assembled A's largest |A_ij - A_ji|, relative to max|A|: J's default
// screen (nonsymmetric_jk's README: ~1e-11 of max|G| on a dense D) is per
// density, so it does not cancel between column j's (ia|jb) and column i's.
// Measured 1e-15 on H2O / STO-3G, and 1.4e-11 on HF / cc-pVTZ's full A
// (assembled once, the same on both cards; too slow to keep).
constexpr double kAssembledSymmetryTolerance = 1e-10;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "cis" / "cis.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in cis/cis.json";
  static const fb::json kNone;
  return kNone;
}

std::vector<double> random_vector(std::size_t n, std::uint32_t seed, double lo, double hi) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{lo, hi};
  std::vector<double> v(n);
  for (auto &x : v) {
    x = dist(gen);
  }
  return v;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *device) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(device, host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

fb::Buffers::device<double> to_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  upload(res, host, d.data());
  return d;
}

std::vector<double> to_host(fb::DeviceResources &res, const double *d, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

// The reference's molecule through the setup steps, the J and K classes, and,
// when with_scf, the engine's RHF.
struct Setup {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  const fb::BasisSet *basis = nullptr;
  const fb::SphericalTransformOutput *transform = nullptr;
  std::vector<fb::CoulombPairClass> j_classes;
  std::vector<fb::OrderedPairClass> k_classes;
  std::optional<fb::DensityFitted> fit;
  std::optional<fb::ScfOutput> scf;

  [[nodiscard]] fb::DeviceResources &res() { return node.device(0); }

  [[nodiscard]] fb::Result<cis::Sigma> sigma(const fb::SphericalMatrixAO &c, std::int32_t n_occ,
                                             const double *delta, cis::Spin spin) {
    return cis::Sigma::create(res(), *basis, transform->u, j_classes, k_classes, c, n_occ, delta,
                              spin);
  }

  [[nodiscard]] fb::Result<cis::DfSigma> df_sigma(const fb::SphericalMatrixAO &c,
                                                  std::int32_t n_occ, const double *delta,
                                                  cis::Spin spin, std::int32_t max_block) {
    return cis::DfSigma::create(res(), fit->b, fit->naux, c, n_occ, delta, spin, max_block);
  }
};

// fitted: the EriOutput is a fit over cc-pVDZ-JKFIT instead, held as fit.
std::unique_ptr<Setup> make_setup(const fb::json &entry, bool with_scf, bool fitted = false) {
  auto setup = std::make_unique<Setup>();
  const fb::test::ScratchDir dir("cis_sigma");
  // Bohr in the reference, Angstrom in an XYZ file.
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
                  "[orthogonalizer]\n[core_hamiltonian]\n[guess]\n",
                  entry.at("basis").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return nullptr;
  }
  if (const fb::Status ran = calc->run(setup->node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc));
  const auto &molecule = setup->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = setup->calc->get<fb::BasisSetOutput>();
  const auto &transform = setup->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = setup->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = setup->calc->get<fb::OverlapOutput>();
  const auto &x = setup->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = setup->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = setup->calc->get<fb::GuessOutput>();
  if (!molecule.has_value() || !basis_set.has_value() || !transform.has_value() ||
      !pairs.has_value() || !overlap.has_value() || !x.has_value() || !h.has_value() ||
      !guess.has_value()) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  setup->basis = &basis_set->basis;
  setup->transform = &*transform;
  EXPECT_EQ(setup->basis->nao_sphe(), entry.at("nao").get<std::uint32_t>());

  fb::DeviceResources &res = setup->res();
  const fb::EriStep eri_step =
      fitted ? fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"}
             : fb::EriStep{};
  auto eri = eri_step.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          *molecule, *basis_set, *pairs},
      res);
  if (!eri.has_value()) {
    ADD_FAILURE() << eri.error().message;
    return nullptr;
  }
  if (with_scf) {
    auto scf = fb::test::kExactScf.run(fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform,
                                                                  *eri, *overlap, *x, *h, *guess},
                                       res);
    if (!scf.has_value()) {
      ADD_FAILURE() << scf.error().message;
      return nullptr;
    }
    EXPECT_NEAR(scf->energy, entry.at("rhf_energy").get<double>(), 1e-8);
    setup->scf.emplace(std::move(*scf));
  }
  if (fitted) {
    setup->fit.emplace(std::move(std::get<fb::DensityFitted>(eri->eri)));
    return setup;
  }
  // The sigma's own classes, as pipeline.eri's IntegralDirect holds them.
  auto &direct = std::get<fb::IntegralDirect>(eri->eri);
  setup->j_classes = std::move(direct.j_classes);
  setup->k_classes = std::move(direct.k_classes);
  return setup;
}

// Synthetic orbitals for the operator checks: C uniform on [-0.5, 0.5), which
// keeps every transition density element of a unit b inside the |S| <= 2
// J's classes were pruned for, and ε ascending.
struct Synthetic {
  fb::SphericalMatrixAO c;
  fb::Buffers::device<double> delta;
  std::int32_t n_occ;
  std::int32_t n_ov;
};

Synthetic synthetic(Setup &setup, std::int32_t n_occ) {
  fb::DeviceResources &res = setup.res();
  const std::uint32_t nao = setup.basis->nao_sphe();
  fb::SphericalMatrixAO c{res, *setup.basis};
  upload(res, random_vector(static_cast<std::size_t>(nao) * nao, 7, -0.5, 0.5), c.data());
  auto eps = random_vector(nao, 11, -2.0, 3.0);
  std::sort(eps.begin(), eps.end());
  const auto d_eps = to_device(res, eps);
  const auto n_virt = static_cast<std::int32_t>(nao) - n_occ;
  fb::Buffers::device<double> delta(static_cast<std::size_t>(n_occ) * n_virt, res.device());
  EXPECT_TRUE(cis::diagonal(res, n_occ, n_virt, d_eps.data(), delta.data()).has_value());
  EXPECT_TRUE(res.sync().has_value());
  return {std::move(c), std::move(delta), n_occ, n_occ * n_virt};
}

// A over the engine's RHF, assembled column by column (the identity block
// applied in one call, which runs its columns in turn), checked symmetric,
// and its eigenvalues ascending.
std::vector<double> spectrum(Setup &setup, cis::Spin spin) {
  fb::DeviceResources &res = setup.res();
  const fb::ScfOutput &scf = *setup.scf;
  const std::int32_t n_occ = scf.n_alpha;
  const auto nmo = static_cast<std::int32_t>(scf.c.n());
  const std::int32_t n_virt = nmo - n_occ;
  const std::int32_t n_ov = n_occ * n_virt;
  const auto n = static_cast<std::size_t>(n_ov);

  const auto eps = to_device(res, scf.mo_energies);
  fb::Buffers::device<double> delta(n, res.device());
  EXPECT_TRUE(cis::diagonal(res, n_occ, n_virt, eps.data(), delta.data()).has_value());
  auto sigma = setup.sigma(scf.c, n_occ, delta.data(), spin);
  if (!sigma.has_value()) {
    ADD_FAILURE() << sigma.error().message;
    return {};
  }
  std::vector<double> identity(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    identity[(i * n) + i] = 1.0;
  }
  const auto x = to_device(res, identity);
  fb::Buffers::device<double> y(n * n, res.device());
  EXPECT_TRUE(sigma->apply(res, n_ov, x.data(), y.data()).has_value());
  auto a = to_host(res, y.data(), n * n);

  double asymmetry = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      asymmetry = std::max(asymmetry, std::abs(a[(i * n) + j] - a[(j * n) + i]));
    }
  }
  std::cout << "A asymmetry: " << asymmetry / max_abs(a) << " of max|A|\n";
  EXPECT_LE(asymmetry, kAssembledSymmetryTolerance * max_abs(a));

  auto d_a = to_device(res, a);
  fb::Buffers::device<double> w(n, res.device());
  EXPECT_TRUE(fb::diagonalize<double>(res, n_ov, d_a.data(), w.data()).has_value());
  return to_host(res, w.data(), n);
}

void expect_roots(const std::vector<double> &got, const fb::json &roots, std::string_view what) {
  const auto want = roots.at("energies").get<std::vector<double>>();
  ASSERT_GE(got.size(), want.size());
  for (std::size_t r = 0; r < want.size(); ++r) {
    std::cout << what << " root " << r << ": " << got[r] - want[r] << " hartree\n";
    EXPECT_NEAR(got[r], want[r], kEnergyTolerance) << what << " root " << r;
  }
}

// The spin by name, so that ctest's entry names it too (gtest_discover_tests
// prints the parameter's value, and an enum's is its bytes).
class CisSigmaSpin : public ::testing::TestWithParam<std::string> {
protected:
  [[nodiscard]] cis::Spin spin() const {
    return GetParam() == "Triplet" ? cis::Spin::Triplet : cis::Spin::Singlet;
  }
};

// <b1, A b2> = <b2, A b1> for two random b, on synthetic C and ε.
TEST_P(CisSigmaSpin, IsSymmetric) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  auto sigma = setup->sigma(syn.c, syn.n_occ, syn.delta.data(), spin());
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  ASSERT_EQ(sigma->size(), syn.n_ov);

  const auto n = static_cast<std::size_t>(syn.n_ov);
  auto b = random_vector(2 * n, 23, -1.0, 1.0);
  const auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_s(2 * n, res.device());
  ASSERT_TRUE(sigma->apply(res, 2, d_b.data(), d_s.data()).has_value());
  const auto s = to_host(res, d_s.data(), 2 * n);
  const std::vector<double> b1(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<double> b2(b.begin() + static_cast<std::ptrdiff_t>(n), b.end());
  const std::vector<double> s1(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<double> s2(s.begin() + static_cast<std::ptrdiff_t>(n), s.end());
  const double scale = max_abs(s) * static_cast<double>(n);
  const double b1_ab2 = dot(b1, s2);
  const double b2_ab1 = dot(b2, s1);
  // Not a symmetry of the diagonal alone: the two-electron part is past it.
  ASSERT_GT(max_abs(s), 1.0);
  std::cout << "symmetry: " << std::abs(b1_ab2 - b2_ab1) / scale << " of max|sigma| n_ov\n";
  EXPECT_LE(std::abs(b1_ab2 - b2_ab1), kOperatorTolerance * scale) << b1_ab2 << " vs " << b2_ab1;
}

// A block of four is its four columns applied one call each.
TEST_P(CisSigmaSpin, BlockEqualsColumns) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  auto sigma = setup->sigma(syn.c, syn.n_occ, syn.delta.data(), spin());
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;

  constexpr std::int32_t kColumns = 4;
  const auto n = static_cast<std::size_t>(syn.n_ov);
  const auto d_b = to_device(res, random_vector(kColumns * n, 31, -1.0, 1.0));
  fb::Buffers::device<double> block(kColumns * n, res.device());
  fb::Buffers::device<double> single(kColumns * n, res.device());
  ASSERT_TRUE(sigma->apply(res, kColumns, d_b.data(), block.data()).has_value());
  for (std::size_t col = 0; col < kColumns; ++col) {
    ASSERT_TRUE(sigma->apply(res, 1, d_b.data() + (col * n), single.data() + (col * n)));
  }
  const auto got = to_host(res, block.data(), kColumns * n);
  const auto want = to_host(res, single.data(), kColumns * n);
  double diff = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  std::cout << "block vs columns: " << diff / max_abs(want) << " of max|sigma|\n";
  EXPECT_LE(diff, kOperatorTolerance * max_abs(want));
}

INSTANTIATE_TEST_SUITE_P(Spins, CisSigmaSpin, ::testing::Values("Singlet", "Triplet"),
                         [](const ::testing::TestParamInfo<std::string> &info) {
                           return info.param;
                         });

// The pool's high-water mark over fn, above what was in use when it started
// (zero if it never rose past that), with res synchronized on both sides. On
// HIP the mark has read 2.8 MB below the starting use after an fn that
// allocates nothing (a second DF apply, RX 9060 XT), which the bare unsigned
// difference turned into nearly 2^64.
std::uint64_t pool_peak(fb::DeviceResources &res, const std::function<void()> &fn) {
  wwr::wwrMemPool_t pool = res.device()->pool().get();
  EXPECT_TRUE(res.sync().has_value());
  std::uint64_t zero = 0;
  EXPECT_EQ(wwr::wwrMemPoolSetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &zero),
            wwr::wwrSuccess);
  std::uint64_t before = 0;
  EXPECT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemCurrent, &before),
            wwr::wwrSuccess);
  fn();
  EXPECT_TRUE(res.sync().has_value());
  std::uint64_t high = 0;
  EXPECT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &high),
            wwr::wwrSuccess);
  return high > before ? high - before : 0;
}

// create allocates every buffer of the sigma's and grows the workspace, and
// coulomb's and exchange's scratch is their classes', so a second apply takes
// nothing from the device's pool and leaves the workspace where it was.
TEST(CisSigma, SecondApplyAllocatesNothing) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  auto sigma = setup->sigma(syn.c, syn.n_occ, syn.delta.data(), cis::Spin::Singlet);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  constexpr std::int32_t kColumns = 4;
  const auto n = static_cast<std::size_t>(syn.n_ov);
  const auto d_b = to_device(res, random_vector(kColumns * n, 41, -1.0, 1.0));
  fb::Buffers::device<double> d_s(kColumns * n, res.device());
  ASSERT_TRUE(sigma->apply(res, kColumns, d_b.data(), d_s.data()).has_value());

  const std::byte *const workspace = res.device_workspace(0).data();
  const std::size_t workspace_bytes = res.device_workspace(0).num_elements();
  const std::uint64_t second = pool_peak(
      res, [&] { EXPECT_TRUE(sigma->apply(res, kColumns, d_b.data(), d_s.data()).has_value()); });
  std::cout << "pool peak: second apply " << second << " B\n";
  EXPECT_EQ(second, 0U);
  EXPECT_EQ(res.device_workspace(0).data(), workspace);
  EXPECT_EQ(res.device_workspace(0).num_elements(), workspace_bytes);
}

TEST(CisSigma, Refusals) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_sto3g"), false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  const auto nao = static_cast<std::int32_t>(setup->basis->nao_sphe());
  for (const std::int32_t n_occ : {0, nao, -1}) {
    const auto bad = setup->sigma(syn.c, n_occ, syn.delta.data(), cis::Spin::Singlet);
    ASSERT_FALSE(bad.has_value()) << n_occ;
    EXPECT_EQ(bad.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(bad.error().message.find("cis::Sigma::create"), std::string::npos);
  }
  const auto no_delta = setup->sigma(syn.c, 5, nullptr, cis::Spin::Singlet);
  ASSERT_FALSE(no_delta.has_value());
  EXPECT_EQ(no_delta.error().kind, fb::ErrorKind::InvalidConfig);

  auto sigma = setup->sigma(syn.c, 5, syn.delta.data(), cis::Spin::Singlet);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  const auto refused = sigma->apply(res, -1, nullptr, nullptr);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_TRUE(sigma->apply(res, 0, nullptr, nullptr).has_value());
}

// The density-fitted operator's symmetry, <b1, A b2> = <b2, A b1>, on the
// same synthetic C and ε. B_voᵀ B_vo is symmetric by construction and the
// exchange term is through B_oo's and B_vv's own symmetry, so only rounding
// separates the two sides.
TEST_P(CisSigmaSpin, DfIsSymmetric) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  auto sigma = setup->df_sigma(syn.c, syn.n_occ, syn.delta.data(), spin(), 2);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  ASSERT_EQ(sigma->size(), syn.n_ov);

  const auto n = static_cast<std::size_t>(syn.n_ov);
  auto b = random_vector(2 * n, 23, -1.0, 1.0);
  const auto d_b = to_device(res, b);
  fb::Buffers::device<double> d_s(2 * n, res.device());
  ASSERT_TRUE(sigma->apply(res, 2, d_b.data(), d_s.data()).has_value());
  const auto s = to_host(res, d_s.data(), 2 * n);
  const std::vector<double> b1(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<double> b2(b.begin() + static_cast<std::ptrdiff_t>(n), b.end());
  const std::vector<double> s1(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<double> s2(s.begin() + static_cast<std::ptrdiff_t>(n), s.end());
  const double scale = max_abs(s) * static_cast<double>(n);
  const double b1_ab2 = dot(b1, s2);
  const double b2_ab1 = dot(b2, s1);
  ASSERT_GT(max_abs(s), 1.0);
  std::cout << "DF symmetry: " << std::abs(b1_ab2 - b2_ab1) / scale << " of max|sigma| n_ov\n";
  EXPECT_LE(std::abs(b1_ab2 - b2_ab1), kOperatorTolerance * scale) << b1_ab2 << " vs " << b2_ab1;
}

// A block of five is its five columns applied one call each, with the sigma
// created for blocks of four: the first four in one pass, the fifth in a
// second.
TEST_P(CisSigmaSpin, DfBlockEqualsColumns) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  auto sigma = setup->df_sigma(syn.c, syn.n_occ, syn.delta.data(), spin(), 4);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;

  constexpr std::int32_t kColumns = 5;
  const auto n = static_cast<std::size_t>(syn.n_ov);
  const auto d_b = to_device(res, random_vector(kColumns * n, 31, -1.0, 1.0));
  fb::Buffers::device<double> block(kColumns * n, res.device());
  fb::Buffers::device<double> single(kColumns * n, res.device());
  ASSERT_TRUE(sigma->apply(res, kColumns, d_b.data(), block.data()).has_value());
  for (std::size_t col = 0; col < kColumns; ++col) {
    ASSERT_TRUE(sigma->apply(res, 1, d_b.data() + (col * n), single.data() + (col * n)));
  }
  const auto got = to_host(res, block.data(), kColumns * n);
  const auto want = to_host(res, single.data(), kColumns * n);
  double diff = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  std::cout << "DF block vs columns: " << diff / max_abs(want) << " of max|sigma|\n";
  EXPECT_LE(diff, kOperatorTolerance * max_abs(want));
}

// create allocates the MO blocks and grows the workspace to apply's scratch
// for its block, so a second apply of that block takes nothing from the
// device's pool and leaves the workspace where it was.
TEST(CisSigma, DfSecondApplyAllocatesNothing) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  constexpr std::int32_t kColumns = 4;
  auto sigma = setup->df_sigma(syn.c, syn.n_occ, syn.delta.data(), cis::Spin::Singlet, kColumns);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  const auto n = static_cast<std::size_t>(syn.n_ov);
  const auto d_b = to_device(res, random_vector(kColumns * n, 41, -1.0, 1.0));
  fb::Buffers::device<double> d_s(kColumns * n, res.device());
  ASSERT_TRUE(sigma->apply(res, kColumns, d_b.data(), d_s.data()).has_value());

  const std::byte *const workspace = res.device_workspace(0).data();
  const std::size_t workspace_bytes = res.device_workspace(0).num_elements();
  const std::uint64_t second = pool_peak(
      res, [&] { EXPECT_TRUE(sigma->apply(res, kColumns, d_b.data(), d_s.data()).has_value()); });
  std::cout << "pool peak: second DF apply " << second << " B\n";
  EXPECT_EQ(second, 0U);
  EXPECT_EQ(res.device_workspace(0).data(), workspace);
  EXPECT_EQ(res.device_workspace(0).num_elements(), workspace_bytes);
}

TEST(CisSigma, DfRefusals) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"), false, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  auto syn = synthetic(*setup, 5);
  const auto nao = static_cast<std::int32_t>(setup->basis->nao_sphe());
  const auto expect_config = [](const fb::Result<cis::DfSigma> &bad, std::string_view what) {
    ASSERT_FALSE(bad.has_value()) << what;
    EXPECT_EQ(bad.error().kind, fb::ErrorKind::InvalidConfig) << bad.error().message;
    EXPECT_NE(bad.error().message.find("cis::DfSigma::create"), std::string::npos)
        << bad.error().message;
  };
  for (const std::int32_t n_occ : {0, nao, -1}) {
    expect_config(setup->df_sigma(syn.c, n_occ, syn.delta.data(), cis::Spin::Singlet, 2), "n_occ");
  }
  expect_config(setup->df_sigma(syn.c, 5, nullptr, cis::Spin::Singlet, 2), "delta");
  expect_config(setup->df_sigma(syn.c, 5, syn.delta.data(), cis::Spin::Singlet, 0), "max_block");
  expect_config(cis::DfSigma::create(res, setup->fit->b, setup->fit->naux + 1, syn.c, 5,
                                     syn.delta.data(), cis::Spin::Singlet, 2),
                "naux");

  auto sigma = setup->df_sigma(syn.c, 5, syn.delta.data(), cis::Spin::Singlet, 2);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  const auto refused = sigma->apply(res, -1, nullptr, nullptr);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_TRUE(sigma->apply(res, 0, nullptr, nullptr).has_value());
}

// The conventional H2O / STO-3G singlets and triplets.
TEST(CisSigma, H2oSto3gMatchesReference) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  expect_roots(spectrum(*setup, cis::Spin::Singlet), entry.at("singlets"), "singlet");
  expect_roots(spectrum(*setup, cis::Spin::Triplet), entry.at("triplets"), "triplet");
}

// HF / cc-pVTZ's singlets: f shells through the spherical bridge, and the Π
// pair, compared root by root (a degenerate group's energies are each
// defined; only its vectors are not). Not by the full A: its 195 columns of
// three f-shell builds each take ~200 s on the RTX 3080 and ~400 s on the
// RX 9060 XT, past ctest's 120 s backstop. So the roots come from
// davidson_solve over the same sigma, from cis::guess's start with
// cis::Preconditioner, at cis::Options' own tolerance: what the solver (#199)
// will run. Still slow for the fast set; H2O / STO-3G stands for the family
// there.
TEST(CisSigma, HfCcPvtzSingletsMatchReference) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &entry = reference_case("hf_ccpvtz");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScfOutput &scf = *setup->scf;
  const std::int32_t n_occ = scf.n_alpha;
  const std::int32_t n_virt = static_cast<std::int32_t>(scf.c.n()) - n_occ;
  const std::int32_t n_ov = n_occ * n_virt;
  const auto n = static_cast<std::size_t>(n_ov);
  const auto want = entry.at("singlets").at("energies").get<std::vector<double>>();
  const auto n_roots = static_cast<std::int32_t>(want.size());
  const std::int32_t guess_count = 2 * n_roots;
  const std::int32_t max_subspace = 8 * n_roots;

  const auto eps = to_device(res, scf.mo_energies);
  fb::Buffers::device<double> delta(n, res.device());
  ASSERT_TRUE(cis::diagonal(res, n_occ, n_virt, eps.data(), delta.data()).has_value());
  auto sigma = setup->sigma(scf.c, n_occ, delta.data(), cis::Spin::Singlet);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  fb::Buffers::device<double> start(n * static_cast<std::size_t>(guess_count), res.device());
  ASSERT_TRUE(cis::guess(res, n_ov, delta.data(), guess_count, start.data()).has_value());
  const cis::Preconditioner precondition{res, n_ov, delta.data(), n_roots};
  fb::Buffers::device<double> values(static_cast<std::size_t>(n_roots), res.device());
  fb::Buffers::device<double> vectors(n * static_cast<std::size_t>(n_roots), res.device());
  const auto info = fb::davidson_solve<double>(
      res, *sigma, precondition, n_ov, n_roots, max_subspace, start.data(), guess_count,
      values.data(), vectors.data(), cis::Options{}.davidson);
  ASSERT_TRUE(info.has_value()) << info.error().message;
  std::cout << "davidson: " << info->iterations << " iterations, residual "
            << info->max_residual_norm << "\n";
  expect_roots(to_host(res, values.data(), static_cast<std::size_t>(n_roots)), entry.at("singlets"),
               "singlet");
}

} // namespace
