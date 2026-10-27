// localization::valence_virtual_orbitals over the IAOs of an RHF SCF, water and N2 in
// cc-pVDZ over MINAO: m - n_occ of them, S-orthonormal, orthogonal to the
// occupied orbitals and inside the canonical virtual space, and a span that a
// rotation of the occupied orbitals does not move. Then localization::livvo_localize on
// water, two mirror-image sigma*_OH LIVVOs, and localization::valence_character over the
// canonical virtuals. The SCF is the pipeline's, from a config naming only
// [scf]. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.localization.ibo;
import free_breakfast.localization.livvo;
import free_breakfast.analysis.population;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// Every check is an orthonormality or a projector built from orbitals the SCF
// and the IAOs give to rounding; nothing here carries the SCF's tolerance.
constexpr double kTolerance = 1e-10;

// A LIVVO is two-centre when O and its H hold more than this of it; the far
// H holds under 1e-6 of either.
constexpr double kTwoCenter = 0.999;
// The two LIVVOs' populations agree to the localization's convergence rather
// than to rounding; they are measured equal to about 1e-11.
constexpr double kMirror = 1e-8;

// Angstrom.
constexpr std::string_view kWater = "3\nwater\n"
                                    "O 0.0 0.0 0.1173\n"
                                    "H 0.0 0.7572 -0.4692\n"
                                    "H 0.0 -0.7572 -0.4692\n";
constexpr std::string_view kNitrogen = "2\nN2\n"
                                       "N 0.0 0.0 0.0\n"
                                       "N 0.0 0.0 1.0977\n";

std::vector<double> download(fb::DeviceResources &res, const double *device, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The calculation of a config naming @p xyz in cc-pVDZ and [scf], run.
std::optional<fb::Calculation> run_scf(std::string_view xyz) {
  const fb::test::ScratchDir dir("livvo");
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", std::string{xyz}).has_value());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\n"
                                               "basis = \"cc-pvdz\"\n[scf]\n")
                  .has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return std::nullopt;
  }
  fb::NodeResources node;
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return std::nullopt;
  }
  return std::move(*calc);
}

// (X^T S Y)_ij for the n x p X and n x q Y, column-major, on the host.
std::vector<double> overlap_of(const std::vector<double> &x, const std::vector<double> &s,
                               const std::vector<double> &y, std::size_t n, std::size_t p,
                               std::size_t q) {
  std::vector<double> sy(n * q, 0.0);
  for (std::size_t j = 0; j < q; ++j) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        sy[mu + (n * j)] += s[mu + (n * nu)] * y[nu + (n * j)];
      }
    }
  }
  std::vector<double> xsy(p * q, 0.0);
  for (std::size_t j = 0; j < q; ++j) {
    for (std::size_t i = 0; i < p; ++i) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        xsy[i + (p * j)] += x[mu + (n * i)] * sy[mu + (n * j)];
      }
    }
  }
  return xsy;
}

// V V^T, the n x n projector-in-coefficients of the n x p V: the same for any
// orthonormal basis of V's span.
std::vector<double> outer(const std::vector<double> &v, std::size_t n, std::size_t p) {
  std::vector<double> vv(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t r = 0; r < p; ++r) {
        vv[i + (n * j)] += v[i + (n * r)] * v[j + (n * r)];
      }
    }
  }
  return vv;
}

// The IAOs of the n x @p n_occ occupied orbitals @p c (device), then their
// VVOs, downloaded.
std::vector<double> build_vvos(fb::DeviceResources &res, const fb::BasisSet &basis,
                               const fb::BasisSet &minao, std::int32_t n_occ, const double *s,
                               const double *c) {
  const std::size_t n = basis.nao_sphe();
  const std::size_t m = minao.nao_sphe();
  const std::size_t n_vvo = m - static_cast<std::size_t>(n_occ);
  fb::Buffers::device<double> iao(n * m, res.device());
  fb::Buffers::device<double> vvo(n * n_vvo, res.device());
  const fb::Status built =
      fb::localization::intrinsic_atomic_orbitals(res, basis, minao, n_occ, s, c, iao.data());
  EXPECT_TRUE(built.has_value()) << built.error().message;
  const fb::Status made = fb::localization::valence_virtual_orbitals(res, basis, minao, n_occ, s, c,
                                                                     iao.data(), vvo.data());
  EXPECT_TRUE(made.has_value()) << made.error().message;
  EXPECT_TRUE(res.sync().has_value());
  return download(res, vvo.data(), n * n_vvo);
}

struct Case {
  std::string_view name;
  std::string_view xyz;
  std::size_t n_vvo;
};

void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class Livvo : public ::testing::TestWithParam<Case> {};

TEST_P(Livvo, AreTheOrthonormalVirtualHalfOfTheIaoSpan) {
  FBF_REQUIRE_GPU();
  auto calc = run_scf(GetParam().xyz);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && overlap.has_value() && scf.has_value());
  auto stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto minao = fb::place(stamp->primitive_normalized(), molecule->molecule);
  ASSERT_TRUE(minao.has_value()) << minao.error().message;

  const std::size_t n = basis->basis.nao_sphe();
  const std::size_t m = minao->nao_sphe();
  const auto occ = static_cast<std::size_t>(molecule->molecule.electron_count() / 2);
  const std::size_t n_vvo = m - occ;
  ASSERT_EQ(n_vvo, GetParam().n_vvo);
  const auto k = static_cast<std::int32_t>(occ);

  fb::DeviceResources res(scf->c.dev_idx());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> v =
      build_vvos(res, basis->basis, *minao, k, overlap->s.data(), scf->c.data());
  const std::vector<double> s = download(res, overlap->s.data(), n * n);
  const std::vector<double> c = download(res, scf->c.data(), n * n);

  // V^T S V = I.
  const std::vector<double> vsv = overlap_of(v, s, v, n, n_vvo, n_vvo);
  for (std::size_t j = 0; j < n_vvo; ++j) {
    for (std::size_t i = 0; i < n_vvo; ++i) {
      EXPECT_NEAR(vsv[i + (n_vvo * j)], i == j ? 1.0 : 0.0, kTolerance) << i << "," << j;
    }
  }
  // C_occ^T S V = 0, and V inside the canonical virtual space: its weight on
  // the n - n_occ virtuals is all of it, ||C_vir^T S V||_F^2 = n_vvo.
  const std::vector<double> csv = overlap_of(c, s, v, n, n, n_vvo);
  double virtual_weight = 0.0;
  for (std::size_t j = 0; j < n_vvo; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      const double x = csv[i + (n * j)];
      if (i < occ) {
        EXPECT_NEAR(x, 0.0, kTolerance) << "occupied " << i << ", VVO " << j;
      } else {
        virtual_weight += x * x;
      }
    }
  }
  EXPECT_NEAR(virtual_weight, static_cast<double>(n_vvo), kTolerance);

  // The occupied orbitals rotated among themselves, by a Givens rotation on
  // each neighbouring pair: the same occupied space, so the same IAOs and the
  // same VVO span, compared as V V^T.
  std::vector<double> rotated(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(n * occ));
  for (std::size_t i = 0; i + 1 < occ; ++i) {
    const double angle = 0.3 + (0.2 * static_cast<double>(i));
    const double cs = std::cos(angle);
    const double sn = std::sin(angle);
    for (std::size_t mu = 0; mu < n; ++mu) {
      const double a = rotated[mu + (n * i)];
      const double b = rotated[mu + (n * (i + 1))];
      rotated[mu + (n * i)] = (cs * a) - (sn * b);
      rotated[mu + (n * (i + 1))] = (sn * a) + (cs * b);
    }
  }
  // Uploaded on res.stream(), which is non-blocking: a plain wwrMemcpy from
  // pageable memory goes on the null stream and may return before its DMA
  // lands, so the IAOs below could read c_rotated half written.
  fb::Buffers::device<double> c_rotated(n * occ, res.device());
  ASSERT_EQ(wwr::wwrMemcpyAsync(c_rotated.data(), rotated.data(), sizeof(double) * rotated.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  const std::vector<double> w =
      build_vvos(res, basis->basis, *minao, k, overlap->s.data(), c_rotated.data());
  const std::vector<double> vv = outer(v, n, n_vvo);
  const std::vector<double> ww = outer(w, n, n_vvo);
  for (std::size_t i = 0; i < n * n; ++i) {
    EXPECT_NEAR(vv[i], ww[i], kTolerance) << i % n << "," << i / n;
  }
}

INSTANTIATE_TEST_SUITE_P(CcPvdzOverMinao, Livvo,
                         ::testing::Values(Case{"h2o", kWater, 2}, Case{"n2", kNitrogen, 3}),
                         [](const ::testing::TestParamInfo<Case> &info) {
                           return std::string{info.param.name};
                         });

// Water's two LIVVOs are the sigma*_OH: each on O and one H by the IAO
// populations, one per H, mirror images of each other. Their valence
// character over the canonical virtuals is a partition of n_vvo, the same
// from the VVOs as from the LIVVOs.
TEST(Livvo, WaterLocalizesIntoTwoMirrorImageSigmaStarOh) {
  FBF_REQUIRE_GPU();
  auto calc = run_scf(kWater);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && overlap.has_value() && scf.has_value());
  auto stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto minao = fb::place(stamp->primitive_normalized(), molecule->molecule);
  ASSERT_TRUE(minao.has_value()) << minao.error().message;

  const std::size_t n = basis->basis.nao_sphe();
  const std::size_t m = minao->nao_sphe();
  const auto occ = static_cast<std::size_t>(molecule->molecule.electron_count() / 2);
  const std::size_t n_vvo = m - occ;
  ASSERT_EQ(n_vvo, 2U);
  const auto k = static_cast<std::int32_t>(occ);

  fb::DeviceResources res(scf->c.dev_idx());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const double *s_dev = overlap->s.data();
  const double *c_dev = scf->c.data();
  fb::Buffers::device<double> iao(n * m, res.device());
  fb::Buffers::device<double> vvo(n * n_vvo, res.device());
  fb::Buffers::device<double> livvo(n * n_vvo, res.device());
  fb::Buffers::device<double> u(n_vvo * n_vvo, res.device());
  const fb::Status built = fb::localization::intrinsic_atomic_orbitals(res, basis->basis, *minao, k,
                                                                       s_dev, c_dev, iao.data());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  const fb::Status made = fb::localization::valence_virtual_orbitals(
      res, basis->basis, *minao, k, s_dev, c_dev, iao.data(), vvo.data());
  ASSERT_TRUE(made.has_value()) << made.error().message;
  const auto info = fb::localization::livvo_localize(
      res, basis->basis, *minao, k, s_dev, iao.data(), vvo.data(), livvo.data(), u.data());
  ASSERT_TRUE(info.has_value()) << info.error().message;
  EXPECT_GE(info->functional, info->initial_functional);

  const std::vector<double> s = download(res, s_dev, n * n);
  const std::vector<double> c = download(res, c_dev, n * n);
  const std::vector<double> l = download(res, livvo.data(), n * n_vvo);

  // L^T S L = I and C_occ^T S L = 0.
  const std::vector<double> lsl = overlap_of(l, s, l, n, n_vvo, n_vvo);
  for (std::size_t j = 0; j < n_vvo; ++j) {
    for (std::size_t i = 0; i < n_vvo; ++i) {
      EXPECT_NEAR(lsl[i + (n_vvo * j)], i == j ? 1.0 : 0.0, kTolerance) << i << "," << j;
    }
  }
  const std::vector<double> csl = overlap_of(c, s, l, n, occ, n_vvo);
  for (const double x : csl) {
    EXPECT_NEAR(x, 0.0, kTolerance);
  }

  // Populations over the IAOs, atoms O, H, H: each LIVVO on O and one H, a
  // different H each, and the two mirror images.
  const auto pop = fb::orbital_populations(
      res, basis->basis, *minao, static_cast<std::int32_t>(n_vvo), s_dev, livvo.data(), iao.data());
  ASSERT_TRUE(pop.has_value()) << pop.error().message;
  ASSERT_EQ(pop->size(), 3 * n_vvo);
  const auto at = [&](std::size_t atom, std::size_t orb) { return (*pop)[atom + (3 * orb)]; };
  std::array<std::size_t, 2> hydrogen{};
  for (std::size_t j = 0; j < n_vvo; ++j) {
    hydrogen[j] = at(1, j) > at(2, j) ? 1 : 2;
    const std::size_t other = 3 - hydrogen[j];
    std::cout << std::format("LIVVO {}: O {:.12f}, H{} {:.12f}, H{} {:.3e}\n", j, at(0, j),
                             hydrogen[j], at(hydrogen[j], j), other, at(other, j));
    EXPECT_GT(at(0, j) + at(hydrogen[j], j), kTwoCenter) << "LIVVO " << j;
  }
  EXPECT_NE(hydrogen[0], hydrogen[1]);
  EXPECT_NEAR(at(0, 0), at(0, 1), kMirror);
  EXPECT_NEAR(at(hydrogen[0], 0), at(hydrogen[1], 1), kMirror);
  EXPECT_NEAR(at(hydrogen[1], 0), at(hydrogen[0], 1), kMirror);

  // t_val over the n - n_occ canonical virtuals: each in [0, 1], summing to
  // n_vvo since the VVOs lie wholly in the virtual space, and the same from
  // the VVOs as from the LIVVOs, which span the same space.
  const auto n_virt = static_cast<std::int32_t>(n - occ);
  const double *virtuals = c_dev + (n * occ);
  const auto from_livvo = fb::localization::valence_character(
      res, basis->basis, n_virt, static_cast<std::int32_t>(n_vvo), s_dev, virtuals, livvo.data());
  const auto from_vvo = fb::localization::valence_character(
      res, basis->basis, n_virt, static_cast<std::int32_t>(n_vvo), s_dev, virtuals, vvo.data());
  ASSERT_TRUE(from_livvo.has_value()) << from_livvo.error().message;
  ASSERT_TRUE(from_vvo.has_value()) << from_vvo.error().message;
  ASSERT_EQ(from_livvo->size(), n - occ);
  ASSERT_EQ(from_vvo->size(), n - occ);
  double total = 0.0;
  for (std::size_t i = 0; i < n - occ; ++i) {
    const double t_val = (*from_livvo)[i];
    EXPECT_GE(t_val, -kTolerance) << "virtual " << i;
    EXPECT_LE(t_val, 1.0 + kTolerance) << "virtual " << i;
    EXPECT_NEAR(t_val, (*from_vvo)[i], kTolerance) << "virtual " << i;
    total += t_val;
  }
  EXPECT_NEAR(total, static_cast<double>(n_vvo), kTolerance);
}

// The arguments are refused before anything is read, so any non-null device
// pointer serves for the matrices.
TEST(Livvo, RefusesAnOutOfRangeNOccMismatchedAtomsAndNullPointers) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule water(res, {fb::Atom{{0.0, 0.0, 0.2}, fb::Element::O},
                                 fb::Atom{{0.0, 1.4, -0.9}, fb::Element::H},
                                 fb::Atom{{0.0, -1.4, -0.9}, fb::Element::H}});
  const fb::Molecule hydrogen(
      res, {fb::Atom{{0.0, 0.0, 0.0}, fb::Element::H}, fb::Atom{{0.0, 0.0, 1.4}, fb::Element::H}});
  auto basis_stamp = fb::resolve_basis("cc-pvdz");
  auto minao_stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(basis_stamp.has_value() && minao_stamp.has_value());
  auto basis = fb::place(basis_stamp->primitive_normalized(), water);
  auto minao = fb::place(minao_stamp->primitive_normalized(), water);
  auto other = fb::place(minao_stamp->primitive_normalized(), hydrogen);
  ASSERT_TRUE(basis.has_value() && minao.has_value() && other.has_value());
  const auto m = static_cast<std::int32_t>(minao->nao_sphe());
  fb::Buffers::device<double> any(1, res.device());
  double *p = any.data();

  const auto refused = [&](const fb::BasisSet &minimal, std::int32_t n_occ, const double *s,
                           const double *c, const double *iao, double *vvo) {
    const fb::Status status =
        fb::localization::valence_virtual_orbitals(res, *basis, minimal, n_occ, s, c, iao, vvo);
    return !status.has_value() && status.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused(*minao, -1, p, p, p, p));
  EXPECT_TRUE(refused(*minao, m + 1, p, p, p, p));
  EXPECT_TRUE(refused(*other, 1, p, p, p, p));
  EXPECT_TRUE(refused(*minao, 5, nullptr, p, p, p));
  EXPECT_TRUE(refused(*minao, 5, p, nullptr, p, p));
  EXPECT_TRUE(refused(*minao, 5, p, p, nullptr, p));
  EXPECT_TRUE(refused(*minao, 5, p, p, p, nullptr));
}

TEST(Livvo, LocalizeAndValenceCharacterRefuseBadArguments) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule water(res, {fb::Atom{{0.0, 0.0, 0.2}, fb::Element::O},
                                 fb::Atom{{0.0, 1.4, -0.9}, fb::Element::H},
                                 fb::Atom{{0.0, -1.4, -0.9}, fb::Element::H}});
  const fb::Molecule hydrogen(
      res, {fb::Atom{{0.0, 0.0, 0.0}, fb::Element::H}, fb::Atom{{0.0, 0.0, 1.4}, fb::Element::H}});
  auto basis_stamp = fb::resolve_basis("cc-pvdz");
  auto minao_stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(basis_stamp.has_value() && minao_stamp.has_value());
  auto basis = fb::place(basis_stamp->primitive_normalized(), water);
  auto minao = fb::place(minao_stamp->primitive_normalized(), water);
  auto other = fb::place(minao_stamp->primitive_normalized(), hydrogen);
  ASSERT_TRUE(basis.has_value() && minao.has_value() && other.has_value());
  const auto m = static_cast<std::int32_t>(minao->nao_sphe());
  fb::Buffers::device<double> any(2, res.device());
  double *p = any.data();
  double *q = p + 1;

  const auto invalid = [](const auto &result) {
    return !result.has_value() && result.error().kind == fb::ErrorKind::InvalidConfig;
  };
  const auto localize = [&](const fb::BasisSet &minimal, std::int32_t n_occ, const double *s,
                            const double *iao, const double *vvo, double *livvo) {
    return invalid(
        fb::localization::livvo_localize(res, *basis, minimal, n_occ, s, iao, vvo, livvo));
  };
  EXPECT_TRUE(localize(*minao, -1, p, p, p, q));
  EXPECT_TRUE(localize(*minao, m + 1, p, p, p, q));
  EXPECT_TRUE(localize(*other, 1, p, p, p, q));
  EXPECT_TRUE(localize(*minao, 5, nullptr, p, p, q));
  EXPECT_TRUE(localize(*minao, 5, p, nullptr, p, q));
  EXPECT_TRUE(localize(*minao, 5, p, p, nullptr, q));
  EXPECT_TRUE(localize(*minao, 5, p, p, p, nullptr));
  EXPECT_TRUE(localize(*minao, 5, p, p, q, q));

  const auto character = [&](std::int32_t n_orb, std::int32_t n_vvo, const double *s,
                             const double *c, const double *livvo) {
    return invalid(fb::localization::valence_character(res, *basis, n_orb, n_vvo, s, c, livvo));
  };
  EXPECT_TRUE(character(-1, 2, p, p, p));
  EXPECT_TRUE(character(1, -1, p, p, p));
  EXPECT_TRUE(character(1, 2, nullptr, p, p));
  EXPECT_TRUE(character(1, 2, p, nullptr, p));
  EXPECT_TRUE(character(1, 2, p, p, nullptr));
}

} // namespace
