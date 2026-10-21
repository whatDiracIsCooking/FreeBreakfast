// Host cases for experimental/atomic_scf: the one-center integrals against
// closed forms and against the oxygen block of test/gtest's libcint golden
// data, the atomic SCF against PySCF's AtomSphAverageRHF
// (data/atomic_scf.json, written by experimental/atomic_scf/emit_reference.py),
// and the block-diagonal molecular guess.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.basis;
import free_breakfast.experimental.atomic_scf;

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

double max_abs(const std::vector<double> &m) {
  double out = 0.0;
  for (const double v : m) {
    out = std::max(out, std::abs(v));
  }
  return out;
}

// tr(A B) for two symmetric n x n matrices.
double trace_product(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// C = A B, n x n column-major.
std::vector<double> matmul(const std::vector<double> &a, const std::vector<double> &b,
                           std::size_t n) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t i = 0; i < n; ++i) {
        c[i + (n * j)] += a[i + (n * k)] * b[k + (n * j)];
      }
    }
  }
  return c;
}

// U^T M U for M (r x r) and U (r x c), column-major.
std::vector<double> sandwich(const std::vector<double> &m, const std::vector<double> &u,
                             std::size_t r, std::size_t c) {
  std::vector<double> out(c * c, 0.0);
  for (std::size_t j = 0; j < c; ++j) {
    for (std::size_t i = 0; i < c; ++i) {
      double sum = 0.0;
      for (std::size_t b = 0; b < r; ++b) {
        for (std::size_t a = 0; a < r; ++a) {
          sum += u[a + (r * i)] * m[a + (r * b)] * u[b + (r * j)];
        }
      }
      out[i + (c * j)] = sum;
    }
  }
  return out;
}

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_EXPERIMENTAL_DATA_DIR} / "atomic_scf.json");
    if (!read) {
      ADD_FAILURE() << read.error().message;
      return fb::json{};
    }
    return *read;
  }();
  return doc;
}

// One element's shells from the reference file's BSE basis, at the origin, the
// way a production run gets them: parse_basis, primitive_normalized, at.
std::vector<fb::Shell> reference_shells(const std::string &basis, std::int32_t z) {
  auto stamp = fb::parse_basis(reference()["bases"][basis].dump());
  EXPECT_TRUE(stamp) << basis;
  if (!stamp) {
    return {};
  }
  const fb::BasisStamp normalized = stamp->primitive_normalized();
  const std::vector<fb::ShellStamp> *element = normalized.shells_for(z);
  EXPECT_NE(element, nullptr) << basis << " Z=" << z;
  std::vector<fb::Shell> shells;
  if (element != nullptr) {
    for (const fb::ShellStamp &s : *element) {
      shells.push_back(s.at(kOrigin));
    }
  }
  return shells;
}

// --- Integrals, closed forms ----------------------------------------------

TEST(AtomicIntegrals, NormalizedSPrimitiveMatchesClosedForms) {
  // One normalized s primitive: S = 1, T = 3a/2, V = -2 z sqrt(2a/pi),
  // (ss|ss) = 2 sqrt(a/pi).
  const double a = 0.7;
  const double norm = std::pow(2.0 * a / std::numbers::pi, 0.75);
  const std::vector<fb::Shell> shells{fb::Shell(0, {{a, norm}}, kOrigin)};
  const auto ints = fb::atomic_integrals(3, shells);
  ASSERT_TRUE(ints) << ints.error().message;
  ASSERT_EQ(ints->nao_sphe, 1U);
  EXPECT_NEAR(ints->overlap[0], 1.0, 1e-14);
  EXPECT_NEAR(ints->kinetic[0], 1.5 * a, 1e-14);
  EXPECT_NEAR(ints->nuclear[0], -2.0 * 3.0 * std::sqrt(2.0 * a / std::numbers::pi), 1e-14);
  EXPECT_NEAR(ints->repulsion[0], 2.0 * std::sqrt(a / std::numbers::pi), 1e-14);
}

TEST(AtomicIntegrals, TranslationInvariant) {
  const std::vector<fb::Shell> at_origin = reference_shells("cc-pvdz", 8);
  std::vector<fb::Shell> moved;
  for (const fb::Shell &s : at_origin) {
    moved.emplace_back(s.angular_momentum(), s.primitives(), fb::Vec<double, 3>{1.5, -0.3, 2.0});
  }
  const auto a = fb::atomic_integrals(8, at_origin);
  const auto b = fb::atomic_integrals(8, moved);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->repulsion, b->repulsion);
}

TEST(AtomicIntegrals, RepulsionHasEightfoldSymmetry) {
  const std::vector<fb::Shell> shells = reference_shells("cc-pvdz", 8);
  const auto ints = fb::atomic_integrals(8, shells);
  ASSERT_TRUE(ints);
  const std::size_t n = ints->nao_sphe;
  const auto at = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return ints->repulsion[i + (n * (j + (n * (k + (n * l)))))];
  };
  const double scale = max_abs(ints->repulsion);
  for (std::size_t l = 0; l < n; ++l) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
          const double v = at(i, j, k, l);
          EXPECT_NEAR(at(j, i, k, l), v, 1e-13 * scale);
          EXPECT_NEAR(at(i, j, l, k), v, 1e-13 * scale);
          EXPECT_NEAR(at(k, l, i, j), v, 1e-13 * scale);
        }
      }
    }
  }
}

// --- Integrals, against libcint ---------------------------------------------

// The oxygen of test/gtest's H2O: it is atom 0 at the origin, so its shells
// come first and its Cartesian block of S and T is one-center. V sums over all
// three nuclei, so it is not compared here (the SCF energies cover it).
class AtomicIntegralsGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(AtomicIntegralsGolden, OxygenBlockMatchesLibcint) {
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "one_electron" /
                                 (GetParam() + ".json"));
  ASSERT_TRUE(doc) << GetParam();
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    if (c != std::vector<double>{0.0, 0.0, 0.0}) {
      break;
    }
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(), kOrigin);
  }
  ASSERT_FALSE(shells.empty());
  const fb::BasisSet o(shells);
  const std::size_t nao = (*doc)["nao"].get<std::size_t>();
  const std::size_t ncart = o.nao_cart();
  const std::size_t nsph = o.nao_sphe();
  const std::vector<double> u = fb::spherical_transform_matrix(o);
  const auto block = [&](const char *name) {
    const auto all = (*doc)["arrays"][name]["values"].get<std::vector<double>>();
    std::vector<double> out(ncart * ncart);
    for (std::size_t j = 0; j < ncart; ++j) {
      for (std::size_t i = 0; i < ncart; ++i) {
        out[i + (ncart * j)] = all[(i * nao) + j];
      }
    }
    return sandwich(out, u, ncart, nsph);
  };

  const auto ints = fb::atomic_integrals(8, shells);
  ASSERT_TRUE(ints) << ints.error().message;
  ASSERT_EQ(ints->nao_sphe, nsph);
  for (const auto &[name, ours] :
       {std::pair{"int1e_ovlp", &ints->overlap}, std::pair{"int1e_kin", &ints->kinetic}}) {
    const std::vector<double> golden = block(name);
    const double tol = 1e-12 * max_abs(golden);
    for (std::size_t i = 0; i < golden.size(); ++i) {
      EXPECT_NEAR((*ours)[i], golden[i], tol) << name << " element " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(H2O, AtomicIntegralsGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

// --- The SCF, against PySCF -------------------------------------------------

struct ReferenceAtom {
  std::string basis;
  std::string symbol;
  std::int32_t z;
  double energy;
  std::vector<std::vector<double>> orbital_energies;
};

std::vector<ReferenceAtom> reference_atoms() {
  std::vector<ReferenceAtom> out;
  for (const auto &a : reference()["atoms"]) {
    out.push_back({a["basis"].get<std::string>(), a["symbol"].get<std::string>(),
                   a["z"].get<std::int32_t>(), a["energy"].get<double>(),
                   a["orbital_energies"].get<std::vector<std::vector<double>>>()});
  }
  return out;
}

TEST(AtomicScf, MatchesPyscfAtomSphAverageRhf) {
  const std::vector<ReferenceAtom> atoms = reference_atoms();
  ASSERT_FALSE(atoms.empty());
  for (const ReferenceAtom &ref : atoms) {
    SCOPED_TRACE(ref.symbol + "/" + ref.basis);
    const std::vector<fb::Shell> shells = reference_shells(ref.basis, ref.z);
    const auto scf = fb::atomic_scf(ref.z, shells);
    ASSERT_TRUE(scf) << scf.error().message;
    EXPECT_NEAR(scf->energy, ref.energy, 1e-8);
    ASSERT_EQ(scf->orbital_energies.size(), ref.orbital_energies.size());
    for (std::size_t l = 0; l < ref.orbital_energies.size(); ++l) {
      ASSERT_EQ(scf->orbital_energies[l].size(), ref.orbital_energies[l].size()) << "l=" << l;
      for (std::size_t k = 0; k < ref.orbital_energies[l].size(); ++k) {
        EXPECT_NEAR(scf->orbital_energies[l][k], ref.orbital_energies[l][k], 1e-6)
            << "l=" << l << " k=" << k;
      }
    }

    // The density holds Z electrons against S, and is symmetric.
    const auto ints = fb::atomic_integrals(ref.z, shells);
    ASSERT_TRUE(ints);
    EXPECT_NEAR(trace_product(scf->density, ints->overlap), ref.z, 1e-9);
    const std::size_t n = ints->nao_sphe;
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < j; ++i) {
        EXPECT_EQ(scf->density[i + (n * j)], scf->density[j + (n * i)]);
      }
    }
  }
}

TEST(AtomicScf, ClosedShellDensityIsIdempotent) {
  // Every orbital of Ne doubly occupied: D S D = 2 D.
  const std::vector<fb::Shell> shells = reference_shells("cc-pvdz", 10);
  const auto scf = fb::atomic_scf(10, shells);
  const auto ints = fb::atomic_integrals(10, shells);
  ASSERT_TRUE(scf && ints);
  const std::size_t n = ints->nao_sphe;
  const std::vector<double> dsd = matmul(matmul(scf->density, ints->overlap, n), scf->density, n);
  for (std::size_t i = 0; i < n * n; ++i) {
    EXPECT_NEAR(dsd[i], 2.0 * scf->density[i], 1e-8);
  }
}

TEST(AtomicScf, GhostAtomIsEmpty) {
  const auto scf = fb::atomic_scf(0, reference_shells("sto-3g", 1));
  ASSERT_TRUE(scf) << scf.error().message;
  EXPECT_EQ(scf->energy, 0.0);
  EXPECT_EQ(max_abs(scf->density), 0.0);
}

// --- Refusals ---------------------------------------------------------------

TEST(AtomicScf, RefusesNoShellsAndSplitCenters) {
  EXPECT_EQ(fb::atomic_scf(1, {}).error().kind, fb::ErrorKind::InvalidConfig);
  const std::vector<fb::Shell> split{fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                                     fb::Shell(0, {{0.5, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})};
  EXPECT_EQ(fb::atomic_scf(2, split).error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(AtomicScf, RefusesABasisThatCannotSeatTheAtom) {
  // O needs a p orbital; two s shells have none.
  const std::vector<fb::Shell> s_only{fb::Shell(0, {{10.0, 1.0}}, kOrigin),
                                      fb::Shell(0, {{1.0, 1.0}}, kOrigin)};
  EXPECT_EQ(fb::atomic_scf(8, s_only).error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(AtomicScf, ReportsAnUnconvergedRun) {
  const auto scf = fb::atomic_scf(8, reference_shells("cc-pvdz", 8), 1e-9, 1);
  ASSERT_FALSE(scf);
  EXPECT_EQ(scf.error().kind, fb::ErrorKind::MaxIterationsExceeded);
}

// --- The molecular guess ----------------------------------------------------

TEST(AtomicScfDensity, IsBlockDiagonalOverTheAtoms) {
  // H2O laid out as place would: O's shells, then each H's, grouped per atom.
  const std::vector<fb::Atom> atoms{
      fb::Atom{kOrigin, fb::Element::O},
      fb::Atom{fb::Vec<double, 3>{1.43, 0.21, -1.07}, fb::Element::H},
      fb::Atom{fb::Vec<double, 3>{-1.38, 0.35, -1.12}, fb::Element::H},
  };
  const std::vector<fb::Shell> o = reference_shells("sto-3g", 8);
  const std::vector<fb::Shell> h = reference_shells("sto-3g", 1);
  std::vector<fb::Shell> shells;
  std::vector<std::uint32_t> offsets{0};
  for (const fb::Atom &atom : atoms) {
    for (const fb::Shell &s : (atom.element == fb::Element::O) ? o : h) {
      shells.emplace_back(s.angular_momentum(), s.primitives(), atom.position);
    }
    offsets.push_back(static_cast<std::uint32_t>(shells.size()));
  }
  const fb::BasisSet basis(shells, offsets);

  const auto density = fb::atomic_scf_density(basis, atoms);
  ASSERT_TRUE(density) << density.error().message;
  const std::size_t n = basis.nao_sphe();
  ASSERT_EQ(n, 7U);
  const auto o_scf = fb::atomic_scf(8, o);
  const auto h_scf = fb::atomic_scf(1, h);
  ASSERT_TRUE(o_scf && h_scf);
  // O is the leading 5 x 5 block, each H one diagonal entry; nothing between.
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      double expected = 0.0;
      if (i < 5 && j < 5) {
        expected = o_scf->density[i + (5 * j)];
      } else if (i == j) {
        expected = h_scf->density[0];
      }
      EXPECT_EQ((*density)[i + (n * j)], expected) << i << ", " << j;
    }
  }
}

TEST(AtomicScfDensity, RefusesAMismatchedAtomCount) {
  const std::vector<fb::Atom> one{fb::Atom{kOrigin, fb::Element::H}};
  const fb::BasisSet ungrouped(reference_shells("sto-3g", 1));
  EXPECT_EQ(fb::atomic_scf_density(ungrouped, one).error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
