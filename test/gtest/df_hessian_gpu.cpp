// GPU cases for integrals.eri.density_fitting.hessian: SUM_PQ W_PQ
// d2(P|Q)/dR dR and SUM_mnP Gam_mnP d2(mn|P)/dR dR over the skew water in
// cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux to f), cc-pVTZ / cc-pVTZ-JKFIT
// (to f, to g) and cc-pVTZ / cc-pVTZ-acCD-6-6 (to f, to i: (ff|i), (i|i)),
// against PySCF's int2c2e_ipip1 / ip1ip2 and int3c2e_ipip1 / ipvip1 / ip1ip2
// contracted with the same recipe matrices, which have no symmetry
// (data/hessian/df_hessian.json, test/gtest/data/README.md); the translational
// sum rule of every class on its own; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bars: 1e-10 of max|H| against PySCF, and the sum over atoms of
// each column below 1e-10 of max|H|, per class.
constexpr double kRelTol = 1e-10;
// The acCD-6-6 metric alone, as df_gradient_gpu.cpp's: libcint's
// int2c2e_ipip1 / ip1ip2 are themselves off on that set's tight O s and p
// contractions. Against a 40-digit mpmath McMurchie-Davidson reference of the
// same contraction (W restricted to the aux s, then s and p functions),
// PySCF is off by 2.8e-12, then 4.0e-11, and ours by 2.1e-14, then 2.6e-13;
// the whole case misses PySCF by 3.2e-11 at max|H| 0.139 (2.3e-10), the miss
// arriving with the p functions and flat from d up (#593's PR has the table).
constexpr double kAcCdMetricRelTol = 1e-9;
constexpr double kSumRelTol = 1e-10;

const std::filesystem::path kDataFile =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "hessian" / "df_hessian.json";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_gradient_gpu.cpp).
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

// Each spherical function's norm N_i: function i here is N_i times PySCF's
// unit-normalised one, so the recipe's matrices are divided by the N of each
// index (test/gtest/data/README.md, "Normalisation").
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

std::int32_t max_momentum(const fb::BasisSet &basis) {
  std::int32_t l = 0;
  for (const fb::Shell &shell : basis.shells()) {
    l = std::max(l, shell.angular_momentum());
  }
  return l;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

double max_abs(const std::vector<double> &h) {
  double m = 0.0;
  for (const double x : h) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

// The worst |SUM_A H[A k][j]| over every column j and axis k.
double sum_rule_drift(const std::vector<double> &h, const std::size_t n_atom) {
  const std::size_t n = 3 * n_atom;
  double worst = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t a = 0; a < n_atom; ++a) {
        sum += h[(((3 * a) + k) * n) + j];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

// @p ours against the reference rows @p ref, to @p rel_tol of max|ref|, and its
// sum rule.
void compare(const std::string &what, const std::vector<double> &ours, const fb::json &ref,
             const double rel_tol = kRelTol) {
  const auto rows = ref.get<std::vector<std::vector<double>>>();
  ASSERT_EQ(ours.size(), rows.size() * rows.size()) << what;
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    ASSERT_EQ(rows[i].size(), rows.size()) << what;
    for (std::size_t j = 0; j < rows.size(); ++j) {
      scale = std::max(scale, std::abs(rows[i][j]));
      worst = std::max(worst, std::abs(ours[(i * rows.size()) + j] - rows[i][j]));
    }
  }
  const double drift = sum_rule_drift(ours, rows.size() / 3);
  EXPECT_LE(worst, rel_tol * scale)
      << what << ": worst |H - PySCF| " << worst << ", max|H| " << scale;
  EXPECT_LE(drift, kSumRelTol * scale) << what << ": max |SUM_A H[A k][j]| " << drift;
  std::cout << std::format("[ dfhess   ] {}: max|H| {:.4f}, worst |H - PySCF| / max|H| {:.3e}, "
                           "max |SUM_A H| / max|H| {:.3e}\n",
                           what, scale, worst / scale, drift / scale);
}

// One reference case: its two bases over its molecule, placed as the DF steps
// will, their shells checked against the manifest's (atom, l), and the
// recipe's W and Gam over this tree's functions, on the device.
struct Case {
  fb::json ref;
  fb::DeviceResources res{0};
  std::optional<fb::Molecule> molecule;
  std::optional<fb::BasisSet> basis;
  std::optional<fb::BasisSet> aux;
  std::size_t nao = 0;
  std::size_t naux = 0;
  std::vector<double> w;
  std::vector<double> gamma;
};

void load(Case &c, const std::string &name, const std::int32_t orbital_l,
          const std::int32_t aux_l) {
  const auto doc = fb::read_json(kDataFile);
  ASSERT_TRUE(doc) << doc.error().message;
  for (const fb::json &entry : (*doc)["density_fitting"]) {
    if (entry["name"].get<std::string>() == name) {
      c.ref = entry;
    }
  }
  ASSERT_FALSE(c.ref.is_null()) << "no density_fitting case " << name;

  std::vector<fb::Atom> atoms;
  for (const auto &atom : c.ref["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  c.molecule.emplace(c.res, std::move(atoms));
  const auto stamp = fb::resolve_basis(c.ref["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  auto basis = fb::place(stamp->primitive_normalized(), *c.molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis(c.ref["aux_basis"].get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  auto aux = fb::place(aux_stamp->primitive_normalized(), *c.molecule);
  ASSERT_TRUE(aux) << aux.error().message;
  c.basis = std::move(*basis);
  c.aux = std::move(*aux);

  c.nao = c.ref["nao"].get<std::size_t>();
  c.naux = c.ref["naux"].get<std::size_t>();
  ASSERT_EQ(c.basis->nao_sphe(), c.nao);
  ASSERT_EQ(c.aux->nao_sphe(), c.naux);
  // PySCF's shells are generally contracted, ours one contraction each: each
  // of its (atom, l, nctr) is nctr of ours in a row.
  const auto same_shells = [](const fb::BasisSet &ours, const fb::json &theirs) {
    std::vector<std::pair<std::int32_t, std::int32_t>> expanded;
    for (const auto &s : theirs) {
      expanded.insert(expanded.end(), s["nctr"].get<std::size_t>(),
                      {s["atom"].get<std::int32_t>(), s["l"].get<std::int32_t>()});
    }
    std::vector<std::pair<std::int32_t, std::int32_t>> mine;
    for (std::uint32_t a = 0; a < ours.natoms(); ++a) {
      for (std::uint32_t i = 0; i < ours.atom_shell_count(a); ++i) {
        mine.emplace_back(static_cast<std::int32_t>(a),
                          ours.shells()[ours.atom_shell_offset(a) + i].angular_momentum());
      }
    }
    return mine == expanded;
  };
  ASSERT_TRUE(same_shells(*c.basis, c.ref["shells"])) << name << ": orbital shells";
  ASSERT_TRUE(same_shells(*c.aux, c.ref["aux_shells"])) << name << ": aux shells";
  EXPECT_EQ(max_momentum(*c.basis), orbital_l);
  EXPECT_EQ(max_momentum(*c.aux), aux_l);

  // The recipe, one IEEE division per element as the tool wrote it, then
  // divided by our norms. W column-major, element (P, Q) at P + naux Q; Gam
  // in three_center's layout, P + naux (n + nao m).
  const std::vector<double> n_ao = norms(*c.basis);
  const std::vector<double> n_aux = norms(*c.aux);
  c.w.resize(c.naux * c.naux);
  for (std::size_t q = 0; q < c.naux; ++q) {
    for (std::size_t p = 0; p < c.naux; ++p) {
      c.w[p + (c.naux * q)] = (1.0 / static_cast<double>(p + (2 * q) + 1)) / (n_aux[p] * n_aux[q]);
    }
  }
  c.gamma.resize(c.nao * c.nao * c.naux);
  for (std::size_t m = 0; m < c.nao; ++m) {
    for (std::size_t n = 0; n < c.nao; ++n) {
      for (std::size_t p = 0; p < c.naux; ++p) {
        c.gamma[p + (c.naux * (n + (c.nao * m)))] =
            (1.0 / static_cast<double>(m + (2 * n) + (3 * p) + 1)) / (n_ao[m] * n_ao[n] * n_aux[p]);
      }
    }
  }
}

void check_hessian(const std::string &name, const std::int32_t orbital_l,
                   const std::int32_t aux_l) {
  FBF_REQUIRE_GPU();
  Case c;
  load(c, name, orbital_l, aux_l);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }
  const fb::Buffers::device<double> d_w = upload(c.res, c.w);
  const fb::Buffers::device<double> d_gamma = upload(c.res, c.gamma);

  const auto metric = fb::metric_hessian(c.res, *c.aux, d_w.data());
  ASSERT_TRUE(metric) << metric.error().message;
  compare(name + " metric", *metric, c.ref["metric"],
          name == "h2o_ccpvtz_accd66" ? kAcCdMetricRelTol : kRelTol);

  const auto classes = fb::primitive_pair_classes(c.res, *c.basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto three_center =
      fb::three_center_hessian(c.res, *c.basis, *classes, *c.aux, d_gamma.data());
  ASSERT_TRUE(three_center) << three_center.error().message;
  compare(name + " three_center", *three_center, c.ref["three_center"]);
}

TEST(DfHessian, DzJkfitMatchesPyscf) {
  check_hessian("h2o_ccpvdz", 2, 3);
}

TEST(DfHessian, TzJkfitMatchesPyscf) {
  check_hessian("h2o_ccpvtz", 3, 4);
}

// The aux i shells reach (ff|i) and (i|i), the ladders' top classes.
TEST(DfHessian, TzAcCdThroughFfIMatchesPyscf) {
  FBF_SKIP_IF_FAST_TESTS(); // about 44 s on the RTX 3080; DzJkfit is the family's cheapest
  check_hessian("h2o_ccpvtz_accd66", 3, 6);
}

// Every class of the ladders on its own, SUM_A H[A k][j] = 0: the kernels
// differentiate every centre, the aux one by the Hermite shift, and infer
// none from the others, so the rule is a check of each class, not an
// identity. Each class gets its own three-atom system (the skew water's
// centres): for d2eri3<la, lb, lP> an la shell on the first atom, an lb shell
// on the second and an lP aux shell on the third, the recipe kept only where
// m and n are on different shells, so (la, lb, lP) is the one class that
// reaches it; for d2metric<lP, lQ> aux shells lP and lQ on the second and
// third, W kept only across them. One primitive a shell: a cell is one
// thread, and with two a shell the whole sweep took 42 s on the RTX 3080
// (5.7 s with one); contraction is the PySCF cases' to check.
TEST(DfHessian, EveryClassKeepsTheSumRule) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const std::array<fb::Vec<double, 3>, 3> centre{fb::Vec<double, 3>{0.0, 0.0, 0.0},
                                                 fb::Vec<double, 3>{1.43, 0.21, -1.07},
                                                 fb::Vec<double, 3>{-1.38, 0.35, -1.12}};
  const auto shell = [&](const std::int32_t l, const std::size_t atom) {
    return fb::Shell{l, std::vector<double>{0.9}, std::vector<double>{1.0}, centre[atom]};
  };
  const auto check = [&](const std::string &what, const std::vector<double> &h) {
    const double scale = max_abs(h);
    EXPECT_GT(scale, 0.0) << what << ": the masked recipe reaches no integral";
    if (scale == 0.0) {
      return 0.0;
    }
    const double drift = sum_rule_drift(h, centre.size());
    EXPECT_LE(drift, kSumRelTol * scale) << what << ": max |SUM_A H| " << drift;
    return drift / scale;
  };

  double worst_metric = 0.0;
  std::int32_t metric_classes = 0;
  for (std::int32_t lp = 0; lp <= 6; ++lp) {
    for (std::int32_t lq = lp; lq <= 6; ++lq) {
      const fb::BasisSet aux{{shell(lp, 1), shell(lq, 2)}, {0, 0, 1, 2}};
      const auto np = static_cast<std::size_t>((2 * lp) + 1);
      const std::size_t n = aux.nao_sphe();
      std::vector<double> w(n * n, 0.0);
      for (std::size_t q = 0; q < n; ++q) {
        for (std::size_t p = 0; p < n; ++p) {
          if ((p < np) != (q < np)) {
            w[p + (n * q)] = 1.0 / static_cast<double>(p + (2 * q) + 1);
          }
        }
      }
      const fb::Buffers::device<double> d_w = upload(res, w);
      const auto h = fb::metric_hessian(res, aux, d_w.data());
      ASSERT_TRUE(h) << h.error().message;
      worst_metric = std::max(worst_metric, check(std::format("metric ({}|{})", lp, lq), *h));
      ++metric_classes;
    }
  }

  double worst_three = 0.0;
  std::int32_t three_classes = 0;
  for (std::int32_t la = 0; la <= 3; ++la) {
    for (std::int32_t lb = la; lb <= 3; ++lb) {
      const fb::BasisSet basis{{shell(la, 0), shell(lb, 1)}, {0, 1, 2, 2}};
      const auto classes = fb::primitive_pair_classes(res, basis);
      ASSERT_TRUE(classes) << classes.error().message;
      const auto na = static_cast<std::size_t>((2 * la) + 1);
      const std::size_t nao = basis.nao_sphe();
      for (std::int32_t lp = 0; lp <= 6; ++lp) {
        const fb::BasisSet aux{{shell(lp, 2)}, {0, 0, 0, 1}};
        const std::size_t naux = aux.nao_sphe();
        std::vector<double> gamma(nao * nao * naux, 0.0);
        for (std::size_t m = 0; m < nao; ++m) {
          for (std::size_t n = 0; n < nao; ++n) {
            if ((m < na) == (n < na)) {
              continue;
            }
            for (std::size_t p = 0; p < naux; ++p) {
              gamma[p + (naux * (n + (nao * m)))] =
                  1.0 / static_cast<double>(m + (2 * n) + (3 * p) + 1);
            }
          }
        }
        const fb::Buffers::device<double> d_gamma = upload(res, gamma);
        const auto h = fb::three_center_hessian(res, basis, *classes, aux, d_gamma.data());
        ASSERT_TRUE(h) << h.error().message;
        worst_three =
            std::max(worst_three, check(std::format("three_center ({}{}|{})", la, lb, lp), *h));
        ++three_classes;
      }
    }
  }
  EXPECT_EQ(metric_classes, 28);
  EXPECT_EQ(three_classes, 70);
  std::cout << std::format("[ dfhess   ] sum rule, worst |SUM_A H| / max|H|: metric {:.3e} over "
                           "{} classes, three_center {:.3e} over {}\n",
                           worst_metric, metric_classes, worst_three, three_classes);
}

// A basis not made by place() has no atoms to put a Hessian on, and a shell
// past the ladders has no kernel; each refused before anything is launched.
TEST(DfHessian, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto placed = [](std::vector<fb::Shell> shells) {
    const auto count = static_cast<std::uint32_t>(shells.size());
    return fb::BasisSet{std::move(shells), {0, count}};
  };
  const fb::Shell s{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}};
  const fb::BasisSet unplaced(std::vector<fb::Shell>{s});
  const fb::BasisSet one_s = placed({s});
  const fb::BasisSet g_orbital = placed({s, fb::Shell{4, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  const fb::BasisSet k_aux = placed({fb::Shell{7, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  fb::Buffers::device<double> scratch(64, res.device());

  const auto metric_refusal = [&](const fb::BasisSet &aux, const std::string &says) {
    const auto h = fb::metric_hessian(res, aux, scratch.data());
    ASSERT_FALSE(h) << says;
    EXPECT_EQ(h.error().kind, fb::ErrorKind::InvalidConfig) << h.error().message;
    EXPECT_NE(h.error().message.find(says), std::string::npos) << h.error().message;
  };
  metric_refusal(unplaced, "metric_hessian: the aux basis has no atoms");
  metric_refusal(k_aux, "no kernel for an aux shell of momentum 7");

  const auto three_center_refusal = [&](const fb::BasisSet &basis, const fb::BasisSet &aux,
                                        const fb::ErrorKind kind, const std::string &says) {
    const auto classes = fb::primitive_pair_classes(res, basis);
    ASSERT_TRUE(classes) << classes.error().message;
    const auto h = fb::three_center_hessian(res, basis, *classes, aux, scratch.data());
    ASSERT_FALSE(h) << says;
    EXPECT_EQ(h.error().kind, kind) << h.error().message;
    EXPECT_NE(h.error().message.find(says), std::string::npos) << h.error().message;
  };
  three_center_refusal(one_s, unplaced, fb::ErrorKind::InvalidConfig,
                       "the bases have 1 and 0 atoms");
  three_center_refusal(g_orbital, one_s, fb::ErrorKind::Unsupported,
                       "orbital shell 1 has momentum 4");
  three_center_refusal(one_s, k_aux, fb::ErrorKind::Unsupported, "aux shell 0 has momentum 7");
}

} // namespace
