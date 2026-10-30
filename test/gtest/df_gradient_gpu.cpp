// GPU cases for integrals.eri.density_fitting.gradient: SUM_PQ W_PQ
// d(P|Q)/dR and SUM_mnP Gam_mnP d(mn|P)/dR over FelizJueves's skew water in
// cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux to f), cc-pVTZ / cc-pVTZ-JKFIT
// (to f, to g) and cc-pVTZ / cc-pVTZ-acCD-6-6 (to f, to i: (ff|i), (i|i)),
// against PySCF's int2c2e_ip1 and int3c2e_ip1 / ip2 contracted with the same
// recipe matrices (data/gradient/gradient.json, test/gtest/data/README.md);
// each gradient's sum over atoms; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.gradient;

namespace {

namespace fb = free_breakfast;

// The issue's bars: 1e-10 of max|G| against PySCF, and the sum over atoms of
// each component below 1e-10.
constexpr double kRelTol = 1e-10;
// The acCD-6-6 metric alone: libcint's int2c2e_ip1 is itself off ~1.7e-11 on
// that set's tight O s and p contractions, where ours matches a 40-digit
// mpmath reference of the same contraction to ~1e-13 (#494).
constexpr double kAcCdMetricRelTol = 1e-9;
constexpr double kSumTol = 1e-10;

const std::filesystem::path kDataFile =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "gradient" / "gradient.json";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_three_center_gpu.cpp).
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
// unit-normalised one, so the recipe's matrices are divided by the N of
// each index (test/gtest/data/README.md, "Normalisation").
std::vector<double> norms(const fb::BasisSet &basis) {
  std::vector<double> n;
  for (const fb::Shell &shell : basis.shells()) {
    n.insert(n.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
  }
  return n;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// @p ours against the reference rows @p ref, to @p rel_tol of max|ref|, and each
// component's sum over atoms to kSumTol.
void compare(const std::string &what, const std::vector<double> &ours, const fb::json &ref,
             const double rel_tol = kRelTol) {
  const auto rows = ref.get<std::vector<std::vector<double>>>();
  ASSERT_EQ(ours.size(), 3 * rows.size()) << what;
  double scale = 0.0;
  double worst = 0.0;
  std::array<double, 3> sum{};
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      scale = std::max(scale, std::abs(rows[a][k]));
      worst = std::max(worst, std::abs(ours[(3 * a) + k] - rows[a][k]));
      sum[k] += ours[(3 * a) + k];
    }
  }
  EXPECT_LE(worst, rel_tol * scale)
      << what << ": worst |G - PySCF| " << worst << ", max|G| " << scale;
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_LE(std::abs(sum[k]), kSumTol) << what << ": SUM_A G[A][" << k << "] = " << sum[k];
  }
  std::cout << std::format("[ dfgrad   ] {}: max|G| {:.4f}, worst |G - PySCF| / max|G| {:.3e}, "
                           "max |SUM_A G| {:.3e}\n",
                           what, scale, worst / scale,
                           std::max({std::abs(sum[0]), std::abs(sum[1]), std::abs(sum[2])}));
}

// The case's two bases over its molecule, placed as the DF steps will, their
// shells checked against the manifest's (atom, l); then both gradients at the
// recipe's W and Gam over this tree's functions, against PySCF's.
void check_gradient(const std::string &name, const std::int32_t orbital_l,
                    const std::int32_t aux_l) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataFile);
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = nullptr;
  for (const fb::json &entry : (*doc)["density_fitting"]) {
    if (entry["name"].get<std::string>() == name) {
      c = &entry;
    }
  }
  ASSERT_NE(c, nullptr) << "no density_fitting case " << name;

  fb::DeviceResources res{0};
  std::vector<fb::Atom> atoms;
  for (const auto &atom : (*c)["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  const fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_basis((*c)["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis((*c)["aux_basis"].get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  const auto aux = fb::place(aux_stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;

  const auto nao = (*c)["nao"].get<std::size_t>();
  const auto naux = (*c)["naux"].get<std::size_t>();
  ASSERT_EQ(basis->nao_sphe(), nao);
  ASSERT_EQ(aux->nao_sphe(), naux);
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
  ASSERT_TRUE(same_shells(*basis, (*c)["shells"])) << name << ": orbital shells";
  ASSERT_TRUE(same_shells(*aux, (*c)["aux_shells"])) << name << ": aux shells";
  const auto lmax = [](const fb::BasisSet &b) {
    std::int32_t l = 0;
    for (const fb::Shell &shell : b.shells()) {
      l = std::max(l, shell.angular_momentum());
    }
    return l;
  };
  EXPECT_EQ(lmax(*basis), orbital_l);
  EXPECT_EQ(lmax(*aux), aux_l);

  const std::vector<double> n_ao = norms(*basis);
  const std::vector<double> n_aux = norms(*aux);
  // The recipe, one IEEE division per element as the tool wrote it, then
  // divided by our norms. W column-major (it is symmetric anyway); Gam in
  // three_center's layout, P + naux (n + nao m).
  std::vector<double> w(naux * naux);
  for (std::size_t q = 0; q < naux; ++q) {
    for (std::size_t p = 0; p < naux; ++p) {
      w[p + (naux * q)] = (1.0 / static_cast<double>(p + q + 1)) / (n_aux[p] * n_aux[q]);
    }
  }
  std::vector<double> gamma(nao * nao * naux);
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t p = 0; p < naux; ++p) {
        gamma[p + (naux * (n + (nao * m)))] =
            (1.0 / static_cast<double>(m + n + p + 1)) / (n_ao[m] * n_ao[n] * n_aux[p]);
      }
    }
  }
  const fb::Buffers::device<double> d_w = upload(res, w);
  const fb::Buffers::device<double> d_gamma = upload(res, gamma);

  const auto metric = fb::metric_gradient(res, *aux, d_w.data());
  ASSERT_TRUE(metric) << metric.error().message;
  compare(name + " metric", *metric, (*c)["metric"],
          name == "h2o_ccpvtz_accd66" ? kAcCdMetricRelTol : kRelTol);

  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto three_center = fb::three_center_gradient(res, *basis, *classes, *aux, d_gamma.data());
  ASSERT_TRUE(three_center) << three_center.error().message;
  compare(name + " three_center", *three_center, (*c)["three_center"]["total"]);
}

TEST(DfGradient, DzJkfitMatchesPyscf) {
  check_gradient("h2o_ccpvdz", 2, 3);
}

TEST(DfGradient, TzJkfitMatchesPyscf) {
  check_gradient("h2o_ccpvtz", 3, 4);
}

// The aux i shells reach (ff|i) and (i|i), the ladders' top classes.
TEST(DfGradient, TzAcCdThroughFfIMatchesPyscf) {
  check_gradient("h2o_ccpvtz_accd66", 3, 6);
}

// A basis not made by place() has no atoms to put a gradient on, and a shell
// past the ladders has no kernel; each refused before anything is launched.
TEST(DfGradient, Refusals) {
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
    const auto g = fb::metric_gradient(res, aux, scratch.data());
    ASSERT_FALSE(g) << says;
    EXPECT_EQ(g.error().kind, fb::ErrorKind::InvalidConfig) << g.error().message;
    EXPECT_NE(g.error().message.find(says), std::string::npos) << g.error().message;
  };
  metric_refusal(unplaced, "metric_gradient: the aux basis has no atoms");
  metric_refusal(k_aux, "no kernel for an aux shell of momentum 7");

  const auto three_center_refusal = [&](const fb::BasisSet &basis, const fb::BasisSet &aux,
                                        const fb::ErrorKind kind, const std::string &says) {
    const auto classes = fb::primitive_pair_classes(res, basis);
    ASSERT_TRUE(classes) << classes.error().message;
    const auto g = fb::three_center_gradient(res, basis, *classes, aux, scratch.data());
    ASSERT_FALSE(g) << says;
    EXPECT_EQ(g.error().kind, kind) << g.error().message;
    EXPECT_NE(g.error().message.find(says), std::string::npos) << g.error().message;
  };
  three_center_refusal(one_s, unplaced, fb::ErrorKind::InvalidConfig,
                       "the bases have 1 and 0 atoms");
  three_center_refusal(g_orbital, one_s, fb::ErrorKind::Unsupported,
                       "orbital shell 1 has momentum 4");
  three_center_refusal(one_s, k_aux, fb::ErrorKind::Unsupported, "aux shell 0 has momentum 7");
}

} // namespace
