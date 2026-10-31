// GPU cases for the density-fitted skeleton Fock derivatives and their
// building blocks (integrals.eri.density_fitting.gradient): df_jk_derivative's
// J^(x) and K^(x) and J^(x) - 1/2 K^(x) against PySCF's
// df.hessian.rhf (make_h1's generator, the same aux set) for the skew water in
// cc-pVDZ / cc-pVDZ-JKFIT, cc-pVTZ / cc-pVTZ-JKFIT and cc-pVTZ /
// cc-pVTZ-acCD-6-6 (data/df_jk_derivative/, test/gtest/data/README.md); each
// contracted with D against gradient::df_rhf's Coulomb and exchange terms;
// metric_derivative and three_center_derivative contracted with the recipe
// matrices against metric_gradient and three_center_gradient; and the
// refusals.

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
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.gradient;

namespace {

namespace fb = free_breakfast;

// The issue's bar: ~1e-10 of the largest element against PySCF over the same
// aux set; the contractions with D against df_rhf, and the building blocks
// against the gradients, are this tree against itself.
constexpr double kRelTol = 1e-10;
constexpr double kSelfRelTol = 1e-11;

const std::filesystem::path kDataDir =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "df_jk_derivative";

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
// unit-normalised one (test/gtest/data/README.md, "Normalisation").
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

std::vector<double> download(fb::DeviceResources &res, const double *device,
                             const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

std::vector<double> read_array(const std::string &file, const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
  return a->data;
}

// The worst |ours - ref| over max|ref|, held to @p rel_tol, and printed.
void compare(const std::string &what, const std::vector<double> &ours,
             const std::vector<double> &ref, const double rel_tol) {
  ASSERT_EQ(ours.size(), ref.size()) << what;
  double scale = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    scale = std::max(scale, std::abs(ref[i]));
    worst = std::max(worst, std::abs(ours[i] - ref[i]));
  }
  EXPECT_LE(worst, rel_tol * scale) << what << ": worst " << worst << ", max " << scale;
  std::cout << std::format("[ dfjkx    ] {}: max {:.4f}, worst / max {:.3e}\n", what, scale,
                           worst / scale);
}

fb::Molecule molecule_of(fb::DeviceResources &res, const fb::json &c) {
  std::vector<fb::Atom> atoms;
  for (const auto &atom : c["atoms"]) {
    const auto xyz = atom["coord"].get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
  }
  return fb::Molecule(res, std::move(atoms));
}

const fb::json *find_case(const fb::json &doc, const std::string &name) {
  for (const fb::json &entry : doc["cases"]) {
    if (entry["name"].get<std::string>() == name) {
      return &entry;
    }
  }
  return nullptr;
}

// The case's bases placed as the DF steps place them, the fit B and its L
// built as the Eri step builds them, PySCF's orbitals made ours (rows over
// N); then df_jk_derivative against PySCF, and contracted with D against
// gradient::df_rhf's two-electron terms over the same orbitals.
void check_case(const std::string &name) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_jk_derivative.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = find_case(*doc, name);
  ASSERT_NE(c, nullptr) << "no case " << name;

  fb::DeviceResources res{0};
  const fb::Molecule molecule = molecule_of(res, *c);
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
  const auto nocc = (*c)["nocc"].get<std::int32_t>();
  ASSERT_EQ(basis->nao_sphe(), nao);
  ASSERT_EQ(aux->nao_sphe(), naux);
  const std::size_t n_x = 3 * basis->natoms();
  const std::size_t nn = nao * nao;

  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  auto u = fb::upload_spherical_transform_matrix(res, *aux);
  ASSERT_TRUE(u) << u.error().message;
  fb::MatrixAOPair metric(res, *aux, std::move(*u));
  fb::Buffers::device<double> b(naux * nn, res.device());
  ASSERT_TRUE(fb::metric(res, *aux, metric));
  ASSERT_TRUE(fb::three_center(res, *basis, *classes, *aux, b.data()));
  ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nao), static_cast<std::int32_t>(naux),
                              metric.spherical().data(), b.data()));

  const std::vector<double> n_ao = norms(*basis);
  std::vector<double> c_host = read_array((*c)["c"].get<std::string>(), {nao, nao});
  for (std::size_t j = 0; j < nao; ++j) {
    for (std::size_t m = 0; m < nao; ++m) {
      c_host[m + (nao * j)] /= n_ao[m];
    }
  }
  fb::SphericalMatrixAO c_mat(res, *basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(c_mat.data(), c_host.data(), sizeof(double) * nn,
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync());

  fb::Buffers::device<double> jx(n_x * nn, res.device());
  fb::Buffers::device<double> kx(n_x * nn, res.device());
  fb::Buffers::device<double> g(naux * n_x, res.device());
  // All-ones bytes are a NaN: an element the call does not write stays one.
  ASSERT_EQ(wwr::wwrMemsetAsync(g.data(), 0xFF, sizeof(double) * naux * n_x, res.stream()),
            wwr::wwrSuccess);
  const auto start = std::chrono::steady_clock::now();
  const fb::Status s =
      fb::df_jk_derivative(res, *basis, *classes, *aux, b.data(), metric.spherical().data(),
                           c_mat.data(), nocc, jx.data(), kx.data(), g.data());
  ASSERT_TRUE(s) << s.error().message;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const std::vector<double> ours_j = download(res, jx.data(), n_x * nn);
  const std::vector<double> ours_k = download(res, kx.data(), n_x * nn);
  const std::vector<double> ours_g = download(res, g.data(), naux * n_x);

  // PySCF's, made ours: J^(x)_mn N_m N_n.
  std::vector<double> ref_j = read_array((*c)["jx"].get<std::string>(), {nao, nao, n_x});
  std::vector<double> ref_k = read_array((*c)["kx"].get<std::string>(), {nao, nao, n_x});
  for (std::size_t x = 0; x < n_x; ++x) {
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t m = 0; m < nao; ++m) {
        const double scale = n_ao[m] * n_ao[n];
        ref_j[m + (nao * n) + (nn * x)] *= scale;
        ref_k[m + (nao * n) + (nn * x)] *= scale;
      }
    }
  }
  std::vector<double> ours_f(n_x * nn);
  std::vector<double> ref_f(n_x * nn);
  for (std::size_t i = 0; i < ours_f.size(); ++i) {
    ours_f[i] = ours_j[i] - (0.5 * ours_k[i]);
    ref_f[i] = ref_j[i] - (0.5 * ref_k[i]);
  }
  compare(name + " J^(x)", ours_j, ref_j, kRelTol);
  compare(name + " K^(x)", ours_k, ref_k, kRelTol);
  compare(name + " J^(x) - K^(x)/2 (make_h1 - hcore)", ours_f, ref_f, kRelTol);
  // u^x - M^x d is checked through J^(x), whose T M^-1 (u^x - M^x d) it is:
  // alone it carries d, which is only as well determined as M is
  // conditioned, so no independent integral code matches it to 1e-10
  // (tools/emit/df_jk_derivative_reference.py has the numbers). Here, only
  // that every element was written.
  EXPECT_TRUE(std::ranges::all_of(ours_g, [](const double v) { return std::isfinite(v); }));
  std::cout << std::format("[ dfjkx    ] {}: {} orbital, {} fitting functions, {:.2f} s\n", name,
                           nao, naux, elapsed);

  // 1/2 tr(D J^(x)) and -1/4 tr(D K^(x)) are df_rhf's Coulomb and exchange
  // terms at the same orbitals (the other terms do not matter here).
  std::vector<double> d(nn, 0.0);
  for (std::size_t n = 0; n < nao; ++n) {
    for (std::size_t m = 0; m < nao; ++m) {
      for (std::size_t i = 0; i < static_cast<std::size_t>(nocc); ++i) {
        d[m + (nao * n)] += 2.0 * c_host[m + (nao * i)] * c_host[n + (nao * i)];
      }
    }
  }
  std::vector<double> coulomb(n_x, 0.0);
  std::vector<double> exchange(n_x, 0.0);
  for (std::size_t x = 0; x < n_x; ++x) {
    for (std::size_t i = 0; i < nn; ++i) {
      coulomb[x] += 0.5 * d[i] * ours_j[(nn * x) + i];
      exchange[x] -= 0.25 * d[i] * ours_k[(nn * x) + i];
    }
  }
  const auto energies = (*c)["mo_energies"].get<std::vector<double>>();
  const auto terms = fb::gradient::df_rhf(res, molecule, *basis, *classes, *aux, b.data(),
                                          metric.spherical().data(), c_mat, energies, nocc);
  ASSERT_TRUE(terms) << terms.error().message;
  compare(name + " 1/2 tr(D J^(x)) vs df_rhf coulomb", coulomb, terms->coulomb, kSelfRelTol);
  compare(name + " -1/4 tr(D K^(x)) vs df_rhf exchange", exchange, terms->exchange, kSelfRelTol);
}

TEST(DfJkDerivative, DzJkfitMatchesPyscf) {
  check_case("h2o_ccpvdz");
}

TEST(DfJkDerivative, TzJkfitMatchesPyscf) {
  check_case("h2o_ccpvtz");
}

// The aux i shells reach (ff|i) and (i|i), the ladders' top classes. About
// 23 s on an RTX 3080 (the call itself 9 s, then df_rhf), past
// slow_test.h's ten, and the family's cheapest cases run in the fast tier.
TEST(DfJkDerivative, TzAcCdThroughFfIMatchesPyscf) {
  FBF_SKIP_IF_FAST_TESTS();
  check_case("h2o_ccpvtz_accd66");
}

// metric_derivative and three_center_derivative, put back together into
// SUM W d(P|Q)/dR and SUM Gam d(mn|P)/dR on the host by their documented
// centre rules, against metric_gradient and three_center_gradient (which
// df_gradient_gpu.cpp holds to PySCF) at the recipe's W and Gam.
TEST(DfJkDerivative, BuildingBlocksMatchTheGradients) {
  FBF_REQUIRE_GPU();
  const auto doc = fb::read_json(kDataDir / "df_jk_derivative.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const fb::json *c = find_case(*doc, "h2o_ccpvtz");
  ASSERT_NE(c, nullptr);
  fb::DeviceResources res{0};
  const fb::Molecule molecule = molecule_of(res, *c);
  const auto stamp = fb::resolve_basis((*c)["basis"].get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis((*c)["aux_basis"].get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  const auto aux = fb::place(aux_stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;
  const std::size_t nao = basis->nao_sphe();
  const std::size_t naux = aux->nao_sphe();
  const std::size_t n_atom = basis->natoms();

  // The atom of each spherical function.
  const auto atom_of = [](const fb::BasisSet &set) {
    std::vector<std::size_t> out;
    for (std::uint32_t a = 0; a < set.natoms(); ++a) {
      for (std::uint32_t i = 0; i < set.atom_shell_count(a); ++i) {
        out.insert(out.end(), set.shells()[set.atom_shell_offset(a) + i].function_count_sphe(), a);
      }
    }
    return out;
  };
  const std::vector<std::size_t> ao_atom = atom_of(*basis);
  const std::vector<std::size_t> aux_atom = atom_of(*aux);

  std::vector<double> w(naux * naux);
  for (std::size_t q = 0; q < naux; ++q) {
    for (std::size_t p = 0; p < naux; ++p) {
      w[p + (naux * q)] = 1.0 / static_cast<double>(p + q + 1);
    }
  }
  std::vector<double> gamma(nao * nao * naux);
  for (std::size_t m = 0; m < nao; ++m) {
    for (std::size_t n = 0; n < nao; ++n) {
      for (std::size_t p = 0; p < naux; ++p) {
        gamma[p + (naux * (n + (nao * m)))] = 1.0 / static_cast<double>(m + n + p + 1);
      }
    }
  }
  const fb::Buffers::device<double> d_w = upload(res, w);
  const fb::Buffers::device<double> d_gamma = upload(res, gamma);

  // d(P|Q)/dR_{X,k} = [P on X] M1[k][P][Q] + [Q on X] M1[k][Q][P].
  fb::Buffers::device<double> m1(3 * naux * naux, res.device());
  const fb::Status ms = fb::metric_derivative(res, *aux, m1.data());
  ASSERT_TRUE(ms) << ms.error().message;
  const std::vector<double> m1_host = download(res, m1.data(), 3 * naux * naux);
  std::vector<double> metric_ours(3 * n_atom, 0.0);
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t p = 0; p < naux; ++p) {
      for (std::size_t q = 0; q < naux; ++q) {
        const double v = m1_host[q + (naux * (p + (naux * k)))];
        // Element (P, Q) of slab k is d(P|Q)/dA_k on P's centre; on Q's it is
        // that of (Q, P), so each term lands on P's atom with W_PQ + W_QP.
        metric_ours[(3 * aux_atom[p]) + k] += v * (w[p + (naux * q)] + w[q + (naux * p)]);
      }
    }
  }
  const auto metric_ref = fb::metric_gradient(res, *aux, d_w.data());
  ASSERT_TRUE(metric_ref) << metric_ref.error().message;
  compare("metric_derivative . W vs metric_gradient", metric_ours, *metric_ref, kSelfRelTol);

  // d(mn|P)/dR_{X,k} = -[m on X] X_mnP - [n on X] X_nmP + [P on X] (X_mnP +
  // X_nmP), walked one aux atom at a time.
  const auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  std::vector<double> three_ours(3 * n_atom, 0.0);
  std::size_t p0 = 0;
  for (std::uint32_t blk = 0; blk < aux->natoms(); ++blk) {
    std::size_t nb = 0;
    for (std::uint32_t i = 0; i < aux->atom_shell_count(blk); ++i) {
      nb += aux->shells()[aux->atom_shell_offset(blk) + i].function_count_sphe();
    }
    fb::Buffers::device<double> x(3 * nao * nao * nb, res.device());
    const fb::Status xs = fb::three_center_derivative(res, *basis, *classes, *aux, blk, x.data());
    ASSERT_TRUE(xs) << xs.error().message;
    const std::vector<double> x_host = download(res, x.data(), 3 * nao * nao * nb);
    for (std::size_t k = 0; k < 3; ++k) {
      for (std::size_t m = 0; m < nao; ++m) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t pb = 0; pb < nb; ++pb) {
            const double v = x_host[pb + (nb * (n + (nao * (m + (nao * k)))))];
            const std::size_t p = p0 + pb;
            const double pair =
                gamma[p + (naux * (n + (nao * m)))] + gamma[p + (naux * (m + (nao * n)))];
            three_ours[(3 * ao_atom[m]) + k] -= v * pair;
            three_ours[(3 * blk) + k] += v * pair;
          }
        }
      }
    }
    p0 += nb;
  }
  ASSERT_EQ(p0, naux);
  const auto three_ref = fb::three_center_gradient(res, *basis, *classes, *aux, d_gamma.data());
  ASSERT_TRUE(three_ref) << three_ref.error().message;
  compare("three_center_derivative . Gam vs three_center_gradient", three_ours, *three_ref,
          kSelfRelTol);
}

// Bases not made by place(), an occupation past the orbitals, an atom past
// the bases, a shell past the ladders, an index past the kernels' 32 bits and
// scratch past the device's memory: each refused before anything is launched.
TEST(DfJkDerivative, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto placed = [](std::vector<fb::Shell> shells) {
    const auto count = static_cast<std::uint32_t>(shells.size());
    return fb::BasisSet{std::move(shells), {0, count}};
  };
  const fb::Shell s{0, {{1.0, 1.0}}, {0.0, 0.0, 0.0}};
  const fb::BasisSet unplaced(std::vector<fb::Shell>{s});
  const fb::BasisSet one_s = placed({s});
  const fb::BasisSet k_aux = placed({fb::Shell{7, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}});
  // Many d shells: 3 n_cart^2 doubles of Cartesian metric derivative alone
  // pass the device's memory while staying inside the 32-bit index, and a
  // few more pass the index.
  const auto d_shells = [&](const std::size_t count) {
    return placed(std::vector<fb::Shell>(count, fb::Shell{2, {{1.0, 1.0}}, {0.0, 0.0, 0.0}}));
  };
  const fb::BasisSet huge_aux = d_shells(4000);     // 24000 Cartesian
  const fb::BasisSet too_wide_aux = d_shells(4600); // 27600 Cartesian
  fb::Buffers::device<double> scratch(64, res.device());
  double *const p = scratch.data();

  const auto classes = fb::primitive_pair_classes(res, one_s);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto refusal = [](const fb::Status &st, const fb::ErrorKind kind, const std::string &says) {
    ASSERT_FALSE(st) << says;
    EXPECT_EQ(st.error().kind, kind) << st.error().message;
    EXPECT_NE(st.error().message.find(says), std::string::npos) << st.error().message;
  };
  const auto jk = [&](const fb::BasisSet &aux, const std::int32_t nocc) {
    return fb::df_jk_derivative(res, one_s, *classes, aux, p, p, p, nocc, p, p, nullptr);
  };
  refusal(jk(unplaced, 1), fb::ErrorKind::InvalidConfig,
          "df_jk_derivative: the bases have 1 and 0 atoms");
  refusal(jk(k_aux, 1), fb::ErrorKind::Unsupported, "df_jk_derivative: aux shell 0 has momentum 7");
  refusal(jk(one_s, 2), fb::ErrorKind::InvalidConfig, "df_jk_derivative: nocc 2 is outside [0, 1]");
  refusal(jk(one_s, -1), fb::ErrorKind::InvalidConfig, "nocc -1 is outside");
  refusal(jk(too_wide_aux, 1), fb::ErrorKind::Unsupported, "32-bit indices");
  refusal(jk(huge_aux, 1), fb::ErrorKind::Unsupported, "bytes of device scratch");

  refusal(fb::metric_derivative(res, unplaced, p), fb::ErrorKind::InvalidConfig,
          "metric_derivative: the aux basis has no atoms");
  refusal(fb::metric_derivative(res, k_aux, p), fb::ErrorKind::InvalidConfig,
          "no kernel for an aux shell of momentum 7");
  refusal(fb::metric_derivative(res, huge_aux, p), fb::ErrorKind::Unsupported,
          "bytes of device scratch");
  refusal(fb::three_center_derivative(res, one_s, *classes, one_s, 1, p),
          fb::ErrorKind::InvalidConfig, "three_center_derivative: atom 1 is not one of");
  refusal(fb::three_center_derivative(res, one_s, *classes, unplaced, 0, p),
          fb::ErrorKind::InvalidConfig, "the bases have 1 and 0 atoms");
}

} // namespace
