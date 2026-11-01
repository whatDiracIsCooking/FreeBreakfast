// GPU cases for mcscf::ThirdDerivative::apply_integrals over
// mcscf::df_integral_derivatives: I^x z, the state-averaged Hessian at the
// derivative integrals (the connection included) applied to z, against
// test/gtest/data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py):
// the stretched water's SA-CAS(4,4) / 6-31G over cc-pVDZ-JKFIT at two roots
// (0.5, 0.5) and (0.7, 0.3), three equal and (0.5, 0.3, 0.2), every root's
// z. The orbitals, roots and z are the fixture's (PySCF's own solutions for
// two roots are saddle points; the fixture's are the minima), the integrals
// this tree's. Checked: the covectors against `lxl_z` part by part (the CI
// part's singlet component, as the fixture's covectors are over the singlet
// space), L_xlambda = root I's CP right-hand side (mcscf::df_derivatives at
// one-hot weights) + I^x z against `lxl_lambda`, the translational sum rule,
// and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.test.ci_oracles;

namespace {

namespace fb = free_breakfast;

// Against the fixture, absolute: a hundredth of the issue's 1e-8. lxl_z and
// lxl_lambda are the same formulas over the same fit (max |lxl_z| 6e-3 to
// 0.14, |lxl_lambda| 0.26 to 0.8); measured <= 3.2e-13 on the 3080.
constexpr double kFixtureTol = 1e-10;
// Sum over the atoms of each axis of every element, absolute: translational
// invariance, which none of the terms is built on.
constexpr double kSumRuleTol = 1e-9;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_jk_derivative_gpu.cpp).
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

// Per AO of this tree's basis: its row in PySCF's AO order (each atom's shells
// sorted by l, stably), and N_i, function i being N_i times PySCF's
// unit-normalised one. 6-31G's split SP shells are where the orders differ.
std::pair<std::vector<std::size_t>, std::vector<double>> pyscf_rows(const fb::BasisSet &basis) {
  std::vector<std::size_t> row;
  std::vector<double> norm;
  const auto shells = basis.shells();
  std::size_t atom_first = 0;
  for (std::uint32_t atom = 0; atom < basis.natoms(); ++atom) {
    const std::uint32_t first = basis.atom_shell_offset(atom);
    const std::uint32_t count = basis.atom_shell_count(atom);
    std::vector<std::uint32_t> order(count);
    std::iota(order.begin(), order.end(), first);
    std::ranges::stable_sort(order, {},
                             [&](std::uint32_t s) { return shells[s].angular_momentum(); });
    std::vector<std::size_t> start(shells.size());
    std::size_t at = atom_first;
    for (const std::uint32_t s : order) {
      start[s] = at;
      at += shells[s].function_count_sphe();
    }
    for (std::uint32_t s = first; s < first + count; ++s) {
      const std::size_t nf = shells[s].function_count_sphe();
      for (std::size_t f = 0; f < nf; ++f) {
        row.push_back(start[s] + f);
        norm.push_back(std::sqrt(self_overlap(shells[s])));
      }
    }
    atom_first = at;
  }
  return {row, norm};
}

std::vector<double> download(fb::DeviceResources &res, const double *device, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *device) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(device, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
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

const fb::json &manifest() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(kDataDir / "casscf_hessian.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : manifest().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_hessian.json";
  static const fb::json kNone;
  return kNone;
}

// The fixture's (orb, ci, theta) layout of one or more vectors, column by
// column, into the operator's: the packed pairs (q core then q active, p
// ascending), each root's d, then Theta_sr for r < s.
std::vector<double> pack_vectors(const std::vector<double> &orb, const std::vector<double> &ci,
                                 const std::vector<double> &theta, std::size_t nmo,
                                 std::size_t ncore, std::size_t ncas, std::size_t dim,
                                 std::size_t nroots, bool rotations, std::size_t columns) {
  const std::size_t nocc = ncore + ncas;
  std::vector<double> out;
  for (std::size_t col = 0; col < columns; ++col) {
    const double *k = orb.data() + (col * nmo * nmo);
    for (std::size_t q = 0; q < nocc; ++q) {
      for (std::size_t p = std::max(q < ncore ? ncore : nocc, q + 1); p < nmo; ++p) {
        out.push_back(k[p + (nmo * q)]);
      }
    }
    const double *c = ci.data() + (col * dim * nroots);
    out.insert(out.end(), c, c + (dim * nroots));
    if (rotations) {
      const double *t = theta.data() + (col * nroots * nroots);
      for (std::size_t r = 0; r < nroots; ++r) {
        for (std::size_t s = r + 1; s < nroots; ++s) {
          out.push_back(t[s + (nroots * r)]);
        }
      }
    }
  }
  return out;
}

// The singlet part of every length-@p dim column of @p v, in place:
// prod_{S = 1..S_max} (S^2 - S(S + 1)) / (0 - S(S + 1)), over the host S^2.
void project_singlet(const fb::ci::StringSet &alpha, const fb::ci::StringSet &beta,
                     std::span<double> v, const std::size_t dim) {
  const std::int32_t s_max = (alpha.n_electrons() + beta.n_electrons()) / 2;
  std::vector<double> s2c(dim);
  for (std::size_t at = 0; at < v.size(); at += dim) {
    const std::span<double> col(v.data() + at, dim);
    for (std::int32_t s = 1; s <= s_max; ++s) {
      const double eig = s * (s + 1.0);
      ASSERT_TRUE(fb::test::host_spin_square(alpha, beta, col, s2c));
      for (std::size_t i = 0; i < dim; ++i) {
        col[i] = (s2c[i] - (eig * col[i])) / -eig;
      }
    }
  }
}

class McscfIntegralDerivative : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfIntegralDerivative, MatchesTheFixture) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam());
  ASSERT_FALSE(entry.is_null());
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const std::size_t nroots = weights.size();
  const bool rotations = !entry.at("equal_weights").get<bool>();
  const auto nmo = entry.at("nmo").get<std::size_t>();
  const auto naux = entry.at("naux").get<std::size_t>();
  const auto ncore = entry.at("ncore").get<std::int32_t>();
  const auto ncas = entry.at("ncas").get<std::int32_t>();
  const auto nelecas = entry.at("nelecas").get<std::vector<std::int32_t>>();
  const auto na = entry.at("na").get<std::size_t>();
  const auto nb = entry.at("nb").get<std::size_t>();
  const std::size_t dim = na * nb;
  const auto arrays = entry.at("arrays");

  fb::DeviceResources res(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  std::vector<fb::Atom> atoms;
  for (const fb::json &atom : entry.at("atoms")) {
    const auto xyz = atom.at("coord").get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom.at("charge").get<std::int32_t>())});
  }
  const fb::Molecule molecule(res, std::move(atoms));
  const auto stamp = fb::resolve_basis(entry.at("basis").get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  auto basis = fb::place(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis) << basis.error().message;
  const auto aux_stamp = fb::resolve_aux_basis(entry.at("aux_basis").get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
  auto aux = fb::place(aux_stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(aux) << aux.error().message;
  ASSERT_EQ(basis->nao_sphe(), nmo);
  ASSERT_EQ(aux->nao_sphe(), naux);

  // h, and B and L over the fitting set.
  auto classes = fb::primitive_pair_classes(res, *basis);
  ASSERT_TRUE(classes) << classes.error().message;
  auto basis_u = fb::upload_spherical_transform_matrix(res, *basis);
  ASSERT_TRUE(basis_u) << basis_u.error().message;
  fb::MatrixAOPair h_pair(res, *basis, *basis_u);
  ASSERT_TRUE(fb::core_hamiltonian(res, *classes, molecule, h_pair.cartesian()));
  ASSERT_TRUE(h_pair.cartesian_to_spherical(res));
  auto aux_u = fb::upload_spherical_transform_matrix(res, *aux);
  ASSERT_TRUE(aux_u) << aux_u.error().message;
  fb::MatrixAOPair metric(res, *aux, std::move(*aux_u));
  fb::Buffers::device<double> b(naux * nmo * nmo, res.device());
  ASSERT_TRUE(fb::metric(res, *aux, metric));
  ASSERT_TRUE(fb::three_center(res, *basis, *classes, *aux, b.data()));
  ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nmo), static_cast<std::int32_t>(naux),
                              metric.spherical().data(), b.data()));
  const double *const l = metric.spherical().data();

  // The fixture's orbitals, rows into this basis's order and normalisation,
  // and its roots.
  const auto [row, norm] = pyscf_rows(*basis);
  const std::vector<double> c_pyscf = read_array(arrays.at("c").get<std::string>(), {nmo, nmo});
  std::vector<double> c_host(nmo * nmo);
  for (std::size_t j = 0; j < nmo; ++j) {
    for (std::size_t m = 0; m < nmo; ++m) {
      c_host[m + (nmo * j)] = c_pyscf[row[m] + (nmo * j)] / norm[m];
    }
  }
  fb::SphericalMatrixAO c(res, *basis);
  upload(res, c_host, c.data());
  const std::vector<double> roots_host =
      read_array(arrays.at("ci").get<std::string>(), {dim, nroots});
  fb::Buffers::device<double> roots(roots_host.size(), res.device());
  upload(res, roots_host, roots.data());
  auto strings = fb::ci::DeviceStringSet::create(res, ncas, nelecas[0]);
  ASSERT_TRUE(strings) << strings.error().message;
  ASSERT_EQ(static_cast<std::size_t>(strings->size()), na);
  const auto alpha = fb::ci::StringSet::create(ncas, nelecas[0]);
  const auto beta = fb::ci::StringSet::create(ncas, nelecas[1]);
  ASSERT_TRUE(alpha && beta);

  // I^x, every coordinate.
  const std::size_t n_atom = molecule.natoms();
  const std::size_t n = 3 * n_atom;
  const auto begin = std::chrono::steady_clock::now();
  fb::Buffers::device<double> h1(n * nmo * nmo, res.device());
  fb::Buffers::device<double> w1(n * nmo * nmo * naux, res.device());
  ASSERT_TRUE(fb::mcscf::df_integral_derivatives(
      res, molecule, *basis, *classes, *aux, h_pair.spherical(), b, l, c, h1.data(), w1.data()));
  const std::chrono::duration<double> integral_seconds = std::chrono::steady_clock::now() - begin;
  {
    const auto refused = [&](const fb::Status &made, std::string_view what) {
      ASSERT_FALSE(made.has_value()) << what;
      EXPECT_EQ(made.error().kind, fb::ErrorKind::InvalidConfig) << what;
      EXPECT_TRUE(made.error().message.starts_with("mcscf::")) << what;
    };
    refused(fb::mcscf::df_integral_derivatives(res, molecule, *basis, *classes, *aux,
                                               h_pair.spherical(), b, l, c, nullptr, w1.data()),
            "no h1");
    refused(fb::mcscf::df_integral_derivatives(res, molecule, *basis, *classes, *aux,
                                               h_pair.spherical(), b, nullptr, c, h1.data(),
                                               w1.data()),
            "no metric factor");
  }

  // lambda^x, as the operator lays a vector out.
  const std::vector<double> lam_orb =
      read_array(arrays.at("lambda_orb").get<std::string>(), {nmo, nmo, n});
  const std::vector<double> lam_ci =
      read_array(arrays.at("lambda_ci").get<std::string>(), {dim, nroots, n});
  const std::vector<double> lam_theta =
      rotations ? read_array(arrays.at("lambda_theta").get<std::string>(), {nroots, nroots, n})
                : std::vector<double>{};
  const auto core = static_cast<std::size_t>(ncore);
  const auto act = static_cast<std::size_t>(ncas);
  const std::vector<double> lambda =
      pack_vectors(lam_orb, lam_ci, lam_theta, nmo, core, act, dim, nroots, rotations, n);
  const std::size_t size = lambda.size() / n;
  const std::size_t pairs = size - (dim * nroots) - (rotations ? nroots * (nroots - 1) / 2 : 0);

  for (std::size_t root = 0; root < nroots; ++root) {
    SCOPED_TRACE(std::format("root {}", root));
    const fb::json &ra = entry.at("roots")[root].at("arrays");
    const std::vector<double> z_host =
        pack_vectors(read_array(ra.at("z_orb").get<std::string>(), {nmo, nmo}),
                     read_array(ra.at("z_ci").get<std::string>(), {dim, nroots}),
                     rotations ? read_array(ra.at("z_theta").get<std::string>(), {nroots, nroots})
                               : std::vector<double>{},
                     nmo, core, act, dim, nroots, rotations, 1);
    ASSERT_EQ(z_host.size(), size);
    fb::Buffers::device<double> z(size, res.device());
    upload(res, z_host, z.data());

    const auto start = std::chrono::steady_clock::now();
    auto op = fb::mcscf::ThirdDerivative::create(
        res, h_pair.spherical(), b, static_cast<std::int32_t>(naux), c, ncore, ncas, *strings,
        *strings, roots.data(), weights, z.data());
    ASSERT_TRUE(op.has_value()) << op.error().message;
    ASSERT_EQ(static_cast<std::size_t>(op->size()), size);
    fb::Buffers::device<double> y(size * n, res.device());
    ASSERT_TRUE(
        op->apply_integrals(res, static_cast<std::int32_t>(n), h1.data(), w1.data(), y.data()));
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    std::vector<double> y_host = download(res, y.data(), size * n);
    {
      const fb::Status made = op->apply_integrals(res, -1, h1.data(), w1.data(), y.data());
      ASSERT_FALSE(made.has_value());
      EXPECT_EQ(made.error().kind, fb::ErrorKind::InvalidConfig);
    }

    // Translational invariance: the integrals' derivatives sum to zero over
    // the atoms, so the covectors do.
    double sum_rule = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
      for (std::size_t i = 0; i < size; ++i) {
        double sum = 0.0;
        for (std::size_t a = 0; a < n_atom; ++a) {
          sum += y_host[i + (size * ((3 * a) + k))];
        }
        sum_rule = std::max(sum_rule, std::abs(sum));
      }
    }
    EXPECT_LT(sum_rule, kSumRuleTol);

    // L_xlambda = (root I's right-hand side + I^x z) . lambda^y, before the
    // CI parts are projected (lambda is a singlet).
    auto rhs =
        fb::mcscf::df_derivatives(res, molecule, *basis, *classes, *aux, h_pair.spherical(), b, l,
                                  c, ncore, ncas, nelecas[0], nelecas[1], roots.data(), [&] {
                                    std::vector<double> one_hot(nroots, 0.0);
                                    one_hot[root] = 1.0;
                                    return one_hot;
                                  }());
    ASSERT_TRUE(rhs) << rhs.error().message;
    const std::vector<double> rhs_packed =
        pack_vectors(download(res, rhs->rhs_orbital.data(), nmo * nmo * n),
                     download(res, rhs->rhs_ci.data(), dim * nroots * n), rhs->rhs_theta, nmo, core,
                     act, dim, nroots, rotations, n);
    std::vector<double> lxl(n * n, 0.0);
    for (std::size_t yy = 0; yy < n; ++yy) {
      for (std::size_t x = 0; x < n; ++x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
          sum += (rhs_packed[i + (size * x)] + y_host[i + (size * x)]) * lambda[i + (size * yy)];
        }
        lxl[x + (n * yy)] = sum;
      }
    }

    // I^x z part by part against lxl_z, the CI parts as their singlet parts.
    const std::vector<double> want = pack_vectors(
        read_array(ra.at("lxl_z_orb").get<std::string>(), {nmo, nmo, n}),
        read_array(ra.at("lxl_z_ci").get<std::string>(), {dim, nroots, n}),
        rotations ? read_array(ra.at("lxl_z_theta").get<std::string>(), {nroots, nroots, n})
                  : std::vector<double>{},
        nmo, core, act, dim, nroots, rotations, n);
    ASSERT_EQ(want.size(), y_host.size());
    for (std::size_t x = 0; x < n; ++x) {
      project_singlet(*alpha, *beta, std::span(y_host).subspan((size * x) + pairs, dim * nroots),
                      dim);
    }
    std::array<double, 3> worst{};
    std::array<double, 3> scale{};
    for (std::size_t x = 0; x < n; ++x) {
      for (std::size_t i = 0; i < size; ++i) {
        const std::size_t part = i < pairs ? 0 : (i < pairs + (dim * nroots) ? 1 : 2);
        const double got = y_host[i + (size * x)];
        const double expect = want[i + (size * x)];
        worst[part] = std::max(worst[part], std::abs(got - expect));
        scale[part] = std::max(scale[part], std::abs(expect));
      }
    }
    EXPECT_LT(worst[0], kFixtureTol) << "orbital part, of max " << scale[0];
    EXPECT_LT(worst[1], kFixtureTol) << "CI part, of max " << scale[1];
    EXPECT_LT(worst[2], kFixtureTol) << "Theta part, of max " << scale[2];

    const std::vector<double> lxl_want = read_array(ra.at("lxl_lambda").get<std::string>(), {n, n});
    double lxl_worst = 0.0;
    double lxl_scale = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      lxl_worst = std::max(lxl_worst, std::abs(lxl[i] - lxl_want[i]));
      lxl_scale = std::max(lxl_scale, std::abs(lxl_want[i]));
    }
    EXPECT_LT(lxl_worst, kFixtureTol) << "L_xlambda, of max " << lxl_scale;

    std::cout << std::format("[ I^x z    ] {} root {}: integrals {:.2f} s, operator and {} "
                             "columns {:.2f} s; |I^x z - fixture| orb {:.1e} ci {:.1e} theta "
                             "{:.1e} (of {:.2e} {:.2e} {:.2e}); |L_xlambda - fixture| {:.1e} "
                             "(of {:.2e}); sum rule {:.1e}\n",
                             entry.at("name").get<std::string>(), root, integral_seconds.count(), n,
                             seconds.count(), worst[0], worst[1], worst[2], scale[0], scale[1],
                             scale[2], lxl_worst, lxl_scale, sum_rule);
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, McscfIntegralDerivative,
                         ::testing::Values("h2o_631g_sa2", "h2o_631g_sa2_unequal", "h2o_631g_sa3",
                                           "h2o_631g_sa3_unequal"),
                         [](const auto &info) { return std::string{info.param}; });

} // namespace
