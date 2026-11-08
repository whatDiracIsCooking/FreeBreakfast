// GPU cases for mcscf::ThirdDerivative, the third derivative of the
// state-averaged DF-CASSCF energy contracted with z, against
// test/gtest/data/casscf_hessian/ (tools/emit/casscf_hessian_reference.py):
// the stretched water's SA-CAS(4,4) / 6-31G over cc-pVDZ-JKFIT at two roots
// (0.5, 0.5) and (0.7, 0.3), three equal and (0.5, 0.3, 0.2), every root's z.
// The orbitals and roots are the fixture's (PySCF's own solutions for two
// roots are saddle points; the fixture's are the minima), the integrals this
// tree's. Checked: lambda^xT (z . d2 g_SA) lambda^y against the fixture's
// `third` (a mixed fourth-order difference of the exact gradient), the
// two random vectors' `third_random`, its symmetry in (x, y), the refusals,
// and partly equal weights (0.4, 0.4, 0.2) taken with every rotation. Beside them, mcscf::df_integral_derivatives' chunks handed to a
// test consumer under a budget of two coordinates per chunk, against a
// Richardson ladder of (h_MO, L) along every coordinate, the connection
// included.

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

namespace {

namespace fb = free_breakfast;

// The issue's bar against the fixture's differences, absolute (max |third| is
// 5e-3 to 5e-2; the differences themselves are good to ~5e-10).
constexpr double kThirdTol = 1e-7;
// The random vectors' value, absolute (|value| 0.15 to 3.4): their norms are
// ~10x lambda's and the fixture's difference is good to ~1e-8 there (the
// numpy prototype of these formulas lands on the same values).
constexpr double kRandomTol = 1e-7;
// max |T - T^T| over max |T|: rounding only.
constexpr double kSymmetryRelTol = 1e-12;
// The chunks' (h1^x, W^x) against the ladder, relative to the largest element
// of the ladder's h_MO or L derivative along that coordinate: the issue's
// bar.
constexpr double kLadderTol = 1e-7;
// The ladder's displacement, bohr, and half of it beside it (Richardson).
constexpr double kLadderStep = 2e-3;

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

// C (m x n) = op(A) B, column-major; op(A) is A (m x k), or A^T for A k x m.
std::vector<double> host_gemm(std::size_t m, std::size_t k, std::size_t n,
                              const std::vector<double> &a, bool transpose_a,
                              const std::vector<double> &b) {
  std::vector<double> c(m * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t l = 0; l < k; ++l) {
      const double blj = b[l + (k * j)];
      for (std::size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += (transpose_a ? a[l + (k * i)] : a[i + (m * l)]) * blj;
      }
    }
  }
  return c;
}

// A^-1/2 of a symmetric A near the identity (||1 - A|| < 1), by the coupled
// Newton-Schulz iteration: Y -> A^1/2, Z -> A^-1/2.
std::vector<double> inverse_sqrt(const std::vector<double> &a, std::size_t n) {
  std::vector<double> y = a;
  std::vector<double> z(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z[i * (n + 1)] = 1.0;
  }
  for (std::int32_t it = 0; it < 20; ++it) {
    // T = (3 - Z Y) / 2; done when Z Y is the identity to rounding.
    std::vector<double> t = host_gemm(n, n, n, z, false, y);
    double off = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      off = std::max(off, std::abs(t[i] - (i % (n + 1) == 0 ? 1.0 : 0.0)));
      t[i] = -0.5 * t[i];
    }
    if (off < 1e-14) {
      break;
    }
    for (std::size_t i = 0; i < n; ++i) {
      t[i * (n + 1)] += 1.5;
    }
    y = host_gemm(n, n, n, y, false, t);
    z = host_gemm(n, n, n, t, false, z);
  }
  return z;
}

// X = L^-1 B (or L^-T B) for the lower triangle of the n x n @p l and the
// n x cols @p b, by substitution.
std::vector<double> lower_solve(const std::vector<double> &l, std::size_t n, std::vector<double> b,
                                std::size_t cols, bool transpose) {
  for (std::size_t j = 0; j < cols; ++j) {
    double *x = b.data() + (n * j);
    if (!transpose) {
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) {
          x[i] -= l[i + (n * k)] * x[k];
        }
        x[i] /= l[i * (n + 1)];
      }
    } else {
      for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = i + 1; k < n; ++k) {
          x[i] -= l[k + (n * i)] * x[k];
        }
        x[i] /= l[i * (n + 1)];
      }
    }
  }
  return b;
}

// A^T over a square n.
std::vector<double> transpose(const std::vector<double> &a, std::size_t n) {
  std::vector<double> t(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      t[j + (n * i)] = a[i + (n * j)];
    }
  }
  return t;
}

class McscfThirdDerivative : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfThirdDerivative, MatchesTheFixture) {
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

  // h, and B over the fitting set.
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
  auto strings = fb::ci::DeviceStringSet::create(
      res, ncas, static_cast<std::int32_t>(entry.at("nelecas")[0].get<std::int32_t>()));
  ASSERT_TRUE(strings) << strings.error().message;
  ASSERT_EQ(static_cast<std::size_t>(strings->size()), na);

  // lambda^x, as the operator lays a vector out.
  const std::size_t n = 3 * molecule.natoms();
  const std::vector<double> lam_orb =
      read_array(arrays.at("lambda_orb").get<std::string>(), {nmo, nmo, n});
  const std::vector<double> lam_ci =
      read_array(arrays.at("lambda_ci").get<std::string>(), {dim, nroots, n});
  const std::vector<double> lam_theta =
      rotations ? read_array(arrays.at("lambda_theta").get<std::string>(), {nroots, nroots, n})
                : std::vector<double>{};
  const auto mo = static_cast<std::size_t>(nmo);
  const auto core = static_cast<std::size_t>(ncore);
  const auto act = static_cast<std::size_t>(ncas);
  const std::vector<double> lambda =
      pack_vectors(lam_orb, lam_ci, lam_theta, mo, core, act, dim, nroots, rotations, n);
  const std::size_t size = lambda.size() / n;

  // The refusals, on the first root's z.
  {
    const fb::json &root0 = entry.at("roots")[0];
    const std::vector<double> z_host =
        pack_vectors(read_array(root0.at("arrays").at("z_orb").get<std::string>(), {nmo, nmo}),
                     read_array(root0.at("arrays").at("z_ci").get<std::string>(), {dim, nroots}),
                     rotations ? read_array(root0.at("arrays").at("z_theta").get<std::string>(),
                                            {nroots, nroots})
                               : std::vector<double>{},
                     mo, core, act, dim, nroots, rotations, 1);
    fb::Buffers::device<double> z(z_host.size(), res.device());
    upload(res, z_host, z.data());
    const auto refused = [&](auto &&made, std::string_view what) {
      ASSERT_FALSE(made.has_value()) << what;
      EXPECT_EQ(made.error().kind, fb::ErrorKind::InvalidConfig) << what;
      EXPECT_TRUE(made.error().message.starts_with("mcscf::ThirdDerivative")) << what;
    };
    refused(fb::mcscf::ThirdDerivative::create(res, h_pair.spherical(), b,
                                               static_cast<std::int32_t>(naux), c, ncore, 0,
                                               *strings, *strings, roots.data(), weights, z.data()),
            "ncas 0");
    refused(fb::mcscf::ThirdDerivative::create(res, h_pair.spherical(), b,
                                               static_cast<std::int32_t>(naux) - 1, c, ncore, ncas,
                                               *strings, *strings, roots.data(), weights, z.data()),
            "b of the wrong size");
    std::vector<double> unnormalized = weights;
    unnormalized[0] += 0.1;
    refused(fb::mcscf::ThirdDerivative::create(
                res, h_pair.spherical(), b, static_cast<std::int32_t>(naux), c, ncore, ncas,
                *strings, *strings, roots.data(), unnormalized, z.data()),
            "weights not summing to 1");
    if (nroots == 3) {
      // Partly equal weights are taken, every pair's rotation a parameter.
      const std::size_t n_theta = rotations ? 3 : 0;
      const std::size_t pairs = z_host.size() - (dim * nroots) - n_theta;
      fb::Buffers::device<double> zero_z(pairs + (dim * nroots) + 3, res.device());
      ASSERT_EQ(wwr::wwrMemsetAsync(zero_z.data(), 0, zero_z.num_elements() * sizeof(double),
                                    res.stream()),
                wwr::wwrSuccess);
      const std::vector<double> partly{0.4, 0.4, 0.2};
      const auto made = fb::mcscf::ThirdDerivative::create(
          res, h_pair.spherical(), b, static_cast<std::int32_t>(naux), c, ncore, ncas, *strings,
          *strings, roots.data(), partly, zero_z.data());
      ASSERT_TRUE(made.has_value()) << made.error().message;
      EXPECT_EQ(static_cast<std::size_t>(made->size()), zero_z.num_elements());
    }
  }

  fb::Buffers::device<double> lambda_device(lambda.size(), res.device());
  upload(res, lambda, lambda_device.data());
  for (std::size_t root = 0; root < nroots; ++root) {
    SCOPED_TRACE(std::format("root {}", root));
    const fb::json &info = entry.at("roots")[root];
    const fb::json &ra = info.at("arrays");
    const std::vector<double> z_host =
        pack_vectors(read_array(ra.at("z_orb").get<std::string>(), {nmo, nmo}),
                     read_array(ra.at("z_ci").get<std::string>(), {dim, nroots}),
                     rotations ? read_array(ra.at("z_theta").get<std::string>(), {nroots, nroots})
                               : std::vector<double>{},
                     mo, core, act, dim, nroots, rotations, 1);
    ASSERT_EQ(z_host.size(), size);
    fb::Buffers::device<double> z(size, res.device());
    upload(res, z_host, z.data());

    const auto begin = std::chrono::steady_clock::now();
    auto op = fb::mcscf::ThirdDerivative::create(
        res, h_pair.spherical(), b, static_cast<std::int32_t>(naux), c, ncore, ncas, *strings,
        *strings, roots.data(), weights, z.data());
    ASSERT_TRUE(op.has_value()) << op.error().message;
    ASSERT_EQ(static_cast<std::size_t>(op->size()), size);
    fb::Buffers::device<double> y(size * n, res.device());
    ASSERT_TRUE(op->apply(res, static_cast<std::int32_t>(n), lambda_device.data(), y.data()));
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
    const std::vector<double> y_host = download(res, y.data(), size * n);

    // third = Lambda^T Y.
    std::vector<double> third(n * n, 0.0);
    for (std::size_t yy = 0; yy < n; ++yy) {
      for (std::size_t x = 0; x < n; ++x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
          sum += lambda[i + (size * x)] * y_host[i + (size * yy)];
        }
        third[x + (n * yy)] = sum;
      }
    }
    const std::vector<double> want = read_array(ra.at("third").get<std::string>(), {n, n});
    double worst = 0.0;
    double scale = 0.0;
    double asym = 0.0;
    for (std::size_t yy = 0; yy < n; ++yy) {
      for (std::size_t x = 0; x < n; ++x) {
        worst = std::max(worst, std::abs(third[x + (n * yy)] - want[x + (n * yy)]));
        scale = std::max(scale, std::abs(want[x + (n * yy)]));
        asym = std::max(asym, std::abs(third[x + (n * yy)] - third[yy + (n * x)]));
      }
    }
    EXPECT_LE(worst, kThirdTol) << "max |third - fixture| of max " << scale;
    EXPECT_LE(asym, kSymmetryRelTol * scale) << "max |T - T^T|";

    // Two random vectors: u_0^T T(., u_1, z).
    const std::vector<double> u = pack_vectors(
        read_array(ra.at("random_orb").get<std::string>(), {nmo, nmo, 2}),
        read_array(ra.at("random_ci").get<std::string>(), {dim, nroots, 2}),
        rotations ? read_array(ra.at("random_theta").get<std::string>(), {nroots, nroots, 2})
                  : std::vector<double>{},
        mo, core, act, dim, nroots, rotations, 2);
    fb::Buffers::device<double> u_device(u.size(), res.device());
    upload(res, u, u_device.data());
    fb::Buffers::device<double> uy(size, res.device());
    ASSERT_TRUE(op->apply(res, 1, u_device.data() + size, uy.data()));
    const std::vector<double> uy_host = download(res, uy.data(), size);
    double random = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
      random += u[i] * uy_host[i];
    }
    const double random_want = info.at("third_random").get<double>();
    EXPECT_LE(std::abs(random - random_want), kRandomTol)
        << "u0^T T u1 " << random << " against " << random_want;

    std::cout << std::format("[ third    ] {} root {}: {} coordinates in {:.2f} s; max |third| "
                             "{:.3e}, |third - fixture| {:.2e}, asym {:.1e}; random {:.10f} vs "
                             "{:.10f} ({:.1e} relative)\n",
                             entry.at("name").get<std::string>(), root, n, seconds.count(), scale,
                             worst, asym, random, random_want,
                             std::abs(random - random_want) / std::abs(random_want));
  }
}

// mcscf::df_integral_derivatives' chunks, handed to a test consumer under a
// budget of exactly two coordinates per chunk, against a ladder of what they
// are the derivatives of, along every coordinate x of the stretched water:
// h_MO(R) = C(R)^T h(R) C(R) and L(R) = C(R)^T T(R) C(R) Z(R), with
// C(R) = C (C^T S(R) C)^-1/2 (the connection: d/dR is C tau^x) and
// Z(R) = L_M^-T (L_M^-1 M(R) L_M^-T)^-1/2 at the reference's factor L_M (the
// gauge W^x is in: Z(R) Z(R)^T = M(R)^-1, so L(R) L(R)^T is the fitted
// (pq|rs) at R). Central differences at kLadderStep and half of it,
// Richardson-extrapolated.
TEST(McscfThirdDerivativeChunks, MatchALadder) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_631g_sa2");
  ASSERT_FALSE(entry.is_null());
  const auto nmo = entry.at("nmo").get<std::size_t>();
  const auto naux = entry.at("naux").get<std::size_t>();
  const std::size_t nn = nmo * nmo;
  const std::size_t factor = nn * naux;

  fb::DeviceResources res(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  std::vector<fb::Atom> atoms;
  for (const fb::json &atom : entry.at("atoms")) {
    const auto xyz = atom.at("coord").get<std::vector<double>>();
    atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                             static_cast<fb::Element>(atom.at("charge").get<std::int32_t>())});
  }
  const auto stamp = fb::resolve_basis(entry.at("basis").get<std::string>());
  ASSERT_TRUE(stamp) << stamp.error().message;
  const auto aux_stamp = fb::resolve_aux_basis(entry.at("aux_basis").get<std::string>());
  ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;

  // Everything at one geometry: the placed bases, S, h, the metric M and the
  // three-center T (unfitted), and the molecule they were placed on.
  struct Geometry {
    std::optional<fb::Molecule> molecule;
    std::optional<fb::BasisSet> basis;
    std::optional<fb::BasisSet> aux;
    std::vector<double> s, h, m, t;
  };
  const auto at = [&](const std::vector<fb::Atom> &where) {
    Geometry g;
    g.molecule.emplace(res, where);
    auto basis = fb::place(stamp->primitive_normalized(), *g.molecule);
    EXPECT_TRUE(basis) << basis.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *g.molecule);
    EXPECT_TRUE(aux) << aux.error().message;
    if (!basis || !aux) {
      return g;
    }
    EXPECT_EQ(basis->nao_sphe(), nmo);
    EXPECT_EQ(aux->nao_sphe(), naux);
    auto classes = fb::primitive_pair_classes(res, *basis);
    EXPECT_TRUE(classes) << classes.error().message;
    auto basis_u = fb::upload_spherical_transform_matrix(res, *basis);
    EXPECT_TRUE(basis_u) << basis_u.error().message;
    auto aux_u = fb::upload_spherical_transform_matrix(res, *aux);
    EXPECT_TRUE(aux_u) << aux_u.error().message;
    if (!classes || !basis_u || !aux_u) {
      return g;
    }
    fb::MatrixAOPair s_pair(res, *basis, *basis_u);
    EXPECT_TRUE(fb::overlap(res, *classes, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(res));
    fb::MatrixAOPair h_pair(res, *basis, *basis_u);
    EXPECT_TRUE(fb::core_hamiltonian(res, *classes, *g.molecule, h_pair.cartesian()));
    EXPECT_TRUE(h_pair.cartesian_to_spherical(res));
    fb::MatrixAOPair metric(res, *aux, std::move(*aux_u));
    EXPECT_TRUE(fb::metric(res, *aux, metric));
    fb::Buffers::device<double> t(factor, res.device());
    EXPECT_TRUE(fb::three_center(res, *basis, *classes, *aux, t.data()));
    g.s = download(res, s_pair.spherical().data(), nn);
    g.h = download(res, h_pair.spherical().data(), nn);
    g.m = download(res, metric.spherical().data(), naux * naux);
    g.t = download(res, t.data(), factor);
    g.basis.emplace(std::move(*basis));
    g.aux.emplace(std::move(*aux));
    return g;
  };
  const Geometry ref = at(atoms);
  ASSERT_TRUE(ref.basis && ref.aux);
  const std::size_t n = 3 * atoms.size();

  // The fit at the reference (L_M = chol(M), B = L_M^-1 T) on the device, as
  // the call takes it.
  fb::Buffers::device<double> b(factor, res.device());
  fb::Buffers::device<double> l(naux * naux, res.device());
  upload(res, ref.t, b.data());
  upload(res, ref.m, l.data());
  ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nmo), static_cast<std::int32_t>(naux),
                              l.data(), b.data()));
  const std::vector<double> l_m = download(res, l.data(), naux * naux);

  // The fixture's orbitals, rows into this basis's order and normalisation,
  // made orthonormal over this S: C (C^T S C)^-1/2.
  const auto [row, norm] = pyscf_rows(*ref.basis);
  const std::vector<double> c_pyscf =
      read_array(entry.at("arrays").at("c").get<std::string>(), {nmo, nmo});
  std::vector<double> c_raw(nn);
  for (std::size_t j = 0; j < nmo; ++j) {
    for (std::size_t m = 0; m < nmo; ++m) {
      c_raw[m + (nmo * j)] = c_pyscf[row[m] + (nmo * j)] / norm[m];
    }
  }
  const auto orthonormal = [&](const std::vector<double> &s) {
    const std::vector<double> sc = host_gemm(nmo, nmo, nmo, s, false, c_raw);
    return host_gemm(nmo, nmo, nmo, c_raw, false,
                     inverse_sqrt(host_gemm(nmo, nmo, nmo, c_raw, true, sc), nmo));
  };
  const std::vector<double> c_host = orthonormal(ref.s);
  fb::SphericalMatrixAO c(res, *ref.basis);
  upload(res, c_host, c.data());
  fb::SphericalMatrixAO h_ref(res, *ref.basis);
  upload(res, ref.h, h_ref.data());

  // The chunks, as the consumer sees them.
  std::vector<double> h1(n * nn);
  std::vector<double> tau(n * nn);
  std::vector<double> w1(n * factor);
  std::vector<std::pair<std::int32_t, std::int32_t>> chunks;
  const auto consume = [&](const std::int32_t x0, const std::int32_t count,
                           const double *const h1_dev, const double *const tau_dev,
                           const double *const w1_dev) -> fb::Status {
    chunks.emplace_back(x0, count);
    const auto k = static_cast<std::size_t>(count);
    const auto from = static_cast<std::size_t>(x0);
    const std::vector<double> hk = download(res, h1_dev, k * nn);
    const std::vector<double> tk = download(res, tau_dev, k * nn);
    const std::vector<double> wk = download(res, w1_dev, k * factor);
    std::ranges::copy(hk, h1.begin() + static_cast<std::ptrdiff_t>(from * nn));
    std::ranges::copy(tk, tau.begin() + static_cast<std::ptrdiff_t>(from * nn));
    std::ranges::copy(wk, w1.begin() + static_cast<std::ptrdiff_t>(from * factor));
    return {};
  };
  const auto classes = fb::primitive_pair_classes(res, *ref.basis);
  ASSERT_TRUE(classes) << classes.error().message;
  const auto call = [&](const std::size_t max_bytes) {
    return fb::mcscf::df_integral_derivatives(res, *ref.molecule, *ref.basis, *classes, *ref.aux,
                                              h_ref, b, l.data(), c, consume, max_bytes);
  };
  // No consumer is refused; so is a budget below one coordinate, naming the
  // bytes one needs, and that plus one more coordinate's W^x is a budget of
  // exactly two.
  {
    const fb::Status none = fb::mcscf::df_integral_derivatives(
        res, *ref.molecule, *ref.basis, *classes, *ref.aux, h_ref, b, l.data(), c,
        std::function<fb::Status(std::int32_t, std::int32_t, const double *, const double *,
                                 const double *)>{});
    ASSERT_FALSE(none.has_value());
    EXPECT_EQ(none.error().kind, fb::ErrorKind::InvalidConfig);
  }
  const fb::Status tiny = call(1);
  ASSERT_FALSE(tiny.has_value());
  EXPECT_EQ(tiny.error().kind, fb::ErrorKind::Unsupported);
  EXPECT_TRUE(chunks.empty());
  const std::string &message = tiny.error().message;
  const std::size_t needs = message.find(" needs ");
  ASSERT_NE(needs, std::string::npos) << message;
  const std::size_t need = std::stoull(message.substr(needs + 7));
  const fb::Status made = call(need + (sizeof(double) * factor));
  ASSERT_TRUE(made) << made.error().message;
  std::vector<std::pair<std::int32_t, std::int32_t>> want_chunks;
  for (std::size_t x0 = 0; x0 < n; x0 += 2) {
    want_chunks.emplace_back(static_cast<std::int32_t>(x0),
                             static_cast<std::int32_t>(std::min<std::size_t>(2, n - x0)));
  }
  EXPECT_EQ(chunks, want_chunks);

  // (h_MO, L) at a displaced geometry, in the reference's gauge.
  const auto integrals = [&](const Geometry &g) {
    const std::vector<double> sc = host_gemm(nmo, nmo, nmo, g.s, false, c_host);
    const std::vector<double> x =
        host_gemm(nmo, nmo, nmo, c_host, false,
                  inverse_sqrt(host_gemm(nmo, nmo, nmo, c_host, true, sc), nmo));
    const auto to_mo = [&](const std::vector<double> &ao) {
      const std::vector<double> ax = host_gemm(nmo, nmo, nmo, ao, false, x);
      return host_gemm(nmo, nmo, nmo, x, true, ax);
    };
    std::vector<double> h_mo = to_mo(g.h);
    // K[p, q, P] = (X^T T_P X)[p, q], T_P[m, n] at P + naux (m + nao n).
    std::vector<double> k(factor);
    std::vector<double> tp(nn);
    for (std::size_t p = 0; p < naux; ++p) {
      for (std::size_t mn = 0; mn < nn; ++mn) {
        tp[mn] = g.t[p + (naux * mn)];
      }
      std::ranges::copy(to_mo(tp), k.begin() + static_cast<std::ptrdiff_t>(p * nn));
    }
    // G = L_M^-1 M L_M^-T (symmetric), Z = L_M^-T G^-1/2.
    const std::vector<double> half = lower_solve(l_m, naux, g.m, naux, false);
    const std::vector<double> gm = lower_solve(l_m, naux, transpose(half, naux), naux, false);
    const std::vector<double> z = lower_solve(l_m, naux, inverse_sqrt(gm, naux), naux, true);
    return std::pair{std::move(h_mo), host_gemm(nn, naux, naux, k, false, z)};
  };

  double worst = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    SCOPED_TRACE(std::format("coordinate {}", x));
    std::array<std::pair<std::vector<double>, std::vector<double>>, 4> f;
    const std::array<double, 4> steps{kLadderStep, -kLadderStep, kLadderStep / 2, -kLadderStep / 2};
    for (std::size_t i = 0; i < steps.size(); ++i) {
      std::vector<fb::Atom> moved = atoms;
      fb::Vec<double, 3> &r = moved[x / 3].position;
      (x % 3 == 0 ? r.x : x % 3 == 1 ? r.y : r.z) += steps[i];
      const Geometry g = at(moved);
      ASSERT_TRUE(g.basis && g.aux);
      f[i] = integrals(g);
    }
    // Richardson: (4 D(step / 2) - D(step)) / 3 per element, then the worst
    // miss of each quantity relative to its largest element.
    const auto miss = [&](const auto &part, const double *analytic, const std::size_t len) {
      double diff = 0.0;
      double scale = 0.0;
      for (std::size_t i = 0; i < len; ++i) {
        const double d1 = (part(f[0])[i] - part(f[1])[i]) / (2 * kLadderStep);
        const double d2 = (part(f[2])[i] - part(f[3])[i]) / kLadderStep;
        const double ladder = ((4 * d2) - d1) / 3;
        diff = std::max(diff, std::abs(analytic[i] - ladder));
        scale = std::max(scale, std::abs(ladder));
      }
      return std::pair{diff / scale, scale};
    };
    const auto [h_miss, h_scale] =
        miss([](const auto &v) -> const std::vector<double> & { return v.first; },
             h1.data() + (x * nn), nn);
    const auto [w_miss, w_scale] =
        miss([](const auto &v) -> const std::vector<double> & { return v.second; },
             w1.data() + (x * factor), factor);
    EXPECT_LT(h_miss, kLadderTol) << "h1^x, of max " << h_scale;
    EXPECT_LT(w_miss, kLadderTol) << "W^x, of max " << w_scale;
    worst = std::max({worst, h_miss, w_miss});
    std::cout << std::format("[ ladder   ] x {}: h1^x {:.1e} of max {:.2e}, W^x {:.1e} of max "
                             "{:.2e} (relative)\n",
                             x, h_miss, h_scale, w_miss, w_scale);
  }
  std::cout << std::format("[ ladder   ] {} chunks of <= 2 coordinates; worst relative miss "
                           "{:.1e}\n",
                           chunks.size(), worst);
}

INSTANTIATE_TEST_SUITE_P(Cases, McscfThirdDerivative,
                         ::testing::Values("h2o_631g_sa2", "h2o_631g_sa2_unequal", "h2o_631g_sa3",
                                           "h2o_631g_sa3_unequal"),
                         [](const auto &info) { return std::string{info.param}; });

} // namespace
