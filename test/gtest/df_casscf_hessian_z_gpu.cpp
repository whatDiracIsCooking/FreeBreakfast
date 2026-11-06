// GPU cases for hessian::df_casscf_z, the Z terms of a single-state
// DF-CASSCF reference's Lagrangian Hessian at given multipliers (Z, z)
// (nuclear_hessian.tex eq. hess), each against a Richardson ladder of Z . g
// itself, g the CASSCF energy's exact gradient in its parameters: the
// stretched water's CAS(4,4) / 6-31G over cc-pVDZ-JKFIT
// (test/gtest/data/casscf_hessian/'s h2o_631g geometry and orbitals, the
// orbitals made orthonormal over this tree's S and the root re-solved at
// them), random (Z, z) and a random block Lambda of three vectors, all in
// mcscf::zvector's coordinates. Z . g is evaluated here, on the host, from
// this tree's integrals at each displaced geometry along the symmetric
// connection C(R) = C (C^T S(R) C)^-1/2 and the exponential parametrization
// C(R) e^K, e^A c (A = d c^T - c d^T), every derivative of the exponentials
// exact (their series), only the RDMs from ci::Rdm:
//
//     Z . d2 g / dR dR   against d2/dR_x dR_y Z.g(R, 0)
//     Z . d2 g / dR dx   against d2/dR_x dt Z.g(R, t lambda_j)
//     Lambda^T (Z . d2 g / dx dx) Lambda  against d2/ds dt Z.g(R_0, s lambda_i + t lambda_j)
//
// Beside it, the refusals.

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
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.hessian;

namespace {

namespace fb = free_breakfast;

// The issue's bar, absolute, for each of the three terms against its ladder.
constexpr double kLadderTol = 1e-7;
// The ladders' steps: the displacement in bohr, and the parameter step along
// a vector (whose elements are at most kAmplitude); each with half of it
// beside it (Richardson). Rounding, not truncation, limits them: the third
// term's miss is 4.1e-9 at these, 3.1e-8 at half and 6.9e-8 at a quarter.
constexpr double kGeometryStep = 4e-3;
constexpr double kParameterStep = 4e-3;
// The random vectors' elements, uniform in [-kAmplitude, kAmplitude]: the
// size of a CASSCF's z here (the fixture's are up to 0.2).
constexpr double kAmplitude = 0.1;
// Lambda's columns.
constexpr std::int32_t kBlock = 3;
// The host energy at the reference against the CASCI root's.
constexpr double kEnergyTol = 1e-10;
// Terms of the exponentials' series: the arguments' norms are below 0.05.
constexpr std::int32_t kSeriesTerms = 24;

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
// unit-normalised one.
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

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(kDataDir / "casscf_hessian.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_hessian.json";
  static const fb::json kNone;
  return kNone;
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

// exp(X) of a small n x n X by its series.
std::vector<double> expm(const std::vector<double> &x, std::size_t n) {
  std::vector<double> out(n * n, 0.0);
  std::vector<double> term(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out[i * (n + 1)] = 1.0;
    term[i * (n + 1)] = 1.0;
  }
  for (std::int32_t k = 1; k < kSeriesTerms; ++k) {
    term = host_gemm(n, n, n, term, false, x);
    for (std::size_t i = 0; i < n * n; ++i) {
      term[i] /= k;
      out[i] += term[i];
    }
  }
  return out;
}

// e^-X d/de e^(X + e Y) at e = 0: Sum_k (-1)^k / (k + 1)! ad_X^k Y.
std::vector<double> dexp(const std::vector<double> &x, const std::vector<double> &y,
                         std::size_t n) {
  std::vector<double> out = y;
  std::vector<double> term = y;
  for (std::int32_t k = 1; k < kSeriesTerms; ++k) {
    const std::vector<double> xt = host_gemm(n, n, n, x, false, term);
    const std::vector<double> tx = host_gemm(n, n, n, term, false, x);
    for (std::size_t i = 0; i < n * n; ++i) {
      term[i] = -(xt[i] - tx[i]) / (k + 1);
      out[i] += term[i];
    }
  }
  return out;
}

// A deterministic uniform draw in [-1, 1] (splitmix64), the same on every
// standard library.
class Draw {
public:
  explicit Draw(std::uint64_t seed) : state_(seed) {}
  double operator()() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    z ^= z >> 31U;
    return (2.0 * static_cast<double>(z >> 11U) / static_cast<double>(1ULL << 53U)) - 1.0;
  }

private:
  std::uint64_t state_;
};

// The case: the bases at the fixture's geometry, the fit, the orbitals and
// the root, and Z . g at any (h_MO, L) and parameters.
class HessianDfCasscfZ : public ::testing::Test {
protected:
  // Everything at one geometry: the placed bases, S, h, the metric M and the
  // three-center T (unfitted), and the molecule they were placed on.
  struct Geometry {
    std::optional<fb::Molecule> molecule;
    std::optional<fb::BasisSet> basis;
    std::optional<fb::BasisSet> aux;
    std::vector<double> s, h, m, t;
  };

  Geometry at(const std::vector<fb::Atom> &where) {
    fb::DeviceResources &res = *res_;
    Geometry g;
    g.molecule.emplace(res, where);
    auto basis = fb::place(stamp_->primitive_normalized(), *g.molecule);
    EXPECT_TRUE(basis) << basis.error().message;
    auto aux = fb::place(aux_stamp_->primitive_normalized(), *g.molecule);
    EXPECT_TRUE(aux) << aux.error().message;
    if (!basis || !aux) {
      return g;
    }
    EXPECT_EQ(basis->nao_sphe(), nmo_);
    EXPECT_EQ(aux->nao_sphe(), naux_);
    auto classes = fb::primitive_pair_classes(res, *basis);
    EXPECT_TRUE(classes) << classes.error().message;
    auto basis_u = fb::upload_spherical_transform_matrix(res, *basis);
    EXPECT_TRUE(basis_u) << basis_u.error().message;
    auto aux_u = fb::upload_spherical_transform_matrix(res, *aux);
    EXPECT_TRUE(aux_u) << aux_u.error().message;
    if (!classes || !basis_u || !aux_u) {
      return g;
    }
    const std::size_t nn = nmo_ * nmo_;
    fb::MatrixAOPair s_pair(res, *basis, *basis_u);
    EXPECT_TRUE(fb::overlap(res, *classes, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(res));
    fb::MatrixAOPair h_pair(res, *basis, *basis_u);
    EXPECT_TRUE(fb::core_hamiltonian(res, *classes, *g.molecule, h_pair.cartesian()));
    EXPECT_TRUE(h_pair.cartesian_to_spherical(res));
    fb::MatrixAOPair metric(res, *aux, std::move(*aux_u));
    EXPECT_TRUE(fb::metric(res, *aux, metric));
    fb::Buffers::device<double> t(nn * naux_, res.device());
    EXPECT_TRUE(fb::three_center(res, *basis, *classes, *aux, t.data()));
    g.s = download(res, s_pair.spherical().data(), nn);
    g.h = download(res, h_pair.spherical().data(), nn);
    g.m = download(res, metric.spherical().data(), naux_ * naux_);
    g.t = download(res, t.data(), nn * naux_);
    g.basis.emplace(std::move(*basis));
    g.aux.emplace(std::move(*aux));
    return g;
  }

  void load() {
    const fb::json &entry = reference_case("h2o_631g");
    ASSERT_FALSE(entry.is_null());
    ASSERT_EQ(entry.at("weights").size(), 1U);
    nmo_ = entry.at("nmo").get<std::size_t>();
    naux_ = entry.at("naux").get<std::size_t>();
    ncore_ = entry.at("ncore").get<std::int32_t>();
    ncas_ = entry.at("ncas").get<std::int32_t>();
    const auto nelecas = entry.at("nelecas").get<std::vector<std::int32_t>>();
    ASSERT_EQ(nelecas[0], nelecas[1]);
    nocc_ = static_cast<std::size_t>(ncore_ + ncas_);
    res_.emplace(0);
    fb::DeviceResources &res = *res_;
    for (const fb::json &atom : entry.at("atoms")) {
      const auto xyz = atom.at("coord").get<std::vector<double>>();
      atoms_.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                                static_cast<fb::Element>(atom.at("charge").get<std::int32_t>())});
    }
    auto stamp = fb::resolve_basis(entry.at("basis").get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    stamp_.emplace(std::move(*stamp));
    auto aux_stamp = fb::resolve_aux_basis(entry.at("aux_basis").get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    aux_stamp_.emplace(std::move(*aux_stamp));
    ref_ = at(atoms_);
    ASSERT_TRUE(ref_.basis && ref_.aux);
    const std::size_t nn = nmo_ * nmo_;
    n_x_ = 3 * atoms_.size();

    // The fit at the reference (L_M = chol(M), B = L_M^-1 T) on the device,
    // as the call takes it.
    b_.emplace(nn * naux_, res.device());
    l_.emplace(naux_ * naux_, res.device());
    upload(res, ref_.t, b_->data());
    upload(res, ref_.m, l_->data());
    ASSERT_TRUE(fb::density_fit(res, static_cast<std::int32_t>(nmo_),
                                static_cast<std::int32_t>(naux_), l_->data(), b_->data()));
    auto classes = fb::primitive_pair_classes(res, *ref_.basis);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    h_.emplace(res, *ref_.basis);
    upload(res, ref_.h, h_->data());

    // The fixture's orbitals, rows into this basis's order and normalisation,
    // made orthonormal over this S: C (C^T S C)^-1/2.
    const auto [row, norm] = pyscf_rows(*ref_.basis);
    const std::vector<double> c_pyscf =
        read_array(entry.at("arrays").at("c").get<std::string>(), {nmo_, nmo_});
    std::vector<double> c_raw(nn);
    for (std::size_t j = 0; j < nmo_; ++j) {
      for (std::size_t m = 0; m < nmo_; ++m) {
        c_raw[m + (nmo_ * j)] = c_pyscf[row[m] + (nmo_ * j)] / norm[m];
      }
    }
    const std::vector<double> sc = host_gemm(nmo_, nmo_, nmo_, ref_.s, false, c_raw);
    c_host_ = host_gemm(nmo_, nmo_, nmo_, c_raw, false,
                        inverse_sqrt(host_gemm(nmo_, nmo_, nmo_, c_raw, true, sc), nmo_));

    // The root re-solved at these orbitals: an eigenvector of the
    // active-space H over these integrals.
    fb::SphericalMatrixAO c(res, *ref_.basis);
    upload(res, c_host_, c.data());
    auto ints = fb::casscf_integrals(res, *ref_.basis, *ref_.molecule, *h_, *b_,
                                     static_cast<std::int32_t>(naux_), c, ncore_, ncas_);
    ASSERT_TRUE(ints) << ints.error().message;
    auto strings = fb::ci::DeviceStringSet::create(res, ncas_, nelecas[0]);
    ASSERT_TRUE(strings) << strings.error().message;
    strings_.emplace(std::move(*strings));
    auto sigma = fb::ci::Sigma::create(res, *strings_, *strings_);
    ASSERT_TRUE(sigma) << sigma.error().message;
    const auto n = static_cast<std::size_t>(ncas_);
    ASSERT_TRUE(sigma->set_integrals(res, ints->hamiltonian.e_core,
                                     download(res, ints->hamiltonian.h_eff.data(), n * n),
                                     download(res, ints->hamiltonian.eri.data(), n * n * n * n)));
    dim_ = static_cast<std::size_t>(sigma->size());
    fb::Buffers::device<double> diagonal(dim_, res.device());
    ASSERT_TRUE(sigma->diagonal(res, diagonal.data()));
    auto casci = fb::ci::solve(res, *sigma, *strings_, *strings_, diagonal.data(), 1);
    ASSERT_TRUE(casci) << casci.error().message;
    root_ = download(res, casci->vectors.data(), dim_);
    energy_ = casci->energies[0];
    auto rdm = fb::ci::Rdm::create(res, *strings_, *strings_);
    ASSERT_TRUE(rdm) << rdm.error().message;
    rdm_.emplace(std::move(*rdm));

    fb::mcscf::Solution s{.c = fb::SphericalMatrixAO(res, *ref_.basis),
                          .roots = {{energy_}, {0.0}, std::move(casci->vectors), {}, 0},
                          .energy = energy_,
                          .gradient_norm = 0.0,
                          .gamma = {},
                          .gamma2 = {},
                          .root_gammas = {},
                          .iterations = {}};
    upload(res, c_host_, s.c.data());
    const auto [gamma, gamma2] = rdms(root_);
    s.gamma = gamma;
    s.gamma2 = gamma2;
    s.root_gammas = gamma;
    solution_.emplace(std::move(s));

    // The packed pairs, (p, q) with p > q in response's order: q core then
    // active, p from the next space up.
    for (std::size_t q = 0; q < nocc_; ++q) {
      for (std::size_t p = q < static_cast<std::size_t>(ncore_) ? ncore_ : nocc_; p < nmo_; ++p) {
        pairs_.emplace_back(p, q);
      }
    }
    size_ = pairs_.size() + dim_;
  }

  // A random vector over the parameters, its CI part off the root.
  std::vector<double> random_vector(Draw &draw) const {
    std::vector<double> v(size_);
    for (double &x : v) {
      x = kAmplitude * draw();
    }
    double along = 0.0;
    for (std::size_t i = 0; i < dim_; ++i) {
      along += v[pairs_.size() + i] * root_[i];
    }
    for (std::size_t i = 0; i < dim_; ++i) {
      v[pairs_.size() + i] -= along * root_[i];
    }
    return v;
  }

  // gamma (ncas^2) and Gamma (ncas^4), row-major, of one vector, unscaled.
  std::pair<std::vector<double>, std::vector<double>> rdms(const std::vector<double> &x) {
    fb::DeviceResources &res = *res_;
    const auto n = static_cast<std::size_t>(ncas_);
    fb::Buffers::device<double> v(dim_, res.device());
    upload(res, x, v.data());
    fb::Buffers::device<double> out((n * n) + (n * n * n * n), res.device());
    const std::array<double, 1> one{1.0};
    EXPECT_TRUE(rdm_->rdm12(res, one, v.data(), out.data(), out.data() + (n * n)).has_value());
    const std::vector<double> all = download(res, out.data(), out.num_elements());
    return {std::vector<double>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n * n)),
            std::vector<double>(all.begin() + static_cast<std::ptrdiff_t>(n * n), all.end())};
  }

  // (h_MO, L) at a geometry: h_MO = X^T h X and L_Q = Sum_P X^T T_P X
  // (L_M^-T)_PQ, X = C (C^T S C)^-1/2 the connection and L_M M's Cholesky
  // factor there, L_Q at p + nmo (q + nmo Q). Any factor of M^-1 would do:
  // the energy sees L only as Sum_Q L_Q L_Q.
  std::pair<std::vector<double>, std::vector<double>> integrals(const Geometry &g) const {
    const std::size_t nn = nmo_ * nmo_;
    const std::vector<double> sc = host_gemm(nmo_, nmo_, nmo_, g.s, false, c_host_);
    const std::vector<double> x =
        host_gemm(nmo_, nmo_, nmo_, c_host_, false,
                  inverse_sqrt(host_gemm(nmo_, nmo_, nmo_, c_host_, true, sc), nmo_));
    const auto to_mo = [&](const std::vector<double> &ao) {
      const std::vector<double> ax = host_gemm(nmo_, nmo_, nmo_, ao, false, x);
      return host_gemm(nmo_, nmo_, nmo_, x, true, ax);
    };
    std::vector<double> h_mo = to_mo(g.h);
    // K^T: row P the MO (X^T T_P X), naux x nn.
    std::vector<double> kt(naux_ * nn);
    std::vector<double> tp(nn);
    for (std::size_t p = 0; p < naux_; ++p) {
      for (std::size_t mn = 0; mn < nn; ++mn) {
        tp[mn] = g.t[p + (naux_ * mn)];
      }
      const std::vector<double> mo = to_mo(tp);
      for (std::size_t pq = 0; pq < nn; ++pq) {
        kt[p + (naux_ * pq)] = mo[pq];
      }
    }
    // M = L_M L_M^T, then L^T = L_M^-1 K^T.
    std::vector<double> chol = g.m;
    for (std::size_t j = 0; j < naux_; ++j) {
      for (std::size_t k = 0; k < j; ++k) {
        for (std::size_t i = j; i < naux_; ++i) {
          chol[i + (naux_ * j)] -= chol[i + (naux_ * k)] * chol[j + (naux_ * k)];
        }
      }
      const double pivot = std::sqrt(chol[j * (naux_ + 1)]);
      for (std::size_t i = j; i < naux_; ++i) {
        chol[i + (naux_ * j)] /= pivot;
      }
    }
    const std::vector<double> lt = lower_solve(chol, naux_, std::move(kt), nn, false);
    std::vector<double> l(nn * naux_);
    for (std::size_t pq = 0; pq < nn; ++pq) {
      for (std::size_t q = 0; q < naux_; ++q) {
        l[pq + (nn * q)] = lt[q + (naux_ * pq)];
      }
    }
    return {std::move(h_mo), std::move(l)};
  }

  // K (nmo x nmo, antisymmetric) of a vector's packed pairs: K_pq = v,
  // K_qp = -v.
  [[nodiscard]] std::vector<double> kappa(const std::vector<double> &v) const {
    std::vector<double> k(nmo_ * nmo_, 0.0);
    for (std::size_t i = 0; i < pairs_.size(); ++i) {
      const auto [p, q] = pairs_[i];
      k[p + (nmo_ * q)] = v[i];
      k[q + (nmo_ * p)] = -v[i];
    }
    return k;
  }

  // A = d c^T - c d^T (dim x dim) of a vector's CI part d.
  [[nodiscard]] std::vector<double> rotation(const std::vector<double> &v) const {
    std::vector<double> a(dim_ * dim_);
    const double *d = v.data() + pairs_.size();
    for (std::size_t j = 0; j < dim_; ++j) {
      for (std::size_t i = 0; i < dim_; ++i) {
        a[i + (dim_ * j)] = (d[i] * root_[j]) - (root_[i] * d[j]);
      }
    }
    return a;
  }

  // Z . g at the integrals (h_MO, L) and the parameters lambda: d/de of the
  // energy at C e^(K(lambda) + e K(Z)) and e^(A(lambda) + e A(Z)) c, exactly
  // (the exponentials' derivatives by their series), the orbital part by
  // dE/dU_o and the CI part as 2 c'^T H dc' = (E(c' + dc') - E(c' - dc')) / 2,
  // E quadratic in the vector (its densities differenced first). With @p energy, E at lambda instead.
  double zg(const std::pair<std::vector<double>, std::vector<double>> &ints,
            const std::vector<double> &lambda, const std::vector<double> &z, bool energy = false) {
    const auto &[h_mo, l_mo] = ints;
    const std::size_t nmo = nmo_;
    const std::size_t nocc = nocc_;
    const std::size_t no2 = nocc * nocc;
    const auto core = static_cast<std::size_t>(ncore_);
    const auto n = static_cast<std::size_t>(ncas_);
    const std::vector<double> k = kappa(lambda);
    const std::vector<double> u = expm(k, nmo);
    const std::vector<double> du = host_gemm(nmo, nmo, nmo, u, false, dexp(k, kappa(z), nmo));
    const std::vector<double> a = rotation(lambda);
    const std::vector<double> y = expm(a, dim_);
    const std::vector<double> c1 = host_gemm(dim_, dim_, 1, y, false, root_);
    const std::vector<double> dc =
        host_gemm(dim_, dim_, 1, y, false,
                  host_gemm(dim_, dim_, 1, dexp(a, rotation(z), dim_), false, root_));

    // U_o, h U_o, h_o = U_o^T h U_o; per Q, L_Q U_o and M_Q = U_o^T L_Q U_o.
    const std::vector<double> uo(u.begin(), u.begin() + static_cast<std::ptrdiff_t>(nmo * nocc));
    const std::vector<double> huo = host_gemm(nmo, nmo, nocc, h_mo, false, uo);
    const std::vector<double> ho = host_gemm(nocc, nmo, nocc, uo, true, huo);
    std::vector<std::vector<double>> luo(naux_);
    std::vector<std::vector<double>> mq(naux_);
    for (std::size_t q = 0; q < naux_; ++q) {
      const std::vector<double> lq(l_mo.begin() + static_cast<std::ptrdiff_t>(q * nmo * nmo),
                                   l_mo.begin() + static_cast<std::ptrdiff_t>((q + 1) * nmo * nmo));
      luo[q] = host_gemm(nmo, nmo, nocc, lq, false, uo);
      mq[q] = host_gemm(nocc, nmo, nocc, uo, true, luo[q]);
    }
    // G_pqrs = Sum_Q M_Q,pq M_Q,rs.
    std::vector<double> g(no2 * no2, 0.0);
    for (std::size_t q = 0; q < naux_; ++q) {
      for (std::size_t rs = 0; rs < no2; ++rs) {
        const double m_rs = mq[q][rs];
        for (std::size_t pq = 0; pq < no2; ++pq) {
          g[pq + (no2 * rs)] += mq[q][pq] * m_rs;
        }
      }
    }
    // D and Gamma over the occupied orbitals of a vector x (its core weighted
    // by |x|^2): D at p + nocc q, Gamma at pq + nocc^2 rs.
    const auto full = [&](const std::vector<double> &x) {
      const auto [g1, g2] = rdms(x);
      const double n0 = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
      std::vector<double> d(no2, 0.0);
      std::vector<double> gg(no2 * no2, 0.0);
      const auto at = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double & {
        return gg[p + (nocc * q) + (no2 * (r + (nocc * s)))];
      };
      for (std::size_t i = 0; i < core; ++i) {
        d[i * (nocc + 1)] = 2 * n0;
        for (std::size_t j = 0; j < core; ++j) {
          at(i, i, j, j) += 4 * n0;
          at(i, j, j, i) -= 2 * n0;
        }
        for (std::size_t t = 0; t < n; ++t) {
          for (std::size_t w = 0; w < n; ++w) {
            const double gtw = g1[(t * n) + w];
            at(i, i, core + t, core + w) += 2 * gtw;
            at(core + t, core + w, i, i) += 2 * gtw;
            at(core + t, i, i, core + w) -= gtw;
            at(i, core + w, core + t, i) -= gtw;
          }
        }
      }
      for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t w = 0; w < n; ++w) {
          d[(core + t) + (nocc * (core + w))] = g1[(t * n) + w];
          for (std::size_t v = 0; v < n; ++v) {
            for (std::size_t x2 = 0; x2 < n; ++x2) {
              at(core + t, core + w, core + v, core + x2) +=
                  g2[(((((t * n) + w) * n) + v) * n) + x2];
            }
          }
        }
      }
      return std::pair{std::move(d), std::move(gg)};
    };
    const auto energy_of = [&](const std::vector<double> &x) {
      const auto [d, gg] = full(x);
      double e = 0.0;
      for (std::size_t i = 0; i < no2; ++i) {
        e += ho[i] * d[i];
      }
      for (std::size_t i = 0; i < no2 * no2; ++i) {
        e += 0.5 * g[i] * gg[i];
      }
      return e;
    };
    if (energy) {
      return energy_of(c1);
    }

    // The orbital part: dE/dU_o = h U_o (D + D^T) + Sum_Q L_Q U_o (W_Q + W_Q^T),
    // W_Q,pq = Sum_rs Gs_pqrs M_Q,rs, against dU_o.
    const auto [d, gg] = full(c1);
    std::vector<double> dsym(no2);
    for (std::size_t p = 0; p < nocc; ++p) {
      for (std::size_t q = 0; q < nocc; ++q) {
        dsym[p + (nocc * q)] = d[p + (nocc * q)] + d[q + (nocc * p)];
      }
    }
    std::vector<double> grad = host_gemm(nmo, nocc, nocc, huo, false, dsym);
    // T[m, q, rs] = Sum_Q (L_Q U_o)_mq M_Q,rs, then the two-electron part
    // Sum_q,rs T[m, q, rs] (Gs_pqrs + Gs_qprs), Gs = (Gamma_pqrs + Gamma_rspq) / 2.
    const std::size_t mo_occ = nmo * nocc;
    std::vector<double> lu_stack(mo_occ * naux_);
    std::vector<double> m_stack(naux_ * no2);
    for (std::size_t q = 0; q < naux_; ++q) {
      std::ranges::copy(luo[q], lu_stack.begin() + static_cast<std::ptrdiff_t>(q * mo_occ));
      for (std::size_t rs = 0; rs < no2; ++rs) {
        m_stack[q + (naux_ * rs)] = mq[q][rs];
      }
    }
    const std::vector<double> t = host_gemm(mo_occ, naux_, no2, lu_stack, false, m_stack);
    const auto gs = [&](std::size_t p, std::size_t q, std::size_t rs) {
      const std::size_t pq = p + (nocc * q);
      return 0.5 * (gg[pq + (no2 * rs)] + gg[rs + (no2 * pq)]);
    };
    for (std::size_t p = 0; p < nocc; ++p) {
      for (std::size_t q = 0; q < nocc; ++q) {
        for (std::size_t rs = 0; rs < no2; ++rs) {
          const double weight = gs(p, q, rs) + gs(q, p, rs);
          const double *tq = t.data() + (nmo * q) + (mo_occ * rs);
          for (std::size_t m = 0; m < nmo; ++m) {
            grad[m + (nmo * p)] += tq[m] * weight;
          }
        }
      }
    }
    double orbital = 0.0;
    for (std::size_t i = 0; i < nmo * nocc; ++i) {
      orbital += du[i] * grad[i];
    }
    // The CI part from the difference of the two vectors' densities, taken
    // before the contraction: the energies themselves are ~80 hartree.
    std::vector<double> plus = c1;
    std::vector<double> minus = c1;
    for (std::size_t i = 0; i < dim_; ++i) {
      plus[i] += dc[i];
      minus[i] -= dc[i];
    }
    const auto [d_plus, g_plus] = full(plus);
    const auto [d_minus, g_minus] = full(minus);
    double ci = 0.0;
    for (std::size_t i = 0; i < no2; ++i) {
      ci += ho[i] * (d_plus[i] - d_minus[i]);
    }
    for (std::size_t i = 0; i < no2 * no2; ++i) {
      ci += 0.5 * g[i] * (g_plus[i] - g_minus[i]);
    }
    return orbital + (0.5 * ci);
  }

  // The atoms with coordinate x moved by @p step.
  [[nodiscard]] std::vector<fb::Atom> moved(std::vector<fb::Atom> atoms, std::size_t x,
                                            double step) const {
    fb::Vec<double, 3> &r = atoms[x / 3].position;
    (x % 3 == 0 ? r.x : x % 3 == 1 ? r.y : r.z) += step;
    return atoms;
  }

  [[nodiscard]] fb::Result<fb::hessian::CasscfZTerms> run(const fb::Buffers::device<double> &z,
                                                          std::int32_t k, const double *lambda) {
    return fb::hessian::df_casscf_z(*res_, *ref_.molecule, *ref_.basis, classes_, *ref_.aux, *h_,
                                    *b_, l_->data(), *solution_, ncore_, ncas_, z, k, lambda);
  }

  std::optional<fb::DeviceResources> res_;
  std::optional<fb::BasisStamp> stamp_;
  std::optional<fb::BasisStamp> aux_stamp_;
  std::vector<fb::Atom> atoms_;
  Geometry ref_;
  std::size_t nmo_ = 0;
  std::size_t naux_ = 0;
  std::size_t nocc_ = 0;
  std::size_t dim_ = 0;
  std::size_t size_ = 0;
  std::size_t n_x_ = 0;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::SphericalMatrixAO> h_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> l_;
  std::vector<double> c_host_;
  std::optional<fb::ci::DeviceStringSet> strings_;
  std::optional<fb::ci::Rdm> rdm_;
  std::vector<double> root_;
  double energy_ = 0.0;
  std::optional<fb::mcscf::Solution> solution_;
  std::vector<std::pair<std::size_t, std::size_t>> pairs_;
};

TEST_F(HessianDfCasscfZ, MatchesLaddersOfZg) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  load();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_FALSE(HasFailure());
  fb::DeviceResources &res = *res_;
  const std::size_t n = n_x_;

  Draw draw(808);
  const std::vector<double> z_host = random_vector(draw);
  std::vector<double> lambda_host;
  for (std::int32_t j = 0; j < kBlock; ++j) {
    const std::vector<double> v = random_vector(draw);
    lambda_host.insert(lambda_host.end(), v.begin(), v.end());
  }
  fb::Buffers::device<double> z(size_, res.device());
  upload(res, z_host, z.data());
  fb::Buffers::device<double> lambda(lambda_host.size(), res.device());
  upload(res, lambda_host, lambda.data());
  const auto column = [&](std::size_t j) {
    return std::vector<double>(lambda_host.begin() + static_cast<std::ptrdiff_t>(j * size_),
                               lambda_host.begin() + static_cast<std::ptrdiff_t>((j + 1) * size_));
  };

  const auto begin = std::chrono::steady_clock::now();
  auto got = run(z, kBlock, lambda.data());
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->lxy.exy.size(), n * n);
  ASSERT_EQ(got->lxlambda.num_elements(), size_ * n);
  ASSERT_EQ(got->third.size(), static_cast<std::size_t>(kBlock * kBlock));
  const std::vector<double> lxl = download(res, got->lxlambda.data(), size_ * n);

  // The host energy is the CASCI root's at the reference.
  const auto ref_ints = integrals(ref_);
  const std::vector<double> zero(size_, 0.0);
  EXPECT_NEAR(zg(ref_ints, zero, z_host, true),
              energy_ - fb::nuclear_repulsion_energy(*ref_.molecule), kEnergyTol);

  const auto ladder_begin = std::chrono::steady_clock::now();
  // The integrals at the reference displaced by a along x and b along y.
  const auto ints_at = [&](std::size_t x, double a, std::size_t y, double b) {
    const Geometry g = at(moved(moved(atoms_, x, a), y, b));
    EXPECT_TRUE(g.basis && g.aux);
    return integrals(g);
  };
  const auto richardson = [](double coarse, double fine) { return ((4 * fine) - coarse) / 3; };

  // Z . d2 g / dR dR: d2/dR_x dR_y Z.g(R, 0), a mixed central difference
  // (at x = y the four points are +-2h and twice 0).
  double worst_xy = 0.0;
  const double scale_xy = std::ranges::fold_left(
      got->lxy.exy, 0.0, [](double m, double v) { return std::max(m, std::abs(v)); });
  for (std::size_t x = 0; x < n; ++x) {
    for (std::size_t y = x; y < n; ++y) {
      std::array<double, 2> level{};
      for (std::size_t i = 0; i < 2; ++i) {
        const double h = kGeometryStep / static_cast<double>(1U << i);
        const double f =
            zg(ints_at(x, h, y, h), zero, z_host) - zg(ints_at(x, h, y, -h), zero, z_host) -
            zg(ints_at(x, -h, y, h), zero, z_host) + zg(ints_at(x, -h, y, -h), zero, z_host);
        level[i] = f / (4 * h * h);
      }
      const double want = richardson(level[0], level[1]);
      const double miss = std::max(std::abs(got->lxy.exy[x + (n * y)] - want),
                                   std::abs(got->lxy.exy[y + (n * x)] - want));
      EXPECT_LE(miss, kLadderTol) << "Z.d2g/dR dR (" << x << ", " << y << ")";
      worst_xy = std::max(worst_xy, miss);
    }
  }

  // Z . d2 g / dR dx: column x of lxlambda against lambda_j, d2/dR_x dt
  // Z.g(R, t lambda_j).
  double worst_xl = 0.0;
  double scale_xl = 0.0;
  for (std::size_t x = 0; x < n; ++x) {
    std::array<std::array<double, kBlock>, 2> level{};
    for (std::size_t i = 0; i < 2; ++i) {
      const double h = kGeometryStep / static_cast<double>(1U << i);
      const double t = kParameterStep / static_cast<double>(1U << i);
      const auto plus = ints_at(x, h, x, 0.0);
      const auto minus = ints_at(x, -h, x, 0.0);
      for (std::size_t j = 0; j < static_cast<std::size_t>(kBlock); ++j) {
        std::vector<double> up = column(j);
        std::vector<double> down = column(j);
        for (std::size_t e = 0; e < size_; ++e) {
          up[e] *= t;
          down[e] *= -t;
        }
        level[i][j] = (zg(plus, up, z_host) - zg(plus, down, z_host) - zg(minus, up, z_host) +
                       zg(minus, down, z_host)) /
                      (4 * h * t);
      }
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(kBlock); ++j) {
      double analytic = 0.0;
      for (std::size_t e = 0; e < size_; ++e) {
        analytic += lxl[e + (size_ * x)] * lambda_host[e + (size_ * j)];
      }
      const double want = richardson(level[0][j], level[1][j]);
      EXPECT_LE(std::abs(analytic - want), kLadderTol) << "Z.d2g/dR dx (" << x << ", " << j << ")";
      worst_xl = std::max(worst_xl, std::abs(analytic - want));
      scale_xl = std::max(scale_xl, std::abs(want));
    }
  }

  // Lambda^T (Z . d2 g / dx dx) Lambda: d2/ds dt Z.g(R_0, s lambda_i + t lambda_j).
  double worst_xx = 0.0;
  const double scale_xx = std::ranges::fold_left(
      got->third, 0.0, [](double m, double v) { return std::max(m, std::abs(v)); });
  for (std::size_t i = 0; i < static_cast<std::size_t>(kBlock); ++i) {
    for (std::size_t j = i; j < static_cast<std::size_t>(kBlock); ++j) {
      std::array<double, 2> level{};
      for (std::size_t l = 0; l < 2; ++l) {
        const double t = kParameterStep / static_cast<double>(1U << l);
        const auto point = [&](double si, double sj) {
          std::vector<double> v(size_);
          const std::vector<double> a = column(i);
          const std::vector<double> b = column(j);
          for (std::size_t e = 0; e < size_; ++e) {
            v[e] = (si * a[e]) + (sj * b[e]);
          }
          return zg(ref_ints, v, z_host);
        };
        level[l] = (point(t, t) - point(t, -t) - point(-t, t) + point(-t, -t)) / (4 * t * t);
      }
      const double want = richardson(level[0], level[1]);
      const auto k = static_cast<std::size_t>(kBlock);
      const double miss = std::max(std::abs(got->third[i + (k * j)] - want),
                                   std::abs(got->third[j + (k * i)] - want));
      EXPECT_LE(miss, kLadderTol) << "Z.d2g/dx dx (" << i << ", " << j << ")";
      worst_xx = std::max(worst_xx, miss);
    }
  }
  const std::chrono::duration<double> ladder = std::chrono::steady_clock::now() - ladder_begin;
  std::cout << std::format(
      "[ z terms  ] h2o_631g, one root: df_casscf_z {:.2f} s, ladders {:.1f} s\n"
      "[ z terms  ] Z.d2g/dRdR {:.1e} of max {:.2e}; Z.d2g/dRdx {:.1e} of max {:.2e}; "
      "Z.d2g/dxdx {:.1e} of max {:.2e}\n",
      seconds.count(), ladder.count(), worst_xy, scale_xy, worst_xl, scale_xl, worst_xx, scale_xx);
}

TEST_F(HessianDfCasscfZ, Refusals) {
  FBF_REQUIRE_GPU();
  load();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_FALSE(HasFailure());
  fb::DeviceResources &res = *res_;
  const auto refused = [](const auto &made, std::string_view what) {
    ASSERT_FALSE(made.has_value()) << what;
    EXPECT_EQ(made.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_TRUE(made.error().message.starts_with("hessian::df_casscf_z")) << what;
  };
  fb::Buffers::device<double> z(size_, res.device());
  fb::Buffers::device<double> short_z(size_ - 1, res.device());
  refused(run(short_z, 0, nullptr), "z one short");
  refused(run(z, -1, nullptr), "a negative k");
  refused(run(z, 1, nullptr), "no Lambda for k = 1");
  refused(fb::hessian::df_casscf_z(res, *ref_.molecule, *ref_.basis, classes_, *ref_.aux, *h_, *b_,
                                   l_->data(), *solution_, ncore_, 0, z),
          "ncas 0");
}

} // namespace
