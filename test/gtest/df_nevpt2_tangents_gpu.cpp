// GPU cases for hessian::df_nevpt2_tangents, the NEVPT2 parameter response
// (lambda^x's kappa_nr and c^x, kappa_oo^x) and the total tangent t_x of
// (h_MO, B_MO, c), on this tree's own DF-CASSCF through the registry. On
// data/nevpt2_hessian/'s skew formaldehyde (#803): t_x against a Richardson
// ladder of (h_MO, B_MO, c) of the CASSCF re-solved at each displaced
// geometry and re-canonicalized, in the gauge the tangent is in (the
// connection's frame, the active orbitals turned by no active-active
// rotation, B's fitted index by the symmetric metric gauge), handed over two
// coordinates a chunk. On a C3v ammonia, whose core and virtual e pairs are
// degenerate: SC refused, naming a pair and a coordinate, and PC's records.
// Then the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.response.zvector;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The reference CASSCF's ||g||, the registry's.
constexpr double kConvTolGrad = 1e-10;
// Every displaced solve's: the ladder divides each displaced solution's
// error, about ||g|| over the parameters' Hessian, by the step. At 1e-10
// (worst 6.8e-11 reached) and 4e-3 bohr, h's ladder sat at 1.7e-7 of the
// bar's 1e-7; at 1e-12 (worst 9.9e-13 reached, gfx1200) and 1e-3 bohr,
// c's at 5.8e-8, the noisiest.
constexpr double kSolveTolGrad = 1e-12;
// The issue's bar: t_x against the ladder, relative to the ladder's largest
// element of each of h_MO, B_MO and c along that coordinate.
constexpr double kLadderTol = 1e-7;
// The ladder's displacement (bohr), with half of it beside it (Richardson),
// for h_MO and c. B_MO takes half that pair, 5e-4 and 2.5e-4: its tight
// JKFIT functions on C and O give it large fifth derivatives along a
// nuclear coordinate, and its Richardson remainder, falling 16x a halving
// (the connection-only ladder of B: 1.4e-5, 8.9e-7, 5.6e-8 absolute at 2e-3,
// 1e-3 and 5e-4, of max |dB| 12.4), was 7.2e-8 relative at 1e-3.
constexpr double kLadderStep = 1e-3;
// The degenerate-pair rule df_nevpt2_tangents applies (the gradient's).
constexpr double kGapFloor = 1e-6;
constexpr double kZeroSource = 1e-8;

const fb::json &fixture(std::string_view name) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto it = docs.find(name); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" /
                            std::format("{}.json", name));
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(std::string{name}, read.has_value() ? std::move(*read) : fb::json{})
      .first->second;
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *d) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(d, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
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

// A^-1/2 of a symmetric A near the identity (||1 - A|| < 1), by the coupled
// Newton-Schulz iteration: Y -> A^1/2, Z -> A^-1/2.
std::vector<double> inverse_sqrt(const std::vector<double> &a, std::size_t n) {
  std::vector<double> y = a;
  std::vector<double> z(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    z[i * (n + 1)] = 1.0;
  }
  for (std::int32_t it = 0; it < 30; ++it) {
    std::vector<double> t = host_gemm(n, n, n, z, false, y);
    double off = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
      off = std::max(off, std::abs(t[i] - (i % (n + 1) == 0 ? 1.0 : 0.0)));
      t[i] = -0.5 * t[i];
    }
    if (off < 1e-15) {
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
                                std::size_t cols, bool transpose_l) {
  for (std::size_t j = 0; j < cols; ++j) {
    double *x = b.data() + (n * j);
    if (!transpose_l) {
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

// The eigenvectors of the symmetric n x n @p a by cyclic Jacobi, columns in
// ascending order of eigenvalue.
std::vector<double> eigenvectors(std::vector<double> a, std::size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    v[i * (n + 1)] = 1.0;
  }
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        off = std::max(off, std::abs(a[p + (n * q)]));
      }
    }
    if (off < 1e-15) {
      break;
    }
    for (std::size_t q = 1; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        const double apq = a[p + (n * q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double theta = (a[q * (n + 1)] - a[p * (n + 1)]) / (2.0 * apq);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double cs = 1.0 / std::sqrt((t * t) + 1.0);
        const double sn = t * cs;
        // A <- J^T A J, V <- V J, J the rotation in the (p, q) plane.
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k + (n * p)];
          const double akq = a[k + (n * q)];
          a[k + (n * p)] = (cs * akp) - (sn * akq);
          a[k + (n * q)] = (sn * akp) + (cs * akq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[p + (n * k)];
          const double aqk = a[q + (n * k)];
          a[p + (n * k)] = (cs * apk) - (sn * aqk);
          a[q + (n * k)] = (sn * apk) + (cs * aqk);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double vkp = v[k + (n * p)];
          const double vkq = v[k + (n * q)];
          v[k + (n * p)] = (cs * vkp) - (sn * vkq);
          v[k + (n * q)] = (sn * vkp) + (cs * vkq);
        }
      }
    }
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, {}, [&](const std::size_t i) { return a[i * (n + 1)]; });
  std::vector<double> sorted(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(order[j] * n), n,
                sorted.begin() + static_cast<std::ptrdiff_t>(j * n));
  }
  return sorted;
}

// det of the k x k @p m (column-major), by elimination with partial pivots.
double determinant(std::vector<double> m, std::size_t k) {
  double det = 1.0;
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t pivot = c;
    for (std::size_t r = c + 1; r < k; ++r) {
      if (std::abs(m[r + (k * c)]) > std::abs(m[pivot + (k * c)])) {
        pivot = r;
      }
    }
    if (pivot != c) {
      det = -det;
      for (std::size_t j = 0; j < k; ++j) {
        std::swap(m[c + (k * j)], m[pivot + (k * j)]);
      }
    }
    det *= m[c + (k * c)];
    if (m[c + (k * c)] == 0.0) {
      return 0.0;
    }
    for (std::size_t r = c + 1; r < k; ++r) {
      const double factor = m[r + (k * c)] / m[c + (k * c)];
      for (std::size_t j = c; j < k; ++j) {
        m[r + (k * j)] -= factor * m[c + (k * j)];
      }
    }
  }
  return det;
}

// This tree's DF-CASSCF of a nevpt2_hessian fixture's start (@p start: its
// atoms, charge, basis, fitting set and active space), at @p conv_tol_grad,
// through the real registry in @p node; nothing on a failure (reported).
std::optional<fb::Calculation> run_casscf(const fb::json &start, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("df_nevpt2_tangents");
  std::string xyz =
      std::format("{}\n{}\n", start.at("atoms").size(), start.at("name").get<std::string>());
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string active;
  for (const std::int32_t a : start.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
      start.at("charge").get<std::int32_t>(), start.at("basis").at("name").get<std::string>(),
      kConvTolGrad, start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
      active, start.at("aux_basis").at("name").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return std::nullopt;
  }
  const fb::Status ran = calc->run(node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return std::nullopt;
  }
  return std::move(*calc);
}

// One case's reference: the registry's products, the one-root Solution of
// its canonical CASSCF, and lambda^x at it.
struct Reference {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::optional<fb::response::ZVectors> cp;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;

  [[nodiscard]] fb::DeviceResources &res() const { return node->device(0); }
  [[nodiscard]] const fb::Molecule &molecule() const {
    return calc->get<fb::MoleculeOutput>()->molecule;
  }
  [[nodiscard]] const fb::BasisSet &basis() const { return calc->get<fb::BasisSetOutput>()->basis; }
  [[nodiscard]] std::span<const fb::PrimitivePairClass> classes() const {
    return calc->get<fb::PrimitivePairClassesOutput>()->classes;
  }
  [[nodiscard]] const fb::DensityFitted &fit() const {
    return std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  }
  [[nodiscard]] const fb::SphericalMatrixAO &h() const {
    return calc->get<fb::CoreHamiltonianOutput>()->h;
  }

  using Consumer = std::function<fb::Status(std::int32_t, std::int32_t, const double *,
                                            const double *, const double *, const double *)>;
  [[nodiscard]] fb::Result<std::vector<fb::hessian::DegeneratePair>>
  tangents(const fb::mcscf::Solution &s, const fb::response::ZVectors &z, const bool pc,
           const Consumer &consume, const std::size_t max_bytes = 0) const {
    return fb::hessian::df_nevpt2_tangents(res(), molecule(), basis(), classes(), fit().aux, h(),
                                           fit().b, fit().l.data(), s, ncore, ncas, z, pc, consume,
                                           max_bytes);
  }
};

Reference reference(const fb::json &start) {
  Reference ref{.node = std::make_unique<fb::NodeResources>(
                    fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces}),
                .calc = std::nullopt,
                .solution = std::nullopt,
                .cp = std::nullopt};
  std::optional<fb::Calculation> calc = run_casscf(start, *ref.node);
  if (!calc.has_value()) {
    return ref;
  }
  ref.calc.emplace(std::move(*calc));
  fb::DeviceResources &res = ref.res();
  const auto &cas = ref.calc->get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (!cas.has_value()) {
    return ref;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  ref.ncore = cas->ncore;
  ref.ncas = cas->ncas;
  ref.nelecas = cas->nelecas;
  ref.solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = cas->rdm1,
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  upload(res, cas->vectors, ref.solution->roots.vectors.data());
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref.solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  auto cp = fb::hessian::df_casscf_cp(res, ref.molecule(), ref.basis(), ref.classes(),
                                      ref.fit().aux, ref.h(), ref.fit().b, ref.fit().l.data(),
                                      *ref.solution, ref.ncore, ref.ncas);
  EXPECT_TRUE(cp.has_value()) << cp.error().message;
  if (cp.has_value()) {
    ref.cp.emplace(std::move(*cp));
  }
  return ref;
}

// (h_MO, B_MO, c) of one point: B at Q + naux (p + nmo q).
struct Point {
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
};

// The ladder's points: at each displaced geometry the CASSCF re-solved from
// the reference moved by the connection, re-canonicalized and put in the
// tangent's gauge.
class Ladder {
public:
  Ladder(const Reference &ref, const fb::json &start) : ref_(ref), res_(ref.res()) {
    const fb::BasisSet &basis = ref.basis();
    nmo_ = basis.nao_sphe();
    naux_ = ref.fit().aux.nao_sphe();
    nn_ = nmo_ * nmo_;
    nocc_ = static_cast<std::size_t>(ref.ncore + ref.ncas);
    c0_ = download(res_, ref.solution->c.data(), nn_);
    ci0_ = download(res_, ref.solution->roots.vectors.data(),
                    ref.solution->roots.vectors.num_elements());
    l_m_ = download(res_, ref.fit().l.data(), naux_ * naux_);
    const auto strings = fb::ci::StringSet::create(ref.ncas, ref.nelecas / 2);
    EXPECT_TRUE(strings.has_value()) << strings.error().message;
    if (strings.has_value()) {
      strings_.assign(strings->strings().begin(), strings->strings().end());
    }
    auto stamp = fb::resolve_basis(start.at("basis").at("name").get<std::string>());
    EXPECT_TRUE(stamp) << stamp.error().message;
    auto aux_stamp = fb::resolve_aux_basis(start.at("aux_basis").at("name").get<std::string>());
    EXPECT_TRUE(aux_stamp) << aux_stamp.error().message;
    if (stamp && aux_stamp) {
      stamp_.emplace(std::move(*stamp));
      aux_stamp_.emplace(std::move(*aux_stamp));
    }
  }

  // The point at the reference's atoms with coordinate @p x moved by @p step.
  Point at(const std::size_t x, const double step) {
    std::vector<fb::Atom> atoms = ref_.molecule().atoms();
    fb::Vec<double, 3> &moved = atoms[x / 3].position;
    (x % 3 == 0 ? moved.x : x % 3 == 1 ? moved.y : moved.z) += step;
    const fb::Molecule molecule(res_, atoms);
    auto basis = fb::place(stamp_->primitive_normalized(), molecule);
    auto aux = fb::place(aux_stamp_->primitive_normalized(), molecule);
    EXPECT_TRUE(basis && aux);
    if (!basis || !aux) {
      return {};
    }
    auto classes = fb::primitive_pair_classes(res_, *basis);
    auto basis_u = fb::upload_spherical_transform_matrix(res_, *basis);
    auto aux_u = fb::upload_spherical_transform_matrix(res_, *aux);
    EXPECT_TRUE(classes && basis_u && aux_u);
    if (!classes || !basis_u || !aux_u) {
      return {};
    }
    fb::MatrixAOPair s_pair(res_, *basis, *basis_u);
    EXPECT_TRUE(fb::overlap(res_, *classes, s_pair.cartesian()));
    EXPECT_TRUE(s_pair.cartesian_to_spherical(res_));
    fb::MatrixAOPair h_pair(res_, *basis, *basis_u);
    EXPECT_TRUE(fb::core_hamiltonian(res_, *classes, molecule, h_pair.cartesian()));
    EXPECT_TRUE(h_pair.cartesian_to_spherical(res_));
    fb::MatrixAOPair metric(res_, *aux, std::move(*aux_u));
    EXPECT_TRUE(fb::metric(res_, *aux, metric));
    fb::Buffers::device<double> b(naux_ * nn_, res_.device());
    EXPECT_TRUE(fb::three_center(res_, *basis, *classes, *aux, b.data()));
    const std::vector<double> s = download(res_, s_pair.spherical().data(), nn_);
    const std::vector<double> h = download(res_, h_pair.spherical().data(), nn_);
    const std::vector<double> m = download(res_, metric.spherical().data(), naux_ * naux_);
    const std::vector<double> t = download(res_, b.data(), naux_ * nn_);
    EXPECT_TRUE(fb::density_fit(res_, static_cast<std::int32_t>(nmo_),
                                static_cast<std::int32_t>(naux_), metric.spherical().data(),
                                b.data()));

    // The reference moved by the connection, C0 (C0^T S C0)^-1/2, and the
    // solve from it, the reference's CI its guess.
    const std::vector<double> c_ref =
        host_gemm(nmo_, nmo_, nmo_, c0_, false,
                  inverse_sqrt(host_gemm(nmo_, nmo_, nmo_, c0_, true,
                                         host_gemm(nmo_, nmo_, nmo_, s, false, c0_)),
                               nmo_));
    fb::SphericalMatrixAO c_start(res_, *basis);
    upload(res_, c_ref, c_start.data());
    fb::SphericalMatrixAO h_ao(res_, *basis);
    upload(res_, h, h_ao.data());
    fb::mcscf::Options options;
    options.conv_tol = 1e-12;
    options.conv_tol_grad = kSolveTolGrad;
    options.density_fitting = fb::mcscf::DensityFitting::Full;
    auto sol = fb::mcscf::solve(
        res_, *basis, molecule, h_ao, nullptr, {}, {}, c_start, ref_.ncore, ref_.ncas, ref_.nelecas,
        1, {}, options, &b, static_cast<std::int32_t>(naux_),
        std::span<const double>{ref_.solution->roots.vectors.data(),
                                ref_.solution->roots.vectors.num_elements()});
    EXPECT_TRUE(sol.has_value()) << sol.error().message;
    if (!sol.has_value()) {
      return {};
    }
    worst_gradient_ = std::max(worst_gradient_, sol->gradient_norm);
    const std::vector<double> c_s = download(res_, sol->c.data(), nn_);
    const std::vector<double> ci_s =
        download(res_, sol->roots.vectors.data(), sol->roots.vectors.num_elements());

    // B's fitted index in the reference's symmetric gauge:
    // Z = L_M^-T (L_M^-1 M L_M^-T)^-1/2 at the reference's L_M.
    const std::vector<double> half = lower_solve(l_m_, naux_, m, naux_, false);
    const std::vector<double> gm = lower_solve(l_m_, naux_, transpose(half, naux_), naux_, false);
    const std::vector<double> z = lower_solve(l_m_, naux_, inverse_sqrt(gm, naux_), naux_, true);
    const auto mo = [&](const std::vector<double> &x) {
      const auto to_mo = [&](const std::vector<double> &ao) {
        return host_gemm(nmo_, nmo_, nmo_, x, true, host_gemm(nmo_, nmo_, nmo_, ao, false, x));
      };
      Point p{.h = to_mo(h), .b = std::vector<double>(naux_ * nn_), .c = {}};
      std::vector<double> k(naux_ * nn_);
      std::vector<double> tp(nn_);
      for (std::size_t q = 0; q < naux_; ++q) {
        for (std::size_t mn = 0; mn < nn_; ++mn) {
          tp[mn] = t[q + (naux_ * mn)];
        }
        std::ranges::copy(to_mo(tp), k.begin() + static_cast<std::ptrdiff_t>(q * nn_));
      }
      const std::vector<double> kz = host_gemm(nn_, naux_, naux_, k, false, z);
      for (std::size_t q = 0; q < naux_; ++q) {
        for (std::size_t pq = 0; pq < nn_; ++pq) {
          p.b[q + (naux_ * pq)] = kz[pq + (nn_ * q)];
        }
      }
      return p;
    };

    // Re-canonicalized: f's core and virtual blocks at the solve's orbitals
    // diagonalized, f = h + B (2 d + g) - sum_j (pj|jq) - 1/2 gamma (pt|uq).
    const Point solved = mo(c_s);
    const std::vector<double> f = fock(solved, sol->gamma);
    std::vector<double> r(nn_, 0.0);
    const auto ncore = static_cast<std::size_t>(ref_.ncore);
    for (const auto &[first, last] : {std::pair{std::size_t{0}, ncore}, std::pair{nocc_, nmo_}}) {
      const std::size_t w = last - first;
      std::vector<double> block(w * w);
      for (std::size_t j = 0; j < w; ++j) {
        for (std::size_t i = 0; i < w; ++i) {
          block[i + (w * j)] = f[(first + i) + (nmo_ * (first + j))];
        }
      }
      const std::vector<double> v = eigenvectors(block, w);
      for (std::size_t j = 0; j < w; ++j) {
        for (std::size_t i = 0; i < w; ++i) {
          r[(first + i) + (nmo_ * (first + j))] = v[i + (w * j)];
        }
      }
    }
    // The active orbitals turned by no active-active rotation from the
    // connection's frame: U = (M^T M)^-1/2 M^T, M = (C_ref^T S C_s) over the
    // active block, so that M U is symmetric.
    const std::vector<double> o_s =
        host_gemm(nmo_, nmo_, nmo_, c_ref, true, host_gemm(nmo_, nmo_, nmo_, s, false, c_s));
    const auto n = static_cast<std::size_t>(ref_.ncas);
    std::vector<double> m_act(n * n);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        m_act[i + (n * j)] = o_s[(ncore + i) + (nmo_ * (ncore + j))];
      }
    }
    const std::vector<double> u =
        host_gemm(n, n, n, inverse_sqrt(host_gemm(n, n, n, m_act, true, m_act), n), false,
                  transpose(m_act, n));
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        r[(ncore + i) + (nmo_ * (ncore + j))] = u[i + (n * j)];
      }
    }
    std::vector<double> c_new = host_gemm(nmo_, nmo_, nmo_, c_s, false, r);
    // Each core and virtual orbital's sign: along its reference one.
    const std::vector<double> o_new =
        host_gemm(nmo_, nmo_, nmo_, c_ref, true, host_gemm(nmo_, nmo_, nmo_, s, false, c_new));
    for (std::size_t p = 0; p < nmo_; ++p) {
      if ((p < ncore || p >= nocc_) && o_new[p * (nmo_ + 1)] < 0.0) {
        for (std::size_t mu = 0; mu < nmo_; ++mu) {
          c_new[mu + (nmo_ * p)] = -c_new[mu + (nmo_ * p)];
        }
      }
    }
    Point point = mo(c_new);
    // The CI vector over the turned active orbitals: c' = D^T c per spin,
    // D_JI = det U[J, I] over the strings' occupied orbitals.
    const std::size_t ns = strings_.size();
    const auto ne = static_cast<std::size_t>(ref_.nelecas / 2);
    std::vector<double> d(ns * ns);
    for (std::size_t i = 0; i < ns; ++i) {
      for (std::size_t j = 0; j < ns; ++j) {
        std::vector<std::size_t> rows;
        std::vector<std::size_t> cols;
        for (std::size_t t = 0; t < n; ++t) {
          if (((strings_[j] >> t) & 1U) != 0U) {
            rows.push_back(t);
          }
          if (((strings_[i] >> t) & 1U) != 0U) {
            cols.push_back(t);
          }
        }
        std::vector<double> sub(ne * ne);
        for (std::size_t b = 0; b < ne; ++b) {
          for (std::size_t a = 0; a < ne; ++a) {
            sub[a + (ne * b)] = u[rows[a] + (n * cols[b])];
          }
        }
        d[j + (ns * i)] = determinant(sub, ne);
      }
    }
    // c'[I, I'] = sum_JJ' D_JI D_J'I' c[J, J'] (alpha and beta share the
    // strings), its sign along the reference's.
    point.c.assign(ns * ns, 0.0);
    const std::vector<double> dc = host_gemm(ns, ns, ns, d, true, ci_s);
    const std::vector<double> dcd = host_gemm(ns, ns, ns, dc, false, d);
    point.c = dcd;
    double overlap = 0.0;
    for (std::size_t i = 0; i < point.c.size(); ++i) {
      overlap += point.c[i] * ci0_[i];
    }
    if (overlap < 0.0) {
      for (double &v : point.c) {
        v = -v;
      }
    }
    return point;
  }

  [[nodiscard]] double worst_gradient() const { return worst_gradient_; }

private:
  // f_pq = h_pq + sum_Q B_Q,pq (2 d_Q + g_Q) - sum_Q,j B_Q,pj B_Q,jq
  //        - 1/2 sum_Q,tu gamma_tu B_Q,pt B_Q,uq.
  [[nodiscard]] std::vector<double> fock(const Point &p, std::span<const double> gamma) const {
    const auto ncore = static_cast<std::size_t>(ref_.ncore);
    const auto n = static_cast<std::size_t>(ref_.ncas);
    const auto bq = [&](const std::size_t q, const std::size_t a, const std::size_t b) {
      return p.b[q + (naux_ * (a + (nmo_ * b)))];
    };
    std::vector<double> f = p.h;
    for (std::size_t q = 0; q < naux_; ++q) {
      double coulomb = 0.0;
      for (std::size_t j = 0; j < ncore; ++j) {
        coulomb += 2.0 * bq(q, j, j);
      }
      for (std::size_t t = 0; t < n; ++t) {
        for (std::size_t u = 0; u < n; ++u) {
          coulomb += gamma[t + (n * u)] * bq(q, ncore + t, ncore + u);
        }
      }
      for (std::size_t b = 0; b < nmo_; ++b) {
        for (std::size_t a = 0; a < nmo_; ++a) {
          double value = coulomb * bq(q, a, b);
          for (std::size_t j = 0; j < ncore; ++j) {
            value -= bq(q, a, j) * bq(q, j, b);
          }
          for (std::size_t t = 0; t < n; ++t) {
            for (std::size_t u = 0; u < n; ++u) {
              value -= 0.5 * gamma[t + (n * u)] * bq(q, a, ncore + t) * bq(q, ncore + u, b);
            }
          }
          f[a + (nmo_ * b)] += value;
        }
      }
    }
    return f;
  }

  const Reference &ref_;
  fb::DeviceResources &res_;
  std::optional<fb::BasisStamp> stamp_;
  std::optional<fb::BasisStamp> aux_stamp_;
  std::size_t nmo_ = 0;
  std::size_t naux_ = 0;
  std::size_t nn_ = 0;
  std::size_t nocc_ = 0;
  std::vector<double> c0_;
  std::vector<double> ci0_;
  std::vector<double> l_m_;
  std::vector<std::uint64_t> strings_;
  double worst_gradient_ = 0.0;
};

// The skew formaldehyde: t_x against the ladder along each of
// @p coordinates (every one handed over, two a chunk).
void expect_ladder(const std::vector<std::size_t> &coordinates) {
  const fb::json &start = fixture("h2co_skew_sto3g");
  ASSERT_FALSE(start.is_null());
  ASSERT_EQ(start.at("frozen_core").get<std::int32_t>(), 0);
  const Reference ref = reference(start);
  ASSERT_TRUE(ref.cp.has_value());
  fb::DeviceResources &res = ref.res();
  const std::size_t nmo = ref.basis().nao_sphe();
  const std::size_t naux = ref.fit().aux.nao_sphe();
  const std::size_t nn = nmo * nmo;
  const std::size_t ndet = ref.solution->roots.vectors.num_elements();
  const std::size_t n_x = 3 * ref.basis().natoms();

  std::vector<double> h_dot(n_x * nn);
  std::vector<double> b_dot(n_x * naux * nn);
  std::vector<double> c_dot(n_x * ndet);
  std::vector<std::pair<std::int32_t, std::int32_t>> chunks;
  const Reference::Consumer consume =
      [&](const std::int32_t x0, const std::int32_t count, const double *const h,
          const double *const b, const double *const c, const double * /*kappa*/) -> fb::Status {
    chunks.emplace_back(x0, count);
    const auto k = static_cast<std::size_t>(count);
    const auto from = static_cast<std::size_t>(x0);
    std::ranges::copy(download(res, h, k * nn),
                      h_dot.begin() + static_cast<std::ptrdiff_t>(from * nn));
    std::ranges::copy(download(res, b, k * naux * nn),
                      b_dot.begin() + static_cast<std::ptrdiff_t>(from * naux * nn));
    std::ranges::copy(download(res, c, k * ndet),
                      c_dot.begin() + static_cast<std::ptrdiff_t>(from * ndet));
    return {};
  };
  // A budget below one coordinate names what one needs; three times that
  // plus one more coordinate's W^x is two coordinates a chunk (a third goes
  // to W^x).
  const auto tiny = ref.tangents(*ref.solution, *ref.cp, false, consume, 3);
  ASSERT_FALSE(tiny.has_value());
  ASSERT_EQ(tiny.error().kind, fb::ErrorKind::Unsupported) << tiny.error().message;
  EXPECT_TRUE(chunks.empty());
  const std::string &message = tiny.error().message;
  const std::size_t needs = message.find(" needs ");
  ASSERT_NE(needs, std::string::npos) << message;
  const std::size_t need = std::stoull(message.substr(needs + 7));
  const auto start_time = std::chrono::steady_clock::now();
  const auto made = ref.tangents(*ref.solution, *ref.cp, false, consume,
                                 3 * (need + (sizeof(double) * naux * nn)));
  ASSERT_TRUE(made.has_value()) << made.error().message;
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  EXPECT_TRUE(made->empty());
  std::vector<std::pair<std::int32_t, std::int32_t>> want_chunks;
  for (std::size_t x0 = 0; x0 < n_x; x0 += 2) {
    want_chunks.emplace_back(static_cast<std::int32_t>(x0),
                             static_cast<std::int32_t>(std::min<std::size_t>(2, n_x - x0)));
  }
  EXPECT_EQ(chunks, want_chunks);

  Ladder ladder(ref, start);
  ASSERT_FALSE(::testing::Test::HasFailure());
  double worst = 0.0;
  std::string worst_at;
  for (const std::size_t x : coordinates) {
    SCOPED_TRACE(std::format("coordinate {}", x));
    const std::array<double, 6> steps{kLadderStep,      -kLadderStep,    kLadderStep / 2,
                                      -kLadderStep / 2, kLadderStep / 4, -kLadderStep / 4};
    std::array<Point, 6> f;
    for (std::size_t i = 0; i < steps.size(); ++i) {
      f[i] = ladder.at(x, steps[i]);
      ASSERT_FALSE(f[i].c.empty());
    }
    // Richardson over the pair from point @p first on, (4 D(t / 2) - D(t)) / 3
    // per element; the worst miss relative to the ladder's largest element.
    const auto miss = [&](std::vector<double> Point::*part, const double *analytic,
                          const std::size_t first) {
      const double t = steps[first];
      const std::size_t len = (f[0].*part).size();
      double diff = 0.0;
      double scale = 0.0;
      for (std::size_t i = 0; i < len; ++i) {
        const double d1 = ((f[first].*part)[i] - (f[first + 1].*part)[i]) / (2 * t);
        const double d2 = ((f[first + 2].*part)[i] - (f[first + 3].*part)[i]) / t;
        const double value = ((4 * d2) - d1) / 3;
        diff = std::max(diff, std::abs(analytic[i] - value));
        scale = std::max(scale, std::abs(value));
      }
      return std::pair{diff / scale, scale};
    };
    const auto [h_miss, h_scale] = miss(&Point::h, h_dot.data() + (x * nn), 0);
    const auto [b_miss, b_scale] = miss(&Point::b, b_dot.data() + (x * naux * nn), 2);
    const auto [c_miss, c_scale] = miss(&Point::c, c_dot.data() + (x * ndet), 0);
    EXPECT_LT(h_miss, kLadderTol) << "h_dot, of max " << h_scale;
    EXPECT_LT(b_miss, kLadderTol) << "B_dot, of max " << b_scale;
    EXPECT_LT(c_miss, kLadderTol) << "c_dot, of max " << c_scale;
    for (const auto &[value, what] :
         {std::pair{h_miss, "h"}, std::pair{b_miss, "B"}, std::pair{c_miss, "c"}}) {
      if (value > worst) {
        worst = value;
        worst_at = std::format("{} along coordinate {}", what, x);
      }
    }
    std::cout << std::format("[ ladder   ] x {:2}: h {:.1e} of {:.2e}, B {:.1e} of {:.2e}, c "
                             "{:.1e} of {:.2e} (relative)\n",
                             x, h_miss, h_scale, b_miss, b_scale, c_miss, c_scale);
  }
  std::cout << std::format("[ ladder   ] {} chunks, {:.2f} s; worst relative miss {:.1e} ({}); "
                           "displaced solves' worst ||g|| {:.1e}\n",
                           chunks.size(), elapsed, worst, worst_at, ladder.worst_gradient());
}

// Every coordinate.
TEST(HessianNevpt2Tangents, MatchesTheLadder) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  std::vector<std::size_t> every(3 * fixture("h2co_skew_sto3g").at("atoms").size());
  std::iota(every.begin(), every.end(), std::size_t{0});
  expect_ladder(every);
}

// The fast set's: the carbon's three coordinates.
TEST(HessianNevpt2Tangents, MatchesTheLadderOnOneAtom) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  expect_ladder({0, 1, 2});
}

// A C3v ammonia in 6-31G, its full valence CAS(8,7) (the 1s core): the
// virtuals past the valence space hold degenerate e pairs, and the
// displacements of E symmetry move f off-diagonal within each. SC refuses,
// naming a pair and a coordinate; PC records them.
fb::json c3v_ammonia() {
  // N-H 1.9 bohr, the hydrogens 0.7 bohr below N, 120 degrees apart.
  const double rho = std::sqrt((1.9 * 1.9) - (0.7 * 0.7));
  fb::json atoms = fb::json::array();
  atoms.push_back({{"symbol", "N"}, {"coord", {0.0, 0.0, 0.0}}});
  for (std::int32_t k = 0; k < 3; ++k) {
    const double phi = 2.0 * std::numbers::pi * k / 3.0;
    atoms.push_back({{"symbol", "H"}, {"coord", {rho * std::cos(phi), rho * std::sin(phi), -0.7}}});
  }
  return {{"name", "nh3_c3v_631g"},
          {"atoms", atoms},
          {"charge", 0},
          {"basis", {{"name", "6-31G"}}},
          {"aux_basis", {{"name", "cc-pVDZ-JKFIT"}}},
          {"ncas", 7},
          {"nelecas", 8},
          {"active", {1, 2, 3, 4, 5, 6, 7}}};
}

TEST(HessianNevpt2Tangents, DegeneratePairRule) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.cp.has_value());
  const auto ncore = static_cast<std::size_t>(ref.ncore);
  const std::size_t nocc = ncore + static_cast<std::size_t>(ref.ncas);
  const auto &eps = ref.calc->get<fb::CasscfOutput>()->mo_energies;
  // The pairs below the floor at the reference: the virtual e partners.
  std::vector<std::pair<std::int32_t, std::int32_t>> degenerate;
  for (const auto &[first, last] :
       {std::pair{std::size_t{0}, ncore}, std::pair{nocc, eps.size()}}) {
    for (std::size_t p = first; p < last; ++p) {
      for (std::size_t q = first; q < p; ++q) {
        if (std::abs(eps[p] - eps[q]) < kGapFloor) {
          degenerate.emplace_back(static_cast<std::int32_t>(p), static_cast<std::int32_t>(q));
          std::cout << std::format("[ c3v      ] orbitals {} and {}: {:.2e} Eh apart\n", p, q,
                                   std::abs(eps[p] - eps[q]));
        }
      }
    }
  }
  std::string all;
  for (const double e : eps) {
    all += std::format(" {:.10f}", e);
  }
  ASSERT_FALSE(degenerate.empty()) << "no degenerate pair:" << all;

  std::int32_t calls = 0;
  const Reference::Consumer count = [&](std::int32_t, std::int32_t, const double *, const double *,
                                        const double *, const double *) -> fb::Status {
    ++calls;
    return {};
  };
  const auto sc = ref.tangents(*ref.solution, *ref.cp, false, count);
  ASSERT_FALSE(sc.has_value()) << "SC took a degenerate pair's kappa_oo";
  EXPECT_EQ(sc.error().kind, fb::ErrorKind::Numerical) << sc.error().message;
  bool names_a_pair = false;
  for (const auto &[p, q] : degenerate) {
    names_a_pair = names_a_pair || sc.error().message.find(std::format("orbitals {} and {}", p,
                                                                       q)) != std::string::npos;
  }
  EXPECT_TRUE(names_a_pair) << sc.error().message;
  EXPECT_NE(sc.error().message.find("along coordinate "), std::string::npos) << sc.error().message;
  std::cout << std::format("[ c3v      ] SC: {}\n", sc.error().message);

  calls = 0;
  const auto pc = ref.tangents(*ref.solution, *ref.cp, true, count);
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  EXPECT_GE(calls, 1);
  ASSERT_FALSE(pc->empty()) << "PC recorded nothing";
  std::int32_t last = -1;
  for (const fb::hessian::DegeneratePair &d : *pc) {
    EXPECT_NE(std::ranges::find(degenerate, std::pair{d.p, d.q}), degenerate.end())
        << d.p << ", " << d.q;
    EXPECT_LT(std::abs(d.gap), kGapFloor);
    EXPECT_GT(std::abs(d.f), kZeroSource);
    EXPECT_GE(d.coordinate, last);
    EXPECT_LT(d.coordinate, static_cast<std::int32_t>(3 * ref.basis().natoms()));
    last = d.coordinate;
    std::cout << std::format("[ c3v      ] PC: coordinate {:2}, orbitals {} and {}, gap {:.1e}, "
                             "f {:+.3e}\n",
                             d.coordinate, d.p, d.q, d.gap, d.f);
  }
}

TEST(HessianNevpt2Tangents, Refusals) {
  FBF_REQUIRE_GPU();
  const Reference ref = reference(fixture("h2co_skew_sto3g"));
  ASSERT_TRUE(ref.cp.has_value());
  fb::DeviceResources &res = ref.res();
  const Reference::Consumer none = [](std::int32_t, std::int32_t, const double *, const double *,
                                      const double *, const double *) -> fb::Status { return {}; };
  const auto expect_refused = [](const auto &r, const char *what) {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_nevpt2_tangents"), std::string::npos) << what;
  };
  expect_refused(ref.tangents(*ref.solution, *ref.cp, false, Reference::Consumer{}), "no consumer");
  // lambda^x of one coordinate fewer.
  fb::response::ZVectors short_cp{
      .orbital = fb::Buffers::device<double>(
          ref.cp->orbital.num_elements() -
              (ref.cp->orbital.num_elements() / (3 * ref.basis().natoms())),
          res.device()),
      .ci = fb::Buffers::device<double>(ref.cp->ci.num_elements(), res.device()),
      .theta = {},
      .iterations = {},
      .residuals = {}};
  expect_refused(ref.tangents(*ref.solution, short_cp, false, none), "lambda^x short");
  // Two core orbitals turned into each other by 0.1 rad: no longer canonical.
  const std::size_t nmo = ref.basis().nao_sphe();
  std::vector<double> c = download(res, ref.solution->c.data(), nmo * nmo);
  for (std::size_t mu = 0; mu < nmo; ++mu) {
    const double a = c[mu + (nmo * 4)];
    const double b = c[mu + (nmo * 5)];
    c[mu + (nmo * 4)] = (std::cos(0.1) * a) - (std::sin(0.1) * b);
    c[mu + (nmo * 5)] = (std::sin(0.1) * a) + (std::cos(0.1) * b);
  }
  fb::mcscf::Solution turned{
      .c = fb::SphericalMatrixAO(res, ref.basis()),
      .roots = {.energies = ref.solution->roots.energies,
                .spin_squares = ref.solution->roots.spin_squares,
                .vectors = fb::Buffers::device<double>(ref.solution->roots.vectors.num_elements(),
                                                       res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = ref.solution->energy,
      .gradient_norm = ref.solution->gradient_norm,
      .gamma = ref.solution->gamma,
      .gamma2 = ref.solution->gamma2,
      .root_gammas = ref.solution->root_gammas,
      .iterations = {},
      .weights = ref.solution->weights};
  upload(res, c, turned.c.data());
  EXPECT_EQ(wwr::wwrMemcpyAsync(turned.roots.vectors.data(), ref.solution->roots.vectors.data(),
                                turned.roots.vectors.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  expect_refused(ref.tangents(turned, *ref.cp, false, none), "core not canonical");
}

} // namespace
