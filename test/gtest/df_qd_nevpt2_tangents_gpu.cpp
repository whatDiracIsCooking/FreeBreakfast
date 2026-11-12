// GPU cases for hessian::df_qd_nevpt2_tangents and hessian::df_sacasscf_cp,
// the QD-NEVPT2 parameter response over a state average and its total tangent
// t_x of (h_MO, B_MO, {c_r}), on this tree's own DF-SA-CASSCF through the
// registry. The conditions that define x_A are each checked on the tangent
// handed over, and given the validated derivative integrals they are
// sufficient (docs/nevpt2/nevpt2.tex, Eqs. qd-K, qd-theta, qd-gsa, koo):
//
// - CP-SA-CASSCF: df_sacasscf_cp's residuals, the unreduced equations' by one
//   more product, within 1e-9 in every column.
// - The rotations: the first-order change of <c_s|H|c_r> along t_x,
//   <c_dot_s|H|c_r> + <c_s|H|c_dot_r> + <c_s|H_dot|c_r>, H_dot the active
//   Hamiltonian's tangent rebuilt here on the host from h_dot and B_dot
//   (h1eff with the core's mean field, and (tu|vw)), each product by
//   ci::Sigma, zero to 1e-10 of its largest summand for every equal-weight
//   pair (Eq. qd-theta's), and the roots kept orthonormal. An unequal pair's
//   is the CP solve's rotation row, held to the residual's 1e-9 absolute.
// - kappa_oo: the change of f^SA's core-core and virtual-virtual
//   off-diagonal along t_x (nevpt2::qd_builders_tangent at the tangent
//   handed over) zero to 1e-10 of its larger summand, f^(x) and kappa_oo's
//   part (kappa^T f + f kappa).
//
// On data/qd_nevpt2/'s h2co_skew_ccpvdz_sa3 (C1, three roots at equal
// weights) and h2co_skew_ccpvdz_sa3_weighted (0.4, 0.4, 0.2: roots 0 and 1
// rotate by Eq. qd-theta, the pairs with root 2 in the solve). Then a C3v
// ammonia's degenerate virtual e pairs (SC refused, PC recorded), and the
// refusals: two roots within mcscf::kRootGapFloor, a mixing rate past 10 per
// bohr, and the inputs.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.response.zvector;
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars: the CP solve's true residual, absolute, and each
// defining condition relative to its largest summand.
constexpr double kResidualTol = 1e-9;
constexpr double kConditionTol = 1e-10;
// c_dot_s . c_r + c_s . c_dot_r: zero by construction (xi off every root,
// the rotations antisymmetric), to rounding.
constexpr double kOrthonormalTol = 1e-12;
// The reference SA-CASSCF's ||g|| and |Delta E|. Eq. qd-theta reads
// H c_r = E_r c_r (xi adds nothing to <c_s|H|c_r> only at eigenvectors), and
// the CI's residual follows ||g|| (mcscf::ci_tolerance_for): at 1e-8 its
// 1e-10 against xi missed the weighted case's 1e-10 bar by 1.06x.
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-12;
// df_qd_nevpt2_tangents' degenerate-pair rule and rotation bound.
constexpr double kGapFloor = 1e-6;
constexpr double kZeroSource = 1e-8;
constexpr double kMaxRotation = 10.0;

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

double max_abs(std::span<const double> v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

// A state average's start: atoms (bohr), basis, fitting set, active space,
// weights, and an explicit active list or none.
struct Start {
  std::string name;
  std::vector<std::pair<std::string, std::array<double, 3>>> atoms;
  std::string basis;
  std::string aux_basis;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<double> weights;
  std::vector<std::int32_t> active;
};

Start qd_case(const std::string &name) {
  auto read =
      fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / "qd_nevpt2.json");
  EXPECT_TRUE(read.has_value()) << read.error().message;
  if (!read.has_value()) {
    return {};
  }
  for (const fb::json &entry : read->at("cases")) {
    if (entry.at("name").get<std::string>() != name) {
      continue;
    }
    Start s{.name = name,
            .atoms = {},
            .basis = entry.at("basis").get<std::string>(),
            .aux_basis = entry.at("aux_basis").at("name").get<std::string>(),
            .ncas = entry.at("ncas").get<std::int32_t>(),
            .nelecas = entry.at("nelecas").get<std::int32_t>(),
            .weights = entry.at("weights").get<std::vector<double>>(),
            .active = {}};
    for (const fb::json &atom : entry.at("atoms")) {
      const auto coord = atom.at(1).get<std::vector<double>>();
      s.atoms.push_back({atom.at(0).get<std::string>(), {coord[0], coord[1], coord[2]}});
    }
    return s;
  }
  ADD_FAILURE() << "no case " << name;
  return {};
}

// A C3v ammonia in 6-31G, SA-2 CAS(8,7) over its full valence (the 1s core),
// both roots A1: the virtual e pairs past the valence space stay degenerate,
// and the displacements of E symmetry move f^SA off-diagonal within each.
Start c3v_ammonia() {
  // N-H 1.9 bohr, the hydrogens 0.7 bohr below N, 120 degrees apart.
  const double rho = std::sqrt((1.9 * 1.9) - (0.7 * 0.7));
  Start s{.name = "nh3_c3v_631g_sa2",
          .atoms = {{"N", {0.0, 0.0, 0.0}}},
          .basis = "6-31G",
          .aux_basis = "cc-pVDZ-JKFIT",
          .ncas = 7,
          .nelecas = 8,
          .weights = {0.5, 0.5},
          .active = {1, 2, 3, 4, 5, 6, 7}};
  for (std::int32_t k = 0; k < 3; ++k) {
    const double phi = 2.0 * std::numbers::pi * k / 3.0;
    s.atoms.push_back({"H", {rho * std::cos(phi), rho * std::sin(phi), -0.7}});
  }
  return s;
}

// One case's reference: the registry's products, the SA Solution of its
// canonical SA-CASSCF, and lambda^x at it.
struct Reference {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::optional<fb::hessian::SaCasscfCp> cp;
  std::vector<double> weights;
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
  [[nodiscard]] fb::Result<fb::hessian::QdNevpt2Tangents> tangents(const fb::mcscf::Solution &s,
                                                                   const fb::response::ZVectors &z,
                                                                   const bool pc,
                                                                   const Consumer &consume) const {
    return fb::hessian::df_qd_nevpt2_tangents(res(), molecule(), basis(), classes(), fit().aux, h(),
                                              fit().b, fit().l.data(), s, weights, ncore, ncas, z,
                                              pc, consume);
  }

  // A copy of the solution with other root energies (the vectors and orbitals
  // the same).
  [[nodiscard]] fb::mcscf::Solution with_energies(std::vector<double> energies) const {
    fb::DeviceResources &r = res();
    fb::mcscf::Solution s{.c = fb::SphericalMatrixAO(r, basis()),
                          .roots = {.energies = std::move(energies),
                                    .spin_squares = solution->roots.spin_squares,
                                    .vectors = fb::Buffers::device<double>(
                                        solution->roots.vectors.num_elements(), r.device()),
                                    .davidson = {},
                                    .fp32_iterations = 0},
                          .energy = solution->energy,
                          .gradient_norm = solution->gradient_norm,
                          .gamma = solution->gamma,
                          .gamma2 = solution->gamma2,
                          .root_gammas = solution->root_gammas,
                          .iterations = {},
                          .weights = solution->weights};
    EXPECT_EQ(wwr::wwrMemcpyAsync(s.c.data(), solution->c.data(),
                                  solution->c.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrMemcpyAsync(s.roots.vectors.data(), solution->roots.vectors.data(),
                                  s.roots.vectors.num_elements() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, r.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(r.sync().has_value());
    return s;
  }
};

std::unique_ptr<Reference> make_reference(const Start &start) {
  auto ref = std::make_unique<Reference>();
  ref->node = std::make_unique<fb::NodeResources>(
      fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces});
  ref->weights = start.weights;
  const fb::test::ScratchDir dir("df_qd_nevpt2_tangents");
  std::string xyz = std::format("{}\n{}\n", start.atoms.size(), start.name);
  for (const auto &[symbol, coord] : start.atoms) {
    xyz += std::format(
        "{} {:.17g} {:.17g} {:.17g}\n", symbol, coord[0] * fb::kAngstromPerBohr<double>,
        coord[1] * fb::kAngstromPerBohr<double>, coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : start.weights) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  std::string active;
  for (const std::int32_t a : start.active) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol = {:g}\n"
      "conv_tol_grad = {:g}\n{}",
      start.basis, start.aux_basis, start.ncas, start.nelecas, start.weights.size(), weights,
      kConvTol, kConvTolGrad, active.empty() ? "" : std::format("active = [{}]\n", active));
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return ref;
  }
  const fb::Status ran = calc->run(*ref->node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return ref;
  }
  ref->calc.emplace(std::move(*calc));
  fb::DeviceResources &res = ref->res();
  const auto &cas = ref->calc->get<fb::CasscfOutput>();
  EXPECT_TRUE(cas.has_value());
  if (!cas.has_value()) {
    return ref;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  ref->ncore = cas->ncore;
  ref->ncas = cas->ncas;
  ref->nelecas = cas->nelecas;
  // The Solution as the [hessian] step builds it: gamma the weighted average
  // of the roots' 1-RDMs, rdm2 already the average.
  const auto n2 = static_cast<std::size_t>(cas->ncas) * static_cast<std::size_t>(cas->ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas->weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas->weights[r] * cas->rdm1[(r * n2) + i];
    }
  }
  ref->solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, ref->basis()),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights});
  upload(res, cas->vectors, ref->solution->roots.vectors.data());
  EXPECT_EQ(wwr::wwrMemcpyAsync(ref->solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  const auto t0 = std::chrono::steady_clock::now();
  auto cp = fb::hessian::df_sacasscf_cp(res, ref->molecule(), ref->basis(), ref->classes(),
                                        ref->fit().aux, ref->h(), ref->fit().b, ref->fit().l.data(),
                                        *ref->solution, ref->weights, ref->ncore, ref->ncas);
  EXPECT_TRUE(cp.has_value()) << cp.error().message;
  if (cp.has_value()) {
    std::cout << std::format(
        "[ qd tang  ] {}: E {}, ||g|| {:.1e}, CI residual {:.1e}; CP-SA-CASSCF {:.2f} s, worst "
        "residual {:.2e}\n",
        start.name, cas->energies, cas->gradient_norm, cas->ci_residual_norm,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
        std::ranges::max(cp->residuals));
    ref->cp.emplace(std::move(*cp));
  }
  return ref;
}

class HessianQdNevpt2Tangents : public ::testing::Test {
protected:
  // One reference per case, kept for the suite: the refusals reuse the
  // weighted case's.
  static const Reference &reference(const Start &start) {
    auto &slot = references()[start.name];
    if (!slot) {
      slot = make_reference(start);
    }
    return *slot;
  }
  static void TearDownTestSuite() { references().clear(); }

private:
  static std::map<std::string, std::unique_ptr<Reference>> &references() {
    static std::map<std::string, std::unique_ptr<Reference>> kept;
    return kept;
  }
};

// B_MO's slabs over the occupied orbitals on the host: (Q, p, q) at
// Q + naux (p + nocc q), p and q below nocc.
std::vector<double> occupied_b(const Reference &ref, const std::size_t nocc) {
  fb::DeviceResources &res = ref.res();
  const std::size_t nao = ref.basis().nao_sphe();
  const std::size_t naux = ref.fit().aux.nao_sphe();
  const std::vector<double> b = download(res, ref.fit().b.data(), naux * nao * nao);
  const std::vector<double> c = download(res, ref.solution->c.data(), nao * nao);
  std::vector<double> half(naux * nao * nocc, 0.0); // (Q, mu, q)
  for (std::size_t q = 0; q < nocc; ++q) {
    for (std::size_t nu = 0; nu < nao; ++nu) {
      const double cnq = c[nu + (nao * q)];
      for (std::size_t mu = 0; mu < nao; ++mu) {
        const double *const slab = b.data() + (naux * (mu + (nao * nu)));
        double *const out = half.data() + (naux * (mu + (nao * q)));
        for (std::size_t l = 0; l < naux; ++l) {
          out[l] += slab[l] * cnq;
        }
      }
    }
  }
  std::vector<double> occ(naux * nocc * nocc, 0.0);
  for (std::size_t q = 0; q < nocc; ++q) {
    for (std::size_t p = 0; p < nocc; ++p) {
      double *const out = occ.data() + (naux * (p + (nocc * q)));
      for (std::size_t mu = 0; mu < nao; ++mu) {
        const double cmp = c[mu + (nao * p)];
        const double *const in = half.data() + (naux * (mu + (nao * q)));
        for (std::size_t l = 0; l < naux; ++l) {
          out[l] += cmp * in[l];
        }
      }
    }
  }
  return occ;
}

// The active Hamiltonian (h1eff with the core's mean field, row-major, and
// (tu|vw), row-major) from @p h (nmo x nmo) and, for the tangent, its first
// order: @p h_dot with @p b_dot (naux x nmo^2) against @p b_occ.
std::pair<std::vector<double>, std::vector<double>>
active_hamiltonian(const std::size_t naux, const std::size_t nmo, const std::size_t nc,
                   const std::size_t na, const std::vector<double> &b_occ,
                   const std::vector<double> &h, const double *b_dot) {
  const std::size_t nocc = nc + na;
  // (pq|rs) over occupied indices; with b_dot, its first order.
  const auto eri = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    const double *const pq = b_occ.data() + (naux * (p + (nocc * q)));
    const double *const rs = b_occ.data() + (naux * (r + (nocc * s)));
    double sum = 0.0;
    if (b_dot == nullptr) {
      for (std::size_t l = 0; l < naux; ++l) {
        sum += pq[l] * rs[l];
      }
      return sum;
    }
    const double *const dpq = b_dot + (naux * (p + (nmo * q)));
    const double *const drs = b_dot + (naux * (r + (nmo * s)));
    for (std::size_t l = 0; l < naux; ++l) {
      sum += (dpq[l] * rs[l]) + (pq[l] * drs[l]);
    }
    return sum;
  };
  std::vector<double> h_eff(na * na);
  std::vector<double> g(na * na * na * na);
  for (std::size_t t = 0; t < na; ++t) {
    for (std::size_t u = 0; u < na; ++u) {
      double value = h[(nc + t) + (nmo * (nc + u))];
      for (std::size_t i = 0; i < nc; ++i) {
        value += (2.0 * eri(i, i, nc + t, nc + u)) - eri(i, nc + t, i, nc + u);
      }
      h_eff[(t * na) + u] = value;
      for (std::size_t v = 0; v < na; ++v) {
        for (std::size_t w = 0; w < na; ++w) {
          g[(((((t * na) + u) * na) + v) * na) + w] = eri(nc + t, nc + u, nc + v, nc + w);
        }
      }
    }
  }
  return {std::move(h_eff), std::move(g)};
}

// Every defining condition on @p name's state average.
void expect_conditions(const Reference &ref, const std::string &name) {
  ASSERT_TRUE(ref.cp.has_value());
  fb::DeviceResources &res = ref.res();
  const std::size_t nmo = ref.basis().nao_sphe();
  const std::size_t naux = ref.fit().aux.nao_sphe();
  const std::size_t nn = nmo * nmo;
  const auto nc = static_cast<std::size_t>(ref.ncore);
  const auto na = static_cast<std::size_t>(ref.ncas);
  const std::size_t nocc = nc + na;
  const std::size_t nv = nmo - nocc;
  const std::size_t nroots = ref.weights.size();
  const std::size_t n_x = 3 * ref.basis().natoms();
  const std::vector<bool> rotates = fb::response::rotating_pairs(ref.weights);

  // CP-SA-CASSCF: the unreduced equations' residual per column.
  ASSERT_EQ(ref.cp->residuals.size(), n_x);
  for (std::size_t x = 0; x < n_x; ++x) {
    EXPECT_LE(ref.cp->residuals[x], kResidualTol) << name << " coordinate " << x;
  }

  // The active Hamiltonian at the point, and a sigma over it (no constant:
  // a shift of H moves <c_s|H|c_r> by its multiple of c_s . c_r, whose
  // tangent is checked to vanish beside it).
  const std::vector<double> b_occ = occupied_b(ref, nocc);
  std::vector<double> h_mo(nn, 0.0);
  {
    const std::vector<double> h_ao = download(res, ref.h().data(), nn);
    const std::vector<double> c = download(res, ref.solution->c.data(), nn);
    for (std::size_t q = 0; q < nocc; ++q) {
      for (std::size_t p = 0; p < nocc; ++p) {
        double sum = 0.0;
        for (std::size_t nu = 0; nu < nmo; ++nu) {
          for (std::size_t mu = 0; mu < nmo; ++mu) {
            sum += c[mu + (nmo * p)] * h_ao[mu + (nmo * nu)] * c[nu + (nmo * q)];
          }
        }
        h_mo[p + (nmo * q)] = sum;
      }
    }
  }
  const std::int32_t two_s = ref.molecule().multiplicity() - 1;
  const std::int32_t nbeta = (ref.nelecas - two_s) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ref.ncas, ref.nelecas - nbeta);
  auto beta = fb::ci::DeviceStringSet::create(res, ref.ncas, nbeta);
  ASSERT_TRUE(alpha && beta);
  auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
  auto sigma_dot = fb::ci::Sigma::create(res, *alpha, *beta);
  ASSERT_TRUE(sigma && sigma_dot);
  {
    const auto [h_eff, g] = active_hamiltonian(naux, nmo, nc, na, b_occ, h_mo, nullptr);
    ASSERT_TRUE(sigma->set_integrals(res, 0.0, h_eff, g).has_value());
  }
  const std::size_t ndet = ref.solution->roots.vectors.num_elements() / nroots;
  const std::vector<double> c_host =
      download(res, ref.solution->roots.vectors.data(), nroots * ndet);
  fb::Buffers::device<double> hc(nroots * ndet, res.device());
  ASSERT_TRUE(sigma
                  ->apply(res, static_cast<std::int32_t>(nroots),
                          ref.solution->roots.vectors.data(), hc.data())
                  .has_value());
  const std::vector<double> hc_host = download(res, hc.data(), nroots * ndet);
  const auto dot = [&](const double *a, const double *b) {
    return std::inner_product(a, a + ndet, b, 0.0);
  };

  double worst_rotation = 0.0;
  double worst_unequal = 0.0;
  double worst_orthonormal = 0.0;
  double worst_kappa_oo = 0.0;
  double largest_equal_coupling = 0.0;
  std::int32_t seen = 0;
  const Reference::Consumer check = [&](const std::int32_t x0, const std::int32_t count,
                                        const double *const h_dot, const double *const b_dot,
                                        const double *const c_dot,
                                        const double *const kappa) -> fb::Status {
    EXPECT_EQ(x0, seen);
    seen += count;
    const auto k = static_cast<std::size_t>(count);
    const std::vector<double> h_dots = download(res, h_dot, k * nn);
    const std::vector<double> b_dots = download(res, b_dot, k * naux * nn);
    const std::vector<double> c_dots = download(res, c_dot, k * nroots * ndet);
    const std::vector<double> kappas = download(res, kappa, k * nn);
    fb::Buffers::device<double> hc_dot(k * nroots * ndet, res.device());
    EXPECT_TRUE(
        sigma->apply(res, static_cast<std::int32_t>(k * nroots), c_dot, hc_dot.data()).has_value());
    const std::vector<double> hc_dots = download(res, hc_dot.data(), k * nroots * ndet);

    // The rotations' condition, every pair.
    for (std::size_t j = 0; j < k; ++j) {
      const std::size_t x = static_cast<std::size_t>(x0) + j;
      const std::vector<double> h_j(h_dots.begin() + static_cast<std::ptrdiff_t>(j * nn),
                                    h_dots.begin() + static_cast<std::ptrdiff_t>((j + 1) * nn));
      const auto [h_eff_dot, g_dot] =
          active_hamiltonian(naux, nmo, nc, na, b_occ, h_j, b_dots.data() + (j * naux * nn));
      EXPECT_TRUE(sigma_dot->set_integrals(res, 0.0, h_eff_dot, g_dot).has_value());
      fb::Buffers::device<double> h_dot_c(nroots * ndet, res.device());
      EXPECT_TRUE(sigma_dot
                      ->apply(res, static_cast<std::int32_t>(nroots),
                              ref.solution->roots.vectors.data(), h_dot_c.data())
                      .has_value());
      const std::vector<double> h_dot_c_host = download(res, h_dot_c.data(), nroots * ndet);
      const double *const cj = c_dots.data() + (j * nroots * ndet);
      const double *const hcj = hc_dots.data() + (j * nroots * ndet);
      std::size_t at = 0;
      for (std::size_t r = 0; r < nroots; ++r) {
        worst_orthonormal =
            std::max(worst_orthonormal, std::abs(dot(cj + (r * ndet), c_host.data() + (r * ndet))));
        for (std::size_t s = r + 1; s < nroots; ++s, ++at) {
          // <c_dot_s|H|c_r> + <c_s|H|c_dot_r> + <c_s|H_dot|c_r>.
          const std::array<double, 3> terms{
              dot(cj + (s * ndet), hc_host.data() + (r * ndet)),
              dot(hc_host.data() + (s * ndet), cj + (r * ndet)),
              dot(c_host.data() + (s * ndet), h_dot_c_host.data() + (r * ndet))};
          const double total = terms[0] + terms[1] + terms[2];
          const double miss = std::abs(total) / max_abs(terms);
          if (rotates[at]) {
            // The CP solve's row, held by its residual to the solve's
            // tolerance, not to this bar.
            worst_unequal = std::max(worst_unequal, std::abs(total));
          } else {
            EXPECT_LE(miss, kConditionTol)
                << name << " coordinate " << x << " roots " << r << ", " << s << ": "
                << std::format("{:+.6e} {:+.6e} {:+.6e}", terms[0], terms[1], terms[2]);
            worst_rotation = std::max(worst_rotation, miss);
            largest_equal_coupling = std::max(largest_equal_coupling, std::abs(terms[2]));
          }
          worst_orthonormal = std::max(worst_orthonormal,
                                       std::abs(dot(cj + (s * ndet), c_host.data() + (r * ndet)) +
                                                dot(c_host.data() + (s * ndet), cj + (r * ndet))));
        }
      }
    }

    // kappa_oo's condition: f^SA's off-diagonal change along t_x.
    auto qd = fb::nevpt2::qd_builders_tangent(
        res, *alpha, *beta, static_cast<std::int32_t>(naux), static_cast<std::int32_t>(nmo),
        static_cast<std::int32_t>(nmo), ref.fit().b.data(), ref.solution->c.data(), ref.h().data(),
        ref.ncore, ref.ncas, ref.weights, ref.solution->roots.vectors.data(), count, h_dot, b_dot,
        c_dot);
    EXPECT_TRUE(qd.has_value()) << qd.error().message;
    if (!qd.has_value()) {
      return {};
    }
    for (std::size_t j = 0; j < k; ++j) {
      const std::size_t x = static_cast<std::size_t>(x0) + j;
      for (const auto &[value, tangent, first, width] :
           {std::tuple{&qd->value.shared.f_core, &qd->tangents[j].shared.f_core, std::size_t{0},
                       nc},
            std::tuple{&qd->value.shared.f_virt, &qd->tangents[j].shared.f_virt, nocc, nv}}) {
        const std::vector<double> f = download(res, value->data(), width * width);
        const std::vector<double> t = download(res, tangent->data(), width * width);
        const double *const kj = kappas.data() + (j * nn);
        double residual = 0.0;
        double largest = 0.0;
        for (std::size_t q = 0; q < width; ++q) {
          for (std::size_t p = 0; p < width; ++p) {
            if (p == q) {
              continue;
            }
            // (kappa^T f + f kappa)_pq over the block.
            double part = 0.0;
            for (std::size_t m = 0; m < width; ++m) {
              part += (kj[(first + m) + (nmo * (first + p))] * f[m + (width * q)]) +
                      (f[p + (width * m)] * kj[(first + m) + (nmo * (first + q))]);
            }
            const double total = t[p + (width * q)];
            residual = std::max(residual, std::abs(total));
            largest = std::max({largest, std::abs(total - part), std::abs(part)});
          }
        }
        if (largest > 0.0) {
          EXPECT_LE(residual, kConditionTol * largest)
              << name << " coordinate " << x << " block at " << first;
          worst_kappa_oo = std::max(worst_kappa_oo, residual / largest);
        }
      }
    }
    return {};
  };
  const auto t0 = std::chrono::steady_clock::now();
  const auto got = ref.tangents(*ref.solution, ref.cp->cp, false, check);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  EXPECT_EQ(seen, static_cast<std::int32_t>(n_x));
  EXPECT_TRUE(got->recorded.empty());
  ASSERT_EQ(got->theta.size(), n_x * nroots * (nroots - 1) / 2);
  // The unequal pairs' rotations are the CP solve's as they come.
  for (std::size_t i = 0; i < got->theta.size(); ++i) {
    if (rotates[i % rotates.size()]) {
      EXPECT_EQ(got->theta[i], ref.cp->cp.theta[i]) << i;
    }
  }
  EXPECT_EQ(got->max_theta, max_abs(got->theta));
  EXPECT_GT(got->max_kappa_oo, 0.0);
  EXPECT_LE(worst_orthonormal, kOrthonormalTol) << name;
  EXPECT_LE(worst_unequal, kResidualTol) << name;
  EXPECT_GT(largest_equal_coupling, 0.0) << name;
  std::cout << std::format(
      "[ qd tang  ] {}: tangents {:.2f} s; worst equal-weight rotation condition {:.2e} "
      "(unequal pairs' rows {:.2e} absolute), kappa_oo condition {:.2e}, orthonormality {:.2e}; "
      "max |theta| {:.3e}, max |kappa_oo| {:.3e}, largest equal-weight coupling {:.3e}\n",
      name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
      worst_rotation, worst_unequal, worst_kappa_oo, worst_orthonormal, got->max_theta,
      got->max_kappa_oo, largest_equal_coupling);
}

TEST_F(HessianQdNevpt2Tangents, DefiningConditionsEqualWeights) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Start start = qd_case("h2co_skew_ccpvdz_sa3");
  expect_conditions(reference(start), start.name);
}

TEST_F(HessianQdNevpt2Tangents, DefiningConditionsUnequalWeights) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  const Start start = qd_case("h2co_skew_ccpvdz_sa3_weighted");
  expect_conditions(reference(start), start.name);
}

TEST_F(HessianQdNevpt2Tangents, DegeneratePairRule) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const Reference &ref = reference(c3v_ammonia());
  ASSERT_TRUE(ref.cp.has_value());
  const auto ncore = static_cast<std::size_t>(ref.ncore);
  const std::size_t nocc = ncore + static_cast<std::size_t>(ref.ncas);
  const auto &eps = ref.calc->get<fb::CasscfOutput>()->mo_energies;
  // The pairs below the floor at the reference: the e partners.
  std::vector<std::pair<std::int32_t, std::int32_t>> degenerate;
  for (const auto &[first, last] :
       {std::pair{std::size_t{0}, ncore}, std::pair{nocc, eps.size()}}) {
    for (std::size_t p = first; p < last; ++p) {
      for (std::size_t q = first; q < p; ++q) {
        if (std::abs(eps[p] - eps[q]) < kGapFloor) {
          degenerate.emplace_back(static_cast<std::int32_t>(p), static_cast<std::int32_t>(q));
        }
      }
    }
  }
  ASSERT_FALSE(degenerate.empty()) << "no degenerate pair";
  const Reference::Consumer none = [](std::int32_t, std::int32_t, const double *, const double *,
                                      const double *, const double *) -> fb::Status { return {}; };
  const auto sc = ref.tangents(*ref.solution, ref.cp->cp, false, none);
  ASSERT_FALSE(sc.has_value()) << "SC took a degenerate pair's kappa_oo";
  EXPECT_EQ(sc.error().kind, fb::ErrorKind::Numerical) << sc.error().message;
  bool names_a_pair = false;
  for (const auto &[p, q] : degenerate) {
    names_a_pair = names_a_pair || sc.error().message.find(std::format("orbitals {} and {}", p,
                                                                       q)) != std::string::npos;
  }
  EXPECT_TRUE(names_a_pair) << sc.error().message;
  EXPECT_NE(sc.error().message.find("along coordinate "), std::string::npos) << sc.error().message;
  std::cout << std::format("[ qd tang  ] nh3 SC: {}\n", sc.error().message);

  const auto pc = ref.tangents(*ref.solution, ref.cp->cp, true, none);
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  // Every pair below the floor along every coordinate (#1277: df_qd_pc_nevpt2
  // takes their off-diagonal f), some of them split past round-off.
  ASSERT_EQ(pc->recorded.size(), degenerate.size() * 3 * ref.basis().natoms()) << "PC's records";
  std::int32_t last = -1;
  std::size_t split = 0;
  for (const fb::hessian::DegeneratePair &d : pc->recorded) {
    EXPECT_NE(std::ranges::find(degenerate, std::pair{d.p, d.q}), degenerate.end())
        << d.p << ", " << d.q;
    EXPECT_LT(std::abs(d.gap), kGapFloor);
    EXPECT_GE(d.coordinate, last);
    EXPECT_LT(d.coordinate, static_cast<std::int32_t>(3 * ref.basis().natoms()));
    last = d.coordinate;
    split += std::abs(d.f) > kZeroSource ? 1 : 0;
  }
  EXPECT_GT(split, 0U) << "no coordinate splits a pair";
  std::cout << std::format("[ qd tang  ] nh3 PC: {} records, {} split past round-off, the first "
                           "coordinate {} orbitals {} and {}\n",
                           pc->recorded.size(), split, pc->recorded.front().coordinate,
                           pc->recorded.front().p, pc->recorded.front().q);
}

TEST_F(HessianQdNevpt2Tangents, Refusals) {
  FBF_REQUIRE_GPU();
  const Reference &ref = reference(qd_case("h2co_skew_ccpvdz_sa3_weighted"));
  ASSERT_TRUE(ref.cp.has_value());
  fb::DeviceResources &res = ref.res();
  const Reference::Consumer none = [](std::int32_t, std::int32_t, const double *, const double *,
                                      const double *, const double *) -> fb::Status { return {}; };
  const auto expect_refused = [](const auto &r, const fb::ErrorKind kind,
                                 const std::string_view what,
                                 const std::string_view names = "hessian::df_qd_nevpt2_tangents") {
    ASSERT_FALSE(r.has_value()) << what;
    EXPECT_EQ(r.error().kind, kind) << what << ": " << r.error().message;
    EXPECT_NE(r.error().message.find("hessian::df_qd_nevpt2_tangents"), std::string::npos) << what;
    EXPECT_NE(r.error().message.find(names), std::string::npos)
        << what << ": " << r.error().message;
  };
  const auto config = fb::ErrorKind::InvalidConfig;
  expect_refused(ref.tangents(*ref.solution, ref.cp->cp, false, Reference::Consumer{}), config,
                 "no consumer");
  // lambda^x of one coordinate fewer.
  const std::size_t n_x = 3 * ref.basis().natoms();
  const fb::response::ZVectors &cp = ref.cp->cp;
  fb::response::ZVectors short_cp{
      .orbital =
          fb::Buffers::device<double>(cp.orbital.num_elements() / n_x * (n_x - 1), res.device()),
      .ci = fb::Buffers::device<double>(cp.ci.num_elements() / n_x * (n_x - 1), res.device()),
      .theta = std::vector<double>(cp.theta.size() / n_x * (n_x - 1), 0.0),
      .iterations = {},
      .residuals = {}};
  expect_refused(ref.tangents(*ref.solution, short_cp, false, none), config, "lambda^x short");

  // Roots 0 and 1 (equal weights) within mcscf::kRootGapFloor.
  std::vector<double> e = ref.solution->roots.energies;
  e[1] = e[0] + (0.1 * fb::mcscf::kRootGapFloor);
  expect_refused(ref.tangents(ref.with_energies(e), cp, false, none), fb::ErrorKind::Numerical,
                 "roots within the floor", "roots 0 and 1");
  // Roots 0 and 1 1e-6 Eh apart: their Eq. qd-theta rotation past 10 per bohr.
  e[1] = e[0] + 1e-6;
  expect_refused(ref.tangents(ref.with_energies(e), cp, false, none), fb::ErrorKind::Numerical,
                 "a mixing rate past the bound", "along coordinate ");
  const auto past = ref.tangents(ref.with_energies(e), cp, false, none);
  if (!past.has_value()) {
    std::cout << std::format("[ qd tang  ] refusal at a 1e-6 Eh gap: {}\n", past.error().message);
    EXPECT_NE(past.error().message.find(std::format("(above {:.0f})", kMaxRotation)),
              std::string::npos);
  }
  // df_sacasscf_cp names itself.
  const auto cp_refused = fb::hessian::df_sacasscf_cp(
      res, ref.molecule(), ref.basis(), ref.classes(), ref.fit().aux, ref.h(), ref.fit().b,
      ref.fit().l.data(), *ref.solution, std::vector<double>{1.0}, ref.ncore, ref.ncas);
  ASSERT_FALSE(cp_refused.has_value());
  EXPECT_EQ(cp_refused.error().kind, config);
  EXPECT_NE(cp_refused.error().message.find("hessian::df_sacasscf_cp"), std::string::npos)
      << cp_refused.error().message;
}

} // namespace
