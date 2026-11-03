// GPU cases for nevpt2.gradient: sc_energy and sc_adjoint on this tree's own
// DF-CASSCF of data/casscf/casscf_df.json's h2o_ccpvdz (CAS(4,4), one root,
// so the CASSCF's core and virtuals are the root's canonical ones), run
// through the real registry as pipeline_nevpt2_gpu.cpp runs it. sc_energy
// against PySCF's DF-NEVPT2 (data/nevpt2/nevpt2_df.json); then each adjoint
// against a Richardson ladder of sc_energy itself, with every input rebuilt
// from (h, B, C, c) at each point, the orbital energies included: <u, y>
// along a spin-pure, symmetric u perpendicular to c; Σ_{p>q} κ_pq A_pq along
// a random rotation of each block (core-core and virtual-virtual included),
// the orbitals not re-canonicalized; and <h_bar, C^T dh C> and
// <Λ, C^T dB C> along seeded integral perturbations. Then z^c's: the
// Lagrangian E_corr + <z^c, f> stationary along core-core and
// virtual-virtual rotations, and each adjoint of Φ = <z^c, f> at fixed z^c
// against a ladder of Φ along the same directions; and the degenerate-pair
// rule. Then Φ over a state average of two roots, γ_SA's Fock, each adjoint
// against a ladder at equal and unequal weights, and the weights refused.
// Then the nuclear gradient (sc_gradient, lagrangian_gradient) on
// data/nevpt2_gradient/'s cases: against the numerical gradient, with
// E_corr zeroed against the DF-CASSCF's, and the full Lagrangian's
// stationarity. Then PC-NEVPT2's (pc_gradient) against the numerical one, its
// z^c's zero branch on a forced near-degenerate pair, and its refusals. Then
// the SC-QD-NEVPT2 functional E[W] (qd_sc_energy, qd_sc_adjoint) on
// data/qd_nevpt2/'s cases through this tree's own DF-SA-CASSCF: each adjoint
// against a ladder at the eigenvector seed and a random W, the one-root call
// against sc_adjoint, and its refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::string_view kCase = "h2o_ccpvdz";
// The fitting set both references were made over, compiled in.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";
// pipeline_nevpt2_gpu.cpp's CASSCF convergence and its bar on E_corr against
// PySCF's, derived there from how closely our CASSCF reaches the reference's
// orbitals.
constexpr double kConvTolGrad = 1e-9;
constexpr double kTolerance = 1e-9;
// sc_adjoint walks the same forward as sc_energy; the einsums' atomics
// reassociate every sum.
constexpr double kSameEnergyTolerance = 1e-12;
// nevpt2_energy_gpu.cpp's ladder: central differences D(t) at t = t0 / 2^k,
// successive changes falling by a factor in (3.2, 4.8) over at least three
// halvings, and the Richardson extrapolant at that run's finest pair against
// the adjoint's prediction to 1e-7 relative.
constexpr double kLadderT0 = 1e-2;
constexpr std::int32_t kRungs = 10;
constexpr double kLadderTolerance = 1e-7;
// The B direction is a tenth the length of the others: uniform in [-1, 1)
// on every fitted element, at full length its ladder ran ratios 5.41 -1.03
// -2.03 -2.02 3.23 4.17 4.08 4.02 (RTX 3080) and its extrapolant missed by
// 3.0e-6 relative, still far from the t^2 regime at t = 2e-5. At a tenth, the
// prediction a tenth of the full one's to all its digits, the ratios settle
// at 4.000 and the miss is 3.6e-10. So the bar was right and the direction
// too long, as nevpt2_energy_gpu.cpp's kRdmScale found; the tolerance is
// unchanged.
constexpr double kBScale = 0.1;

const fb::json &reference(std::string_view dir, std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  const std::string key = std::format("{}/{}", dir, file);
  if (const auto it = docs.find(key); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / dir / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(key, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &reference_case(std::string_view dir, std::string_view file, std::string_view name) {
  for (const fb::json &entry : reference(dir, file).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in " << dir << "/" << file;
  static const fb::json kNone;
  return kNone;
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> out(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size_bytes(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

// The converged state, on the host: B as density_fit leaves it, C, the AO
// core Hamiltonian and the CI vector.
struct State {
  std::int32_t naux = 0;
  std::int32_t nao = 0;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> h;
  std::vector<double> vector;
};

// The state in one gauge, whatever signs the CASSCF's eigensolvers left:
// each orbital's largest coefficient positive, the CI vector carried along
// (a determinant's sign is the product of its occupied active orbitals'
// flips, per spin), then its largest coefficient positive. A seeded direction
// is then the same rotation or displacement from run to run: unfixed, one
// seed's κ on the active-core block was a different rotation in another
// run's signs, and that one's ladder had not reached the t^2 regime by its
// last rung (ratios 2.98 3.91 5.00 6.49 6.17 4.87 4.25 4.08) while the
// first run's had, at 6.6e-9.
// The first element of @p v within 1e-8 of its largest magnitude. Not the
// largest itself: by H2O's mirror symmetry an orbital's coefficients on the
// two hydrogens (and a CI vector's mirrored determinants) tie in magnitude,
// and which of a tied pair is larger is round-off's choice.
double leading(std::span<const double> v) {
  double largest = 0.0;
  for (const double x : v) {
    largest = std::max(largest, std::abs(x));
  }
  for (const double x : v) {
    if (std::abs(x) >= largest - 1e-8) {
      return x;
    }
  }
  return 0.0;
}

void fix_gauge(State &s) {
  const auto nao = static_cast<std::size_t>(s.nao);
  std::vector<bool> flipped(nao, false);
  for (std::size_t p = 0; p < nao; ++p) {
    const auto column = std::span<double>(s.c).subspan(p * nao, nao);
    if (leading(column) < 0.0) {
      flipped[p] = true;
      for (double &v : column) {
        v = -v;
      }
    }
  }
  auto strings = fb::ci::StringSet::create(s.ncas, s.nelecas / 2);
  EXPECT_TRUE(strings.has_value()) << strings.error().message;
  if (!strings.has_value()) {
    return;
  }
  std::vector<double> sign;
  for (const std::uint64_t string : strings->strings()) {
    double value = 1.0;
    for (std::int32_t t = 0; t < s.ncas; ++t) {
      if (((string >> t) & 1U) != 0U && flipped[static_cast<std::size_t>(s.ncore + t)]) {
        value = -value;
      }
    }
    sign.push_back(value);
  }
  // Each root's, when s.vector holds several, root-major.
  const std::size_t m = sign.size();
  EXPECT_EQ(s.vector.size() % (m * m), 0U);
  for (std::size_t first = 0; first + (m * m) <= s.vector.size(); first += m * m) {
    const auto root = std::span<double>(s.vector).subspan(first, m * m);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        root[(i * m) + j] *= sign[i] * sign[j];
      }
    }
    if (leading(root) < 0.0) {
      for (double &v : root) {
        v = -v;
      }
    }
  }
}

// This tree's DF-CASSCF from a reference's start (@p start: the molecule,
// basis, fitting set and active space of a casscf_df.json or
// nevpt2_gradient.json case) through the real registry, in @p node; the
// Calculation, which holds every product, or nothing on a failure (already
// reported).
std::optional<fb::Calculation> run_casscf(const fb::json &start, fb::NodeResources &node) {
  EXPECT_EQ(start.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_gradient");
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
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                  start.at("charge").get<std::int32_t>(), start.at("basis").get<std::string>(),
                  kConvTolGrad, start.at("ncas").get<std::int32_t>(),
                  start.at("nelecas").get<std::int32_t>(), active, kDfAuxBasis);
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

// The converged state of @p calc, on the host, in one gauge (fix_gauge): its
// @p nroots roots' vectors root-major in State::vector.
State host_state(fb::Calculation &calc, fb::DeviceResources &res, const std::size_t nroots = 1) {
  State s;
  const auto &casscf = calc.get<fb::CasscfOutput>();
  const auto &eri = calc.get<fb::EriOutput>();
  const auto &h = calc.get<fb::CoreHamiltonianOutput>();
  EXPECT_TRUE(casscf.has_value() && eri.has_value() && h.has_value());
  if (!casscf.has_value() || !eri.has_value() || !h.has_value()) {
    return s;
  }
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
  EXPECT_EQ(casscf->energies.size(), nroots);
  const auto &df = std::get<fb::DensityFitted>(eri->eri);
  s.naux = df.naux;
  s.nao = static_cast<std::int32_t>(casscf->c.n());
  s.ncore = casscf->ncore;
  s.ncas = casscf->ncas;
  s.nelecas = casscf->nelecas;
  const auto nao = static_cast<std::size_t>(s.nao);
  s.b = download(res, df.b.data(), static_cast<std::size_t>(s.naux) * nao * nao);
  s.c = download(res, casscf->c.data(), nao * nao);
  s.h = download(res, h->h.data(), nao * nao);
  s.vector = casscf->vectors;
  fix_gauge(s);
  return s;
}

// Our DF-CASSCF from casscf_df.json's start, run once per process.
const State &state() {
  static const State kState = [] {
    fb::NodeResources node;
    std::optional<fb::Calculation> calc =
        run_casscf(reference_case("casscf", "casscf_df.json", kCase), node);
    return calc.has_value() ? host_state(*calc, node.device(0)) : State{};
  }();
  return kState;
}

// The state on a device, and its strings: one set for both spins.
struct DeviceState {
  fb::Buffers::device<double> b;
  fb::Buffers::device<double> c;
  fb::Buffers::device<double> h;
  fb::Buffers::device<double> vector;
  std::optional<fb::ci::DeviceStringSet> strings;
};

DeviceState upload_state(fb::DeviceResources &res, const State &s) {
  DeviceState d{.b = upload(res, s.b),
                .c = upload(res, s.c),
                .h = upload(res, s.h),
                .vector = upload(res, s.vector),
                .strings = std::nullopt};
  auto strings = fb::ci::DeviceStringSet::create(res, s.ncas, s.nelecas / 2);
  EXPECT_TRUE(strings.has_value()) << strings.error().message;
  if (strings.has_value()) {
    d.strings.emplace(std::move(*strings));
  }
  return d;
}

// sc_energy's E_corr at the state @p s, uploaded.
double energy(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  const auto e = nevpt2::sc_energy(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                   d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
  EXPECT_TRUE(e.has_value()) << e.error().message;
  return e.has_value() ? e->total : std::nan("");
}

// The adjoint at @p s, on the host: h_bar, Lambda, y and A.
struct HostAdjoint {
  double total = 0.0;
  std::vector<double> h_bar;
  std::vector<double> lambda;
  std::vector<double> y;
  std::vector<double> x;
  std::vector<double> a;
  std::vector<double> epsilon;
};

HostAdjoint adjoint(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  const auto got = nevpt2::sc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                      d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  if (!got.has_value()) {
    return {};
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  return {.total = got->energies.total,
          .h_bar = download(res, got->h_bar.data(), nmo * nmo),
          .lambda = download(res, got->lambda.data(), static_cast<std::size_t>(s.naux) * nmo * nmo),
          .y = download(res, got->y.data(), s.vector.size()),
          .x = download(res, got->x.data(), nmo * nmo),
          .a = download(res, got->a.data(), nmo * nmo),
          .epsilon = download(res, got->epsilon.data(), nmo)};
}

// pc_adjoint at @p s, on the host, as adjoint() takes sc_adjoint's; or its
// error.
fb::Result<HostAdjoint> pc_adjoint_of(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  const auto got = nevpt2::pc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                      d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
  if (!got.has_value()) {
    return std::unexpected(got.error());
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  return HostAdjoint{
      .total = got->energies.total,
      .h_bar = download(res, got->h_bar.data(), nmo * nmo),
      .lambda = download(res, got->lambda.data(), static_cast<std::size_t>(s.naux) * nmo * nmo),
      .y = download(res, got->y.data(), s.vector.size()),
      .x = download(res, got->x.data(), nmo * nmo),
      .a = download(res, got->a.data(), nmo * nmo),
      .epsilon = download(res, got->epsilon.data(), nmo)};
}

// pc_energy at @p s.
fb::Result<nevpt2::PcClassEnergies> pc_energy_of(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  return nevpt2::pc_energy(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                           d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
}

// C = A B, column-major: A m x k, B k x n.
std::vector<double> matmul(std::span<const double> a, std::span<const double> b, std::size_t m,
                           std::size_t k, std::size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t l = 0; l < k; ++l) {
      const double blj = b[l + (k * j)];
      for (std::size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += a[i + (m * l)] * blj;
      }
    }
  }
  return c;
}

// exp(K) for a small n x n K, by its Taylor series: |t K| <= 1e-2 |K| here.
std::vector<double> expm(std::span<const double> k, std::size_t n) {
  std::vector<double> out(n * n, 0.0);
  std::vector<double> term(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out[i * (n + 1)] = 1.0;
    term[i * (n + 1)] = 1.0;
  }
  for (std::int32_t m = 1; m < 60; ++m) {
    term = matmul(term, k, n, n, n);
    double largest = 0.0;
    for (std::size_t i = 0; i < term.size(); ++i) {
      term[i] /= m;
      out[i] += term[i];
      largest = std::max(largest, std::abs(term[i]));
    }
    if (largest < 1e-20) {
      break;
    }
  }
  return out;
}

// The AO core Hamiltonian whose MO view C^T h C is @p s's plus @p delta on
// the diagonal of each of @p orbitals and nothing else: h + delta Σ_j v_j
// v_jᵀ, v_j = C^-T e_j (C is square), so every orbital energy and F^I
// element but those of the moved orbitals is the state's own, and theirs
// move by delta. C^-1 by Gauss-Jordan with partial pivoting.
std::vector<double> shifted_h(const State &s, std::span<const std::size_t> orbitals,
                              const double delta) {
  const auto n = static_cast<std::size_t>(s.nao);
  // [C | I], row-major, reduced to [I | C^-1].
  std::vector<double> m(n * 2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[(i * 2 * n) + j] = s.c[i + (n * j)];
    }
    m[(i * 2 * n) + n + i] = 1.0;
  }
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(m[(i * 2 * n) + k]) > std::abs(m[(pivot * 2 * n) + k])) {
        pivot = i;
      }
    }
    for (std::size_t j = 0; j < 2 * n; ++j) {
      std::swap(m[(k * 2 * n) + j], m[(pivot * 2 * n) + j]);
    }
    const double inv = 1.0 / m[(k * 2 * n) + k];
    for (std::size_t j = 0; j < 2 * n; ++j) {
      m[(k * 2 * n) + j] *= inv;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double f = m[(i * 2 * n) + k];
      if (i != k && f != 0.0) {
        for (std::size_t j = 0; j < 2 * n; ++j) {
          m[(i * 2 * n) + j] -= f * m[(k * 2 * n) + j];
        }
      }
    }
  }
  std::vector<double> h = s.h;
  for (const std::size_t j : orbitals) {
    // v_mu = (C^-T)_{mu j} = (C^-1)_{j mu}.
    const auto v = std::span<const double>(m).subspan((j * 2 * n) + n, n);
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        h[mu + (n * nu)] += delta * v[mu] * v[nu];
      }
    }
  }
  return h;
}

// C^T M C for an nao x nao M: the MO basis's view of an AO matrix.
std::vector<double> to_mo(const State &s, std::span<const double> m) {
  const auto n = static_cast<std::size_t>(s.nao);
  std::vector<double> ct(n * n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t mu = 0; mu < n; ++mu) {
      ct[p + (n * mu)] = s.c[mu + (n * p)];
    }
  }
  return matmul(matmul(ct, m, n, n, n), s.c, n, n, n);
}

double dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// A ladder of central differences D_k at t = t0 / 2^k: the longest run of
// ratios (D_k - D_{k+1}) / (D_{k+1} - D_{k+2}) in (3.2, 4.8) (the O(t²)
// regime), and the Richardson extrapolant R_k = (4 D_{k+2} - D_{k+1}) / 3
// at that run's finest pair. With @p steadiest, it is instead the finer of
// the run's two neighbouring extrapolants that agree best: R_k's truncation
// falls 16-fold a rung and its rounding grows 2-fold, so where a ladder
// reaches its rounding inside the run, the finest pair reads the most of it.
struct Ladder {
  std::int32_t run = 0;
  double richardson = 0.0;
  std::string ratios;
};

Ladder richardson_ladder(std::span<const double> d, const bool steadiest = false) {
  Ladder l;
  std::int32_t best_end = -1;
  std::int32_t len = 0;
  for (std::size_t k = 0; k + 2 < d.size(); ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    l.ratios += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > l.run) {
      l.run = len;
      best_end = static_cast<std::int32_t>(k);
    }
  }
  if (l.run == 0) {
    return l;
  }
  const auto r = [&](const std::size_t k) { return ((4.0 * d[k + 2]) - d[k + 1]) / 3.0; };
  auto at = static_cast<std::size_t>(best_end);
  if (steadiest && l.run > 1) {
    double gap = std::numeric_limits<double>::infinity();
    for (auto k = static_cast<std::size_t>(best_end - l.run + 1);
         k < static_cast<std::size_t>(best_end); ++k) {
      if (std::abs(r(k + 1) - r(k)) < gap) {
        gap = std::abs(r(k + 1) - r(k));
        at = k + 1;
      }
    }
  }
  l.richardson = r(at);
  return l;
}

// Central differences of @p at over the ladder, against @p predicted, to
// kLadderTolerance relative to @p scale: |predicted| unless given, which a
// stationarity check, predicting 0, gives as the slope it cancels.
void expect_ladder(const std::string &name, const std::function<double(double)> &at,
                   const double predicted, const std::optional<double> scale = std::nullopt) {
  std::vector<double> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(kLadderT0, -k);
    d.push_back((at(t) - at(-t)) / (2.0 * t));
  }
  const Ladder l = richardson_ladder(d);
  ASSERT_GE(l.run, 3) << name << " ratios:" << l.ratios;
  const double against = scale.value_or(std::abs(predicted));
  EXPECT_NEAR(l.richardson, predicted, kLadderTolerance * against)
      << name << " ratios:" << l.ratios;
  std::cout << std::format("[ nevpt2   ] gradient ladder {}: predicted {:.12e}, Richardson "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           name, predicted, l.richardson,
                           std::abs(l.richardson - predicted) / against, l.ratios);
}

TEST(Nevpt2Gradient, EnergyMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const fb::json &want = reference_case("nevpt2", "nevpt2_df.json", kCase);
  const double e_corr = want.at("roots").at(0).at("e_corr").get<double>();
  const double got = energy(res, s);
  EXPECT_NEAR(got, e_corr, kTolerance);
  const HostAdjoint adj = adjoint(res, s);
  EXPECT_NEAR(adj.total, got, kSameEnergyTolerance);
  std::cout << std::format("[ nevpt2   ] {}: sc_energy {:.12f}, dE_corr {:+.2e}\n", kCase, got,
                           got - e_corr);
}

// u = P H_r c, normalized: the CI direction of every ladder along c(t) =
// (c + t u) / |c + t u|. H_r is spin-free, so u is in c's spin; and it is a
// random combination of operators that commute with the molecule's point
// group (the active block of C^T h C, its square and the active (tu|vw)), so
// u is in c's symmetry sector too. A direction outside either takes the
// hole-side metrics where they are not positive, and the SC finish has
// poles there.
std::vector<double> ci_direction(fb::DeviceResources &res, const State &s) {
  const auto n = static_cast<std::size_t>(s.ncas);
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  const auto nc = static_cast<std::size_t>(s.ncore);

  std::mt19937_64 engine(562);
  std::uniform_real_distribution<double> scale(0.5, 1.5);
  const std::vector<double> h_mo = to_mo(s, s.h);
  std::vector<double> h_act(n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      h_act[(t * n) + u] = h_mo[(nc + t) + (nao * (nc + u))];
    }
  }
  const std::vector<double> h_sq = matmul(h_act, h_act, n, n, n);
  const double s1 = scale(engine);
  const double s2 = scale(engine);
  const double s3 = scale(engine);
  std::vector<double> h_r(n * n);
  for (std::size_t k = 0; k < h_r.size(); ++k) {
    h_r[k] = (s1 * h_act[k]) + (s2 * h_sq[k]);
  }
  // (tu|vw) over the fitted B: B_L,tu = Σ_mn C_m,t C_n,u B_L,mn.
  std::vector<double> b_act(naux * n * n, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      for (std::size_t mu = 0; mu < nao; ++mu) {
        for (std::size_t nu = 0; nu < nao; ++nu) {
          const double cc = s.c[mu + (nao * (nc + t))] * s.c[nu + (nao * (nc + u))];
          for (std::size_t l = 0; l < naux; ++l) {
            b_act[l + (naux * (t + (n * u)))] += cc * s.b[l + (naux * (mu + (nao * nu)))];
          }
        }
      }
    }
  }
  std::vector<double> eri_r(n * n * n * n);
  for (std::size_t tu = 0; tu < n * n; ++tu) {
    for (std::size_t vw = 0; vw < n * n; ++vw) {
      double sum = 0.0;
      for (std::size_t l = 0; l < naux; ++l) {
        sum += b_act[l + (naux * tu)] * b_act[l + (naux * vw)];
      }
      // Row-major (t, u, v, w); B_tu is symmetric, so tu's order is moot.
      eri_r[(tu * n * n) + vw] = s3 * sum;
    }
  }
  const DeviceState d = upload_state(res, s);
  auto sigma = fb::ci::Sigma::create(res, *d.strings, *d.strings);
  EXPECT_TRUE(sigma.has_value()) << sigma.error().message;
  if (!sigma.has_value()) {
    return {};
  }
  EXPECT_TRUE(sigma->set_integrals(res, 0.0, h_r, eri_r).has_value());
  fb::Buffers::device<double> hc(s.vector.size(), res.device());
  EXPECT_TRUE(sigma->apply(res, 1, d.vector.data(), hc.data()).has_value());
  std::vector<double> u = download(res, hc.data(), s.vector.size());
  const double overlap = dot(u, s.vector);
  for (std::size_t i = 0; i < u.size(); ++i) {
    u[i] -= overlap * s.vector[i];
  }
  const double norm = std::sqrt(dot(u, u));
  EXPECT_GT(norm, 1e-6);
  for (double &v : u) {
    v /= norm;
  }
  return u;
}

// The state moved along c(t) = (c + t u) / |c + t u| into @p moved.
void move_vector(const State &s, std::span<const double> u, const double t, State &moved) {
  for (std::size_t i = 0; i < u.size(); ++i) {
    moved.vector[i] = s.vector[i] + (t * u[i]);
  }
  const double length = std::sqrt(dot(moved.vector, moved.vector));
  for (double &v : moved.vector) {
    v /= length;
  }
}

// <u, y> along u = ci_direction's.
TEST(Nevpt2Gradient, CiAdjointMatchesLadder) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> u = ci_direction(res, s);
  ASSERT_EQ(u.size(), s.vector.size());

  State moved = s;
  expect_ladder(
      "ci",
      [&](const double t) {
        move_vector(s, u, t, moved);
        return energy(res, moved);
      },
      dot(u, adj.y));
}

// Σ_{p>q} κ_pq A_pq along C(t) = C exp(t κ), κ random and antisymmetric on
// one block, everything rebuilt from C(t) and nothing re-canonicalized.
struct Block {
  std::string_view name;
  // The two orbital spaces, each 0 core, 1 active, 2 virtual.
  std::int32_t row;
  std::int32_t col;
};

void PrintTo(const Block &b, std::ostream *os) {
  *os << b.name;
}

// Each block alone: the three inside a space and the three between.
constexpr std::array<Block, 6> kBlocks{{{.name = "core_core", .row = 0, .col = 0},
                                        {.name = "active_active", .row = 1, .col = 1},
                                        {.name = "virtual_virtual", .row = 2, .col = 2},
                                        {.name = "active_core", .row = 1, .col = 0},
                                        {.name = "virtual_core", .row = 2, .col = 0},
                                        {.name = "virtual_active", .row = 2, .col = 1}}};

// A random antisymmetric κ on @p block alone, seeded by the block, nmo x nmo
// column-major.
std::vector<double> block_kappa(const State &s, const Block &block) {
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::array<std::size_t, 4> first{0, static_cast<std::size_t>(s.ncore),
                                         static_cast<std::size_t>(s.ncore + s.ncas), nmo};
  std::mt19937_64 engine(562 + (3 * block.row) + block.col);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> kappa(nmo * nmo, 0.0);
  for (std::size_t p = first[block.row]; p < first[block.row + 1]; ++p) {
    for (std::size_t q = first[block.col]; q < first[block.col + 1]; ++q) {
      if (p <= q && block.row == block.col) {
        continue;
      }
      const double k = uniform(engine);
      kappa[p + (nmo * q)] = k;
      kappa[q + (nmo * p)] = -k;
    }
  }
  return kappa;
}

// Σ_{p>q} κ_pq A_pq: the first-order change along C exp(t κ).
double source_dot(std::span<const double> kappa, std::span<const double> a, std::size_t nmo) {
  double sum = 0.0;
  for (std::size_t q = 0; q < nmo; ++q) {
    for (std::size_t p = q + 1; p < nmo; ++p) {
      sum += kappa[p + (nmo * q)] * a[p + (nmo * q)];
    }
  }
  return sum;
}

// The state with C(t) = C exp(t κ) into @p moved.
void rotate(const State &s, std::span<const double> kappa, const double t, State &moved) {
  const auto nmo = static_cast<std::size_t>(s.nao);
  std::vector<double> tk(kappa.begin(), kappa.end());
  for (double &v : tk) {
    v *= t;
  }
  moved.c = matmul(s.c, expm(tk, nmo), nmo, nmo, nmo);
}

class Nevpt2GradientRotation : public ::testing::TestWithParam<Block> {};

TEST_P(Nevpt2GradientRotation, OrbitalSourceMatchesLadder) {
  FBF_REQUIRE_GPU();
  const Block &block = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> kappa = block_kappa(s, block);

  State moved = s;
  expect_ladder(
      std::string{block.name},
      [&](const double t) {
        rotate(s, kappa, t, moved);
        return energy(res, moved);
      },
      source_dot(kappa, adj.a, static_cast<std::size_t>(s.nao)));
}

INSTANTIATE_TEST_SUITE_P(Blocks, Nevpt2GradientRotation, ::testing::ValuesIn(kBlocks),
                         [](const auto &info) { return std::string(info.param.name); });

// A seeded symmetric AO dh ("h"), or a seeded dB symmetric in its AO pair
// ("B"), the other zero.
struct IntegralDirection {
  std::vector<double> dh;
  std::vector<double> db;
};

IntegralDirection integral_direction(const State &s, const std::string &which) {
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  std::mt19937_64 engine(which == "h" ? 5621 : 5622);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  IntegralDirection dir{.dh = std::vector<double>(nao * nao, 0.0),
                        .db = std::vector<double>(naux * nao * nao, 0.0)};
  for (std::size_t mu = 0; mu < nao; ++mu) {
    for (std::size_t nu = 0; nu <= mu; ++nu) {
      if (which == "h") {
        dir.dh[mu + (nao * nu)] = dir.dh[nu + (nao * mu)] = uniform(engine);
        continue;
      }
      for (std::size_t l = 0; l < naux; ++l) {
        dir.db[l + (naux * (mu + (nao * nu)))] = dir.db[l + (naux * (nu + (nao * mu)))] =
            kBScale * uniform(engine);
      }
    }
  }
  return dir;
}

// <h_bar, C^T dh C> + Σ_L <Λ^L, C^T dB^L C>.
double integral_dot(const State &s, const IntegralDirection &dir, std::span<const double> h_bar,
                    std::span<const double> lambda) {
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  double sum = dot(h_bar, to_mo(s, dir.dh));
  std::vector<double> slice(nao * nao);
  for (std::size_t l = 0; l < naux; ++l) {
    for (std::size_t mn = 0; mn < nao * nao; ++mn) {
      slice[mn] = dir.db[l + (naux * mn)];
    }
    const std::vector<double> mo = to_mo(s, slice);
    for (std::size_t pq = 0; pq < nao * nao; ++pq) {
      sum += lambda[l + (naux * pq)] * mo[pq];
    }
  }
  return sum;
}

// The integrals moved by @p step along @p dir into @p moved.
void perturb(const State &s, const IntegralDirection &dir, const double step, State &moved) {
  for (std::size_t k = 0; k < dir.dh.size(); ++k) {
    moved.h[k] = s.h[k] + (step * dir.dh[k]);
  }
  for (std::size_t k = 0; k < dir.db.size(); ++k) {
    moved.b[k] = s.b[k] + (step * dir.db[k]);
  }
}

// <h_bar, C^T dh C> along a seeded symmetric dh, and <Λ, C^T dB C> along a
// seeded dB symmetric in its AO pair, each alone, at fixed C and c.
class Nevpt2GradientIntegrals : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2GradientIntegrals, IntegralAdjointMatchesLadder) {
  FBF_REQUIRE_GPU();
  const std::string &which = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const IntegralDirection dir = integral_direction(s, which);

  State moved = s;
  expect_ladder(
      which,
      [&](const double t) {
        perturb(s, dir, t, moved);
        return energy(res, moved);
      },
      integral_dot(s, dir, adj.h_bar, adj.lambda));
}

INSTANTIATE_TEST_SUITE_P(Integrals, Nevpt2GradientIntegrals,
                         ::testing::Values(std::string{"h"}, std::string{"B"}),
                         [](const auto &info) { return info.param; });

// The orbital source term's parts: A is X's antisymmetric part, and
// h_bar and every Λ^L symmetric, as the AO half-transform assumes.
TEST(Nevpt2Gradient, SourceTermShapes) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  double worst_a = 0.0;
  double worst_h = 0.0;
  double worst_lambda = 0.0;
  for (std::size_t p = 0; p < nmo; ++p) {
    for (std::size_t q = 0; q < nmo; ++q) {
      worst_a = std::max(
          worst_a, std::abs(adj.a[p + (nmo * q)] - (adj.x[p + (nmo * q)] - adj.x[q + (nmo * p)])));
      worst_h = std::max(worst_h, std::abs(adj.h_bar[p + (nmo * q)] - adj.h_bar[q + (nmo * p)]));
      for (std::size_t l = 0; l < naux; ++l) {
        worst_lambda = std::max(worst_lambda, std::abs(adj.lambda[l + (naux * (p + (nmo * q)))] -
                                                       adj.lambda[l + (naux * (q + (nmo * p)))]));
      }
    }
  }
  EXPECT_LT(worst_a, 1e-14);
  EXPECT_EQ(worst_h, 0.0);
  EXPECT_LT(worst_lambda, 1e-15);
}

// zc_multipliers at the source term @p a and orbital energies @p epsilon, on
// the host.
fb::Result<nevpt2::ZcMultipliers> multipliers(fb::DeviceResources &res, const State &s,
                                              std::span<const double> a,
                                              std::span<const double> epsilon) {
  const fb::Buffers::device<double> a_d = upload(res, a);
  const fb::Buffers::device<double> e_d = upload(res, epsilon);
  return nevpt2::zc_multipliers(res, s.nao, s.ncore, s.ncas, a_d.data(), e_d.data());
}

// z^c at the state, from sc_adjoint's A and ε, on the host.
std::vector<double> zc_of(fb::DeviceResources &res, const State &s, const HostAdjoint &adj) {
  const auto got = multipliers(res, s, adj.a, adj.epsilon);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  if (!got.has_value()) {
    return {};
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  std::cout << std::format("[ nevpt2   ] z^c: max |z| {:.3e}, smallest inactive gap {:.3e}\n",
                           got->max_z, got->min_gap);
  return download(res, got->z.data(), nmo * nmo);
}

// zc_adjoint at @p s for the fixed @p z, on the host.
struct HostZcAdjoint {
  double phi = 0.0;
  std::vector<double> h_bar;
  std::vector<double> lambda;
  std::vector<double> y;
  std::vector<double> x;
  std::vector<double> a;
};

// The state's own root alone, at weight 1: the single-state Φ.
constexpr std::array<double, 1> kOneRoot{1.0};

// zc_adjoint at @p s's integrals and orbitals for the fixed @p z, over the
// roots @p roots (root-major, each s.vector's length; empty for s.vector
// itself) at @p weights.
fb::Result<nevpt2::ZcAdjoint> zc_at(fb::DeviceResources &res, const State &s, const DeviceState &d,
                                    const fb::Buffers::device<double> &z,
                                    std::span<const double> weights,
                                    const fb::Buffers::device<double> &roots) {
  return nevpt2::zc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                            d.c.data(), d.h.data(), s.ncore, s.ncas, weights, roots.data(),
                            z.data());
}

HostZcAdjoint zc_adjoint(fb::DeviceResources &res, const State &s, std::span<const double> z,
                         std::span<const double> weights = kOneRoot,
                         std::span<const double> roots = {}) {
  const DeviceState d = upload_state(res, s);
  const fb::Buffers::device<double> z_d = upload(res, z);
  const std::span<const double> vectors = roots.empty() ? std::span<const double>(s.vector) : roots;
  const fb::Buffers::device<double> roots_d = upload(res, vectors);
  const auto got = zc_at(res, s, d, z_d, weights, roots_d);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  if (!got.has_value()) {
    return {};
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  return {.phi = got->phi,
          .h_bar = download(res, got->h_bar.data(), nmo * nmo),
          .lambda = download(res, got->lambda.data(), static_cast<std::size_t>(s.naux) * nmo * nmo),
          .y = download(res, got->y.data(), vectors.size()),
          .x = download(res, got->x.data(), nmo * nmo),
          .a = download(res, got->a.data(), nmo * nmo)};
}

// Φ = <z, f> at @p s for the fixed @p z, over @p roots at @p weights as
// zc_adjoint's.
double phi(fb::DeviceResources &res, const State &s, std::span<const double> z,
           std::span<const double> weights = kOneRoot, std::span<const double> roots = {}) {
  const DeviceState d = upload_state(res, s);
  const fb::Buffers::device<double> z_d = upload(res, z);
  const fb::Buffers::device<double> roots_d =
      upload(res, roots.empty() ? std::span<const double>(s.vector) : roots);
  const auto got = zc_at(res, s, d, z_d, weights, roots_d);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return got.has_value() ? got->phi : std::nan("");
}

// The Lagrangian E_corr + <z^c, f> along C exp(t κ) for κ on the core-core
// or the virtual-virtual block, z^c fixed and ε rebuilt from C(t), not
// re-canonicalized: flat at first order, to the ladder's 1e-7 of the slope
// Σ κ A it cancels. And the same at the point, Σ κ (A + A^{z^c}).
class Nevpt2ZcStationarity : public ::testing::TestWithParam<Block> {};

TEST_P(Nevpt2ZcStationarity, LagrangianIsFlat) {
  FBF_REQUIRE_GPU();
  const Block &block = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const HostZcAdjoint zadj = zc_adjoint(res, s, z);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::vector<double> kappa = block_kappa(s, block);
  const double slope = source_dot(kappa, adj.a, nmo);
  ASSERT_GT(std::abs(slope), 0.0);
  const double at_point = slope + source_dot(kappa, zadj.a, nmo);
  EXPECT_NEAR(at_point, 0.0, kLadderTolerance * std::abs(slope));
  std::cout << std::format("[ nevpt2   ] z^c {}: slope {:.6e}, with A^zc {:.3e}\n", block.name,
                           slope, at_point);

  State moved = s;
  expect_ladder(
      std::format("lagrangian {}", block.name),
      [&](const double t) {
        rotate(s, kappa, t, moved);
        return energy(res, moved) + phi(res, moved, z);
      },
      0.0, std::abs(slope));
}

INSTANTIATE_TEST_SUITE_P(InactiveBlocks, Nevpt2ZcStationarity,
                         ::testing::Values(kBlocks[0], kBlocks[2]),
                         [](const auto &info) { return std::string(info.param.name); });

// Σ_{p>q} κ_pq A^{z^c}_pq against a ladder of Φ at fixed z^c along
// C exp(t κ), κ on each block alone.
class Nevpt2ZcRotation : public ::testing::TestWithParam<Block> {};

TEST_P(Nevpt2ZcRotation, OrbitalSourceMatchesLadder) {
  FBF_REQUIRE_GPU();
  const Block &block = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const HostZcAdjoint zadj = zc_adjoint(res, s, z);
  const std::vector<double> kappa = block_kappa(s, block);

  State moved = s;
  expect_ladder(
      std::format("phi {}", block.name),
      [&](const double t) {
        rotate(s, kappa, t, moved);
        return phi(res, moved, z);
      },
      source_dot(kappa, zadj.a, static_cast<std::size_t>(s.nao)));
}

INSTANTIATE_TEST_SUITE_P(Blocks, Nevpt2ZcRotation, ::testing::ValuesIn(kBlocks),
                         [](const auto &info) { return std::string(info.param.name); });

// <u, y^c> against a ladder of Φ at fixed z^c along ci_direction's u.
TEST(Nevpt2Zc, CiSourceMatchesLadder) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const HostZcAdjoint zadj = zc_adjoint(res, s, z);
  const std::vector<double> u = ci_direction(res, s);
  ASSERT_EQ(u.size(), s.vector.size());

  State moved = s;
  expect_ladder(
      "phi ci",
      [&](const double t) {
        move_vector(s, u, t, moved);
        return phi(res, moved, z);
      },
      dot(u, zadj.y));
}

// <h_bar^{z^c}, C^T dh C> and <Λ^{z^c}, C^T dB C> against a ladder of Φ at
// fixed z^c, along the sc_adjoint ladders' directions. Φ is linear in h and
// quadratic in B, so along a straight line its central difference is exact
// at every t and the ladder's changes are round-off, with no t^2 regime for
// the ratio rule to find. The path is bent instead, (t + t^3 / t0^2) along
// the direction: the same tangent, so the same prediction, and a central
// difference of exactly D(0) (1 + t^2 / t0^2), whose changes fall by 4.
class Nevpt2ZcIntegrals : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2ZcIntegrals, IntegralAdjointMatchesLadder) {
  FBF_REQUIRE_GPU();
  const std::string &which = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const HostZcAdjoint zadj = zc_adjoint(res, s, z);
  const IntegralDirection dir = integral_direction(s, which);

  State moved = s;
  expect_ladder(
      std::format("phi {}", which),
      [&](const double t) {
        perturb(s, dir, t * (1.0 + ((t * t) / (kLadderT0 * kLadderT0))), moved);
        return phi(res, moved, z);
      },
      integral_dot(s, dir, zadj.h_bar, zadj.lambda));
}

INSTANTIATE_TEST_SUITE_P(Integrals, Nevpt2ZcIntegrals,
                         ::testing::Values(std::string{"h"}, std::string{"B"}),
                         [](const auto &info) { return info.param; });

// The degenerate-pair rule on a forced pair: the first virtual and a partner
// put 1e-7 Eh apart, under the 1e-6 floor. With A as sc_adjoint gives it the
// pair is refused, Numerical, naming it; with A_pq at round-off (1e-12) its
// z is zero, and every pair without the moved orbital keeps its division.
TEST(Nevpt2Zc, DegeneratePairRule) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto q = static_cast<std::size_t>(s.ncore + s.ncas);
  const std::size_t p = q + 1;
  ASSERT_LT(p, nmo);
  // The two lowest virtuals of different symmetry have A at round-off, so
  // the pair is the first above it: then the refusal is A's, not chance.
  std::size_t partner = p;
  while (partner < nmo && std::abs(adj.a[partner + (nmo * q)]) <= 1e-6) {
    ++partner;
  }
  ASSERT_LT(partner, nmo) << "no virtual pair with q = " << q << " and |A| > 1e-6";
  std::vector<double> epsilon = adj.epsilon;
  epsilon[partner] = epsilon[q] + 1e-7;

  const auto refused = multipliers(res, s, adj.a, epsilon);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical);
  const std::string named = std::format("orbitals {} and {}", partner, q);
  EXPECT_NE(refused.error().message.find(named), std::string::npos) << refused.error().message;

  std::vector<double> a = adj.a;
  a[partner + (nmo * q)] = 1e-12;
  a[q + (nmo * partner)] = -1e-12;
  const auto kept = multipliers(res, s, a, epsilon);
  ASSERT_TRUE(kept.has_value()) << kept.error().message;
  EXPECT_LE(kept->min_gap, 1e-7 * (1.0 + 1e-6));
  const std::vector<double> z = download(res, kept->z.data(), nmo * nmo);
  EXPECT_EQ(z[partner + (nmo * q)], 0.0);
  EXPECT_EQ(z[q + (nmo * partner)], 0.0);
  // Every pair without the moved orbital keeps its division to the bit.
  const auto unforced = multipliers(res, s, adj.a, adj.epsilon);
  ASSERT_TRUE(unforced.has_value()) << unforced.error().message;
  const std::vector<double> z0 = download(res, unforced->z.data(), nmo * nmo);
  for (std::size_t i = 0; i < nmo; ++i) {
    for (std::size_t j = 0; j < nmo; ++j) {
      if (i != partner && j != partner) {
        EXPECT_EQ(z[i + (nmo * j)], z0[i + (nmo * j)]) << i << ", " << j;
      }
    }
  }
  EXPECT_GT(unforced->max_z, 0.0);
}

// PC's z^c on a forced near-degenerate inactive pair takes the zero branch,
// where SC's is refused. PC is invariant to a rotation of two inactive
// orbitals only when their orbital energies are equal (nevpt2_energy_gpu.cpp's
// Nevpt2PcEnergyAdjoint.InactiveRotationsAreRoundOff, at the class level):
// at the real state, with distinct ε, its A_pq is not zero and z^c is the
// same division as SC's. So in each of the core and the virtual space, the
// pair with |A^SC| > 1e-6 closest in ε is pushed together, the upper one
// moved to 1e-9 Eh above the lower through h alone (shifted_h: C^T h C
// moves on that orbital's diagonal, so f stays diagonal and nothing else
// does): PC's A_pq there is round-off, under zc_multipliers' 1e-8, and its
// z_pq exactly 0, while SC's A_pq stays past it and the pair is refused,
// Numerical, naming both orbitals. SC's A_pq falls too as the pair closes
// (on the gfx1200, 7.3e-3 to 1.5e-7 on the core pair, 4.6e-5 to 7.4e-8 on
// the virtual one), but stops there, where PC's reaches 1e-13: so the
// control is the rule's own 1e-8, and PC's at least 1e3 below SC's.
TEST(Nevpt2PcZc, DegeneratePairTakesTheZeroBranch) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const HostAdjoint sc = adjoint(res, s);
  const auto pc = pc_adjoint_of(res, s);
  ASSERT_TRUE(pc.has_value()) << pc.error().message;
  const std::array<std::size_t, 3> first{0, static_cast<std::size_t>(s.ncore),
                                         static_cast<std::size_t>(s.ncore + s.ncas)};
  for (const bool core : {true, false}) {
    const char *const what = core ? "core-core" : "virtual-virtual";
    const std::size_t lo = core ? first[0] : first[2];
    const std::size_t hi = core ? first[1] : nmo;
    std::optional<std::pair<std::size_t, std::size_t>> pair;
    double gap = std::numeric_limits<double>::infinity();
    for (std::size_t q = lo; q < hi; ++q) {
      for (std::size_t p = q + 1; p < hi; ++p) {
        const double g = std::abs(sc.epsilon[p] - sc.epsilon[q]);
        if (std::abs(sc.a[p + (nmo * q)]) > 1e-6 && g < gap) {
          gap = g;
          pair = {p, q};
        }
      }
    }
    ASSERT_TRUE(pair.has_value()) << what << ": no pair with |A^SC| > 1e-6";
    const auto [p, q] = *pair;
    // The upper of the two onto the lower + 1e-9.
    const std::size_t upper = sc.epsilon[p] > sc.epsilon[q] ? p : q;
    const std::size_t lower = upper == p ? q : p;
    const double pc_before = pc->a[p + (nmo * q)];
    State forced = s;
    const std::array<std::size_t, 1> moved{upper};
    forced.h = shifted_h(s, moved, (sc.epsilon[lower] + 1e-9) - sc.epsilon[upper]);

    const HostAdjoint sc_forced = adjoint(res, forced);
    const auto pc_forced = pc_adjoint_of(res, forced);
    ASSERT_TRUE(pc_forced.has_value()) << what << ": " << pc_forced.error().message;
    const double forced_gap = std::abs(pc_forced->epsilon[p] - pc_forced->epsilon[q]);
    EXPECT_LT(forced_gap, 1e-8) << what;
    EXPECT_LT(std::abs(sc_forced.epsilon[p] - sc_forced.epsilon[q]), 1e-8) << what;
    const double pc_a = pc_forced->a[p + (nmo * q)];
    const double sc_a = sc_forced.a[p + (nmo * q)];
    EXPECT_LE(std::abs(pc_a), 1e-8) << what << ": PC's A at the forced pair";
    EXPECT_GT(std::abs(sc_a), 1e-8) << what << ": SC's A at the forced pair, the control";
    EXPECT_LT(std::abs(pc_a), 1e-3 * std::abs(sc_a)) << what;

    const auto pc_z = multipliers(res, forced, pc_forced->a, pc_forced->epsilon);
    ASSERT_TRUE(pc_z.has_value()) << what << ": " << pc_z.error().message;
    const std::vector<double> z = download(res, pc_z->z.data(), nmo * nmo);
    EXPECT_EQ(z[p + (nmo * q)], 0.0) << what;
    EXPECT_EQ(z[q + (nmo * p)], 0.0) << what;
    EXPECT_LE(pc_z->min_gap, 1e-8) << what;

    const auto sc_z = multipliers(res, forced, sc_forced.a, sc_forced.epsilon);
    ASSERT_FALSE(sc_z.has_value()) << what;
    EXPECT_EQ(sc_z.error().kind, fb::ErrorKind::Numerical) << what;
    const std::string named = std::format("orbitals {} and {}", p, q);
    EXPECT_NE(sc_z.error().message.find(named), std::string::npos) << sc_z.error().message;
    std::cout << std::format("[ nevpt2   ] {} pair ({}, {}), gap {:.3e} -> {:.3e}: PC A {:+.3e} "
                             "-> {:+.3e}, SC A {:+.3e} -> {:+.3e}\n",
                             what, p, q, gap, forced_gap, pc_before, pc_a, sc.a[p + (nmo * q)],
                             sc_a);
  }
}

// z^c symmetric and zero off the core-core and virtual-virtual blocks and on
// the diagonal; A^{z^c} = X - X^T, h_bar^{z^c} = z, every Λ^L symmetric; and
// y^c swap-symmetric in its two strings, as the singlet c is, since
// Σ (W + W^T)_tu E_tu is spin-free.
TEST(Nevpt2Zc, Shapes) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const HostZcAdjoint zadj = zc_adjoint(res, s, z);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  const auto space = [&](const std::size_t p) {
    return p < static_cast<std::size_t>(s.ncore)            ? 0
           : p < static_cast<std::size_t>(s.ncore + s.ncas) ? 1
                                                            : 2;
  };
  double worst_lambda = 0.0;
  for (std::size_t p = 0; p < nmo; ++p) {
    for (std::size_t q = 0; q < nmo; ++q) {
      const double zpq = z[p + (nmo * q)];
      EXPECT_EQ(zpq, z[q + (nmo * p)]);
      if (p == q || space(p) != space(q) || space(p) == 1) {
        EXPECT_EQ(zpq, 0.0) << p << ", " << q;
      }
      EXPECT_EQ(zadj.h_bar[p + (nmo * q)], zpq);
      EXPECT_LT(std::abs(zadj.a[p + (nmo * q)] - (zadj.x[p + (nmo * q)] - zadj.x[q + (nmo * p)])),
                1e-15);
      for (std::size_t l = 0; l < naux; ++l) {
        worst_lambda = std::max(worst_lambda, std::abs(zadj.lambda[l + (naux * (p + (nmo * q)))] -
                                                       zadj.lambda[l + (naux * (q + (nmo * p)))]));
      }
    }
  }
  EXPECT_LT(worst_lambda, 1e-15);
  const auto m = static_cast<std::size_t>(std::llround(std::sqrt(zadj.y.size())));
  ASSERT_EQ(m * m, zadj.y.size());
  double worst_y = 0.0;
  double largest_y = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      worst_y = std::max(worst_y, std::abs(zadj.y[(i * m) + j] - zadj.y[(j * m) + i]));
      largest_y = std::max(largest_y, std::abs(zadj.y[(i * m) + j]));
    }
  }
  EXPECT_GT(largest_y, 0.0);
  EXPECT_LT(worst_y, 1e-14 * largest_y);
}

// ---- Φ over a state average ----------------------------------------------
//
// Φ = <z^c, f(γ_SA)>, γ_SA = Σ_r w_r γ_r, over two roots: the state's c and
// ci_direction's u, a unit vector in c's spin and symmetry sector
// perpendicular to it. Φ is a function of γ_SA alone, so the roots need not
// be eigenvectors. z^c is the single-state one, fixed.

struct SaWeights {
  std::string_view name;
  std::array<double, 2> w;
};

void PrintTo(const SaWeights &w, std::ostream *os) {
  *os << w.name;
}

constexpr std::array<SaWeights, 2> kSaWeights{
    {{.name = "equal", .w = {0.5, 0.5}}, {.name = "unequal", .w = {0.7, 0.3}}}};

// Each of Φ's adjoints against a ladder of Φ at fixed z^c and weights: along
// h and B (the bent path of Nevpt2ZcIntegrals), along C exp(t κ) with κ every
// block's at once, and along each root c_r(t) = (c_r + t d_r) / |c_r + t d_r|
// alone, d_0 = u and d_1 = c (each perpendicular to its root), against
// <d_r, y^c_r>.
class Nevpt2ZcStateAverage : public ::testing::TestWithParam<SaWeights> {};

TEST_P(Nevpt2ZcStateAverage, AdjointsMatchLadders) {
  FBF_REQUIRE_GPU();
  const SaWeights &weights = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const std::vector<double> u = ci_direction(res, s);
  const std::size_t ndet = s.vector.size();
  ASSERT_EQ(u.size(), ndet);
  std::vector<double> roots = s.vector;
  roots.insert(roots.end(), u.begin(), u.end());
  const std::span<const double> w = weights.w;
  const HostZcAdjoint zadj = zc_adjoint(res, s, z, w, roots);
  ASSERT_EQ(zadj.y.size(), 2 * ndet);
  const auto nmo = static_cast<std::size_t>(s.nao);

  for (const std::string which : {"h", "B"}) {
    const IntegralDirection dir = integral_direction(s, which);
    State moved = s;
    expect_ladder(
        std::format("sa {} phi {}", weights.name, which),
        [&](const double t) {
          perturb(s, dir, t * (1.0 + ((t * t) / (kLadderT0 * kLadderT0))), moved);
          return phi(res, moved, z, w, roots);
        },
        integral_dot(s, dir, zadj.h_bar, zadj.lambda));
  }

  std::vector<double> kappa(nmo * nmo, 0.0);
  for (const Block &block : kBlocks) {
    const std::vector<double> k = block_kappa(s, block);
    for (std::size_t i = 0; i < kappa.size(); ++i) {
      kappa[i] += k[i];
    }
  }
  {
    State moved = s;
    expect_ladder(
        std::format("sa {} phi C", weights.name),
        [&](const double t) {
          rotate(s, kappa, t, moved);
          return phi(res, moved, z, w, roots);
        },
        source_dot(kappa, zadj.a, nmo));
  }

  for (std::size_t r = 0; r < 2; ++r) {
    const std::span<const double> root = std::span<const double>(roots).subspan(r * ndet, ndet);
    const std::span<const double> d = r == 0 ? std::span<const double>(u) : s.vector;
    ASSERT_LT(std::abs(dot(root, d)), 1e-12);
    std::vector<double> moved = roots;
    expect_ladder(
        std::format("sa {} phi c_{}", weights.name, r),
        [&](const double t) {
          const auto out = std::span<double>(moved).subspan(r * ndet, ndet);
          for (std::size_t i = 0; i < ndet; ++i) {
            out[i] = root[i] + (t * d[i]);
          }
          const double length = std::sqrt(dot(out, out));
          for (double &v : out) {
            v /= length;
          }
          return phi(res, s, z, w, moved);
        },
        dot(d, std::span<const double>(zadj.y).subspan(r * ndet, ndet)));
  }
}

INSTANTIATE_TEST_SUITE_P(Weights, Nevpt2ZcStateAverage, ::testing::ValuesIn(kSaWeights),
                         [](const auto &info) { return std::string(info.param.name); });

// The one-root call at weight 1 is the single-state Φ: against a second
// call of itself, and against a state average at weights (1, 0) over (c, u),
// whose γ_SA is γ_c + 0 γ_u and whose root-1 y^c is scaled by 0. h̄ is z
// itself, so it is equal to the bit; root 1's y^c is exactly 0. The rest
// cannot be bitwise: the single-root zc_adjoint before the state average was
// not bitwise against itself, its einsums' atomics reassociating each sum
// (on the gfx1200, five calls against the first: Φ apart by up to 1.4e-18,
// Λ 2.6e-18, y^c 3.5e-18, X and A 5.2e-18), and the one-root call against
// it sat at the same spread (at most 6.9e-18). An absolute 1e-17 is under
// that floor: 50 repeats of this case on the gfx1200 reached 1.39e-17 on A,
// one ulp of its largest element. So each array is held to
// kRunToRunEps of its largest element's machine epsilon (the 50 repeats
// reached 10.7 on A, 4.0 on y^c, 2.5 on X, 2.4 on Λ), and Φ, which is
// round-off itself (f is diagonal where z lives, so Φ ~ 1e-17), to
// kRunToRunPhi absolute (the repeats reached 6.2e-18).
constexpr double kRunToRunEps = 32.0;
constexpr double kRunToRunPhi = 5e-17;

TEST(Nevpt2Zc, OneRootIsTheSingleStatePhi) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const std::vector<double> z = zc_of(res, s, adj);
  ASSERT_FALSE(z.empty());
  const std::vector<double> u = ci_direction(res, s);
  const std::size_t ndet = s.vector.size();
  ASSERT_EQ(u.size(), ndet);
  std::vector<double> roots = s.vector;
  roots.insert(roots.end(), u.begin(), u.end());
  const std::array<double, 2> first_only{1.0, 0.0};

  const HostZcAdjoint one = zc_adjoint(res, s, z);
  const HostZcAdjoint again = zc_adjoint(res, s, z);
  const HostZcAdjoint averaged = zc_adjoint(res, s, z, first_only, roots);
  ASSERT_EQ(one.y.size(), ndet);
  ASSERT_EQ(averaged.y.size(), 2 * ndet);
  // max |a - b| in units of b's largest magnitude's machine epsilon.
  const auto worst = [](std::span<const double> a, std::span<const double> b) {
    double out = 0.0;
    double largest = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      out = std::max(out, std::abs(a[i] - b[i]));
      largest = std::max(largest, std::abs(b[i]));
    }
    return out / (std::numeric_limits<double>::epsilon() * largest);
  };
  for (const auto &[name, other] :
       {std::pair{"again", &again}, std::pair{"weights (1, 0)", &averaged}}) {
    EXPECT_EQ(other->h_bar, one.h_bar) << name;
    const std::array<std::pair<const char *, double>, 5> apart{
        {{"phi", std::abs(other->phi - one.phi)},
         {"lambda", worst(other->lambda, one.lambda)},
         {"y", worst(std::span<const double>(other->y).first(ndet), one.y)},
         {"x", worst(other->x, one.x)},
         {"a", worst(other->a, one.a)}}};
    for (const auto &[field, difference] : apart) {
      EXPECT_LE(difference, std::string_view{field} == "phi" ? kRunToRunPhi : kRunToRunEps)
          << name << ": " << field;
      std::cout << std::format("[ nevpt2   ] one root, {} {}: apart by {:.3e}{}\n", name, field,
                               difference, std::string_view{field} == "phi" ? "" : " eps max");
    }
  }
  for (std::size_t i = ndet; i < 2 * ndet; ++i) {
    EXPECT_EQ(averaged.y[i], 0.0) << i;
  }
}

// zc_adjoint refuses weights it cannot average over: none, a negative one,
// and a sum off 1.
TEST(Nevpt2Zc, RefusesBadWeights) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::vector<double> z(nmo * nmo, 0.0);
  const DeviceState d = upload_state(res, s);
  const fb::Buffers::device<double> z_d = upload(res, z);
  std::vector<double> roots = s.vector;
  roots.insert(roots.end(), s.vector.begin(), s.vector.end());
  const fb::Buffers::device<double> roots_d = upload(res, roots);
  const std::array<double, 2> negative{1.5, -0.5};
  const std::array<double, 2> short_sum{0.5, 0.4};
  for (const std::span<const double> weights :
       {std::span<const double>{}, std::span<const double>(negative),
        std::span<const double>(short_sum)}) {
    const auto got = zc_at(res, s, d, z_d, weights, roots_d);
    ASSERT_FALSE(got.has_value());
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << got.error().message;
  }
}

// ---- The nuclear gradient: lagrangian_gradient and sc_gradient ----------
//
// On test/gtest/data/nevpt2_gradient/'s cases (skew H2O CAS(4,4) and H2CO
// CAS(6,6), cc-pVDZ / cc-pVDZ-JKFIT), each this tree's own DF-CASSCF from
// the case's start through the registry: the SC-NEVPT2 gradient against the
// numerical one (Richardson over re-converged energies) per component; the
// identical pipeline with E_corr's adjoint and z^c zeroed against the
// DF-CASSCF's own gradient; and the full Lagrangian's stationarity along a
// non-redundant rotation and a CI direction.

// The issue's bars: each component within 1e-6 Eh/a0 of the numerical
// gradient (itself a difference, so not the tree's usual 1e-7), the sum
// over atoms below 1e-9, and X^L symmetric per block to ~1e-9.
constexpr double kNumericalTolerance = 1e-6;
constexpr double kTranslationTolerance = 1e-9;
constexpr double kAsymmetryTolerance = 1e-9;
// The CASSCF energy against the reference's, as mcscf_gradient_gpu.cpp
// holds it, and with E_corr zeroed the gradient against PySCF's analytic
// DF-CASSCF one to that file's 1e-8.
constexpr double kCasscfEnergyTolerance = 1e-8;
constexpr double kAnalyticTolerance = 1e-8;
// With E_corr zeroed the right-hand side is the CASSCF's own orbital
// gradient g, below kConvTolGrad: the multipliers are that small, not zero
// (the skew H2O's max |Z| 4.1e-10 on an RTX 3080; H2CO's g is under
// zvector's tolerance, and its Z is exactly 0). So the Lagrangian differs
// from df_gradient's own single-root formula, which drops Z, by terms first
// order in g: the skew H2O's 1.5e-10 on a component. That is
// mcscf_gradient_gpu.cpp's reasoning for its kConvTolGrad, and the bar.
constexpr double kVanishTolerance = 1e-8;
constexpr double kZeroedTolerance = kConvTolGrad;

const fb::json &gradient_case(std::string_view name) {
  return reference_case("nevpt2_gradient", "nevpt2_gradient.json", name);
}

fb::Buffers::device<double> zeros(fb::DeviceResources &res, std::size_t count) {
  return upload(res, std::vector<double>(count, 0.0));
}

// One case's run: the registry's products (the Calculation, kept alive in
// its node), the state in one gauge on the host, and the mcscf::Solution of
// that state, its RDMs taken from the gauge-fixed vector.
struct NuclearRun {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  State state;
  std::optional<fb::mcscf::Solution> solution;
};

NuclearRun nuclear_run(const fb::json &entry) {
  NuclearRun run{.node = std::make_unique<fb::NodeResources>(),
                 .calc = std::nullopt,
                 .state = {},
                 .solution = std::nullopt};
  // A Calculation moves but does not assign.
  std::optional<fb::Calculation> calc = run_casscf(entry, *run.node);
  if (!calc.has_value()) {
    return run;
  }
  run.calc.emplace(std::move(*calc));
  fb::DeviceResources &res = run.node->device(0);
  run.state = host_state(*run.calc, res);
  const State &s = run.state;
  const auto &casscf = run.calc->get<fb::CasscfOutput>();
  const auto &basis = run.calc->get<fb::BasisSetOutput>();
  if (s.b.empty() || !casscf.has_value() || !basis.has_value()) {
    return run;
  }
  EXPECT_NEAR(casscf->energy, entry.at("energies").at("casscf").get<double>(),
              kCasscfEnergyTolerance);
  EXPECT_EQ(s.ncore, entry.at("ncore").get<std::int32_t>());
  const auto n = static_cast<std::size_t>(s.ncas);
  auto strings = fb::ci::DeviceStringSet::create(res, s.ncas, s.nelecas / 2);
  EXPECT_TRUE(strings.has_value()) << strings.error().message;
  if (!strings.has_value()) {
    return run;
  }
  auto rdm = fb::ci::Rdm::create(res, *strings, *strings);
  EXPECT_TRUE(rdm.has_value()) << rdm.error().message;
  if (!rdm.has_value()) {
    return run;
  }
  fb::Buffers::device<double> vector = upload(res, s.vector);
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  const std::array<double, 1> one{1.0};
  EXPECT_TRUE(rdm->rdm12(res, one, vector.data(), gamma.data(), gamma2.data()).has_value());
  std::vector<double> g1 = download(res, gamma.data(), n * n);
  run.solution.emplace(fb::mcscf::Solution{.c = fb::SphericalMatrixAO(res, basis->basis),
                                           .roots = {.energies = casscf->energies,
                                                     .spin_squares = casscf->spin_squares,
                                                     .vectors = std::move(vector),
                                                     .davidson = {},
                                                     .fp32_iterations = 0},
                                           .energy = casscf->energy,
                                           .gradient_norm = casscf->gradient_norm,
                                           .gamma = g1,
                                           .gamma2 = download(res, gamma2.data(), n * n * n * n),
                                           .root_gammas = g1,
                                           .iterations = casscf->iterations});
  EXPECT_EQ(wwr::wwrMemcpyAsync(run.solution->c.data(), s.c.data(), s.c.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return run;
}

// The registry's products df_gradient takes, by reference into the run.
struct GradientInputs {
  const fb::Molecule &molecule;
  const fb::BasisSet &basis;
  std::span<const fb::PrimitivePairClass> classes;
  const fb::DensityFitted &df;
  const fb::SphericalMatrixAO &h;
};

GradientInputs gradient_inputs(NuclearRun &run) {
  return {.molecule = run.calc->get<fb::MoleculeOutput>()->molecule,
          .basis = run.calc->get<fb::BasisSetOutput>()->basis,
          .classes = run.calc->get<fb::PrimitivePairClassesOutput>()->classes,
          .df = std::get<fb::DensityFitted>(run.calc->get<fb::EriOutput>()->eri),
          .h = run.calc->get<fb::CoreHamiltonianOutput>()->h};
}

fb::Result<nevpt2::ScGradient> sc_gradient_of(NuclearRun &run) {
  const GradientInputs in = gradient_inputs(run);
  return nevpt2::sc_gradient(run.node->device(0), in.molecule, in.basis, in.classes, in.df.aux,
                             in.h, in.df.b, in.df.l.data(), *run.solution, run.state.ncore,
                             run.state.ncas, 0);
}

fb::Result<nevpt2::PcGradient> pc_gradient_of(NuclearRun &run, const fb::SphericalMatrixAO &h) {
  const GradientInputs in = gradient_inputs(run);
  return nevpt2::pc_gradient(run.node->device(0), in.molecule, in.basis, in.classes, in.df.aux, h,
                             in.df.b, in.df.l.data(), *run.solution, run.state.ncore,
                             run.state.ncas, 0);
}

double largest_abs(std::span<const double> v) {
  double largest = 0.0;
  for (const double x : v) {
    largest = std::max(largest, std::abs(x));
  }
  return largest;
}

// The worst |sum over atoms| of each Cartesian component.
double translation(std::span<const double> g) {
  std::array<double, 3> sum{};
  for (std::size_t i = 0; i < g.size(); ++i) {
    sum[i % 3] += g[i];
  }
  return std::abs(std::ranges::max(sum, {}, [](double v) { return std::abs(v); }));
}

constexpr std::array<std::string_view, 6> kAsymmetryBlocks{"core-core",       "active-active",
                                                           "virtual-virtual", "active-core",
                                                           "virtual-core",    "virtual-active"};

void expect_symmetric(const std::string &label, const std::array<double, 6> &asymmetry) {
  std::string report;
  for (std::size_t k = 0; k < asymmetry.size(); ++k) {
    EXPECT_LT(asymmetry[k], kAsymmetryTolerance) << label << " X^L " << kAsymmetryBlocks[k];
    report += std::format(" {} {:.2e}", kAsymmetryBlocks[k], asymmetry[k]);
  }
  std::cout << std::format("[ nevpt2   ] {} X^L asymmetry:{}\n", label, report);
}

class Nevpt2NuclearGradient : public ::testing::TestWithParam<std::string_view> {};

// The SC-NEVPT2 gradient against the numerical one per component, its sum
// over atoms, X^L's symmetry, E_corr against PySCF's at the same state, and
// a state average refused.
TEST_P(Nevpt2NuclearGradient, MatchesNumerical) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const auto start = std::chrono::steady_clock::now();
  const fb::json &entry = gradient_case(name);
  NuclearRun run = nuclear_run(entry);
  ASSERT_TRUE(run.solution.has_value());
  const auto setup = std::chrono::steady_clock::now();
  const auto g = sc_gradient_of(run);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const std::chrono::duration<double> setup_seconds = setup - start;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - setup;

  EXPECT_NEAR(g->energies.total, entry.at("e_corr").at("sc").get<double>(), kTolerance);
  const fb::json &want = entry.at("gradients").at("sc");
  const fb::json &rows = want.at("richardson");
  const fb::json &bars = want.at("error_bar");
  const std::vector<double> &total = g->terms.total;
  ASSERT_EQ(total.size(), 3 * rows.size());
  std::cout << std::format("[ nevpt2   ] {} SC gradient (setup {:.2f} s, gradient {:.2f} s): "
                           "E_corr {:.12f} ({:+.2e}), max |z^c| {:.3e}, smallest inactive gap "
                           "{:.3e}, PCG {} iterations, residual {:.2e}\n",
                           name, setup_seconds.count(), seconds.count(), g->energies.total,
                           g->energies.total - entry.at("e_corr").at("sc").get<double>(),
                           g->zc.max_z, g->zc.min_gap, g->multipliers.iterations,
                           g->multipliers.residual);
  double worst = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double ours = total[(3 * a) + k];
      const double ref = rows[a][k].get<double>();
      EXPECT_NEAR(ours, ref, kNumericalTolerance) << name << " [" << a << "][" << k << "]";
      worst = std::max(worst, std::abs(ours - ref));
      std::cout << std::format(
          "[ nevpt2   ] {} {} {}: analytic {:+.10f} numerical {:+.10f} "
          "diff {:+.2e} (error bar {:.1e})\n",
          name, entry.at("atoms")[a].at("symbol").get<std::string>() + std::to_string(a), "xyz"[k],
          ours, ref, ours - ref, bars[a][k].get<double>());
    }
  }
  EXPECT_LT(translation(total), kTranslationTolerance) << name;
  std::cout << std::format("[ nevpt2   ] {} worst component {:.2e}, translation {:.2e}\n", name,
                           worst, translation(total));
  expect_symmetric(name, g->asymmetry);

  // A state average is refused.
  run.solution->roots.energies.push_back(run.solution->roots.energies.front());
  const auto averaged = sc_gradient_of(run);
  ASSERT_FALSE(averaged.has_value());
  EXPECT_EQ(averaged.error().kind, fb::ErrorKind::Unsupported) << averaged.error().message;
}

// The identical pipeline with E_corr's adjoint and z^c zeroed: the
// multipliers vanish (to the CASSCF's own convergence) and the gradient is
// the DF-CASSCF's, df_gradient's own and PySCF's analytic one.
TEST_P(Nevpt2NuclearGradient, MultipliersVanishWithoutCorrelation) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = gradient_case(name);
  NuclearRun run = nuclear_run(entry);
  ASSERT_TRUE(run.solution.has_value());
  fb::DeviceResources &res = run.node->device(0);
  const GradientInputs in = gradient_inputs(run);
  const State &s = run.state;
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::size_t nmo2 = nmo * nmo;
  const std::size_t lambda = static_cast<std::size_t>(s.naux) * nmo2;
  const std::size_t dim = s.vector.size();
  nevpt2::ScAdjoint corr{.energies = {},
                         .h_bar = zeros(res, nmo2),
                         .lambda = zeros(res, lambda),
                         .y = zeros(res, dim),
                         .x = zeros(res, nmo2),
                         .a = zeros(res, nmo2),
                         .epsilon = zeros(res, nmo)};
  nevpt2::ZcMultipliers zc{.z = zeros(res, nmo2)};
  const nevpt2::ZcAdjoint phi{.phi = 0.0,
                              .h_bar = zeros(res, nmo2),
                              .lambda = zeros(res, lambda),
                              .y = zeros(res, dim),
                              .x = zeros(res, nmo2),
                              .a = zeros(res, nmo2)};
  const auto g = nevpt2::lagrangian_gradient(res, in.molecule, in.basis, in.classes, in.df.aux,
                                             in.h, in.df.b, in.df.l.data(), *run.solution, s.ncore,
                                             s.ncas, std::move(corr), std::move(zc), phi);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const double z_orbital = largest_abs(
      download(res, g->multipliers.orbital.data(), g->multipliers.orbital.num_elements()));
  const double z_ci =
      largest_abs(download(res, g->multipliers.ci.data(), g->multipliers.ci.num_elements()));
  EXPECT_LT(z_orbital, kVanishTolerance) << name;
  EXPECT_LT(z_ci, kVanishTolerance) << name;

  const auto own =
      fb::mcscf::df_gradient(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                             in.df.l.data(), *run.solution, s.ncore, s.ncas, 0);
  ASSERT_TRUE(own.has_value()) << own.error().message;
  const fb::json &analytic = entry.at("casscf_gradient");
  double worst_own = 0.0;
  double worst_analytic = 0.0;
  for (std::size_t i = 0; i < g->terms.total.size(); ++i) {
    const double ours = g->terms.total[i];
    const double pyscf = analytic[i / 3][i % 3].get<double>();
    EXPECT_NEAR(ours, own->total[i], kZeroedTolerance) << name << " " << i;
    EXPECT_NEAR(ours, pyscf, kAnalyticTolerance) << name << " " << i;
    worst_own = std::max(worst_own, std::abs(ours - own->total[i]));
    worst_analytic = std::max(worst_analytic, std::abs(ours - pyscf));
  }
  std::cout << std::format("[ nevpt2   ] {} E_corr zeroed: max |Z| {:.2e}, max |z| {:.2e}, PCG "
                           "{} iterations; against df_gradient {:.2e}, against PySCF {:.2e}\n",
                           name, z_orbital, z_ci, g->multipliers.iterations, worst_own,
                           worst_analytic);
  expect_symmetric(name + " E_corr zeroed", g->asymmetry);
}

// The PC-NEVPT2 gradient against the numerical one (gradients.pc, block2's
// WickICNEVPT2 over PySCF's fitted integrals) per component to 1e-6, its
// sum over atoms, X^L's symmetry, E_corr against block2's at the same
// state, and a state average refused. PC's z^c is not round-off here: the
// inactive orbital energies are distinct, and PC is invariant to an
// inactive rotation only between equal ones (Nevpt2PcZc).
TEST_P(Nevpt2NuclearGradient, PcMatchesNumerical) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const auto start = std::chrono::steady_clock::now();
  const fb::json &entry = gradient_case(name);
  NuclearRun run = nuclear_run(entry);
  ASSERT_TRUE(run.solution.has_value());
  const auto setup = std::chrono::steady_clock::now();
  const GradientInputs in = gradient_inputs(run);
  const auto g = pc_gradient_of(run, in.h);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  const std::chrono::duration<double> setup_seconds = setup - start;
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - setup;

  const double e_ref = entry.at("e_corr").at("pc").get<double>();
  EXPECT_NEAR(g->energies.total, e_ref, kTolerance);
  const fb::json &want = entry.at("gradients").at("pc");
  const fb::json &rows = want.at("richardson");
  const fb::json &bars = want.at("error_bar");
  const std::vector<double> &total = g->terms.total;
  ASSERT_EQ(total.size(), 3 * rows.size());
  std::cout << std::format("[ nevpt2   ] {} PC gradient (setup {:.2f} s, gradient {:.2f} s): "
                           "E_corr {:.12f} ({:+.2e}), max |z^c| {:.3e}, smallest inactive gap "
                           "{:.3e}, PCG {} iterations, residual {:.2e}\n",
                           name, setup_seconds.count(), seconds.count(), g->energies.total,
                           g->energies.total - e_ref, g->zc.max_z, g->zc.min_gap,
                           g->multipliers.iterations, g->multipliers.residual);
  double worst = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      const double ours = total[(3 * a) + k];
      const double ref = rows[a][k].get<double>();
      EXPECT_NEAR(ours, ref, kNumericalTolerance) << name << " PC [" << a << "][" << k << "]";
      worst = std::max(worst, std::abs(ours - ref));
      std::cout << std::format(
          "[ nevpt2   ] {} PC {} {}: analytic {:+.10f} numerical {:+.10f} "
          "diff {:+.2e} (error bar {:.1e})\n",
          name, entry.at("atoms")[a].at("symbol").get<std::string>() + std::to_string(a), "xyz"[k],
          ours, ref, ours - ref, bars[a][k].get<double>());
    }
  }
  EXPECT_LT(translation(total), kTranslationTolerance) << name;
  std::cout << std::format("[ nevpt2   ] {} PC worst component {:.2e}, translation {:.2e}\n", name,
                           worst, translation(total));
  expect_symmetric(name + " PC", g->asymmetry);

  run.solution->roots.energies.push_back(run.solution->roots.energies.front());
  const auto averaged = pc_gradient_of(run, in.h);
  ASSERT_FALSE(averaged.has_value());
  EXPECT_EQ(averaged.error().kind, fb::ErrorKind::Unsupported) << averaged.error().message;
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2NuclearGradient,
                         ::testing::Values(std::string_view{"h2o_skew_ccpvdz"},
                                           std::string_view{"h2co_ccpvdz"}),
                         [](const auto &info) { return std::string(info.param); });

// A state PC refuses is refused by pc_gradient with pc_energy's error, kind
// and message, never a gradient with the class summed in as zero. Two
// states, after nevpt2_energy_gpu.cpp's: a zero CI vector, whose RDMs and so
// Sr's metric are zero ("no positive eigenvalue"); and every virtual orbital
// energy lowered by 1000 Eh through h (shifted_h), so that Sr's denominator
// λ + ε_r is negative ("not positive").
TEST(Nevpt2PcGradient, RefusesWhatTheEnergyRefuses) {
  FBF_REQUIRE_GPU();
  NuclearRun run = nuclear_run(gradient_case("h2o_skew_ccpvdz"));
  ASSERT_TRUE(run.solution.has_value());
  fb::DeviceResources &res = run.node->device(0);
  const GradientInputs in = gradient_inputs(run);
  const State &s = run.state;
  const auto nmo = static_cast<std::size_t>(s.nao);

  std::vector<std::size_t> virtuals;
  for (auto p = static_cast<std::size_t>(s.ncore + s.ncas); p < nmo; ++p) {
    virtuals.push_back(p);
  }
  State low = s;
  low.h = shifted_h(s, virtuals, -1000.0);
  fb::SphericalMatrixAO h_low(res, in.basis);
  ASSERT_EQ(wwr::wwrMemcpyAsync(h_low.data(), low.h.data(), low.h.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  {
    const auto energy = pc_energy_of(res, low);
    ASSERT_FALSE(energy.has_value()) << "virtuals at -1000 Eh";
    const auto g = pc_gradient_of(run, h_low);
    ASSERT_FALSE(g.has_value()) << "virtuals at -1000 Eh";
    EXPECT_EQ(g.error().kind, fb::ErrorKind::Numerical);
    EXPECT_EQ(g.error().kind, energy.error().kind);
    EXPECT_EQ(g.error().message, energy.error().message);
    EXPECT_NE(g.error().message.find("Sr"), std::string::npos) << g.error().message;
    EXPECT_NE(g.error().message.find("not positive"), std::string::npos) << g.error().message;
  }

  State zero = s;
  std::ranges::fill(zero.vector, 0.0);
  ASSERT_EQ(wwr::wwrMemcpyAsync(run.solution->roots.vectors.data(), zero.vector.data(),
                                zero.vector.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  const auto energy = pc_energy_of(res, zero);
  ASSERT_FALSE(energy.has_value()) << "zero vector";
  const auto g = pc_gradient_of(run, in.h);
  ASSERT_FALSE(g.has_value()) << "zero vector";
  EXPECT_EQ(g.error().kind, fb::ErrorKind::Numerical);
  EXPECT_EQ(g.error().kind, energy.error().kind);
  EXPECT_EQ(g.error().message, energy.error().message);
  EXPECT_NE(g.error().message.find("Sr"), std::string::npos) << g.error().message;
  EXPECT_NE(g.error().message.find("no positive eigenvalue"), std::string::npos)
      << g.error().message;
}

// The full Lagrangian L = E_cas + E_corr + <z^c, f> + Z . g_orb + z . g_ci
// at the moved state @p moved, the multipliers fixed: g_orb response::fock's
// gradient over the packed pairs, g_ci = 2 (H_0 - E) c in zvector's
// coordinates, every input rebuilt from (C, c) and nothing re-canonicalized.
// As {E_corr, the rest}, each kept apart for a ladder of its own.
std::array<double, 2> lagrangian(NuclearRun &run, const State &moved,
                                 std::span<const double> z_orbital, std::span<const double> z_ci,
                                 std::span<const double> zc) {
  fb::DeviceResources &res = run.node->device(0);
  const GradientInputs in = gradient_inputs(run);
  const auto n = static_cast<std::size_t>(moved.ncas);
  const DeviceState d = upload_state(res, moved);
  fb::SphericalMatrixAO c(res, in.basis);
  EXPECT_EQ(wwr::wwrMemcpyAsync(c.data(), moved.c.data(), moved.c.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  auto ints = fb::casscf_integrals(res, in.basis, in.molecule, in.h, in.df.b, moved.naux, c,
                                   moved.ncore, moved.ncas);
  EXPECT_TRUE(ints.has_value()) << ints.error().message;
  auto rdm = fb::ci::Rdm::create(res, *d.strings, *d.strings);
  auto sigma = fb::ci::Sigma::create(res, *d.strings, *d.strings);
  EXPECT_TRUE(rdm.has_value() && sigma.has_value());
  if (!ints.has_value() || !rdm.has_value() || !sigma.has_value()) {
    return {std::nan(""), std::nan("")};
  }
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  const std::array<double, 1> one{1.0};
  EXPECT_TRUE(rdm->rdm12(res, one, d.vector.data(), gamma.data(), gamma2.data()).has_value());
  auto fock = fb::response::fock(
      res, in.molecule, in.h, in.df.b, moved.naux, c, moved.ncore, moved.ncas, ints->ppaa,
      ints->papa, download(res, gamma.data(), n * n), download(res, gamma2.data(), n * n * n * n));
  EXPECT_TRUE(fock.has_value()) << fock.error().message;
  if (!fock.has_value()) {
    return {std::nan(""), std::nan("")};
  }
  const std::vector<double> g_orb = download(res, fock->gradient.data(), z_orbital.size());
  EXPECT_TRUE(sigma
                  ->set_integrals(res, ints->hamiltonian.e_core,
                                  download(res, ints->hamiltonian.h_eff.data(), n * n),
                                  download(res, ints->hamiltonian.eri.data(), n * n * n * n))
                  .has_value());
  fb::Buffers::device<double> hc(moved.vector.size(), res.device());
  EXPECT_TRUE(sigma->apply(res, 1, d.vector.data(), hc.data()).has_value());
  const std::vector<double> h_c = download(res, hc.data(), moved.vector.size());
  const double e = dot(moved.vector, h_c);
  const double ci = 2.0 * (dot(z_ci, h_c) - (e * dot(z_ci, moved.vector)));
  return {energy(res, moved), fock->energy + phi(res, moved, zc) + dot(z_orbital, g_orb) + ci};
}

// expect_ladder over L = E_corr + the rest (@p at's two parts), predicting 0
// against @p slope, each part's derivative extrapolated by a ladder of its
// own and the two extrapolants summed. One ladder over L rarely sees a run:
// E_corr is still far from its O(t²) regime (orbitals: 9.9 5.6 4.9 4.8 4.4
// 4.1) where the rest, at 4.000 from the first rung, has reached the
// rounding of E_cas ≈ −76 Eh (an ulp over 2t is 3.6e-10 at the finest
// rung), so L's ratios (12.4 6.8 5.8 6.0 5.2 4.40 4.06, then 4.3-4.9) held
// three in (3.2, 4.8) by chance and fell short 4 runs in 50 on the gfx1200.
// Apart, each part has a run of four or more, and the extrapolant is linear
// in D, so the sum is L's. Each is taken where its extrapolants are
// steadiest (richardson_ladder): at the finest pair the rest's reads
// E_cas's rounding (up to 8.7e-8 of the slope over 50 runs, against 1e-7),
// at its coarsest its t⁴ truncation (3.6e-7). Steadiest, the worst of 50
// runs on the gfx1200 was 1.4e-8 (orbitals) and 2.5e-9 (CI), of 20 on the
// RTX 3080 1.7e-8 and 1.2e-9.
void expect_stationary(const std::string &name,
                       const std::function<std::array<double, 2>(double)> &at, const double slope) {
  constexpr std::array<std::string_view, 2> kParts{"E_corr", "rest"};
  std::array<std::vector<double>, 2> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(kLadderT0, -k);
    const std::array<double, 2> plus = at(t);
    const std::array<double, 2> minus = at(-t);
    for (std::size_t part = 0; part < d.size(); ++part) {
      d[part].push_back((plus[part] - minus[part]) / (2.0 * t));
    }
  }
  double richardson = 0.0;
  std::string ladders;
  for (std::size_t part = 0; part < d.size(); ++part) {
    const Ladder l = richardson_ladder(d[part], true);
    ASSERT_GE(l.run, 3) << name << " " << kParts[part] << " ratios:" << l.ratios;
    richardson += l.richardson;
    ladders += std::format(" {}:{}", kParts[part], l.ratios);
  }
  EXPECT_NEAR(richardson, 0.0, kLadderTolerance * slope) << name << ladders;
  std::cout << std::format("[ nevpt2   ] stationarity {}: Σ Richardson {:.12e}, relative {:.2e},"
                           "{}\n",
                           name, richardson, std::abs(richardson) / slope, ladders);
}

// The full Lagrangian flat at first order along a random rotation of the
// non-redundant pairs (core-active, core-virtual, active-virtual at once),
// C exp(t kappa), and along ci_direction's u perpendicular to c: each
// direction's summed extrapolant (expect_stationary) within 1e-7 of the
// slope of E_corr + <z^c, f> it cancels.
TEST(Nevpt2NuclearGradient, LagrangianIsStationary) {
  FBF_REQUIRE_GPU();
  NuclearRun run = nuclear_run(gradient_case("h2o_skew_ccpvdz"));
  ASSERT_TRUE(run.solution.has_value());
  const auto g = sc_gradient_of(run);
  ASSERT_TRUE(g.has_value()) << g.error().message;
  fb::DeviceResources &res = run.node->device(0);
  const State &s = run.state;
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::vector<double> z_orbital =
      download(res, g->multipliers.orbital.data(), g->multipliers.orbital.num_elements());
  const std::vector<double> z_ci =
      download(res, g->multipliers.ci.data(), g->multipliers.ci.num_elements());
  const std::vector<double> zc = download(res, g->zc.z.data(), nmo * nmo);
  const HostAdjoint adj = adjoint(res, s);
  const HostZcAdjoint zadj = zc_adjoint(res, s, zc);

  std::vector<double> kappa(nmo * nmo, 0.0);
  for (const Block &block : {kBlocks[3], kBlocks[4], kBlocks[5]}) {
    const std::vector<double> part = block_kappa(s, block);
    for (std::size_t i = 0; i < kappa.size(); ++i) {
      kappa[i] += part[i];
    }
  }
  const double orbital_slope = source_dot(kappa, adj.a, nmo) + source_dot(kappa, zadj.a, nmo);
  ASSERT_GT(std::abs(orbital_slope), 0.0);
  State moved = s;
  expect_stationary(
      "full lagrangian orbitals",
      [&](const double t) {
        rotate(s, kappa, t, moved);
        return lagrangian(run, moved, z_orbital, z_ci, zc);
      },
      std::abs(orbital_slope));

  const std::vector<double> u = ci_direction(res, s);
  ASSERT_EQ(u.size(), s.vector.size());
  const double ci_slope = dot(u, adj.y) + dot(u, zadj.y);
  ASSERT_GT(std::abs(ci_slope), 0.0);
  moved = s;
  expect_stationary(
      "full lagrangian ci",
      [&](const double t) {
        move_vector(s, u, t, moved);
        return lagrangian(run, moved, z_orbital, z_ci, zc);
      },
      std::abs(ci_slope));
}

// ---- SC-QD-NEVPT2: qd_sc_energy and qd_sc_adjoint -----------------------
//
// On data/qd_nevpt2/'s cases (#670), each this tree's own DF-SA-CASSCF from
// the case's start through the registry, as pipeline_nevpt2_gpu.cpp's
// PipelineQdNevpt2 runs it, put in one gauge (every root's vector carried
// along). E[W] = Σ_MN W_MN H_MN of qd_sc_energy's elements, every input
// rebuilt from (h, B, C, c_1 ... c_n) at each point, the state-averaged
// orbital energies included, against qd_sc_adjoint's adjoints by the
// gradient ladders' rule, at two W: u_K u_Kᵀ, K the highest state of the
// symmetrized H_eff, and a seeded random symmetric W.

// A case's SA-CASSCF: its state (State::vector every root, root-major), the
// weights, and the CASSCF's root energies and SA-Fock orbital energies.
struct QdState {
  State s;
  std::vector<double> weights;
  std::vector<double> energies;
  std::vector<double> mo_energies;
};

// This tree's DF-SA-CASSCF of qd_nevpt2.json's @p entry through the real
// registry, in @p node: pipeline_nevpt2_gpu.cpp's write_qd_config, with no
// Nevpt2 step.
std::optional<fb::Calculation> run_qd_casscf(const fb::json &entry, fb::NodeResources &node) {
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_gradient_qd");
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : entry.at("weights").get<std::vector<double>>()) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n",
      entry.at("basis").get<std::string>(), kDfAuxBasis, entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), entry.at("weights").size(), weights, kConvTolGrad);
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

// The case @p name's SA-CASSCF, run once per process.
const QdState &qd_state(const std::string &name) {
  static std::map<std::string, QdState, std::less<>> states;
  if (const auto it = states.find(name); it != states.end()) {
    return it->second;
  }
  const fb::json &entry = reference_case("qd_nevpt2", "qd_nevpt2.json", name);
  QdState q{.weights = entry.at("weights").get<std::vector<double>>()};
  fb::NodeResources node;
  std::optional<fb::Calculation> calc = run_qd_casscf(entry, node);
  if (calc.has_value()) {
    q.s = host_state(*calc, node.device(0), q.weights.size());
    const auto &casscf = calc->get<fb::CasscfOutput>();
    q.energies = casscf->energies;
    q.mo_energies = casscf->mo_energies;
  }
  return states.emplace(name, std::move(q)).first->second;
}

// qd_sc_energy's element totals at @p s, its roots s.vector at @p weights:
// n², [M n + N].
std::vector<double> qd_totals(fb::DeviceResources &res, const State &s,
                              std::span<const double> weights) {
  const DeviceState d = upload_state(res, s);
  const auto got =
      nevpt2::qd_sc_energy(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                           d.c.data(), d.h.data(), s.ncore, s.ncas, weights, d.vector.data(), 0);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  std::vector<double> totals;
  if (got.has_value()) {
    for (const nevpt2::ClassEnergies &e : *got) {
      totals.push_back(e.total);
    }
  }
  return totals;
}

// E[W] = Σ_MN W_MN H_MN over the totals.
double weighted(std::span<const double> w, std::span<const double> totals) {
  return totals.size() == w.size() ? dot(w, totals) : std::nan("");
}

// qd_sc_adjoint at @p s along @p w, on the host: HostAdjoint's arrays (y
// every root's), its total E[W].
fb::Result<HostAdjoint> qd_adjoint_of(fb::DeviceResources &res, const State &s,
                                      std::span<const double> weights, std::span<const double> w) {
  const DeviceState d = upload_state(res, s);
  const auto got = nevpt2::qd_sc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao,
                                         d.b.data(), d.c.data(), d.h.data(), s.ncore, s.ncas,
                                         weights, d.vector.data(), w, 0);
  if (!got.has_value()) {
    return std::unexpected(got.error());
  }
  std::vector<double> totals;
  for (const nevpt2::ClassEnergies &e : got->elements) {
    totals.push_back(e.total);
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  return HostAdjoint{
      .total = weighted(w, totals),
      .h_bar = download(res, got->h_bar.data(), nmo * nmo),
      .lambda = download(res, got->lambda.data(), static_cast<std::size_t>(s.naux) * nmo * nmo),
      .y = download(res, got->y.data(), s.vector.size()),
      .x = download(res, got->x.data(), nmo * nmo),
      .a = download(res, got->a.data(), nmo * nmo),
      .epsilon = download(res, got->epsilon.data(), nmo)};
}

HostAdjoint qd_adjoint(fb::DeviceResources &res, const State &s, std::span<const double> weights,
                       std::span<const double> w) {
  auto got = qd_adjoint_of(res, s, weights, w);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return got.has_value() ? std::move(*got) : HostAdjoint{};
}

// The eigenvalues, ascending, and eigenvectors (columns, n x n column-major)
// of the symmetric n x n @p a, by cyclic Jacobi.
std::pair<std::vector<double>, std::vector<double>> symmetric_eigen(std::vector<double> a,
                                                                    const std::size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    v[i * (n + 1)] = 1.0;
  }
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        off += a[p + (n * q)] * a[p + (n * q)];
      }
    }
    if (off < 1e-60) {
      break;
    }
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t p = 0; p < q; ++p) {
        const double apq = a[p + (n * q)];
        if (apq == 0.0) {
          continue;
        }
        const double theta = (a[q + (n * q)] - a[p + (n * p)]) / (2.0 * apq);
        const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
        const double c = 1.0 / std::hypot(t, 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double kp = a[k + (n * p)];
          const double kq = a[k + (n * q)];
          a[k + (n * p)] = (c * kp) - (s * kq);
          a[k + (n * q)] = (s * kp) + (c * kq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double pk = a[p + (n * k)];
          const double qk = a[q + (n * k)];
          a[p + (n * k)] = (c * pk) - (s * qk);
          a[q + (n * k)] = (s * pk) + (c * qk);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double kp = v[k + (n * p)];
          const double kq = v[k + (n * q)];
          v[k + (n * p)] = (c * kp) - (s * kq);
          v[k + (n * q)] = (s * kp) + (c * kq);
        }
      }
    }
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(order, [&](const std::size_t i, const std::size_t j) {
    return a[i * (n + 1)] < a[j * (n + 1)];
  });
  std::vector<double> values;
  std::vector<double> vectors;
  for (const std::size_t k : order) {
    values.push_back(a[k * (n + 1)]);
    vectors.insert(vectors.end(), v.begin() + static_cast<std::ptrdiff_t>(k * n),
                   v.begin() + static_cast<std::ptrdiff_t>((k + 1) * n));
  }
  return {values, vectors};
}

// The QD gradient's seed, u_K u_Kᵀ for K the highest state of H_eff =
// diag(E_CASSCF) + the totals, symmetrized; and H_eff's eigenvalues.
std::pair<std::vector<double>, std::vector<double>> eigen_seed(const QdState &q,
                                                               std::span<const double> totals) {
  const std::size_t n = q.weights.size();
  std::vector<double> heff(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      heff[i + (n * j)] =
          (0.5 * (totals[(i * n) + j] + totals[(j * n) + i])) + (i == j ? q.energies[i] : 0.0);
    }
  }
  const auto [values, vectors] = symmetric_eigen(heff, n);
  const auto u = std::span<const double>(vectors).subspan((n - 1) * n, n);
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      w[(i * n) + j] = u[i] * u[j];
    }
  }
  return {w, values};
}

// A seeded random symmetric n x n W, uniform in [-1, 1).
std::vector<double> random_seed(const std::size_t n) {
  std::mt19937_64 engine(702);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      w[(i * n) + j] = w[(j * n) + i] = uniform(engine);
    }
  }
  return w;
}

// The ladders' bar on ε against the CASSCF's SA-Fock diagonal: the same
// Fock from the same B and roots, summed in another order.
constexpr double kQdEpsilonTolerance = 1e-10;

// Which ladders are smooth at the ladder's scale, by case, as "seed:path"
// words, or "all". Measured on the RTX 3080 over five runs, each ladder's
// ratios the same to three decimals from run to run (the forward repeats to
// far below the ladder's differences): the rest are not smooth there, their
// central differences ending in a kink or never settling, as #724 found
// along raw directions, and are not run. Most break a symmetry the state
// has. On the two C2v h2co cases the roots are 1A1 and 1A2, so every
// coupling, and every perturber whose own norm N^J_t is zero by symmetry
// (dropped at |N| <= 1e-14), vanishes at the point; along h, B or C, which
// break C2v, such a perturber's N and H grow as t², their ratio staying
// finite, and a coupling T^t_IJ / (Δ + H/N) grows as t: a first-order
// change that no derivative at the point can see. So the random W, whose
// off-diagonal reads those couplings, runs a ratio run that then breaks
// (h: 4.769 4.162 3.997 4.077 4.740 18.722 -0.908 2.349) and misses by
// 2.6e-4 (h), 8.0e-4 (B); the eigenvector seed, diagonal there, does not
// read them. Elsewhere it is #724's: a perturber with a small own norm is
// moved toward the pole Δ + H/N = 0 (h2co_skew_ccpvdz_sa3 along h: 4.298
// 4.901 3.191 4.805 -2.122 0.031 -0.703 -16.157; h6_631g_sa2 along h: ...
// 0.000 -1.000 -13037, a step); a random diagonal W (measured, not kept)
// was not smooth along the eigenvector seed's failing paths either. Every
// case's c_0 ladders at both seeds, and h8_sto3g_sa3 along every path.
const std::map<std::string, std::string, std::less<>> kQdSmooth{
    {"h2co_ccpvdz_sa2", "eigen:h eigen:B eigen:C eigen:c_0 eigen:c_1 random:c_0 random:c_1"},
    {"h2co_ccpvdz_sa2_weighted", "eigen:h eigen:B eigen:c_0 eigen:c_1 random:c_0 random:c_1"},
    {"h2co_skew_ccpvdz_sa3", "eigen:B eigen:c_0 random:c_0"},
    {"h6_631g_sa2", "eigen:B eigen:C eigen:c_0 random:B random:C random:c_0"},
    {"h8_sto3g_sa3", "all"},
    {"h8_ring_sto3g_sa3",
     "eigen:h eigen:c_0 eigen:c_1 eigen:c_2 random:h random:c_0 random:c_1 random:c_2"}};

// Each of E[W]'s adjoints against a ladder of E[W] at both seeds, the two
// seeds' ladders reading the same forwards: along h and B (straight; E is
// not linear in either), along C exp(t κ) with every block's κ at once, and
// along each root c_r(t) = (c_r + t d_r) / |c_r + t d_r| alone, d_r
// ci_direction's at c_r (in its spin and symmetry sector, ⊥ c_r), against
// <d_r, y_r>; each where kQdSmooth lists it. And ε against the CASSCF's
// mo_energies; H_eff's eigenvalues are printed beside the goldens'.
class Nevpt2QdAdjoint : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2QdAdjoint, AdjointsMatchLadders) {
  FBF_REQUIRE_GPU();
  const std::string &name = GetParam();
  const QdState &q = qd_state(name);
  const State &s = q.s;
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const std::size_t n = q.weights.size();
  const std::size_t ndet = s.vector.size() / n;
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::vector<double> totals = qd_totals(res, s, q.weights);
  ASSERT_EQ(totals.size(), n * n);

  const auto [eigen, values] = eigen_seed(q, totals);
  const std::array<std::pair<std::string_view, std::vector<double>>, 2> seeds{
      {{"eigen", eigen}, {"random", random_seed(n)}}};
  const auto smooth = kQdSmooth.find(name);
  ASSERT_NE(smooth, kQdSmooth.end()) << name;
  const auto laddered = [&](const std::string_view seed, const std::string_view path) {
    return smooth->second == "all" ||
           std::format(" {} ", smooth->second).find(std::format(" {}:{} ", seed, path)) !=
               std::string::npos;
  };
  std::vector<HostAdjoint> adj;
  for (const auto &[seed, w] : seeds) {
    adj.push_back(qd_adjoint(res, s, q.weights, w));
    ASSERT_EQ(adj.back().y.size(), s.vector.size()) << seed;
    EXPECT_NEAR(adj.back().total, weighted(w, totals), kSameEnergyTolerance) << seed;
  }

  // ε is the SA Fock's diagonal, as the CASSCF's own.
  ASSERT_EQ(q.mo_energies.size(), nmo);
  double epsilon_worst = 0.0;
  for (std::size_t p = 0; p < nmo; ++p) {
    if (p >= static_cast<std::size_t>(s.ncore) && p < static_cast<std::size_t>(s.ncore + s.ncas)) {
      continue;
    }
    EXPECT_NEAR(adj[0].epsilon[p], q.mo_energies[p], kQdEpsilonTolerance) << "orbital " << p;
    epsilon_worst = std::max(epsilon_worst, std::abs(adj[0].epsilon[p] - q.mo_energies[p]));
  }
  const auto want = reference_case("qd_nevpt2", "qd_nevpt2.json", name)
                        .at("eigenvalues")
                        .get<std::vector<double>>();
  std::cout << std::format("[ nevpt2   ] qd {}: H_eff eigenvalues {::.10f}, goldens' {::.10f}; ε "
                           "from mo_energies {:.2e}\n",
                           name, values, want, epsilon_worst);

  // Each path's forwards are evaluated once, both seeds reading them.
  const auto ladders = [&](const std::string &path,
                           const std::function<void(double, State &)> &move,
                           const std::function<double(const HostAdjoint &)> &predicted) {
    std::map<double, std::vector<double>> forwards;
    State moved = s;
    for (std::size_t k = 0; k < seeds.size(); ++k) {
      if (!laddered(seeds[k].first, path)) {
        std::cout << std::format("[ nevpt2   ] qd {} {} {}: not laddered, not smooth at the "
                                 "ladder's scale (kQdSmooth)\n",
                                 name, seeds[k].first, path);
        continue;
      }
      expect_ladder(
          std::format("qd {} {} {}", name, seeds[k].first, path),
          [&](const double t) {
            auto it = forwards.find(t);
            if (it == forwards.end()) {
              move(t, moved);
              it = forwards.emplace(t, qd_totals(res, moved, q.weights)).first;
            }
            return weighted(seeds[k].second, it->second);
          },
          predicted(adj[k]));
    }
  };

  for (const std::string which : {"h", "B"}) {
    const IntegralDirection dir = integral_direction(s, which);
    ladders(
        which, [&](const double t, State &moved) { perturb(s, dir, t, moved); },
        [&](const HostAdjoint &a) { return integral_dot(s, dir, a.h_bar, a.lambda); });
  }

  std::vector<double> kappa(nmo * nmo, 0.0);
  for (const Block &block : kBlocks) {
    const std::vector<double> k = block_kappa(s, block);
    for (std::size_t i = 0; i < kappa.size(); ++i) {
      kappa[i] += k[i];
    }
  }
  ladders(
      "C", [&](const double t, State &moved) { rotate(s, kappa, t, moved); },
      [&](const HostAdjoint &a) { return source_dot(kappa, a.a, nmo); });

  for (std::size_t r = 0; r < n; ++r) {
    const std::span<const double> root = std::span<const double>(s.vector).subspan(r * ndet, ndet);
    State alone = s;
    alone.vector.assign(root.begin(), root.end());
    const std::vector<double> d = ci_direction(res, alone);
    ASSERT_EQ(d.size(), ndet);
    ladders(
        std::format("c_{}", r),
        [&](const double t, State &moved) {
          const auto out = std::span<double>(moved.vector).subspan(r * ndet, ndet);
          for (std::size_t i = 0; i < ndet; ++i) {
            out[i] = root[i] + (t * d[i]);
          }
          const double length = std::sqrt(dot(out, out));
          for (double &v : out) {
            v /= length;
          }
        },
        [&](const HostAdjoint &a) {
          return dot(d, std::span<const double>(a.y).subspan(r * ndet, ndet));
        });
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdAdjoint,
                         ::testing::Values("h2co_ccpvdz_sa2", "h2co_ccpvdz_sa2_weighted",
                                           "h2co_skew_ccpvdz_sa3", "h6_631g_sa2", "h8_sto3g_sa3",
                                           "h8_ring_sto3g_sa3"),
                         [](const auto &info) { return info.param; });

// The one-root call at weight 1 and W = 1 is sc_adjoint at the state, to
// sc_adjoint's own run-to-run spread: each array within kQdOneRootEps of
// its largest element's machine epsilon, and the energy within
// kSameEnergyTolerance. Not bitwise: sc_adjoint is not bitwise against
// itself (class_energies_adjoint's and the einsums' atomics reassociate each
// sum), and over 35 runs on the RTX 3080 two of its calls came out up to
// 2476 ε apart (A; y 2324, Λ and h̄ 1326, X 574, ε 4.7), and the one-root
// call against it up to 2532 (A; y 1950, Λ and h̄ 1315, X 656, ε 3.9): the
// same spread, the QD path adding the walk's γ̄ and the builders' in another
// order. The bound is that floor with room, about 1.6 times it.
constexpr double kQdOneRootEps = 4096.0;

TEST(Nevpt2Qd, OneRootIsScAdjoint) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint sc = adjoint(res, s);
  const HostAdjoint again = adjoint(res, s);
  const std::array<double, 1> w{1.0};
  const HostAdjoint qd = qd_adjoint(res, s, kOneRoot, w);
  ASSERT_EQ(qd.y.size(), sc.y.size());
  // max |a - b| in units of b's largest magnitude's machine epsilon.
  const auto worst = [](std::span<const double> a, std::span<const double> b) {
    double out = 0.0;
    double largest = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      out = std::max(out, std::abs(a[i] - b[i]));
      largest = std::max(largest, std::abs(b[i]));
    }
    return out / (std::numeric_limits<double>::epsilon() * largest);
  };
  for (const auto &[name, other] : {std::pair{"sc_adjoint again", &again}, std::pair{"qd", &qd}}) {
    const std::array<std::pair<const char *, double>, 6> apart{
        {{"h_bar", worst(other->h_bar, sc.h_bar)},
         {"lambda", worst(other->lambda, sc.lambda)},
         {"y", worst(other->y, sc.y)},
         {"x", worst(other->x, sc.x)},
         {"a", worst(other->a, sc.a)},
         {"epsilon", worst(other->epsilon, sc.epsilon)}}};
    for (const auto &[field, difference] : apart) {
      EXPECT_LE(difference, kQdOneRootEps) << name << ": " << field;
      std::cout << std::format("[ nevpt2   ] one root, {} {}: apart by {:.3e} eps max\n", name,
                               field, difference);
    }
    EXPECT_NEAR(other->total, sc.total, kSameEnergyTolerance) << name;
  }
}

// qd_sc_energy and qd_sc_adjoint refuse what they cannot take: bad weights
// (none, a negative one, a sum off 1) and a W that is not n².
TEST(Nevpt2Qd, RefusesBadInputs) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  State two = s;
  two.vector.insert(two.vector.end(), s.vector.begin(), s.vector.end());
  const DeviceState d = upload_state(res, two);
  const std::array<double, 2> equal{0.5, 0.5};
  const std::array<double, 2> negative{1.5, -0.5};
  const std::array<double, 2> short_sum{0.5, 0.4};
  const std::array<double, 4> w{1.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> short_w{1.0, 0.0, 0.0};
  const auto energy_at = [&](std::span<const double> weights) {
    return nevpt2::qd_sc_energy(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                d.c.data(), d.h.data(), s.ncore, s.ncas, weights, d.vector.data(),
                                0);
  };
  const auto adjoint_at = [&](std::span<const double> weights, std::span<const double> seed) {
    return nevpt2::qd_sc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                 d.c.data(), d.h.data(), s.ncore, s.ncas, weights, d.vector.data(),
                                 seed, 0);
  };
  for (const std::span<const double> weights :
       {std::span<const double>{}, std::span<const double>(negative),
        std::span<const double>(short_sum)}) {
    const auto e = energy_at(weights);
    ASSERT_FALSE(e.has_value());
    EXPECT_EQ(e.error().kind, fb::ErrorKind::InvalidConfig) << e.error().message;
    const auto a = adjoint_at(weights, w);
    ASSERT_FALSE(a.has_value());
    EXPECT_EQ(a.error().kind, fb::ErrorKind::InvalidConfig) << a.error().message;
  }
  const auto a = adjoint_at(equal, short_w);
  ASSERT_FALSE(a.has_value());
  EXPECT_EQ(a.error().kind, fb::ErrorKind::InvalidConfig) << a.error().message;
}

} // namespace
