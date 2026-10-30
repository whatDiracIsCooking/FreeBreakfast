// GPU cases for cis.ris_precondition over H2O / STO-3G's own RHF
// (data/cis/cis.json's geometry): the correction z satisfies
// (θ − Δε − 2 L Lᵀ) z = r to 1e-7 against the operator rebuilt on the host
// from Δε and the factor L the preconditioner holds, the clamp included, at
// Ritz values between, below and on the Δε; in place it is the same bytes; a
// θ exactly on a Δε still gives a finite z; L Lᵀ is (ia|P) M⁻¹ (P|jb), so
// symmetric positive semidefinite with L's RIS basis the Sp one's six
// functions; and the refusals. With the exchange (RIS-JK), against the
// operator rebuilt from Δε, L, T_oo and T_vv: at a tight tolerance and a
// Krylov cap above n_ov, z satisfies (θ − Δε − 2 L Lᵀ + K_RI) z = r to 1e-6;
// zero inner iterations is the RIS-J answer to 1e-12; and θ on a Δε still
// gives a finite z. CisStep's use of both, against the reference's roots, is
// cis_step_gpu.cpp's.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.cis.precondition;
import free_breakfast.cis.ris_precondition;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace cis = free_breakfast::cis;

// The issue's bar on the system's residual, relative to r's largest entry.
constexpr double kSystemTolerance = 1e-7;
// RIS-JK's: the GMRES's system, solved to its own tolerance.
constexpr double kExchangeTolerance = 1e-6;
// Zero inner iterations against RIS-J.
constexpr double kDegradeTolerance = 1e-12;
// The GMRES at a tight tolerance and a Krylov cap above H2O / STO-3G's
// n_ov = 10, so an unrestarted run spans the whole space.
constexpr cis::RisOptions kTightExchange{
    .exchange = true, .k_max_inner_iters = 20, .k_inner_tol = 1e-14};
// Δε's clamp, the diagonal correction's (#195).
constexpr double kClamp = 1e-3;

// H2O / STO-3G's RHF, through the real registry: the reference's molecule and
// basis, the SCF converged tightly with every Fock build full (as
// cis_step_gpu.cpp's), nothing past it.
struct Setup {
  fb::test::ScratchDir dir{"cis_ris_precondition"};
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
};

std::unique_ptr<Setup> h2o_sto3g() {
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "cis" / "cis.json");
  EXPECT_TRUE(read.has_value()) << read.error().message;
  if (!read.has_value()) {
    return nullptr;
  }
  const fb::json *entry = nullptr;
  for (const fb::json &e : read->at("cases")) {
    if (e.at("name").get<std::string>() == "h2o_sto3g") {
      entry = &e;
    }
  }
  EXPECT_NE(entry, nullptr);
  if (entry == nullptr) {
    return nullptr;
  }
  auto setup = std::make_unique<Setup>();
  std::string xyz = std::format("{}\nh2o\n", entry->at("atoms").size());
  for (const fb::json &atom : entry->at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(setup->dir / "molecule.xyz", xyz).has_value());
  EXPECT_TRUE(fb::write_file(setup->dir / "run.toml",
                             "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
                             "[scf]\nenergy_tolerance = 1e-11\ncommutator_tolerance = 1e-10\n"
                             "fock_rebuild_interval = 1\n")
                  .has_value());
  auto calc = fb::read_config(setup->dir / "run.toml", setup->dir.path(), setup->dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc));
  const fb::Status ran = setup->calc->run(setup->node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  return ran.has_value() ? std::move(setup) : nullptr;
}

fb::Buffers::device<double> to_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> to_host(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, count * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The preconditioner's inputs over the setup's RHF: Δε on the device, and the
// split.
struct Problem {
  std::int32_t n_occ = 0;
  std::int32_t n_ov = 0;
  fb::Buffers::device<double> delta;
};

Problem problem(fb::DeviceResources &res, const fb::Calculation &calc) {
  const fb::ScfOutput &scf = *calc.get<fb::ScfOutput>();
  const std::int32_t n_occ = scf.n_alpha;
  const auto n_virt = static_cast<std::int32_t>(scf.mo_energies.size()) - n_occ;
  const auto energies = to_device(res, scf.mo_energies);
  fb::Buffers::device<double> delta(static_cast<std::size_t>(n_occ * n_virt), res.device());
  EXPECT_TRUE(cis::diagonal(res, n_occ, n_virt, energies.data(), delta.data()).has_value());
  EXPECT_TRUE(res.sync().has_value());
  return Problem{.n_occ = n_occ, .n_ov = n_occ * n_virt, .delta = std::move(delta)};
}

fb::Result<cis::RisPreconditioner> make_ris(fb::DeviceResources &res, const fb::Calculation &calc,
                                            const Problem &p, const std::int32_t max_columns,
                                            const cis::RisOptions ris = {}) {
  return cis::RisPreconditioner::create(res, calc.get<fb::BasisSetOutput>()->basis,
                                        calc.get<fb::PrimitivePairClassesOutput>()->classes,
                                        calc.get<fb::MoleculeOutput>()->molecule.atoms(),
                                        calc.get<fb::ScfOutput>()->c, p.n_occ, p.delta.data(),
                                        max_columns, ris);
}

// The exchange's blocks on the host, for worst_residual: empty for RIS-J.
struct Exchange {
  std::int32_t n_occ = 0;
  std::vector<double> oo;
  std::vector<double> vv;
};

// max |M z − r| / max |r| per column, M = clamp(θ − Δε) − 2 L Lᵀ + K_RI
// rebuilt on the host from Δε, Lᵀ (naux × n_ov) and, unless they are empty,
// T_oo and T_vv: (K_RI)_ia,jb = Σ_Q T_ij^Q T_ab^Q.
double worst_residual(const std::vector<double> &delta, const std::vector<double> &factor,
                      const std::int32_t naux, const std::vector<double> &theta,
                      const std::vector<double> &r, const std::vector<double> &z,
                      const Exchange &blocks = {}) {
  const std::size_t n = delta.size();
  const auto aux = static_cast<std::size_t>(naux);
  const auto occ = static_cast<std::size_t>(blocks.n_occ);
  const std::size_t virt = occ == 0 ? 0 : n / occ;
  const auto exchange = [&](const std::size_t ia, const std::size_t jb) {
    if (blocks.oo.empty()) {
      return 0.0;
    }
    const std::size_t i = ia / virt;
    const std::size_t a = ia % virt;
    const std::size_t j = jb / virt;
    const std::size_t b = jb % virt;
    double sum = 0.0;
    for (std::size_t q = 0; q < aux; ++q) {
      sum += blocks.oo[q + (aux * (i + (occ * j)))] * blocks.vv[q + (aux * (a + (virt * b)))];
    }
    return sum;
  };
  double worst = 0.0;
  for (std::size_t c = 0; c < theta.size(); ++c) {
    double r_max = 0.0;
    double res_max = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double d = theta[c] - delta[j];
      if (std::abs(d) < kClamp) {
        d = d < 0.0 ? -kClamp : kClamp;
      }
      double mz = d * z[j + (c * n)];
      for (std::size_t k = 0; k < n; ++k) {
        double llt = 0.0;
        for (std::size_t q = 0; q < aux; ++q) {
          llt += factor[q + (aux * j)] * factor[q + (aux * k)];
        }
        mz += (exchange(j, k) - (2.0 * llt)) * z[k + (c * n)];
      }
      r_max = std::max(r_max, std::abs(r[j + (c * n)]));
      res_max = std::max(res_max, std::abs(mz - r[j + (c * n)]));
    }
    worst = std::max(worst, res_max / r_max);
  }
  return worst;
}

std::vector<double> random_block(const std::size_t count, const std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> v(count);
  for (double &x : v) {
    x = dist(gen);
  }
  return v;
}

// Ritz values below every Δε, between two, and well above them, one per
// column: z solves the clamped system to 1e-7, and the same call in place
// gives the same bytes.
TEST(CisRisPrecondition, SolvesTheSystemExactly) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  auto delta = to_host(res, p.delta.data(), p.delta.num_elements());
  std::vector<double> sorted = delta;
  std::ranges::sort(sorted);
  const std::vector<double> theta{sorted.front() - 0.1, 0.5 * (sorted[2] + sorted[3]),
                                  sorted.back() + 0.7};
  const auto columns = static_cast<std::int32_t>(theta.size());
  auto ris = make_ris(res, *setup->calc, p, columns);
  ASSERT_TRUE(ris.has_value()) << ris.error().message;
  // Sp: O's s and p, and each H's s.
  EXPECT_EQ(ris->naux(), 6);
  const auto n = static_cast<std::size_t>(p.n_ov);
  ASSERT_EQ(ris->factor().num_elements(), static_cast<std::size_t>(ris->naux()) * n);

  const auto r = random_block(n * theta.size(), 2026U);
  const auto d_r = to_device(res, r);
  fb::Buffers::device<double> d_z(r.size(), res.device());
  const fb::Status ran = (*ris)(res, theta, nullptr, d_r.data(), d_z.data());
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto z = to_host(res, d_z.data(), r.size());
  const auto factor = to_host(res, ris->factor().data(), ris->factor().num_elements());
  const double worst = worst_residual(delta, factor, ris->naux(), theta, r, z);
  std::cout << "max |M z - r| / max |r|: " << worst << "\n";
  EXPECT_LE(worst, kSystemTolerance);
  // The low-rank term is not nothing: z is not the diagonal correction.
  double off_diagonal = 0.0;
  for (std::size_t c = 0; c < theta.size(); ++c) {
    for (std::size_t j = 0; j < n; ++j) {
      off_diagonal = std::max(off_diagonal,
                              std::abs(z[j + (c * n)] - (r[j + (c * n)] / (theta[c] - delta[j]))));
    }
  }
  EXPECT_GT(off_diagonal, 1e-6);

  auto in_place = to_device(res, r);
  const fb::Status again = (*ris)(res, theta, nullptr, in_place.data(), in_place.data());
  ASSERT_TRUE(again.has_value()) << again.error().message;
  EXPECT_EQ(to_host(res, in_place.data(), r.size()), z);
}

// θ exactly on a Δε puts a zero on Dt's diagonal, which the clamp lifts to
// 1e-3: z is finite, and solves the clamped system.
TEST(CisRisPrecondition, TheClampKeepsZFinite) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  const auto delta = to_host(res, p.delta.data(), p.delta.num_elements());
  const std::vector<double> theta{delta.front(), delta.back()};
  auto ris = make_ris(res, *setup->calc, p, 2);
  ASSERT_TRUE(ris.has_value()) << ris.error().message;
  const auto n = static_cast<std::size_t>(p.n_ov);
  const auto r = random_block(n * theta.size(), 7U);
  auto d_z = to_device(res, r);
  const fb::Status ran = (*ris)(res, theta, nullptr, d_z.data(), d_z.data());
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto z = to_host(res, d_z.data(), r.size());
  for (std::size_t e = 0; e < z.size(); ++e) {
    ASSERT_TRUE(std::isfinite(z[e])) << e;
  }
  const auto factor = to_host(res, ris->factor().data(), ris->factor().num_elements());
  EXPECT_LE(worst_residual(delta, factor, ris->naux(), theta, r, z), kSystemTolerance);
}

// The richer basis carries more functions, Spd's 11 (O's s, p and d, each
// H's s), and its factor is finite and not empty.
TEST(CisRisPrecondition, SpdCarriesElevenFunctions) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  auto spd = make_ris(res, *setup->calc, p, 1, cis::RisOptions{.shells = cis::RisShells::Spd});
  ASSERT_TRUE(spd.has_value()) << spd.error().message;
  EXPECT_EQ(spd->naux(), 11);
  const auto factor = to_host(res, spd->factor().data(), spd->factor().num_elements());
  double largest = 0.0;
  for (const double x : factor) {
    ASSERT_TRUE(std::isfinite(x));
    largest = std::max(largest, std::abs(x));
  }
  EXPECT_GT(largest, 1e-3);
}

// RIS-JK at a tight tolerance and a Krylov cap above n_ov: z solves
// (θ − Δε − 2 L Lᵀ + K_RI) z = r to 1e-6 at Ritz values below, between and
// above the Δε, the exchange's blocks are the symmetric ones they should be,
// and z is not the RIS-J answer.
TEST(CisRisPrecondition, ExchangeSolvesTheSystem) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  const auto delta = to_host(res, p.delta.data(), p.delta.num_elements());
  std::vector<double> sorted = delta;
  std::ranges::sort(sorted);
  const std::vector<double> theta{sorted.front() - 0.1, 0.5 * (sorted[2] + sorted[3]),
                                  sorted.back() + 0.7};
  const auto columns = static_cast<std::int32_t>(theta.size());
  auto jk = make_ris(res, *setup->calc, p, columns, kTightExchange);
  ASSERT_TRUE(jk.has_value()) << jk.error().message;
  auto j_only = make_ris(res, *setup->calc, p, columns);
  ASSERT_TRUE(j_only.has_value()) << j_only.error().message;
  const auto n = static_cast<std::size_t>(p.n_ov);
  const auto aux = static_cast<std::size_t>(jk->naux());
  const auto occ = static_cast<std::size_t>(p.n_occ);
  const std::size_t virt = n / occ;
  ASSERT_EQ(jk->factor_oo().num_elements(), aux * occ * occ);
  ASSERT_EQ(jk->factor_vv().num_elements(), aux * virt * virt);
  EXPECT_EQ(j_only->factor_oo().num_elements(), 0U);
  const Exchange blocks{
      .n_occ = p.n_occ,
      .oo = to_host(res, jk->factor_oo().data(), jk->factor_oo().num_elements()),
      .vv = to_host(res, jk->factor_vv().data(), jk->factor_vv().num_elements()),
  };
  // T_ij^Q = T_ji^Q, which the exchange's product leans on.
  for (std::size_t q = 0; q < aux; ++q) {
    for (std::size_t i = 0; i < occ; ++i) {
      for (std::size_t j = 0; j < occ; ++j) {
        EXPECT_NEAR(blocks.oo[q + (aux * (i + (occ * j)))], blocks.oo[q + (aux * (j + (occ * i)))],
                    1e-12);
      }
    }
  }

  const auto r = random_block(n * theta.size(), 2026U);
  const auto d_r = to_device(res, r);
  fb::Buffers::device<double> d_z(r.size(), res.device());
  const fb::Status ran = (*jk)(res, theta, nullptr, d_r.data(), d_z.data());
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto z = to_host(res, d_z.data(), r.size());
  const auto factor = to_host(res, jk->factor().data(), jk->factor().num_elements());
  const double worst = worst_residual(delta, factor, jk->naux(), theta, r, z, blocks);
  std::cout << "max |(M + K) z - r| / max |r|: " << worst << "\n";
  EXPECT_LE(worst, kExchangeTolerance);

  // The exchange is not nothing: z is not RIS-J's.
  fb::Buffers::device<double> d_j(r.size(), res.device());
  const fb::Status ran_j = (*j_only)(res, theta, nullptr, d_r.data(), d_j.data());
  ASSERT_TRUE(ran_j.has_value()) << ran_j.error().message;
  const auto z_j = to_host(res, d_j.data(), r.size());
  double change = 0.0;
  for (std::size_t e = 0; e < z.size(); ++e) {
    change = std::max(change, std::abs(z[e] - z_j[e]));
  }
  EXPECT_GT(change, 1e-6);

  auto in_place = to_device(res, r);
  const fb::Status again = (*jk)(res, theta, nullptr, in_place.data(), in_place.data());
  ASSERT_TRUE(again.has_value()) << again.error().message;
  EXPECT_EQ(to_host(res, in_place.data(), r.size()), z);
}

// With no inner iterations allowed, RIS-JK is the RIS-J answer: every column
// left at its seed.
TEST(CisRisPrecondition, ZeroInnerIterationsIsRisJ) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  const auto delta = to_host(res, p.delta.data(), p.delta.num_elements());
  std::vector<double> sorted = delta;
  std::ranges::sort(sorted);
  const std::vector<double> theta{sorted.front() - 0.1, 0.5 * (sorted[2] + sorted[3])};
  const auto columns = static_cast<std::int32_t>(theta.size());
  auto none = make_ris(res, *setup->calc, p, columns,
                       cis::RisOptions{.exchange = true, .k_max_inner_iters = 0});
  ASSERT_TRUE(none.has_value()) << none.error().message;
  auto j_only = make_ris(res, *setup->calc, p, columns);
  ASSERT_TRUE(j_only.has_value()) << j_only.error().message;
  const auto r = random_block(static_cast<std::size_t>(p.n_ov) * theta.size(), 99U);
  const auto d_r = to_device(res, r);
  fb::Buffers::device<double> d_z(r.size(), res.device());
  fb::Buffers::device<double> d_j(r.size(), res.device());
  const fb::Status ran = (*none)(res, theta, nullptr, d_r.data(), d_z.data());
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const fb::Status ran_j = (*j_only)(res, theta, nullptr, d_r.data(), d_j.data());
  ASSERT_TRUE(ran_j.has_value()) << ran_j.error().message;
  const auto z = to_host(res, d_z.data(), r.size());
  const auto z_j = to_host(res, d_j.data(), r.size());
  for (std::size_t e = 0; e < z.size(); ++e) {
    EXPECT_NEAR(z[e], z_j[e], kDegradeTolerance * std::max(1.0, std::abs(z_j[e]))) << e;
  }
}

// θ exactly on a Δε, with the exchange on: the clamp keeps z finite, and it
// still solves the clamped system.
TEST(CisRisPrecondition, ExchangeTheClampKeepsZFinite) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  const auto delta = to_host(res, p.delta.data(), p.delta.num_elements());
  const std::vector<double> theta{delta.front(), delta.back()};
  auto jk = make_ris(res, *setup->calc, p, 2, kTightExchange);
  ASSERT_TRUE(jk.has_value()) << jk.error().message;
  const auto n = static_cast<std::size_t>(p.n_ov);
  const auto r = random_block(n * theta.size(), 7U);
  auto d_z = to_device(res, r);
  const fb::Status ran = (*jk)(res, theta, nullptr, d_z.data(), d_z.data());
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto z = to_host(res, d_z.data(), r.size());
  for (std::size_t e = 0; e < z.size(); ++e) {
    ASSERT_TRUE(std::isfinite(z[e])) << e;
  }
  const auto factor = to_host(res, jk->factor().data(), jk->factor().num_elements());
  const Exchange blocks{
      .n_occ = p.n_occ,
      .oo = to_host(res, jk->factor_oo().data(), jk->factor_oo().num_elements()),
      .vv = to_host(res, jk->factor_vv().data(), jk->factor_vv().num_elements()),
  };
  EXPECT_LE(worst_residual(delta, factor, jk->naux(), theta, r, z, blocks), kExchangeTolerance);
}

TEST(CisRisPrecondition, Refusals) {
  FBF_REQUIRE_GPU();
  const auto setup = h2o_sto3g();
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->node.device(0);
  const Problem p = problem(res, *setup->calc);
  const auto expect_config = [](const auto &refused, std::string_view needle) {
    ASSERT_FALSE(refused.has_value()) << needle;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(refused.error().message.find(needle), std::string::npos) << refused.error().message;
  };
  expect_config(make_ris(res, *setup->calc, p, 0), "cis::RisPreconditioner::create: needs delta");
  expect_config(make_ris(res, *setup->calc, p, 1, cis::RisOptions{.theta = -1.0}),
                "cis::ris_basis: theta");
  const fb::Calculation &calc = *setup->calc;
  expect_config(cis::RisPreconditioner::create(res, calc.get<fb::BasisSetOutput>()->basis,
                                               calc.get<fb::PrimitivePairClassesOutput>()->classes,
                                               calc.get<fb::MoleculeOutput>()->molecule.atoms(),
                                               calc.get<fb::ScfOutput>()->c, 0, p.delta.data(), 1,
                                               {}),
                "cis::RisPreconditioner::create: needs 1 <= n_occ < nao");

  expect_config(
      make_ris(res, calc, p, 1, cis::RisOptions{.exchange = true, .k_max_inner_iters = -1}),
      "cis::RisPreconditioner::create: the exchange needs k_max_inner_iters >= 0");
  expect_config(make_ris(res, calc, p, 1, cis::RisOptions{.exchange = true, .k_inner_tol = 0.0}),
                "cis::RisPreconditioner::create: the exchange needs k_max_inner_iters >= 0");

  auto ris = make_ris(res, calc, p, 1);
  ASSERT_TRUE(ris.has_value()) << ris.error().message;
  const std::vector<double> two{0.5, 0.6};
  fb::Buffers::device<double> block(2 * static_cast<std::size_t>(p.n_ov), res.device());
  expect_config((*ris)(res, two, nullptr, block.data(), block.data()),
                "cis::RisPreconditioner: 2 columns, more than the 1");
}

} // namespace
