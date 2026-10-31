// GPU cases for response.cphf: B fitted by density_fit from PySCF's int2c2e
// and int3c2e (data/density_fitting/'s h2o_ccpvdz), and that molecule's
// DF-RHF orbitals, orbital energies and CPHF right-hand sides from
// data/cphf/ (tools/emit/cphf_reference.py). The nine nuclear-displacement
// columns against pyscf.df.hessian.rhf's solve_mo1 over the same fitting
// set; the three dipole columns' static polarizability against PySCF's; one
// column solved alone against the same column of the lockstep block; the
// batched product against one column at a time; and the refusals and an
// exhausted budget.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.response;

namespace {

namespace fb = free_breakfast;

// The issue's bar: U against PySCF's mo1, absolute (max |U| is 0.15). Both
// stop near 1e-10 in the residual and M's smallest eigenvalue is 0.35, so
// each U is good to a few 1e-10.
constexpr double kSolutionTol = 1e-8;
// alpha against PySCF's, of max |alpha|: quadratic in U's error.
constexpr double kPolarizabilityRelTol = 1e-9;
// A column alone against the same column of the block: the same products in
// the same order, so to rounding.
constexpr double kLockstepTol = 1e-13;
constexpr double kTolerance = 1e-10;
// The batched product against one column at a time, absolute: the same
// products in other gemm shapes, through the same sweeps (1.8e-15 measured
// on decane's 96 columns).
constexpr double kBatchedTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR};

fb::json read_manifest(const std::filesystem::path &path) {
  auto read = fb::read_json(path);
  EXPECT_TRUE(read) << (read ? "" : read.error().message);
  return read ? std::move(*read) : fb::json{};
}

std::vector<double> read_array(const std::filesystem::path &path,
                               const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(path);
  EXPECT_TRUE(a) << path;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << path;
  return a->data;
}

// The fitting inputs as df_mo_gpu.cpp reads them: the metric, and (P|mn)
// transposed from the pieces' [m][n][P] into [P][n][m].
struct Golden {
  std::int32_t nao = 0;
  std::int32_t naux = 0;
  std::int32_t nocc = 0;
  std::vector<double> metric;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> energies;
  std::vector<double> nuclear_rhs;
  std::vector<double> nuclear_solution;
  std::vector<double> dipole_rhs;
  std::vector<double> polarizability;
};

Golden read_golden() {
  const fb::json cphf = read_manifest(kDataDir / "cphf" / "cphf.json");
  const fb::json df = read_manifest(kDataDir / "density_fitting" / "density_fitting.json");
  Golden g;
  for (const auto &c : df["integrals"]) {
    if (c["name"].get<std::string>() != "h2o_ccpvdz") {
      continue;
    }
    g.nao = c["nao"].get<std::int32_t>();
    g.naux = c["naux"].get<std::int32_t>();
    const auto nao = static_cast<std::size_t>(g.nao);
    const auto naux = static_cast<std::size_t>(g.naux);
    g.metric =
        read_array(kDataDir / "density_fitting" / c["int2c2e"].get<std::string>(), {naux, naux});
    g.b.assign(naux * nao * nao, std::numeric_limits<double>::quiet_NaN());
    for (const auto &piece : c["int3c2e"]) {
      const auto range = piece["aux"].get<std::vector<std::size_t>>();
      const auto slab = read_array(kDataDir / "density_fitting" / piece["file"].get<std::string>(),
                                   {nao, nao, range[1] - range[0]});
      if (slab.empty()) {
        return {};
      }
      for (std::size_t p = range[0]; p < range[1]; ++p) {
        for (std::size_t n = 0; n < nao; ++n) {
          for (std::size_t m = 0; m < nao; ++m) {
            g.b[p + naux * (n + nao * m)] = slab[m + nao * (n + nao * (p - range[0]))];
          }
        }
      }
    }
  }
  if (g.nao == 0 || cphf["nao"].get<std::int32_t>() != g.nao ||
      cphf["naux"].get<std::int32_t>() != g.naux) {
    ADD_FAILURE() << "cphf.json and density_fitting.json's h2o_ccpvdz disagree";
    return {};
  }
  const auto nao = static_cast<std::size_t>(g.nao);
  g.nocc = cphf["occupied"].get<std::int32_t>();
  const auto pairs = (nao - static_cast<std::size_t>(g.nocc)) * static_cast<std::size_t>(g.nocc);
  const auto path = [&](const fb::json &name) {
    return kDataDir / "cphf" / name.get<std::string>();
  };
  g.c = read_array(path(cphf["mo_coefficients"]), {nao, nao});
  g.energies = read_array(path(cphf["mo_energies"]), {nao});
  g.nuclear_rhs = read_array(path(cphf["nuclear"]["rhs"]), {pairs, 9});
  g.nuclear_solution = read_array(path(cphf["nuclear"]["solution"]), {pairs, 9});
  g.dipole_rhs = read_array(path(cphf["dipole"]["rhs"]), {pairs, 3});
  g.polarizability = read_array(path(cphf["dipole"]["polarizability"]), {3, 3});
  return g;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *d) {
  ASSERT_EQ(wwr::wwrMemcpyAsync(d, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // host is pageable: the copy has not finished reading it until the stream has.
  ASSERT_TRUE(res.sync().has_value());
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &d) {
  std::vector<double> host(d.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// The fitted B and the orbitals on the device.
class ResponseCphf : public ::testing::Test {
protected:
  void SetUp() override {
    FBF_REQUIRE_GPU();
    g_ = read_golden();
    ASSERT_GT(g_.nocc, 0);
    res_.emplace(0);
    fb::Buffers::device<double> metric(g_.metric.size(), res_->device());
    b_.emplace(g_.b.size(), res_->device());
    c_.emplace(g_.c.size(), res_->device());
    upload(*res_, g_.metric, metric.data());
    upload(*res_, g_.b, b_->data());
    upload(*res_, g_.c, c_->data());
    const fb::Status fitted = fb::density_fit(*res_, g_.nao, g_.naux, metric.data(), b_->data());
    ASSERT_TRUE(fitted) << fitted.error().message;
  }

  fb::Buffers::device<double> device(const std::vector<double> &host) {
    fb::Buffers::device<double> d(host.size(), res_->device());
    upload(*res_, host, d.data());
    return d;
  }

  fb::Result<fb::response::Cphf> solve(const fb::Buffers::device<double> &rhs,
                                       const std::int32_t max_iterations = 100,
                                       const bool batched = false) {
    return fb::response::cphf(*res_, *b_, g_.naux, *c_, g_.energies, g_.nocc, rhs, kTolerance,
                              max_iterations, batched);
  }

  Golden g_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::Buffers::device<double>> c_;
};

TEST_F(ResponseCphf, NuclearColumnsMatchPyscfSolveMo1) {
  const auto rhs = device(g_.nuclear_rhs);
  auto solved = solve(rhs);
  ASSERT_TRUE(solved) << solved.error().message;
  ASSERT_EQ(solved->residuals.size(), 9u);
  for (std::size_t x = 0; x < solved->residuals.size(); ++x) {
    EXPECT_LE(solved->residuals[x], 10 * kTolerance) << "column " << x;
  }
  const double worst = max_diff(download(*res_, solved->solution), g_.nuclear_solution);
  EXPECT_LE(worst, kSolutionTol) << "max |U - mo1| " << worst << " of max "
                                 << max_abs(g_.nuclear_solution) << " after " << solved->iterations
                                 << " sweeps";
}

TEST_F(ResponseCphf, DipoleColumnsGivePyscfPolarizability) {
  const auto rhs = device(g_.dipole_rhs);
  auto solved = solve(rhs);
  ASSERT_TRUE(solved) << solved.error().message;
  const std::vector<double> u = download(*res_, solved->solution);
  const std::size_t pairs = g_.dipole_rhs.size() / 3;
  // alpha_xy = 4 B_x . U_y, B = -r.
  std::vector<double> alpha(9);
  for (std::size_t y = 0; y < 3; ++y) {
    for (std::size_t x = 0; x < 3; ++x) {
      double sum = 0.0;
      for (std::size_t k = 0; k < pairs; ++k) {
        sum += g_.dipole_rhs[k + pairs * x] * u[k + pairs * y];
      }
      alpha[x + 3 * y] = 4.0 * sum;
    }
  }
  const double scale = max_abs(g_.polarizability);
  ASSERT_GT(scale, 1.0);
  const double worst = max_diff(alpha, g_.polarizability);
  EXPECT_LE(worst, kPolarizabilityRelTol * scale)
      << "max |alpha - PySCF| " << worst << " of max " << scale;
}

TEST_F(ResponseCphf, OneColumnAloneMatchesTheLockstepBlock) {
  const auto block = device(g_.nuclear_rhs);
  const std::size_t pairs = g_.nuclear_rhs.size() / 9;
  // Column 4, the second atom's y: one that stops neither first nor last.
  const std::size_t column = 4;
  const std::vector<double> one(
      g_.nuclear_rhs.begin() + static_cast<std::ptrdiff_t>(column * pairs),
      g_.nuclear_rhs.begin() + static_cast<std::ptrdiff_t>((column + 1) * pairs));
  const auto single = device(one);
  auto all = solve(block);
  auto alone = solve(single);
  ASSERT_TRUE(all) << all.error().message;
  ASSERT_TRUE(alone) << alone.error().message;
  const std::vector<double> u_all = download(*res_, all->solution);
  const std::vector<double> u_alone = download(*res_, alone->solution);
  const std::vector<double> u_column(u_all.begin() + static_cast<std::ptrdiff_t>(column * pairs),
                                     u_all.begin() +
                                         static_cast<std::ptrdiff_t>((column + 1) * pairs));
  EXPECT_LE(max_diff(u_alone, u_column), kLockstepTol * max_abs(u_column));
  EXPECT_LE(alone->iterations, all->iterations);
}

// batched builds each sweep's products as one block: the nuclear and the
// dipole columns, each against the default path's solution, and the nuclear
// ones against PySCF's mo1 too.
TEST_F(ResponseCphf, BatchedMatchesOneColumnAtATime) {
  for (const auto *host : {&g_.nuclear_rhs, &g_.dipole_rhs}) {
    const auto rhs = device(*host);
    auto one = solve(rhs);
    auto batched = solve(rhs, 100, true);
    ASSERT_TRUE(one) << one.error().message;
    ASSERT_TRUE(batched) << batched.error().message;
    EXPECT_EQ(batched->iterations, one->iterations);
    for (std::size_t x = 0; x < batched->residuals.size(); ++x) {
      EXPECT_LE(batched->residuals[x], 10 * kTolerance) << "column " << x;
    }
    const std::vector<double> u_batched = download(*res_, batched->solution);
    const double worst = max_diff(u_batched, download(*res_, one->solution));
    EXPECT_LE(worst, kBatchedTol) << "max |U_batched - U| " << worst;
    std::cout << std::format("[ cphf     ] {} columns: max |U_batched - U| {:.3e}\n",
                             batched->residuals.size(), worst);
    if (host == &g_.nuclear_rhs) {
      EXPECT_LE(max_diff(u_batched, g_.nuclear_solution), kSolutionTol);
    }
  }
}

TEST_F(ResponseCphf, ExhaustedBudgetIsReported) {
  const auto rhs = device(g_.nuclear_rhs);
  auto solved = solve(rhs, 2);
  ASSERT_FALSE(solved);
  EXPECT_EQ(solved.error().kind, fb::ErrorKind::MaxIterationsExceeded);
  EXPECT_NE(solved.error().message.find("response::cphf"), std::string::npos);
}

TEST_F(ResponseCphf, Refusals) {
  const auto rhs = device(g_.nuclear_rhs);
  const std::span<const double> energies{g_.energies};
  const auto expect_refused = [](const fb::Result<fb::response::Cphf> &r, const char *what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
  };
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies, 0, rhs), "nocc 0");
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies, g_.nao, rhs),
                 "no virtuals");
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux - 1, *c_, energies, g_.nocc, rhs),
                 "B of another naux");
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies.first(energies.size() - 1),
                                    g_.nocc, rhs),
                 "C not whole columns of nmo");
  const fb::Buffers::device<double> ragged(g_.nuclear_rhs.size() - 1, res_->device());
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies, g_.nocc, ragged),
                 "a partial column");
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies, g_.nocc, rhs, 0.0),
                 "tolerance 0");
  expect_refused(fb::response::cphf(*res_, *b_, g_.naux, *c_, energies, g_.nocc, rhs, 1e-10, 0),
                 "max_iterations 0");
}

} // namespace
