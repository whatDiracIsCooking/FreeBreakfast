// GPU cases for recipes.einsum: matmul, diagonal, trace, a full reduction to
// a scalar, a strided view, transpose, a three-operand contraction, SC-NEVPT2's
// small-output / large-sum shape and one of its diagonal-plus-eight-label
// shapes, and the accumulate / scale semantics, each against a host loop.
//
// The kernel adds every term with an atomicAdd, so the sum's order varies and
// nothing here is compared bit for bit: every check is at kTol, absolute, on
// entries in [-1, 1] whose sums stay within a few hundred.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;

namespace {

namespace fb = free_breakfast;

constexpr double kTol = 1e-11;

std::vector<double> random_values(const std::size_t n, const std::uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> v(n);
  for (double &x : v) {
    x = dist(gen);
  }
  return v;
}

std::size_t element_count(const std::vector<std::int64_t> &dims) {
  std::size_t n = 1;
  for (const std::int64_t d : dims) {
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

// A device copy of a host array, queued on res's stream.
class Uploaded {
public:
  Uploaded(fb::DeviceResources &res, const std::vector<double> &host)
      : buffer_(std::max<std::size_t>(host.size(), 1), res.device()) {
    EXPECT_EQ(wwr::wwrMemcpyAsync(buffer_.data(), host.data(), host.size() * sizeof(double),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
  }
  [[nodiscard]] double *data() { return buffer_.data(); }

private:
  fb::Buffers::device<double> buffer_;
};

std::vector<double> download(fb::DeviceResources &res, const double *device, std::size_t n) {
  std::vector<double> host(n);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, n * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// out_init + alpha * einsum(subscripts, operands) on the device, read back.
std::vector<double>
run_einsum(fb::DeviceResources &res, const std::string_view subscripts,
           const std::vector<std::pair<std::vector<double>, std::vector<std::int64_t>>> &operands,
           const std::vector<std::int64_t> &out_dims, const std::vector<double> &out_init,
           const double alpha) {
  // A deque, so an Uploaded is never moved once its pointer is taken.
  std::deque<Uploaded> uploaded;
  std::vector<fb::EinsumOperand> ops;
  for (const auto &[values, dims] : operands) {
    uploaded.emplace_back(res, values);
    ops.push_back({uploaded.back().data(), dims, {}});
  }
  Uploaded out{res, out_init};
  const fb::Status status = fb::einsum(res.stream(), subscripts, ops, out.data(), out_dims, alpha);
  EXPECT_TRUE(status.has_value()) << status.error().message;
  return download(res, out.data(), out_init.size());
}

std::vector<double>
run_einsum(fb::DeviceResources &res, const std::string_view subscripts,
           const std::vector<std::pair<std::vector<double>, std::vector<std::int64_t>>> &operands,
           const std::vector<std::int64_t> &out_dims) {
  return run_einsum(res, subscripts, operands, out_dims,
                    std::vector<double>(element_count(out_dims), 0.0), 1.0);
}

void expect_near_all(const std::vector<double> &got, const std::vector<double> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], kTol) << "at " << i;
  }
}

TEST(RecipesEinsum, Matmul) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::size_t m = 7;
  constexpr std::size_t k = 33;
  constexpr std::size_t n = 5;
  const auto a = random_values(m * k, 1);
  const auto b = random_values(k * n, 2);
  std::vector<double> want(m * n, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t p = 0; p < k; ++p) {
        want[(i * n) + j] += a[(i * k) + p] * b[(p * n) + j];
      }
    }
  }
  expect_near_all(run_einsum(res, "ik,kj->ij", {{a, {m, k}}, {b, {k, n}}}, {m, n}), want);
}

TEST(RecipesEinsum, DiagonalAndTrace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::size_t n = 9;
  const auto a = random_values(n * n, 3);
  std::vector<double> diagonal(n);
  double trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    diagonal[i] = a[(i * n) + i];
    trace += a[(i * n) + i];
  }
  expect_near_all(run_einsum(res, "ii->i", {{a, {n, n}}}, {n}), diagonal);
  expect_near_all(run_einsum(res, "ii->", {{a, {n, n}}}, {}), {trace});
}

TEST(RecipesEinsum, FullReductionToAScalar) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::size_t m = 40;
  constexpr std::size_t n = 31;
  const auto a = random_values(m * n, 4);
  const auto b = random_values(m * n, 5);
  double dot = 0.0;
  for (std::size_t i = 0; i < m * n; ++i) {
    dot += a[i] * b[i];
  }
  expect_near_all(run_einsum(res, "ij,ij->", {{a, {m, n}}, {b, {m, n}}}, {}), {dot});
}

TEST(RecipesEinsum, ReadsAStridedViewInPlace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // Columns [c0, c0 + k) of an m x wide row-major matrix, times a k-vector.
  constexpr std::size_t m = 6;
  constexpr std::size_t wide = 11;
  constexpr std::size_t c0 = 3;
  constexpr std::size_t k = 5;
  const auto big = random_values(m * wide, 6);
  const auto v = random_values(k, 7);
  std::vector<double> want(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      want[i] += big[(i * wide) + c0 + p] * v[p];
    }
  }
  Uploaded d_big{res, big};
  Uploaded d_v{res, v};
  Uploaded out{res, std::vector<double>(m, 0.0)};
  const fb::EinsumOperand view{d_big.data() + c0, {m, k}, {wide, 1}};
  const fb::Status status =
      fb::einsum(res.stream(), "ip,p->i", {view, {d_v.data(), {k}, {}}}, out.data(), {m});
  ASSERT_TRUE(status.has_value()) << status.error().message;
  expect_near_all(download(res, out.data(), m), want);
}

TEST(RecipesEinsum, TransposeIsNumpysAxesConvention) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // The rank-6 permutation SC-NEVPT2's a16 is built with, on unequal extents
  // so a wrong reading of `axes` cannot pass.
  const std::vector<std::int64_t> dims{2, 3, 4, 2, 3, 2};
  const std::vector<std::int32_t> axes{1, 4, 0, 2, 5, 3};
  const auto x = random_values(element_count(dims), 8);
  std::vector<std::int64_t> out_dims;
  for (const std::int32_t a : axes) {
    out_dims.push_back(dims[static_cast<std::size_t>(a)]);
  }
  const std::size_t total = x.size();
  const auto out_init = random_values(total, 9);
  constexpr double kAlpha = -1.5;
  // out[i0..i5] += alpha * x[j], j's axis axes[d] taking out's index d.
  std::vector<double> want = out_init;
  for (std::size_t flat = 0; flat < total; ++flat) {
    std::array<std::size_t, 6> idx{};
    std::size_t rest = flat;
    for (std::size_t d = 6; d-- > 0;) {
      idx[d] = rest % static_cast<std::size_t>(out_dims[d]);
      rest /= static_cast<std::size_t>(out_dims[d]);
    }
    std::array<std::size_t, 6> src{};
    for (std::size_t d = 0; d < 6; ++d) {
      src[static_cast<std::size_t>(axes[d])] = idx[d];
    }
    std::size_t at = 0;
    for (std::size_t d = 0; d < 6; ++d) {
      at = (at * static_cast<std::size_t>(dims[d])) + src[d];
    }
    want[flat] += kAlpha * x[at];
  }
  Uploaded d_x{res, x};
  Uploaded out{res, out_init};
  const fb::Status status =
      fb::transpose(res.stream(), {d_x.data(), dims, {}}, axes, out.data(), kAlpha);
  ASSERT_TRUE(status.has_value()) << status.error().message;
  expect_near_all(download(res, out.data(), total), want);
}

TEST(RecipesEinsum, ThreeOperandContraction) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // SC-NEVPT2's "ap,lbkl,qk->pqab": a diagonal on the middle operand too.
  constexpr std::size_t na = 5;
  constexpr std::size_t np = 4;
  constexpr std::size_t nl = 3;
  const auto ap = random_values(na * np, 10);
  const auto lbkl = random_values(nl * na * np * nl, 11);
  const auto qk = random_values(np * np, 12);
  // out (p, q, a, b); a, b in na; p, q, k in np; l in nl.
  std::vector<double> want(np * np * na * na, 0.0);
  for (std::size_t p = 0; p < np; ++p) {
    for (std::size_t q = 0; q < np; ++q) {
      for (std::size_t a = 0; a < na; ++a) {
        for (std::size_t b = 0; b < na; ++b) {
          double sum = 0.0;
          for (std::size_t l = 0; l < nl; ++l) {
            for (std::size_t k = 0; k < np; ++k) {
              sum += ap[(a * np) + p] * lbkl[(((l * na + b) * np + k) * nl) + l] * qk[(q * np) + k];
            }
          }
          want[(((p * np + q) * na + a) * na) + b] = sum;
        }
      }
    }
  }
  expect_near_all(run_einsum(res, "ap,lbkl,qk->pqab",
                             {{ap, {na, np}}, {lbkl, {nl, na, np, nl}}, {qk, {np, np}}},
                             {np, np, na, na}),
                  want);
}

TEST(RecipesEinsum, SmallOutputLargeSum) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // Sr's "ipqr,pqrabc,iabc->i": an output of ni summed over six labels.
  constexpr std::size_t ni = 4;
  constexpr std::size_t no = 6;
  constexpr std::size_t nv = 7;
  const auto ipqr = random_values(ni * no * no * no, 13);
  const auto pqrabc = random_values(no * no * no * nv * nv * nv, 14);
  const auto iabc = random_values(ni * nv * nv * nv, 15);
  constexpr std::size_t o3 = no * no * no;
  constexpr std::size_t v3 = nv * nv * nv;
  std::vector<double> want(ni, 0.0);
  for (std::size_t i = 0; i < ni; ++i) {
    for (std::size_t pqr = 0; pqr < o3; ++pqr) {
      for (std::size_t abc = 0; abc < v3; ++abc) {
        want[i] += ipqr[(i * o3) + pqr] * pqrabc[(pqr * v3) + abc] * iabc[(i * v3) + abc];
      }
    }
  }
  expect_near_all(
      run_einsum(
          res, "ipqr,pqrabc,iabc->i",
          {{ipqr, {ni, no, no, no}}, {pqrabc, {no, no, no, nv, nv, nv}}, {iabc, {ni, nv, nv, nv}}},
          {ni}),
      want);
}

TEST(RecipesEinsum, DiagonalWithEightDistinctLabels) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  // energy.cpp's "jbij,rpqiac->pqrabc": every limit but the operand count at
  // once -- eight distinct labels, a rank-6 operand and output, and j
  // repeated on the first operand. One extent n throughout, as there.
  constexpr std::size_t n = 3;
  const auto jbij = random_values(n * n * n * n, 16);
  const auto rpqiac = random_values(n * n * n * n * n * n, 17);
  const auto at4 = [](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return (((a * n + b) * n + c) * n) + d;
  };
  const auto at6 = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d, std::size_t e,
                       std::size_t f) { return (at4(a, b, c, d) * n * n) + (e * n) + f; };
  std::vector<double> want(n * n * n * n * n * n, 0.0);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t a = 0; a < n; ++a) {
          for (std::size_t b = 0; b < n; ++b) {
            for (std::size_t c = 0; c < n; ++c) {
              double sum = 0.0;
              for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                  sum += jbij[at4(j, b, i, j)] * rpqiac[at6(r, p, q, i, a, c)];
                }
              }
              want[at6(p, q, r, a, b, c)] = sum;
            }
          }
        }
      }
    }
  }
  const std::vector<std::int64_t> six(6, n);
  expect_near_all(
      run_einsum(res, "jbij,rpqiac->pqrabc", {{jbij, {n, n, n, n}}, {rpqiac, six}}, six), want);
}

TEST(RecipesEinsum, AccumulatesAndScales) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::size_t m = 6;
  constexpr std::size_t k = 9;
  const auto a = random_values(m * k, 18);
  const auto b = random_values(k * m, 19);
  const auto out_init = random_values(m * m, 20);
  constexpr double kAlpha = -0.75;
  std::vector<double> product(m * m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      for (std::size_t p = 0; p < k; ++p) {
        product[(i * m) + j] += a[(i * k) + p] * b[(p * m) + j];
      }
    }
  }
  // Into a non-zero output, scaled, twice over: out_init + 2 alpha A B.
  Uploaded d_a{res, a};
  Uploaded d_b{res, b};
  Uploaded out{res, out_init};
  const std::vector<fb::EinsumOperand> ops{{d_a.data(), {m, k}, {}}, {d_b.data(), {k, m}, {}}};
  for (std::int32_t pass = 0; pass < 2; ++pass) {
    const fb::Status status =
        fb::einsum(res.stream(), "ik,kj->ij", ops, out.data(), {m, m}, kAlpha);
    ASSERT_TRUE(status.has_value()) << status.error().message;
  }
  std::vector<double> want(m * m);
  for (std::size_t i = 0; i < m * m; ++i) {
    want[i] = out_init[i] + (2.0 * kAlpha * product[i]);
  }
  expect_near_all(download(res, out.data(), m * m), want);
}

} // namespace
