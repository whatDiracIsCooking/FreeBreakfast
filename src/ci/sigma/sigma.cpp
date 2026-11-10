/// @file
/// @brief Sigma's uploads and launches, and unpack and pack.

module;

// The launchers, defined in ci_sigma.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "ci_sigma_bridge.h"

module free_breakfast.ci.sigma;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kOne, kZero
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.recipes.linear_operator;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast::ci {

static_assert(LinearOperator<Sigma, double>);

Status unpack(DeviceResources &res, const int32_t n_strings, const int32_t k, const double *const x,
              double *const y) {
  if (n_strings < 0 || k < 0) {
    return err_config("ci::unpack: {} vectors over {} strings", k, n_strings);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::unpack(res.stream(), n_strings, k, x, y));
}

Status pack(DeviceResources &res, const int32_t n_strings, const int32_t k, const double *const x,
            double *const y) {
  if (n_strings < 0 || k < 0) {
    return err_config("ci::pack: {} vectors over {} strings", k, n_strings);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::pack(res.stream(), n_strings, k, x, y));
}

// The bytes of D one chunk of apply's rows may take: big enough that the gemm
// over a chunk is a wide one, small enough that G dominates the scratch.
constexpr size_t kCiSigmaChunkBytes = size_t{64} << 20;

// The rows of D one chunk of g_rows rows holds, elements of element_bytes.
size_t ci_sigma_chunk(const size_t element_bytes, const size_t pairs, const size_t g_rows) {
  return std::clamp<size_t>(kCiSigmaChunkBytes / (element_bytes * pairs), 1, g_rows);
}

// The bytes of apply's scratch for G over g_rows rows and D over one chunk of
// them, elements of element_bytes each, D starting a whole alignment step past
// G.
size_t ci_sigma_workspace(const size_t element_bytes, const size_t pairs, const size_t g_rows) {
  return math::align_up(g_rows * (pairs + 1) * element_bytes,
                        DeviceResources::kWorkspaceAlignment) +
         math::align_up(ci_sigma_chunk(element_bytes, pairs, g_rows) * pairs * element_bytes,
                        DeviceResources::kWorkspaceAlignment);
}

// A blocked apply's blocks: the strings per alpha block and per beta block,
// and the rows of G the larger holds.
struct CiSigmaBlocks {
  int32_t alpha_block;
  int32_t beta_block;
  size_t rows;
};

// The blocks of a blocked apply over elements of element_bytes under limit:
// the most rows of G a block may hold, at least one, found by bisection,
// whole strings of one spin against every string of the other.
CiSigmaBlocks ci_sigma_blocks(const size_t element_bytes, const size_t pairs, const int32_t n_alpha,
                              const int32_t n_beta, const bool symmetric, const size_t limit) {
  size_t fit = 1;
  size_t top = static_cast<size_t>(n_alpha) * static_cast<size_t>(n_beta);
  while (fit < top) {
    const size_t mid = fit + ((top - fit) / 2) + 1;
    if (ci_sigma_workspace(element_bytes, pairs, mid) <= limit) {
      fit = mid;
    } else {
      top = mid - 1;
    }
  }
  const auto block_strings = [&](const int32_t n_major, const int32_t n_minor) {
    return static_cast<int32_t>(
        std::clamp<size_t>(fit / static_cast<size_t>(n_minor), 1, static_cast<size_t>(n_major)));
  };
  CiSigmaBlocks out{.alpha_block = block_strings(n_alpha, n_beta),
                    .beta_block = block_strings(n_beta, n_alpha),
                    .rows = 0};
  out.rows = static_cast<size_t>(out.alpha_block) * static_cast<size_t>(n_beta);
  if (!symmetric) {
    out.rows =
        std::max(out.rows, static_cast<size_t>(out.beta_block) * static_cast<size_t>(n_alpha));
  }
  return out;
}

// The bytes one apply over elements of element_bytes takes in the layout:
// G for every row and a chunk of D where that fits limit, else the blocks'.
size_t ci_sigma_need(const size_t element_bytes, const size_t pairs, const int32_t n_alpha,
                     const int32_t n_beta, const bool symmetric, const size_t limit) {
  const size_t rows = symmetric ? static_cast<size_t>(n_alpha) * (n_alpha + 1) / 2
                                : static_cast<size_t>(n_alpha) * static_cast<size_t>(n_beta);
  const size_t whole = ci_sigma_workspace(element_bytes, pairs, rows);
  if (whole <= limit) {
    return whole;
  }
  return ci_sigma_workspace(
      element_bytes, pairs,
      ci_sigma_blocks(element_bytes, pairs, n_alpha, n_beta, symmetric, limit).rows);
}

// The packed index of orbital pair (t, u), symmetric in the two.
size_t ci_pair(const size_t t, const size_t u) {
  const size_t hi = std::max(t, u);
  return (hi * (hi + 1) / 2) + std::min(t, u);
}

template <typename T>
Status ci_upload(const wwr::wwrStream_t stream, Buffers::device<T> &to,
                 const std::vector<T> &from) {
  return gpu_status(wwr::wwrMemcpyAsync(to.data(), from.data(), sizeof(T) * from.size(),
                                        wwr::wwrMemcpyHostToDevice, stream));
}

Sigma::Sigma(DeviceResources &res, const DeviceStringSet &alpha, const DeviceStringSet &beta)
    : dev_idx_{res.dev_idx()}, n_orbitals_{alpha.n_orbitals()}, n_alpha_{alpha.size()},
      n_beta_{beta.size()}, alpha_per_{alpha.replacements_per_string()},
      beta_per_{beta.replacements_per_string()},
      one_set_{alpha.n_electrons() == beta.n_electrons()},
      alpha_links_{alpha.device_replacements()}, beta_links_{beta.device_replacements()},
      alpha_strings_{alpha.device_strings()}, beta_strings_{beta.device_strings()},
      contraction_(ci_pair(static_cast<size_t>(n_orbitals_), 0) *
                       (ci_pair(static_cast<size_t>(n_orbitals_), 0) + 1),
                   res.device()),
      contraction_fp32_(contraction_.num_elements(), res.device()),
      diagonal_terms_(static_cast<size_t>(n_orbitals_) *
                          (1 + (2 * static_cast<size_t>(n_orbitals_))),
                      res.device()),
      apply_scratch_(0, res.device()) {}

Result<Sigma> Sigma::create(DeviceResources &res, const DeviceStringSet &alpha,
                            const DeviceStringSet &beta) {
  if (alpha.dev_idx() != res.dev_idx() || beta.dev_idx() != res.dev_idx()) {
    return err_config("ci::Sigma: strings are on devices {} and {}, but the call passed device {}",
                      alpha.dev_idx(), beta.dev_idx(), res.dev_idx());
  }
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("ci::Sigma: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("ci::Sigma: the strings span no orbitals");
  }
  if (static_cast<int64_t>(alpha.size()) * beta.size() > std::numeric_limits<int32_t>::max()) {
    return err_config("ci::Sigma: {} x {} determinants is more than an int32_t counts",
                      alpha.size(), beta.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return Sigma(res, alpha, beta);
}

Status Sigma::set_integrals(DeviceResources &res, const double e_core,
                            const std::span<const double> h_eff,
                            const std::span<const double> eri) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("ci::Sigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  const auto n = static_cast<size_t>(n_orbitals_);
  if (h_eff.size() != n * n || eri.size() != n * n * n * n) {
    return err_config("ci::Sigma: {} orbitals want h_eff of {} and eri of {}, given {} and {}", n,
                      n * n, n * n * n * n, h_eff.size(), eri.size());
  }
  const auto at = [&](size_t t, size_t u, size_t v, size_t w) {
    return eri[((t * n + u) * n + v) * n + w];
  };

  // Column pair(t, u) holds ½ (tu|vw) down the rows pair(v, w), and the last
  // column k_vw = h_vw − ½ Σ_u (vu|uw): the one-electron part of H, packed.
  const size_t n_pairs = ci_pair(n, 0);
  std::vector<double> contraction(n_pairs * (n_pairs + 1));
  for (size_t v = 0; v < n; ++v) {
    for (size_t w = 0; w <= v; ++w) {
      const size_t row = ci_pair(v, w);
      for (size_t t = 0; t < n; ++t) {
        for (size_t u = 0; u <= t; ++u) {
          contraction[(ci_pair(t, u) * n_pairs) + row] = 0.5 * at(t, u, v, w);
        }
      }
      double sum = 0.0;
      for (size_t u = 0; u < n; ++u) {
        sum += at(v, u, u, w);
      }
      contraction[(n_pairs * n_pairs) + row] = h_eff[v * n + w] - 0.5 * sum;
    }
  }
  const std::vector<float> contraction_fp32{contraction.begin(), contraction.end()};
  std::vector<double> terms(n * (1 + 2 * n));
  for (size_t i = 0; i < n; ++i) {
    terms[i] = h_eff[i * n + i];
    for (size_t j = 0; j < n; ++j) {
      terms[n + (i * n) + j] = at(i, i, j, j);
      terms[n + (n * n) + (i * n) + j] = at(i, j, j, i);
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  // A failed upload leaves the tables half-written, so they count as unset
  // until all three are in.
  has_integrals_ = false;
  FBF_TRY(ci_upload(res.stream(), contraction_, contraction));
  FBF_TRY(ci_upload(res.stream(), contraction_fp32_, contraction_fp32));
  FBF_TRY(ci_upload(res.stream(), diagonal_terms_, terms));
  FBF_TRY(res.sync());
  e_core_ = e_core;
  has_integrals_ = true;
  return {};
}

template <typename T>
Status Sigma::apply_as(DeviceResources &res, const int32_t k, const double *const x,
                       double *const y, const T *const contraction, const Layout layout) {
  if (!has_integrals_) {
    return err_config("ci::Sigma: apply before set_integrals");
  }
  if (res.dev_idx() != dev_idx_) {
    return err_config("ci::Sigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  if (k < 0) {
    return err_config("ci::Sigma: a block of {} vectors", k);
  }
  const bool symmetric = layout == Layout::symmetric;
  if (symmetric && !one_set_) {
    return err_config("ci::Sigma: the symmetric layout needs one set of strings for both spins");
  }
  if (k == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // A vector's rows: every determinant, or the triangle's.
  const int32_t rows = size(layout);
  const auto n_pairs = static_cast<int32_t>(ci_pair(static_cast<size_t>(n_orbitals_), 0));
  const auto pairs = static_cast<size_t>(n_pairs);
  const int32_t chunk = static_cast<int32_t>(
      std::clamp<size_t>(kCiSigmaChunkBytes / (sizeof(T) * pairs), 1, static_cast<size_t>(rows)));

  // The scratch, sized once for apply and apply_fp32 alike, so a float call
  // and then a double one share it. A growth (a call in a layout of more
  // rows) frees the old before it allocates the new.
  if (const size_t bytes =
          sigma_apply_bytes(n_orbitals_, n_alpha_, n_beta_, layout, scratch_limit_);
      apply_scratch_.num_elements() < bytes) {
    apply_scratch_ = Buffers::device<std::byte>(0, res.device());
    apply_scratch_ = Buffers::device<std::byte>(bytes, res.device());
  }
  if (ci_sigma_workspace(sizeof(T), pairs, static_cast<size_t>(rows)) > scratch_limit_) {
    return apply_blocked(res, k, x, y, contraction, layout);
  }

  // G for one vector's rows, then D for one chunk of them. σ_I reads G only
  // over I's own vector, so the vectors go one at a time and G is never wider
  // than one: k · dim · (P + 1) doubles would be 60 GB at CAS(14,14), k = 6.
  // In the symmetric layout it is the triangle's rows alone.
  T *const g = reinterpret_cast<T *>(apply_scratch_.data());
  T *const d = reinterpret_cast<T *>(
      apply_scratch_.data() + math::align_up(static_cast<size_t>(rows) * (pairs + 1) * sizeof(T),
                                             DeviceResources::kWorkspaceAlignment));

  for (int32_t j = 0; j < k; ++j) {
    const double *const xj = x + (static_cast<size_t>(j) * static_cast<size_t>(rows));
    double *const yj = y + (static_cast<size_t>(j) * static_cast<size_t>(rows));
    for (int32_t begin = 0; begin < rows; begin += chunk) {
      const int32_t count = std::min(chunk, rows - begin);
      FBF_TRY(gpu_status(
          wwr::wwrMemsetAsync(d, 0, sizeof(T) * static_cast<size_t>(count) * pairs, stream)));
      FBF_TRY(
          gpu_status(device::gather_d(stream, symmetric, begin, count, rows, n_beta_, alpha_links_,
                                      alpha_per_, beta_links_, beta_per_, false, xj, d)));
      FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, count, n_pairs + 1, n_pairs,
          &kOne<T>, d, count, contraction, n_pairs, &kZero<T>, g + begin, rows)));
    }
    FBF_TRY(gpu_status(device::sigma_from_g(stream, symmetric, rows, rows, n_beta_, n_pairs,
                                            alpha_links_, alpha_per_, beta_links_, beta_per_,
                                            e_core_, xj, g, yj)));
  }
  return {};
}

template <typename T>
Status Sigma::apply_blocked(DeviceResources &res, const int32_t k, const double *const x,
                            double *const y, const T *const contraction, const Layout layout) {
  const wwr::wwrStream_t stream = res.stream();
  const bool symmetric = layout == Layout::symmetric;
  const int32_t rows = size(layout);
  const auto n_pairs = static_cast<int32_t>(ci_pair(static_cast<size_t>(n_orbitals_), 0));
  const auto pairs = static_cast<size_t>(n_pairs);

  // A pass is one spin's strings in blocks, each against every string of the
  // other spin: in the full layout the α blocks (β links), then the β blocks
  // (α links); in the symmetric one the one set's alone, serving both. The
  // blocks are the largest whose G and D fit the limit (ci_sigma_blocks), in
  // the scratch apply_as sized for them.
  const CiSigmaBlocks blocks =
      ci_sigma_blocks(sizeof(T), pairs, n_alpha_, n_beta_, symmetric, scratch_limit_);
  const int32_t alpha_block = blocks.alpha_block;
  const int32_t beta_block = blocks.beta_block;
  const size_t block_rows = blocks.rows;
  const size_t chunk = ci_sigma_chunk(sizeof(T), pairs, block_rows);
  T *const g = reinterpret_cast<T *>(apply_scratch_.data());
  T *const d = reinterpret_cast<T *>(
      apply_scratch_.data() +
      math::align_up(block_rows * (pairs + 1) * sizeof(T), DeviceResources::kWorkspaceAlignment));

  for (int32_t j = 0; j < k; ++j) {
    const double *const xj = x + (static_cast<size_t>(j) * static_cast<size_t>(rows));
    double *const yj = y + (static_cast<size_t>(j) * static_cast<size_t>(rows));
    // Every block adds its share of σ.
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(yj, 0, sizeof(double) * static_cast<size_t>(rows), stream)));
    for (const bool beta_major : {false, true}) {
      if (beta_major && symmetric) {
        break;
      }
      const int32_t n_major = beta_major ? n_beta_ : n_alpha_;
      const int32_t n_minor = beta_major ? n_alpha_ : n_beta_;
      const int32_t strings = beta_major ? beta_block : alpha_block;
      for (int32_t first = 0; first < n_major; first += strings) {
        const int32_t count = std::min(strings, n_major - first);
        const int32_t rows_here = count * n_minor;
        const int32_t step = static_cast<int32_t>(std::min(chunk, static_cast<size_t>(rows_here)));
        for (int32_t begin = 0; begin < rows_here; begin += step) {
          const int32_t many = std::min(step, rows_here - begin);
          FBF_TRY(gpu_status(
              wwr::wwrMemsetAsync(d, 0, sizeof(T) * static_cast<size_t>(many) * pairs, stream)));
          FBF_TRY(gpu_status(device::gather_d_block(stream, symmetric, beta_major, first, begin,
                                                    many, n_alpha_, n_beta_, alpha_links_,
                                                    alpha_per_, beta_links_, beta_per_, xj, d)));
          FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(
              res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, many, n_pairs + 1, n_pairs,
              &kOne<T>, d, many, contraction, n_pairs, &kZero<T>, g + begin, rows_here)));
        }
        FBF_TRY(gpu_status(device::sigma_from_g_block(
            stream, symmetric, beta_major, first, count, n_alpha_, n_beta_, n_pairs, alpha_links_,
            alpha_per_, beta_links_, beta_per_, e_core_, xj, g, yj)));
      }
    }
  }
  return {};
}

Status Sigma::apply(DeviceResources &res, const int32_t k, const double *const x, double *const y,
                    const Layout layout) {
  return apply_as(res, k, x, y, contraction_.data(), layout);
}

Status Sigma::apply_fp32(DeviceResources &res, const int32_t k, const double *const x,
                         double *const y, const Layout layout) {
  return apply_as(res, k, x, y, contraction_fp32_.data(), layout);
}

Status Sigma::diagonal(DeviceResources &res, double *const diagonal) {
  if (!has_integrals_) {
    return err_config("ci::Sigma: diagonal before set_integrals");
  }
  if (res.dev_idx() != dev_idx_) {
    return err_config("ci::Sigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::diagonal(res.stream(), n_orbitals_, n_alpha_, n_beta_, alpha_strings_,
                                     beta_strings_, e_core_, diagonal_terms_.data(), diagonal));
}

Status Sigma::rdm1(DeviceResources &res, const double *const c, double *const gamma,
                   const Layout layout) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("ci::Sigma: tables are on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  const bool symmetric = layout == Layout::symmetric;
  if (symmetric && !one_set_) {
    return err_config("ci::Sigma: the symmetric layout needs one set of strings for both spins");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // Every determinant, or the triangle's, each off-diagonal row scaled by √2
  // so that with the packed c's own √2 it carries the weight 2 of itself and
  // its mirror.
  const int32_t rows = size(layout);
  const auto n_pairs = static_cast<int32_t>(ci_pair(static_cast<size_t>(n_orbitals_), 0));
  const auto pairs = static_cast<size_t>(n_pairs);
  const auto chunk =
      static_cast<int32_t>(ci_sigma_chunk(sizeof(double), pairs, static_cast<size_t>(rows)));

  // The packed sums, then D for one chunk.
  const auto [packed, d] = FBF_TRY(
      res.device_regions<double>("ci::Sigma::rdm1", pairs, static_cast<size_t>(chunk) * pairs));

  // The first chunk's gemv overwrites the sums, every later one adds to them.
  for (int32_t begin = 0; begin < rows; begin += chunk) {
    const int32_t count = std::min(chunk, rows - begin);
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(d, 0, sizeof(double) * static_cast<size_t>(count) * pairs, stream)));
    FBF_TRY(
        gpu_status(device::gather_d(stream, symmetric, begin, count, rows, n_beta_, alpha_links_,
                                    alpha_per_, beta_links_, beta_per_, true, c, d)));
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_T, count, n_pairs, &kOne<double>, d, count, c + begin, 1,
        begin == 0 ? &kZero<double> : &kOne<double>, packed, 1)));
  }
  return gpu_status(device::unpack_rdm1(stream, n_orbitals_, packed, gamma));
}

MemoryFootprint sigma_footprint(const int32_t n_orbitals, const int32_t n_alpha_strings,
                                const int32_t n_beta_strings, const Layout rdm1_layout) {
  const size_t pairs = ci_pair(static_cast<size_t>(n_orbitals), 0);
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t rows = rdm1_layout == Layout::symmetric
                          ? static_cast<size_t>(n_alpha_strings) * (n_alpha_strings + 1) / 2
                          : static_cast<size_t>(n_alpha_strings) * n_beta_strings;
  MemoryFootprint out;
  // The contraction in double and in float, and the diagonal's n + 2 n².
  out.device.persistent = (pairs * (pairs + 1) * (sizeof(double) + sizeof(float))) +
                          (sizeof(double) * n * (1 + (2 * n)));
  // rdm1's packed sums and D for one chunk.
  out.device.workspace =
      math::align_up(pairs * sizeof(double), DeviceResources::kWorkspaceAlignment) +
      math::align_up(ci_sigma_chunk(sizeof(double), pairs, rows) * pairs * sizeof(double),
                     DeviceResources::kWorkspaceAlignment);
  return out;
}

size_t sigma_apply_bytes(const int32_t n_orbitals, const int32_t n_alpha_strings,
                         const int32_t n_beta_strings, const Layout layout, const size_t limit) {
  const size_t pairs = ci_pair(static_cast<size_t>(n_orbitals), 0);
  const bool symmetric = layout == Layout::symmetric;
  return std::max(
      ci_sigma_need(sizeof(double), pairs, n_alpha_strings, n_beta_strings, symmetric, limit),
      ci_sigma_need(sizeof(float), pairs, n_alpha_strings, n_beta_strings, symmetric, limit));
}

} // namespace free_breakfast::ci
