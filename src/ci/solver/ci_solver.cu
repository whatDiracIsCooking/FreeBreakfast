// The device half of ci_solve's per-iteration work: S² on a block of CI
// vectors, which the spin penalty adds into sigma's output, and the diagonal
// preconditioner. One source for both backends -- nvcc under CUDA, clang -x
// hip under HIP.
//
// S² is a gather, one thread per row: (Σ_pq E^β_pq E^α_qp x)_J walks J's α
// links a†_p a_q and, for each, the β partner a†_q a_p of J's β string, read
// from that string's slot table. S² is real symmetric, so J's own links reach
// every K with <J|S²|K> nonzero. A thread owns its row, so nothing is atomic
// and the summation order is fixed.
//
// S² and the penalized diagonal come in the two layouts of CiLayout, as a
// template on Symmetric: in the symmetric one a row is a triangle element
// (a, b), a >= b, and a link below the triangle reads its mirror, as the GPU
// sigma's gathers do (see ../sigma/ci_symmetric.h).

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"

#include "ci_solver_bridge.h"
#include "ci_symmetric.h"

namespace free_breakfast::device {

namespace {

constexpr uint32_t kCiBlock = kFourWarps;

// The fields of one α link (see the bridge header).
constexpr int32_t kLinkWidth = 3;

// The preconditioner's floor on |theta − diag|: a root concentrated on one
// determinant has theta ≈ its diagonal.
constexpr double kDenominatorFloor = 1e-3;

// Determinant (a, b) of element @p det of a vector, in the layout's terms.
template <bool Symmetric>
__device__ void ci_strings(const int32_t det, const int32_t n_beta, int32_t &a, int32_t &b) {
  if constexpr (Symmetric) {
    ci_triangle_strings(det, a, b);
  } else {
    a = det / n_beta;
    b = det % n_beta;
  }
}

template <bool Symmetric>
__global__ void ci_spin_square_kernel(const int32_t rows, const int32_t n_det, const int32_t n_beta,
                                      const int32_t n, const int32_t *const alpha_links,
                                      const int32_t alpha_per, const int32_t *const beta_slots,
                                      const double constant, const double shift, const double scale,
                                      const bool accumulate, const double *const x,
                                      double *const y) {
  const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (r >= rows) {
    return;
  }
  const auto row = static_cast<size_t>(r);
  const auto det = static_cast<int32_t>(row % static_cast<size_t>(n_det));
  const size_t vector_base = row - static_cast<size_t>(det);
  int32_t ja = 0;
  int32_t jb = 0;
  ci_strings<Symmetric>(det, n_beta, ja, jb);

  // Σ sign · c_K over J's links. In the symmetric layout c_K is K's stored
  // element (its mirror's, below the triangle) over √2 off the diagonal, and
  // the row's own is stored times √2 off the diagonal.
  double links = 0.0;
  const int32_t *a = alpha_links + (static_cast<size_t>(ja) * alpha_per * kLinkWidth);
  const int32_t *const slots =
      beta_slots + (static_cast<size_t>(jb) * static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    const int32_t slot = slots[a[1]];
    if (slot == 0) {
      continue;
    }
    const int32_t kb = (slot < 0 ? -slot : slot) - 1;
    const double sign = static_cast<double>(slot < 0 ? -a[2] : a[2]);
    if constexpr (Symmetric) {
      const double weight = a[0] == kb ? 1.0 : kCiInvSqrt2;
      links += sign * (weight * x[vector_base + ci_triangle(a[0], kb)]);
    } else {
      links += sign * x[vector_base + (static_cast<size_t>(a[0]) * static_cast<size_t>(n_beta)) +
                        static_cast<size_t>(kb)];
    }
  }
  double sum = (constant - shift) * x[row];
  if constexpr (Symmetric) {
    sum -= ja == jb ? links : kCiSqrt2 * links;
  } else {
    sum -= links;
  }
  y[row] = (accumulate ? y[row] : 0.0) + (scale * sum);
}

__global__ void ci_precondition_kernel(const int32_t n, const double theta,
                                       const double *const shifted, const double *const residual,
                                       double *const correction) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n) {
    return;
  }
  double denominator = theta - shifted[i];
  if (denominator < kDenominatorFloor && denominator > -kDenominatorFloor) {
    denominator = denominator < 0.0 ? -kDenominatorFloor : kDenominatorFloor;
  }
  correction[i] = residual[i] / denominator;
}

template <bool Symmetric>
__global__ void ci_spin_diagonal_kernel(const int32_t n_det, const int32_t n_beta, const int32_t n,
                                        const int32_t *const alpha_links, const int32_t alpha_per,
                                        const int32_t *const beta_slots, const double constant,
                                        const double shift, const double scale,
                                        const double *const diagonal, double *const shifted) {
  const int64_t det = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (det >= n_det) {
    return;
  }
  int32_t ja = 0;
  int32_t jb = 0;
  ci_strings<Symmetric>(static_cast<int32_t>(det), n_beta, ja, jb);
  // Only a†_p a_p leaves the α string where it was, and its partner a†_p a_p
  // survives the β string exactly when p is occupied there too.
  int32_t doubles = 0;
  const int32_t *a = alpha_links + (static_cast<size_t>(ja) * alpha_per * kLinkWidth);
  const int32_t *const slots =
      beta_slots + (static_cast<size_t>(jb) * static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int32_t e = 0; e < alpha_per; ++e, a += kLinkWidth) {
    if (a[0] == ja && slots[a[1]] != 0) {
      ++doubles;
    }
  }
  const size_t full =
      (static_cast<size_t>(ja) * static_cast<size_t>(n_beta)) + static_cast<size_t>(jb);
  shifted[det] = diagonal[full] + (scale * (constant - static_cast<double>(doubles) - shift));
}

// One (value, address) candidate of ci_lowest; address −1 is none.
struct CiPick {
  double value;
  int32_t address;
};

// Whether a is below b in (value, address) order, none above everything.
__device__ bool ci_below(const CiPick a, const CiPick b) {
  if (a.address < 0) {
    return false;
  }
  if (b.address < 0) {
    return true;
  }
  return a.value < b.value || (a.value == b.value && a.address < b.address);
}

// The least of a block's candidates, into thread 0's: a tree in shared memory.
__device__ CiPick ci_block_least(CiPick mine) {
  __shared__ double values[kCiBlock];
  __shared__ int32_t addresses[kCiBlock];
  values[threadIdx.x] = mine.value;
  addresses[threadIdx.x] = mine.address;
  __syncthreads();
  for (uint32_t half = kCiBlock / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      const CiPick other{values[threadIdx.x + half], addresses[threadIdx.x + half]};
      if (ci_below(other, CiPick{values[threadIdx.x], addresses[threadIdx.x]})) {
        values[threadIdx.x] = other.value;
        addresses[threadIdx.x] = other.address;
      }
    }
    __syncthreads();
  }
  return CiPick{values[0], addresses[0]};
}

// Pass one of pick j: each block's least candidate above pick j − 1 (none for
// j = 0, @p previous null).
__global__ void ci_lowest_partial_kernel(const int32_t n, const double *const values,
                                         const int32_t *const previous,
                                         double *const partial_values,
                                         int32_t *const partial_index) {
  const CiPick floor = previous == nullptr ? CiPick{0.0, -1} : CiPick{values[*previous], *previous};
  CiPick best{0.0, -1};
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const CiPick candidate{values[i], static_cast<int32_t>(i)};
    if ((floor.address < 0 || ci_below(floor, candidate)) && ci_below(candidate, best)) {
      best = candidate;
    }
  }
  const CiPick least = ci_block_least(best);
  if (threadIdx.x == 0) {
    partial_values[blockIdx.x] = least.value;
    partial_index[blockIdx.x] = least.address;
  }
}

// Pass two: one block, the least of the partials, into pick j.
__global__ void ci_lowest_final_kernel(const int32_t partials, const double *const partial_values,
                                       const int32_t *const partial_index, int32_t *const pick) {
  CiPick best{0.0, -1};
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < partials;
       i += static_cast<int32_t>(blockDim.x)) {
    const CiPick candidate{partial_values[i], partial_index[i]};
    if (ci_below(candidate, best)) {
      best = candidate;
    }
  }
  const CiPick least = ci_block_least(best);
  if (threadIdx.x == 0) {
    *pick = least.address;
  }
}

__global__ void ci_guess_kernel(const int32_t n, const int32_t count, const int32_t *const picks,
                                double *const guess) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(n) * count) {
    return;
  }
  // splitmix64 of the position, as ci_solve's spread always was.
  uint64_t z = (static_cast<uint64_t>(at) + 1) * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  z ^= z >> 31U;
  double x = 1e-3 * ((static_cast<double>(z >> 11U) * 0x1.0p-52) - 1.0); // [-1e-3, 1e-3)
  if (at % n == picks[at / n]) {
    x += 1.0;
  }
  guess[at] = x;
}

uint32_t ci_blocks(const int64_t items) {
  return static_cast<uint32_t>((items + kCiBlock - 1) / kCiBlock);
}

} // namespace

wwr::wwrError_t ci_spin_square(const wwr::wwrStream_t stream, const bool symmetric,
                               const int32_t rows, const int32_t n_det, const int32_t n_beta,
                               const int32_t n, const int32_t *const alpha_links,
                               const int32_t alpha_per, const int32_t *const beta_slots,
                               const double constant, const double shift, const double scale,
                               const bool accumulate, const double *const x, double *const y) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  if (symmetric) {
    ci_spin_square_kernel<true><<<ci_blocks(rows), kCiBlock, 0, stream>>>(
        rows, n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale,
        accumulate, x, y);
  } else {
    ci_spin_square_kernel<false><<<ci_blocks(rows), kCiBlock, 0, stream>>>(
        rows, n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale,
        accumulate, x, y);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_precondition(const wwr::wwrStream_t stream, const int32_t n, const double theta,
                                const double *const shifted, const double *const residual,
                                double *const correction) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  ci_precondition_kernel<<<ci_blocks(n), kCiBlock, 0, stream>>>(n, theta, shifted, residual,
                                                                correction);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_spin_diagonal(const wwr::wwrStream_t stream, const bool symmetric,
                                 const int32_t n_det, const int32_t n_beta, const int32_t n,
                                 const int32_t *const alpha_links, const int32_t alpha_per,
                                 const int32_t *const beta_slots, const double constant,
                                 const double shift, const double scale,
                                 const double *const diagonal, double *const shifted) {
  if (n_det == 0) {
    return wwr::wwrSuccess;
  }
  if (symmetric) {
    ci_spin_diagonal_kernel<true><<<ci_blocks(n_det), kCiBlock, 0, stream>>>(
        n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale, diagonal,
        shifted);
  } else {
    ci_spin_diagonal_kernel<false><<<ci_blocks(n_det), kCiBlock, 0, stream>>>(
        n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale, diagonal,
        shifted);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_lowest(const wwr::wwrStream_t stream, const int32_t n,
                          const double *const values, const int32_t count,
                          double *const partial_values, int32_t *const partial_index,
                          int32_t *const picks) {
  const auto blocks = static_cast<uint32_t>(
      ci_blocks(n) < static_cast<uint32_t>(kCiLowestBlocks) ? ci_blocks(n) : kCiLowestBlocks);
  for (int32_t j = 0; j < count; ++j) {
    ci_lowest_partial_kernel<<<blocks, kCiBlock, 0, stream>>>(
        n, values, j == 0 ? nullptr : picks + (j - 1), partial_values, partial_index);
    ci_lowest_final_kernel<<<1, kCiBlock, 0, stream>>>(static_cast<int32_t>(blocks), partial_values,
                                                       partial_index, picks + j);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t ci_guess(const wwr::wwrStream_t stream, const int32_t n, const int32_t count,
                         const int32_t *const picks, double *const guess) {
  const int64_t elements = static_cast<int64_t>(n) * count;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  ci_guess_kernel<<<ci_blocks(elements), kCiBlock, 0, stream>>>(n, count, picks, guess);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
