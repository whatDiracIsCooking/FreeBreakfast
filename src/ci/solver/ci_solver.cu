// The device half of ci::solve: S² on a block of CI vectors, which the spin
// penalty adds into sigma's output, the diagonal preconditioner with Olsen's
// correction, the select of the lowest diagonals, and the P-space's
// Hamiltonian by Slater–Condon. One source for both backends -- nvcc under
// CUDA, clang -x hip under HIP.
//
// S² is a gather, one thread per row: (Σ_pq E^β_pq E^α_qp x)_J walks J's α
// links a†_p a_q and, for each, the β partner a†_q a_p of J's β string, read
// from that string's slot table. S² is real symmetric, so J's own links reach
// every K with <J|S²|K> nonzero. A thread owns its row, so nothing is atomic
// and the summation order is fixed.
//
// S² and the penalized diagonal come in the two layouts of ci::Layout, as a
// template on Symmetric: in the symmetric one a row is a triangle element
// (a, b), a >= b, and a link below the triangle reads its mirror, as the GPU
// sigma's gathers do (see ../sigma/ci_symmetric.h).

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "ci_solver_bridge.h"
#include "ci_symmetric.h"

namespace free_breakfast::ci::device {

namespace {

// The β slot of an α link a†_p a_q's partner a†_q a_p: q · n + p.
__device__ int32_t ci_partner_slot(const Replacement l, const int32_t n) {
  return (static_cast<int32_t>(l.q) * n) + static_cast<int32_t>(l.p);
}

// The preconditioner's floor on |theta − diag|: a root concentrated on one
// determinant has theta ≈ its diagonal.
constexpr double kDenominatorFloor = 1e-3;

// Determinant (a, b) of element @p det of a vector, in the layout's terms.
template <bool Symmetric>
__device__ void ci_strings(const int32_t det, const int32_t n_beta, int32_t &a, int32_t &b) {
  if constexpr (Symmetric) {
    triangle_strings(det, a, b);
  } else {
    a = det / n_beta;
    b = det % n_beta;
  }
}

template <bool Symmetric>
__global__ void ci_spin_square_kernel(const int32_t rows, const int32_t n_det, const int32_t n_beta,
                                      const int32_t n, const Replacement *const alpha_links,
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
  const Replacement *const a = alpha_links + (static_cast<size_t>(ja) * alpha_per);
  const int32_t *const slots =
      beta_slots + (static_cast<size_t>(jb) * static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    const int32_t slot = slots[ci_partner_slot(l, n)];
    if (slot == 0) {
      continue;
    }
    const int32_t kb = (slot < 0 ? -slot : slot) - 1;
    const double sign = static_cast<double>(slot < 0 ? -l.sign : l.sign);
    if constexpr (Symmetric) {
      const double weight = l.target == kb ? 1.0 : kInvSqrt2;
      links += sign * (weight * x[vector_base + triangle(l.target, kb)]);
    } else {
      links +=
          sign * x[vector_base + (static_cast<size_t>(l.target) * static_cast<size_t>(n_beta)) +
                   static_cast<size_t>(kb)];
    }
  }
  double sum = (constant - shift) * x[row];
  if constexpr (Symmetric) {
    sum -= ja == jb ? links : kSqrt2 * links;
  } else {
    sum -= links;
  }
  y[row] = (accumulate ? y[row] : 0.0) + (scale * sum);
}

// theta − shifted, held off zero at ±kDenominatorFloor.
__device__ double ci_denominator(const double theta, const double shifted) {
  const double denominator = theta - shifted;
  if (denominator < kDenominatorFloor && denominator > -kDenominatorFloor) {
    return denominator < 0.0 ? -kDenominatorFloor : kDenominatorFloor;
  }
  return denominator;
}

__global__ void ci_precondition_kernel(const int32_t n, const double theta,
                                       const double *const shifted, const double *const in,
                                       double *const out) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = in[i] / ci_denominator(theta, shifted[i]);
}

// Olsen's correction, M⁻¹ (r − ε x) with ε = dots[1] / dots[0] read on the
// device. A zero x^T M⁻¹ x leaves ε at 0: the plain correction M⁻¹ r. With
// shifted null, r − ε x alone.
__global__ void ci_olsen_kernel(const int32_t n, const double theta, const double *const shifted,
                                const double *const x, const double *const residual,
                                const double *const dots, double *const correction) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double epsilon = dots[0] == 0.0 ? 0.0 : dots[1] / dots[0];
  const double numerator = residual[i] - (epsilon * x[i]);
  correction[i] = shifted == nullptr ? numerator : numerator / ci_denominator(theta, shifted[i]);
}

// With @p exchange, diagonal_J is first spin-averaged: over the open shells
// (occupied in one string of J), the exchange −δ_{σ_i σ_j} K_ij of each pair is
// replaced by −p K_ij, p the probability two open shells have the same spin
// over the configuration's determinants, which is the same for every pair.
template <bool Symmetric>
__global__ void ci_spin_diagonal_kernel(const int32_t n_det, const int32_t n_beta, const int32_t n,
                                        const Replacement *const alpha_links,
                                        const int32_t alpha_per, const int32_t *const beta_slots,
                                        const double constant, const double shift,
                                        const double scale, const double *const exchange,
                                        const double *const diagonal, double *const shifted) {
  const int64_t det = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (det >= n_det) {
    return;
  }
  int32_t ja = 0;
  int32_t jb = 0;
  ci_strings<Symmetric>(static_cast<int32_t>(det), n_beta, ja, jb);
  // Only a†_p a_p leaves the α string where it was, and its partner a†_p a_p
  // survives the β string exactly when p is occupied there too. Its slot
  // index is p · n + p.
  int32_t doubles = 0;
  uint64_t alpha_occupied = 0;
  const Replacement *const a = alpha_links + (static_cast<size_t>(ja) * alpha_per);
  const int32_t *const slots =
      beta_slots + (static_cast<size_t>(jb) * static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    if (l.target == ja) {
      alpha_occupied |= uint64_t{1} << l.p;
      if (slots[ci_partner_slot(l, n)] != 0) {
        ++doubles;
      }
    }
  }
  // Σ_{i<j open} (δ_{σ_i σ_j} − p) K_ij, which takes J's own open-shell
  // exchange out and puts the average in.
  double average = 0.0;
  if (exchange != nullptr) {
    uint64_t beta_occupied = 0;
    for (int32_t p = 0; p < n; ++p) {
      if (slots[p * (n + 1)] != 0) {
        beta_occupied |= uint64_t{1} << p;
      }
    }
    int32_t open_alpha = 0;
    int32_t open_beta = 0;
    for (int32_t p = 0; p < n; ++p) {
      open_alpha += static_cast<int32_t>(((alpha_occupied & ~beta_occupied) >> p) & 1U);
      open_beta += static_cast<int32_t>(((beta_occupied & ~alpha_occupied) >> p) & 1U);
    }
    const int32_t open = open_alpha + open_beta;
    if (open >= 2) {
      const double same =
          static_cast<double>((open_alpha * (open_alpha - 1)) + (open_beta * (open_beta - 1))) /
          static_cast<double>(open * (open - 1));
      const uint64_t open_shells = alpha_occupied ^ beta_occupied;
      for (int32_t i = 0; i < n; ++i) {
        if (((open_shells >> i) & 1U) == 0) {
          continue;
        }
        const bool i_alpha = ((alpha_occupied >> i) & 1U) != 0;
        for (int32_t j = i + 1; j < n; ++j) {
          if (((open_shells >> j) & 1U) == 0) {
            continue;
          }
          const double delta = (((alpha_occupied >> j) & 1U) != 0) == i_alpha ? 1.0 : 0.0;
          average += (delta - same) * exchange[(static_cast<size_t>(i) * n) + j];
        }
      }
    }
  }
  const size_t full =
      (static_cast<size_t>(ja) * static_cast<size_t>(n_beta)) + static_cast<size_t>(jb);
  shifted[det] =
      diagonal[full] + average + (scale * (constant - static_cast<double>(doubles) - shift));
}

// pick_lowest's state, in its scratch: the histogram of the current byte, the
// (value key, address) prefix chosen so far, the rank still to find under it,
// and the survivors' counter.
struct CiLowestState {
  uint64_t key;
  uint32_t address;
  uint32_t remaining;
  uint32_t found;
  uint32_t histogram[256];
};
static_assert(sizeof(CiLowestState) <= sizeof(uint64_t) * kCiLowestScratch);

// The blocks of pick_lowest's histogram passes, at most.
constexpr uint32_t kCiLowestBlocks = 1024;

// The select's passes: eight bytes of the value's key, then four of the
// address.
constexpr int32_t kCiKeyBytes = 8;
constexpr int32_t kCiLowestPasses = 12;

// A double's bits as an unsigned key in the double's order, −0 as +0.
__device__ uint64_t ci_order_key(const double value) {
  const auto bits = static_cast<uint64_t>(__double_as_longlong(value == 0.0 ? 0.0 : value));
  return (bits >> 63U) != 0 ? ~bits : bits | (uint64_t{1} << 63U);
}

// Whether (key, address) matches the prefix above byte @p pass, and if so its
// byte there.
__device__ bool ci_lowest_digit(const CiLowestState &state, const int32_t pass, const uint64_t key,
                                const uint32_t address, uint32_t &digit) {
  if (pass < kCiKeyBytes) {
    const auto above = static_cast<uint32_t>(64 - (8 * pass));
    if (pass > 0 && (key >> above) != (state.key >> above)) {
      return false;
    }
    digit = static_cast<uint32_t>(key >> (above - 8U)) & 0xFFU;
    return true;
  }
  const auto above = static_cast<uint32_t>(32 - (8 * (pass - kCiKeyBytes)));
  if (key != state.key || (pass > kCiKeyBytes && (address >> above) != (state.address >> above))) {
    return false;
  }
  digit = (address >> (above - 8U)) & 0xFFU;
  return true;
}

// One pass's histogram: each block counts its stride in shared memory, then
// adds its bins into the state's.
__global__ void ci_lowest_histogram_kernel(const int32_t n, const double *const values,
                                           const int32_t pass, CiLowestState *const state) {
  __shared__ uint32_t bins[256];
  for (uint32_t b = threadIdx.x; b < 256; b += blockDim.x) {
    bins[b] = 0;
  }
  __syncthreads();
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t digit = 0;
    if (ci_lowest_digit(*state, pass, ci_order_key(values[i]), static_cast<uint32_t>(i), digit)) {
      atomicAdd(&bins[digit], 1U);
    }
  }
  __syncthreads();
  for (uint32_t b = threadIdx.x; b < 256; b += blockDim.x) {
    if (bins[b] != 0) {
      atomicAdd(&state->histogram[b], bins[b]);
    }
  }
}

// One block: the byte under which the remaining rank lies, into the prefix;
// then the histogram cleared for the next pass.
__global__ void ci_lowest_choose_kernel(const int32_t pass, const int32_t count,
                                        CiLowestState *const state) {
  if (threadIdx.x == 0) {
    uint32_t remaining = pass == 0 ? static_cast<uint32_t>(count) : state->remaining;
    uint32_t digit = 0;
    for (; digit < 255 && remaining > state->histogram[digit]; ++digit) {
      remaining -= state->histogram[digit];
    }
    if (pass < kCiKeyBytes) {
      state->key |= uint64_t{digit} << static_cast<uint32_t>(56 - (8 * pass));
    } else {
      state->address |= digit << static_cast<uint32_t>(24 - (8 * (pass - kCiKeyBytes)));
    }
    state->remaining = remaining;
  }
  __syncthreads();
  for (uint32_t b = threadIdx.x; b < 256; b += blockDim.x) {
    state->histogram[b] = 0;
  }
}

// Every element at or below the threshold (key, address) into survivors.
__global__ void ci_lowest_compact_kernel(const int32_t n, const double *const values,
                                         CiLowestState *const state, int32_t *const survivors) {
  const uint64_t key = state->key;
  const uint32_t address = state->address;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint64_t mine = ci_order_key(values[i]);
    if (mine < key || (mine == key && static_cast<uint32_t>(i) <= address)) {
      survivors[atomicAdd(&state->found, 1U)] = static_cast<int32_t>(i);
    }
  }
}

// Each survivor to its rank among them in (key, address) order.
__global__ void ci_lowest_rank_kernel(const int32_t count, const double *const values,
                                      const int32_t *const survivors, int32_t *const picks) {
  const int64_t s = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (s >= count) {
    return;
  }
  const int32_t address = survivors[s];
  const uint64_t key = ci_order_key(values[address]);
  int32_t rank = 0;
  for (int32_t t = 0; t < count; ++t) {
    const int32_t other = survivors[t];
    const uint64_t other_key = ci_order_key(values[other]);
    rank += static_cast<int32_t>(other_key < key || (other_key == key && other < address));
  }
  picks[rank] = address;
}

__global__ void ci_guess_kernel(const int32_t n, const int32_t count, const int32_t *const picks,
                                double *const guess) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(n) * count) {
    return;
  }
  // splitmix64 of the position, as ci::solve's spread always was.
  uint64_t z = (static_cast<uint64_t>(at) + 1) * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  z ^= z >> 31U;
  double x = 1e-3 * ((static_cast<double>(z >> 11U) * 0x1.0p-52) - 1.0); // [-1e-3, 1e-3)
  if (picks != nullptr && at % n == picks[at / n]) {
    x += 1.0;
  }
  guess[at] = x;
}

__global__ void ci_gather_kernel(const int32_t n, const int32_t count, const int32_t p,
                                 const int32_t *const picks, const double *const from,
                                 double *const to) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(p) * count) {
    return;
  }
  const int64_t column = at / p;
  to[at] = from[(column * n) + picks[at % p]];
}

__global__ void ci_scatter_kernel(const int32_t n, const int32_t count, const int32_t p,
                                  const int32_t *const picks, const double *const from,
                                  double *const to) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(p) * count) {
    return;
  }
  const int64_t column = at / p;
  to[(column * n) + picks[at % p]] = from[at];
}

// The occupied orbitals a_k or a†_k passes on the bit pattern @p string,
// those below k, before it flips bit k there.
__device__ int32_t ci_flip(uint64_t &string, const int32_t k) {
  const uint64_t bit = uint64_t{1} << static_cast<uint32_t>(k);
  const int32_t passed = __popcll(string & (bit - 1U));
  string ^= bit;
  return passed;
}

// ±1 for an even or odd count of orbitals passed.
__device__ double ci_parity(const int32_t passed) {
  return (passed & 1) == 0 ? 1.0 : -1.0;
}

// The sign of a†_p a_q on @p string, q occupied and p not: a_q, then a†_p.
__device__ double ci_replacement_sign(uint64_t string, const int32_t p, const int32_t q) {
  const int32_t passed = ci_flip(string, q);
  return ci_parity(passed + ci_flip(string, p));
}

// The lowest and highest set bits of a nonzero pattern, through popcounts
// alone: the bits below the lowest one, and those at or below the highest
// once it is smeared down.
__device__ int32_t ci_lowest_bit(const uint64_t bits) {
  return __popcll((bits & (~bits + 1U)) - 1U);
}
__device__ int32_t ci_highest_bit(uint64_t bits) {
  for (uint32_t shift = 1; shift < 64; shift *= 2) {
    bits |= bits >> shift;
  }
  return __popcll(bits) - 1;
}

// (tu|vw) from ci::Sigma's contraction: twice its ½ (tu|vw).
__device__ double ci_eri(const double *const contraction, const int32_t pairs, const int32_t t,
                         const int32_t u, const int32_t v, const int32_t w) {
  return 2.0 * contraction[(static_cast<size_t>(triangle(t, u)) * pairs) + triangle(v, w)];
}

// h_pq from ci::Sigma's contraction: k_pq + ½ Σ_u (pu|uq).
__device__ double ci_one_electron(const double *const contraction, const int32_t pairs,
                                  const int32_t n, const int32_t p, const int32_t q) {
  double h = contraction[(static_cast<size_t>(pairs) * pairs) + triangle(p, q)];
  for (int32_t u = 0; u < n; ++u) {
    h += contraction[(static_cast<size_t>(triangle(p, u)) * pairs) + triangle(u, q)];
  }
  return h;
}

// <I| H + λ (S² − target) |J> for the determinants I = (ia, ib) and
// J = (ja, jb), by Slater–Condon's rules on the strings, each spin's
// replacement carrying its own string's sign (a β operator passes every α
// creator an even number of times). The orbitals occupied in J of the
// replaced spin and the other one sum separately.
__device__ double ci_determinant_element(const int32_t ia, const int32_t ib, const int32_t ja,
                                         const int32_t jb, const int32_t n_beta, const int32_t n,
                                         const uint64_t *const alpha_strings,
                                         const uint64_t *const beta_strings,
                                         const double *const contraction,
                                         const double *const diagonal, const double penalty,
                                         const double spin_shift) {
  const uint64_t bra_a = alpha_strings[ia];
  const uint64_t bra_b = beta_strings[ib];
  const uint64_t ket_a = alpha_strings[ja];
  const uint64_t ket_b = beta_strings[jb];
  if (ia == ja && ib == jb) {
    const size_t full =
        (static_cast<size_t>(ia) * static_cast<size_t>(n_beta)) + static_cast<size_t>(ib);
    return diagonal[full] + (penalty * (spin_shift - __popcll(ket_a & ket_b)));
  }
  const uint64_t particles_a = bra_a & ~ket_a;
  const uint64_t holes_a = ket_a & ~bra_a;
  const uint64_t particles_b = bra_b & ~ket_b;
  const uint64_t holes_b = ket_b & ~bra_b;
  const int32_t excited_a = __popcll(particles_a);
  const int32_t excited_b = __popcll(particles_b);
  if (excited_a + excited_b > 2) {
    return 0.0;
  }
  const int32_t pairs = n * (n + 1) / 2;
  if (excited_a + excited_b == 1) {
    // a†_p a_q on one spin's string: h_pq + Σ_j∈J [(pq|jj) − δ_σ (pj|jq)].
    const bool alpha = excited_a == 1;
    const uint64_t same = alpha ? ket_a : ket_b;
    const uint64_t other = alpha ? ket_b : ket_a;
    const int32_t p = ci_lowest_bit(alpha ? particles_a : particles_b);
    const int32_t q = ci_lowest_bit(alpha ? holes_a : holes_b);
    double sum = ci_one_electron(contraction, pairs, n, p, q);
    for (int32_t j = 0; j < n; ++j) {
      if (((same >> j) & 1U) != 0 && j != q) {
        sum += ci_eri(contraction, pairs, p, q, j, j) - ci_eri(contraction, pairs, p, j, j, q);
      }
      if (((other >> j) & 1U) != 0) {
        sum += ci_eri(contraction, pairs, p, q, j, j);
      }
    }
    return ci_replacement_sign(same, p, q) * sum;
  }
  if (excited_a == 1) {
    // a†_pα a_qα a†_rβ a_sβ: (pq|rs), and S²'s −λ where it is the swap r = q,
    // s = p.
    const int32_t p = ci_lowest_bit(particles_a);
    const int32_t q = ci_lowest_bit(holes_a);
    const int32_t r = ci_lowest_bit(particles_b);
    const int32_t s = ci_lowest_bit(holes_b);
    const double sign = ci_replacement_sign(ket_a, p, q) * ci_replacement_sign(ket_b, r, s);
    const double spin = r == q && s == p ? penalty : 0.0;
    return sign * (ci_eri(contraction, pairs, p, q, r, s) - spin);
  }
  // Two of one spin, particles a < b and holes i < j: I = a†_a a†_b a_j a_i J,
  // its sign that of a_i, a_j, a†_b and a†_a in turn; <ab||ij>.
  const bool alpha = excited_a == 2;
  const uint64_t particles = alpha ? particles_a : particles_b;
  const uint64_t holes = alpha ? holes_a : holes_b;
  const int32_t a = ci_lowest_bit(particles);
  const int32_t b = ci_highest_bit(particles);
  const int32_t i = ci_lowest_bit(holes);
  const int32_t j = ci_highest_bit(holes);
  uint64_t string = alpha ? ket_a : ket_b;
  int32_t passed = ci_flip(string, i);
  passed += ci_flip(string, j);
  passed += ci_flip(string, b);
  passed += ci_flip(string, a);
  return ci_parity(passed) *
         (ci_eri(contraction, pairs, a, i, b, j) - ci_eri(contraction, pairs, a, j, b, i));
}

template <bool Symmetric>
__global__ void ci_pspace_kernel(const int32_t p, const int32_t *const picks, const int32_t n_beta,
                                 const int32_t n, const uint64_t *const alpha_strings,
                                 const uint64_t *const beta_strings,
                                 const double *const contraction, const double *const diagonal,
                                 const double penalty, const double spin_shift, double *const h) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (at >= static_cast<int64_t>(p) * p) {
    return;
  }
  const auto row = static_cast<int32_t>(at % p);
  const auto column = static_cast<int32_t>(at / p);
  int32_t ia = 0;
  int32_t ib = 0;
  int32_t ja = 0;
  int32_t jb = 0;
  ci_strings<Symmetric>(picks[row], n_beta, ia, ib);
  ci_strings<Symmetric>(picks[column], n_beta, ja, jb);
  const auto element = [&](const int32_t a, const int32_t b, const int32_t c, const int32_t d) {
    return ci_determinant_element(a, b, c, d, n_beta, n, alpha_strings, beta_strings, contraction,
                                  diagonal, penalty, spin_shift);
  };
  if constexpr (Symmetric) {
    // φ = (|ab> + |ba>) / √2 off the triangle's diagonal, |aa> on it.
    double sum = element(ia, ib, ja, jb);
    if (ja != jb) {
      sum += element(ia, ib, jb, ja);
    }
    if (ia != ib) {
      sum += element(ib, ia, ja, jb);
      if (ja != jb) {
        sum += element(ib, ia, jb, ja);
      }
    }
    const double weight = (ia == ib ? 1.0 : kInvSqrt2) * (ja == jb ? 1.0 : kInvSqrt2);
    h[at] = weight * sum;
  } else {
    h[at] = element(ia, ib, ja, jb);
  }
}

uint32_t ci_blocks(const int64_t items) {
  return static_cast<uint32_t>(math::idivup(items, int64_t{kFourWarps}));
}

} // namespace

wwr::wwrError_t spin_square(const wwr::wwrStream_t stream, const bool symmetric, const int32_t rows,
                            const int32_t n_det, const int32_t n_beta, const int32_t n,
                            const Replacement *const alpha_links, const int32_t alpha_per,
                            const int32_t *const beta_slots, const double constant,
                            const double shift, const double scale, const bool accumulate,
                            const double *const x, double *const y) {
  if (rows == 0) {
    return wwr::wwrSuccess;
  }
  if (symmetric) {
    ci_spin_square_kernel<true><<<ci_blocks(rows), kFourWarps, 0, stream>>>(
        rows, n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale,
        accumulate, x, y);
  } else {
    ci_spin_square_kernel<false><<<ci_blocks(rows), kFourWarps, 0, stream>>>(
        rows, n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale,
        accumulate, x, y);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t precondition(const wwr::wwrStream_t stream, const int32_t n, const double theta,
                             const double *const shifted, const double *const in,
                             double *const out) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  ci_precondition_kernel<<<ci_blocks(n), kFourWarps, 0, stream>>>(n, theta, shifted, in, out);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t olsen(const wwr::wwrStream_t stream, const int32_t n, const double theta,
                      const double *const shifted, const double *const x,
                      const double *const residual, const double *const dots,
                      double *const correction) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  ci_olsen_kernel<<<ci_blocks(n), kFourWarps, 0, stream>>>(n, theta, shifted, x, residual, dots,
                                                           correction);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t spin_diagonal(const wwr::wwrStream_t stream, const bool symmetric,
                              const int32_t n_det, const int32_t n_beta, const int32_t n,
                              const Replacement *const alpha_links, const int32_t alpha_per,
                              const int32_t *const beta_slots, const double constant,
                              const double shift, const double scale, const double *const exchange,
                              const double *const diagonal, double *const shifted) {
  if (n_det == 0) {
    return wwr::wwrSuccess;
  }
  if (symmetric) {
    ci_spin_diagonal_kernel<true><<<ci_blocks(n_det), kFourWarps, 0, stream>>>(
        n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale, exchange,
        diagonal, shifted);
  } else {
    ci_spin_diagonal_kernel<false><<<ci_blocks(n_det), kFourWarps, 0, stream>>>(
        n_det, n_beta, n, alpha_links, alpha_per, beta_slots, constant, shift, scale, exchange,
        diagonal, shifted);
  }
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pick_lowest(const wwr::wwrStream_t stream, const int32_t n,
                            const double *const values, const int32_t count,
                            uint64_t *const scratch, int32_t *const survivors,
                            int32_t *const picks) {
  auto *const state = reinterpret_cast<CiLowestState *>(scratch);
  if (const wwr::wwrError_t status = wwr::wwrMemsetAsync(state, 0, sizeof(CiLowestState), stream);
      status != wwr::wwrSuccess) {
    return status;
  }
  const uint32_t blocks = ci_blocks(n) < kCiLowestBlocks ? ci_blocks(n) : kCiLowestBlocks;
  for (int32_t pass = 0; pass < kCiLowestPasses; ++pass) {
    ci_lowest_histogram_kernel<<<blocks, kFourWarps, 0, stream>>>(n, values, pass, state);
    ci_lowest_choose_kernel<<<1, kFourWarps, 0, stream>>>(pass, count, state);
  }
  ci_lowest_compact_kernel<<<blocks, kFourWarps, 0, stream>>>(n, values, state, survivors);
  ci_lowest_rank_kernel<<<ci_blocks(count), kFourWarps, 0, stream>>>(count, values, survivors,
                                                                     picks);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t guess(const wwr::wwrStream_t stream, const int32_t n, const int32_t count,
                      const int32_t *const picks, double *const guess) {
  const int64_t elements = static_cast<int64_t>(n) * count;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  ci_guess_kernel<<<ci_blocks(elements), kFourWarps, 0, stream>>>(n, count, picks, guess);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t gather(const wwr::wwrStream_t stream, const int32_t n, const int32_t count,
                       const int32_t p, const int32_t *const picks, const double *const from,
                       double *const to) {
  const int64_t elements = static_cast<int64_t>(p) * count;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  ci_gather_kernel<<<ci_blocks(elements), kFourWarps, 0, stream>>>(n, count, p, picks, from, to);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t scatter(const wwr::wwrStream_t stream, const int32_t n, const int32_t count,
                        const int32_t p, const int32_t *const picks, const double *const from,
                        double *const to) {
  const int64_t elements = static_cast<int64_t>(p) * count;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  ci_scatter_kernel<<<ci_blocks(elements), kFourWarps, 0, stream>>>(n, count, p, picks, from, to);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t pspace(const wwr::wwrStream_t stream, const bool symmetric, const int32_t p,
                       const int32_t *const picks, const int32_t n_beta, const int32_t n,
                       const uint64_t *const alpha_strings, const uint64_t *const beta_strings,
                       const double *const contraction, const double *const diagonal,
                       const double penalty, const double spin_shift, double *const h) {
  const int64_t elements = static_cast<int64_t>(p) * p;
  if (elements == 0) {
    return wwr::wwrSuccess;
  }
  if (symmetric) {
    ci_pspace_kernel<true><<<ci_blocks(elements), kFourWarps, 0, stream>>>(
        p, picks, n_beta, n, alpha_strings, beta_strings, contraction, diagonal, penalty,
        spin_shift, h);
  } else {
    ci_pspace_kernel<false><<<ci_blocks(elements), kFourWarps, 0, stream>>>(
        p, picks, n_beta, n, alpha_strings, beta_strings, contraction, diagonal, penalty,
        spin_shift, h);
  }
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::ci::device
