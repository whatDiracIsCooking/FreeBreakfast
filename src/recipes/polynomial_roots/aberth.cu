// The device half of free_breakfast.recipes.polynomial_roots: the
// Aberth-Ehrlich iteration, one warp per polynomial and one lane per root. One
// source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h). <cfloat>
// for FLT/DBL_EPSILON: nvcc rejects std::numeric_limits in device code.
#include <cfloat>

#include "primitives.h"

#include "block_params.h"
#include "constants.h"
#include "warp_reduce.cuh"

#include "aberth_bridge.h"

#include "complex.h"

namespace free_breakfast::device {

namespace {

namespace cg = cooperative_groups;

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

// Polynomials per block.
constexpr uint32_t kWarpsPerBlock = kFourWarps / kWarpSize;

// The iteration's arithmetic, on a plain (re, im) pair. The vendor complex
// types carry no portable operators (cuDoubleComplex is a bare double2,
// hipDoubleComplex a class), so they are read and written only at the edges,
// and one body serves both precisions and both backends. Division and modulus
// are scaled, so |z|^2 is never formed: no overflow for |z| near sqrt(max).
template <typename R>
struct Complex {
  R re;
  R im;
};

template <typename R>
__device__ __forceinline__ Complex<R> operator+(const Complex<R> a, const Complex<R> b) {
  return {a.re + b.re, a.im + b.im};
}

template <typename R>
__device__ __forceinline__ Complex<R> operator-(const Complex<R> a, const Complex<R> b) {
  return {a.re - b.re, a.im - b.im};
}

template <typename R>
__device__ __forceinline__ Complex<R> operator*(const Complex<R> a, const Complex<R> b) {
  return {(a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)};
}

template <typename R>
__device__ __forceinline__ Complex<R> operator*(const Complex<R> a, const R s) {
  return {a.re * s, a.im * s};
}

// cuCdiv's algorithm: both operands scaled by 1 / (|re b| + |im b|). b == 0
// gives NaN, as a general complex division would.
template <typename R>
__device__ __forceinline__ Complex<R> operator/(const Complex<R> a, const Complex<R> b) {
  const R s = R{1} / (math::fabs(b.re) + math::fabs(b.im));
  const R ar = a.re * s;
  const R ai = a.im * s;
  const R br = b.re * s;
  const R bi = b.im * s;
  const R t = R{1} / ((br * br) + (bi * bi));
  return {((ar * br) + (ai * bi)) * t, ((ai * br) - (ar * bi)) * t};
}

template <typename R>
__device__ __forceinline__ R modulus(const Complex<R> a) {
  return math::hypot(a.re, a.im);
}

// The edges: a coefficient (real or complex) in, a root out. Components go
// through complex.h's accessors, never .x/.y, which hipFloatComplex lacks.
__device__ __forceinline__ Complex<float> load(const float a) {
  return {a, 0.0F};
}
__device__ __forceinline__ Complex<double> load(const double a) {
  return {a, 0.0};
}
__device__ __forceinline__ Complex<float> load(const wwrFloatComplex a) {
  return {wwr::wwrCrealf(a), wwr::wwrCimagf(a)};
}
__device__ __forceinline__ Complex<double> load(const wwrDoubleComplex a) {
  return {wwr::wwrCreal(a), wwr::wwrCimag(a)};
}
__device__ __forceinline__ void store(wwrFloatComplex &out, const Complex<float> z) {
  out = wwr::make_wwrFloatComplex(z.re, z.im);
}
__device__ __forceinline__ void store(wwrDoubleComplex &out, const Complex<double> z) {
  out = wwr::make_wwrDoubleComplex(z.re, z.im);
}

// The larger of two, with NaN winning (`a != a`): a NaN correction must not
// read as converged.
struct MaxNanOp {
  template <typename R>
  __device__ R operator()(const R a, const R b) const {
    return (a != a || a > b) ? a : b;
  }
};

template <typename R>
constexpr R kEpsilon = sizeof(R) == sizeof(float) ? R{FLT_EPSILON} : R{DBL_EPSILON};

// Every root of a_0 + a_1 z + ... + a_n z^n, n <= kWarpSize, iterated across
// warp_tile: lane i owns root i in a register and returns it in @p root (lanes
// >= n own nothing). Every lane calls, with identical arguments. Returns the
// iteration at which every root settled, else -1 (a NaN never settles); the
// final estimates are in @p root either way.
//
// Each iteration, every lane reads every other estimate by shuffle. The
// shuffles are tile-wide collectives, so each lane has read them all before
// any lane moves its own: the update is Jacobi-style, independent of lane
// scheduling, and needs no shared memory and no barrier.
//
// A root settles when its relative correction |w| / |z| is <= tol, or when
// |p(z)| is inside Horner's rounding bound and no correction can improve it.
// |z| > 1 is evaluated through the reversed polynomial in 1/z, so no |z|^n is
// formed.
template <typename ParentT, typename T, typename R>
__device__ int32_t aberth_warp(cg::thread_block_tile<kWarpSize, ParentT> &warp_tile,
                               const T *const coeffs, const int32_t n, Complex<R> &root,
                               const R tol, const int32_t max_iter) {
  const auto lane = static_cast<int32_t>(warp_tile.thread_rank());
  const bool owner = lane < n;
  const Complex<R> one{R{1}, R{0}};

  // Initial estimates: n points on the circle of radius |a_0 / a_n|^(1/n) --
  // the geometric mean of the root moduli -- turned off the real axis by a
  // fixed angle so a real polynomial's conjugate pairs are not started
  // symmetric.
  Complex<R> z{R{0}, R{0}};
  if (owner) {
    R radius =
        math::pow(modulus(load(coeffs[0])) / modulus(load(coeffs[n])), R{1} / static_cast<R>(n));
    if (!(radius > R{0}) || !(radius - radius == R{0})) {
      radius = R{1}; // a_0 == 0, or the ratio overflowed: any distinct start works
    }
    const R theta = (kTwo<R> * kPi<R> * static_cast<R>(lane) / static_cast<R>(n)) + R{0.4};
    z = {radius * math::cos(theta), radius * math::sin(theta)};
  }

  // gamma ~ n unit roundoffs: |p(z)| below gamma * sum |a_k| |z|^k is rounding
  // noise, and further corrections cannot improve z.
  const R gamma = static_cast<R>(n) * kEpsilon<R>;
  int32_t iters = -1;
  for (int32_t iter = 1; iter <= max_iter; ++iter) {
    // The Aberth sum over the other estimates. Every lane shuffles, owner or
    // not; a non-owner's sum is never used.
    Complex<R> s{R{0}, R{0}};
    for (int32_t j = 0; j < n; ++j) {
      const auto src = static_cast<uint32_t>(j);
      const Complex<R> zj{warp_tile.shfl(z.re, src), warp_tile.shfl(z.im, src)};
      if (j != lane) {
        s = s + (one / (z - zj));
      }
    }

    R ratio = R{0};
    if (owner) {
      // Newton's ratio as num / den = p / p'. Inside the unit disc, p and p' by
      // Horner; outside it, the reversal q(y) = y^n p(1/y) at y = 1/z, where
      // p'/p = y (n q - y q') / q. bound is sum |a_k| |x|^k for the polynomial
      // actually evaluated.
      const R zmod = modulus(z);
      const bool inside = zmod <= R{1};
      const Complex<R> x = inside ? z : one / z;
      const R xmod = modulus(x);
      const int32_t k0 = inside ? n : 0; // the leading coefficient, then toward the other end
      const int32_t dk = inside ? -1 : 1;
      Complex<R> num = load(coeffs[k0]);
      Complex<R> dnum{R{0}, R{0}};
      R bound = modulus(num);
      for (int32_t k = k0 + dk; k >= 0 && k <= n; k += dk) {
        const Complex<R> a = load(coeffs[k]);
        dnum = (dnum * x) + num;
        num = (num * x) + a;
        bound = (bound * xmod) + modulus(a);
      }
      const Complex<R> den = inside ? dnum : x * ((num * static_cast<R>(n)) - (x * dnum));
      // w = N / (1 - N s) with N = num / den, rearranged so den == 0 is no
      // special case.
      const Complex<R> w = num / (den - (num * s));
      z = z - w;
      const R wmod = modulus(w);
      // wmod == 0 is settled even at z == 0, where the ratio would be 0/0. A
      // NaN num fails the <= and keeps its NaN ratio, so it reaches the fold.
      const bool noise = modulus(num) <= gamma * bound;
      ratio = (noise || wmod == R{0}) ? R{0} : wmod / zmod;
    }
    // Only the n owners' ratios enter the fold; every lane gets the result, so
    // the exit is warp-uniform.
    if (warp_reduce(warp_tile, ratio, MaxNanOp{}, static_cast<uint32_t>(n)) <= tol) {
      iters = iter;
      break;
    }
  }
  root = z;
  return iters;
}

// One warp per polynomial: warp w of block blk takes polynomial
// blk * kWarpsPerBlock + w.
template <typename T, typename CT, typename R>
__global__ void __launch_bounds__(kFourWarps)
    aberth_kernel(const int32_t n, const int32_t batch, const T *const coeffs, CT *const roots,
                  int32_t *const iters, const R tol, const int32_t max_iter) {
  auto warp_tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int64_t b =
      (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) + warp_tile.meta_group_rank();
  // Warp-uniform exit: the collectives are tile-wide, never block-wide.
  if (b >= batch) {
    return;
  }
  Complex<R> z;
  const int32_t it = aberth_warp(warp_tile, coeffs + (b * (n + 1)), n, z, tol, max_iter);
  const auto lane = static_cast<int32_t>(warp_tile.thread_rank());
  if (lane < n) {
    store(roots[(b * n) + lane], z);
  }
  if (lane == 0) {
    iters[b] = it;
  }
}

} // namespace

template <typename T, typename CT, typename R>
wwr::wwrError_t aberth(const wwr::wwrStream_t stream, const int32_t n, const int32_t batch,
                       const T *const coeffs, CT *const roots, int32_t *const iters, const R tol,
                       const int32_t max_iter) {
  const auto grid =
      static_cast<uint32_t>(math::idivup(batch, static_cast<int32_t>(kWarpsPerBlock)));
  aberth_kernel<T, CT, R>
      <<<grid, kFourWarps, 0, stream>>>(n, batch, coeffs, roots, iters, tol, max_iter);
  return wwr::wwrGetLastError();
}

// One per supported pair, matching polynomial_roots.cpp's list.
template wwr::wwrError_t aberth<float, wwrFloatComplex, float>(wwr::wwrStream_t, int32_t, int32_t,
                                                               const float *, wwrFloatComplex *,
                                                               int32_t *, float, int32_t);
template wwr::wwrError_t aberth<double, wwrDoubleComplex, double>(wwr::wwrStream_t, int32_t,
                                                                  int32_t, const double *,
                                                                  wwrDoubleComplex *, int32_t *,
                                                                  double, int32_t);
template wwr::wwrError_t aberth<wwrFloatComplex, wwrFloatComplex, float>(wwr::wwrStream_t, int32_t,
                                                                         int32_t,
                                                                         const wwrFloatComplex *,
                                                                         wwrFloatComplex *,
                                                                         int32_t *, float, int32_t);
template wwr::wwrError_t
aberth<wwrDoubleComplex, wwrDoubleComplex, double>(wwr::wwrStream_t, int32_t, int32_t,
                                                   const wwrDoubleComplex *, wwrDoubleComplex *,
                                                   int32_t *, double, int32_t);

} // namespace free_breakfast::device
