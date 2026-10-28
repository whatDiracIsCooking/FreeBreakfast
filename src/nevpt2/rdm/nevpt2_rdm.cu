// The device half of nevpt2::rdms: the eri's conversion at entry, the per-tile
// R and L2 over each determinant's own replacement lists, and fdm2 and the
// wedge once after the tile loop. The products between them (the digests and
// the consume) are BLAS calls on the module's side. One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP.
//
// Ported from nevpt2-gpu-demo's src/rdm/kernels.cu (commit 7c5a674): the same
// sums in the same order, over this tree's Replacement lists in place of the
// demo's Knowles-Handy tables. A forward link a†_p a_q |K> = s |J> of K is
// <J| E_pq |K> = s, which is also the demo's reverse link of K (E_qp |J> =
// s |K>), so R reads K's own lists and no reverse table is built.
//
// One thread per determinant for R and L2 (adjacent threads, adjacent columns,
// so the writes are coalesced), and one thread per output element for the
// rest. A thread owns what it writes and nothing is atomic, so the result does
// not depend on the run.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "nevpt2_rdm_bridge.h"

namespace free_breakfast::nevpt2::device {

// The strings' link, borrowed from ci (docs/style.md: a using-declaration
// naming what the file borrows).
using ci::Replacement;

namespace {

__global__ void nevpt2_rdm_convert_eri_kernel(const int64_t n, const double *const in,
                                              double *const eri, double *const consume) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn) {
    return;
  }
  // at = ((i n + j) n + k) n + l, read once as eri's (a, b, c, d) and once as
  // consume's (x, a, p, q).
  const int64_t i = at / (nn * n);
  const int64_t j = (at / nn) % n;
  const int64_t k = (at / n) % n;
  const int64_t l = at % n;
  // eri[i, j, k, l] = (ij|kl).
  eri[at] = in[i + (n * j) + (nn * k) + (nn * n * l)];
  // consume[x = i, a = j, p = k, q = l] = (ax|qp) = (j i|l k).
  consume[at] = in[j + (n * i) + (nn * l) + (nn * n * k)];
}

__global__ void nevpt2_rdm_produce_kernel(const int32_t n, const int32_t n_beta,
                                          const Replacement *const alpha_links,
                                          const int32_t alpha_per,
                                          const Replacement *const beta_links,
                                          const int32_t beta_per, const double *const c,
                                          const int32_t begin, const int32_t width, double *const r,
                                          double *const l2) {
  const int64_t k = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (k >= width) {
    return;
  }
  const int32_t det = begin + static_cast<int32_t>(k);
  const int32_t ka = det / n_beta;
  const int32_t kb = det % n_beta;
  const auto at = [&](const int32_t a, const int32_t b) {
    return c[(static_cast<int64_t>(a) * n_beta) + b];
  };
  const auto pair = [&](const Replacement l) {
    return (static_cast<int64_t>(l.p) * n) + static_cast<int64_t>(l.q);
  };
  const auto links_of = [](const Replacement *const links, const int32_t per, const int32_t s) {
    return links + (static_cast<int64_t>(s) * per);
  };
  const int64_t nn = static_cast<int64_t>(n) * n;
  const int64_t ld = width;
  const Replacement *const a = links_of(alpha_links, alpha_per, ka);
  const Replacement *const b = links_of(beta_links, beta_per, kb);

  // R[tu, K] = <K| E_tu |c>: each link (p, q, J, s) of K is <K| E_qp |J> = s.
  double *const r_col = r + k;
  for (int64_t tu = 0; tu < nn; ++tu) {
    r_col[tu * ld] = 0.0;
  }
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    r_col[((static_cast<int64_t>(l.q) * n) + l.p) * ld] +=
        static_cast<double>(l.sign) * at(l.target, kb);
  }
  for (int32_t e = 0; e < beta_per; ++e) {
    const Replacement l = b[e];
    r_col[((static_cast<int64_t>(l.q) * n) + l.p) * ld] +=
        static_cast<double>(l.sign) * at(ka, l.target);
  }

  // L2[pqrs, K] = <c| E_pq E_rs |K>: E_rs over K's links to M, then E_pq over
  // M's to J, each pair of links adding s s' c_J.
  double *const l2_col = l2 + k;
  for (int64_t pqrs = 0; pqrs < nn * nn; ++pqrs) {
    l2_col[pqrs * ld] = 0.0;
  }
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    const int64_t rs = pair(l);
    const auto s = static_cast<double>(l.sign);
    const Replacement *const m = links_of(alpha_links, alpha_per, l.target);
    for (int32_t f = 0; f < alpha_per; ++f) {
      const Replacement o = m[f];
      l2_col[((pair(o) * nn) + rs) * ld] += s * static_cast<double>(o.sign) * at(o.target, kb);
    }
    for (int32_t f = 0; f < beta_per; ++f) {
      const Replacement o = b[f];
      l2_col[((pair(o) * nn) + rs) * ld] +=
          s * static_cast<double>(o.sign) * at(l.target, o.target);
    }
  }
  for (int32_t e = 0; e < beta_per; ++e) {
    const Replacement l = b[e];
    const int64_t rs = pair(l);
    const auto s = static_cast<double>(l.sign);
    for (int32_t f = 0; f < alpha_per; ++f) {
      const Replacement o = a[f];
      l2_col[((pair(o) * nn) + rs) * ld] +=
          s * static_cast<double>(o.sign) * at(o.target, l.target);
    }
    const Replacement *const m = links_of(beta_links, beta_per, l.target);
    for (int32_t f = 0; f < beta_per; ++f) {
      const Replacement o = m[f];
      l2_col[((pair(o) * nn) + rs) * ld] += s * static_cast<double>(o.sign) * at(ka, o.target);
    }
  }
}

__global__ void nevpt2_rdm_fdm2_ca_kernel(const int64_t n, const double *const eri,
                                          const double *const dm3, double *const fdm2) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn) {
    return;
  }
  const int64_t i = at / (nn * n);
  const int64_t j = (at / nn) % n;
  const int64_t k = (at / n) % n;
  const int64_t l = at % n;
  double acc = 0.0;
  for (int64_t cc = 0; cc < n; ++cc) {
    for (int64_t bb = 0; bb < n; ++bb) {
      for (int64_t aa = 0; aa < n; ++aa) {
        const double e = eri[(((l * n + cc) * n + bb) * n) + aa];
        acc += e * dm3[(((((aa * n + bb) * n + cc) * n + k) * n + j) * n) + i];
      }
    }
  }
  fdm2[at] = acc;
}

__global__ void nevpt2_rdm_fdm2_ac_kernel(const int64_t n, const double *const eri,
                                          const double *const dm3, double *const fdm2) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn) {
    return;
  }
  const int64_t i = at / (nn * n);
  const int64_t j = (at / nn) % n;
  const int64_t k = (at / n) % n;
  const int64_t l = at % n;
  double acc = 0.0;
  for (int64_t cc = 0; cc < n; ++cc) {
    for (int64_t ee = 0; ee < n; ++ee) {
      for (int64_t ff = 0; ff < n; ++ff) {
        const double e = eri[(((k * n + cc) * n + ee) * n) + ff];
        acc += e * dm3[(((((i * n + j) * n + cc) * n + l) * n + ee) * n) + ff];
      }
    }
  }
  fdm2[at] = acc;
}

// One thread per (i, j); only j < i writes, so no two threads touch the same
// (first, third) index pair, and what one reads (first index i > third j) no
// thread writes.
__global__ void nevpt2_rdm_wedge_kernel(const int64_t n, const double *const fdm2,
                                        double *const f3) {
  const int64_t ij = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (ij >= n * n) {
    return;
  }
  const int64_t i = ij / n;
  const int64_t j = ij % n;
  if (j >= i) {
    return;
  }
  const auto at6 = [n](const int64_t p, const int64_t q, const int64_t r, const int64_t s,
                       const int64_t t, const int64_t u) {
    return (((((((p * n + q) * n + r) * n + s) * n) + t) * n) + u);
  };
  const auto at4 = [n](const int64_t p, const int64_t q, const int64_t r, const int64_t s) {
    return (((p * n + q) * n + r) * n) + s;
  };
  // f3[j, :, i] = f3[i, :, j] with its two free middle axes swapped.
  for (int64_t b = 0; b < n; ++b) {
    for (int64_t c = 0; c < n; ++c) {
      for (int64_t d = 0; d < n; ++d) {
        for (int64_t e = 0; e < n; ++e) {
          f3[at6(j, b, i, c, d, e)] = f3[at6(i, c, j, b, d, e)];
        }
      }
    }
  }
  for (int64_t c = 0; c < n; ++c) {
    for (int64_t d = 0; d < n; ++d) {
      for (int64_t e = 0; e < n; ++e) {
        f3[at6(j, i, i, c, d, e)] += fdm2[at4(j, c, d, e)];
      }
    }
  }
  for (int64_t b = 0; b < n; ++b) {
    for (int64_t d = 0; d < n; ++d) {
      for (int64_t e = 0; e < n; ++e) {
        f3[at6(j, b, i, j, d, e)] -= fdm2[at4(i, b, d, e)];
      }
    }
  }
}

// Blocks of kFourWarps over @p items threads.
uint32_t nevpt2_rdm_blocks(const int64_t items) {
  return static_cast<uint32_t>(math::idivup(items, int64_t{kFourWarps}));
}

} // namespace

wwr::wwrError_t rdm_convert_eri(const wwr::wwrStream_t stream, const int32_t n,
                                const double *const in, double *const eri, double *const consume) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_convert_eri_kernel<<<nevpt2_rdm_blocks(nn * nn), kFourWarps, 0, stream>>>(n, in, eri,
                                                                                       consume);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_produce(const wwr::wwrStream_t stream, const int32_t n, const int32_t n_beta,
                            const Replacement *const alpha_links, const int32_t alpha_per,
                            const Replacement *const beta_links, const int32_t beta_per,
                            const double *const c, const int32_t begin, const int32_t width,
                            double *const r, double *const l2) {
  nevpt2_rdm_produce_kernel<<<nevpt2_rdm_blocks(width), kFourWarps, 0, stream>>>(
      n, n_beta, alpha_links, alpha_per, beta_links, beta_per, c, begin, width, r, l2);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_fdm2_ca(const wwr::wwrStream_t stream, const int32_t n, const double *const eri,
                            const double *const dm3, double *const fdm2) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_fdm2_ca_kernel<<<nevpt2_rdm_blocks(nn * nn), kFourWarps, 0, stream>>>(n, eri, dm3,
                                                                                   fdm2);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_fdm2_ac(const wwr::wwrStream_t stream, const int32_t n, const double *const eri,
                            const double *const dm3, double *const fdm2) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_fdm2_ac_kernel<<<nevpt2_rdm_blocks(nn * nn), kFourWarps, 0, stream>>>(n, eri, dm3,
                                                                                   fdm2);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_wedge(const wwr::wwrStream_t stream, const int32_t n, const double *const fdm2,
                          double *const f3) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_wedge_kernel<<<nevpt2_rdm_blocks(nn), kFourWarps, 0, stream>>>(n, fdm2, f3);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::nevpt2::device
