// The device half of nevpt2::rdms: the eri's conversion at entry, the per-tile
// R and L2 over each determinant's own replacement lists, and fdm2 and the
// wedge once after the tile loop; and of nevpt2::rdms_adjoint, the transpose
// of each. The products between them (the digests and
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

// The wedge's transpose onto f3 before it: one thread per element P. The
// wedge copied raw[i, c, j, b, d, e] (i > j) onto out[j, b, i, c, d, e] and
// overwrote everything with first index below the third, so those raw
// elements reached nothing, and the ones above reached two places.
__global__ void nevpt2_rdm_wedge_raw_adjoint_kernel(const int64_t n, const double *const f3_bar,
                                                    double *const raw_bar) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;
  if (at >= n4 * nn) {
    return;
  }
  // at is (i, c, j, b, d, e), row-major.
  const int64_t i = at / (n4 * n);
  const int64_t c = (at / n4) % n;
  const int64_t j = (at / (nn * n)) % n;
  const int64_t b = (at / nn) % n;
  const int64_t de = at % nn;
  if (i < j) {
    raw_bar[at] = 0.0;
  } else if (i == j) {
    raw_bar[at] = f3_bar[at];
  } else {
    raw_bar[at] = f3_bar[at] + f3_bar[(((((j * n) + b) * n + i) * n + c) * nn) + de];
  }
}

// The wedge's transpose onto fdm2: one thread per fdm2[i, b, d, e]. The wedge
// added fdm2[j, c, d, e] to f3[j, i, i, c, d, e] for every i > j, and took
// fdm2[i, b, d, e] from f3[j, b, i, j, d, e] for every j < i.
__global__ void nevpt2_rdm_wedge_fdm2_adjoint_kernel(const int64_t n, const double *const f3_bar,
                                                     double *const fdm2_bar) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn) {
    return;
  }
  const int64_t i = at / (nn * n);
  const int64_t b = (at / nn) % n;
  const int64_t de = at % nn;
  // The start of f3's row (p, q, r, s, ·, ·).
  const auto row = [n, nn](const int64_t p, const int64_t q, const int64_t r, const int64_t s) {
    return ((((p * n + q) * n + r) * n + s) * nn);
  };
  double acc = 0.0;
  for (int64_t j = i + 1; j < n; ++j) {
    acc += f3_bar[row(i, j, j, b) + de];
  }
  for (int64_t j = 0; j < i; ++j) {
    acc -= f3_bar[row(j, b, i, j) + de];
  }
  fdm2_bar[at] = acc;
}

// One thread per dm3 element P = (P0, …, P5), each fdm2's sum over the one
// index the element leaves free.
__global__ void nevpt2_rdm_fdm2_dm3_adjoint_kernel(const int64_t n, const double *const eri,
                                                   const double *const fdm2_ca_bar,
                                                   const double *const fdm2_ac_bar,
                                                   const double *const dm3_bar_in,
                                                   double *const dm3_bar_out) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn * nn) {
    return;
  }
  const int64_t p0 = at / (nn * nn * n);
  const int64_t p1 = (at / (nn * nn)) % n;
  const int64_t p2 = (at / (nn * n)) % n;
  const int64_t p3 = (at / nn) % n;
  const int64_t p4 = (at / n) % n;
  const int64_t p5 = at % n;
  const auto at4 = [n](const int64_t p, const int64_t q, const int64_t r, const int64_t s) {
    return (((p * n + q) * n + r) * n) + s;
  };
  double acc = dm3_bar_in[at];
  // ca: dm3[A, B, C, k, j, i] met eri[l, C, B, A] fdm2_ca[i, j, k, l].
  for (int64_t l = 0; l < n; ++l) {
    acc += eri[at4(l, p2, p1, p0)] * fdm2_ca_bar[at4(p5, p4, p3, l)];
  }
  // ac: dm3[i, j, C, l, E, F] met eri[k, C, E, F] fdm2_ac[i, j, k, l].
  for (int64_t k = 0; k < n; ++k) {
    acc += eri[at4(k, p2, p4, p5)] * fdm2_ac_bar[at4(p0, p1, k, p3)];
  }
  dm3_bar_out[at] = acc;
}

// One thread per element of (tu|vw) in the caller's layout, at = t + n (u +
// n (v + n w)): what the conversion read it as, eri[t, u, v, w] (through the
// two fdm2s) and consume[u, t, w, v].
__global__ void nevpt2_rdm_eri_adjoint_kernel(const int64_t n, const double *const dm3,
                                              const double *const fdm2_ca_bar,
                                              const double *const fdm2_ac_bar,
                                              const double *const consume_bar,
                                              double *const eri_bar) {
  const int64_t at = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const int64_t nn = n * n;
  if (at >= nn * nn) {
    return;
  }
  const int64_t t = at % n;
  const int64_t u = (at / n) % n;
  const int64_t v = (at / nn) % n;
  const int64_t w = at / (nn * n);
  const auto at4 = [n](const int64_t p, const int64_t q, const int64_t r, const int64_t s) {
    return (((p * n + q) * n + r) * n) + s;
  };
  const auto at6 = [n, nn](const int64_t p, const int64_t q, const int64_t r, const int64_t s,
                           const int64_t x, const int64_t y) {
    return ((((p * n + q) * n + r) * n + s) * nn) + (x * n) + y;
  };
  double acc = consume_bar[at4(u, t, w, v)];
  // ca: eri[l, C, B, A] = eri[t, u, v, w] met fdm2_ca[i, j, k, t] dm3[w, v, u, k, j, i].
  // ac: eri[k, C, E, F] = eri[t, u, v, w] met fdm2_ac[i, j, t, l] dm3[i, j, u, l, v, w].
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t k = 0; k < n; ++k) {
        acc += fdm2_ca_bar[at4(i, j, k, t)] * dm3[at6(w, v, u, k, j, i)];
        acc += fdm2_ac_bar[at4(i, j, t, k)] * dm3[at6(i, j, u, k, v, w)];
      }
    }
  }
  eri_bar[at] = acc;
}

// rdm_produce's transpose: one thread per determinant J, walking J's own
// lists as the produce walks K's. A link (p, q, M, s) of J is <M| E_pq |J> =
// s, which is R's <K| E_tu |J> at K = M, tu = pq; and it is also <J| E_qp |M>
// = s, so two links in a row, J to M by (p, q) and M to K by (p', q'), are
// <J| E_qp E_q'p' |K> = s s', L2's element (q, p, q', p') of K. Only the K in
// the tile add.
__global__ void nevpt2_rdm_produce_adjoint_kernel(
    const int32_t n, const int32_t n_beta, const int32_t ndet, const Replacement *const alpha_links,
    const int32_t alpha_per, const Replacement *const beta_links, const int32_t beta_per,
    const int32_t begin, const int32_t width, const double *const r_bar, const double *const l2_bar,
    double *const y) {
  const int64_t det = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (det >= ndet) {
    return;
  }
  const auto ja = static_cast<int32_t>(det / n_beta);
  const auto jb = static_cast<int32_t>(det % n_beta);
  const int64_t ld = width;
  // The tile's column of (a, b), or −1 outside it.
  const auto column = [&](const int32_t a, const int32_t b) {
    const int64_t k = (static_cast<int64_t>(a) * n_beta) + b - begin;
    return k >= 0 && k < width ? k : int64_t{-1};
  };
  const auto pair = [&](const Replacement l) {
    return (static_cast<int64_t>(l.p) * n) + static_cast<int64_t>(l.q);
  };
  const auto reversed = [&](const Replacement l) {
    return (static_cast<int64_t>(l.q) * n) + static_cast<int64_t>(l.p);
  };
  const auto links_of = [](const Replacement *const links, const int32_t per, const int32_t s) {
    return links + (static_cast<int64_t>(s) * per);
  };
  const int64_t nn = static_cast<int64_t>(n) * n;
  const Replacement *const a = links_of(alpha_links, alpha_per, ja);
  const Replacement *const b = links_of(beta_links, beta_per, jb);
  double acc = 0.0;

  // R̄: one link of J, to K.
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    if (const int64_t k = column(l.target, jb); k >= 0) {
      acc += static_cast<double>(l.sign) * r_bar[(pair(l) * ld) + k];
    }
  }
  for (int32_t e = 0; e < beta_per; ++e) {
    const Replacement l = b[e];
    if (const int64_t k = column(ja, l.target); k >= 0) {
      acc += static_cast<double>(l.sign) * r_bar[(pair(l) * ld) + k];
    }
  }

  // L2̄: J to M by l, then M to K by o.
  for (int32_t e = 0; e < alpha_per; ++e) {
    const Replacement l = a[e];
    const int64_t pq = reversed(l) * nn;
    const auto s = static_cast<double>(l.sign);
    const Replacement *const m = links_of(alpha_links, alpha_per, l.target);
    for (int32_t f = 0; f < alpha_per; ++f) {
      const Replacement o = m[f];
      if (const int64_t k = column(o.target, jb); k >= 0) {
        acc += s * static_cast<double>(o.sign) * l2_bar[((pq + reversed(o)) * ld) + k];
      }
    }
    for (int32_t f = 0; f < beta_per; ++f) {
      const Replacement o = b[f];
      if (const int64_t k = column(l.target, o.target); k >= 0) {
        acc += s * static_cast<double>(o.sign) * l2_bar[((pq + reversed(o)) * ld) + k];
      }
    }
  }
  for (int32_t e = 0; e < beta_per; ++e) {
    const Replacement l = b[e];
    const int64_t pq = reversed(l) * nn;
    const auto s = static_cast<double>(l.sign);
    for (int32_t f = 0; f < alpha_per; ++f) {
      const Replacement o = a[f];
      if (const int64_t k = column(o.target, l.target); k >= 0) {
        acc += s * static_cast<double>(o.sign) * l2_bar[((pq + reversed(o)) * ld) + k];
      }
    }
    const Replacement *const m = links_of(beta_links, beta_per, l.target);
    for (int32_t f = 0; f < beta_per; ++f) {
      const Replacement o = m[f];
      if (const int64_t k = column(ja, o.target); k >= 0) {
        acc += s * static_cast<double>(o.sign) * l2_bar[((pq + reversed(o)) * ld) + k];
      }
    }
  }
  y[det] += acc;
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

wwr::wwrError_t rdm_wedge_adjoint(const wwr::wwrStream_t stream, const int32_t n,
                                  const double *const f3_bar, double *const raw_bar,
                                  double *const fdm2_bar) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_wedge_raw_adjoint_kernel<<<nevpt2_rdm_blocks(nn * nn * nn), kFourWarps, 0, stream>>>(
      n, f3_bar, raw_bar);
  if (const wwr::wwrError_t status = wwr::wwrGetLastError(); status != wwr::wwrSuccess) {
    return status;
  }
  nevpt2_rdm_wedge_fdm2_adjoint_kernel<<<nevpt2_rdm_blocks(nn * nn), kFourWarps, 0, stream>>>(
      n, f3_bar, fdm2_bar);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_fdm2_dm3_adjoint(const wwr::wwrStream_t stream, const int32_t n,
                                     const double *const eri, const double *const fdm2_ca_bar,
                                     const double *const fdm2_ac_bar,
                                     const double *const dm3_bar_in, double *const dm3_bar_out) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_fdm2_dm3_adjoint_kernel<<<nevpt2_rdm_blocks(nn * nn * nn), kFourWarps, 0, stream>>>(
      n, eri, fdm2_ca_bar, fdm2_ac_bar, dm3_bar_in, dm3_bar_out);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_eri_adjoint(const wwr::wwrStream_t stream, const int32_t n,
                                const double *const dm3, const double *const fdm2_ca_bar,
                                const double *const fdm2_ac_bar, const double *const consume_bar,
                                double *const eri_bar) {
  const int64_t nn = static_cast<int64_t>(n) * n;
  nevpt2_rdm_eri_adjoint_kernel<<<nevpt2_rdm_blocks(nn * nn), kFourWarps, 0, stream>>>(
      n, dm3, fdm2_ca_bar, fdm2_ac_bar, consume_bar, eri_bar);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t rdm_produce_adjoint(const wwr::wwrStream_t stream, const int32_t n,
                                    const int32_t n_beta, const int32_t ndet,
                                    const Replacement *const alpha_links, const int32_t alpha_per,
                                    const Replacement *const beta_links, const int32_t beta_per,
                                    const int32_t begin, const int32_t width,
                                    const double *const r_bar, const double *const l2_bar,
                                    double *const y) {
  nevpt2_rdm_produce_adjoint_kernel<<<nevpt2_rdm_blocks(ndet), kFourWarps, 0, stream>>>(
      n, n_beta, ndet, alpha_links, alpha_per, beta_links, beta_per, begin, width, r_bar, l2_bar,
      y);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::nevpt2::device
