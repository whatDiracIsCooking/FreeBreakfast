/// @file
/// @brief Implementation of the atomic SCF: one-center McMurchie-Davidson
/// integrals, the per-l SCF, and the molecular assembly.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives;
import free_breakfast.basis;
import free_breakfast.sad;

#include "diagnostics.h"

namespace free_breakfast {

// The n^4 two-electron tensor is held whole, over the Cartesian layout first:
// 96^4 doubles is 680 MB. An atom in a triple-zeta basis is well inside it.
constexpr uint32_t kMaxAtomicAoCart = 96;

// Linear dependence within one l block: canonical orthogonalization drops the
// overlap eigenvectors below this, relative to the largest.
constexpr double kAtomicLinearDependence = 1e-10;

// DIIS history length.
constexpr size_t kAtomicDiisDepth = 8;

// ---------------------------------------------------------------------------
// One-center integrals
// ---------------------------------------------------------------------------

// Hermite expansion coefficients of x^n exp(-p x^2) about its own center:
// x^n exp(-p x^2) = Sum_t e[n][t] (d/dx)^t exp(-p x^2). With every
// McMurchie-Davidson distance zero, the two-center recurrence reduces to
//   e[n+1][t] = e[n][t-1] / (2p) + (t+1) e[n][t+1],   e[0][0] = 1,
// and e[n][t] vanishes unless t and n have the same parity.
std::vector<std::vector<double>> one_center_hermite(int32_t nmax, double p) {
  const auto size = static_cast<size_t>(nmax) + 1;
  std::vector<std::vector<double>> e(size, std::vector<double>(size + 1, 0.0));
  e[0][0] = 1.0;
  for (size_t n = 0; n + 1 < size; ++n) {
    for (size_t t = 0; t <= n + 1; ++t) {
      const double down = (t > 0) ? e[n][t - 1] / (2.0 * p) : 0.0;
      const double up = static_cast<double>(t + 1) * e[n][t + 1];
      e[n + 1][t] = down + up;
    }
  }
  return e;
}

// R_{tuv}(alpha) at zero distance: the (t, u, v)-th derivative of F_0(alpha R^2)
// at R = 0. Only even t, u, v survive, and then it is
//   (t-1)!! (u-1)!! (v-1)!! (-2 alpha)^{N/2} / (N + 1),   N = t + u + v,
// the recurrence R^n_{t+1} = t R^{n+1}_{t-1} unrolled down to
// R^n_{000} = (-2 alpha)^n F_n(0) with F_n(0) = 1 / (2n + 1).
double one_center_hermite_coulomb(int32_t t, int32_t u, int32_t v, double alpha) {
  if ((t % 2) != 0 || (u % 2) != 0 || (v % 2) != 0) {
    return 0.0;
  }
  const int32_t half = (t + u + v) / 2;
  return math::double_factorial_odd(t - 1) * math::double_factorial_odd(u - 1) *
         math::double_factorial_odd(v - 1) * std::pow(-2.0 * alpha, half) /
         static_cast<double>((2 * half) + 1);
}

// Rank-agnostic change of one index: contracts the slowest index of @p in
// (extent u_rows) with the u_rows x u_cols column-major @p u and makes the result
// the fastest index. Applied once per index, it carries a rank-r tensor over
// the Cartesian layout to the spherical one with every index back in place.
std::vector<double> rotate_transform(const std::vector<double> &in, const std::vector<double> &u,
                                     size_t u_rows, size_t u_cols) {
  const size_t rest = in.size() / u_rows;
  std::vector<double> out(rest * u_cols, 0.0);
  for (size_t d = 0; d < u_rows; ++d) {
    for (size_t l = 0; l < u_cols; ++l) {
      const double w = u[d + (u_rows * l)];
      if (w == 0.0) {
        continue;
      }
      for (size_t r = 0; r < rest; ++r) {
        out[l + (u_cols * r)] += w * in[r + (rest * d)];
      }
    }
  }
  return out;
}

std::vector<double> to_spherical(std::vector<double> m, const std::vector<double> &u, size_t ncart,
                                 size_t nsph, int32_t rank) {
  for (int32_t i = 0; i < rank; ++i) {
    m = rotate_transform(m, u, ncart, nsph);
  }
  return m;
}

Result<AtomicIntegrals> atomic_integrals(int32_t z, std::span<const Shell> shells) {
  if (shells.empty()) {
    return err_config("atomic_integrals: the atom has no shells");
  }
  const Vec<double, 3> center = shells.front().center();
  for (const Shell &shell : shells) {
    const Vec<double, 3> c = shell.center();
    if (c.x != center.x || c.y != center.y || c.z != center.z) {
      return err_config("atomic_integrals: the shells are not all on one center");
    }
  }
  const BasisSet basis(std::vector<Shell>(shells.begin(), shells.end()));
  const size_t ncart = basis.nao_cart();
  const size_t nsph = basis.nao_sphe();
  if (ncart > kMaxAtomicAoCart) {
    return err_unsupported("atomic_integrals: {} Cartesian AOs, past the {} the n^4 tensor is "
                           "held for",
                           ncart, kMaxAtomicAoCart);
  }

  std::vector<std::vector<std::array<int32_t, 3>>> powers;
  int32_t lmax = 0;
  for (const Shell &shell : shells) {
    powers.push_back(cartesian_powers(shell.angular_momentum()));
    lmax = std::max(lmax, shell.angular_momentum());
  }

  std::vector<double> overlap(ncart * ncart, 0.0);
  std::vector<double> kinetic(ncart * ncart, 0.0);
  std::vector<double> nuclear(ncart * ncart, 0.0);
  const double charge = static_cast<double>(z);
  const double pi = std::numbers::pi;

  for (size_t sa = 0; sa < shells.size(); ++sa) {
    for (size_t sb = 0; sb < shells.size(); ++sb) {
      const size_t oa = basis.ao_offset_cart(static_cast<uint32_t>(sa));
      const size_t ob = basis.ao_offset_cart(static_cast<uint32_t>(sb));
      const int32_t lab = shells[sa].angular_momentum() + shells[sb].angular_momentum();
      for (const Primitive &pa : shells[sa].primitives()) {
        for (const Primitive &pb : shells[sb].primitives()) {
          const double p = pa.exponent + pb.exponent;
          const double beta = pb.exponent;
          const double cc = pa.coefficient * pb.coefficient;
          // +2 for the kinetic energy's x^{n+2}.
          const auto e = one_center_hermite(lab + 2, p);
          const double s1 = std::sqrt(pi / p);
          const auto overlap_1d = [&](int32_t n) { return n < 0 ? 0.0 : e[n][0] * s1; };
          for (size_t i = 0; i < powers[sa].size(); ++i) {
            for (size_t j = 0; j < powers[sb].size(); ++j) {
              const auto &a = powers[sa][i];
              const auto &b = powers[sb][j];
              std::array<double, 3> s{};
              std::array<double, 3> t{};
              for (size_t x = 0; x < 3; ++x) {
                const int32_t n = a[x] + b[x];
                s[x] = overlap_1d(n);
                // d^2/dx^2 on the ket's x^b exp(-beta x^2), then integrated.
                const auto bx = static_cast<double>(b[x]);
                t[x] = (bx * (bx - 1.0) * overlap_1d(n - 2)) -
                       (2.0 * beta * ((2.0 * bx) + 1.0) * overlap_1d(n)) +
                       (4.0 * beta * beta * overlap_1d(n + 2));
              }
              const double sv = s[0] * s[1] * s[2];
              const double tv =
                  -0.5 * ((t[0] * s[1] * s[2]) + (s[0] * t[1] * s[2]) + (s[0] * s[1] * t[2]));
              double vv = 0.0;
              const int32_t nx = a[0] + b[0];
              const int32_t ny = a[1] + b[1];
              const int32_t nz = a[2] + b[2];
              for (int32_t tx = nx % 2; tx <= nx; tx += 2) {
                for (int32_t ty = ny % 2; ty <= ny; ty += 2) {
                  for (int32_t tz = nz % 2; tz <= nz; tz += 2) {
                    vv += e[nx][tx] * e[ny][ty] * e[nz][tz] *
                          one_center_hermite_coulomb(tx, ty, tz, p);
                  }
                }
              }
              const size_t idx = (oa + i) + (ncart * (ob + j));
              overlap[idx] += cc * sv;
              kinetic[idx] += cc * tv;
              nuclear[idx] += cc * (-charge * 2.0 * pi / p) * vv;
            }
          }
        }
      }
    }
  }

  // Two-electron: (ab|cd) = 2 pi^{5/2} / (p q sqrt(p+q))
  //   Sum_{tuv} E^{ab}_{tuv} Sum_{t'u'v'} (-1)^{t'+u'+v'} E^{cd}_{t'u'v'} R_{t+t',u+u',v+v'}(alpha)
  // with alpha = p q / (p + q). Every shell quartet, no permutational
  // symmetry: an atom's quartet count is small, and this is checked code.
  std::vector<double> repulsion(ncart * ncart * ncart * ncart, 0.0);
  const double prefactor = 2.0 * std::pow(pi, 2.5);
  const size_t nsh = shells.size();
  for (size_t sa = 0; sa < nsh; ++sa) {
    for (size_t sb = 0; sb < nsh; ++sb) {
      for (size_t sc = 0; sc < nsh; ++sc) {
        for (size_t sd = 0; sd < nsh; ++sd) {
          const std::array<size_t, 4> off{basis.ao_offset_cart(static_cast<uint32_t>(sa)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sb)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sc)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sd))};
          const int32_t lab = shells[sa].angular_momentum() + shells[sb].angular_momentum();
          const int32_t lcd = shells[sc].angular_momentum() + shells[sd].angular_momentum();
          for (const Primitive &pa : shells[sa].primitives()) {
            for (const Primitive &pb : shells[sb].primitives()) {
              const double p = pa.exponent + pb.exponent;
              const auto ep = one_center_hermite(lab, p);
              for (const Primitive &pc : shells[sc].primitives()) {
                for (const Primitive &pd : shells[sd].primitives()) {
                  const double q = pc.exponent + pd.exponent;
                  const auto eq = one_center_hermite(lcd, q);
                  const double alpha = p * q / (p + q);
                  const double scale = pa.coefficient * pb.coefficient * pc.coefficient *
                                       pd.coefficient * prefactor / (p * q * std::sqrt(p + q));
                  for (size_t i = 0; i < powers[sa].size(); ++i) {
                    for (size_t j = 0; j < powers[sb].size(); ++j) {
                      const auto &a = powers[sa][i];
                      const auto &b = powers[sb][j];
                      const int32_t nx = a[0] + b[0];
                      const int32_t ny = a[1] + b[1];
                      const int32_t nz = a[2] + b[2];
                      for (size_t k = 0; k < powers[sc].size(); ++k) {
                        for (size_t l = 0; l < powers[sd].size(); ++l) {
                          const auto &c = powers[sc][k];
                          const auto &d = powers[sd][l];
                          const int32_t mx = c[0] + d[0];
                          const int32_t my = c[1] + d[1];
                          const int32_t mz = c[2] + d[2];
                          // Odd total power on any axis integrates to zero.
                          if (((nx + mx) % 2) != 0 || ((ny + my) % 2) != 0 ||
                              ((nz + mz) % 2) != 0) {
                            continue;
                          }
                          double sum = 0.0;
                          for (int32_t tx = nx % 2; tx <= nx; tx += 2) {
                            for (int32_t ty = ny % 2; ty <= ny; ty += 2) {
                              for (int32_t tz = nz % 2; tz <= nz; tz += 2) {
                                const double eab = ep[nx][tx] * ep[ny][ty] * ep[nz][tz];
                                for (int32_t ux = mx % 2; ux <= mx; ux += 2) {
                                  for (int32_t uy = my % 2; uy <= my; uy += 2) {
                                    for (int32_t uz = mz % 2; uz <= mz; uz += 2) {
                                      const double sign = (((ux + uy + uz) % 2) == 0) ? 1.0 : -1.0;
                                      sum += eab * sign * eq[mx][ux] * eq[my][uy] * eq[mz][uz] *
                                             one_center_hermite_coulomb(tx + ux, ty + uy, tz + uz,
                                                                        alpha);
                                    }
                                  }
                                }
                              }
                            }
                          }
                          const size_t idx =
                              (off[0] + i) +
                              (ncart *
                               ((off[1] + j) + (ncart * ((off[2] + k) + (ncart * (off[3] + l))))));
                          repulsion[idx] += scale * sum;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  const std::vector<double> u = spherical_transform_matrix(basis);
  return AtomicIntegrals{
      .nao_sphe = static_cast<uint32_t>(nsph),
      .overlap = to_spherical(std::move(overlap), u, ncart, nsph, 2),
      .kinetic = to_spherical(std::move(kinetic), u, ncart, nsph, 2),
      .nuclear = to_spherical(std::move(nuclear), u, ncart, nsph, 2),
      .repulsion = to_spherical(std::move(repulsion), u, ncart, nsph, 4),
  };
}

// ---------------------------------------------------------------------------
// Small dense linear algebra, n x n column-major
// ---------------------------------------------------------------------------

// C = A^T B for A (k x m) and B (k x n).
std::vector<double> atomic_gemm_tn(const std::vector<double> &a, const std::vector<double> &b,
                                   size_t k, size_t m, size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      double sum = 0.0;
      for (size_t r = 0; r < k; ++r) {
        sum += a[r + (k * i)] * b[r + (k * j)];
      }
      c[i + (m * j)] = sum;
    }
  }
  return c;
}

// C = A B for A (m x k) and B (k x n).
std::vector<double> atomic_gemm_nn(const std::vector<double> &a, const std::vector<double> &b,
                                   size_t m, size_t k, size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (size_t j = 0; j < n; ++j) {
    for (size_t r = 0; r < k; ++r) {
      const double w = b[r + (k * j)];
      for (size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += a[i + (m * r)] * w;
      }
    }
  }
  return c;
}

struct SymmetricEigen {
  std::vector<double> values;  // ascending
  std::vector<double> vectors; // n x n, column j for values[j]
};

// Cyclic Jacobi. The l blocks are a handful of shells, where it is exact to
// round-off and needs no LAPACK.
SymmetricEigen atomic_jacobi_eigen(std::vector<double> a, size_t n) {
  std::vector<double> v(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    v[i + (n * i)] = 1.0;
  }
  for (int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < j; ++i) {
        off += a[i + (n * j)] * a[i + (n * j)];
      }
    }
    if (off < 1e-30) {
      break;
    }
    for (size_t p = 0; p < n; ++p) {
      for (size_t q = p + 1; q < n; ++q) {
        const double apq = a[p + (n * q)];
        if (std::abs(apq) < 1e-300) {
          continue;
        }
        const double theta = (a[q + (n * q)] - a[p + (n * p)]) / (2.0 * apq);
        const double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
        const double c = 1.0 / std::hypot(t, 1.0);
        const double s = t * c;
        for (size_t k = 0; k < n; ++k) {
          const double akp = a[k + (n * p)];
          const double akq = a[k + (n * q)];
          a[k + (n * p)] = (c * akp) - (s * akq);
          a[k + (n * q)] = (s * akp) + (c * akq);
        }
        for (size_t k = 0; k < n; ++k) {
          const double apk = a[p + (n * k)];
          const double aqk = a[q + (n * k)];
          a[p + (n * k)] = (c * apk) - (s * aqk);
          a[q + (n * k)] = (s * apk) + (c * aqk);
        }
        for (size_t k = 0; k < n; ++k) {
          const double vkp = v[k + (n * p)];
          const double vkq = v[k + (n * q)];
          v[k + (n * p)] = (c * vkp) - (s * vkq);
          v[k + (n * q)] = (s * vkp) + (c * vkq);
        }
      }
    }
  }
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::ranges::sort(order, [&](size_t x, size_t y) { return a[x + (n * x)] < a[y + (n * y)]; });
  SymmetricEigen out{std::vector<double>(n), std::vector<double>(n * n)};
  for (size_t j = 0; j < n; ++j) {
    out.values[j] = a[order[j] + (n * order[j])];
    std::copy_n(v.begin() + static_cast<std::ptrdiff_t>(n * order[j]), n,
                out.vectors.begin() + static_cast<std::ptrdiff_t>(n * j));
  }
  return out;
}

// Solve A x = b by Gaussian elimination with partial pivoting; nullopt if a
// pivot vanishes (the DIIS matrix goes singular as its residuals align).
std::optional<std::vector<double>> atomic_solve(std::vector<double> a, std::vector<double> b,
                                                size_t n) {
  for (size_t col = 0; col < n; ++col) {
    size_t pivot = col;
    for (size_t r = col + 1; r < n; ++r) {
      if (std::abs(a[r + (n * col)]) > std::abs(a[pivot + (n * col)])) {
        pivot = r;
      }
    }
    if (std::abs(a[pivot + (n * col)]) < 1e-14) {
      return std::nullopt;
    }
    if (pivot != col) {
      for (size_t c = 0; c < n; ++c) {
        std::swap(a[col + (n * c)], a[pivot + (n * c)]);
      }
      std::swap(b[col], b[pivot]);
    }
    for (size_t r = col + 1; r < n; ++r) {
      const double f = a[r + (n * col)] / a[col + (n * col)];
      for (size_t c = col; c < n; ++c) {
        a[r + (n * c)] -= f * a[col + (n * c)];
      }
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n, 0.0);
  for (size_t r = n; r-- > 0;) {
    double sum = b[r];
    for (size_t c = r + 1; c < n; ++c) {
      sum -= a[r + (n * c)] * x[c];
    }
    x[r] = sum / a[r + (n * r)];
  }
  return x;
}

// ---------------------------------------------------------------------------
// The SCF
// ---------------------------------------------------------------------------

// One momentum's radial block: the atom's shells of that l, in order, and the
// subshells the aufbau fill seats on its orbitals.
struct AngularBlock {
  int32_t l{0};
  std::vector<size_t> ao_first;   // each shell's first spherical AO (its m = -l)
  size_t n{0};                    // shells of this l
  std::vector<double> overlap;    // n x n
  std::vector<double> x;          // n x m canonical orthogonalizer, X^T S X = 1
  size_t m{0};                    // independent orbitals
  std::vector<double> occupation; // per orbital, spin-summed, over all 2l+1 m
  std::vector<double> density;    // n x n, per m: Sum_k occ_k / (2l+1) c_k c_k^T
};

// The m-averaged l block of a full spherical matrix.
std::vector<double> l_block_average(const AngularBlock &block, const std::vector<double> &full,
                                    size_t nao) {
  const size_t degen = static_cast<size_t>((2 * block.l) + 1);
  std::vector<double> out(block.n * block.n, 0.0);
  for (size_t j = 0; j < block.n; ++j) {
    for (size_t i = 0; i < block.n; ++i) {
      double sum = 0.0;
      for (size_t mm = 0; mm < degen; ++mm) {
        sum += full[(block.ao_first[i] + mm) + (nao * (block.ao_first[j] + mm))];
      }
      out[i + (block.n * j)] = sum / static_cast<double>(degen);
    }
  }
  return out;
}

// Diagonalize F in the block's orthonormal span; refresh its density. Returns
// the orbital energies.
std::vector<double> l_block_diagonalize(AngularBlock &block, const std::vector<double> &fock) {
  const std::vector<double> fx = atomic_gemm_nn(fock, block.x, block.n, block.n, block.m);
  const std::vector<double> xfx = atomic_gemm_tn(block.x, fx, block.n, block.m, block.m);
  const SymmetricEigen eig = atomic_jacobi_eigen(xfx, block.m);
  const std::vector<double> c = atomic_gemm_nn(block.x, eig.vectors, block.n, block.m, block.m);
  const auto degen = static_cast<double>((2 * block.l) + 1);
  block.density.assign(block.n * block.n, 0.0);
  for (size_t k = 0; k < block.occupation.size(); ++k) {
    const double w = block.occupation[k] / degen;
    for (size_t j = 0; j < block.n; ++j) {
      for (size_t i = 0; i < block.n; ++i) {
        block.density[i + (block.n * j)] += w * (c[i + (block.n * k)] * c[j + (block.n * k)]);
      }
    }
  }
  return eig.values;
}

Result<AtomicScf> atomic_scf(int32_t z, std::span<const Shell> shells, double tolerance,
                             int32_t max_iterations) {
  const AtomicIntegrals ints = FBF_TRY(atomic_integrals(z, shells));
  const size_t nao = ints.nao_sphe;

  // The blocks, and each one's share of the aufbau fill.
  int32_t lmax = 0;
  for (const Shell &shell : shells) {
    lmax = std::max(lmax, shell.angular_momentum());
  }
  std::vector<AngularBlock> blocks(static_cast<size_t>(lmax) + 1);
  {
    size_t ao = 0;
    for (const Shell &shell : shells) {
      AngularBlock &block = blocks[static_cast<size_t>(shell.angular_momentum())];
      block.ao_first.push_back(ao);
      ao += shell.function_count_sphe();
    }
  }
  for (size_t l = 0; l < blocks.size(); ++l) {
    AngularBlock &block = blocks[l];
    block.l = static_cast<int32_t>(l);
    block.n = block.ao_first.size();
    block.overlap = l_block_average(block, ints.overlap, nao);
    if (block.n > 0) {
      const SymmetricEigen s = atomic_jacobi_eigen(block.overlap, block.n);
      const double largest = s.values.back();
      for (size_t k = 0; k < block.n; ++k) {
        if (s.values[k] <= kAtomicLinearDependence * largest) {
          continue;
        }
        const double inv = 1.0 / std::sqrt(s.values[k]);
        for (size_t i = 0; i < block.n; ++i) {
          block.x.push_back(s.vectors[i + (block.n * k)] * inv);
        }
        ++block.m;
      }
    }
  }
  const std::vector<int32_t> occupation = subshell_occupation(z);
  for (size_t k = 0; k < occupation.size(); ++k) {
    const auto l = static_cast<size_t>(kAufbauOrder[k].l);
    if (l >= blocks.size() || blocks[l].occupation.size() >= blocks[l].m) {
      return err_config("atomic_scf: Z={} occupies more l={} subshells than the basis has "
                        "independent orbitals of that momentum",
                        z, l);
    }
    blocks[l].occupation.push_back(static_cast<double>(occupation[k]));
  }

  std::vector<double> hcore(nao * nao);
  for (size_t i = 0; i < hcore.size(); ++i) {
    hcore[i] = ints.kinetic[i] + ints.nuclear[i];
  }
  for (AngularBlock &block : blocks) {
    if (block.m > 0) {
      (void)l_block_diagonalize(block, l_block_average(block, hcore, nao));
    }
  }

  std::deque<std::vector<double>> diis_fock;
  std::deque<std::vector<double>> diis_error;
  double previous_energy = 0.0;
  for (int32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    // The full spherical density from the blocks: P_l on every m.
    std::vector<double> density(nao * nao, 0.0);
    for (const AngularBlock &block : blocks) {
      const size_t degen = static_cast<size_t>((2 * block.l) + 1);
      for (size_t j = 0; j < block.n; ++j) {
        for (size_t i = 0; i < block.n; ++i) {
          for (size_t mm = 0; mm < degen; ++mm) {
            density[(block.ao_first[i] + mm) + (nao * (block.ao_first[j] + mm))] =
                block.density[i + (block.n * j)];
          }
        }
      }
    }

    // F = H + J - K/2, J_ij = (ij|kl) D_kl, K_ij = (ik|jl) D_kl.
    std::vector<double> fock = hcore;
    for (size_t j = 0; j < nao; ++j) {
      for (size_t i = 0; i < nao; ++i) {
        double coulomb = 0.0;
        double exchange = 0.0;
        for (size_t l = 0; l < nao; ++l) {
          for (size_t k = 0; k < nao; ++k) {
            const double d = density[k + (nao * l)];
            coulomb += ints.repulsion[i + (nao * (j + (nao * (k + (nao * l)))))] * d;
            exchange += ints.repulsion[i + (nao * (k + (nao * (j + (nao * l)))))] * d;
          }
        }
        fock[i + (nao * j)] += coulomb - (0.5 * exchange);
      }
    }
    double energy = 0.0;
    for (size_t i = 0; i < nao * nao; ++i) {
      energy += 0.5 * density[i] * (hcore[i] + fock[i]);
    }

    // Per block: the averaged Fock matrix and its commutator with P against S.
    std::vector<double> stacked_fock;
    std::vector<double> stacked_error;
    double max_error = 0.0;
    for (const AngularBlock &block : blocks) {
      const std::vector<double> f = l_block_average(block, fock, nao);
      const std::vector<double> fp = atomic_gemm_nn(f, block.density, block.n, block.n, block.n);
      const std::vector<double> fps = atomic_gemm_nn(fp, block.overlap, block.n, block.n, block.n);
      for (size_t j = 0; j < block.n; ++j) {
        for (size_t i = 0; i < block.n; ++i) {
          // S P F = (F P S)^T: every factor is symmetric.
          const double e = fps[i + (block.n * j)] - fps[j + (block.n * i)];
          stacked_error.push_back(e);
          max_error = std::max(max_error, std::abs(e));
        }
      }
      stacked_fock.insert(stacked_fock.end(), f.begin(), f.end());
    }

    if (max_error < tolerance && std::abs(energy - previous_energy) < tolerance) {
      AtomicScf out{.energy = energy, .iterations = iteration, .density = std::move(density)};
      for (AngularBlock &block : blocks) {
        out.orbital_energies.push_back(
            block.m > 0 ? l_block_diagonalize(block, l_block_average(block, fock, nao))
                        : std::vector<double>{});
      }
      return out;
    }
    previous_energy = energy;

    // Pulay DIIS over the stacked blocks.
    diis_fock.push_back(std::move(stacked_fock));
    diis_error.push_back(std::move(stacked_error));
    if (diis_fock.size() > kAtomicDiisDepth) {
      diis_fock.pop_front();
      diis_error.pop_front();
    }
    std::vector<double> extrapolated;
    while (extrapolated.empty()) {
      const size_t depth = diis_fock.size();
      const size_t dim = depth + 1;
      std::vector<double> b(dim * dim, 0.0);
      std::vector<double> rhs(dim, 0.0);
      for (size_t i = 0; i < depth; ++i) {
        for (size_t j = 0; j < depth; ++j) {
          b[i + (dim * j)] = std::inner_product(diis_error[i].begin(), diis_error[i].end(),
                                                diis_error[j].begin(), 0.0);
        }
        b[i + (dim * depth)] = -1.0;
        b[depth + (dim * i)] = -1.0;
      }
      rhs[depth] = -1.0;
      const std::optional<std::vector<double>> w = atomic_solve(b, rhs, dim);
      if (!w && depth > 1) {
        diis_fock.pop_front();
        diis_error.pop_front();
        continue;
      }
      extrapolated.assign(diis_fock.back().size(), 0.0);
      for (size_t i = 0; i < depth; ++i) {
        const double weight = w ? (*w)[i] : 1.0;
        for (size_t x = 0; x < extrapolated.size(); ++x) {
          extrapolated[x] += weight * diis_fock[i][x];
        }
      }
    }

    size_t at = 0;
    for (AngularBlock &block : blocks) {
      const std::vector<double> f(extrapolated.begin() + static_cast<std::ptrdiff_t>(at),
                                  extrapolated.begin() +
                                      static_cast<std::ptrdiff_t>(at + (block.n * block.n)));
      at += block.n * block.n;
      if (block.m > 0) {
        (void)l_block_diagonalize(block, f);
      }
    }
  }
  return err_max_iterations_exceeded("atomic_scf: Z={} not converged to {} in {} iterations", z,
                                     tolerance, max_iterations);
}

// ---------------------------------------------------------------------------
// The molecular guess
// ---------------------------------------------------------------------------

bool same_shells(std::span<const Shell> a, std::span<const Shell> b) {
  return std::ranges::equal(a, b, [](const Shell &x, const Shell &y) {
    return x.angular_momentum() == y.angular_momentum() &&
           std::ranges::equal(x.primitives(), y.primitives(),
                              [](const Primitive &p, const Primitive &q) {
                                return p.exponent == q.exponent && p.coefficient == q.coefficient;
                              });
  });
}

Result<std::vector<double>> atomic_scf_density(const BasisSet &basis, std::span<const Atom> atoms) {
  if (basis.natoms() != atoms.size()) {
    return err_config("atomic_scf_density: the basis groups its shells over {} atoms but {} were "
                      "given; the basis was not placed over these atoms",
                      basis.natoms(), atoms.size());
  }
  const size_t n = basis.nao_sphe();
  std::vector<double> density(n * n, 0.0);

  struct Solved {
    int32_t z;
    std::span<const Shell> shells;
    std::vector<double> density;
  };
  std::vector<Solved> solved;

  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t count = basis.atom_shell_count(a);
    if (count == 0) {
      continue;
    }
    const uint32_t first = basis.atom_shell_offset(a);
    const std::span<const Shell> shells(basis.shells().data() + first, count);
    const int32_t z = nuclear_charge(atoms[a]);
    auto it = std::ranges::find_if(
        solved, [&](const Solved &s) { return s.z == z && same_shells(s.shells, shells); });
    if (it == solved.end()) {
      AtomicScf scf = FBF_TRY(atomic_scf(z, shells));
      solved.push_back(Solved{z, shells, std::move(scf.density)});
      it = std::prev(solved.end());
    }
    const size_t offset = basis.ao_offset_sphe(first);
    const size_t block = basis.ao_offset_sphe(first + count - 1) +
                         basis.shells()[first + count - 1].function_count_sphe() - offset;
    for (size_t j = 0; j < block; ++j) {
      for (size_t i = 0; i < block; ++i) {
        density[(offset + i) + (n * (offset + j))] = it->density[i + (block * j)];
      }
    }
  }
  return density;
}

} // namespace free_breakfast
