/// @file
/// @brief One atom's two-electron integrals over the Cartesian layout.

// An internal partition of free_breakfast.sad. atomic_integrals
// calls it and carries the tensor to the spherical layout.
module free_breakfast.sad:eri;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;
import :hermite;

namespace free_breakfast {

/// @brief (ab|cd) over @p basis's Cartesian layout, every shell on one center:
/// nao_cart^4 values, (ab|cd) at a + n*(b + n*(c + n*d)).
///
/// (ab|cd) = 2 pi^{5/2} / (p q sqrt(p+q)) Sum_{tuv} E^{ab}_{tuv}
///   Sum_{t'u'v'} (-1)^{t'+u'+v'} E^{cd}_{t'u'v'} R_{t+t',u+u',v+v'}(alpha),
/// alpha = p q / (p + q). Only the canonical shell quartets are evaluated,
/// sa >= sb, sc >= sd and pair (sa, sb) >= pair (sc, sd), and each value is
/// written to all eight of (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) =
/// (dc|ab) = (cd|ba) = (dc|ba), so the tensor is exactly symmetric.
[[nodiscard]] std::vector<double> one_center_eri(const BasisSet &basis) {
  const size_t ncart = basis.nao_cart();
  const std::vector<Shell> &shells = basis.shells();
  const size_t nsh = shells.size();
  std::vector<std::vector<std::array<int32_t, 3>>> powers;
  for (const Shell &shell : shells) {
    powers.push_back(cartesian_powers(shell.angular_momentum()));
  }

  std::vector<double> eri(ncart * ncart * ncart * ncart, 0.0);
  const auto at = [ncart](size_t a, size_t b, size_t c, size_t d) {
    return a + (ncart * (b + (ncart * (c + (ncart * d)))));
  };
  const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5);
  std::vector<double> block;
  for (size_t sa = 0; sa < nsh; ++sa) {
    for (size_t sb = 0; sb <= sa; ++sb) {
      for (size_t sc = 0; sc <= sa; ++sc) {
        // Pair (sc, sd) may not pass pair (sa, sb): at sc == sa, sd stops at sb.
        const size_t sd_last = (sc == sa) ? sb : sc;
        for (size_t sd = 0; sd <= sd_last; ++sd) {
          const std::array<size_t, 4> off{basis.ao_offset_cart(static_cast<uint32_t>(sa)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sb)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sc)),
                                          basis.ao_offset_cart(static_cast<uint32_t>(sd))};
          const std::array<size_t, 4> len{powers[sa].size(), powers[sb].size(), powers[sc].size(),
                                          powers[sd].size()};
          const int32_t lab = shells[sa].angular_momentum() + shells[sb].angular_momentum();
          const int32_t lcd = shells[sc].angular_momentum() + shells[sd].angular_momentum();
          // The quartet's values, contracted over every primitive before they
          // are scattered: i + na*(j + nb*(k + nc*l)).
          block.assign(len[0] * len[1] * len[2] * len[3], 0.0);
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
                  for (size_t i = 0; i < len[0]; ++i) {
                    for (size_t j = 0; j < len[1]; ++j) {
                      const auto &a = powers[sa][i];
                      const auto &b = powers[sb][j];
                      const int32_t nx = a[0] + b[0];
                      const int32_t ny = a[1] + b[1];
                      const int32_t nz = a[2] + b[2];
                      for (size_t k = 0; k < len[2]; ++k) {
                        for (size_t l = 0; l < len[3]; ++l) {
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
                          block[i + (len[0] * (j + (len[1] * (k + (len[2] * l)))))] += scale * sum;
                        }
                      }
                    }
                  }
                }
              }
            }
          }

          // Assigned, not accumulated: every element of the tensor lies in
          // exactly one canonical quartet's orbit, and a quartet with a
          // repeated shell or pair writes some elements more than once, always
          // with the same value.
          for (size_t l = 0; l < len[3]; ++l) {
            for (size_t k = 0; k < len[2]; ++k) {
              for (size_t j = 0; j < len[1]; ++j) {
                for (size_t i = 0; i < len[0]; ++i) {
                  const double v = block[i + (len[0] * (j + (len[1] * (k + (len[2] * l)))))];
                  const size_t a = off[0] + i;
                  const size_t b = off[1] + j;
                  const size_t c = off[2] + k;
                  const size_t d = off[3] + l;
                  eri[at(a, b, c, d)] = v;
                  eri[at(b, a, c, d)] = v;
                  eri[at(a, b, d, c)] = v;
                  eri[at(b, a, d, c)] = v;
                  eri[at(c, d, a, b)] = v;
                  eri[at(d, c, a, b)] = v;
                  eri[at(c, d, b, a)] = v;
                  eri[at(d, c, b, a)] = v;
                }
              }
            }
          }
        }
      }
    }
  }
  return eri;
}

} // namespace free_breakfast
