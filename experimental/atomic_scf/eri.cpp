/// @file
/// @brief The one-center two-electron integrals, every shell quartet.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;
import :hermite;
import :eri;

namespace free_breakfast {

std::vector<double> one_center_eri(const BasisSet &basis) {
  const size_t ncart = basis.nao_cart();
  const std::vector<Shell> &shells = basis.shells();
  const size_t nsh = shells.size();
  std::vector<std::vector<std::array<int32_t, 3>>> powers;
  for (const Shell &shell : shells) {
    powers.push_back(cartesian_powers(shell.angular_momentum()));
  }

  std::vector<double> eri(ncart * ncart * ncart * ncart, 0.0);
  const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5);
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
                          eri[idx] += scale * sum;
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
  return eri;
}

} // namespace free_breakfast
