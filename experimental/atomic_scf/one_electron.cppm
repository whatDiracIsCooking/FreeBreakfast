/// @file
/// @brief One atom's one-electron matrices over the Cartesian layout: S, T and
/// the attraction V to its own nucleus.

// A private interface partition: nothing here is exported. atomic_integrals
// calls these and carries their results to the spherical layout.
export module free_breakfast.experimental.atomic_scf:one_electron;

import std;
import free_breakfast.core.types;
import free_breakfast.basis;
import :hermite;

namespace free_breakfast {

// The loop the three matrices share: every shell pair, every primitive pair,
// every Cartesian component pair, accumulating cc * integrand(a, b, p, beta, e)
// into (a, b). The Hermite table runs to la + lb + @p extra_n.
template <typename Integrand>
std::vector<double> one_center_pair_loop(const BasisSet &basis, int32_t extra_n,
                                         Integrand integrand) {
  const size_t ncart = basis.nao_cart();
  const std::vector<Shell> &shells = basis.shells();
  std::vector<double> out(ncart * ncart, 0.0);
  for (size_t sa = 0; sa < shells.size(); ++sa) {
    const auto powers_a = cartesian_powers(shells[sa].angular_momentum());
    const size_t oa = basis.ao_offset_cart(static_cast<uint32_t>(sa));
    for (size_t sb = 0; sb < shells.size(); ++sb) {
      const auto powers_b = cartesian_powers(shells[sb].angular_momentum());
      const size_t ob = basis.ao_offset_cart(static_cast<uint32_t>(sb));
      const int32_t lab = shells[sa].angular_momentum() + shells[sb].angular_momentum();
      for (const Primitive &pa : shells[sa].primitives()) {
        for (const Primitive &pb : shells[sb].primitives()) {
          const double p = pa.exponent + pb.exponent;
          const double cc = pa.coefficient * pb.coefficient;
          const auto e = one_center_hermite(lab + extra_n, p);
          for (size_t i = 0; i < powers_a.size(); ++i) {
            for (size_t j = 0; j < powers_b.size(); ++j) {
              out[(oa + i) + (ncart * (ob + j))] +=
                  cc * integrand(powers_a[i], powers_b[j], p, pb.exponent, e);
            }
          }
        }
      }
    }
  }
  return out;
}

// The one-dimensional overlap of x^n exp(-p x^2): its Hermite t = 0 term.
double one_center_overlap_1d(const std::vector<std::vector<double>> &e, int32_t n, double p) {
  return n < 0 ? 0.0 : e[static_cast<size_t>(n)][0] * std::sqrt(std::numbers::pi / p);
}

/// @brief S over @p basis's Cartesian layout, nao_cart x nao_cart,
/// column-major. Every shell of @p basis is taken to sit on one center.
[[nodiscard]] std::vector<double> one_center_overlap(const BasisSet &basis) {
  return one_center_pair_loop(
      basis, 0, [](const auto &a, const auto &b, double p, double, const auto &e) {
        return one_center_overlap_1d(e, a[0] + b[0], p) * one_center_overlap_1d(e, a[1] + b[1], p) *
               one_center_overlap_1d(e, a[2] + b[2], p);
      });
}

/// @brief T = -1/2 <a|nabla^2|b>, as @ref one_center_overlap.
[[nodiscard]] std::vector<double> one_center_kinetic(const BasisSet &basis) {
  // +2 for the x^{n+2} the Laplacian makes.
  return one_center_pair_loop(
      basis, 2, [](const auto &a, const auto &b, double p, double beta, const auto &e) {
        std::array<double, 3> s{};
        std::array<double, 3> t{};
        for (size_t x = 0; x < 3; ++x) {
          const int32_t n = a[x] + b[x];
          s[x] = one_center_overlap_1d(e, n, p);
          // d^2/dx^2 on the ket's x^b exp(-beta x^2), then integrated.
          const auto bx = static_cast<double>(b[x]);
          t[x] = (bx * (bx - 1.0) * one_center_overlap_1d(e, n - 2, p)) -
                 (2.0 * beta * ((2.0 * bx) + 1.0) * one_center_overlap_1d(e, n, p)) +
                 (4.0 * beta * beta * one_center_overlap_1d(e, n + 2, p));
        }
        return -0.5 * ((t[0] * s[1] * s[2]) + (s[0] * t[1] * s[2]) + (s[0] * s[1] * t[2]));
      });
}

/// @brief V, the attraction to a nucleus of charge @p z at the shells' common
/// center, as @ref one_center_overlap.
[[nodiscard]] std::vector<double> one_center_nuclear(int32_t z, const BasisSet &basis) {
  // V = -z (2 pi / p) Sum_{tuv} E_t E_u E_v R_{tuv}(p), the nucleus at the
  // product's center.
  const double charge = static_cast<double>(z);
  return one_center_pair_loop(
      basis, 0, [charge](const auto &a, const auto &b, double p, double, const auto &e) {
        const int32_t nx = a[0] + b[0];
        const int32_t ny = a[1] + b[1];
        const int32_t nz = a[2] + b[2];
        double sum = 0.0;
        for (int32_t tx = nx % 2; tx <= nx; tx += 2) {
          for (int32_t ty = ny % 2; ty <= ny; ty += 2) {
            for (int32_t tz = nz % 2; tz <= nz; tz += 2) {
              sum += e[nx][tx] * e[ny][ty] * e[nz][tz] * one_center_hermite_coulomb(tx, ty, tz, p);
            }
          }
        }
        return -charge * 2.0 * std::numbers::pi / p * sum;
      });
}

} // namespace free_breakfast
