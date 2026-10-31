/// @file
/// @brief The orthonormal basis of the rigid-body motions of a set of points,
/// shared by optimize's projection and harmonic_analysis's.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.geometry. See docs/style.md, "Module-private code".

module free_breakfast.geometry:rigid_body;

import std;
import free_breakfast.core.types;

namespace free_breakfast::geometry {

// The orthonormal basis of the rigid-body motions of the points x holds: the
// three translations and the three rotations about their centroid, by
// Gram-Schmidt. Point i's three components of every motion are multiplied by
// scale[i] (all 1 when scale is empty): sqrt(m_i) gives the motions in
// mass-weighted coordinates. The centroid stays unweighted either way: moving
// the rotation centre adds a translation to each rotation, and the
// translations are in the basis, so the span does not depend on it. A motion
// with nothing left after the ones before it (the rotation about a linear
// arrangement's axis, every rotation of one point) is dropped, so the basis
// spans exactly what moves the points rigidly.
std::vector<std::vector<double>> rigid_body_basis(std::span<const double> x,
                                                  std::span<const double> scale) {
  const size_t n = x.size();
  const size_t points = n / 3;
  std::array<double, 3> centroid{};
  for (size_t i = 0; i < points; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      centroid[k] += x[3 * i + k];
    }
  }
  for (double &c : centroid) {
    c /= static_cast<double>(points);
  }

  std::vector<std::vector<double>> motions(6, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < points; ++i) {
    const double s = scale.empty() ? 1.0 : scale[i];
    const std::array<double, 3> d{x[3 * i] - centroid[0], x[3 * i + 1] - centroid[1],
                                  x[3 * i + 2] - centroid[2]};
    for (size_t k = 0; k < 3; ++k) {
      motions[k][3 * i + k] = s;
      // e_k x d: the velocity of point i under a unit rotation about axis k.
      const size_t a = (k + 1) % 3;
      const size_t b = (k + 2) % 3;
      motions[3 + k][3 * i + a] = -s * d[b];
      motions[3 + k][3 * i + b] = s * d[a];
    }
  }

  std::vector<std::vector<double>> basis;
  for (std::vector<double> &v : motions) {
    const double before = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    for (const std::vector<double> &u : basis) {
      const double overlap = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
      for (size_t j = 0; j < n; ++j) {
        v[j] -= overlap * u[j];
      }
    }
    const double after = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    // Relative to the motion's own length, so the cut does not depend on the
    // units of x; 1e-8 is far below any real rotation and far above the
    // rounding left of one that is not there.
    if (after > 1e-8 * before) {
      for (double &value : v) {
        value /= after;
      }
      basis.push_back(std::move(v));
    }
  }
  return basis;
}

} // namespace free_breakfast::geometry
