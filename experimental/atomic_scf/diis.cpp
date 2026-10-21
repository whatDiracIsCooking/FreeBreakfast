/// @file
/// @brief Pulay DIIS extrapolation.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import :diis;

namespace free_breakfast {

// Solve A x = b by Gaussian elimination with partial pivoting; nullopt if a
// pivot vanishes (the DIIS matrix goes singular as its residuals align).
std::optional<std::vector<double>> bordered_solve(std::vector<double> a, std::vector<double> b,
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

std::vector<double> AtomicDiis::extrapolate(std::vector<double> fock, std::vector<double> error) {
  fock_.push_back(std::move(fock));
  error_.push_back(std::move(error));
  if (fock_.size() > depth_) {
    fock_.pop_front();
    error_.pop_front();
  }
  while (true) {
    const size_t depth = fock_.size();
    const size_t dim = depth + 1;
    // [B -1; -1 0] [w; lambda] = [0; -1], B_ij = e_i . e_j.
    std::vector<double> b(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    for (size_t i = 0; i < depth; ++i) {
      for (size_t j = 0; j < depth; ++j) {
        b[i + (dim * j)] =
            std::inner_product(error_[i].begin(), error_[i].end(), error_[j].begin(), 0.0);
      }
      b[i + (dim * depth)] = -1.0;
      b[depth + (dim * i)] = -1.0;
    }
    rhs[depth] = -1.0;
    const std::optional<std::vector<double>> w = bordered_solve(b, rhs, dim);
    if (!w && depth > 1) {
      fock_.pop_front();
      error_.pop_front();
      continue;
    }
    if (!w) {
      return fock_.back();
    }
    std::vector<double> out(fock_.back().size(), 0.0);
    for (size_t i = 0; i < depth; ++i) {
      for (size_t x = 0; x < out.size(); ++x) {
        out[x] += (*w)[i] * fock_[i][x];
      }
    }
    return out;
  }
}

} // namespace free_breakfast
