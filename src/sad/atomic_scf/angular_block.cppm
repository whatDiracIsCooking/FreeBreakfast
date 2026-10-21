/// @file
/// @brief One momentum's radial block of the atomic SCF.

// An internal partition of free_breakfast.sad.
module free_breakfast.sad:angular_block;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :aufbau;
import :integrals;
import :gemm;
import :eigen;

namespace free_breakfast {

// Linear dependence within one l block: canonical orthogonalization drops the
// overlap eigenvectors below this, relative to the largest.
constexpr double kLinearDependence = 1e-10;

/// @brief One momentum's radial block: the atom's shells of that l, in order,
/// the orthonormal span they cover, and the subshells the aufbau fill seats on
/// its orbitals.
///
/// A spherical density makes the Fock matrix block-diagonal by (l, m) and the
/// same for every m, so each block's matrices are n x n over its shells -- the
/// m-average of the full spherical matrix -- not (n (2l+1))^2.
struct AngularBlock {
  int32_t l{0};                   ///< The momentum.
  std::vector<size_t> ao_first;   ///< Each shell's first spherical AO, its m = -l.
  size_t n{0};                    ///< Shells of this l.
  std::vector<double> overlap;    ///< n x n.
  std::vector<double> x;          ///< n x m canonical orthogonalizer, X^T S X = 1.
  size_t m{0};                    ///< Linearly independent orbitals.
  std::vector<double> occupation; ///< Per orbital, spin-summed, over all 2l+1 m.
  std::vector<double> density;    ///< n x n, per m: Sum_k occ_k / (2l+1) c_k c_k^T.

  /// @brief The m-averaged block of the full spherical @p nao x @p nao matrix.
  [[nodiscard]] std::vector<double> average(const std::vector<double> &full, size_t nao) const {
    const auto degen = static_cast<size_t>((2 * l) + 1);
    std::vector<double> out(n * n, 0.0);
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t mm = 0; mm < degen; ++mm) {
          sum += full[(ao_first[i] + mm) + (nao * (ao_first[j] + mm))];
        }
        out[i + (n * j)] = sum / static_cast<double>(degen);
      }
    }
    return out;
  }

  /// @brief Diagonalize @p fock (n x n) in the block's orthonormal span and
  /// refresh @ref density from the result.
  /// @return The orbital energies, ascending, m of them.
  std::vector<double> diagonalize(const std::vector<double> &fock) {
    const std::vector<double> fx = gemm_nn(fock, x, n, n, m);
    const std::vector<double> xfx = gemm_tn(x, fx, n, m, m);
    const SymmetricEigen eig = jacobi_eigen(xfx, m);
    const std::vector<double> c = gemm_nn(x, eig.vectors, n, m, m);
    const auto degen = static_cast<double>((2 * l) + 1);
    density.assign(n * n, 0.0);
    for (size_t k = 0; k < occupation.size(); ++k) {
      const double w = occupation[k] / degen;
      for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
          // c_i c_j grouped before the weight, so the density is exactly
          // symmetric.
          density[i + (n * j)] += w * (c[i + (n * k)] * c[j + (n * k)]);
        }
      }
    }
    return eig.values;
  }
};

/// @brief One block per l from 0 to the highest momentum in @p shells, each
/// with its overlap, its orthogonalizer and its share of @p z's aufbau fill.
/// Densities are left empty, for the caller's first @ref
/// AngularBlock::diagonalize.
///
/// @return ErrorKind::InvalidConfig if some l has more occupied subshells than
///   independent orbitals to seat them.
[[nodiscard]] Result<std::vector<AngularBlock>>
angular_blocks(int32_t z, std::span<const Shell> shells, const AtomicIntegrals &ints) {
  int32_t lmax = 0;
  for (const Shell &shell : shells) {
    lmax = std::max(lmax, shell.angular_momentum());
  }
  std::vector<AngularBlock> blocks(static_cast<size_t>(lmax) + 1);
  size_t ao = 0;
  for (const Shell &shell : shells) {
    blocks[static_cast<size_t>(shell.angular_momentum())].ao_first.push_back(ao);
    ao += shell.function_count_sphe();
  }

  for (size_t l = 0; l < blocks.size(); ++l) {
    AngularBlock &block = blocks[l];
    block.l = static_cast<int32_t>(l);
    block.n = block.ao_first.size();
    block.overlap = block.average(ints.overlap, ints.nao_sphe);
    if (block.n == 0) {
      continue;
    }
    const SymmetricEigen s = jacobi_eigen(block.overlap, block.n);
    const double largest = s.values.back();
    for (size_t k = 0; k < block.n; ++k) {
      if (s.values[k] <= kLinearDependence * largest) {
        continue;
      }
      const double inv = 1.0 / std::sqrt(s.values[k]);
      for (size_t i = 0; i < block.n; ++i) {
        block.x.push_back(s.vectors[i + (block.n * k)] * inv);
      }
      ++block.m;
    }
  }

  // The k-th subshell of momentum l on the k-th orbital of the l block.
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
  return blocks;
}

/// @brief The full spherical @p nao x @p nao density: each block's
/// @ref AngularBlock::density on every one of its m.
[[nodiscard]] std::vector<double> angular_density(const std::vector<AngularBlock> &blocks,
                                                  size_t nao) {
  std::vector<double> density(nao * nao, 0.0);
  for (const AngularBlock &block : blocks) {
    const auto degen = static_cast<size_t>((2 * block.l) + 1);
    for (size_t j = 0; j < block.n; ++j) {
      for (size_t i = 0; i < block.n; ++i) {
        for (size_t mm = 0; mm < degen; ++mm) {
          density[(block.ao_first[i] + mm) + (nao * (block.ao_first[j] + mm))] =
              block.density[i + (block.n * j)];
        }
      }
    }
  }
  return density;
}

} // namespace free_breakfast
