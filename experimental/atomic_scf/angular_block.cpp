/// @file
/// @brief The l blocks: setup, averaging, diagonalization and the density
/// scatter.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.sad;
import :integrals;
import :gemm;
import :eigen;
import :angular_block;

namespace free_breakfast {

// Linear dependence within one l block: canonical orthogonalization drops the
// overlap eigenvectors below this, relative to the largest.
constexpr double kLinearDependence = 1e-10;

std::vector<double> AngularBlock::average(const std::vector<double> &full, size_t nao) const {
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

std::vector<double> AngularBlock::diagonalize(const std::vector<double> &fock) {
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

Result<std::vector<AngularBlock>> angular_blocks(int32_t z, std::span<const Shell> shells,
                                                 const AtomicIntegrals &ints) {
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

std::vector<double> angular_density(const std::vector<AngularBlock> &blocks, size_t nao) {
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
