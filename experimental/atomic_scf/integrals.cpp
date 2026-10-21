/// @file
/// @brief atomic_integrals: check the shells, build the Cartesian integrals,
/// carry them to the spherical layout.

module free_breakfast.experimental.atomic_scf;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :integrals;
import :one_electron;
import :eri;

namespace free_breakfast {

// The n^4 two-electron tensor is held whole, over the Cartesian layout first:
// 96^4 doubles is 680 MB. An atom in a triple-zeta basis is well inside it.
constexpr uint32_t kMaxAtomicAoCart = 96;

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

  const std::vector<double> u = spherical_transform_matrix(basis);
  return AtomicIntegrals{
      .nao_sphe = static_cast<uint32_t>(nsph),
      .overlap = to_spherical(one_center_overlap(basis), u, ncart, nsph, 2),
      .kinetic = to_spherical(one_center_kinetic(basis), u, ncart, nsph, 2),
      .nuclear = to_spherical(one_center_nuclear(z, basis), u, ncart, nsph, 2),
      .eri = to_spherical(one_center_eri(basis), u, ncart, nsph, 4),
  };
}

} // namespace free_breakfast
