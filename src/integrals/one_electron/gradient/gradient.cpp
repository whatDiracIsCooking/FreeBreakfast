/// @file
/// @brief overlap_gradient, kinetic_gradient and potential_gradient: the
/// checks, the spherical matrix carried onto the Cartesian layout, each
/// Cartesian AO's atom, and the per-class launches into one device gradient;
/// the three derivative-matrix builds over the ip1 kernels; and
/// overlap_half_gradient, the overlap's ip1 matrices contracted row by row.

module;

// The kernel launchers, defined in gradient_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "gradient_bridge.h"

module free_breakfast.integrals.one_electron.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// Every check passed but launch_class's own: carry m onto the Cartesian
// layout, then launch_class(cls, ao_atom, atoms, m_cart, g) for every class,
// each on its own stream between a fork from and a join into res.stream(),
// into a zeroed [natoms, 3] device gradient, and download it. ao_atom is each
// Cartesian AO's atom; atoms is the class's own 2 n_rec of scratch.
Result<std::vector<double>> contract_gradient(DeviceResources &res, const BasisSet &basis,
                                              const std::span<const PrimitivePairClass> classes,
                                              const SphericalMatrixAO &m,
                                              const std::string_view what,
                                              const auto &launch_class) {
  if (res.dev_idx() != m.dev_idx()) {
    return err_config("{}: the matrix is on device {}, but the call passed device {}", what,
                      m.dev_idx(), res.dev_idx());
  }
  if (m.n() != basis.nao_sphe()) {
    return err_config("{}: the matrix is {} square, but the basis has {} spherical AOs", what,
                      m.n(), basis.nao_sphe());
  }
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("{}: the basis carries no per-atom shell map (make it with place)", what);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  FBF_TRY(check_one_electron_build(res, classes, pair.cartesian(), what));
  std::vector<double> g(static_cast<size_t>(n_atom) * 3);
  if (m.n() == 0) {
    return g;
  }
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(pair.spherical().data(), m.data(),
                                         sizeof(double) * m.num_elements(),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(pair.spherical_to_cartesian(res));

  // Each Cartesian AO's atom, from the basis's per-atom shell map.
  std::vector<int32_t> ao_atom(basis.nao_cart());
  for (uint32_t atom = 0; atom < n_atom; ++atom) {
    const uint32_t first = basis.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + basis.atom_shell_count(atom); ++shell) {
      const uint32_t off = basis.ao_offset_cart(shell);
      std::fill_n(ao_atom.begin() + off, basis.shells()[shell].function_count_cart(),
                  static_cast<int32_t>(atom));
    }
  }
  // Each class's slice of the record-atom scratch: classes run at once.
  std::vector<size_t> atoms_begin{0};
  for (const PrimitivePairClass &cls : classes) {
    atoms_begin.push_back(atoms_begin.back() + (2 * static_cast<size_t>(cls.n_rec)));
  }
  Buffers::device<int32_t> ao_atom_d(ao_atom.size(), res.device());
  Buffers::device<int32_t> atoms(std::max<size_t>(atoms_begin.back(), 1), res.device());
  Buffers::device<double> g_d(g.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ao_atom_d.data(), ao_atom.data(),
                                         sizeof(int32_t) * ao_atom.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(g_d.data(), 0, sizeof(double) * g.size(), stream)));
  const double *m_cart = pair.cartesian().data();
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return launch_class(cls, ao_atom_d.data(),
                        atoms.data() + atoms_begin[static_cast<size_t>(&cls - classes.data())],
                        m_cart, g_d.data());
  })));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), g_d.data(), sizeof(double) * g.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what keeps ao_atom, pageable, alive until its upload has read it.
  FBF_TRY(res.sync());
  return g;
}

Result<std::vector<double>> overlap_gradient(DeviceResources &res, const BasisSet &basis,
                                             const std::span<const PrimitivePairClass> classes,
                                             const SphericalMatrixAO &m) {
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const auto n_atom = static_cast<int32_t>(basis.natoms());
  return contract_gradient(res, basis, classes, m, "overlap_gradient",
                           [&](const PrimitivePairClass &cls, const int32_t *ao_atom,
                               int32_t *atoms, const double *mc, double *g) {
                             return device::overlap_gradient_class(cls.device_stream()->stream(),
                                                                   cls.view(), ao_atom, atoms, n_ao,
                                                                   n_atom, mc, g);
                           });
}

Result<std::vector<double>> kinetic_gradient(DeviceResources &res, const BasisSet &basis,
                                             const std::span<const PrimitivePairClass> classes,
                                             const SphericalMatrixAO &m) {
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const auto n_atom = static_cast<int32_t>(basis.natoms());
  return contract_gradient(res, basis, classes, m, "kinetic_gradient",
                           [&](const PrimitivePairClass &cls, const int32_t *ao_atom,
                               int32_t *atoms, const double *mc, double *g) {
                             return device::kinetic_gradient_class(cls.device_stream()->stream(),
                                                                   cls.view(), ao_atom, atoms, n_ao,
                                                                   n_atom, mc, g);
                           });
}

Result<std::vector<double>> potential_gradient(DeviceResources &res, const BasisSet &basis,
                                               const std::span<const PrimitivePairClass> classes,
                                               const Molecule &molecule,
                                               const SphericalMatrixAO &m) {
  FBF_TRY(check_nuclei(res, molecule, "potential_gradient"));
  if (molecule.natoms() != basis.natoms()) {
    return err_config("potential_gradient: the molecule has {} atoms, but the basis {}",
                      molecule.natoms(), basis.natoms());
  }
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const NucleiView nuclei = molecule.view();
  return contract_gradient(res, basis, classes, m, "potential_gradient",
                           [&](const PrimitivePairClass &cls, const int32_t *ao_atom,
                               int32_t *atoms, const double *mc, double *g) {
                             return device::potential_gradient_class(cls.device_stream()->stream(),
                                                                     cls.view(), ao_atom, atoms,
                                                                     nuclei, n_ao, mc, g);
                           });
}

// Each spherical AO's atom, from the basis's per-atom shell map.
std::vector<int32_t> spherical_ao_atoms(const BasisSet &basis) {
  std::vector<int32_t> ao_atom(basis.nao_sphe());
  for (uint32_t atom = 0; atom < basis.natoms(); ++atom) {
    const uint32_t first = basis.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + basis.atom_shell_count(atom); ++shell) {
      std::fill_n(ao_atom.begin() + basis.ao_offset_sphe(shell),
                  basis.shells()[shell].function_count_sphe(), static_cast<int32_t>(atom));
    }
  }
  return ao_atom;
}

// Zero the Cartesian [3, n_cart, n_cart] x1, then launch_class(cls, x1) for
// every class, each on its own stream between a fork from and a join into
// res.stream().
Status build_cartesian_slabs(DeviceResources &res,
                             const std::span<const PrimitivePairClass> classes,
                             Buffers::device<double> &x1, const auto &launch_class) {
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(
      gpu_status(wwr::wwrMemsetAsync(x1.data(), 0, sizeof(double) * x1.num_elements(), stream)));
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return launch_class(cls, x1.data());
  })));
  return {};
}

// x1's three Cartesian slabs, each carried onto the spherical layout at to, a
// [3, n, n], through pair's two matrices, on res.stream(). Copied in
// row-major, a slab is the transpose of the column-major matrix it fills, and
// U^T X^T U = (U^T X U)^T: read row-major, to is U^T X U.
Status carry_slabs_to_spherical(DeviceResources &res, MatrixAOPair &pair,
                                const Buffers::device<double> &x1, double *to) {
  const wwr::wwrStream_t stream = res.stream();
  const size_t slab_cart = pair.cartesian().num_elements();
  const size_t slab = pair.spherical().num_elements();
  for (size_t k = 0; k < 3; ++k) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(pair.cartesian().data(), x1.data() + (k * slab_cart),
                                           sizeof(double) * slab_cart, wwr::wwrMemcpyDeviceToDevice,
                                           stream)));
    FBF_TRY(pair.cartesian_to_spherical(res));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to + (k * slab), pair.spherical().data(),
                                           sizeof(double) * slab, wwr::wwrMemcpyDeviceToDevice,
                                           stream)));
  }
  return {};
}

// The derivative matrices' common body: check, launch launch_ip1(cls, x1) for
// every class into a zeroed Cartesian [3, n_cart, n_cart], carry each slab onto
// the spherical layout, and add every atom's centre terms from it into a zeroed
// [natoms, 3, n, n]. For V, n_rinv is the nucleus count, and before the centre
// terms each nucleus C's launch_rinv(cls, C, xc) matrices, carried onto the
// spherical layout, are its own slabs; for S and T n_rinv is 0.
Result<Buffers::device<double>>
derivative_matrices(DeviceResources &res, const BasisSet &basis,
                    const std::span<const PrimitivePairClass> classes, const std::string_view what,
                    const auto &launch_ip1, const uint32_t n_rinv, const auto &launch_rinv) {
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("{}: the basis carries no per-atom shell map (make it with place)", what);
  }
  // add_centre_derivatives runs one grid row per matrix.
  if (3 * static_cast<uint64_t>(n_atom) > std::numeric_limits<uint16_t>::max()) {
    return err_config("{}: {} atoms is too many for one launch's {} matrices", what, n_atom,
                      3 * static_cast<uint64_t>(n_atom));
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  FBF_TRY(check_one_electron_build(res, classes, pair.cartesian(), what));
  const uint32_t n_cart = basis.nao_cart();
  // The kernels index all three slabs as one 32-bit (k * n + m) * n + n'.
  if (3 * static_cast<uint64_t>(n_cart) * n_cart >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("{}: {} Cartesian AOs is too many for the kernels' 32-bit index", what,
                      n_cart);
  }
  const uint32_t n = basis.nao_sphe();
  const size_t slab_cart = static_cast<size_t>(n_cart) * n_cart;
  const size_t slab = static_cast<size_t>(n) * n;
  Buffers::device<double> out(std::max<size_t>(3 * n_atom * slab, 1), res.device());
  if (n == 0) {
    return out;
  }
  const wwr::wwrStream_t stream = res.stream();

  const std::vector<int32_t> ao_atom = spherical_ao_atoms(basis);
  Buffers::device<int32_t> ao_atom_d(ao_atom.size(), res.device());
  Buffers::device<double> x1(3 * slab_cart, res.device());
  Buffers::device<double> y(3 * slab, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(ao_atom_d.data(), ao_atom.data(),
                                         sizeof(int32_t) * ao_atom.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(
      gpu_status(wwr::wwrMemsetAsync(out.data(), 0, sizeof(double) * out.num_elements(), stream)));

  // V's operator terms first: nucleus C's [3, n, n] is out's slabs 3C..3C+2.
  for (uint32_t c = 0; c < n_rinv; ++c) {
    FBF_TRY(build_cartesian_slabs(res, classes, x1, [&](const PrimitivePairClass &cls, double *xc) {
      return launch_rinv(cls, static_cast<int32_t>(c), xc);
    }));
    FBF_TRY(carry_slabs_to_spherical(res, pair, x1, out.data() + (3 * c * slab)));
  }
  FBF_TRY(build_cartesian_slabs(res, classes, x1, launch_ip1));
  FBF_TRY(carry_slabs_to_spherical(res, pair, x1, y.data()));
  FBF_TRY(gpu_status(device::add_centre_derivatives(stream, static_cast<int32_t>(n_atom),
                                                    static_cast<int32_t>(n), ao_atom_d.data(),
                                                    y.data(), out.data())));
  // What keeps ao_atom, pageable, and the scratch alive until the stream has
  // read them.
  FBF_TRY(res.sync());
  return out;
}

Result<Buffers::device<double>>
overlap_derivative(DeviceResources &res, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes) {
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  return derivative_matrices(
      res, basis, classes, "overlap_derivative",
      [&](const PrimitivePairClass &cls, double *x1) {
        return device::overlap_ip1_class(cls.device_stream()->stream(), cls.view(), n_ao, x1);
      },
      0, [](const PrimitivePairClass &, int32_t, double *) { return wwr::wwrSuccess; });
}

Result<Buffers::device<double>>
kinetic_derivative(DeviceResources &res, const BasisSet &basis,
                   const std::span<const PrimitivePairClass> classes) {
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  return derivative_matrices(
      res, basis, classes, "kinetic_derivative",
      [&](const PrimitivePairClass &cls, double *x1) {
        return device::kinetic_ip1_class(cls.device_stream()->stream(), cls.view(), n_ao, x1);
      },
      0, [](const PrimitivePairClass &, int32_t, double *) { return wwr::wwrSuccess; });
}

Result<Buffers::device<double>>
potential_derivative(DeviceResources &res, const BasisSet &basis,
                     const std::span<const PrimitivePairClass> classes, const Molecule &molecule) {
  FBF_TRY(check_nuclei(res, molecule, "potential_derivative"));
  if (molecule.natoms() != basis.natoms()) {
    return err_config("potential_derivative: the molecule has {} atoms, but the basis {}",
                      molecule.natoms(), basis.natoms());
  }
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const NucleiView nuclei = molecule.view();
  return derivative_matrices(
      res, basis, classes, "potential_derivative",
      [&](const PrimitivePairClass &cls, double *x1) {
        return device::potential_ip1_class(cls.device_stream()->stream(), cls.view(), nuclei, n_ao,
                                           x1);
      },
      molecule.natoms(),
      [&](const PrimitivePairClass &cls, const int32_t nuc, double *xc) {
        return device::potential_rinv_class(cls.device_stream()->stream(), cls.view(), nuclei, nuc,
                                            n_ao, xc);
      });
}

Result<std::vector<double>> overlap_half_gradient(DeviceResources &res, const BasisSet &basis,
                                                  const std::span<const PrimitivePairClass> classes,
                                                  const SphericalMatrixAO &a) {
  constexpr std::string_view what = "overlap_half_gradient";
  if (res.dev_idx() != a.dev_idx()) {
    return err_config("{}: the matrix is on device {}, but the call passed device {}", what,
                      a.dev_idx(), res.dev_idx());
  }
  if (a.n() != basis.nao_sphe()) {
    return err_config("{}: the matrix is {} square, but the basis has {} spherical AOs", what,
                      a.n(), basis.nao_sphe());
  }
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("{}: the basis carries no per-atom shell map (make it with place)", what);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  FBF_TRY(check_one_electron_build(res, classes, pair.cartesian(), what));
  const uint32_t n_cart = basis.nao_cart();
  // The ip1 kernels index all three slabs as one 32-bit (k * n + m) * n + n'.
  if (3 * static_cast<uint64_t>(n_cart) * n_cart >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("{}: {} Cartesian AOs is too many for the kernels' 32-bit index", what,
                      n_cart);
  }
  std::vector<double> g(static_cast<size_t>(n_atom) * 3);
  const uint32_t n = basis.nao_sphe();
  if (n == 0) {
    return g;
  }
  const wwr::wwrStream_t stream = res.stream();
  Buffers::device<double> x1(3 * static_cast<size_t>(n_cart) * n_cart, res.device());
  Buffers::device<double> y(3 * static_cast<size_t>(n) * n, res.device());
  Buffers::device<double> rows_d(3 * static_cast<size_t>(n), res.device());
  const auto n_ao = static_cast<int32_t>(n_cart);
  FBF_TRY(build_cartesian_slabs(res, classes, x1, [&](const PrimitivePairClass &cls, double *x) {
    return device::overlap_ip1_class(cls.device_stream()->stream(), cls.view(), n_ao, x);
  }));
  FBF_TRY(carry_slabs_to_spherical(res, pair, x1, y.data()));
  FBF_TRY(gpu_status(device::first_index_rows(stream, static_cast<int32_t>(n), a.data(), y.data(),
                                              rows_d.data())));
  std::vector<double> rows(rows_d.num_elements());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(rows.data(), rows_d.data(), sizeof(double) * rows.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  // Each AO row's sum into its atom's, in AO order.
  const std::vector<int32_t> ao_atom = spherical_ao_atoms(basis);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t m = 0; m < n; ++m) {
      g[(3 * static_cast<size_t>(ao_atom[m])) + k] += rows[(k * n) + m];
    }
  }
  return g;
}

} // namespace free_breakfast
