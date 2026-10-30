/// @file
/// @brief overlap_gradient, kinetic_gradient and potential_gradient: the
/// checks, the spherical matrix carried onto the Cartesian layout, each
/// Cartesian AO's atom, and the per-class launches into one device gradient.

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

} // namespace free_breakfast
