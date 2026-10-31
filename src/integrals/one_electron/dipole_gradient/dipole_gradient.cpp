/// @file
/// @brief dipole_gradient: the checks, the spherical matrix carried onto the
/// Cartesian layout, each Cartesian AO's atom, and the per-class launches into
/// one device gradient.

module;

// The kernel launcher, defined in dipole_gradient_launch.cu: declared in the
// GMF so it keeps external linkage (see the header).
#include "dipole_gradient_bridge.h"

module free_breakfast.integrals.one_electron.dipole_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Result<std::vector<double>> dipole_gradient(DeviceResources &res, const BasisSet &basis,
                                            const std::span<const PrimitivePairClass> classes,
                                            const SphericalMatrixAO &m) {
  constexpr std::string_view kWhat = "dipole_gradient";
  if (res.dev_idx() != m.dev_idx()) {
    return err_config("{}: the matrix is on device {}, but the call passed device {}", kWhat,
                      m.dev_idx(), res.dev_idx());
  }
  if (m.n() != basis.nao_sphe()) {
    return err_config("{}: the matrix is {} square, but the basis has {} spherical AOs", kWhat,
                      m.n(), basis.nao_sphe());
  }
  const uint32_t n_atom = basis.natoms();
  if (n_atom == 0) {
    return err_config("{}: the basis carries no per-atom shell map (make it with place)", kWhat);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  MatrixAOPair pair(res, basis, FBF_TRY(upload_spherical_transform_matrix(res, basis)));
  FBF_TRY(check_one_electron_build(res, classes, pair.cartesian(), kWhat));
  // [3, n_atom, 3]: component, atom, axis.
  std::vector<double> g(static_cast<size_t>(n_atom) * 9);
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
  const auto n_ao = static_cast<int32_t>(basis.nao_cart());
  const double *m_cart = pair.cartesian().data();
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return device::dipole_gradient_class(
        cls.device_stream()->stream(), cls.view(), ao_atom_d.data(),
        atoms.data() + atoms_begin[static_cast<size_t>(&cls - classes.data())], n_ao,
        static_cast<int32_t>(n_atom), m_cart, g_d.data());
  })));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(g.data(), g_d.data(), sizeof(double) * g.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what keeps ao_atom, pageable, alive until its upload has read it.
  FBF_TRY(res.sync());
  return g;
}

} // namespace free_breakfast
