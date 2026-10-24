/// @file
/// @brief LocalizationStep::run: the method's localizer, then the Molden file
/// if one was asked for.

module free_breakfast.pipeline.localization;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.io.molden;

#include "diagnostics.h"

namespace free_breakfast {

// The localized orbitals to @p path as Molden. They are not eigenvectors of F,
// so each is given its expectation value, <i|F|i> = sum_k U_ki^2 eps_k over the
// canonical occupied energies; U is all that needs downloading for it.
[[nodiscard]] Status write_localization_molden(DeviceResources &res,
                                               const std::filesystem::path &path,
                                               const Molecule &molecule, const BasisSet &basis,
                                               const Localization &out,
                                               const std::vector<double> &mo_energies) {
  const auto occ = static_cast<size_t>(out.n_occ);
  std::vector<double> c(basis.nao_sphe() * occ);
  std::vector<double> u(occ * occ);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data(), out.c.data(), c.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u.data(), out.u.data(), u.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> energies(occ, 0.0);
  for (size_t i = 0; i < occ; ++i) {
    for (size_t k = 0; k < occ; ++k) {
      energies[i] += u[k + (occ * i)] * u[k + (occ * i)] * mo_energies[k];
    }
  }
  const std::vector<double> occupations(occ, 2.0);
  return write_molden(path, molecule.atoms(), basis, c, energies, occupations);
}

[[nodiscard]] Result<Localization> LocalizationStep::run(View<Molecule, BasisSet, Overlap, Scf> in,
                                                         DeviceResources &res) const {
  const Molecule &molecule = in.get<Molecule>();
  const BasisSet &basis = in.get<BasisSet>();
  const Scf &scf = in.get<Scf>();
  const int32_t n_occ = molecule.electron_count() / 2;
  const size_t n_ao = basis.nao_sphe();
  const auto occ = static_cast<size_t>(n_occ);
  Localization out{
      .c = Buffers::device<double>(n_ao * occ, res.device()),
      .u = Buffers::device<double>(occ * occ, res.device()),
      .n_occ = n_occ,
      .info = {},
  };
  // C is AO by MO with the occupied orbitals first, so its first n_occ
  // columns are the occupied block, packed.
  switch (mode) {
  case LocalizationMode::PipekMezey:
    out.info = FBF_TRY(pipek_mezey_localize(res, basis, n_occ, in.get<Overlap>().s.data(),
                                            scf.c.data(), out.c.data(), out.u.data(), pipek_mezey));
    break;
  case LocalizationMode::Ibo: {
    const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
    const BasisSet minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
    out.info = FBF_TRY(ibo_localize(res, basis, minao, n_occ, in.get<Overlap>().s.data(),
                                    scf.c.data(), out.c.data(), out.u.data(), pipek_mezey));
    break;
  }
  }
  if (!molden.empty()) {
    FBF_TRY(write_localization_molden(res, molden, molecule, basis, out, scf.mo_energies));
  }
  return out;
}

} // namespace free_breakfast
