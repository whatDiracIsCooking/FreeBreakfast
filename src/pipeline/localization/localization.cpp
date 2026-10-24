/// @file
/// @brief LocalizationStep::run: the method's localizer, then the Molden file
/// if one was asked for.

module free_breakfast.pipeline.localization;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.atom; // element_symbol
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.io.molden;

#include "diagnostics.h"

namespace free_breakfast {

// One IBO for the log: its kind, then each center as its element, atom
// index, share of the orbital and angular character, as
// "bond  O0 61.2% (s 21% p 79%)  H1 38.6% (s 100%)". A share under half a
// percent is left out of the character.
[[nodiscard]] std::string describe_bond_orbital(const BondOrbital &orbital,
                                                const std::span<const Atom> atoms) {
  static constexpr std::array<std::string_view, 3> kKinds{"lone", "bond", "3c+ "};
  static constexpr std::array<char, 4> kMomenta{'s', 'p', 'd', 'f'};
  std::string text{kKinds[std::min(orbital.centers.size(), kKinds.size()) - 1]};
  for (const BondCenter &center : orbital.centers) {
    text += std::format("  {}{} {:.1f}% (", element_symbol(atoms[center.atom].element), center.atom,
                        100.0 * center.population);
    std::string_view gap;
    for (size_t l = 0; l < kMomenta.size(); ++l) {
      if (center.character[l] >= 0.005) {
        text += std::format("{}{} {:.0f}%", gap, kMomenta[l], 100.0 * center.character[l]);
        gap = " ";
      }
    }
    text += ')';
  }
  return text;
}

// The localized orbitals to @p path as Molden. They are not eigenvectors of F,
// so each is given its expectation value, <i|F|i> = sum_k U_ki^2 eps_k over the
// canonical occupied energies; U is all that needs downloading for it.
[[nodiscard]] Status write_localization_molden(DeviceResources &res,
                                               const std::filesystem::path &path,
                                               const Molecule &molecule, const BasisSet &basis,
                                               const LocalizationOutput &out,
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

[[nodiscard]] Result<LocalizationOutput>
LocalizationStep::run(View<MoleculeOutput, BasisSetOutput, OverlapOutput, ScfOutput> in,
                      DeviceResources &res) const {
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const ScfOutput &scf = in.get<ScfOutput>();
  const int32_t n_occ = molecule.electron_count() / 2;
  const size_t n_ao = basis.nao_sphe();
  const auto occ = static_cast<size_t>(n_occ);
  LocalizationOutput out{
      .c = Buffers::device<double>(n_ao * occ, res.device()),
      .u = Buffers::device<double>(occ * occ, res.device()),
      .n_occ = n_occ,
      .info = {},
  };
  // C is AO by MO with the occupied orbitals first, so its first n_occ
  // columns are the occupied block, packed.
  switch (mode) {
  case LocalizationMode::PipekMezey:
    out.info = FBF_TRY(pipek_mezey_localize(res, basis, n_occ, in.get<OverlapOutput>().s.data(),
                                            scf.c.data(), out.c.data(), out.u.data(), pipek_mezey));
    break;
  case LocalizationMode::Ibo: {
    const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
    const BasisSet minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
    const double *s = in.get<OverlapOutput>().s.data();
    Buffers::device<double> iao(n_ao * minao.nao_sphe(), res.device());
    out.info = FBF_TRY(ibo_localize(res, basis, minao, n_occ, s, scf.c.data(), out.c.data(),
                                    out.u.data(), iao.data(), pipek_mezey));
    // Over the IBOs, so each orbital's populations are its bonding picture;
    // the charges are the same over any rotation of the occupied orbitals.
    const std::vector<double> weights =
        FBF_TRY(orbital_weights(res, basis, minao, n_occ, s, out.c.data(), iao.data()));
    out.populations = FBF_TRY(orbital_populations(minao, n_occ, weights));
    out.charges = FBF_TRY(partial_charges(molecule.atoms(), out.populations));
    out.bonds = FBF_TRY(bond_orbitals(minao, n_occ, weights));
    for (size_t a = 0; a < out.charges.size(); ++a) {
      FBF_LOG_INFO("localization: IAO charge {:>3} {:<2} {:+.6f}", a,
                   element_symbol(molecule.atoms()[a].element), out.charges[a]);
    }
    for (size_t i = 0; i < out.bonds.size(); ++i) {
      FBF_LOG_INFO("localization: IBO {:>3} {}", i,
                   describe_bond_orbital(out.bonds[i], molecule.atoms()));
    }
    break;
  }
  }
  if (!molden.empty()) {
    FBF_TRY(write_localization_molden(res, molden, molecule, basis, out, scf.mo_energies));
  }
  return out;
}

} // namespace free_breakfast
