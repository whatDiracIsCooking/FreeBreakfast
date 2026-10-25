/// @file
/// @brief LocalizationStep::run: the method's localizer, the LIVVOs if they
/// were asked for, then the Molden files.

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
import free_breakfast.localization.livvo;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast {

// One IBO for the log: sigma or pi, its kind by center count, then each
// center as its element, atom index, share of the orbital and angular
// character, as "sigma bond  O0 61.2% (s 21% p 79%)  H1 38.6% (s 100%)". A
// share under half a percent is left out of the character.
[[nodiscard]] std::string describe_bond_orbital(const BondOrbital &orbital,
                                                const std::span<const Atom> atoms) {
  static constexpr std::array<std::string_view, 3> kKinds{"lone", "bond", "3c+ "};
  static constexpr std::array<char, 4> kMomenta{'s', 'p', 'd', 'f'};
  std::string text = std::format("{:<5} {}", orbital.kind == BondKind::Pi ? "pi" : "sigma",
                                 kKinds[std::min(orbital.centers.size(), kKinds.size()) - 1]);
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

// The LIVVOs to @p path as Molden, at occupation 0. They are not eigenvectors
// of F either, so each is given <l|F|l> = sum_a O_al^2 eps_a over the
// canonical virtuals, O = C_vir^T S L; the LIVVOs lie wholly in the virtual
// space, so that is their whole expectation value.
[[nodiscard]] Status write_livvo_molden(DeviceResources &res, const std::filesystem::path &path,
                                        const Molecule &molecule, const BasisSet &basis,
                                        const ScfOutput &scf, const double *s,
                                        const LocalizationOutput &out) {
  const auto n_ao = static_cast<int32_t>(basis.nao_sphe());
  const auto n_vir = static_cast<int32_t>(scf.mo_energies.size()) - out.n_occ;
  const auto vvo = static_cast<size_t>(out.n_vvo);
  const auto vir = static_cast<size_t>(n_vir);
  std::vector<double> l(static_cast<size_t>(n_ao) * vvo);
  std::vector<double> o(vir * vvo);
  if (vvo > 0) {
    const double one = 1.0;
    const double zero = 0.0;
    const double *c_vir = scf.c.data() + (static_cast<size_t>(n_ao) * out.n_occ);
    const auto [sl, o_dev] = res.device_regions<double>(l.size(), o.size());
    FBF_TRY(
        gpu_status(gemm(res.blas(), n_ao, out.n_vvo, n_ao, &one, s, out.livvo.data(), &zero, sl)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), n_vir, out.n_vvo, n_ao, &one, c_vir, sl,
                                               &zero, o_dev)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(o.data(), o_dev, o.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(l.data(), out.livvo.data(), l.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
  }
  std::vector<double> energies(vvo, 0.0);
  for (size_t i = 0; i < vvo; ++i) {
    for (size_t a = 0; a < vir; ++a) {
      const double x = o[a + (vir * i)];
      energies[i] += x * x * scf.mo_energies[static_cast<size_t>(out.n_occ) + a];
    }
  }
  const std::vector<double> occupations(vvo, 0.0);
  return write_molden(path, molecule.atoms(), basis, l, energies, occupations);
}

[[nodiscard]] Result<LocalizationOutput>
LocalizationStep::run(View<MoleculeOutput, BasisSetOutput, OverlapOutput, ScfOutput> in,
                      DeviceResources &res) const {
  if (!livvo_molden.empty() && !livvo) {
    return err_config("localization: livvo_molden needs livvo");
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const ScfOutput &scf = in.get<ScfOutput>();
  const double *s = in.get<OverlapOutput>().s.data();
  const int32_t n_occ = molecule.electron_count() / 2;
  const size_t n_ao = basis.nao_sphe();
  const auto occ = static_cast<size_t>(n_occ);
  // MINAO and the IAOs over it, for IBO or the LIVVOs or both: built once.
  std::optional<BasisSet> minao;
  if (mode == LocalizationMode::Ibo || livvo) {
    const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
    minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
  }
  const size_t n_iao = minao ? minao->nao_sphe() : 0;
  // Sized for the buffer only: n_occ above MINAO's size is
  // valence_virtual_orbitals' InvalidConfig.
  const int32_t n_vvo = livvo ? std::max(static_cast<int32_t>(n_iao) - n_occ, 0) : 0;
  Buffers::device<double> iao(n_ao * n_iao, res.device());
  LocalizationOutput out{
      .c = Buffers::device<double>(n_ao * occ, res.device()),
      .u = Buffers::device<double>(occ * occ, res.device()),
      .n_occ = n_occ,
      .info = {},
      .populations = {},
      .charges = {},
      .bonds = {},
      .livvo = Buffers::device<double>(n_ao * static_cast<size_t>(n_vvo), res.device()),
      .n_vvo = n_vvo,
  };
  // C is AO by MO with the occupied orbitals first, so its first n_occ
  // columns are the occupied block, packed.
  switch (mode) {
  case LocalizationMode::PipekMezey:
    out.info = FBF_TRY(pipek_mezey_localize(res, basis, n_occ, s, scf.c.data(), out.c.data(),
                                            out.u.data(), pipek_mezey));
    if (livvo) {
      FBF_TRY(intrinsic_atomic_orbitals(res, basis, *minao, n_occ, s, scf.c.data(), iao.data()));
    }
    break;
  case LocalizationMode::Ibo: {
    out.info = FBF_TRY(ibo_localize(res, basis, *minao, n_occ, s, scf.c.data(), out.c.data(),
                                    out.u.data(), iao.data(), pipek_mezey));
    // Over the IBOs, so each orbital's populations are its bonding picture;
    // the charges are the same over any rotation of the occupied orbitals.
    const std::vector<double> weights =
        FBF_TRY(orbital_weights(res, basis, *minao, n_occ, s, out.c.data(), iao.data()));
    out.populations = FBF_TRY(orbital_populations(*minao, n_occ, weights));
    out.charges = FBF_TRY(partial_charges(molecule.atoms(), out.populations));
    out.bonds = FBF_TRY(bond_orbitals(*minao, n_occ, weights));
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
  if (livvo) {
    // The VVOs are a step-local buffer: the product is their localized form.
    Buffers::device<double> vvo(n_ao * static_cast<size_t>(n_vvo), res.device());
    FBF_TRY(valence_virtual_orbitals(res, basis, *minao, n_occ, s, scf.c.data(), iao.data(),
                                     vvo.data()));
    PipekMezeyOptions options = pipek_mezey;
    options.exponent = PipekMezeyExponent::Four;
    out.livvo_info = FBF_TRY(livvo_localize(res, basis, *minao, n_occ, s, iao.data(), vvo.data(),
                                            out.livvo.data(), nullptr, options));
    const std::vector<double> weights =
        FBF_TRY(orbital_weights(res, basis, *minao, n_vvo, s, out.livvo.data(), iao.data()));
    out.livvo_bonds = FBF_TRY(bond_orbitals(*minao, n_vvo, weights));
    const auto n_vir = static_cast<int32_t>(scf.mo_energies.size()) - n_occ;
    out.valence_character = FBF_TRY(valence_character(
        res, basis, n_vir, n_vvo, s, scf.c.data() + (n_ao * occ), out.livvo.data()));
    FBF_LOG_INFO("localization: {} LIVVOs in {} sweeps, L {:.6f} -> {:.6f}", n_vvo,
                 out.livvo_info.sweeps, out.livvo_info.initial_functional,
                 out.livvo_info.functional);
    for (size_t l = 0; l < out.livvo_bonds.size(); ++l) {
      FBF_LOG_INFO("localization: LIVVO {:>3} {}", l,
                   describe_bond_orbital(out.livvo_bonds[l], molecule.atoms()));
    }
    for (size_t a = 0; a < out.valence_character.size(); ++a) {
      FBF_LOG_INFO("localization: virtual {:>3} eps {:+.6f} t_val {:.4f}", occ + a,
                   scf.mo_energies[occ + a], out.valence_character[a]);
    }
  }
  if (!molden.empty()) {
    FBF_TRY(write_localization_molden(res, molden, molecule, basis, out, scf.mo_energies));
  }
  if (!livvo_molden.empty()) {
    FBF_TRY(write_livvo_molden(res, livvo_molden, molecule, basis, scf, s, out));
  }
  return out;
}

} // namespace free_breakfast
