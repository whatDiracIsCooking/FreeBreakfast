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
import free_breakfast.recipes.blas; // gemm, gmmwich
import free_breakfast.recipes.sort; // merge_sort, gather_columns

#include "diagnostics.h"

namespace free_breakfast {

// The localized orbitals in ascending order of their Fock expectation value,
// <i|F|i> = sum_k U_ki^2 eps_k, the diagonal of U^T diag(eps_occ) U, each
// space within its own columns: the doubly occupied, then the singly
// occupied. The sort is stable, so orbitals of equal energy keep the
// localizer's order. C and U are permuted together, so C_local = C_occ U
// still holds.
[[nodiscard]] Status sort_localized_orbitals(DeviceResources &res, const size_t n_ao,
                                             const ScfOutput &scf, LocalizationOutput &out) {
  const int32_t n = out.n_occ;
  if (n < 2) {
    return {};
  }
  const auto occ = static_cast<size_t>(n);
  // Not workspace: merge_sort carves its scratch from it.
  Buffers::device<double> keys(occ, res.device());
  Buffers::device<uint32_t> perm(occ, res.device());
  {
    const auto [eps, work, fock] = res.device_regions<double>(occ, occ * occ, occ * occ);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eps, scf.mo_energies.data(), occ * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(gpu_status(gmmwich(res.blas(), n, out.u.data(), eps, work, fock)));
    FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(keys.data(), sizeof(double), fock,
                                             (occ + 1) * sizeof(double), sizeof(double), occ,
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  // Each space's permutation counts from its own first column.
  const auto doubly = static_cast<size_t>(out.n_doubly);
  std::vector<uint32_t> identity(occ);
  std::iota(identity.begin(), identity.begin() + doubly, uint32_t{0});
  std::iota(identity.begin() + doubly, identity.end(), uint32_t{0});
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(perm.data(), identity.data(), occ * sizeof(uint32_t),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  Buffers::device<double> c(n_ao * occ, res.device());
  Buffers::device<double> u(occ * occ, res.device());
  for (const auto [first, count] :
       {std::pair{size_t{0}, out.n_doubly}, std::pair{doubly, n - out.n_doubly}}) {
    FBF_TRY(merge_sort(res, count, keys.data() + first, perm.data() + first));
    FBF_TRY(gather_columns(res, static_cast<int32_t>(n_ao), count, perm.data() + first,
                           out.c.data() + (n_ao * first), c.data() + (n_ao * first)));
    FBF_TRY(gather_columns(res, n, count, perm.data() + first, out.u.data() + (occ * first),
                           u.data() + (occ * first)));
  }
  // Pageable uploads above, and the unsorted buffers freed below: both wait
  // for the stream.
  FBF_TRY(res.sync());
  out.c = std::move(c);
  out.u = std::move(u);
  return {};
}

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

// The localized orbitals to @p path as Molden, the doubly occupied at
// occupation 2 and an open shell's singly occupied at 1, then the canonical
// virtuals:
// the localization mixes only occupied orbitals, so together they are still
// an orthonormal set spanning every MO. The localized ones are not
// eigenvectors of F, so each is given its expectation value,
// <i|F|i> = sum_k U_ki^2 eps_k over the canonical occupied energies; F is
// block-diagonal between occupied and virtual, so each virtual keeps eps_a.
// sort_localized_orbitals has already put the localized ones in that order.
[[nodiscard]] Status write_localization_molden(DeviceResources &res,
                                               const std::filesystem::path &path,
                                               const Molecule &molecule, const BasisSet &basis,
                                               const LocalizationOutput &out,
                                               const ScfOutput &scf) {
  const size_t n_ao = basis.nao_sphe();
  const auto occ = static_cast<size_t>(out.n_occ);
  const size_t n_mo = scf.mo_energies.size();
  std::vector<double> c(n_ao * n_mo);
  std::vector<double> u(occ * occ);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data(), out.c.data(), n_ao * occ * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data() + (n_ao * occ), scf.c.data() + (n_ao * occ),
                                         n_ao * (n_mo - occ) * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u.data(), out.u.data(), u.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> energies(scf.mo_energies);
  for (size_t i = 0; i < occ; ++i) {
    energies[i] = 0.0;
    for (size_t k = 0; k < occ; ++k) {
      energies[i] += u[k + (occ * i)] * u[k + (occ * i)] * scf.mo_energies[k];
    }
  }
  std::vector<double> occupations(n_mo, 0.0);
  std::fill_n(occupations.begin(), out.n_doubly, 2.0);
  std::fill_n(occupations.begin() + out.n_doubly, out.n_occ - out.n_doubly, 1.0);
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
  const int32_t n_occ = scf.n_alpha;
  const int32_t n_doubly = scf.n_beta;
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
  // localization::valence_virtual_orbitals' InvalidConfig.
  const int32_t n_vvo = livvo ? std::max(static_cast<int32_t>(n_iao) - n_occ, 0) : 0;
  Buffers::device<double> iao(n_ao * n_iao, res.device());
  LocalizationOutput out{
      .c = Buffers::device<double>(n_ao * occ, res.device()),
      .u = Buffers::device<double>(occ * occ, res.device()),
      .n_occ = n_occ,
      .n_doubly = n_doubly,
      .info = {},
      .populations = {},
      .charges = {},
      .bonds = {},
      .livvo = Buffers::device<double>(n_ao * static_cast<size_t>(n_vvo), res.device()),
      .n_vvo = n_vvo,
  };
  // The IAOs of the whole occupied space, n_alpha, which span each space's
  // orbitals and the LIVVOs' complement alike.
  if (minao) {
    FBF_TRY(localization::intrinsic_atomic_orbitals(res, basis, *minao, n_occ, s, scf.c.data(),
                                                    iao.data()));
  }
  // Each space localized on its own, from its columns of C into the same
  // columns of out.c and its diagonal block of U: the doubly occupied
  // orbitals, then an open shell's singly occupied ones. C is AO by MO with
  // the occupied orbitals first, doubly before singly, so each space is a
  // packed run of columns. A closed shell is the first space alone, whose U
  // is all of out.u.
  Buffers::device<double> block(n_doubly < n_occ ? occ * occ : 0, res.device());
  if (n_doubly < n_occ) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(out.u.data(), 0, occ * occ * sizeof(double), res.stream())));
  }
  for (const auto [first, count] : {std::pair{size_t{0}, n_doubly},
                                    std::pair{static_cast<size_t>(n_doubly), n_occ - n_doubly}}) {
    if (count == 0) {
      continue;
    }
    const double *c_space = scf.c.data() + (n_ao * first);
    double *c_local = out.c.data() + (n_ao * first);
    double *u = count == n_occ ? out.u.data() : block.data();
    const localization::Info info = FBF_TRY(
        mode == LocalizationMode::PipekMezey
            ? localization::pm_localize(res, basis, count, s, c_space, c_local, u, pipek_mezey)
            : localization::iao_localize(res, basis, *minao, count, s, iao.data(), c_space, c_local,
                                         u, pipek_mezey));
    out.info.sweeps += info.sweeps;
    out.info.initial_functional += info.initial_functional;
    out.info.functional += info.functional;
    if (u != out.u.data()) {
      const auto k = static_cast<size_t>(count);
      FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
          out.u.data() + first + (occ * first), occ * sizeof(double), u, k * sizeof(double),
          k * sizeof(double), k, wwr::wwrMemcpyDeviceToDevice, res.stream())));
    }
  }
  if (n_doubly < n_occ) {
    // The block is freed below, and its copies wait for the stream.
    FBF_TRY(res.sync());
  }
  FBF_TRY(sort_localized_orbitals(res, n_ao, scf, out));
  if (mode == LocalizationMode::Ibo) {
    // Over the IBOs, so each orbital's populations are its bonding picture;
    // the charges are the same over any rotation within each space.
    const std::vector<double> weights =
        FBF_TRY(orbital_weights(res, basis, *minao, n_occ, s, out.c.data(), iao.data()));
    out.populations = FBF_TRY(orbital_populations(*minao, n_occ, weights));
    // Eq 3 at each orbital's occupation: two for the doubly occupied, then
    // one for each singly occupied orbital's population.
    const size_t n_atoms = molecule.atoms().size();
    out.charges = FBF_TRY(
        partial_charges(molecule.atoms(),
                        std::span(out.populations).first(n_atoms * static_cast<size_t>(n_doubly))));
    for (size_t i = static_cast<size_t>(n_doubly); i < occ; ++i) {
      for (size_t a = 0; a < n_atoms; ++a) {
        out.charges[a] -= out.populations[a + (n_atoms * i)];
      }
    }
    out.bonds = FBF_TRY(bond_orbitals(*minao, n_occ, weights));
    for (size_t a = 0; a < out.charges.size(); ++a) {
      FBF_LOG_INFO("localization: IAO charge {:>3} {:<2} {:+.6f}", a,
                   element_symbol(molecule.atoms()[a].element), out.charges[a]);
    }
    for (size_t i = 0; i < out.bonds.size(); ++i) {
      FBF_LOG_INFO("localization: IBO {:>3} {}", i,
                   describe_bond_orbital(out.bonds[i], molecule.atoms()));
    }
  }
  if (livvo) {
    // The VVOs are a step-local buffer: the product is their localized form.
    Buffers::device<double> vvo(n_ao * static_cast<size_t>(n_vvo), res.device());
    FBF_TRY(localization::valence_virtual_orbitals(res, basis, *minao, n_occ, s, scf.c.data(),
                                                   iao.data(), vvo.data()));
    localization::PmOptions options = pipek_mezey;
    options.exponent = localization::PmExponent::Four;
    out.livvo_info = FBF_TRY(localization::livvo_localize(
        res, basis, *minao, n_occ, s, iao.data(), vvo.data(), out.livvo.data(), nullptr, options));
    const std::vector<double> weights =
        FBF_TRY(orbital_weights(res, basis, *minao, n_vvo, s, out.livvo.data(), iao.data()));
    out.livvo_bonds = FBF_TRY(bond_orbitals(*minao, n_vvo, weights));
    const auto n_vir = static_cast<int32_t>(scf.mo_energies.size()) - n_occ;
    out.valence_character = FBF_TRY(localization::valence_character(
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
    FBF_TRY(write_localization_molden(res, molden, molecule, basis, out, scf));
  }
  if (!livvo_molden.empty()) {
    FBF_TRY(write_livvo_molden(res, livvo_molden, molecule, basis, scf, s, out));
  }
  return out;
}

} // namespace free_breakfast
