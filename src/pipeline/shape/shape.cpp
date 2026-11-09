/// @file
/// @brief plan_shape: the molecule read, the bases placed, and the counts
/// taken from them.

module free_breakfast.pipeline.shape;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.io.basis;
import free_breakfast.recipes.diagonalize; // syevd_lwork
import free_breakfast.recipes.expm;        // skew_eig_lwork
import free_breakfast.pipeline.step;       // View

#include "diagnostics.h"

namespace free_breakfast {

// The geometry the plan's molecule is read from: the Molecule step's, or, when
// an [optimize] loads it, the start the optimization's chain reads. The loaded
// final.xyz does not exist before the run, and must hold the same atoms.
[[nodiscard]] Result<std::optional<MoleculeOutput>> shape_molecule(const Calculation &calc,
                                                                   DeviceResources &res) {
  if (const std::optional<MoleculeStep> &step = calc.step<MoleculeStep>()) {
    return std::optional{FBF_TRY(step->run(View<>{}, res))};
  }
  if (const std::optional<OptimizeStep> &optimize = calc.step<OptimizeStep>()) {
    return std::optional{FBF_TRY(optimize->molecule.run(View<>{}, res))};
  }
  return std::optional<MoleculeOutput>{};
}

// The step of type S the plan configures: its own, else the copy an
// [optimize]'s or a [hessian]'s chain holds (member pointer @p member), which
// the config reads from the same table. Null if none does.
template <class S>
[[nodiscard]] const S *shape_step(const Calculation &calc, S OptimizeStep::*in_optimize,
                                  S HessianStep::*in_hessian) {
  if (const std::optional<S> &step = calc.step<S>()) {
    return &*step;
  }
  if (const std::optional<OptimizeStep> &optimize = calc.step<OptimizeStep>()) {
    return &((*optimize).*in_optimize);
  }
  if (const std::optional<HessianStep> &hessian = calc.step<HessianStep>()) {
    return &((*hessian).*in_hessian);
  }
  return nullptr;
}

// The aux basis an [eri] places: its aux_basis, or empty, the Eri step's
// default.
[[nodiscard]] Result<BasisSet> shape_aux_basis(const EriStep &eri, const BasisSet &basis,
                                               const Molecule &molecule) {
  const std::string spec =
      eri.aux_basis.empty() ? FBF_TRY(default_aux_basis(basis)) : eri.aux_basis;
  const BasisStamp stamp = FBF_TRY(resolve_aux_basis(spec));
  return FBF_TRY(place(stamp.primitive_normalized(), molecule));
}

// How many of MINAO's spherical AOs over @p molecule the AVAS @p labels name,
// by the active-space selection's own rule.
[[nodiscard]] Result<size_t> avas_labelled_ao_count(const Molecule &molecule,
                                                    std::span<const std::string> labels) {
  const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
  const BasisSet minao = FBF_TRY(place(stamp, molecule));
  const std::vector<bool> chosen = FBF_TRY(avas_minao_aos(molecule.atoms(), minao, labels));
  return static_cast<size_t>(std::ranges::count(chosen, true));
}

// The largest order a planned @p selection's CI diagonalizes at, past its
// active space: its Davidson's Rayleigh-Ritz problem, 16 nroots columns at
// most (ci::solve's subspace), and for a CASSCF the augmented Hessian's
// projected problem, ah_max_subspace + 1.
[[nodiscard]] size_t ci_eigensolve_order(const int32_t nroots, const bool casscf) {
  size_t order = 16 * static_cast<size_t>(std::max(nroots, 1));
  if (casscf) {
    order = std::max(order, static_cast<size_t>(mcscf::Options{}.ah_max_subspace) + 1);
  }
  return order;
}

// The active space @p selection may take, at its largest, over @p nao
// orbitals and @p n_electrons: see plan_shape. {0, 0} for none (a default
// selection, as an unread chain copy holds).
[[nodiscard]] Result<std::pair<size_t, size_t>>
largest_active_space(const ActiveSpaceSelection &selection, const Molecule &molecule, size_t nao,
                     size_t n_electrons) {
  if (selection.avas.empty()) {
    return std::pair{static_cast<size_t>(std::max(selection.ncas, 0)),
                     static_cast<size_t>(std::max(selection.nelecas, 0))};
  }
  size_t ncas = 2 * FBF_TRY(avas_labelled_ao_count(molecule, selection.avas));
  if (selection.avas_max_ncas > 0) {
    ncas = std::min(ncas, static_cast<size_t>(selection.avas_max_ncas));
  }
  ncas = std::min(ncas, nao);
  return std::pair{ncas, std::min(ncas, n_electrons)};
}

Result<PlanShape> plan_shape(const Calculation &calc, DeviceResources &res) {
  PlanShape shape;
  const std::optional<MoleculeOutput> read = FBF_TRY(shape_molecule(calc, res));
  if (!read) {
    return shape;
  }
  const Molecule &molecule = read->molecule;
  FootprintDimensions &dims = shape.dims;
  dims.natoms = molecule.natoms();
  dims.atoms = molecule.atoms();

  // n_alpha - n_beta = 2S = multiplicity - 1, and N has its parity: the Scf's
  // split.
  const int32_t electrons = molecule.electron_count();
  const int32_t n_open = molecule.multiplicity() - 1;
  if (n_open < 0 || electrons < n_open || (electrons - n_open) % 2 != 0) {
    return err_config("plan shape: {} electrons cannot be at multiplicity {}", electrons,
                      molecule.multiplicity());
  }
  const auto n_electrons = static_cast<size_t>(electrons);
  dims.n_alpha = static_cast<size_t>((electrons + n_open) / 2);
  dims.n_beta = static_cast<size_t>((electrons - n_open) / 2);
  dims.nocc = dims.n_beta;

  const BasisSetStep *basis_step =
      shape_step(calc, &OptimizeStep::basis_set, &HessianStep::basis_set);
  if (basis_step == nullptr) {
    return shape;
  }
  shape.basis =
      std::make_unique<const BasisSet>(FBF_TRY(basis_step->run(View<MoleculeOutput>{*read})).basis);
  const BasisSet &basis = *shape.basis;
  dims.basis = shape.basis.get();
  dims.nao = basis.nao_sphe();

  if (const EriStep *eri = shape_step(calc, &OptimizeStep::eri, &HessianStep::eri);
      eri != nullptr && eri->mode == EriMode::DensityFitting) {
    shape.aux = std::make_unique<const BasisSet>(FBF_TRY(shape_aux_basis(*eri, basis, molecule)));
    dims.aux = shape.aux.get();
    dims.naux = shape.aux->nao_sphe();
  }

  // Every i <= j shell pair's primitive pairs, in its la <= lb class.
  const std::vector<Shell> &shells = basis.shells();
  int32_t l_max = -1;
  for (const Shell &shell : shells) {
    l_max = std::max(l_max, shell.angular_momentum());
  }
  const auto nl = static_cast<size_t>(l_max + 1);
  dims.mmd_records.assign(nl, std::vector<size_t>(nl, 0));
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      const int32_t li = shells[i].angular_momentum();
      const int32_t lj = shells[j].angular_momentum();
      dims.mmd_records[static_cast<size_t>(std::min(li, lj))]
                      [static_cast<size_t>(std::max(li, lj))] +=
          size_t{shells[i].nprimitives()} * shells[j].nprimitives();
    }
  }

  // The largest active space: of most orbitals, then of nelecas nearest half
  // filling (ncas electrons), which spans the most determinants. And the
  // largest order any planned step diagonalizes at: nao, or a CI's subspace.
  std::vector<const ActiveSpaceSelection *> selections;
  size_t max_order = dims.nao;
  bool casscf_planned = false;
  if (const std::optional<CasciStep> &casci = calc.step<CasciStep>()) {
    selections.push_back(&*casci);
    max_order = std::max(max_order, ci_eigensolve_order(casci->nroots, false));
  }
  if (const std::optional<CasscfStep> &casscf = calc.step<CasscfStep>()) {
    selections.push_back(&*casscf);
    max_order = std::max(max_order, ci_eigensolve_order(casscf->nroots, true));
    casscf_planned = true;
  }
  if (const std::optional<OptimizeStep> &optimize = calc.step<OptimizeStep>();
      optimize && (optimize->meci || optimize->casscf_gradient || optimize->nevpt2_gradient)) {
    selections.push_back(&optimize->casscf);
    max_order = std::max(max_order, ci_eigensolve_order(optimize->casscf.nroots, true));
    casscf_planned = true;
  }
  if (const std::optional<HessianStep> &hessian = calc.step<HessianStep>();
      hessian && (hessian->casscf_gradient || hessian->nevpt2_gradient)) {
    selections.push_back(&hessian->casscf);
    max_order = std::max(max_order, ci_eigensolve_order(hessian->casscf.nroots, true));
    casscf_planned = true;
  }
  if (const std::optional<FciStep> &fci = calc.step<FciStep>()) {
    max_order = std::max(max_order, ci_eigensolve_order(fci->nroots, false));
  }

  // MINAO, placed as AVAS places it, when a selection is AVAS's.
  if (std::ranges::any_of(selections, [](const ActiveSpaceSelection *selection) {
        return !selection->avas.empty();
      })) {
    const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
    shape.minao =
        std::make_unique<const BasisSet>(FBF_TRY(place(stamp.primitive_normalized(), molecule)));
    dims.minao = shape.minao.get();
    // AVAS diagonalizes the labelled MINAO AOs' overlap.
    max_order = std::max(max_order, static_cast<size_t>(shape.minao->nao_sphe()));
  }

  // diagonalize's own query, at every order through the largest.
  dims.syevd_lwork.resize(max_order + 1);
  for (size_t order = 0; order <= max_order; ++order) {
    dims.syevd_lwork[order] = FBF_TRY(syevd_lwork<double>(res, static_cast<int32_t>(order)));
  }
  // A CASSCF's orbital step diagonalizes kappa through skew_eig at nmo = nao.
  if (casscf_planned) {
    dims.skew_eig_lwork = FBF_TRY(skew_eig_lwork<double>(res, static_cast<int32_t>(dims.nao)));
  }

  std::vector<std::pair<size_t, size_t>> spaces;
  for (const ActiveSpaceSelection *selection : selections) {
    spaces.push_back(FBF_TRY(largest_active_space(*selection, molecule, dims.nao, n_electrons)));
  }
  if (calc.step<FciStep>()) {
    spaces.emplace_back(dims.nao, n_electrons);
  }
  for (const auto [ncas, nelecas] : spaces) {
    const auto from_half = [](size_t o, size_t e) { return o > e ? o - e : e - o; };
    if (ncas > dims.ncas ||
        (ncas == dims.ncas && from_half(ncas, nelecas) < from_half(dims.ncas, dims.nelecas))) {
      dims.ncas = ncas;
      dims.nelecas = nelecas;
    }
  }
  return shape;
}

} // namespace free_breakfast
