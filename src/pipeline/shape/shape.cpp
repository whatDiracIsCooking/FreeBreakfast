/// @file
/// @brief plan_shape: the molecule read, the bases placed, and the counts
/// taken from them; and preflight, the plan held to the [memory] caps.

module free_breakfast.pipeline.shape;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.core.wwr; // Buffers, ScopedDeviceIndex, PoolUsage
import free_breakfast.core.log;
import free_breakfast.core.primitives; // math::idivup
import free_breakfast.io.basis;
import free_breakfast.recipes.diagonalize; // syevd_lwork
import free_breakfast.recipes.expm;        // skew_eig_lwork
import free_breakfast.recipes.lu;          // getrf_lwork
import free_breakfast.response.zvector;    // rotating_pairs
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

// The largest order a planned @p cis diagonalizes at over @p nocc of @p nao
// orbitals: its Davidson's Rayleigh-Ritz problem, at every subspace size up to
// the step's clamped subspace, max(2 n_states, min(max_subspace, n_ov)). The
// clustered preconditioner's blocks are of at most the guess's columns,
// min(n_ov, 2 n_states), which that never falls below.
[[nodiscard]] size_t cis_eigensolve_order(const CisStep &cis, const size_t nocc, const size_t nao) {
  const size_t n_ov = nocc * (nao > nocc ? nao - nocc : 0);
  const auto n_states = static_cast<size_t>(std::max(cis.options.n_states, 1));
  return std::max(2 * n_states, std::min(static_cast<size_t>(std::max(cis.max_subspace, 0)), n_ov));
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
    // What the steps reading its product are sized over.
    dims.casscf_nroots = static_cast<size_t>(std::max(casscf->nroots, 1));
    dims.casscf_unequal_weights =
        std::ranges::contains(response::rotating_pairs(casscf->weights), true);
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
  const std::optional<CisStep> &cis_step = calc.step<CisStep>();
  if (cis_step) {
    max_order = std::max(max_order, cis_eigensolve_order(*cis_step, dims.nocc, dims.nao));
  }

  // MINAO, placed as AVAS and the Localization step place it, when a
  // selection is AVAS's or a localization builds IAOs.
  const std::optional<LocalizationStep> &localization = calc.step<LocalizationStep>();
  const bool localization_iaos =
      localization && (localization->mode == LocalizationMode::Ibo || localization->livvo);
  if (localization_iaos ||
      std::ranges::any_of(selections, [](const ActiveSpaceSelection *selection) {
        return !selection->avas.empty();
      })) {
    const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
    shape.minao =
        std::make_unique<const BasisSet>(FBF_TRY(place(stamp.primitive_normalized(), molecule)));
    dims.minao = shape.minao.get();
    // AVAS diagonalizes the labelled MINAO AOs' overlap; the IAOs
    // orthonormalize at MINAO's order, and the LIVVOs diagonalize there.
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
  // A CIS step's RIS preconditioner solves its capacitance matrix by LU, at
  // the RIS basis's order.
  if (cis_step && cis_step->options.ris) {
    const BasisSet ris = FBF_TRY(
        cis::ris_basis(dims.atoms, cis_step->options.ris->theta, cis_step->options.ris->shells));
    dims.getrf_lwork = FBF_TRY(getrf_lwork(res, static_cast<int32_t>(ris.nao_sphe())));
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

// The bound on ROCm's reuse rounding, as a divisor of the request: a request
// is served whole from a freed block of up to 9/8 its size, and counted as
// the block (core/wwr's README, "Reuse rounding (HIP)").
constexpr size_t kReuseRoundingDivisor = 8;

// Whether @p res's pool counts a reused block whole rather than the request:
// a 1024-byte block freed, then 1000 bytes asked for. ROCm's pool does, CUDA's
// does not. The probe's block stays cached in the pool.
[[nodiscard]] Result<bool> pool_rounds_reused_blocks(DeviceResources &res) {
  {
    const Buffers::device<std::byte> block(1024, res.device());
  }
  FBF_TRY(res.sync());
  const uint64_t before = res.device()->pool_usage().used_current;
  const Buffers::device<std::byte> probe(1000, res.device());
  FBF_TRY(res.sync());
  return res.device()->pool_usage().used_current - before > 1000;
}

// The most a plan's device peak may reach in a pool capped at @p pool_cap:
// the cap less the fragmentation headroom, and on a pool that rounds a reused
// block, less the rounding of every allocation live at the peak. Those
// allocations' requests sum to at most the peak P, each is counted at most
// an eighth past its request, so the largest P with P + ceil(P / 8) within
// what the headroom leaves. Both terms are left: the rounding fills the
// blocks it reuses, and the headroom covers the chunks' free parts beside
// them.
[[nodiscard]] size_t preflight_device_room(const uint64_t pool_cap, const bool rounds_reuse) {
  const auto limit = static_cast<size_t>(pool_cap - kPoolHeadroomBytes);
  if (!rounds_reuse) {
    return limit;
  }
  const auto rounded = [](size_t peak) { return peak + math::idivup(peak, kReuseRoundingDivisor); };
  size_t peak = limit - (limit / (kReuseRoundingDivisor + 1));
  while (peak > 0 && rounded(peak) > limit) {
    --peak;
  }
  while (rounded(peak + 1) <= limit) {
    ++peak;
  }
  return peak;
}

// The node's MemAvailable, in bytes, from /proc/meminfo.
[[nodiscard]] Result<size_t> read_mem_available() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo) {
    return err_io("pre-flight: cannot open '/proc/meminfo'");
  }
  constexpr std::string_view kKey = "MemAvailable:";
  for (std::string line; std::getline(meminfo, line);) {
    if (line.starts_with(kKey)) {
      std::istringstream fields(line.substr(kKey.size()));
      size_t kib = 0;
      std::string unit;
      if (!(fields >> kib >> unit) || unit != "kB") {
        return err_io("pre-flight: cannot parse '{}' in '/proc/meminfo'", line);
      }
      return kib * 1024;
    }
  }
  return err_io("pre-flight: no MemAvailable in '/proc/meminfo'");
}

// The planned steps' active-space selections, when every active space the
// plan holds is AVAS's: the top-level CASCI and CASSCF, and no FCI (every
// orbital) and no space by index, which the shared worst case also sizes.
// Empty when there is none, or any other. A chain's copies need no look: the
// steps holding them ([optimize], [hessian]) have no footprint yet.
[[nodiscard]] std::vector<ActiveSpaceSelection *> avas_only_selections(Calculation &calc) {
  std::vector<ActiveSpaceSelection *> selections;
  if (std::optional<CasciStep> &casci = calc.step<CasciStep>()) {
    selections.push_back(&*casci);
  }
  if (std::optional<CasscfStep> &casscf = calc.step<CasscfStep>()) {
    selections.push_back(&*casscf);
  }
  const bool all_avas = std::ranges::all_of(
      selections, [](const ActiveSpaceSelection *selection) { return !selection->avas.empty(); });
  if (!all_avas || calc.step<FciStep>()) {
    selections.clear();
  }
  return selections;
}

// Whether @p steps, every one footprinted, fit @p caps: check_plan's test,
// without its log lines.
[[nodiscard]] bool plan_fits(std::span<const StepFootprint> steps, const PlanCaps &caps) {
  std::vector<MemoryFootprint> footprints;
  for (const StepFootprint &step : steps) {
    if (!step.footprint) {
      return false;
    }
    footprints.push_back(*step.footprint);
  }
  const MemoryFootprint peak = plan_footprint(footprints).footprint;
  return (caps.device_room == 0 || peak.device.peak() <= caps.device_room) &&
         (caps.max_pinned_mem_per_gpu == 0 || peak.pinned.peak() <= caps.max_pinned_mem_per_gpu);
}

// Under AVAS, narrow the plan to the most active orbitals that fit @p caps,
// when its worst case does not: each AVAS step's avas_max_ncas is set to it,
// so a pick of more is refused at the step (#1042's check), and @p dims
// shaped at it. The fold grows with the active space, so the largest fitting
// one is found by bisection. Returns the orbitals the plan now fits for any
// pick up to, 0 when not even one does (@p dims unchanged), and none when
// there is nothing to narrow: no AVAS-only plan, or a step with no footprint,
// which check_plan refuses on its own.
[[nodiscard]] std::optional<size_t> narrow_avas(Calculation &calc, FootprintDimensions &dims,
                                                const PlanCaps &caps) {
  const std::vector<ActiveSpaceSelection *> selections = avas_only_selections(calc);
  if (selections.empty() || dims.ncas == 0 ||
      !std::ranges::all_of(calc.footprints(dims),
                           [](const StepFootprint &step) { return step.footprint.has_value(); })) {
    return std::nullopt;
  }
  const size_t worst = dims.ncas;
  const size_t electrons = dims.n_alpha + dims.n_beta;
  const auto at = [&](size_t ncas) {
    FootprintDimensions narrowed = dims;
    narrowed.ncas = ncas;
    narrowed.nelecas = std::min(ncas, electrons);
    return narrowed;
  };
  if (plan_fits(calc.footprints(dims), caps)) {
    log::info(std::format("pre-flight: fits for any AVAS pick up to {} orbitals", worst));
    return worst;
  }
  size_t fits = 0;
  size_t past = worst;
  while (past - fits > 1) {
    const size_t mid = fits + ((past - fits) / 2);
    (plan_fits(calc.footprints(at(mid)), caps) ? fits : past) = mid;
  }
  if (fits == 0) {
    return 0;
  }
  for (ActiveSpaceSelection *selection : selections) {
    selection->avas_max_ncas = static_cast<int32_t>(fits);
  }
  dims = at(fits);
  log::warn(std::format("pre-flight: AVAS's worst case, {} orbitals, does not fit under the "
                        "[memory] caps; fits for any AVAS pick up to {} orbitals, so a pick of "
                        "more is refused at the step (avas_max_ncas is set to {})",
                        worst, fits, fits));
  return fits;
}

Status preflight(Calculation &calc, NodeResources &node) {
  const size_t device_cap = calc.max_device_mem_per_gpu;
  const size_t pinned_cap = calc.max_pinned_mem_per_gpu;
  if (device_cap == 0 && pinned_cap == 0) {
    return {};
  }
  constexpr uint64_t kReserve = DeviceResources::kDeviceReserve;
  if (device_cap != 0 && device_cap <= kReserve + kPoolHeadroomBytes) {
    return err_config("pre-flight: max_device_mem_per_gpu = {} bytes leaves no pool: {} bytes of "
                      "it are the reserve outside the pool, and {} the fragmentation headroom",
                      device_cap, kReserve, kPoolHeadroomBytes);
  }
  if (pinned_cap != 0) {
    FBF_TRY(check_pinned_node(pinned_cap, node.device_count(), FBF_TRY(read_mem_available())));
  }
  if (node.device_count() == 0) {
    return {};
  }

  // The caps are set when a DeviceResources is built, so the node's are
  // rebuilt with them: the pool's maxSize, and its release threshold, are
  // the device cap less the reserve outside the pool.
  const uint64_t pool_cap = device_cap != 0 ? device_cap - kReserve : 0;
  for (DeviceResources &res : node.devices()) {
    const ScopedDeviceIndex on{res.dev_idx()};
    res = DeviceResources{res.dev_idx(), DeviceResources::kDefaultPoolReleaseThreshold, pinned_cap,
                          pool_cap};
  }

  DeviceResources &res = node.device(0);
  const ScopedDeviceIndex on{res.dev_idx()};
  PlanShape shape = FBF_TRY(plan_shape(calc, res));
  const bool rounds_reuse = device_cap != 0 && FBF_TRY(pool_rounds_reused_blocks(res));
  const PlanCaps caps{.max_device_mem_per_gpu = device_cap,
                      .device_room =
                          device_cap != 0 ? preflight_device_room(pool_cap, rounds_reuse) : 0,
                      .max_pinned_mem_per_gpu = pinned_cap};
  const std::optional<size_t> avas_fits = narrow_avas(calc, shape.dims, caps);
  auto plan = check_plan(calc.footprints(shape.dims), caps);
  if (!plan) {
    if (avas_fits == 0U) {
      plan.error().message += "; no AVAS pick fits, not even of 1 orbital: raise the cap";
    }
    return std::unexpected(std::move(plan).error());
  }

  // Grown once to the plan's maximum, so no footprinted step grows them and
  // the fold needs no old-plus-new term for a growth (#1123 sizes them at
  // construction instead).
  (void)res.device_workspace(plan->footprint.device.workspace);
  FBF_TRY(res.pinned_workspace("pre-flight", plan->footprint.pinned.workspace));

  // The pre-warm: what the plan may use of the pool, taken once and freed.
  // The cap is the pool's release threshold, so the pool keeps it, and
  // memory another process holds is reported here rather than at a step.
  if (device_cap != 0) {
    const PoolUsage usage = res.device()->pool_usage();
    // min(pool_cap, free + reserved) - used: below pool_cap - used only when
    // the driver cannot supply the cap.
    const size_t budget = FBF_TRY(res.memory_budget());
    if (budget + usage.used_current < pool_cap) {
      const uint64_t free_bytes = budget + usage.used_current - usage.reserved_current;
      return err_allocation(
          "pre-flight: device {} has {} bytes free ({:.3f} GiB), and its pool "
          "holds {} more, short of the pool's cap of {} bytes ({:.3f} GiB): "
          "max_device_mem_per_gpu = {} less the {} reserve outside the pool; "
          "another process holds the rest",
          res.dev_idx(), free_bytes, static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
          usage.reserved_current - usage.used_current, pool_cap,
          static_cast<double>(pool_cap) / (1024.0 * 1024.0 * 1024.0), device_cap, kReserve);
    }
    const uint64_t taken = usage.used_current + kPoolHeadroomBytes;
    if (pool_cap > taken) {
      const Buffers::device<std::byte> warm(static_cast<size_t>(pool_cap - taken), res.device());
    }
    FBF_TRY(res.sync());
  }
  return {};
}

} // namespace free_breakfast
