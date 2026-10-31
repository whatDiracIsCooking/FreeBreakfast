/// @file
/// @brief The `free-breakfast` executable: read a TOML config, run the
///        pipeline it asks for, and report what it built.
///
/// `main` is the top-level boundary, and the only one: the one place a
/// value-tier Error is logged and turned into a process exit code
/// (`to_exit_code`). Everything below it returns through Result; nothing
/// below it logs a value-tier error itself.
///
/// `free-breakfast [-i <input_dir>] [-o <output_dir>] <config.toml>`: the
/// config's paths are relative to the input directory (default `.`), and every
/// file the run writes lands in the output directory (default `fbf_output`),
/// named after the step that wrote it. `read_config` plans the pipeline,
/// `run` runs it, and main prints each product the pipeline built.

#include "diagnostics.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// One block per product type, each printed only if the pipeline built it.
void print_products(const fb::Calculation &calc) {
  // After an [optimize], the molecule below is the optimized one.
  if (const auto &out = calc.get<fb::OptimizeOutput>()) {
    std::println("optimized energy:  {:.10f} Ha", out->energy);
  }
  if (const auto &out = calc.get<fb::MoleculeOutput>()) {
    const fb::Molecule &molecule = out->molecule;
    std::println("atoms:             {}", molecule.natoms());
    std::println("charge:            {}", molecule.charge());
    std::println("multiplicity:      {}", molecule.multiplicity());
    std::println("electrons:         {}", molecule.electron_count());
    std::println("nuclear repulsion: {:.10f} Ha", fb::nuclear_repulsion_energy(molecule));
  }
  if (const auto &out = calc.get<fb::BasisSetOutput>()) {
    const fb::BasisSet &basis = out->basis;
    std::println("shells:            {}", basis.nshells());
    std::println("AOs (cartesian):   {}", basis.nao_cart());
    std::println("AOs (spherical):   {}", basis.nao_sphe());
  }
  if (const auto &transform = calc.get<fb::SphericalTransformOutput>()) {
    std::println("spherical U:       {}", transform->u ? "uploaded" : "identity");
  }
  if (const auto &pairs = calc.get<fb::PrimitivePairClassesOutput>()) {
    std::println("pair classes:      {}", pairs->classes.size());
  }
  if (const auto &eri = calc.get<fb::EriOutput>()) {
    if (const auto *fit = std::get_if<fb::DensityFitted>(&eri->eri)) {
      std::println("ERIs:              density-fitted, {} fitting functions", fit->naux);
    } else {
      std::println("ERIs:              integral-direct");
    }
  }
  if (const auto &overlap = calc.get<fb::OverlapOutput>()) {
    std::println("overlap:           {0} x {0} (spherical)", overlap->s.n());
  }
  if (const auto &orthogonalizer = calc.get<fb::OrthogonalizerOutput>()) {
    std::println("orthogonalizer:    {0} x {0}, eigenvalue floor {1:g}", orthogonalizer->x.n(),
                 calc.step<fb::OrthogonalizerStep>()->eigenvalue_floor);
  }
  if (const auto &core_hamiltonian = calc.get<fb::CoreHamiltonianOutput>()) {
    std::println("core Hamiltonian:  {0} x {0} (spherical)", core_hamiltonian->h.n());
  }
  // The guess read the basis, so a guess means a basis to size it by.
  if (calc.get<fb::GuessOutput>()) {
    std::println("initial guess:     {0} x {0} (spherical, SAD)",
                 calc.get<fb::BasisSetOutput>()->basis.nao_sphe());
  }
  if (const auto &scf = calc.get<fb::ScfOutput>()) {
    std::println("SCF iterations:    {}", scf->iterations);
    std::println("RHF energy:        {:.10f} Ha", scf->energy);
  }
  if (const auto &pm = calc.get<fb::LocalizationOutput>()) {
    std::println("Pipek-Mezey:       {} orbitals, {} sweeps", pm->n_occ, pm->info.sweeps);
    std::println("PM functional:     {:.10f} -> {:.10f}", pm->info.initial_functional,
                 pm->info.functional);
  }
  if (const auto &casci = calc.get<fb::CasciOutput>()) {
    std::println("CASCI:             CAS({}, {}), {} core, {} Davidson iterations", casci->nelecas,
                 casci->ncas, casci->ncore, casci->davidson.iterations);
    for (std::size_t r = 0; r < casci->energies.size(); ++r) {
      std::println("CASCI root {}:      {:.10f} Ha, <S^2> {:.6f}", r, casci->energies[r],
                   casci->spin_squares[r]);
    }
  }
  if (const auto &casscf = calc.get<fb::CasscfOutput>()) {
    std::println("CASSCF:            CAS({}, {}), {} core, {} macro iterations, |g| {:.3e}",
                 casscf->nelecas, casscf->ncas, casscf->ncore, casscf->iterations.size(),
                 casscf->gradient_norm);
    for (std::size_t r = 0; r < casscf->energies.size(); ++r) {
      std::println("CASSCF root {}:     {:.10f} Ha, <S^2> {:.6f}, weight {:.4f}", r,
                   casscf->energies[r], casscf->spin_squares[r], casscf->weights[r]);
    }
    std::println("CASSCF energy:     {:.10f} Ha", casscf->energy);
  }
  if (const auto &nevpt2 = calc.get<fb::Nevpt2Output>()) {
    const auto &casscf = calc.get<fb::CasscfOutput>();
    for (std::size_t r = 0; r < nevpt2->roots.size(); ++r) {
      std::println("SC-NEVPT2 root {}:  {:.10f} Ha, E_corr {:.10f} Ha", r,
                   casscf->energies[r] + nevpt2->roots[r].total, nevpt2->roots[r].total);
    }
    for (std::size_t r = 0; r < nevpt2->pc_roots.size(); ++r) {
      if (const auto &pc = nevpt2->pc_roots[r]) {
        std::println("PC-NEVPT2 root {}:  {:.10f} Ha, E_corr {:.10f} Ha", r,
                     casscf->energies[r] + pc->total, pc->total);
      } else {
        std::println("PC-NEVPT2 root {}:  refused ({})", r, nevpt2->pc_refusals[r]);
      }
    }
  }
  if (const auto &fci = calc.get<fb::FciOutput>()) {
    std::println("FCI:               {} electrons in {} orbitals, {} Davidson iterations",
                 fci->n_electrons, fci->nmo, fci->davidson.iterations);
    for (std::size_t r = 0; r < fci->energies.size(); ++r) {
      std::println("FCI root {}:        {:.10f} Ha, <S^2> {:.6f}", r, fci->energies[r],
                   fci->spin_squares[r]);
    }
  }
  if (const auto &out = calc.get<fb::CisOutput>()) {
    std::println("CIS:               {} occupied, {} virtual, {} Davidson iterations", out->n_occ,
                 out->n_virt, out->iterations);
    for (std::size_t r = 0; r < out->states.size(); ++r) {
      std::println("CIS state {}:       {:.10f} Ha above the reference", r,
                   out->states[r].excitation_energy);
    }
  }
  if (const auto &out = calc.get<fb::GradientOutput>()) {
    const std::vector<double> &g = out->gradient.total;
    for (std::size_t a = 0; a + 2 < g.size(); a += 3) {
      std::println("gradient atom {}:   {:16.10f} {:16.10f} {:16.10f} Ha/bohr", a / 3, g[a],
                   g[a + 1], g[a + 2]);
    }
  }
  if (const auto &out = calc.get<fb::Nevpt2GradientOutput>()) {
    const fb::nevpt2::ScGradient &sc = out->gradient;
    std::println("SC-NEVPT2 gradient: root {}, E_corr {:.10f} Ha, max |z^c| {:.3e}, smallest "
                 "inactive gap {:.3e}, {} PCG iterations (residual {:.2e})",
                 calc.step<fb::Nevpt2GradientStep>()->root, sc.energies.total, sc.zc.max_z,
                 sc.zc.min_gap, sc.multipliers.iterations, sc.multipliers.residual);
    const std::vector<double> &g = sc.terms.total;
    for (std::size_t a = 0; a + 2 < g.size(); a += 3) {
      std::println("NEVPT2 grad atom {}: {:16.10f} {:16.10f} {:16.10f} Ha/bohr", a / 3, g[a],
                   g[a + 1], g[a + 2]);
    }
  }
}

fb::Status run(std::span<char *const> args) {
  const std::string_view program = args.empty() ? "free-breakfast" : args[0];
  std::filesystem::path input_dir = ".";
  std::filesystem::path output_dir = "fbf_output";
  std::optional<std::filesystem::path> config;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if ((arg == "-i" || arg == "-o") && i + 1 < args.size()) {
      (arg == "-i" ? input_dir : output_dir) = args[++i];
    } else if (!config && !arg.starts_with('-')) {
      config = arg;
    } else {
      config.reset();
      break;
    }
  }
  if (!config) {
    return fb::err_config("usage: {} [-i <input_dir>] [-o <output_dir>] <config.toml>", program);
  }
  // Read and planned, and the output directory made, before the node is
  // built: a bad config or path fails without touching a GPU. The products
  // hold their Device, so they may outlive node.
  fb::Calculation calc = FBF_TRY(fb::read_config(*config, input_dir, output_dir));

  fb::NodeResources node;
  node.log();
  FBF_TRY(calc.run(node));
  print_products(calc);
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status) {
    fb::log::error(status.error().message, status.error().origin);
  }
  std::println("completed: {} errors, {} warnings", status ? 0 : 1, fb::log::warning_count());
  return status ? 0 : fb::to_exit_code(status.error().kind);
}
