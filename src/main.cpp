/// @file
/// @brief The `free-breakfast` executable: read a TOML config, run the
///        pipeline it asks for, and report what it built.
///
/// `main` is the top-level boundary, and the only one: the one place a
/// value-tier Error is logged and turned into a process exit code
/// (`to_exit_code`). Everything below it returns through Result; nothing
/// below it logs a value-tier error itself.
///
/// The config file is the only thing it knows about: `read_config` plans the
/// pipeline, `run` runs it, and main prints each product the pipeline built.

#include "diagnostics.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// One block per product type, each printed only if the pipeline built it.
void print_products(const fb::Calculation &calc) {
  if (const auto &molecule = calc.get<fb::Molecule>()) {
    std::println("atoms:             {}", molecule->natoms());
    std::println("charge:            {}", molecule->charge());
    std::println("multiplicity:      {}", molecule->multiplicity());
    std::println("electrons:         {}", molecule->electron_count());
    std::println("nuclear repulsion: {:.10f} Ha", fb::nuclear_repulsion_energy(*molecule));
  }
  if (const auto &basis = calc.get<fb::BasisSet>()) {
    std::println("shells:            {}", basis->nshells());
    std::println("AOs (cartesian):   {}", basis->nao_cart());
    std::println("AOs (spherical):   {}", basis->nao_sphe());
  }
  if (const auto &transform = calc.get<fb::SphericalTransform>()) {
    std::println("spherical U:       {}", transform->u ? "uploaded" : "identity");
  }
  if (const auto &pairs = calc.get<fb::PrimitivePairClasses>()) {
    std::println("pair classes:      {}", pairs->classes.size());
  }
  if (const auto &overlap = calc.get<fb::Overlap>()) {
    std::println("overlap:           {0} x {0} (spherical)", overlap->s.n());
  }
  if (const auto &orthogonalizer = calc.get<fb::Orthogonalizer>()) {
    std::println("orthogonalizer:    {0} x {0}, eigenvalue floor {1:g}", orthogonalizer->x.n(),
                 calc.step<fb::OrthogonalizerStep>()->eigenvalue_floor);
  }
  if (const auto &core_hamiltonian = calc.get<fb::CoreHamiltonian>()) {
    std::println("core Hamiltonian:  {0} x {0} (spherical)", core_hamiltonian->h.n());
  }
  // The guess read the basis, so a guess means a basis to size it by.
  if (calc.get<fb::Guess>()) {
    std::println("initial guess:     {0} x {0} (spherical, SAD)",
                 calc.get<fb::BasisSet>()->nao_sphe());
  }
  if (const auto &scf = calc.get<fb::Scf>()) {
    std::println("SCF iterations:    {}", scf->iterations);
    std::println("RHF energy:        {:.10f} Ha", scf->energy);
  }
  if (const auto &pm = calc.get<fb::Localization>()) {
    std::println("Pipek-Mezey:       {} orbitals, {} sweeps", pm->n_occ, pm->info.sweeps);
    std::println("PM functional:     {:.10f} -> {:.10f}", pm->info.initial_functional,
                 pm->info.functional);
  }
}

fb::Status run(std::span<char *const> args) {
  if (args.size() != 2) {
    return fb::err_config("usage: {} <config.toml>", args.empty() ? "free-breakfast" : args[0]);
  }
  // Read and planned before the node is built: a bad config fails without
  // touching a GPU. The products hold their Device, so they may outlive node.
  fb::Calculation calc = FBF_TRY(fb::read_config(args[1]));

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
