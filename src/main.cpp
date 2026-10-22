/// @file
/// @brief The `free-breakfast` executable: read a TOML config, run the steps
///        it names, and report what they built.
///
/// `main` is the top-level boundary, and the only one: the one place a
/// value-tier Error is logged and turned into a process exit code
/// (`to_exit_code`). Everything below it returns through Result; nothing
/// below it logs a value-tier error itself.
///
/// It runs the steps by hand, in the order `Pipeline<Steps...>` will plan
/// them, binding each output into its consumer's `View`. When the planner
/// lands, that becomes `read_config`'s step vector handed to
/// `Pipeline::create` / `run`, and the config file stays the only thing main
/// knows about.

#include "diagnostics.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

fb::Status run(std::span<char *const> args) {
  if (args.size() != 2) {
    return fb::err_config("usage: {} <config.toml>", args.empty() ? "free-breakfast" : args[0]);
  }
  // Read before the node is built: a bad config fails without touching a GPU.
  const auto [molecule_step, basis_set_step] = FBF_TRY(fb::read_config(args[1]));

  fb::NodeResources node;
  node.log();
  if (node.device_count() == 0) {
    return fb::err_gpu("no GPU device visible");
  }
  // One device for now, driven from this thread. Opened before any product,
  // so it is closed after every product's frees.
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};

  const fb::Molecule molecule = FBF_TRY(molecule_step.run(fb::View<>{}, res));
  const fb::BasisSet basis = FBF_TRY(basis_set_step.run(fb::View<fb::Molecule>{molecule}));

  std::println("atoms:             {}", molecule.natoms());
  std::println("charge:            {}", molecule.charge());
  std::println("multiplicity:      {}", molecule.multiplicity());
  std::println("electrons:         {}", molecule.electron_count());
  std::println("nuclear repulsion: {:.10f} Ha", fb::nuclear_repulsion_energy(molecule));
  std::println("shells:            {}", basis.nshells());
  std::println("AOs (cartesian):   {}", basis.nao_cart());
  std::println("AOs (spherical):   {}", basis.nao_sphe());
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
