/// @file
/// @brief `free-breakfast-sad-stamps`: compute the SAD atomic density of every
///        element of every compiled-in basis set, and write each one as
///        `data/sad/<basis>/<Z>.npy`.
///
/// The generator behind the baked SAD stamps that `src/io/sad` compiles in. It
/// runs exactly the computation the guess would run at runtime -- the
/// compiled-in basis through @c resolve_basis, its primitives normalized as the
/// BasisSet step normalizes them, the atom's shells alone at the origin, and
/// @c sad_density over them -- so a baked stamp is the block a run would
/// otherwise compute. It passes no @c SadStampLookup, so it can only compute:
/// the stamps it overwrites never feed it.
///
/// The basis sets are the `<data>/basis/*.json` files, each resolved by its
/// file name. Resolving goes through the binary, not the file, so a JSON file
/// this build was not configured with is an error ("not compiled in"), not a
/// stamp computed over shells the runtime would never see. Rebuild first.
///
/// Every subdirectory of `<data>/sad/` is removed before the new stamps are
/// written, so an element or a basis set that has gone leaves nothing behind.
/// An element whose SCF is refused as too large (ErrorKind::Unsupported) is
/// reported and skipped: the runtime refuses it the same way. Any other failure
/// fails the run, and nothing is written.

#include "diagnostics.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.npy;
import free_breakfast.sad;

namespace {

namespace fb = free_breakfast;

// One element of one basis set: its shells at the origin, grouped as one atom,
// and what its SCF made of them.
struct Job {
  std::string basis;
  std::int32_t z = 0;
  fb::BasisSet shells;
  fb::Result<std::vector<double>> density;
  double seconds = 0.0;
};

// A job per element of the compiled-in set @p spec names. The BasisSet is
// built without a name, so nothing downstream could look a baked stamp up for
// it even if asked to.
fb::Result<std::vector<Job>> jobs_for(const std::string &spec) {
  const fb::BasisStamp stamp = FBF_TRY(fb::resolve_basis(spec));
  if (stamp.name().empty()) {
    return fb::err_config("{} is not compiled into this binary; reconfigure and rebuild first",
                          spec);
  }
  const fb::BasisStamp normalized = stamp.primitive_normalized();
  std::vector<Job> jobs;
  for (const auto &[z, element] : normalized.element_shells()) {
    std::vector<fb::Shell> shells;
    for (const fb::ShellStamp &shell : element) {
      shells.push_back(fb::place(shell, fb::Vec<double, 3>{0.0, 0.0, 0.0}));
    }
    const auto count = static_cast<std::uint32_t>(shells.size());
    jobs.push_back(Job{stamp.name(), z, fb::BasisSet(std::move(shells), {0, count}), {}, 0.0});
  }
  return jobs;
}

// Run every job's SCF, one task each: the elements are independent, and the
// heaviest of them take seconds.
void compute(std::vector<Job> &jobs) {
  fb::tf::Executor executor;
  fb::tf::Taskflow flow;
  for (Job &job : jobs) {
    flow.emplace([&job] {
      const std::vector<fb::Atom> atom{
          fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(job.z)}};
      const auto start = std::chrono::steady_clock::now();
      job.density = fb::sad_density(job.shells, atom);
      job.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    });
  }
  executor.run(flow).wait();
}

fb::Status run(std::span<char *const> args) {
  if (args.size() != 2) {
    return fb::err_config("usage: {} <repo>/data",
                          args.empty() ? "free-breakfast-sad-stamps" : args[0]);
  }
  const std::filesystem::path data{args[1]};
  const std::filesystem::path basis_dir = data / "basis";
  const std::filesystem::path sad_dir = data / "sad";

  std::error_code ec;
  std::vector<std::string> specs;
  for (const auto &entry : std::filesystem::directory_iterator(basis_dir, ec)) {
    if (entry.path().extension() == ".json") {
      specs.push_back(entry.path().stem().string());
    }
  }
  if (ec) {
    return fb::err_io("cannot list {}: {}", basis_dir.string(), ec.message());
  }
  std::ranges::sort(specs);

  std::vector<Job> jobs;
  for (const std::string &spec : specs) {
    std::vector<Job> more = FBF_TRY(jobs_for(spec));
    std::ranges::move(more, std::back_inserter(jobs));
  }
  compute(jobs);

  // Every failure but a too-large atom is checked before anything is removed,
  // so a failed run leaves the old stamps in place.
  for (const Job &job : jobs) {
    if (!job.density && job.density.error().kind != fb::ErrorKind::Unsupported) {
      return std::unexpected(job.density.error());
    }
  }

  std::filesystem::create_directories(sad_dir, ec);
  if (ec) {
    return fb::err_io("cannot create {}: {}", sad_dir.string(), ec.message());
  }
  for (const auto &entry : std::filesystem::directory_iterator(sad_dir, ec)) {
    if (entry.is_directory()) {
      std::filesystem::remove_all(entry.path(), ec);
      if (ec) {
        return fb::err_io("cannot remove {}: {}", entry.path().string(), ec.message());
      }
    }
  }
  if (ec) {
    return fb::err_io("cannot list {}: {}", sad_dir.string(), ec.message());
  }

  for (const Job &job : jobs) {
    if (!job.density) {
      std::println("{:>8} Z={:<3} skipped: {}", job.basis, job.z, job.density.error().message);
      continue;
    }
    const std::filesystem::path dir = sad_dir / job.basis;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      return fb::err_io("cannot create {}: {}", dir.string(), ec.message());
    }
    const std::size_t n = job.shells.nao_sphe();
    FBF_TRY(
        fb::write_npy(dir / std::format("{:03}.npy", job.z), fb::NpyArray{{n, n}, *job.density}));
    std::println("{:>8} Z={:<3} {:>3} AOs  {:8.3f} s", job.basis, job.z, n, job.seconds);
  }
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status) {
    std::println(std::cerr, "error: {}", status.error().message);
    return fb::to_exit_code(status.error().kind);
  }
  return 0;
}
