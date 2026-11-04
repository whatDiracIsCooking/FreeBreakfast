// hessian::df_rhf_two_electron's wall clock and peak device memory over an
// all-trans alkane (C_n H_2n+2, built here; propane, 11 atoms, by default) in a
// DF-RHF over the given basis and cc-pVDZ-JKFIT, at the SCF's orbitals.
//
//   free_breakfast_hessian_df_rhf_two_electron [carbons = 3] [basis = cc-pvdz] [reps = 1]
//
// On device 0, where the pipeline runs. Prints the system's sizes, the
// seconds per call (the best of reps, after one to warm up; reps 0 times the
// one call), the peak device memory a call took beyond what was in use before
// it (the device's free bytes, polled from a second thread every
// millisecond, so the workspace's growth on the first call counts and a peak
// shorter than a millisecond may not) and the largest |H| of each term. Hold the card (devtools/lib.sh's with_gpu_card)
// while it runs.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(std::string_view what, const fb::Error &error) {
  std::println(std::cerr, "{}: {}", what, error.message);
  std::exit(1);
}

// All-trans C_n H_2n+2 in angstrom, as bench/cphf_batched.cpp builds decane.
void write_alkane(const std::filesystem::path &path, const std::int32_t carbons) {
  constexpr double kDx = 1.26;
  constexpr double kDy = 0.89;
  constexpr double kHy = 0.63;
  constexpr double kHz = 0.89;
  std::vector<std::tuple<std::string_view, double, double, double>> atoms;
  for (std::int32_t i = 0; i < carbons; ++i) {
    const double x = kDx * i;
    const double y = (i % 2 == 0) ? 0.0 : kDy;
    const double away = (i % 2 == 0) ? -1.0 : 1.0;
    atoms.emplace_back("C", x, y, 0.0);
    atoms.emplace_back("H", x, y + (away * kHy), kHz);
    atoms.emplace_back("H", x, y + (away * kHy), -kHz);
  }
  const double scale = 1.09 / std::hypot(kDx, kDy);
  atoms.emplace_back("H", -kDx * scale, kDy * scale, 0.0);
  const double last_x = kDx * (carbons - 1);
  const double last_y = ((carbons - 1) % 2 == 0) ? 0.0 : kDy;
  const double next_y = (carbons % 2 == 0) ? 0.0 : kDy;
  atoms.emplace_back("H", last_x + (kDx * scale), last_y + ((next_y - last_y) * scale), 0.0);
  std::ofstream xyz{path};
  std::println(xyz, "{}\nalkane", atoms.size());
  for (const auto &[symbol, x, y, z] : atoms) {
    std::println(xyz, "{} {:.6f} {:.6f} {:.6f}", symbol, x, y, z);
  }
}

std::size_t free_bytes() {
  std::size_t free = 0;
  std::size_t total = 0;
  (void)wwr::wwrMemGetInfo(&free, &total);
  return free;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::int32_t carbons = argc > 1 ? std::stoi(argv[1]) : 3;
  const std::string basis = argc > 2 ? argv[2] : "cc-pvdz";
  const std::int32_t reps = argc > 3 ? std::stoi(argv[3]) : 1;
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "fb_hessian_df_rhf_two_electron";
  std::filesystem::create_directories(dir / "out");
  write_alkane(dir / "alkane.xyz", carbons);
  auto calc = fb::parse_config(
      std::format("[molecule]\nxyz = \"alkane.xyz\"\n[basis_set]\nbasis = \"{}\"\n[eri]\nmode = "
                  "\"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n",
                  basis),
      dir, dir / "out");
  if (!calc) {
    fail("config", calc.error());
  }
  fb::NodeResources node;
  if (const fb::Status ran = calc->run(node); !ran) {
    fail("scf", ran.error());
  }
  fb::DeviceResources &res = node.device(0);
  const auto &fit = std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  const fb::ScfOutput &scf = *calc->get<fb::ScfOutput>();
  const fb::BasisSet &orbital = calc->get<fb::BasisSetOutput>()->basis;
  const auto &classes = calc->get<fb::PrimitivePairClassesOutput>()->classes;
  const std::size_t nmo_elements = static_cast<std::size_t>(scf.c.n()) * scf.mo_energies.size();
  fb::Buffers::device<double> c(nmo_elements, res.device());
  (void)wwr::wwrMemcpyAsync(c.data(), scf.c.data(), nmo_elements * sizeof(double),
                            wwr::wwrMemcpyDeviceToDevice, res.stream());
  if (const fb::Status s = res.sync(); !s) {
    fail("sync", s.error());
  }
  std::println("{} atoms, nao {}, naux {}, nocc {}", orbital.natoms(), orbital.nao_sphe(),
               fit.aux.nao_sphe(), scf.n_alpha);

  double best = std::numeric_limits<double>::infinity();
  std::size_t peak = 0;
  fb::hessian::RhfTwoElectron out;
  for (std::int32_t r = 0; r <= reps; ++r) {
    const std::size_t before = free_bytes();
    std::atomic<bool> done{false};
    std::size_t lowest = before;
    std::thread poll{[&] {
      while (!done.load()) {
        lowest = std::min(lowest, free_bytes());
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
      }
    }};
    const auto start = Clock::now();
    auto got = fb::hessian::df_rhf_two_electron(res, orbital, classes, fit.aux, fit.b, fit.l.data(),
                                                c, scf.n_alpha);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    done.store(true);
    poll.join();
    if (!got) {
      fail("df_rhf_two_electron", got.error());
    }
    out = std::move(*got);
    std::println("  call {}: {:.3f} s, peak {:.1f} MiB", r, seconds,
                 static_cast<double>(before - lowest) / (1 << 20));
    // The first call's peak counts the workspace's growth; the timing is the
    // later calls', once it has grown.
    peak = std::max(peak, before - lowest);
    if (r > 0 || reps == 0) {
      best = std::min(best, seconds);
    }
  }
  std::println("best {:.3f} s, peak {:.1f} MiB; max |H| coulomb {:.6f}, exchange {:.6f}", best,
               static_cast<double>(peak) / (1 << 20), max_abs(out.coulomb), max_abs(out.exchange));
  return 0;
}
