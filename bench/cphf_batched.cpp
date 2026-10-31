// response::cphf's wall clock on 3 n_atom right-hand sides, the count the
// DF-RHF Hessian solves, over decane (C10H22, 32 atoms, an all-trans zigzag
// built here) in a DF-RHF over the given basis and cc-pVDZ-JKFIT. The
// right-hand sides are random, from a fixed seed, so two builds of the solver
// see the same equations: run it once per build and pass the first run's
// solution file to the second to read how far apart the two solutions are.
// The second run solves over the first run's orbitals (u_out.orbitals), not
// its own SCF's, which are not bitwise reproducible.
//
//   free_breakfast_cphf_batched [basis = cc-pvdz] [reps = 3] [u_out] [u_reference]
//
// On device 0, where the pipeline runs. Prints the system's sizes, the
// seconds per solve (the best of reps, after one to warm up), the sweeps and
// the largest residual; writes U to u_out as raw doubles when given, and
// prints max |U - U_reference| when u_reference is. Hold the card
// (devtools/lib.sh's with_gpu_card) while it runs.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.response;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

using Clock = std::chrono::steady_clock;

constexpr std::int32_t kCarbons = 10;

[[noreturn]] void fail(std::string_view what, const fb::Error &error) {
  std::println(std::cerr, "{}: {}", what, error.message);
  std::exit(1);
}

// All-trans C_n H_2n+2 in angstrom: the carbons zigzag in the xy plane, C-C
// 1.54, each C's two hydrogens out of the plane on the side away from its
// neighbours, and each end's third along the next carbon's place.
void write_alkane(const std::filesystem::path &path) {
  constexpr double kDx = 1.26;
  constexpr double kDy = 0.89;
  constexpr double kHy = 0.63;
  constexpr double kHz = 0.89;
  std::vector<std::tuple<std::string_view, double, double, double>> atoms;
  for (std::int32_t i = 0; i < kCarbons; ++i) {
    const double x = kDx * i;
    const double y = (i % 2 == 0) ? 0.0 : kDy;
    const double away = (i % 2 == 0) ? -1.0 : 1.0;
    atoms.emplace_back("C", x, y, 0.0);
    atoms.emplace_back("H", x, y + (away * kHy), kHz);
    atoms.emplace_back("H", x, y + (away * kHy), -kHz);
  }
  // The ends' third hydrogens, 1.09 along where carbon -1 and carbon n would be.
  const double scale = 1.09 / std::hypot(kDx, kDy);
  atoms.emplace_back("H", -kDx * scale, kDy * scale, 0.0);
  const double last_x = kDx * (kCarbons - 1);
  const double last_y = ((kCarbons - 1) % 2 == 0) ? 0.0 : kDy;
  const double next_y = (kCarbons % 2 == 0) ? 0.0 : kDy;
  atoms.emplace_back("H", last_x + (kDx * scale), last_y + ((next_y - last_y) * scale), 0.0);
  std::ofstream xyz{path};
  std::println(xyz, "{}\ndecane", atoms.size());
  for (const auto &[symbol, x, y, z] : atoms) {
    std::println(xyz, "{} {:.6f} {:.6f} {:.6f}", symbol, x, y, z);
  }
}

void write_doubles(std::ofstream &file, const std::vector<double> &v) {
  file.write(reinterpret_cast<const char *>(v.data()),
             static_cast<std::streamsize>(v.size() * sizeof(double)));
}

void read_doubles(std::ifstream &file, std::vector<double> &v) {
  file.read(reinterpret_cast<char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::string basis = argc > 1 ? argv[1] : "cc-pvdz";
  const std::int32_t reps = argc > 2 ? std::stoi(argv[2]) : 3;
  const std::filesystem::path u_out = argc > 3 ? argv[3] : "";
  const std::filesystem::path u_reference = argc > 4 ? argv[4] : "";
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fb_cphf_batched";
  std::filesystem::create_directories(dir / "out");
  write_alkane(dir / "decane.xyz");
  auto calc = fb::parse_config(
      std::format("[molecule]\nxyz = \"decane.xyz\"\n[basis_set]\nbasis = \"{}\"\n[eri]\nmode = "
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
  const std::size_t n_atom = calc->get<fb::MoleculeOutput>()->molecule.atoms().size();

  const std::int32_t nocc = scf.n_alpha;
  const std::size_t nmo = scf.mo_energies.size();
  const std::size_t pairs = (nmo - static_cast<std::size_t>(nocc)) * static_cast<std::size_t>(nocc);
  const std::size_t ncol = 3 * n_atom;
  std::mt19937_64 engine{595};
  std::uniform_real_distribution<double> uniform{-0.1, 0.1};
  std::vector<double> host(pairs * ncol);
  std::ranges::generate(host, [&] { return uniform(engine); });
  // C and the orbital energies as cphf takes them: the SCF's, or the
  // reference run's. The SCF is not bitwise reproducible from run to run, and
  // decane's near-degenerate orbitals rotate among themselves with it, so two
  // runs' U (and right-hand sides, in the MO basis) compare only over the
  // same orbitals: each run writes its own beside U, and a run given a
  // reference solves over the reference's.
  const std::size_t nao = scf.c.n();
  std::vector<double> c_host(nao * nmo);
  std::vector<double> energies = scf.mo_energies;
  (void)wwr::wwrMemcpyAsync(c_host.data(), scf.c.data(), c_host.size() * sizeof(double),
                            wwr::wwrMemcpyDeviceToHost, res.stream());
  (void)res.sync();
  if (!u_reference.empty()) {
    std::ifstream file{std::filesystem::path{u_reference} += ".orbitals", std::ios::binary};
    read_doubles(file, c_host);
    read_doubles(file, energies);
    if (!file) {
      std::println(std::cerr, "{}.orbitals: not {} doubles", u_reference.string(),
                   c_host.size() + energies.size());
      return 1;
    }
  }
  if (!u_out.empty()) {
    std::ofstream file{std::filesystem::path{u_out} += ".orbitals", std::ios::binary};
    write_doubles(file, c_host);
    write_doubles(file, energies);
  }
  fb::Buffers::device<double> c(c_host.size(), res.device());
  (void)wwr::wwrMemcpyAsync(c.data(), c_host.data(), c_host.size() * sizeof(double),
                            wwr::wwrMemcpyHostToDevice, res.stream());
  fb::Buffers::device<double> rhs(host.size(), res.device());
  (void)wwr::wwrMemcpyAsync(rhs.data(), host.data(), host.size() * sizeof(double),
                            wwr::wwrMemcpyHostToDevice, res.stream());
  (void)res.sync();

  std::println("decane / {}: {} atoms, nao {}, n_occ {}, n_vir n_occ {}, naux {} "
               "(cc-pVDZ-JKFIT), {} columns",
               basis, n_atom, scf.c.n(), nocc, pairs, fit.naux, ncol);
  double best = std::numeric_limits<double>::infinity();
  std::optional<fb::response::Cphf> solved;
  for (std::int32_t rep = 0; rep <= reps; ++rep) {
    const auto start = Clock::now();
    auto out = fb::response::cphf(res, fit.b, fit.naux, c, energies, nocc, rhs);
    if (!out) {
      fail("cphf", out.error());
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (rep > 0) {
      best = std::min(best, seconds);
    }
    std::println("  solve {}: {:.3f} s{}", rep, seconds, rep == 0 ? " (warm-up)" : "");
    solved = std::move(*out);
  }
  std::println("cphf: {:.3f} s (best of {}), {} sweeps, max residual {:.3e}", best, reps,
               solved->iterations, std::ranges::max(solved->residuals));

  std::vector<double> u(pairs * ncol);
  (void)wwr::wwrMemcpyAsync(u.data(), solved->solution.data(), u.size() * sizeof(double),
                            wwr::wwrMemcpyDeviceToHost, res.stream());
  (void)res.sync();
  if (!u_out.empty()) {
    std::ofstream file{u_out, std::ios::binary};
    write_doubles(file, u);
  }
  if (!u_reference.empty()) {
    std::vector<double> reference(u.size());
    std::ifstream file{u_reference, std::ios::binary};
    read_doubles(file, reference);
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      worst = std::max(worst, std::abs(u[i] - reference[i]));
      scale = std::max(scale, std::abs(reference[i]));
    }
    std::println("max |U - U_reference| {:.3e} (max |U_reference| {:.3e})", worst, scale);
  }
  std::filesystem::remove_all(dir);
  return 0;
}
