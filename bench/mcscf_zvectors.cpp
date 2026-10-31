// mcscf::zvectors' wall clock on 3 n_atom right-hand sides, the count the
// DF-CASSCF Hessian solves, by each CpSolver over the same H~: water, or
// all-trans octatetraene (C8H10, 18 atoms, built here), in a DF-CASSCF over
// the given basis and cc-pVDZ-JKFIT, CAS(4,4) at the frontier, one root or a
// state average of two at equal weights. The right-hand sides are random from
// a fixed seed: each column an orbital one and a CI one, y_r = P H_y c_r for a
// random spin-free H_y. Every solver runs on the same equations, so their
// solutions are compared with PerColumn's.
//
//   free_breakfast_mcscf_zvectors [molecule = octatetraene] [nroots = 1]
//                                 [reps = 2] [basis = cc-pvdz] [solvers = psc]
//
// solvers is a string of p (PerColumn), s (Schur) and c (Coupled). On device
// 0, where the pipeline runs. Prints the system's sizes, then per solver the
// seconds (the best of reps, after one to warm up), the products per column
// (smallest, largest, total), the largest residual and the largest
// |z - z_PerColumn| of each block. Hold the card (devtools/lib.sh's
// with_gpu_card) while it runs.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

namespace {

namespace fb = free_breakfast;

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(std::string_view what, const fb::Error &error) {
  std::println(std::cerr, "{}: {}", what, error.message);
  std::exit(1);
}

template <typename T>
T must(std::string_view what, fb::Result<T> result) {
  if (!result) {
    fail(what, result.error());
  }
  return std::move(*result);
}

void must(std::string_view what, const fb::Status &status) {
  if (!status) {
    fail(what, status.error());
  }
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  must("download", fb::gpu_status(wwr::wwrMemcpyAsync(host.data(), data, count * sizeof(double),
                                                      wwr::wwrMemcpyDeviceToHost, res.stream())));
  must("download", res.sync());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *data) {
  must("upload", fb::gpu_status(wwr::wwrMemcpyAsync(data, host.data(), host.size_bytes(),
                                                    wwr::wwrMemcpyHostToDevice, res.stream())));
  must("upload", res.sync());
}

// All-trans C8H10 in angstrom, in the xy plane: C=C 1.34 and C-C 1.45
// alternating, 120 degrees at each carbon, C-H 1.09 along the outer bisector
// (two at each end carbon, 120 degrees from its bond).
void write_octatetraene(const std::filesystem::path &path) {
  constexpr std::int32_t kCarbons = 8;
  constexpr double kDouble = 1.34;
  constexpr double kSingle = 1.45;
  constexpr double kCh = 1.09;
  const double tilt = std::numbers::pi / 6.0;
  std::vector<std::array<double, 2>> carbons{{0.0, 0.0}};
  for (std::int32_t i = 0; i + 1 < kCarbons; ++i) {
    const double length = (i % 2 == 0) ? kDouble : kSingle;
    const double angle = (i % 2 == 0) ? tilt : -tilt;
    const auto [x, y] = carbons.back();
    carbons.push_back({x + (length * std::cos(angle)), y + (length * std::sin(angle))});
  }
  std::vector<std::tuple<std::string_view, double, double>> atoms;
  const auto unit = [](std::array<double, 2> from, std::array<double, 2> to) {
    const double dx = to[0] - from[0];
    const double dy = to[1] - from[1];
    const double r = std::hypot(dx, dy);
    return std::array{dx / r, dy / r};
  };
  const auto rotate = [](std::array<double, 2> v, double angle) {
    return std::array{(v[0] * std::cos(angle)) - (v[1] * std::sin(angle)),
                      (v[0] * std::sin(angle)) + (v[1] * std::cos(angle))};
  };
  for (std::size_t i = 0; i < carbons.size(); ++i) {
    const auto c = carbons[i];
    atoms.emplace_back("C", c[0], c[1]);
    if (i == 0 || i + 1 == carbons.size()) {
      const auto u = unit(c, carbons[i == 0 ? 1 : i - 1]);
      for (const double angle : {2.0 * std::numbers::pi / 3.0, -2.0 * std::numbers::pi / 3.0}) {
        const auto h = rotate(u, angle);
        atoms.emplace_back("H", c[0] + (kCh * h[0]), c[1] + (kCh * h[1]));
      }
    } else {
      const auto a = unit(c, carbons[i - 1]);
      const auto b = unit(c, carbons[i + 1]);
      const double r = std::hypot(a[0] + b[0], a[1] + b[1]);
      atoms.emplace_back("H", c[0] - (kCh * (a[0] + b[0]) / r), c[1] - (kCh * (a[1] + b[1]) / r));
    }
  }
  std::ofstream xyz{path};
  std::println(xyz, "{}\noctatetraene", atoms.size());
  for (const auto &[symbol, x, y] : atoms) {
    std::println(xyz, "{} {:.6f} {:.6f} 0.0", symbol, x, y);
  }
}

void write_water(const std::filesystem::path &path) {
  std::ofstream xyz{path};
  std::println(xyz, "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692");
}

// A random spin-free active-space Hamiltonian, h1 and (tu|vw) uniform in
// [-1, 1) given their symmetries, row-major.
void random_hamiltonian(std::mt19937_64 &engine, std::size_t n, std::vector<double> &h1,
                        std::vector<double> &eri) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const auto at = [n](std::size_t t, std::size_t u, std::size_t v, std::size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  h1.assign(n * n, 0.0);
  eri.assign(n * n * n * n, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      h1[(t * n) + u] = h1[(u * n) + t] = uniform(engine);
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w <= v; ++w) {
          if ((t * n) + u < (v * n) + w) {
            continue;
          }
          const double value = uniform(engine);
          for (const auto [p, q, r, s] :
               {std::array{t, u, v, w}, std::array{u, t, v, w}, std::array{t, u, w, v},
                std::array{u, t, w, v}, std::array{v, w, t, u}, std::array{w, v, t, u},
                std::array{v, w, u, t}, std::array{w, v, u, t}}) {
            eri[at(p, q, r, s)] = value;
          }
        }
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::string molecule_name = argc > 1 ? argv[1] : "octatetraene";
  const std::int32_t nroots = argc > 2 ? std::stoi(argv[2]) : 1;
  const std::int32_t reps = argc > 3 ? std::stoi(argv[3]) : 2;
  const std::string basis_name = argc > 4 ? argv[4] : "cc-pvdz";
  const std::string solvers = argc > 5 ? argv[5] : "psc";
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fb_mcscf_zvectors";
  std::filesystem::create_directories(dir / "out");
  if (molecule_name == "water") {
    write_water(dir / "molecule.xyz");
  } else {
    write_octatetraene(dir / "molecule.xyz");
  }
  auto calc = must(
      "config", fb::parse_config(std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\n"
                                             "basis = \"{}\"\n[eri]\nmode = \"density_fitting\"\n"
                                             "aux_basis = \"cc-pvdz-jkfit\"\n[scf]\n",
                                             basis_name),
                                 dir, dir / "out"));
  fb::NodeResources node;
  must("scf", calc.run(node));
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto &df = std::get<fb::DensityFitted>(calc.get<fb::EriOutput>()->eri);
  const fb::ScfOutput &scf = *calc.get<fb::ScfOutput>();
  const fb::Molecule &molecule = calc.get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = calc.get<fb::BasisSetOutput>()->basis;
  const auto &h = calc.get<fb::CoreHamiltonianOutput>()->h;
  const auto &u = calc.get<fb::SphericalTransformOutput>()->u;
  const std::size_t n_atom = molecule.atoms().size();

  // The DF-CASSCF, CAS(4,4) at the frontier.
  fb::CasciStep casci;
  casci.ncas = 4;
  casci.nelecas = 4;
  casci.nroots = nroots;
  const auto start = must("start", casci.select(res, molecule, basis, scf));
  const std::int32_t ncore = start.ncore;
  const std::int32_t ncas = start.ncas;
  const std::int32_t nelecas = start.nelecas;
  const std::vector<double> weights(static_cast<std::size_t>(nroots), 1.0 / nroots);
  fb::mcscf::Options options;
  options.conv_tol_grad = 1e-8;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  const auto sol =
      must("casscf", fb::mcscf::solve(res, basis, molecule, h, u, {}, {}, start.c, ncore, ncas,
                                      nelecas, nroots, weights, options, &df.b, df.naux));
  const fb::SphericalMatrixAO &c = sol.c;

  // H and H~ at the solution, the sigma's integrals set as mcscf::solve sets
  // them (h_eff symmetric, (tu|vw) column-major read as row-major).
  const auto mo = static_cast<std::size_t>(c.n());
  const auto core = static_cast<std::size_t>(ncore);
  const auto n = static_cast<std::size_t>(ncas);
  const auto ints = must(
      "integrals", fb::casscf_integrals(res, basis, molecule, h, df.b, df.naux, c, ncore, ncas));
  const std::int32_t n_beta = (nelecas - (molecule.multiplicity() - 1)) / 2;
  const auto alpha = must("strings", fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta));
  const auto beta = must("strings", fb::ci::DeviceStringSet::create(res, ncas, n_beta));
  auto sigma = must("sigma", fb::ci::Sigma::create(res, alpha, beta));
  auto response_sigma = must("sigma", fb::ci::Sigma::create(res, alpha, beta));
  auto rdm = must("rdm", fb::ci::Rdm::create(res, alpha, beta));
  must("sigma", sigma.set_integrals(res, ints.hamiltonian.e_core,
                                    download(res, ints.hamiltonian.h_eff.data(), n * n),
                                    download(res, ints.hamiltonian.eri.data(), n * n * n * n)));
  const auto fock = must("fock", fb::response::fock(res, molecule, h, df.b, df.naux, c, ncore, ncas,
                                                    ints.ppaa, ints.papa, sol.gamma, sol.gamma2));
  auto hessian = must("hessian", fb::response::Hessian::create(res, h, df.b, df.naux, c, ncore,
                                                               ncas, ints.ppaa, ints.papa, fock));
  auto relaxed =
      must("relaxed", fb::mcscf::RelaxedHessian::create(
                          res, hessian, sigma, rdm, alpha, beta, sol.roots, weights,
                          fb::mcscf::ResponseIntegrals{
                              mo, core, n, download(res, fock.fock_inactive.data(), mo * mo),
                              download(res, ints.ppaa.data(), mo * mo * n * n),
                              download(res, ints.papa.data(), mo * mo * n * n)}));
  const auto pairs = static_cast<std::size_t>(relaxed.size());
  const std::size_t dim = static_cast<std::size_t>(sigma.size());
  const std::size_t ci_block = dim * static_cast<std::size_t>(nroots);
  const std::size_t ncol = 3 * n_atom;

  // The right-hand sides: orbital ones uniform in [-0.1, 0.1), CI ones
  // P H_y c with each H_y random.
  std::mt19937_64 engine{611};
  std::uniform_real_distribution<double> uniform{-0.1, 0.1};
  std::vector<double> gradients(pairs * ncol);
  std::ranges::generate(gradients, [&] { return uniform(engine); });
  const std::vector<double> c_host = download(res, sol.roots.vectors.data(), ci_block);
  std::vector<double> y(ci_block * ncol);
  fb::Buffers::device<double> sigma_out(ci_block, res.device());
  for (std::size_t col = 0; col < ncol; ++col) {
    std::vector<double> h1;
    std::vector<double> eri1;
    random_hamiltonian(engine, n, h1, eri1);
    must("sigma", response_sigma.set_integrals(res, 0.0, h1, eri1));
    must("sigma", response_sigma.apply(res, nroots, sol.roots.vectors.data(), sigma_out.data()));
    std::vector<double> block = download(res, sigma_out.data(), ci_block);
    for (std::size_t r = 0; r < static_cast<std::size_t>(nroots); ++r) {
      for (std::size_t s = 0; s < static_cast<std::size_t>(nroots); ++s) {
        double dot = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
          dot += block[(r * dim) + i] * c_host[(s * dim) + i];
        }
        for (std::size_t i = 0; i < dim; ++i) {
          block[(r * dim) + i] -= dot * c_host[(s * dim) + i];
        }
      }
    }
    std::ranges::copy(block, y.begin() + static_cast<std::ptrdiff_t>(col * ci_block));
  }
  fb::Buffers::device<double> gradients_device(gradients.size(), res.device());
  fb::Buffers::device<double> y_device(y.size(), res.device());
  upload(res, gradients, gradients_device.data());
  upload(res, y, y_device.data());

  std::println("{} / {}: {} atoms, nmo {}, ncore {}, CAS({},{}), {} root(s), naux {} "
               "(cc-pVDZ-JKFIT), {} pairs, {} determinants, {} columns",
               molecule_name, basis_name, n_atom, mo, ncore, nelecas, ncas, nroots, df.naux, pairs,
               dim, ncol);
  std::optional<std::pair<std::vector<double>, std::vector<double>>> reference;
  for (const char letter : solvers) {
    const auto [solver, name] =
        letter == 'p'   ? std::pair{fb::mcscf::CpSolver::PerColumn, "PerColumn"}
        : letter == 's' ? std::pair{fb::mcscf::CpSolver::Schur, "Schur"}
                        : std::pair{fb::mcscf::CpSolver::Coupled, "Coupled"};
    double best = std::numeric_limits<double>::infinity();
    std::optional<fb::mcscf::ZVectors> solved;
    for (std::int32_t rep = 0; rep <= reps; ++rep) {
      const auto begin = Clock::now();
      auto out = must(name, fb::mcscf::zvectors(res, relaxed, hessian, sol.roots,
                                                static_cast<std::int32_t>(ncol),
                                                gradients_device.data(), y_device.data(), solver));
      const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
      if (rep > 0) {
        best = std::min(best, seconds);
      }
      std::println("  {} solve {}: {:.3f} s{}", name, rep, seconds, rep == 0 ? " (warm-up)" : "");
      solved = std::move(out);
    }
    const std::vector<double> z_o = download(res, solved->orbital.data(), pairs * ncol);
    const std::vector<double> z_c = download(res, solved->ci.data(), ci_block * ncol);
    const auto max_abs = [](std::span<const double> v) {
      return std::ranges::fold_left(v, 0.0,
                                    [](double m, double x) { return std::max(m, std::abs(x)); });
    };
    const auto max_diff = [](std::span<const double> a, std::span<const double> b) {
      double m = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
      }
      return m;
    };
    std::string against = "the reference";
    if (!reference) {
      reference.emplace(z_o, z_c);
    } else {
      against = std::format("max |dz_o| {:.3e}, max |dz_c| {:.3e} against the first",
                            max_diff(z_o, reference->first), max_diff(z_c, reference->second));
    }
    std::println("{}: {:.3f} s (best of {}), products per column {}..{} ({} in all), max "
                 "residual {:.3e}, max |z_o| {:.3e}, max |z_c| {:.3e}; {}",
                 name, best, reps, std::ranges::min(solved->iterations),
                 std::ranges::max(solved->iterations),
                 std::reduce(solved->iterations.begin(), solved->iterations.end()),
                 std::ranges::max(solved->residuals), max_abs(z_o), max_abs(z_c), against);
  }
  std::filesystem::remove_all(dir);
  return 0;
}
