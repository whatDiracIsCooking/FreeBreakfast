// CiSigma's apply and apply_fp32 unbounded against blocked under a workspace
// limit, per column, with the workspace each grows to; or ci_solve's singlets
// in the symmetric layout under that limit, the solve that past ~CAS(13,13)
// does not fit a 10 GB card unblocked. Over the active spaces of a
// direct_spin1-schema JSON file (tools/emit_ci_bench_integrals.py's schema).
//
//   free_breakfast_ci_sigma_blocked CASES.json columns LIMIT_MIB [device = 0]
//   free_breakfast_ci_sigma_blocked CASES.json solve   LIMIT_MIB [n_roots = 3]
//                                   [device = 0]
//
// LIMIT_MIB 0 leaves CiSigma unbounded: the solve is then the unblocked one,
// as before set_workspace_limit existed. `columns` times every layout and
// precision unbounded too, so it is for spaces whose unblocked G fits.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

constexpr double kMiB = 1024.0 * 1024.0;

// Milliseconds per column of apply (apply_fp32 with fp32) of x into y on a
// fresh DeviceResources and CiSigma under bound (0: unbounded), over reps
// after two to warm up (the first grows the workspace), and the MiB of
// workspace it grew to.
std::pair<double, double> time_column(std::int32_t device, std::int32_t reps,
                                      const fb::StringSet &alpha, const fb::StringSet &beta,
                                      double e_core, const std::vector<double> &h1,
                                      const std::vector<double> &eri, std::size_t bound,
                                      fb::CiLayout layout, bool fp32, const double *x, double *y) {
  fb::DeviceResources res{device};
  auto sigma = fb::CiSigma::create(res, alpha, beta);
  if (!sigma || !sigma->set_integrals(res, e_core, h1, eri)) {
    std::println(std::cerr, "CiSigma setup failed");
    std::exit(1);
  }
  if (bound != 0) {
    sigma->set_workspace_limit(bound);
  }
  const auto apply = [&] {
    return fp32 ? sigma->apply_fp32(res, 1, x, y, layout) : sigma->apply(res, 1, x, y, layout);
  };
  for (std::int32_t i = 0; i < 2; ++i) {
    if (const fb::Status s = apply(); !s) {
      std::println(std::cerr, "apply: {}", s.error().message);
      std::exit(1);
    }
  }
  (void)res.sync();
  const auto start = Clock::now();
  for (std::int32_t i = 0; i < reps; ++i) {
    (void)apply();
  }
  (void)res.sync();
  return {1e3 * seconds_since(start) / reps,
          static_cast<double>(res.device_workspace(0).num_elements()) / kMiB};
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  if (argc < 4) {
    std::println(std::cerr, "usage: {} CASES.json columns|solve LIMIT_MIB [n_roots] [device]",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[2];
  const auto limit = static_cast<std::size_t>(std::stod(argv[3]) * kMiB);
  const bool solve = mode == "solve";
  const std::int32_t n_roots = solve && argc > 4 ? std::stoi(argv[4]) : 3;
  const std::int32_t device = std::stoi(argc > (solve ? 5 : 4) ? argv[solve ? 5 : 4] : "0");
  const auto doc = fb::read_json(argv[1]);
  if (!doc) {
    std::println(std::cerr, "{}", doc.error().message);
    return 1;
  }

  for (const auto &entry : (*doc)["cases"]) {
    const auto n = entry["n_orbitals"].get<std::int32_t>();
    const auto alpha = fb::StringSet::create(n, entry["n_alpha"].get<std::int32_t>());
    const auto beta = fb::StringSet::create(n, entry["n_beta"].get<std::int32_t>());
    const auto h1 = entry["h1"]["values"].get<std::vector<double>>();
    const auto eri = entry["eri"]["values"].get<std::vector<double>>();
    const auto e_core = entry["e_core"].get<double>();
    const auto name = entry["name"].get<std::string>();

    fb::DeviceResources res{device};
    auto sigma = fb::CiSigma::create(res, *alpha, *beta);
    if (!sigma || !sigma->set_integrals(res, e_core, h1, eri)) {
      std::println(std::cerr, "{}: CiSigma setup failed", name);
      return 1;
    }
    const auto dim = static_cast<std::size_t>(sigma->size());
    fb::Buffers::device<double> diagonal(dim, res.device());
    (void)sigma->diagonal(res, diagonal.data());
    std::println("{}  dim {}, symmetric {}, limit {:.0f} MiB", name, dim,
                 sigma->size(fb::CiLayout::symmetric), static_cast<double>(limit) / kMiB);

    if (!solve) {
      // Any column will do: the cost does not depend on c. In the symmetric
      // layout the first size(symmetric) elements of diag(H) stand in for a
      // packed vector.
      fb::Buffers::device<double> y(dim, res.device());
      const std::int32_t reps = dim > 200'000 ? 5 : 20;
      for (const fb::CiLayout layout : {fb::CiLayout::full, fb::CiLayout::symmetric}) {
        for (const bool fp32 : {false, true}) {
          const auto cost = [&](std::size_t bound) {
            return time_column(device, reps, *alpha, *beta, e_core, h1, eri, bound, layout, fp32,
                               diagonal.data(), y.data());
          };
          const auto [free_ms, free_mib] = cost(0);
          const auto [cap_ms, cap_mib] = cost(limit);
          std::println("  {:9} {}: unbounded {:7.2f} ms, {:7.1f} MiB; blocked {:7.2f} ms, "
                       "{:7.1f} MiB",
                       layout == fb::CiLayout::full ? "full" : "symmetric", fp32 ? "FP32" : "FP64",
                       free_ms, free_mib, cap_ms, cap_mib);
        }
      }
      continue;
    }

    if (limit != 0) {
      sigma->set_workspace_limit(limit);
    }
    const auto start = Clock::now();
    const auto roots = fb::ci_solve(res, *sigma, *alpha, *beta, diagonal.data(), n_roots);
    const double wall = seconds_since(start);
    if (!roots) {
      std::println("  ci_solve: {}", roots.error().message);
      return 1;
    }
    std::println("  ci_solve: {:.1f} s, iterations FP32 {} + FP64 {}, workspace {:.1f} MiB", wall,
                 roots->fp32_iterations, roots->davidson.iterations,
                 static_cast<double>(res.device_workspace(0).num_elements()) / kMiB);
    for (std::size_t r = 0; r < roots->energies.size(); ++r) {
      std::println("  root {}: E {:.10f}  <S^2> {:.2e}", r, roots->energies[r],
                   roots->spin_squares[r]);
    }
  }
  return 0;
}
