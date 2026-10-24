// CiSigma's apply against apply_fp32, one column at a time, and ci_solve with
// its single-precision stage off and at several fp32_tolerance values, wall
// clock, over the active spaces of a direct_spin1-schema JSON file
// (tools/emit_ci_bench_integrals.py writes N2's CAS(10,10) and CAS(12,12)).
//
//   free_breakfast_ci_sigma_precision CASES.json [n_roots = 3] [max_iterations = 100]
//                                     [device = 0]
//
// Each solve's energies are compared with the FP64-only solve's, the worst
// root's difference printed, so a speedup that cost accuracy shows as one.

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

// Milliseconds per column of @p apply, over @p reps after two to warm up
// (the first grows the workspace).
template <typename Apply>
double ms_per_column(fb::DeviceResources &res, std::int32_t reps, Apply &&apply) {
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
  return 1e3 * seconds_since(start) / reps;
}

// ‖apply_fp32 c − apply c‖ / ‖apply c − E_core c‖ for the device vector @p c.
double fp32_error(fb::DeviceResources &res, fb::CiSigma &sigma, double e_core, const double *c,
                  std::size_t dim) {
  fb::Buffers::device<double> y64(dim, res.device());
  fb::Buffers::device<double> y32(dim, res.device());
  (void)sigma.apply(res, 1, c, y64.data());
  (void)sigma.apply_fp32(res, 1, c, y32.data());
  std::vector<double> x(dim);
  std::vector<double> a(dim);
  std::vector<double> b(dim);
  (void)wwr::wwrMemcpyAsync(x.data(), c, dim * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                            res.stream());
  (void)wwr::wwrMemcpyAsync(a.data(), y64.data(), dim * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                            res.stream());
  (void)wwr::wwrMemcpyAsync(b.data(), y32.data(), dim * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                            res.stream());
  (void)res.sync();
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    error += (b[i] - a[i]) * (b[i] - a[i]);
    norm += (a[i] - e_core * x[i]) * (a[i] - e_core * x[i]);
  }
  return std::sqrt(error / norm);
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  if (argc < 2) {
    std::println(std::cerr, "usage: {} CASES.json [n_roots] [max_iterations] [device]", argv[0]);
    return 2;
  }
  const std::int32_t n_roots = argc > 2 ? std::stoi(argv[2]) : 3;
  const std::int32_t max_iterations = argc > 3 ? std::stoi(argv[3]) : 100;
  const std::int32_t device = argc > 4 ? std::stoi(argv[4]) : 0;
  const auto doc = fb::read_json(argv[1]);
  if (!doc) {
    std::println(std::cerr, "{}", doc.error().message);
    return 1;
  }
  fb::DeviceResources res{device};

  for (const auto &entry : (*doc)["cases"]) {
    const auto n = entry["n_orbitals"].get<std::int32_t>();
    const auto alpha = fb::StringSet::create(n, entry["n_alpha"].get<std::int32_t>());
    const auto beta = fb::StringSet::create(n, entry["n_beta"].get<std::int32_t>());
    auto sigma = fb::CiSigma::create(res, *alpha, *beta);
    if (!sigma || !sigma->set_integrals(res, entry["e_core"].get<double>(),
                                        entry["h1"]["values"].get<std::vector<double>>(),
                                        entry["eri"]["values"].get<std::vector<double>>())) {
      std::println(std::cerr, "{}: CiSigma setup failed", entry["name"].get<std::string>());
      return 1;
    }
    const auto dim = static_cast<std::size_t>(sigma->size());
    fb::Buffers::device<double> diagonal(dim, res.device());
    fb::Buffers::device<double> y(dim, res.device());
    (void)sigma->diagonal(res, diagonal.data());

    // Any column will do for the timing: the cost does not depend on c.
    const std::int32_t reps = dim > 200'000 ? 5 : 20;
    const double fp64_ms =
        ms_per_column(res, reps, [&] { return sigma->apply(res, 1, diagonal.data(), y.data()); });
    const double fp32_ms = ms_per_column(
        res, reps, [&] { return sigma->apply_fp32(res, 1, diagonal.data(), y.data()); });
    std::println("{}  dim {}  sigma per column: FP64 {:.2f} ms, FP32 {:.2f} ms ({:.1f}x)",
                 entry["name"].get<std::string>(), dim, fp64_ms, fp32_ms, fp64_ms / fp32_ms);

    std::vector<double> reference;
    for (const double tolerance : {0.0, 1e-4, 1e-5, 1e-6, 1e-7}) {
      fb::CiSolverOptions options;
      options.fp32_tolerance = tolerance;
      options.davidson.max_iterations = max_iterations;
      const auto start = Clock::now();
      const auto roots =
          fb::ci_solve(res, *sigma, *alpha, *beta, diagonal.data(), n_roots, options);
      const double wall = seconds_since(start);
      if (!roots) {
        std::println("  fp32_tolerance {:.0e}: {}", tolerance, roots.error().message);
        continue;
      }
      if (reference.empty()) {
        reference = roots->energies;
        std::println(
            "  apply_fp32 on the ground state: {:.2e} of ||(H - E_core) c||",
            fp32_error(res, *sigma, entry["e_core"].get<double>(), roots->vectors.data(), dim));
      }
      double worst = 0.0;
      for (std::size_t r = 0; r < reference.size(); ++r) {
        worst = std::max(worst, std::abs(roots->energies[r] - reference[r]));
      }
      std::println("  fp32_tolerance {:7.0e}: {:7.2f} s  iterations FP32 {:3} + FP64 {:3}  "
                   "E0 {:.10f}  max |dE| vs FP64-only {:.1e}",
                   tolerance, wall, roots->fp32_iterations, roots->davidson.iterations,
                   roots->energies[0], worst);
    }
  }
  return 0;
}
