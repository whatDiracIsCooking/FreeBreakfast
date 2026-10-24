// CiSigma's apply against apply_fp32, one column at a time, in the full and
// the symmetric layout with the device workspace each takes, and ci_solve with
// its single-precision stage off and at several fp32_tolerance values, wall
// clock, over the active spaces of a direct_spin1-schema JSON file
// (tools/emit_ci_bench_integrals.py writes N2's CAS(10,10) and CAS(12,12)).
// ci_solve runs a singlet in the symmetric layout; it is also run held to the
// full one, as it was before that layout, FP64-only and at 1e-6.
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

// CiSigma without the layout parameter, so ci_solve takes the full layout.
struct FullLayout {
  fb::CiSigma &sigma;
  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    return sigma.apply(res, k, x, y);
  }
  fb::Status apply_fp32(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    return sigma.apply_fp32(res, k, x, y);
  }
};

// The MiB of device workspace one column of @p apply grows a fresh
// DeviceResources to.
template <typename Apply>
double workspace_mib(std::int32_t device, Apply &&apply) {
  fb::DeviceResources fresh{device};
  (void)apply(fresh);
  (void)fresh.sync();
  return static_cast<double>(fresh.device_workspace(0).num_elements()) / (1024.0 * 1024.0);
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
    std::println("{}  dim {}, symmetric {}", entry["name"].get<std::string>(), dim,
                 sigma->size(fb::CiLayout::symmetric));

    // Any column will do for the timing: the cost does not depend on c. In
    // the symmetric layout the first size(symmetric) elements of diag(H)
    // stand in for a packed vector.
    const std::int32_t reps = dim > 200'000 ? 5 : 20;
    for (const fb::CiLayout layout : {fb::CiLayout::full, fb::CiLayout::symmetric}) {
      const auto fp64 = [&](fb::DeviceResources &r) {
        return sigma->apply(r, 1, diagonal.data(), y.data(), layout);
      };
      const auto fp32 = [&](fb::DeviceResources &r) {
        return sigma->apply_fp32(r, 1, diagonal.data(), y.data(), layout);
      };
      const double fp64_ms = ms_per_column(res, reps, [&] { return fp64(res); });
      const double fp32_ms = ms_per_column(res, reps, [&] { return fp32(res); });
      std::println("  {:9} sigma per column: FP64 {:6.2f} ms, FP32 {:5.2f} ms; workspace FP64 "
                   "{:6.1f} MiB, FP32 {:6.1f} MiB",
                   layout == fb::CiLayout::full ? "full" : "symmetric", fp64_ms, fp32_ms,
                   workspace_mib(device, fp64), workspace_mib(device, fp32));
    }

    std::vector<double> reference;
    const auto solve = [&](auto &op, const char *label, double tolerance) {
      fb::CiSolverOptions options;
      options.fp32_tolerance = tolerance;
      options.davidson.max_iterations = max_iterations;
      const auto start = Clock::now();
      const auto roots = fb::ci_solve(res, op, *alpha, *beta, diagonal.data(), n_roots, options);
      const double wall = seconds_since(start);
      if (!roots) {
        std::println("  {:9} fp32_tolerance {:.0e}: {}", label, tolerance, roots.error().message);
        return;
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
      std::println("  {:9} fp32_tolerance {:7.0e}: {:7.2f} s  iterations FP32 {:3} + FP64 {:3}  "
                   "E0 {:.10f}  max |dE| vs full FP64-only {:.1e}",
                   label, tolerance, wall, roots->fp32_iterations, roots->davidson.iterations,
                   roots->energies[0], worst);
    };
    FullLayout full{*sigma};
    for (const double tolerance : {0.0, 1e-6}) {
      solve(full, "full", tolerance);
    }
    for (const double tolerance : {0.0, 1e-4, 1e-5, 1e-6, 1e-7}) {
      solve(*sigma, "symmetric", tolerance);
    }
  }
  return 0;
}
