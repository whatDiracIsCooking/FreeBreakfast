// recipes/trsm against the BLAS library's trsm (cuBLAS's or rocBLAS's, through
// wwr::trsm) on the same A and B, in one process (#1782). The timing depends
// only on the shape, so A is a random well-conditioned triangle and B random,
// from a fixed seed; the two X are compared as a check.
//
//   free_breakfast_trsm_vendor [reps = 3] [SFOD m n]...
//
// SFOD is four letters: side L or R, fill L or U, op N or T, diag N or U.
// Without shapes it runs a built-in ladder: the density-fitting solve
// (LLNN, m = naux, n = nao^2) up to a B of 3.2 GB, the two solves of
// cholesky_solve and of lu_solve at a few right-hand sides and at many, and
// the Cholesky factor's panel (RLTN, n = 64).
//
// On device 0; prints, per shape, the best of reps after one warm-up for the
// recipe and for the library, their ratio, and (left side only) the time of
// the recipe's gemms alone, replayed in its order: what is left of the
// recipe's time is its diagonal-block kernel. B is restored from a device copy
// before each solve, outside the timing, so a shape takes A and two B on the
// card, and on ROCm whatever rocBLAS takes besides. Hold the card
// (devtools/lib.sh's with_gpu_card) while it runs.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes.trsm;

namespace {

namespace fb = free_breakfast;

// The diagonal blocks' order in recipes/trsm (kTrsmBlock, in its bridge
// header, which only the recipe sees): the inner dimension of the replayed
// gemms.
constexpr std::int32_t kBlock = 64;

// The workspaces this benchmark's DeviceResources hold, sized at
// construction (src/core/resources/README.md): the recipe borrows nothing.
constexpr fb::MemoryFootprint kWorkspaces{.device = {.workspace = std::size_t{64} << 20},
                                          .pinned = {.workspace = std::size_t{64} << 20}};

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(std::string_view what, std::string_view why) {
  std::println(std::cerr, "{}: {}", what, why);
  std::exit(1);
}

void check(std::string_view what, const fb::Status &s) {
  if (!s) {
    fail(what, s.error().message);
  }
}

struct Shape {
  std::string name;
  wwr::wwrblasSideMode_t side;
  wwr::wwrblasFillMode_t fill;
  wwr::wwrblasOperation_t op;
  wwr::wwrblasDiagType_t diag;
  std::int32_t m;
  std::int32_t n;
};

Shape parse(std::string_view sfod, const std::int32_t m, const std::int32_t n) {
  if (sfod.size() != 4) {
    fail(sfod, "expected four letters: side L|R, fill L|U, op N|T, diag N|U");
  }
  return {.name = std::string{sfod},
          .side = sfod[0] == 'L' ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT,
          .fill = sfod[1] == 'L' ? wwr::WWRBLAS_FILL_MODE_LOWER : wwr::WWRBLAS_FILL_MODE_UPPER,
          .op = sfod[2] == 'N' ? wwr::WWRBLAS_OP_N : wwr::WWRBLAS_OP_T,
          .diag = sfod[3] == 'N' ? wwr::WWRBLAS_DIAG_NON_UNIT : wwr::WWRBLAS_DIAG_UNIT,
          .m = m,
          .n = n};
}

// A full k x k matrix whose either triangle is a well-conditioned factor:
// a diagonal in [1, 2) and off-diagonal entries of order 1/k, so no solve
// grows.
std::vector<double> random_triangle(const std::int32_t k) {
  std::mt19937_64 engine{1};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  const auto order = static_cast<std::size_t>(k);
  std::vector<double> a(order * order);
  for (std::size_t j = 0; j < order; ++j) {
    for (std::size_t i = 0; i < order; ++i) {
      a[i + (j * order)] = i == j ? 1.5 + uniform(engine) : uniform(engine) / k;
    }
  }
  return a;
}

std::vector<double> random_values(const std::size_t count) {
  std::mt19937_64 engine{2};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  std::vector<double> a(count);
  std::ranges::generate(a, [&] { return uniform(engine); });
  return a;
}

void copy(fb::DeviceResources &res, void *to, const void *from, const std::size_t bytes,
          const wwr::wwrMemcpyKind kind) {
  if (wwr::wwrMemcpyAsync(to, from, bytes, kind, res.stream()) != wwr::wwrSuccess) {
    fail("copy", "wwrMemcpyAsync failed");
  }
}

// The gemms recipes/trsm makes on the left side, in its order and with its
// shapes, and nothing else: its time less its diagonal-block kernel's.
fb::Status replay_gemms(fb::DeviceResources &res, const Shape &s, const double *a, double *b) {
  const bool trans = s.op != wwr::WWRBLAS_OP_N;
  const bool lower = (s.fill == wwr::WWRBLAS_FILL_MODE_LOWER) != trans;
  const std::int32_t k = s.m;
  const auto ld = static_cast<std::int64_t>(k);
  const std::int32_t blocks = fb::math::idivup(k, kBlock);
  for (std::int32_t step = 0; step < blocks; ++step) {
    const std::int32_t block = lower ? step : blocks - 1 - step;
    const std::int32_t k0 = block * kBlock;
    const std::int32_t jb = std::min(kBlock, k - k0);
    const std::int32_t r0 = lower ? k0 + jb : 0;
    const std::int32_t rows = lower ? k - r0 : k0;
    if (rows == 0) {
      continue;
    }
    const double *const off = trans ? a + k0 + (r0 * ld) : a + r0 + (k0 * ld);
    if (fb::Status st = fb::gpu_status(wwr::gemm<double, std::int32_t>(
            res.blas().get(), s.op, wwr::WWRBLAS_OP_N, rows, s.n, jb, &fb::kNegOne<double>, off, k,
            b + k0, k, &fb::kOne<double>, b + r0, k));
        !st) {
      return st;
    }
  }
  return {};
}

// The best of reps seconds of call(), after one warm-up, each on a B restored
// from b0 and synchronized on both sides of the clock.
template <typename Call>
double best_of(fb::DeviceResources &res, const std::int32_t reps, double *b, const double *b0,
               const std::size_t bytes, Call call) {
  double best = std::numeric_limits<double>::infinity();
  for (std::int32_t r = 0; r <= reps; ++r) {
    copy(res, b, b0, bytes, wwr::wwrMemcpyDeviceToDevice);
    check("sync", res.sync());
    const auto start = Clock::now();
    check("solve", call());
    check("sync", res.sync());
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (r > 0) {
      best = std::min(best, seconds);
    }
  }
  return best;
}

void run(fb::DeviceResources &res, const Shape &s, const std::int32_t reps) {
  const bool left = s.side == wwr::WWRBLAS_SIDE_LEFT;
  const std::int32_t k = left ? s.m : s.n;
  const std::size_t count = static_cast<std::size_t>(s.m) * static_cast<std::size_t>(s.n);
  const std::size_t bytes = sizeof(double) * count;
  const std::vector<double> host_a = random_triangle(k);
  const std::vector<double> host_b = random_values(count);
  fb::Buffers::device<double> a(host_a.size(), res.device());
  fb::Buffers::device<double> b0(count, res.device());
  fb::Buffers::device<double> b(count, res.device());
  copy(res, a.data(), host_a.data(), sizeof(double) * host_a.size(), wwr::wwrMemcpyHostToDevice);
  copy(res, b0.data(), host_b.data(), bytes, wwr::wwrMemcpyHostToDevice);
  check("upload", res.sync());

  std::vector<double> x_recipe(count);
  std::vector<double> x_vendor(count);
  const auto download = [&](std::vector<double> &x) {
    copy(res, x.data(), b.data(), bytes, wwr::wwrMemcpyDeviceToHost);
    check("download", res.sync());
  };

  const double recipe_s = best_of(res, reps, b.data(), b0.data(), bytes, [&] {
    return fb::trsm<double>(res, s.side, s.fill, s.op, s.diag, s.m, s.n, 1.0, a.data(), k, b.data(),
                            s.m);
  });
  download(x_recipe);
  const double vendor_s = best_of(res, reps, b.data(), b0.data(), bytes, [&] {
    return fb::gpu_status(wwr::trsm<double, std::int32_t>(res.blas().get(), s.side, s.fill, s.op,
                                                          s.diag, s.m, s.n, &fb::kOne<double>,
                                                          a.data(), k, b.data(), s.m));
  });
  download(x_vendor);
  const double gemm_s = left ? best_of(res, reps, b.data(), b0.data(), bytes,
                                       [&] { return replay_gemms(res, s, a.data(), b.data()); })
                             : std::numeric_limits<double>::quiet_NaN();

  double worst = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    worst = std::max(worst, std::abs(x_recipe[i] - x_vendor[i]));
    scale = std::max(scale, std::abs(x_vendor[i]));
  }
  std::println("{} m {:6} n {:7}  recipe {:10.3f} ms  library {:10.3f} ms  {:5.2f}x"
               "  recipe's gemms {:10.3f} ms  |dX|/max {:.1e}",
               s.name, s.m, s.n, 1e3 * recipe_s, 1e3 * vendor_s, recipe_s / vendor_s, 1e3 * gemm_s,
               worst / scale);
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  const std::int32_t reps = args.empty() ? 3 : std::stoi(std::string{args[0]});
  std::vector<Shape> shapes;
  if (args.size() > 1) {
    for (std::size_t i = 1; i + 2 < args.size(); i += 3) {
      shapes.push_back(
          parse(args[i], std::stoi(std::string{args[i + 1]}), std::stoi(std::string{args[i + 2]})));
    }
  } else {
    // The DF fit, naux ~ 4.5 nao: water in cc-pVDZ upward, then two orders
    // that are multiples of 128 (rocBLAS's other path).
    for (const auto &[naux, nao] :
         std::initializer_list<std::pair<std::int32_t, std::int32_t>>{{116, 24},
                                                                      {540, 120},
                                                                      {1100, 250},
                                                                      {1350, 300},
                                                                      {2000, 440},
                                                                      {1280, 300},
                                                                      {2048, 440}}) {
      shapes.push_back(parse("LLNN", naux, nao * nao));
    }
    // cholesky_solve's two solves and lu_solve's two, at 1, 32 and 1024
    // right-hand sides.
    for (const std::string_view sfod : {"LLNN", "LLTN", "LLNU", "LUNN"}) {
      for (const std::int32_t order : {512, 2000}) {
        for (const std::int32_t nrhs : {1, 32, 1024}) {
          shapes.push_back(parse(sfod, order, nrhs));
        }
      }
    }
    // The Cholesky factor's panel: the rows below a 64-column diagonal block.
    for (const std::int32_t rest : {500, 2000, 8000}) {
      shapes.push_back(parse("RLTN", rest, 64));
    }
  }
  fb::DeviceResources res{0, kWorkspaces};
  for (const Shape &s : shapes) {
    run(res, s, reps);
  }
  return 0;
}
