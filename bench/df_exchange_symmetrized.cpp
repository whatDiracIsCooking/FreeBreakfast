// df_exchange_symmetrized against the block two-factor df_exchange (#595's
// path) on the same K: count densities D_d = L R_d^T sharing L, rank n_occ,
// symmetrized. The new call builds K[(D_d + D_d^T) / 2] from L once and the
// R_d; the old one K[D_d + D_d^T] from L_d = [R_d L] and R_d = [L R_d], rank
// 2 n_occ. The timing depends only on the shapes, so B and the factors are
// random, from a fixed seed; the two K are compared as a check.
//
//   free_breakfast_df_exchange_symmetrized [reps = 5] [nao naux nocc count]...
//
// Without shapes it runs a ladder of molecule-like ones (naux ~ 4.5 nao),
// the larger ones at a count that keeps #595's scratch, 4 naux n_occ count
// nao doubles, on a 10 GB card. On device 0; prints, per shape, the
// best of reps after one warm-up for each path, and their ratio. Hold the
// card (devtools/lib.sh's with_gpu_card) while it runs.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.integrals.eri.density_fitting.jk;

namespace {

namespace fb = free_breakfast;

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(std::string_view what, const fb::Error &error) {
  std::println(std::cerr, "{}: {}", what, error.message);
  std::exit(1);
}

struct Shape {
  std::int32_t nao;
  std::int32_t naux;
  std::int32_t nocc;
  std::int32_t count;
};

std::vector<double> random_values(const std::size_t count, const std::uint64_t seed) {
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{-0.5, 0.5};
  std::vector<double> a(count);
  std::ranges::generate(a, [&] { return uniform(engine); });
  return a;
}

fb::Buffers::device<double> on_device(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  if (wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                          wwr::wwrMemcpyHostToDevice, res.stream()) != wwr::wwrSuccess) {
    std::println(std::cerr, "upload failed");
    std::exit(1);
  }
  return d;
}

// The best of reps seconds of call(), after one warm-up, each synchronized.
template <typename Call>
double best_of(fb::DeviceResources &res, const std::int32_t reps, Call call) {
  double best = std::numeric_limits<double>::infinity();
  for (std::int32_t r = 0; r <= reps; ++r) {
    const auto start = Clock::now();
    if (const fb::Status s = call(); !s) {
      fail("df_exchange", s.error());
    }
    if (const fb::Status s = res.sync(); !s) {
      fail("sync", s.error());
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (r > 0) {
      best = std::min(best, seconds);
    }
  }
  return best;
}

void run(fb::DeviceResources &res, const Shape &s, const std::int32_t reps) {
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto nn = nao * nao;
  const std::size_t factor = nao * static_cast<std::size_t>(s.nocc);
  const auto count = static_cast<std::size_t>(s.count);
  const std::vector<double> left = random_values(factor, 1);
  const std::vector<double> right = random_values(count * factor, 2);
  std::vector<double> left2(2 * count * factor);
  std::vector<double> right2(left2.size());
  for (std::size_t d = 0; d < count; ++d) {
    const auto rd = right.begin() + static_cast<std::ptrdiff_t>(d * factor);
    const auto ld = left2.begin() + static_cast<std::ptrdiff_t>(2 * d * factor);
    const auto sd = right2.begin() + static_cast<std::ptrdiff_t>(2 * d * factor);
    std::copy_n(rd, factor, ld);
    std::ranges::copy(left, ld + static_cast<std::ptrdiff_t>(factor));
    std::ranges::copy(left, sd);
    std::copy_n(rd, factor, sd + static_cast<std::ptrdiff_t>(factor));
  }
  const auto b = on_device(res, random_values(static_cast<std::size_t>(s.naux) * nn, 3));
  const auto d_left = on_device(res, left);
  const auto d_right = on_device(res, right);
  const auto d_left2 = on_device(res, left2);
  const auto d_right2 = on_device(res, right2);
  fb::Buffers::device<double> k_old(count * nn, res.device());
  fb::Buffers::device<double> k_new(count * nn, res.device());

  const double old_s = best_of(res, reps, [&] {
    return fb::df_exchange(res, s.nao, s.naux, 2 * s.nocc, s.count, b.data(), d_left2.data(),
                           d_right2.data(), k_old.data());
  });
  const double new_s = best_of(res, reps, [&] {
    return fb::df_exchange_symmetrized(res, s.nao, s.naux, s.nocc, s.count, b.data(), d_left.data(),
                                       d_right.data(), k_new.data());
  });

  std::vector<double> ko(count * nn);
  std::vector<double> kn(count * nn);
  (void)wwr::wwrMemcpyAsync(ko.data(), k_old.data(), sizeof(double) * ko.size(),
                            wwr::wwrMemcpyDeviceToHost, res.stream());
  (void)wwr::wwrMemcpyAsync(kn.data(), k_new.data(), sizeof(double) * kn.size(),
                            wwr::wwrMemcpyDeviceToHost, res.stream());
  if (const fb::Status st = res.sync(); !st) {
    fail("download", st.error());
  }
  double worst = 0.0;
  double scale = 0.0;
  for (std::size_t i = 0; i < ko.size(); ++i) {
    worst = std::max(worst, std::abs((2.0 * kn[i]) - ko[i]));
    scale = std::max(scale, std::abs(ko[i]));
  }
  std::println("nao {:5} naux {:5} nocc {:4} count {:4}  #595 {:9.3f} ms  symmetrized {:9.3f} ms"
               "  {:5.2f}x  |2K - K595|/max {:.1e}",
               s.nao, s.naux, s.nocc, s.count, 1e3 * old_s, 1e3 * new_s, old_s / new_s,
               worst / scale);
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::vector<std::string_view> args(argv + 1, argv + argc);
  const std::int32_t reps = args.empty() ? 5 : std::stoi(std::string{args[0]});
  std::vector<Shape> shapes;
  if (args.size() > 1) {
    for (std::size_t i = 1; i + 3 < args.size(); i += 4) {
      shapes.push_back({std::stoi(std::string{args[i]}), std::stoi(std::string{args[i + 1]}),
                        std::stoi(std::string{args[i + 2]}), std::stoi(std::string{args[i + 3]})});
    }
  } else {
    // Water in cc-pVDZ (CAS(4,4): 7 occupied, 9 coordinates) upward.
    shapes = {{24, 116, 7, 9},    {58, 260, 9, 9},     {120, 540, 15, 27},
              {200, 900, 25, 24}, {250, 1100, 41, 12}, {300, 1350, 40, 8}};
  }
  fb::DeviceResources res{0};
  for (const Shape &s : shapes) {
    run(res, s, reps);
  }
  return 0;
}
