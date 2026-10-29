// gemm_lt against plain Dgemm on the shapes nevpt2::rdms' digest runs: a short
// m = n_act^2 (or 3 n_act^2, the fused digest's), a long n = n_act^4 and a long
// k = the tile width. Both are fp64 and both accumulate, as the digest does.
//
//   free_breakfast_blas_lt_bench [reps = 5] [budget_gib = 6]
//
// Each timing is the median of `reps` calls, GPU-event timed around the call
// alone, after one warm-up. The Lt call includes its own heuristic search, as a
// caller doing no algorithm caching would pay it. A shape whose operands exceed
// `budget_gib` is skipped and said to be.
//
// `algos` is the heuristic's returned count. **0 means the backend offers no
// fp64 algorithm for that shape** -- the question this bench exists to ask of
// hipBLASLt -- and the GFLOP/s column is then blank rather than zero.
//
// `max rel` compares the two results elementwise against max|C|, from a single
// call of each into a zeroed C. Lt picking a different algorithm is free to sum
// in a different order, so this is an agreement check, not a bitwise one.
//
// Not a test: read the numbers, do not gate on them (bench/README.md). Hold the
// whole card first, or another job on it is in the number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.blas_lt;

namespace fb = free_breakfast;

namespace {

void check(const bool ok, const std::string_view what) {
  if (!ok) {
    std::println(std::cerr, "blas_lt_bench: {} failed", what);
    std::exit(1);
  }
}

void check_gpu(const wwr::wwrError_t err, const std::string_view what) {
  check(err == wwr::wwrSuccess, what);
}

void check_blas(const wwr::wwrblasStatus_t status, const std::string_view what) {
  check(status == wwr::WWRBLAS_STATUS_SUCCESS, what);
}

// Median milliseconds of `call` over `reps` runs, after one warm-up.
template <typename F>
float time_call(fb::DeviceResources &res, const std::int32_t reps, F &&call) {
  const wwr::wwrStream_t stream = res.stream();
  const fb::Event start{res.dev_idx(), 0U};
  const fb::Event stop{res.dev_idx(), 0U};
  call();
  check_gpu(wwr::wwrStreamSynchronize(stream), "warm-up synchronize");
  std::vector<float> ms(static_cast<std::size_t>(reps));
  for (float &t : ms) {
    check_gpu(wwr::wwrEventRecord(start, stream), "recording start");
    call();
    check_gpu(wwr::wwrEventRecord(stop, stream), "recording stop");
    check_gpu(wwr::wwrEventSynchronize(stop), "synchronizing stop");
    check_gpu(wwr::wwrEventElapsedTime(&t, start, stop), "reading the elapsed time");
  }
  std::ranges::sort(ms);
  return ms[ms.size() / 2];
}

// Fill a device buffer with a repeating varied pattern, a chunk at a time: the
// operands are gigabytes, and a host copy of one would dwarf the GEMM.
void fill_device(fb::DeviceResources &res, double *const dst, const std::size_t count,
                 const double scale) {
  constexpr std::size_t kChunk = 1UZ << 20;
  std::vector<double> chunk(std::min(kChunk, count));
  for (std::size_t i = 0; i < chunk.size(); ++i) {
    chunk[i] = scale * std::sin(static_cast<double>(i) * 0.37);
  }
  for (std::size_t done = 0; done < count; done += chunk.size()) {
    const std::size_t take = std::min(chunk.size(), count - done);
    check_gpu(wwr::wwrMemcpyAsync(dst + done, chunk.data(), take * sizeof(double),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              "uploading an operand chunk");
  }
  check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the upload");
}

std::vector<double> download(fb::DeviceResources &res, const double *const src,
                             const std::size_t count) {
  std::vector<double> host(count);
  check_gpu(wwr::wwrMemcpyAsync(host.data(), src, count * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            "downloading C");
  check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the download");
  return host;
}

std::int32_t parse_arg(const char *arg, const std::int32_t fallback) {
  std::int32_t value = fallback;
  if (arg != nullptr) {
    const std::string_view s{arg};
    std::from_chars(s.data(), s.data() + s.size(), value);
  }
  return value;
}

// One digest shape: n_act, the tile width, and whether the three orders are
// fused into one call (m = 3 n_act^2 instead of n_act^2).
struct Shape {
  std::int64_t n_act;
  std::int64_t width;
  bool fused;
};

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::int32_t reps = parse_arg(argc > 1 ? argv[1] : nullptr, 5);
  const std::int32_t budget_gib = parse_arg(argc > 2 ? argv[2] : nullptr, 6);
  check(reps > 0, "reps > 0");
  check(budget_gib > 0, "budget_gib > 0");
  const auto budget = static_cast<std::size_t>(budget_gib) * (1UZ << 30);

  // The demo's measured cases: CAS(10,10) at 3 tiles, CAS(12,12) at 80 and 40
  // (docs/performance.md, "The digests"), each unfused and fused.
  constexpr std::array kShapes{
      Shape{10, 21168, false}, Shape{10, 21168, true},  Shape{12, 10673, false},
      Shape{12, 10673, true},  Shape{12, 21345, false}, Shape{12, 21345, true},
  };

  fb::DeviceResources res;
  fb::BlasLtHandle lt{};

  std::println("reps={} budget={} GiB   (m = n_act^2, or 3 n_act^2 fused;"
               " n = n_act^4; k = tile width)",
               reps, budget_gib);
  std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>12} {:>12} {:>8} {:>10}", "nact", "width",
               "fused", "m", "n", "k", "dgemm GF/s", "lt GF/s", "lt/dgemm", "max rel");

  for (const Shape shape : kShapes) {
    const std::int64_t n_act2 = shape.n_act * shape.n_act;
    const std::int64_t m = shape.fused ? 3 * n_act2 : n_act2;
    const std::int64_t n = n_act2 * n_act2;
    const std::int64_t k = shape.width;

    const auto a_count = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const auto b_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    const auto c_count = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    // A, B and two C's -- one per path, so they can be compared.
    const std::size_t bytes = (a_count + b_count + 2 * c_count) * sizeof(double);
    if (bytes > budget) {
      std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6}   skipped: {} GiB of operands exceeds"
                   " the budget",
                   shape.n_act, shape.width, shape.fused, m, n, k, bytes >> 30);
      continue;
    }

    fb::Buffers::device<double> a{a_count, res.device()};
    fb::Buffers::device<double> b{b_count, res.device()};
    fb::Buffers::device<double> c_blas{c_count, res.device()};
    fb::Buffers::device<double> c_lt{c_count, res.device()};
    // 1/sqrt(k) per operand keeps a sum of k products at O(1).
    const double scale = 1.0 / std::sqrt(static_cast<double>(k));
    fill_device(res, a.data(), a_count, scale);
    fill_device(res, b.data(), b_count, scale);

    // The digest's own call: C[m, n] += op(A)^T op(B), column-major, every ld
    // the tile width. rdms_digest issues it as wwr::gemm(OP_T, OP_N, ...).
    const double one = 1.0;
    const auto dgemm = [&] {
      check_blas(wwr::gemm<double, std::int64_t>(res.blas().get(), wwr::WWRBLAS_OP_T,
                                                 wwr::WWRBLAS_OP_N, m, n, k, &one, a.data(), k,
                                                 b.data(), k, &one, c_blas.data(), m),
                 "dgemm");
    };
    fb::LtResult lt_result{};
    const auto lt_call = [&] {
      lt_result = fb::gemm_lt(res, lt, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, m, n, k, a.data(), k,
                              b.data(), k, c_lt.data(), m, nullptr, 0);
    };

    // One call of each into a zeroed C, for the agreement check.
    for (auto *buffer : {&c_blas, &c_lt}) {
      check_gpu(wwr::wwrMemsetAsync(buffer->data(), 0, c_count * sizeof(double), res.stream()),
                "zeroing C");
    }
    dgemm();
    lt_call();
    check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the agreement calls");

    if (lt_result.algorithms == 0) {
      std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>12.1f} {:>12} {:>8} {:>10}", shape.n_act,
                   shape.width, shape.fused, m, n, k,
                   2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) /
                       (static_cast<double>(time_call(res, reps, dgemm)) * 1e6),
                   "no algo", "-", "-");
      continue;
    }
    check_blas(lt_result.status, "gemm_lt");

    const std::vector<double> host_blas = download(res, c_blas.data(), c_count);
    const std::vector<double> host_lt = download(res, c_lt.data(), c_count);
    double max_abs = 0.0;
    double max_diff = 0.0;
    for (std::size_t i = 0; i < c_count; ++i) {
      max_abs = std::max(max_abs, std::abs(host_blas[i]));
      max_diff = std::max(max_diff, std::abs(host_blas[i] - host_lt[i]));
    }
    const double max_rel = max_abs > 0.0 ? max_diff / max_abs : max_diff;

    const float blas_ms = time_call(res, reps, dgemm);
    const float lt_ms = time_call(res, reps, lt_call);
    const double flops =
        2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const auto gflops = [flops](const float ms) { return flops / (static_cast<double>(ms) * 1e6); };
    std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>12.1f} {:>12.1f} {:>8.2f} {:>10.2e}",
                 shape.n_act, shape.width, shape.fused, m, n, k, gflops(blas_ms), gflops(lt_ms),
                 static_cast<double>(blas_ms) / static_cast<double>(lt_ms), max_rel);
  }
  return 0;
}
