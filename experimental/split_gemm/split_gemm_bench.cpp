// Two probes, in one run, both cards.
//
//   free_breakfast_split_gemm_bench [reps = 5] [budget_gib = 6]
//
// **Probe 1, the fp64 ceiling.** Square Dgemm and Sgemm, n = 1024..8192. The
// digest's own shapes all land at 423-438 GF/s on both cards, and tripling m
// moved that by 1%, which says the fp64 ALU rather than the shape -- but a
// square GEMM has far better reuse, so this is the clean test. If square
// Dgemm also tops out there, 425 GF/s IS the card's fp64 ceiling and no
// fp64 work can help the digest. The Sgemm column is the headroom probe 2
// tries to spend.
//
// **Probe 2, what fp32 products can buy.** The digest's shape through
// gemm_split_fp32 at terms 1 (plain fp32) and 3 (double-single), swept over
// k_chunk, against Dgemm. fp64-equivalent GF/s (2mnk either way) and the max
// relative error against Dgemm's answer.
//
// Read the two columns together: a row is only interesting if its error is
// inside what the caller needs. For SC-NEVPT2's E_corr the gate is 1e-7 and
// the demo measured ~640x of margin at the int8 scheme's coarsest working
// truncation; `nevpt2_rdm_gpu`'s dm3 check is 1e-12 of max|dm3|, which is a
// much harder target and probably out of reach here.
//
// Not a test: read the numbers, do not gate on them (bench/README.md). Hold
// the whole card first, or another job on it is in the number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.split_gemm;

namespace fb = free_breakfast;

namespace {

void check(const bool ok, const std::string_view what) {
  if (!ok) {
    std::println(std::cerr, "split_gemm_bench: {} failed", what);
    std::exit(1);
  }
}

void check_gpu(const wwr::wwrError_t err, const std::string_view what) {
  check(err == wwr::wwrSuccess, what);
}

void check_blas(const wwr::wwrblasStatus_t status, const std::string_view what) {
  check(status == wwr::WWRBLAS_STATUS_SUCCESS, what);
}

void check_status(const fb::Status &status, const std::string_view what) {
  if (!status) {
    std::println(std::cerr, "split_gemm_bench: {} failed: {}", what, status.error().message);
    std::exit(1);
  }
}

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

// A varied repeating pattern, uploaded a chunk at a time: the operands are
// gigabytes and a host copy of one would dwarf the GEMM.
template <typename T>
void fill_device(fb::DeviceResources &res, T *const dst, const std::size_t count,
                 const double scale) {
  constexpr std::size_t kChunk = 1UZ << 20;
  std::vector<T> chunk(std::min(kChunk, count));
  for (std::size_t i = 0; i < chunk.size(); ++i) {
    chunk[i] = static_cast<T>(scale * std::sin(static_cast<double>(i) * 0.37));
  }
  for (std::size_t done = 0; done < count; done += chunk.size()) {
    const std::size_t take = std::min(chunk.size(), count - done);
    check_gpu(wwr::wwrMemcpyAsync(dst + done, chunk.data(), take * sizeof(T),
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

double max_relative(const std::vector<double> &reference, const std::vector<double> &test) {
  double max_abs = 0.0;
  double max_diff = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(reference[i]));
    max_diff = std::max(max_diff, std::abs(reference[i] - test[i]));
  }
  return max_abs > 0.0 ? max_diff / max_abs : max_diff;
}

std::int32_t parse_arg(const char *arg, const std::int32_t fallback) {
  std::int32_t value = fallback;
  if (arg != nullptr) {
    const std::string_view s{arg};
    std::from_chars(s.data(), s.data() + s.size(), value);
  }
  return value;
}

double gflops(const std::int64_t m, const std::int64_t n, const std::int64_t k, const float ms) {
  const double flops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  return flops / (static_cast<double>(ms) * 1e6);
}

// Probe 1: is ~425 GF/s the card's fp64 ceiling, or the digest shape's?
void ceiling_probe(fb::DeviceResources &res, const std::int32_t reps, const std::size_t budget) {
  std::println("\n== probe 1: square GEMM, the card's own ceiling ==");
  std::println("{:>6} {:>14} {:>14} {:>10}", "n", "dgemm GF/s", "sgemm GF/s", "fp32/fp64");
  for (std::int64_t n = 1024; n <= 8192; n *= 2) {
    const auto count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if ((3 * count * sizeof(double)) + (3 * count * sizeof(float)) > budget) {
      std::println("{:>6}   skipped: over the budget", n);
      continue;
    }
    fb::Buffers::device<double> a{count, res.device()};
    fb::Buffers::device<double> b{count, res.device()};
    fb::Buffers::device<double> c{count, res.device()};
    fb::Buffers::device<float> fa{count, res.device()};
    fb::Buffers::device<float> fb_{count, res.device()};
    fb::Buffers::device<float> fc{count, res.device()};
    const double scale = 1.0 / std::sqrt(static_cast<double>(n));
    fill_device(res, a.data(), count, scale);
    fill_device(res, b.data(), count, scale);
    fill_device(res, fa.data(), count, scale);
    fill_device(res, fb_.data(), count, scale);

    constexpr double kOne = 1.0;
    constexpr double kZero = 0.0;
    constexpr float kOneF = 1.0F;
    constexpr float kZeroF = 0.0F;
    const float d_ms = time_call(res, reps, [&] {
      check_blas(wwr::gemm<double, std::int64_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                 wwr::WWRBLAS_OP_N, n, n, n, &kOne, a.data(), n,
                                                 b.data(), n, &kZero, c.data(), n),
                 "square dgemm");
    });
    const float s_ms = time_call(res, reps, [&] {
      check_blas(wwr::gemm<float, std::int64_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                wwr::WWRBLAS_OP_N, n, n, n, &kOneF, fa.data(), n,
                                                fb_.data(), n, &kZeroF, fc.data(), n),
                 "square sgemm");
    });
    const double d_gf = gflops(n, n, n, d_ms);
    const double s_gf = gflops(n, n, n, s_ms);
    std::println("{:>6} {:>14.1f} {:>14.1f} {:>10.1f}", n, d_gf, s_gf, s_gf / d_gf);
  }
}

// One digest shape: n_act, the tile width, and whether the three orders fuse.
struct Shape {
  std::int64_t n_act;
  std::int64_t width;
  bool fused;
};

// Probe 2: what the fp32 split buys on the digest's own shape, and at what error.
void split_probe(fb::DeviceResources &res, const std::int32_t reps, const std::size_t budget) {
  constexpr std::array kShapes{
      Shape{10, 21168, false},
      Shape{12, 10673, false},
      Shape{12, 10673, true},
  };
  std::println("\n== probe 2: the digest shape through fp32 products ==");
  std::println("(fp64-equivalent GF/s; 'rel' is the max elementwise error against Dgemm)");
  std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>8} {:>12} {:>8} {:>10}", "nact",
               "width", "fused", "m", "n", "k", "terms", "k_chunk", "split GF/s", "vs d", "rel");

  for (const Shape shape : kShapes) {
    const std::int64_t n_act2 = shape.n_act * shape.n_act;
    const std::int64_t m = shape.fused ? 3 * n_act2 : n_act2;
    const std::int64_t n = n_act2 * n_act2;
    const std::int64_t k = shape.width;
    const auto a_count = static_cast<std::size_t>(m * k);
    const auto b_count = static_cast<std::size_t>(k * n);
    const auto c_count = static_cast<std::size_t>(m * n);
    const auto scratch_floats = static_cast<std::size_t>(fb::split_scratch_floats(m, n, k));
    const std::size_t bytes =
        ((a_count + b_count + (2 * c_count)) * sizeof(double)) + (scratch_floats * sizeof(float));
    if (bytes > budget) {
      std::println("{:>4} {:>6} {:>6}   skipped: {} GiB over the budget", shape.n_act, shape.width,
                   shape.fused, bytes >> 30);
      continue;
    }

    fb::Buffers::device<double> a{a_count, res.device()};
    fb::Buffers::device<double> b{b_count, res.device()};
    fb::Buffers::device<double> c_ref{c_count, res.device()};
    fb::Buffers::device<double> c_split{c_count, res.device()};
    fb::Buffers::device<float> scratch{scratch_floats, res.device()};
    const double scale = 1.0 / std::sqrt(static_cast<double>(k));
    fill_device(res, a.data(), a_count, scale);
    fill_device(res, b.data(), b_count, scale);

    const fb::SplitScratch planes{
        .a_hi = scratch.data(),
        .a_lo = scratch.data() + (m * k),
        .b_hi = scratch.data() + (2 * m * k),
        .b_lo = scratch.data() + (2 * m * k) + (k * n),
        .c32 = scratch.data() + (2 * ((m * k) + (k * n))),
    };

    constexpr double kOne = 1.0;
    constexpr double kZero = 0.0;
    const auto dgemm = [&] {
      check_blas(wwr::gemm<double, std::int64_t>(res.blas().get(), wwr::WWRBLAS_OP_T,
                                                 wwr::WWRBLAS_OP_N, m, n, k, &kOne, a.data(), k,
                                                 b.data(), k, &kZero, c_ref.data(), m),
                 "digest dgemm");
    };
    const float d_ms = time_call(res, reps, dgemm);
    const double d_gf = gflops(m, n, k, d_ms);
    std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>8} {:>12.1f} {:>8} {:>10}",
                 shape.n_act, shape.width, shape.fused, m, n, k, "-", "dgemm", d_gf, "1.00", "-");
    const std::vector<double> reference = download(res, c_ref.data(), c_count);

    for (const std::int32_t terms : {1, 3}) {
      for (const std::int64_t k_chunk :
           {k, std::int64_t{4096}, std::int64_t{1024}, std::int64_t{256}, std::int64_t{64}}) {
        const auto split = [&] {
          check_status(fb::gemm_split_fp32(res, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, m, n, k,
                                           a.data(), k, b.data(), k, c_split.data(), m, k_chunk,
                                           terms, planes),
                       "gemm_split_fp32");
        };
        // C accumulates, so zero it for the one call the error is read from.
        check_gpu(wwr::wwrMemsetAsync(c_split.data(), 0, c_count * sizeof(double), res.stream()),
                  "zeroing C");
        split();
        check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the accuracy call");
        const double rel = max_relative(reference, download(res, c_split.data(), c_count));

        const float s_ms = time_call(res, reps, split);
        const double s_gf = gflops(m, n, k, s_ms);
        std::println("{:>4} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>8} {:>12.1f} {:>8.2f} {:>10.2e}",
                     shape.n_act, shape.width, shape.fused, m, n, k, terms, k_chunk, s_gf,
                     s_gf / d_gf, rel);
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::int32_t reps = parse_arg(argc > 1 ? argv[1] : nullptr, 5);
  const std::int32_t budget_gib = parse_arg(argc > 2 ? argv[2] : nullptr, 6);
  check(reps > 0, "reps > 0");
  check(budget_gib > 0, "budget_gib > 0");
  const auto budget = static_cast<std::size_t>(budget_gib) * (1UZ << 30);

  fb::DeviceResources res;
  std::println("reps={} budget={} GiB", reps, budget_gib);
  ceiling_probe(res, reps, budget);
  split_probe(res, reps, budget);
  return 0;
}
