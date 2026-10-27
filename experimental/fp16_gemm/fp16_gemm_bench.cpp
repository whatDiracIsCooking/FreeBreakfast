// gemm_fp16 under both accumulators against fp32 Sgemm, square n x n x n,
// for n = 512, 1024, ... up to max_n. A half GEMM well above Sgemm's rate is
// the card's half (tensor-core / WMMA) path at work; one level with it means
// the library supports the call but not fast.
//
//   free_breakfast_fp16_gemm_bench [max_n = 8192] [reps = 10]
//
// Each timing is the median of `reps` calls, timed with GPU events around the
// call alone, after one warm-up call. TFLOP/s counts 2 n^3 flops.
//
// Not a test: read the numbers, do not gate on them (bench/README.md). Bound
// the container first, or another job on the GPU is in the number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.fp16_gemm;
import wwr.fp16;

namespace fb = free_breakfast;

namespace {

// Aborts on a failed call: a benchmark has nothing to recover to, and a
// timing taken past a failure would be meaningless.
void check(const bool ok, const std::string_view what) {
  if (!ok) {
    std::println(std::cerr, "fp16_gemm_bench: {} failed", what);
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

std::int32_t parse_arg(const char *arg, const std::int32_t fallback) {
  std::int32_t value = fallback;
  if (arg != nullptr) {
    const std::string_view s{arg};
    std::from_chars(s.data(), s.data() + s.size(), value);
  }
  return value;
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::int32_t max_n = parse_arg(argc > 1 ? argv[1] : nullptr, 8192);
  const std::int32_t reps = parse_arg(argc > 2 ? argv[2] : nullptr, 10);
  check(reps > 0, "reps > 0");

  fb::DeviceResources res;
  std::println("{:>6} {:>12} {:>12} {:>12}   TFLOP/s", "n", "hgemm", "gemmex-f32", "sgemm");
  for (std::int32_t n = 512; n <= max_n; n *= 2) {
    const auto count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    // Constant inputs: the timing does not depend on the values, and 1/n keeps
    // every sum of n products at 1/n, far inside half's range.
    const std::vector<wwr::wwrHalf> h(count, wwr::wwrFloat2Half(1.0F / static_cast<float>(n)));
    const std::vector<float> f(count, 1.0F / static_cast<float>(n));
    fb::Buffers::device<wwr::wwrHalf> ha{count, res.device()};
    fb::Buffers::device<wwr::wwrHalf> hb{count, res.device()};
    fb::Buffers::device<wwr::wwrHalf> hc{count, res.device()};
    fb::Buffers::device<float> fa{count, res.device()};
    fb::Buffers::device<float> fb_{count, res.device()};
    fb::Buffers::device<float> fc{count, res.device()};
    for (auto *d : {&ha, &hb}) {
      check_gpu(wwr::wwrMemcpyAsync(d->data(), h.data(), count * sizeof(wwr::wwrHalf),
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                "uploading half inputs");
    }
    for (auto *d : {&fa, &fb_}) {
      check_gpu(wwr::wwrMemcpyAsync(d->data(), f.data(), count * sizeof(float),
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                "uploading float inputs");
    }

    const auto half_call = [&](const fb::Fp16Accumulate acc) {
      return [&res, &ha, &hb, &hc, n, acc] {
        check_blas(fb::gemm_fp16(res, acc, n, n, n, ha.data(), hb.data(), hc.data()), "gemm_fp16");
      };
    };
    const float hgemm = time_call(res, reps, half_call(fb::Fp16Accumulate::kHalf));
    const float gemmex = time_call(res, reps, half_call(fb::Fp16Accumulate::kFloat));
    const float one = 1.0F;
    const float zero = 0.0F;
    const float sgemm = time_call(res, reps, [&] {
      check_blas(wwr::gemm<float, std::int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                wwr::WWRBLAS_OP_N, n, n, n, &one, fa.data(), n,
                                                fb_.data(), n, &zero, fc.data(), n),
                 "sgemm");
    });

    const double flops =
        2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    const auto tflops = [flops](const float ms) { return flops / (static_cast<double>(ms) * 1e9); };
    std::println("{:>6} {:>12.2f} {:>12.2f} {:>12.2f}", n, tflops(hgemm), tflops(gemmex),
                 tflops(sgemm));
  }
  return 0;
}
