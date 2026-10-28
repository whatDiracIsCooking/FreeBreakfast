// OutputRing against the download it replaces -- a pageable vector, a stream
// sync and io.npy's write_npy, one array after another -- writing k arrays of
// n x n doubles, for n = 512, 1024, ... up to max_n.
//
//   free_breakfast_output_ring_bench [max_n = 8192] [k = 3] [reps = 5]
//
// Each timing is the median wall clock of `reps` runs, after one warm-up, from
// the first copy queued to the last file closed. The files go to the system
// temp dir, so the number is mostly the page cache, not the disk: it measures
// the overlap, which is what the ring is for.
//
// Not a test: read the numbers, do not gate on them (bench/README.md). Bound
// the container first, or another job on the GPU is in the number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.experimental.output_ring;

namespace fb = free_breakfast;

namespace {

// Aborts on a failed call: a benchmark has nothing to recover to.
void check(const fb::Status &status, const std::string_view what) {
  if (!status) {
    std::println(std::cerr, "output_ring_bench: {} failed: {}", what, status.error().message);
    std::exit(1);
  }
}

// Median milliseconds of `call` over `reps` runs, after one warm-up.
template <typename F>
double time_call(const std::int32_t reps, F &&call) {
  call();
  std::vector<double> ms(static_cast<std::size_t>(reps));
  for (double &t : ms) {
    const auto start = std::chrono::steady_clock::now();
    call();
    t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
  const std::int32_t k = parse_arg(argc > 2 ? argv[2] : nullptr, 3);
  const std::int32_t reps = parse_arg(argc > 3 ? argv[3] : nullptr, 5);

  fb::DeviceResources res{0};
  fb::tf::Executor executor{static_cast<std::size_t>(std::max(k, 1))};
  fb::OutputRing ring{res.dev_idx(), executor};
  const auto dir = std::filesystem::temp_directory_path() /
                   std::format("free_breakfast-output_ring_bench-{}",
                               std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(dir);

  std::println("k = {} arrays per run; ring {} x {} MiB", k, 4, 32);
  std::println("{:>6} {:>10} {:>12} {:>12} {:>8}", "n", "MiB/array", "serial ms", "ring ms",
               "speedup");
  for (std::int32_t n = 512; n <= max_n; n *= 2) {
    const auto un = static_cast<std::size_t>(n);
    const std::vector<std::size_t> shape{un, un};
    std::vector<fb::Buffers::device<double>> arrays;
    for (std::int32_t i = 0; i < k; ++i) {
      arrays.emplace_back(un * un, res.device());
      check(fb::gpu_status(wwr::wwrMemsetAsync(arrays.back().data(), i + 1,
                                               un * un * sizeof(double), res.stream())),
            "memset");
    }
    check(res.sync(), "sync");

    const double serial = time_call(reps, [&] {
      for (std::int32_t i = 0; i < k; ++i) {
        fb::NpyArray host{.shape = shape, .data = std::vector<double>(un * un)};
        check(fb::gpu_status(wwr::wwrMemcpyAsync(host.data.data(), arrays[i].data(),
                                                 host.data.size() * sizeof(double),
                                                 wwr::wwrMemcpyDeviceToHost, res.stream())),
              "download");
        check(res.sync(), "sync");
        check(fb::write_npy(dir / std::format("serial{}.npy", i), host), "write_npy");
      }
    });
    const double ringed = time_call(reps, [&] {
      for (std::int32_t i = 0; i < k; ++i) {
        check(ring.write_npy(res, dir / std::format("ring{}.npy", i), shape,
                             {arrays[i].data(), un * un}),
              "ring write_npy");
      }
      check(ring.wait(), "ring wait");
    });
    std::println("{:>6} {:>10.1f} {:>12.2f} {:>12.2f} {:>7.2f}x", n,
                 static_cast<double>(un * un * sizeof(double)) / (1 << 20), serial, ringed,
                 serial / ringed);
  }
  std::error_code ignored;
  std::filesystem::remove_all(dir, ignored);
  return 0;
}
