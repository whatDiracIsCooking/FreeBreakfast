// merge_sort against radix_sort and the vendor's two sorts, DeviceMergeSort
// (cub_merge_sort) and DeviceRadixSort (cub_radix_sort), side by side, keys
// only and keys with uint32_t values. The
// sizes are radix_sort_bench's: a spread below 16384, then every second power
// of two from 2^10 up to 2^max_log2.
//
//   free_breakfast_merge_sort_bench [max_log2 = 26] [reps = 10]
//
// The method is radix_sort_bench's: each timing is the median of `reps` sorts
// timed with GPU events around the sort alone, the input restored from a
// pristine device copy before every rep, outside the timed region, and every
// sort's scratch grown in a warm-up so no timed rep allocates. The keys are
// finite doubles of both signs over 40 binary decades.
//
// "match" compares merge_sort's output with radix_sort's and cub_merge_sort's
// bit for bit, keys and values: all three are stable in the same total order,
// -0.0 before +0.0 included, so they must agree exactly. (The vendor radix
// sort parts from them at zero; see ../radix/README.md.)
//
// Not a test: read the numbers, do not gate on them (bench/README.md).

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.sort.merge;
import free_breakfast.experimental.sort.radix;

namespace fb = free_breakfast;

namespace {

// Aborts on a failed call: a benchmark has nothing to recover to, and a
// timing taken past a failure would be meaningless.
void check(const bool ok, const std::string_view what) {
  if (!ok) {
    std::println(std::cerr, "merge_sort_bench: {} failed", what);
    std::exit(1);
  }
}

void check_gpu(const wwr::wwrError_t err, const std::string_view what) {
  check(err == wwr::wwrSuccess, what);
}

using SortFn = fb::Status (*)(fb::DeviceResources &, std::int32_t, double *, std::uint32_t *);

struct Timing {
  float median_ms;
  std::vector<double> keys;
  std::vector<std::uint32_t> values;
};

// One sort, timed over `reps` restored-input runs after one warm-up run.
Timing time_sort(fb::DeviceResources &res, const SortFn sort, const std::int32_t n,
                 const fb::Buffers::device<double> &pristine_keys,
                 const fb::Buffers::device<std::uint32_t> &pristine_values, const bool with_values,
                 const std::int32_t reps) {
  const auto count = static_cast<std::size_t>(n);
  const wwr::wwrStream_t stream = res.stream();
  fb::Buffers::device<double> keys{count, res.device()};
  fb::Buffers::device<std::uint32_t> values{count, res.device()};
  const fb::Event start{res.dev_idx(), 0U};
  const fb::Event stop{res.dev_idx(), 0U};

  const auto restore = [&] {
    check_gpu(wwr::wwrMemcpyAsync(keys.data(), pristine_keys.data(), count * sizeof(double),
                                  wwr::wwrMemcpyDeviceToDevice, stream),
              "restoring keys");
    if (with_values) {
      check_gpu(wwr::wwrMemcpyAsync(values.data(), pristine_values.data(),
                                    count * sizeof(std::uint32_t), wwr::wwrMemcpyDeviceToDevice,
                                    stream),
                "restoring values");
    }
  };
  std::uint32_t *const values_arg = with_values ? values.data() : nullptr;

  // Warm-up: first-launch costs, and growing res's workspace to fit.
  restore();
  check(sort(res, n, keys.data(), values_arg).has_value(), "warm-up sort");
  check_gpu(wwr::wwrStreamSynchronize(stream), "warm-up synchronize");

  std::vector<float> ms(static_cast<std::size_t>(reps));
  for (float &t : ms) {
    restore();
    check_gpu(wwr::wwrEventRecord(start, stream), "recording start");
    check(sort(res, n, keys.data(), values_arg).has_value(), "sort");
    check_gpu(wwr::wwrEventRecord(stop, stream), "recording stop");
    check_gpu(wwr::wwrEventSynchronize(stop), "synchronizing stop");
    check_gpu(wwr::wwrEventElapsedTime(&t, start, stop), "reading the elapsed time");
  }
  std::ranges::sort(ms);

  Timing timing{ms[ms.size() / 2], std::vector<double>(count), {}};
  check_gpu(wwr::wwrMemcpyAsync(timing.keys.data(), keys.data(), count * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, stream),
            "downloading keys");
  if (with_values) {
    timing.values.resize(count);
    check_gpu(wwr::wwrMemcpyAsync(timing.values.data(), values.data(),
                                  count * sizeof(std::uint32_t), wwr::wwrMemcpyDeviceToHost,
                                  stream),
              "downloading values");
  }
  check_gpu(wwr::wwrStreamSynchronize(stream), "synchronizing the download");
  return timing;
}

bool same_bits(const Timing &a, const Timing &b) {
  const auto bits = [](const double d) { return std::bit_cast<std::uint64_t>(d); };
  return std::ranges::equal(a.keys, b.keys, {}, bits, bits) && a.values == b.values;
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
  const std::int32_t max_log2 = std::clamp(parse_arg(argc > 1 ? argv[1] : nullptr, 26), 10, 30);
  const std::int32_t reps = std::max(parse_arg(argc > 2 ? argv[2] : nullptr, 10), 1);

  std::int32_t device_count = 0;
  if (wwr::wwrGetDeviceCount(&device_count) != wwr::wwrSuccess || device_count < 1) {
    std::println(std::cerr, "merge_sort_bench: no GPU device visible to the runtime");
    return 1;
  }

  fb::DeviceResources res;
  const auto max_count = std::size_t{1} << max_log2;

  // One input, generated once at the largest size; each size sorts a prefix.
  std::mt19937_64 gen{2026};
  std::uniform_real_distribution<double> mantissa{-1.0, 1.0};
  std::uniform_int_distribution<std::int32_t> exponent{-20, 20};
  std::vector<double> host_keys(max_count);
  for (double &k : host_keys) {
    k = std::ldexp(mantissa(gen), exponent(gen));
  }
  std::vector<std::uint32_t> host_values(max_count);
  std::iota(host_values.begin(), host_values.end(), 0U);

  fb::Buffers::device<double> pristine_keys{max_count, res.device()};
  fb::Buffers::device<std::uint32_t> pristine_values{max_count, res.device()};
  check_gpu(wwr::wwrMemcpyAsync(pristine_keys.data(), host_keys.data(), max_count * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            "uploading keys");
  check_gpu(wwr::wwrMemcpyAsync(pristine_values.data(), host_values.data(),
                                max_count * sizeof(std::uint32_t), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            "uploading values");
  check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the upload");

  std::println("merge_sort vs radix_sort vs the vendor's DeviceMergeSort and DeviceRadixSort, "
               "double keys, median of {} reps",
               reps);
  std::println("{:>10} {:>6} {:>9} {:>9} {:>10} {:>10} {:>8} {:>9} {:>9} {:>10} {:>6}", "n", "kind",
               "merge ms", "radix ms", "vnd mrg ms", "vnd rdx ms", "mrg/rdx", "mrg/vmrg",
               "mrg/vrdx", "mrg Gkey/s", "match");
  std::vector<std::int32_t> ns{32, 100, 512, 2048, 8192, 12'000, 16'383};
  for (std::int32_t log2 = 10; log2 <= max_log2; log2 += 2) {
    ns.push_back(std::int32_t{1} << log2);
  }
  std::ranges::sort(ns);
  for (const std::int32_t n : ns) {
    for (const bool with_values : {false, true}) {
      const Timing merge =
          time_sort(res, fb::merge_sort, n, pristine_keys, pristine_values, with_values, reps);
      const Timing radix =
          time_sort(res, fb::radix_sort, n, pristine_keys, pristine_values, with_values, reps);
      const Timing vendor_merge =
          time_sort(res, fb::cub_merge_sort, n, pristine_keys, pristine_values, with_values, reps);
      const Timing vendor_radix =
          time_sort(res, fb::cub_radix_sort, n, pristine_keys, pristine_values, with_values, reps);
      std::println(
          "{:>10} {:>6} {:>9.3f} {:>9.3f} {:>10.3f} {:>10.3f} {:>8.2f} {:>9.2f} {:>9.2f} {:>10.3f} "
          "{:>6}",
          n, with_values ? "pairs" : "keys", merge.median_ms, radix.median_ms,
          vendor_merge.median_ms, vendor_radix.median_ms, merge.median_ms / radix.median_ms,
          merge.median_ms / vendor_merge.median_ms, merge.median_ms / vendor_radix.median_ms,
          static_cast<double>(n) / (merge.median_ms * 1e6),
          same_bits(merge, radix) && same_bits(merge, vendor_merge) ? "yes" : "NO");
    }
  }
  return 0;
}
