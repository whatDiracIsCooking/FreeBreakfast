// GPU lossless compression on the HIP backend: hipCOMP's batched LZ4 and
// Cascaded codecs over device-resident arrays, round-tripped and checked bit
// for bit.
//
//   free_breakfast_hip_comp_demo [mib = 64] [reps = 5]
//
// Each input is `mib` MiB, cut into 64 KiB chunks that are compressed as one
// batch; the compressed chunks are decompressed as one batch into a fresh
// buffer, which must equal the input byte for byte. Times are the median of
// `reps` event-timed calls after one warm-up, and GB/s is uncompressed bytes
// over that time, both directions.
//
// The inputs are chosen to show where each codec earns its keep, and where
// nothing can:
//   - sparse fp64: ~90% exact zeros, the rest random -- screened-integral-like.
//     LZ4 finds the zero runs.
//   - dense fp64: uniform random doubles. Incompressible; the ratio is ~1 by
//     construction, and saying so is the point.
//   - sorted int64 indices: a sparse pattern's ascending column indices.
//     Cascaded (delta + RLE + bitpack, as 8-byte integers) is made for this.
//
// Raw hipCOMP, not wwr.comp: the portable layer needs WWR_WITH_COMP for the
// whole tree, which this build cannot turn on (README.md). Plain C++, not a
// module unit, so the vendor headers are simply included.

// Standard headers first, before any vendor runtime header (primitives.h has
// the reason: HIP's host_defines.h poisons names libc++'s headers use).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipcomp/cascaded.h>
#include <hipcomp/lz4.h>

// math::idivup, core's ceiling division.
#include "primitives.h"

namespace {

constexpr std::size_t kChunkBytes = std::size_t{64} << 10;

void check_hip(const hipError_t err, const char *const what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
    std::abort();
  }
}

void check_comp(const hipcompStatus_t status, const char *const what) {
  if (status != hipcompSuccess) {
    std::fprintf(stderr, "%s: hipcompStatus_t %d\n", what, static_cast<std::int32_t>(status));
    std::abort();
  }
}

template <typename T>
T *device_alloc(const std::size_t count) {
  void *ptr = nullptr;
  check_hip(hipMalloc(&ptr, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
  return static_cast<T *>(ptr);
}

template <typename T>
T *device_upload(const std::vector<T> &host) {
  T *ptr = device_alloc<T>(host.size());
  check_hip(hipMemcpy(ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
            "hipMemcpy H2D");
  return ptr;
}

// The three entry points a codec run needs, bound to that codec's options.
struct Codec {
  std::string name;
  std::function<void(std::size_t batch, std::size_t *temp, std::size_t *max_out)> compress_sizes;
  std::function<void(const void *const *in, const std::size_t *in_bytes, std::size_t batch,
                     void *temp, std::size_t temp_bytes, void *const *out, std::size_t *out_bytes,
                     hipStream_t stream)>
      compress;
  std::function<void(std::size_t batch, std::size_t *temp)> decompress_temp;
  std::function<void(const void *const *in, const std::size_t *in_bytes,
                     const std::size_t *out_bytes, std::size_t *actual_bytes, std::size_t batch,
                     void *temp, std::size_t temp_bytes, void *const *out,
                     hipcompStatus_t *statuses, hipStream_t stream)>
      decompress;
};

Codec lz4_codec() {
  const hipcompBatchedLZ4Opts_t opts = hipcompBatchedLZ4DefaultOpts;
  return {
      "lz4",
      [opts](std::size_t batch, std::size_t *temp, std::size_t *max_out) {
        check_comp(hipcompBatchedLZ4CompressGetTempSize(batch, kChunkBytes, opts, temp),
                   "LZ4 compress temp size");
        check_comp(hipcompBatchedLZ4CompressGetMaxOutputChunkSize(kChunkBytes, opts, max_out),
                   "LZ4 max output chunk size");
      },
      [opts](const void *const *in, const std::size_t *in_bytes, std::size_t batch, void *temp,
             std::size_t temp_bytes, void *const *out, std::size_t *out_bytes, hipStream_t stream) {
        check_comp(hipcompBatchedLZ4CompressAsync(in, in_bytes, kChunkBytes, batch, temp,
                                                  temp_bytes, out, out_bytes, opts, stream),
                   "LZ4 compress");
      },
      [](std::size_t batch, std::size_t *temp) {
        check_comp(hipcompBatchedLZ4DecompressGetTempSize(batch, kChunkBytes, temp),
                   "LZ4 decompress temp size");
      },
      [](const void *const *in, const std::size_t *in_bytes, const std::size_t *out_bytes,
         std::size_t *actual_bytes, std::size_t batch, void *temp, std::size_t temp_bytes,
         void *const *out, hipcompStatus_t *statuses, hipStream_t stream) {
        check_comp(hipcompBatchedLZ4DecompressAsync(in, in_bytes, out_bytes, actual_bytes, batch,
                                                    temp, temp_bytes, out, statuses, stream),
                   "LZ4 decompress");
      },
  };
}

// One delta then one RLE over 8-byte integers, bit-packed: sorted indices
// become small, mostly-equal gaps.
Codec cascaded_codec() {
  const hipcompBatchedCascadedOpts_t opts{4096, HIPCOMP_TYPE_LONGLONG, 1, 1, 1};
  return {
      "cascaded",
      [opts](std::size_t batch, std::size_t *temp, std::size_t *max_out) {
        check_comp(hipcompBatchedCascadedCompressGetTempSize(batch, kChunkBytes, opts, temp),
                   "Cascaded compress temp size");
        check_comp(hipcompBatchedCascadedCompressGetMaxOutputChunkSize(kChunkBytes, opts, max_out),
                   "Cascaded max output chunk size");
      },
      [opts](const void *const *in, const std::size_t *in_bytes, std::size_t batch, void *temp,
             std::size_t temp_bytes, void *const *out, std::size_t *out_bytes, hipStream_t stream) {
        check_comp(hipcompBatchedCascadedCompressAsync(in, in_bytes, kChunkBytes, batch, temp,
                                                       temp_bytes, out, out_bytes, opts, stream),
                   "Cascaded compress");
      },
      [](std::size_t batch, std::size_t *temp) {
        check_comp(hipcompBatchedCascadedDecompressGetTempSize(batch, kChunkBytes, temp),
                   "Cascaded decompress temp size");
      },
      [](const void *const *in, const std::size_t *in_bytes, const std::size_t *out_bytes,
         std::size_t *actual_bytes, std::size_t batch, void *temp, std::size_t temp_bytes,
         void *const *out, hipcompStatus_t *statuses, hipStream_t stream) {
        check_comp(hipcompBatchedCascadedDecompressAsync(in, in_bytes, out_bytes, actual_bytes,
                                                         batch, temp, temp_bytes, out, statuses,
                                                         stream),
                   "Cascaded decompress");
      },
  };
}

// Median milliseconds of `reps` event-timed calls of `call`, after one warm-up.
float median_ms(const std::int32_t reps, hipStream_t stream, const std::function<void()> &call) {
  hipEvent_t start{};
  hipEvent_t stop{};
  check_hip(hipEventCreate(&start), "hipEventCreate");
  check_hip(hipEventCreate(&stop), "hipEventCreate");
  call();
  check_hip(hipStreamSynchronize(stream), "warm-up");
  std::vector<float> times;
  for (std::int32_t r = 0; r < reps; ++r) {
    check_hip(hipEventRecord(start, stream), "hipEventRecord");
    call();
    check_hip(hipEventRecord(stop, stream), "hipEventRecord");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize");
    float ms = 0.0F;
    check_hip(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
    times.push_back(ms);
  }
  check_hip(hipEventDestroy(start), "hipEventDestroy");
  check_hip(hipEventDestroy(stop), "hipEventDestroy");
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

// Round-trips `input` through `codec` on the device; returns false on any
// mismatch, which is also printed.
bool run(const std::string &label, const Codec &codec, const std::vector<std::uint8_t> &input,
         const std::int32_t reps, hipStream_t stream) {
  const std::size_t total = input.size();
  const std::size_t batch = free_breakfast::math::idivup(total, kChunkBytes);

  std::size_t comp_temp_bytes = 0;
  std::size_t max_out = 0;
  codec.compress_sizes(batch, &comp_temp_bytes, &max_out);
  std::size_t decomp_temp_bytes = 0;
  codec.decompress_temp(batch, &decomp_temp_bytes);

  auto *d_in = device_upload(input);
  auto *d_comp = device_alloc<std::uint8_t>(batch * max_out);
  auto *d_out = device_alloc<std::uint8_t>(total);
  auto *d_comp_temp = device_alloc<std::uint8_t>(comp_temp_bytes);
  auto *d_decomp_temp = device_alloc<std::uint8_t>(decomp_temp_bytes);

  std::vector<const void *> in_ptrs(batch);
  std::vector<void *> comp_ptrs(batch);
  std::vector<void *> out_ptrs(batch);
  std::vector<std::size_t> chunk_bytes(batch);
  for (std::size_t i = 0; i < batch; ++i) {
    in_ptrs[i] = d_in + i * kChunkBytes;
    comp_ptrs[i] = d_comp + i * max_out;
    out_ptrs[i] = d_out + i * kChunkBytes;
    chunk_bytes[i] = std::min(kChunkBytes, total - i * kChunkBytes);
  }
  auto *d_in_ptrs = device_upload(in_ptrs);
  auto *d_comp_ptrs = device_upload(comp_ptrs);
  auto *d_out_ptrs = device_upload(out_ptrs);
  auto *d_chunk_bytes = device_upload(chunk_bytes);
  auto *d_comp_bytes = device_alloc<std::size_t>(batch);
  auto *d_actual_bytes = device_alloc<std::size_t>(batch);
  auto *d_statuses = device_alloc<hipcompStatus_t>(batch);

  const float compress_ms = median_ms(reps, stream, [&] {
    codec.compress(d_in_ptrs, d_chunk_bytes, batch, d_comp_temp, comp_temp_bytes, d_comp_ptrs,
                   d_comp_bytes, stream);
  });
  check_hip(hipMemsetAsync(d_out, 0, total, stream), "hipMemsetAsync");
  const float decompress_ms = median_ms(reps, stream, [&] {
    codec.decompress(d_comp_ptrs, d_comp_bytes, d_chunk_bytes, d_actual_bytes, batch, d_decomp_temp,
                     decomp_temp_bytes, d_out_ptrs, d_statuses, stream);
  });
  check_hip(hipStreamSynchronize(stream), "decompress");

  std::vector<std::size_t> comp_bytes(batch);
  std::vector<std::size_t> actual_bytes(batch);
  std::vector<hipcompStatus_t> statuses(batch);
  std::vector<std::uint8_t> output(total);
  check_hip(hipMemcpy(comp_bytes.data(), d_comp_bytes, batch * sizeof(std::size_t),
                      hipMemcpyDeviceToHost),
            "D2H compressed sizes");
  check_hip(hipMemcpy(actual_bytes.data(), d_actual_bytes, batch * sizeof(std::size_t),
                      hipMemcpyDeviceToHost),
            "D2H actual sizes");
  check_hip(hipMemcpy(statuses.data(), d_statuses, batch * sizeof(hipcompStatus_t),
                      hipMemcpyDeviceToHost),
            "D2H statuses");
  check_hip(hipMemcpy(output.data(), d_out, total, hipMemcpyDeviceToHost), "D2H output");

  std::size_t compressed = 0;
  bool ok = true;
  for (std::size_t i = 0; i < batch; ++i) {
    compressed += comp_bytes[i];
    if (statuses[i] != hipcompSuccess || actual_bytes[i] != chunk_bytes[i]) {
      if (ok) {
        std::fprintf(stderr, "  chunk %zu: status %d, %zu of %zu bytes\n", i,
                     static_cast<std::int32_t>(statuses[i]), actual_bytes[i], chunk_bytes[i]);
      }
      ok = false;
    }
  }
  const bool identical = std::memcmp(output.data(), input.data(), total) == 0;
  ok = ok && identical;

  const double gb = static_cast<double>(total) / 1e9;
  std::printf("%-22s %-9s %8.2f MiB -> %8.2f MiB  ratio %6.2f  comp %7.1f GB/s  "
              "decomp %7.1f GB/s  %s\n",
              label.c_str(), codec.name.c_str(), static_cast<double>(total) / (1 << 20),
              static_cast<double>(compressed) / (1 << 20),
              static_cast<double>(total) / static_cast<double>(compressed),
              gb / (compress_ms / 1e3), gb / (decompress_ms / 1e3), ok ? "bit-exact" : "MISMATCH");

  for (void *ptr : {static_cast<void *>(d_in), static_cast<void *>(d_comp),
                    static_cast<void *>(d_out), static_cast<void *>(d_comp_temp),
                    static_cast<void *>(d_decomp_temp), static_cast<void *>(d_in_ptrs),
                    static_cast<void *>(d_comp_ptrs), static_cast<void *>(d_out_ptrs),
                    static_cast<void *>(d_chunk_bytes), static_cast<void *>(d_comp_bytes),
                    static_cast<void *>(d_actual_bytes), static_cast<void *>(d_statuses)}) {
    check_hip(hipFree(ptr), "hipFree");
  }
  return ok;
}

template <typename T>
std::vector<std::uint8_t> as_bytes(const std::vector<T> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

} // namespace

int main(std::int32_t argc, char **argv) {
  const std::size_t mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 64;
  const std::int32_t reps = argc > 2 ? std::atoi(argv[2]) : 5;
  const std::size_t n = (mib << 20) / sizeof(double);

  std::int32_t device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 2;
  }
  hipDeviceProp_t prop{};
  check_hip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::printf("device: %s (%s), %zu KiB chunks, median of %d\n\n", prop.name, prop.gcnArchName,
              kChunkBytes >> 10, reps);

  std::mt19937_64 rng{20261113};
  std::uniform_real_distribution<double> value{-1.0, 1.0};
  std::bernoulli_distribution nonzero{0.1};

  std::vector<double> sparse(n);
  for (double &x : sparse) {
    x = nonzero(rng) ? value(rng) : 0.0;
  }
  std::vector<double> dense(n);
  for (double &x : dense) {
    x = value(rng);
  }
  // Ascending column indices of a ~10%-dense pattern: gaps of mostly 1..20.
  std::vector<std::int64_t> indices(n);
  std::geometric_distribution<std::int64_t> gap{0.1};
  std::int64_t col = 0;
  for (std::int64_t &i : indices) {
    col += 1 + gap(rng);
    i = col;
  }

  hipStream_t stream{};
  check_hip(hipStreamCreate(&stream), "hipStreamCreate");

  const Codec lz4 = lz4_codec();
  const Codec cascaded = cascaded_codec();
  bool ok = true;
  ok = run("sparse fp64 (90% zero)", lz4, as_bytes(sparse), reps, stream) && ok;
  ok = run("dense fp64 (random)", lz4, as_bytes(dense), reps, stream) && ok;
  ok = run("sorted int64 indices", lz4, as_bytes(indices), reps, stream) && ok;
  ok = run("sorted int64 indices", cascaded, as_bytes(indices), reps, stream) && ok;

  check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
  return ok ? 0 : 1;
}
