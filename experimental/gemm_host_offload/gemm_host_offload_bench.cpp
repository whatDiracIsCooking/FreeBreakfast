// The DF exchange build K = X^T X, X_m = B_m C, cut into batches, to measure
// what each batch costs on top of the one-call build: the per-batch constant
// delta in T(n_b) = T_1 + n_b delta, and whether it is launch overhead, the
// cross-stream event waits, or the GEMMs' own tail and tile fill.
//
//   free_breakfast_gemm_host_offload_bench [--reps 10] [--warmup 3]
//       [--nao 256,448,640] [--only resident|pipeline] [--ring-gib 6]
//       [--csv gemm_host_offload.csv]
//
// The calls are df_exchange's (src/integrals/eri/density_fitting/jk/jk.cpp):
// B is [m][n][P] with P fastest, so B_m is naux x nao; X_m = B_m C is one
// strided-batched gemm over m, and K = X^T X one syrk. B and C are random
// (a hash of the index), not a real fit: dense fp64 GEMM time does not depend
// on the values.
//
// Two ways to cut it into batches:
//
//   m  a batch is b consecutive slabs B_m. The batched gemm runs over b slabs
//      instead of nao and writes its part of a device-resident X; the one syrk
//      runs after the last batch. Slabs are contiguous in the host layout.
//   P  a batch is b consecutive aux functions. The gemm is b x nocc per slab,
//      and each batch adds its own syrk into K (beta = 1 after the first). On
//      the device the batch is a strided view (lda = naux); streamed from the
//      host it needs B laid out in P blocks, [block][m][n][p].
//
// The variants, each a curve over the batch count n_b:
//
//   chopped   B resident, the batches enqueued back to back on one stream.
//             n_b = 1 is the one-call build itself, df_exchange's T_1.
//   events    chopped, plus the ring's event protocol with a second stream
//             (copied[s] / consumed[s], R = 2) but no copies: the waits alone.
//   graph     chopped, captured once into a graph and launched: no per-call
//             CPU launch. graph ~= chopped means launch overhead is not delta.
//   pipeline  B in pinned host memory, a ring of R device slots: the copy of
//             batch i+1 overlaps the gemms of batch i.
//   copies    pipeline with the gemms removed: the copies alone, the same
//             chunks, the same waits.
//
// Every config is run `warmup` times untimed, then `reps` times, GPU-event
// timed; within a group the configs are run round-robin, one run of each per
// pass, so clock and thermal drift spreads over all of them instead of landing
// on whichever ran last. The pipeline's P curve is the exception: each batch
// size needs its own host layout, so its configs are round-robin per batch
// size only.
//
// Columns: median, min and interquartile range (as % of the median) of the
// reps; `excess/b` = (median - T_1) / (n_b - 1), the per-batch cost read off
// one point; for pipeline, `over` = median - max(T_1, copies at the same
// batch), what the overlap failed to hide. Per curve, a least-squares fit of
// median against n_b gives delta (the slope) with its standard error. K is
// checked once per config against the one-call build: m keeps every product
// identical, so it should agree bitwise; P sums in another order, so it is a
// relative difference.
//
// Not a test: read the numbers, do not gate on them. Hold the whole card
// first (with_gpu_card, devtools/lib.sh), or another job on it is in the
// number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;

namespace fb = free_breakfast;

namespace {

constexpr double kOne = 1.0;
constexpr double kZero = 0.0;

void check(const bool ok, const std::string_view what) {
  if (!ok) {
    std::println(std::cerr, "gemm_host_offload_bench: {} failed", what);
    std::exit(1);
  }
}

void check_gpu(const wwr::wwrError_t err, const std::string_view what) {
  check(err == wwr::wwrSuccess, what);
}

void check_blas(const wwr::wwrblasStatus_t status, const std::string_view what) {
  check(status == wwr::WWRBLAS_STATUS_SUCCESS, what);
}

// ---------------------------------------------------------------------------
// Operands

// splitmix64's finalizer: a well-mixed 64-bit value from an index.
std::uint64_t mix(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

// Uniform in [-1, 1), from an index.
double uniform(const std::uint64_t key) {
  return static_cast<double>(mix(key) >> 11U) * 0x1.0p-52 - 1.0;
}

// B at (m, n, P): a function of the index alone, so every layout holds the same
// tensor, and symmetric in (m, n), as a real B is.
double b_value(const std::int64_t m, const std::int64_t n, const std::int64_t p,
               const std::int64_t nao, const std::int64_t naux) {
  const std::int64_t lo = std::min(m, n);
  const std::int64_t hi = std::max(m, n);
  return uniform(static_cast<std::uint64_t>((lo * nao + hi) * naux + p));
}

// Run f(i) for i in [0, count) over every hardware thread.
template <typename F>
void parallel_for(const std::int64_t count, F &&f) {
  const auto threads =
      std::min<std::int64_t>(std::max(1U, std::thread::hardware_concurrency()), count);
  std::vector<std::jthread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (std::int64_t t = 0; t < threads; ++t) {
    pool.emplace_back([&f, t, threads, count] {
      for (std::int64_t i = t; i < count; i += threads) {
        f(i);
      }
    });
  }
}

// B in df_exchange's layout, [m][n][P]: slab m is naux x nao, lda naux.
void fill_slabs(double *const host, const std::int64_t nao, const std::int64_t naux) {
  parallel_for(nao, [=](const std::int64_t m) {
    for (std::int64_t n = 0; n < nao; ++n) {
      double *const row = host + (m * nao + n) * naux;
      for (std::int64_t p = 0; p < naux; ++p) {
        row[p] = b_value(m, n, p, nao, naux);
      }
    }
  });
}

// B in aux blocks of `batch` (the last one shorter): block j, of size s, starts
// at its first aux index times nao^2 and is [m][n][p] with lda s, which is what
// a ring slot then holds.
void fill_aux_blocks(double *const host, const std::int64_t nao, const std::int64_t naux,
                     const std::int64_t batch) {
  for (std::int64_t start = 0; start < naux; start += batch) {
    const std::int64_t size = std::min(batch, naux - start);
    double *const block = host + start * nao * nao;
    parallel_for(nao, [=](const std::int64_t m) {
      for (std::int64_t n = 0; n < nao; ++n) {
        double *const row = block + (m * nao + n) * size;
        for (std::int64_t q = 0; q < size; ++q) {
          row[q] = b_value(m, n, start + q, nao, naux);
        }
      }
    });
  }
}

// C, nao x nocc column-major, scaled by 1/sqrt(nao) so X stays O(1).
std::vector<double> make_c(const std::int64_t nao, const std::int64_t nocc) {
  std::vector<double> c(static_cast<std::size_t>(nao * nocc));
  const double scale = 1.0 / std::sqrt(static_cast<double>(nao));
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = scale * uniform((1ULL << 62U) + i);
  }
  return c;
}

// ---------------------------------------------------------------------------
// The problem and its batches

struct Problem {
  std::int32_t nao;
  std::int32_t naux;
  std::int32_t nocc;

  // X's rows, naux nocc, and the syrk's k in the one-call build.
  std::int64_t rows() const { return std::int64_t{naux} * nocc; }
  // B_m's stride, naux nao.
  std::int64_t slab() const { return std::int64_t{naux} * nao; }
  std::int64_t nn() const { return std::int64_t{nao} * nao; }
};

enum class Mode : std::uint8_t { kSlab, kAux };

std::string_view mode_name(const Mode mode) {
  return mode == Mode::kSlab ? "m" : "P";
}

struct Batch {
  std::int32_t start;
  std::int32_t size;
};

std::vector<Batch> batches_of(const Problem &prob, const Mode mode, const std::int32_t batch) {
  const std::int32_t total = mode == Mode::kSlab ? prob.nao : prob.naux;
  std::vector<Batch> out;
  for (std::int32_t start = 0; start < total; start += batch) {
    out.push_back({start, std::min(batch, total - start)});
  }
  return out;
}

// One batch's B as a device pointer, with the leading dimension and slab stride
// it is read with.
struct BatchOperand {
  const double *a;
  std::int32_t lda;
  std::int64_t stride;
};

// The batch's B inside the device-resident [m][n][P] tensor.
BatchOperand resident_operand(const Problem &prob, const Mode mode, const double *const b,
                              const Batch batch) {
  if (mode == Mode::kSlab) {
    return {b + std::int64_t{batch.start} * prob.slab(), prob.naux, prob.slab()};
  }
  return {b + batch.start, prob.naux, prob.slab()};
}

// The batch's B as a ring slot holds it: whole slabs (m), or an aux block (P).
BatchOperand slot_operand(const Problem &prob, const Mode mode, const double *const slot,
                          const Batch batch) {
  if (mode == Mode::kSlab) {
    return {slot, prob.naux, prob.slab()};
  }
  return {slot, batch.size, std::int64_t{batch.size} * prob.nao};
}

// Where the batch's B starts in the host layout for `mode`, and how many
// doubles it is.
std::pair<std::int64_t, std::int64_t> host_extent(const Problem &prob, const Mode mode,
                                                  const Batch batch) {
  if (mode == Mode::kSlab) {
    return {std::int64_t{batch.start} * prob.slab(), std::int64_t{batch.size} * prob.slab()};
  }
  return {std::int64_t{batch.start} * prob.nn(), std::int64_t{batch.size} * prob.nn()};
}

struct Operands {
  wwr::wwrblasHandle_t blas;
  const double *c;
  double *x;
  double *k;
};

// X for the batch: m writes its slabs' part of the whole X (naux x nocc per
// slab, stride rows); P writes a batch-sized X, b x nocc per slab.
void batch_gemm(const Problem &prob, const Mode mode, const Batch batch, const BatchOperand op,
                const Operands &ops) {
  if (mode == Mode::kSlab) {
    check_blas(wwr::gemmStridedBatched<double, std::int32_t>(
                   ops.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, prob.naux, prob.nocc, prob.nao,
                   &kOne, op.a, op.lda, op.stride, ops.c, prob.nao, 0, &kZero,
                   ops.x + std::int64_t{batch.start} * prob.rows(), prob.naux, prob.rows(),
                   batch.size),
               "the slab batch's gemm");
    return;
  }
  check_blas(wwr::gemmStridedBatched<double, std::int32_t>(
                 ops.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, batch.size, prob.nocc, prob.nao,
                 &kOne, op.a, op.lda, op.stride, ops.c, prob.nao, 0, &kZero, ops.x, batch.size,
                 std::int64_t{batch.size} * prob.nocc, prob.nao),
             "the aux batch's gemm");
}

// P's per-batch K += X_b^T X_b, lower triangle; the first batch overwrites.
void batch_syrk(const Problem &prob, const Batch batch, const Operands &ops) {
  const std::int32_t k = batch.size * prob.nocc;
  check_blas(wwr::syrk<double, std::int32_t>(ops.blas, wwr::WWRBLAS_FILL_MODE_LOWER,
                                             wwr::WWRBLAS_OP_T, prob.nao, k, &kOne, ops.x, k,
                                             batch.start == 0 ? &kZero : &kOne, ops.k, prob.nao),
             "the aux batch's syrk");
}

// m's one K = X^T X after the last batch, lower triangle.
void final_syrk(const Problem &prob, const Operands &ops) {
  const auto rows = static_cast<std::int32_t>(prob.rows());
  check_blas(wwr::syrk<double, std::int32_t>(ops.blas, wwr::WWRBLAS_FILL_MODE_LOWER,
                                             wwr::WWRBLAS_OP_T, prob.nao, rows, &kOne, ops.x, rows,
                                             &kZero, ops.k, prob.nao),
             "the final syrk");
}

// ---------------------------------------------------------------------------
// Enqueueing one build

constexpr std::int32_t kMaxRing = 3;

// The streams and events the ring uses. compute is res.stream(), which the
// BLAS handle is bound to; copy is a second non-blocking stream.
struct Ring {
  wwr::wwrStream_t compute;
  fb::Stream copy;
  std::vector<fb::Event> copied;
  std::vector<fb::Event> consumed;
  fb::Event fork;
  fb::Event join;

  explicit Ring(fb::DeviceResources &res)
      : compute(res.stream()), copy(res.dev_idx()), fork(res.dev_idx()), join(res.dev_idx()) {
    copied.reserve(kMaxRing);
    consumed.reserve(kMaxRing);
    for (std::int32_t s = 0; s < kMaxRing; ++s) {
      copied.emplace_back(res.dev_idx());
      consumed.emplace_back(res.dev_idx());
    }
  }
};

// chopped: the batches back to back on the compute stream, B resident.
void enqueue_chopped(const Problem &prob, const Mode mode, const std::vector<Batch> &batches,
                     const double *const b, const Operands &ops) {
  for (const Batch batch : batches) {
    batch_gemm(prob, mode, batch, resident_operand(prob, mode, b, batch), ops);
    if (mode == Mode::kAux) {
      batch_syrk(prob, batch, ops);
    }
  }
  if (mode == Mode::kSlab) {
    final_syrk(prob, ops);
  }
}

// What the ring does each batch.
struct RingWork {
  std::int32_t depth;
  bool copies;            // copy each batch into its slot
  bool gemms;             // run the batch's gemms (and P's syrk)
  const double *resident; // B on the device, when the gemms read it there (events)
  const double *host;     // B in pinned memory, in `mode`'s layout (pipeline, copies)
  std::span<double *const> slots;
};

// The ring: the copy of batch i into slot i % depth waits until the gemm of
// batch i - depth has consumed that slot, and the gemm of batch i waits until
// its copy has landed. The copy stream is forked from the compute stream first
// and joined back into it last, so the build is bracketed by the compute
// stream's own events.
void enqueue_ring(Ring &ring, const Problem &prob, const Mode mode,
                  const std::vector<Batch> &batches, const RingWork &work, const Operands &ops) {
  check_gpu(wwr::wwrEventRecord(ring.fork, ring.compute), "recording the fork");
  check_gpu(wwr::wwrStreamWaitEvent(ring.copy, ring.fork, 0), "forking the copy stream");
  for (std::size_t i = 0; i < batches.size(); ++i) {
    const Batch batch = batches[i];
    const auto s = static_cast<std::size_t>(i % static_cast<std::size_t>(work.depth));
    if (i >= static_cast<std::size_t>(work.depth)) {
      check_gpu(wwr::wwrStreamWaitEvent(ring.copy, ring.consumed[s], 0), "waiting for a slot");
    }
    if (work.copies) {
      const auto [offset, count] = host_extent(prob, mode, batch);
      check_gpu(wwr::wwrMemcpyAsync(work.slots[s], work.host + offset,
                                    static_cast<std::size_t>(count) * sizeof(double),
                                    wwr::wwrMemcpyHostToDevice, ring.copy),
                "copying a batch into its slot");
    }
    check_gpu(wwr::wwrEventRecord(ring.copied[s], ring.copy), "recording a copy");
    check_gpu(wwr::wwrStreamWaitEvent(ring.compute, ring.copied[s], 0), "waiting for a copy");
    if (work.gemms) {
      const BatchOperand op = work.copies ? slot_operand(prob, mode, work.slots[s], batch)
                                          : resident_operand(prob, mode, work.resident, batch);
      batch_gemm(prob, mode, batch, op, ops);
    }
    check_gpu(wwr::wwrEventRecord(ring.consumed[s], ring.compute), "recording a consume");
    if (work.gemms && mode == Mode::kAux) {
      batch_syrk(prob, batch, ops);
    }
  }
  if (work.gemms && mode == Mode::kSlab) {
    final_syrk(prob, ops);
  }
  check_gpu(wwr::wwrEventRecord(ring.join, ring.copy), "recording the join");
  check_gpu(wwr::wwrStreamWaitEvent(ring.compute, ring.join, 0), "joining the copy stream");
}

using GraphExec = wwr::extension::GraphExecWrapper<fb::CreatePolicy<wwr::wwrError_t>,
                                                   fb::DestroyPolicy<wwr::wwrError_t>>;

// `enqueue` captured on `stream` into an executable graph.
template <typename F>
std::unique_ptr<GraphExec> capture(const wwr::wwrStream_t stream, F &&enqueue) {
  check_gpu(wwr::wwrStreamBeginCapture(stream, wwr::wwrStreamCaptureModeGlobal),
            "beginning the capture");
  enqueue();
  wwr::wwrGraph_t graph = nullptr;
  check_gpu(wwr::wwrStreamEndCapture(stream, &graph), "ending the capture");
  auto exec = std::make_unique<GraphExec>(graph);
  check_gpu(wwr::wwrGraphDestroy(graph), "destroying the captured graph");
  return exec;
}

// ---------------------------------------------------------------------------
// Timing

enum class Variant : std::uint8_t { kChopped, kEvents, kGraph, kPipeline, kCopies };

std::string_view variant_name(const Variant v) {
  switch (v) {
  case Variant::kChopped:
    return "chopped";
  case Variant::kEvents:
    return "events";
  case Variant::kGraph:
    return "graph";
  case Variant::kPipeline:
    return "pipeline";
  case Variant::kCopies:
    return "copies";
  }
  return "?";
}

struct Run {
  Variant variant;
  Mode mode;
  std::int32_t depth; // ring depth; 0 for the variants without a ring
  std::int32_t batch;
  std::int64_t n_b;
  std::function<void()> enqueue;
  std::vector<float> ms;
  std::string check; // the K agreement, filled before timing
};

struct Timer {
  wwr::wwrStream_t stream;
  fb::Event start;
  fb::Event stop;

  explicit Timer(fb::DeviceResources &res)
      : stream(res.stream()), start(res.dev_idx(), 0U), stop(res.dev_idx(), 0U) {}

  float time(const std::function<void()> &enqueue) {
    check_gpu(wwr::wwrEventRecord(start, stream), "recording start");
    enqueue();
    check_gpu(wwr::wwrEventRecord(stop, stream), "recording stop");
    check_gpu(wwr::wwrEventSynchronize(stop), "synchronizing stop");
    float ms = 0.0F;
    check_gpu(wwr::wwrEventElapsedTime(&ms, start, stop), "reading the elapsed time");
    return ms;
  }
};

struct Settings {
  std::int32_t reps = 10;
  std::int32_t warmup = 3;
  std::vector<std::int32_t> naos{256, 448, 640};
  bool resident = true;
  bool pipeline = true;
  std::size_t ring_bytes = 6UZ << 30U;
  std::string csv = "gemm_host_offload.csv";
};

// `warmup` untimed passes, then `reps` timed ones, each pass one run of every
// config in order.
void time_round_robin(Timer &timer, std::vector<Run> &runs, const Settings &settings) {
  for (std::int32_t pass = 0; pass < settings.warmup + settings.reps; ++pass) {
    for (Run &run : runs) {
      const float ms = timer.time(run.enqueue);
      if (pass >= settings.warmup) {
        run.ms.push_back(ms);
      }
    }
  }
}

double percentile(std::vector<float> sorted, const double q) {
  std::ranges::sort(sorted);
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double frac = pos - static_cast<double>(lo);
  return static_cast<double>(sorted[lo]) * (1.0 - frac) + static_cast<double>(sorted[hi]) * frac;
}

double median(const Run &run) {
  return percentile(run.ms, 0.5);
}

// ---------------------------------------------------------------------------
// K agreement

std::vector<double> download_k(fb::DeviceResources &res, const Problem &prob, const double *k) {
  std::vector<double> host(static_cast<std::size_t>(prob.nn()));
  check_gpu(wwr::wwrMemcpyAsync(host.data(), k, host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            "downloading K");
  check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing the download");
  return host;
}

// "bitwise", or the largest lower-triangle difference relative to max|K|.
std::string compare_k(const Problem &prob, const std::vector<double> &ref,
                      const std::vector<double> &got) {
  double max_abs = 0.0;
  double max_diff = 0.0;
  bool same = true;
  for (std::int64_t col = 0; col < prob.nao; ++col) {
    for (std::int64_t row = col; row < prob.nao; ++row) {
      const auto i = static_cast<std::size_t>(row + col * prob.nao);
      max_abs = std::max(max_abs, std::abs(ref[i]));
      max_diff = std::max(max_diff, std::abs(ref[i] - got[i]));
      same = same && std::bit_cast<std::uint64_t>(ref[i]) == std::bit_cast<std::uint64_t>(got[i]);
    }
  }
  if (same) {
    return "bitwise";
  }
  return std::format("{:.1e}", max_abs > 0.0 ? max_diff / max_abs : max_diff);
}

// Run each config once into K and compare with the one-call build's K. The
// copies variant computes no K and is marked "-".
void check_runs(fb::DeviceResources &res, const Problem &prob, std::vector<Run> &runs,
                const std::vector<double> &ref, double *const k) {
  for (Run &run : runs) {
    if (run.variant == Variant::kCopies) {
      run.check = "-";
      continue;
    }
    check_gpu(wwr::wwrMemsetAsync(k, 0xFF, static_cast<std::size_t>(prob.nn()) * sizeof(double),
                                  res.stream()),
              "poisoning K");
    run.enqueue();
    run.check = compare_k(prob, ref, download_k(res, prob, k));
  }
}

// ---------------------------------------------------------------------------
// Reporting

struct Fit {
  double slope_us;
  double se_us;
  double intercept_ms;
  std::size_t points;
};

// Least squares of median (ms) against n_b; slope and its standard error in us.
Fit fit_curve(const std::vector<const Run *> &curve) {
  const auto n = static_cast<double>(curve.size());
  double sx = 0.0;
  double sy = 0.0;
  for (const Run *run : curve) {
    sx += static_cast<double>(run->n_b);
    sy += median(*run);
  }
  const double mx = sx / n;
  const double my = sy / n;
  double sxx = 0.0;
  double sxy = 0.0;
  for (const Run *run : curve) {
    const double dx = static_cast<double>(run->n_b) - mx;
    sxx += dx * dx;
    sxy += dx * (median(*run) - my);
  }
  if (curve.size() < 3 || sxx == 0.0) {
    return {0.0, 0.0, my, curve.size()};
  }
  const double slope = sxy / sxx;
  const double intercept = my - slope * mx;
  double ss = 0.0;
  for (const Run *run : curve) {
    const double r = median(*run) - (intercept + slope * static_cast<double>(run->n_b));
    ss += r * r;
  }
  const double se = std::sqrt(ss / (n - 2.0) / sxx);
  return {slope * 1e3, se * 1e3, intercept, curve.size()};
}

std::string curve_key(const Run &run) {
  return run.depth > 0
             ? std::format("{} {} R{}", variant_name(run.variant), mode_name(run.mode), run.depth)
             : std::format("{} {}", variant_name(run.variant), mode_name(run.mode));
}

void report(const std::string_view title, const Problem &prob, const std::vector<Run> &runs,
            const double t1_ms, std::ofstream &csv) {
  std::println("\n== {}: nao {} naux {} nocc {}   T_1 = {:.3f} ms", title, prob.nao, prob.naux,
               prob.nocc, t1_ms);
  std::println("{:<15} {:>6} {:>6} {:>10} {:>10} {:>6} {:>10} {:>9} {:>9}", "curve", "b", "n_b",
               "median ms", "min ms", "iqr %", "excess/b", "over ms", "K");

  // The copies time at each (mode, depth, batch), for the pipeline's `over`.
  std::map<std::tuple<Mode, std::int32_t, std::int32_t>, double> copies;
  for (const Run &run : runs) {
    if (run.variant == Variant::kCopies) {
      copies[{run.mode, run.depth, run.batch}] = median(run);
    }
  }

  std::map<std::string, std::vector<const Run *>> curves;
  for (const Run &run : runs) {
    curves[curve_key(run)].push_back(&run);
  }
  for (auto &[key, curve] : curves) {
    std::ranges::sort(curve, {}, &Run::n_b);
    for (const Run *run : curve) {
      const double med = median(*run);
      const double lo = percentile(run->ms, 0.0);
      const double iqr = percentile(run->ms, 0.75) - percentile(run->ms, 0.25);
      const std::string excess =
          run->n_b > 1 && run->variant != Variant::kCopies
              ? std::format("{:.1f}us", (med - t1_ms) * 1e3 / static_cast<double>(run->n_b - 1))
              : std::string{"-"};
      std::string over = "-";
      if (run->variant == Variant::kPipeline) {
        const auto it = copies.find({run->mode, run->depth, run->batch});
        if (it != copies.end()) {
          over = std::format("{:.3f}", med - std::max(t1_ms, it->second));
        }
      }
      std::println("{:<15} {:>6} {:>6} {:>10.3f} {:>10.3f} {:>6.2f} {:>10} {:>9} {:>9}", key,
                   run->batch, run->n_b, med, lo, 100.0 * iqr / med, excess, over, run->check);
      for (std::size_t r = 0; r < run->ms.size(); ++r) {
        std::println(csv, "{},{},{},{},{},{},{},{},{},{},{}", title, prob.nao, prob.naux, prob.nocc,
                     variant_name(run->variant), mode_name(run->mode), run->depth, run->batch,
                     run->n_b, r, run->ms[r]);
      }
    }
  }
  std::println("  fits, median ms = T_0 + n_b delta:");
  for (const auto &[key, curve] : curves) {
    const Fit fit = fit_curve(curve);
    std::println("  {:<15} delta = {:8.2f} +- {:6.2f} us/batch   T_0 = {:.3f} ms   ({} points)",
                 key, fit.slope_us, fit.se_us, fit.intercept_ms, fit.points);
  }
  std::cout.flush();
}

// ---------------------------------------------------------------------------
// The sweeps

std::vector<std::int32_t> slab_batches(const std::int32_t nao) {
  std::vector<std::int32_t> out;
  for (std::int32_t b = 1; b < nao; b *= 2) {
    out.push_back(b);
  }
  out.push_back(nao);
  return out;
}

std::vector<std::int32_t> aux_batches(const std::int32_t naux) {
  constexpr std::array kSizes{8,   12,  16,  24,  32,  48,  64,   96,
                              128, 192, 256, 384, 512, 768, 1024, 1536};
  std::vector<std::int32_t> out;
  for (const std::int32_t b : kSizes) {
    if (b < naux) {
      out.push_back(b);
    }
  }
  out.push_back(naux);
  return out;
}

// The device buffers one shape's builds share: C, X (sized for the largest
// nocc) and K.
struct Workspace {
  fb::Buffers::device<double> c;
  fb::Buffers::device<double> x;
  fb::Buffers::device<double> k;

  Workspace(fb::DeviceResources &res, const Problem &largest)
      : c(static_cast<std::size_t>(std::int64_t{largest.nao} * largest.nocc), res.device()),
        x(static_cast<std::size_t>(largest.rows() * largest.nao), res.device()),
        k(static_cast<std::size_t>(largest.nn()), res.device()) {}

  void upload_c(fb::DeviceResources &res, const Problem &prob) {
    const std::vector<double> host = make_c(prob.nao, prob.nocc);
    check_gpu(wwr::wwrMemcpyAsync(c.data(), host.data(), host.size() * sizeof(double),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              "uploading C");
    check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing C");
  }

  Operands ops(fb::DeviceResources &res) {
    return {res.blas().get(), c.data(), x.data(), k.data()};
  }
};

// chopped, events and graph over both modes, B resident.
void resident_group(fb::DeviceResources &res, Ring &ring, Timer &timer, const Problem &prob,
                    const double *const b, Workspace &ws, const std::vector<double> &ref,
                    const Settings &settings, std::ofstream &csv) {
  const Operands ops = ws.ops(res);
  std::vector<Run> runs;
  std::vector<std::unique_ptr<GraphExec>> graphs;
  for (const Mode mode : {Mode::kSlab, Mode::kAux}) {
    const std::vector<std::int32_t> sizes =
        mode == Mode::kSlab ? slab_batches(prob.nao) : aux_batches(prob.naux);
    for (const std::int32_t size : sizes) {
      const std::vector<Batch> batches = batches_of(prob, mode, size);
      const auto n_b = static_cast<std::int64_t>(batches.size());
      runs.push_back(
          {Variant::kChopped,
           mode,
           0,
           size,
           n_b,
           [&prob, mode, batches, b, ops] { enqueue_chopped(prob, mode, batches, b, ops); },
           {},
           {}});
      const RingWork work{2, false, true, b, nullptr, {}};
      runs.push_back({Variant::kEvents,
                      mode,
                      0,
                      size,
                      n_b,
                      [&ring, &prob, mode, batches, work, ops] {
                        enqueue_ring(ring, prob, mode, batches, work, ops);
                      },
                      {},
                      {}});
      graphs.push_back(
          capture(res.stream(), [&] { enqueue_chopped(prob, mode, batches, b, ops); }));
      const wwr::wwrGraphExec_t exec = graphs.back()->get();
      const wwr::wwrStream_t stream = res.stream();
      runs.push_back({Variant::kGraph,
                      mode,
                      0,
                      size,
                      n_b,
                      [exec, stream] { check_gpu(wwr::wwrGraphLaunch(exec, stream), "launching"); },
                      {},
                      {}});
    }
  }
  check_runs(res, prob, runs, ref, ws.k.data());
  time_round_robin(timer, runs, settings);

  // T_1 is the chopped m curve's n_b = 1 point, from the same passes.
  double t1 = 0.0;
  for (const Run &run : runs) {
    if (run.variant == Variant::kChopped && run.mode == Mode::kSlab && run.n_b == 1) {
      t1 = median(run);
    }
  }
  report("resident", prob, runs, t1, csv);
}

// pipeline and copies over both modes and both ring depths, B in pinned memory.
// Each config's ring must fit in `settings.ring_bytes`; one that does not is
// skipped and said to be.
void pipeline_group(fb::DeviceResources &res, Ring &ring, Timer &timer, const Problem &prob,
                    Workspace &ws, double *const arena, const double *const host_slabs,
                    double *const host_blocks, const std::vector<double> &ref, const double t1,
                    const Settings &settings, std::ofstream &csv) {
  const Operands ops = ws.ops(res);
  const std::size_t arena_count = settings.ring_bytes / sizeof(double);
  std::vector<Run> all;

  // The slot pointers outlive every closure that reads them.
  std::vector<std::unique_ptr<std::array<double *, kMaxRing>>> slot_sets;
  const auto add = [&](std::vector<Run> &runs, const Mode mode, const std::int32_t size,
                       const std::int32_t depth, const double *const host) {
    const std::int64_t slot = std::int64_t{size} * (mode == Mode::kSlab ? prob.slab() : prob.nn());
    if (static_cast<std::size_t>(slot * depth) > arena_count) {
      std::println("  skipped: {} b {} R{} needs {:.1f} GiB of ring", mode_name(mode), size, depth,
                   static_cast<double>(slot * depth) * sizeof(double) / (1U << 30U));
      return;
    }
    auto &slots = *slot_sets.emplace_back(std::make_unique<std::array<double *, kMaxRing>>());
    for (std::int32_t s = 0; s < kMaxRing; ++s) {
      slots[static_cast<std::size_t>(s)] = s < depth ? arena + s * slot : nullptr;
    }
    const std::vector<Batch> batches = batches_of(prob, mode, size);
    const auto n_b = static_cast<std::int64_t>(batches.size());
    const std::span<double *const> view{slots.data(), static_cast<std::size_t>(depth)};
    for (const bool gemms : {true, false}) {
      const RingWork work{depth, true, gemms, nullptr, host, view};
      runs.push_back({gemms ? Variant::kPipeline : Variant::kCopies,
                      mode,
                      depth,
                      size,
                      n_b,
                      [&ring, &prob, mode, batches, work, ops] {
                        enqueue_ring(ring, prob, mode, batches, work, ops);
                      },
                      {},
                      {}});
    }
  };

  // m: one host layout serves every batch size, so all of it is one
  // round-robin.
  {
    std::vector<Run> runs;
    for (const std::int32_t size : slab_batches(prob.nao)) {
      for (const std::int32_t depth : {2, 3}) {
        add(runs, Mode::kSlab, size, depth, host_slabs);
      }
    }
    check_runs(res, prob, runs, ref, ws.k.data());
    time_round_robin(timer, runs, settings);
    std::ranges::move(runs, std::back_inserter(all));
  }

  // P: each batch size lays the host copy out again, so each is its own
  // round-robin.
  for (const std::int32_t size : aux_batches(prob.naux)) {
    std::vector<Run> runs;
    for (const std::int32_t depth : {2, 3}) {
      add(runs, Mode::kAux, size, depth, host_blocks);
    }
    if (runs.empty()) {
      continue;
    }
    fill_aux_blocks(host_blocks, prob.nao, prob.naux, size);
    check_runs(res, prob, runs, ref, ws.k.data());
    time_round_robin(timer, runs, settings);
    std::ranges::move(runs, std::back_inserter(all));
  }
  report("pipeline", prob, all, t1, csv);
}

std::vector<std::int32_t> parse_list(const std::string_view s) {
  std::vector<std::int32_t> out;
  for (const auto part : std::views::split(s, ',')) {
    std::int32_t v = 0;
    const std::string_view p{part.begin(), part.end()};
    check(std::from_chars(p.data(), p.data() + p.size(), v).ec == std::errc{}, "parsing a list");
    out.push_back(v);
  }
  return out;
}

std::int32_t parse_int(const std::string_view s) {
  std::int32_t v = 0;
  check(std::from_chars(s.data(), s.data() + s.size(), v).ec == std::errc{}, "parsing a number");
  return v;
}

Settings parse(const std::span<char *const> args) {
  Settings settings;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view flag{args[i]};
    check(i + 1 < args.size(), "a flag's value");
    const std::string_view value{args[++i]};
    if (flag == "--reps") {
      settings.reps = parse_int(value);
    } else if (flag == "--warmup") {
      settings.warmup = parse_int(value);
    } else if (flag == "--nao") {
      settings.naos = parse_list(value);
    } else if (flag == "--only") {
      check(value == "resident" || value == "pipeline", "--only resident|pipeline");
      settings.resident = value == "resident";
      settings.pipeline = value == "pipeline";
    } else if (flag == "--ring-gib") {
      settings.ring_bytes = static_cast<std::size_t>(parse_int(value)) << 30U;
    } else if (flag == "--csv") {
      settings.csv = value;
    } else {
      check(false, std::format("recognising {}", flag));
    }
  }
  check(settings.reps > 0 && settings.warmup >= 0, "reps > 0 and warmup >= 0");
  return settings;
}

// The resident sweep's nocc values, and the pipeline's, on its one shape: one
// above the 3080's break-even (gemms limit) and one below it (copies limit).
constexpr std::array kResidentNocc{64, 128};
constexpr std::array kPipelineNocc{32, 128};
constexpr std::int32_t kPipelineNao = 640;

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const Settings settings = parse(std::span<char *const>{argv, static_cast<std::size_t>(argc)});
  fb::DeviceResources res;
  Ring ring{res};
  Timer timer{res};
  std::ofstream csv{settings.csv};
  check(csv.good(), "opening the csv");
  std::println(csv, "group,nao,naux,nocc,variant,mode,depth,batch,n_b,rep,ms");
  const auto t_start = std::chrono::steady_clock::now();
  const auto elapsed = [t_start] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  };

  std::int64_t max_nn_aux = 0;
  for (const std::int32_t nao : settings.naos) {
    max_nn_aux = std::max(max_nn_aux, std::int64_t{nao} * nao * 3 * nao);
  }
  std::println("reps={} warmup={} ring={} GiB csv={}", settings.reps, settings.warmup,
               settings.ring_bytes >> 30U, settings.csv);
  fb::Buffers::pinned<double> host_slabs(static_cast<std::size_t>(max_nn_aux));

  for (const std::int32_t nao : settings.naos) {
    const std::int32_t naux = 3 * nao;
    const bool pipelined = settings.pipeline && nao == kPipelineNao;
    std::vector<std::int32_t> noccs;
    if (settings.resident) {
      noccs.insert(noccs.end(), kResidentNocc.begin(), kResidentNocc.end());
    }
    if (pipelined) {
      noccs.insert(noccs.end(), kPipelineNocc.begin(), kPipelineNocc.end());
    }
    std::ranges::sort(noccs);
    noccs.erase(std::ranges::unique(noccs).begin(), noccs.end());
    if (noccs.empty()) {
      continue;
    }
    const Problem largest{nao, naux, noccs.back()};
    std::println(std::cerr, "[{:7.1f}s] nao {}: filling and uploading {:.2f} GiB of B", elapsed(),
                 nao, static_cast<double>(largest.slab() * nao) * sizeof(double) / (1U << 30U));
    fill_slabs(host_slabs.data(), nao, naux);

    Workspace ws{res, largest};
    std::map<std::int32_t, std::vector<double>> refs;
    std::map<std::int32_t, double> t1s;
    {
      fb::Buffers::device<double> b{static_cast<std::size_t>(largest.slab() * nao), res.device()};
      check_gpu(wwr::wwrMemcpyAsync(b.data(), host_slabs.data(), b.num_elements() * sizeof(double),
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                "uploading B");
      check_gpu(wwr::wwrStreamSynchronize(res.stream()), "synchronizing B");

      // The one-call build for every nocc this shape needs: the reference K,
      // and a standalone T_1 for the pipeline group, which runs after B is
      // freed.
      for (const std::int32_t nocc : noccs) {
        const Problem prob{nao, naux, nocc};
        ws.upload_c(res, prob);
        const Operands ops = ws.ops(res);
        const std::vector<Batch> one = batches_of(prob, Mode::kSlab, nao);
        const std::function<void()> build = [&] {
          enqueue_chopped(prob, Mode::kSlab, one, b.data(), ops);
        };
        build();
        refs[nocc] = download_k(res, prob, ws.k.data());
        std::vector<float> ms;
        for (std::int32_t r = 0; r < settings.warmup + settings.reps; ++r) {
          const float t = timer.time(build);
          if (r >= settings.warmup) {
            ms.push_back(t);
          }
        }
        t1s[nocc] = percentile(ms, 0.5);
      }

      if (settings.resident) {
        for (const std::int32_t nocc : kResidentNocc) {
          const Problem prob{nao, naux, nocc};
          std::println(std::cerr, "[{:7.1f}s] resident nao {} nocc {}", elapsed(), nao, nocc);
          ws.upload_c(res, prob);
          resident_group(res, ring, timer, prob, b.data(), ws, refs[nocc], settings, csv);
        }
      }
    }

    if (pipelined) {
      fb::Buffers::pinned<double> host_blocks(static_cast<std::size_t>(largest.slab() * nao));
      fb::Buffers::device<double> arena{settings.ring_bytes / sizeof(double), res.device()};
      for (const std::int32_t nocc : kPipelineNocc) {
        const Problem prob{nao, naux, nocc};
        std::println(std::cerr, "[{:7.1f}s] pipeline nao {} nocc {}", elapsed(), nao, nocc);
        ws.upload_c(res, prob);
        pipeline_group(res, ring, timer, prob, ws, arena.data(), host_slabs.data(),
                       host_blocks.data(), refs[nocc], t1s[nocc], settings, csv);
      }
    }
  }
  std::println(std::cerr, "[{:7.1f}s] done", elapsed());
  return 0;
}
