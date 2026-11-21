/// @file
/// @brief Does a vendor library allocate device memory behind a caller-sized
/// workspace? One case per process: a fresh context, a fresh handle, the card's
/// free memory read around each step. README.md has the question and results.
///
///   free_breakfast_library_handle_workspace <routine> <mode>
///
/// Solver routines (syevd, potrf, gesvd) take mode exact, short (half the
/// queried lwork, in a buffer of that size) or null (work = nullptr, lwork 0),
/// or exact_guarded / short_guarded: the same lwork passed, in a buffer twice
/// the queried size filled with a sentinel, so touched_mib reports how far the
/// call wrote, past the length it was given or not.
/// BLAS routines (dgemm_square, dgemm_splitk) take mode default (no workspace
/// set), user (a 32 MiB workspace set), tiny (a 256-byte workspace set) or
/// reset (a 32 MiB workspace set, then set back to nullptr, 0).
///
/// Prints one line of key=value pairs. Deltas are the drop in free device
/// memory over the step, in MiB: positive means the step took memory.

import std;
import wwr.runtime_api;
import wwr.blas;
import wwr.solver;

namespace {

constexpr std::int32_t kSolverOrder = 2048;
constexpr std::size_t kUserWorkspaceBytes = std::size_t{32} << 20;
constexpr std::size_t kTinyWorkspaceBytes = 256;

[[noreturn]] void die(const std::string &what) {
  std::println(std::cerr, "library_handle_workspace: {}", what);
  std::exit(2);
}

void check_rt(wwr::wwrError_t e, const char *what) {
  if (e != wwr::wwrSuccess) {
    die(std::format("{}: {}", what, wwr::wwrGetErrorString(e)));
  }
}

// Free device memory after the device has drained, in bytes.
std::int64_t free_bytes() {
  check_rt(wwr::wwrDeviceSynchronize(), "synchronize");
  std::size_t free = 0;
  std::size_t total = 0;
  check_rt(wwr::wwrMemGetInfo(&free, &total), "mem_get_info");
  return static_cast<std::int64_t>(free);
}

double mib(std::int64_t before, std::int64_t after) {
  return static_cast<double>(before - after) / double(1 << 20);
}

template <class T>
T *device_alloc(std::size_t count) {
  void *p = nullptr;
  if (count > 0) {
    check_rt(wwr::wwrMalloc(&p, count * sizeof(T)), "malloc");
  }
  return static_cast<T *>(p);
}

template <class T>
T *upload(const std::vector<T> &h) {
  T *d = device_alloc<T>(h.size());
  check_rt(
      wwr::wwrMemcpyAsync(d, h.data(), h.size() * sizeof(T), wwr::wwrMemcpyHostToDevice, nullptr),
      "upload");
  check_rt(wwr::wwrStreamSynchronize(nullptr), "upload");
  return d;
}

template <class T>
std::vector<T> download(const T *d, std::size_t count) {
  std::vector<T> h(count);
  check_rt(wwr::wwrMemcpyAsync(h.data(), d, count * sizeof(T), wwr::wwrMemcpyDeviceToHost, nullptr),
           "download");
  check_rt(wwr::wwrStreamSynchronize(nullptr), "download");
  return h;
}

std::int32_t download_info(const std::int32_t *d) {
  return download(d, 1)[0];
}

double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

// A symmetric positive-definite n x n matrix, column-major: M M^T / n + I.
std::vector<double> spd_matrix(std::int32_t n) {
  std::mt19937_64 gen{12345};
  std::uniform_real_distribution<double> u{-1.0, 1.0};
  std::vector<double> m(std::size_t(n) * n);
  for (auto &x : m) {
    x = u(gen);
  }
  std::vector<double> a(std::size_t(n) * n, 0.0);
  for (std::int32_t j = 0; j < n; ++j) {
    for (std::int32_t i = j; i < n; ++i) {
      double s = 0.0;
      for (std::int32_t k = 0; k < n; ++k) {
        s += m[std::size_t(k) * n + i] * m[std::size_t(k) * n + j];
      }
      s /= n;
      if (i == j) {
        s += 1.0;
      }
      a[std::size_t(j) * n + i] = s;
      a[std::size_t(i) * n + j] = s;
    }
  }
  return a;
}

std::vector<double> random_matrix(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 gen{seed};
  std::uniform_real_distribution<double> u{-1.0, 1.0};
  std::vector<double> m(count);
  for (auto &x : m) {
    x = u(gen);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Solver routines. Each is three calls over one handle: the size query, the
// call with a given work buffer and length, and a fetch of the output to
// compare.
// ---------------------------------------------------------------------------

struct SolverRoutine {
  // Returns lwork in elements.
  std::function<std::int32_t(wwr::wwrsolverDnHandle_t)> query;
  // Runs on a fresh copy of the input; returns the status, writes the output
  // to compare into `out` and devInfo into `info`.
  std::function<wwr::wwrsolverStatus_t(wwr::wwrsolverDnHandle_t, double *work, std::int32_t lwork,
                                       std::vector<double> &out, std::int32_t &info)>
      call;
};

SolverRoutine make_solver_routine(const std::string &name, const std::vector<double> &a_host) {
  const std::int32_t n = kSolverOrder;
  auto dev_info = std::make_shared<std::int32_t *>(device_alloc<std::int32_t>(1));
  if (name == "syevd") {
    return {[n](wwr::wwrsolverDnHandle_t h) {
              std::int32_t lwork = 0;
              auto s = wwr::wwrsolverDnDsyevd_bufferSize(h, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                                         wwr::WWRSOLVER_FILL_MODE_LOWER, n, nullptr,
                                                         n, nullptr, &lwork);
              if (s != wwr::WWRSOLVER_STATUS_SUCCESS) {
                die(std::format("syevd_bufferSize: {}", wwr::wwrsolverGetStatusName(s)));
              }
              return lwork;
            },
            [n, &a_host, dev_info](wwr::wwrsolverDnHandle_t h, double *work, std::int32_t lwork,
                                   std::vector<double> &out, std::int32_t &info) {
              double *a = upload(a_host);
              double *w = device_alloc<double>(n);
              auto s = wwr::wwrsolverDnDsyevd(h, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                                              wwr::WWRSOLVER_FILL_MODE_LOWER, n, a, n, w, work,
                                              lwork, *dev_info);
              check_rt(wwr::wwrDeviceSynchronize(), "syevd");
              out = download(w, n);
              info = download_info(*dev_info);
              check_rt(wwr::wwrFree(a), "free");
              check_rt(wwr::wwrFree(w), "free");
              return s;
            }};
  }
  if (name == "potrf") {
    return {[n](wwr::wwrsolverDnHandle_t h) {
              std::int32_t lwork = 0;
              auto s = wwr::wwrsolverDnDpotrf_bufferSize(h, wwr::WWRSOLVER_FILL_MODE_LOWER, n,
                                                         nullptr, n, &lwork);
              if (s != wwr::WWRSOLVER_STATUS_SUCCESS) {
                die(std::format("potrf_bufferSize: {}", wwr::wwrsolverGetStatusName(s)));
              }
              return lwork;
            },
            [n, &a_host, dev_info](wwr::wwrsolverDnHandle_t h, double *work, std::int32_t lwork,
                                   std::vector<double> &out, std::int32_t &info) {
              double *a = upload(a_host);
              auto s = wwr::wwrsolverDnDpotrf(h, wwr::WWRSOLVER_FILL_MODE_LOWER, n, a, n, work,
                                              lwork, *dev_info);
              check_rt(wwr::wwrDeviceSynchronize(), "potrf");
              out = download(a, std::size_t(n) * n);
              // Only the lower triangle is the factor.
              for (std::int32_t j = 0; j < n; ++j) {
                for (std::int32_t i = 0; i < j; ++i) {
                  out[std::size_t(j) * n + i] = 0.0;
                }
              }
              info = download_info(*dev_info);
              check_rt(wwr::wwrFree(a), "free");
              return s;
            }};
  }
  if (name == "gesvd") {
    return {[n](wwr::wwrsolverDnHandle_t h) {
              std::int32_t lwork = 0;
              auto s = wwr::wwrsolverDnDgesvd_bufferSize(h, n, n, &lwork);
              if (s != wwr::WWRSOLVER_STATUS_SUCCESS) {
                die(std::format("gesvd_bufferSize: {}", wwr::wwrsolverGetStatusName(s)));
              }
              return lwork;
            },
            [n, &a_host, dev_info](wwr::wwrsolverDnHandle_t h, double *work, std::int32_t lwork,
                                   std::vector<double> &out, std::int32_t &info) {
              double *a = upload(a_host);
              double *sv = device_alloc<double>(n);
              double *rwork = device_alloc<double>(n);
              auto s = wwr::wwrsolverDnDgesvd(h, 'N', 'N', n, n, a, n, sv, nullptr, n, nullptr, n,
                                              work, lwork, rwork, *dev_info);
              check_rt(wwr::wwrDeviceSynchronize(), "gesvd");
              out = download(sv, n);
              info = download_info(*dev_info);
              check_rt(wwr::wwrFree(a), "free");
              check_rt(wwr::wwrFree(sv), "free");
              check_rt(wwr::wwrFree(rwork), "free");
              return s;
            }};
  }
  die("unknown solver routine " + name);
}

// The highest element of a sentinel-filled buffer the library wrote, plus
// one: how much of it the call touched.
std::size_t touched_elements(const double *work, std::size_t count) {
  const auto h = download(work, count);
  for (std::size_t i = count; i > 0; --i) {
    if (std::bit_cast<std::uint64_t>(h[i - 1]) != ~std::uint64_t{0}) {
      return i;
    }
  }
  return 0;
}

std::int32_t run_solver(const std::string &routine, const std::string &mode) {
  const bool guarded = mode.ends_with("_guarded");
  if (mode != "exact" && mode != "short" && mode != "null" && mode != "exact_guarded" &&
      mode != "short_guarded") {
    die("solver mode must be exact, short, null, exact_guarded or short_guarded");
  }
  const auto a_host = routine == "gesvd"
                          ? random_matrix(std::size_t(kSolverOrder) * kSolverOrder, 7)
                          : spd_matrix(kSolverOrder);
  const SolverRoutine r = make_solver_routine(routine, a_host);

  wwr::wwrStream_t stream = nullptr;
  check_rt(wwr::wwrStreamCreate(&stream), "stream create");

  const std::int64_t f0 = free_bytes();
  wwr::wwrsolverDnHandle_t h = nullptr;
  if (wwr::wwrsolverDnCreate(&h) != wwr::WWRSOLVER_STATUS_SUCCESS ||
      wwr::wwrsolverDnSetStream(h, stream) != wwr::WWRSOLVER_STATUS_SUCCESS) {
    die("solver handle");
  }
  const std::int64_t f1 = free_bytes();
  const std::int32_t lwork = r.query(h);
  const std::int64_t f2 = free_bytes();

  const std::int32_t pass_lwork = mode.starts_with("exact")   ? lwork
                                  : mode.starts_with("short") ? lwork / 2
                                                              : 0;
  // A guarded buffer is twice the queried size and filled with a sentinel, so
  // a write past the length passed lands in memory we own and can be seen.
  const std::size_t work_count =
      guarded ? 2 * static_cast<std::size_t>(lwork) : static_cast<std::size_t>(pass_lwork);
  double *work = device_alloc<double>(work_count);
  if (guarded) {
    check_rt(wwr::wwrMemset(work, 0xFF, work_count * sizeof(double)), "memset");
  }
  const std::int64_t f3 = free_bytes();

  std::vector<double> out1;
  std::int32_t info1 = 0;
  const auto s1 = r.call(h, work, pass_lwork, out1, info1);
  const std::int64_t f4 = free_bytes();
  std::vector<double> out2;
  std::int32_t info2 = 0;
  const auto s2 = r.call(h, work, pass_lwork, out2, info2);
  const std::int64_t f5 = free_bytes();
  // -1 when unguarded: nothing to read the high-water mark from.
  const double touched_mib = guarded ? touched_elements(work, work_count) * 8.0 / (1 << 20) : -1.0;

  // The reference: a second, exact-sized buffer on the same handle, after
  // every reading.
  double *ref_work = device_alloc<double>(static_cast<std::size_t>(lwork));
  std::vector<double> ref;
  std::int32_t ref_info = 0;
  const auto sr = r.call(h, ref_work, lwork, ref, ref_info);
  if (sr != wwr::WWRSOLVER_STATUS_SUCCESS || ref_info != 0) {
    die(std::format("reference call failed: {} info={}", wwr::wwrsolverGetStatusName(sr),
                    ref_info));
  }
  const double diff = s1 == wwr::WWRSOLVER_STATUS_SUCCESS ? max_abs_diff(out1, ref) : -1.0;

  std::println("routine={} mode={} lwork_mib={:.2f} passed_mib={:.2f} handle_mib={:.2f} "
               "query_mib={:.2f} buffer_mib={:.2f} call1_status={} call1_info={} call1_mib={:.2f} "
               "call2_status={} call2_info={} call2_mib={:.2f} touched_mib={:.2f} "
               "max_diff_vs_exact={:.3g}",
               routine, mode, lwork * 8.0 / (1 << 20), pass_lwork * 8.0 / (1 << 20), mib(f0, f1),
               mib(f1, f2), mib(f2, f3), wwr::wwrsolverGetStatusName(s1), info1, mib(f3, f4),
               wwr::wwrsolverGetStatusName(s2), info2, mib(f4, f5), touched_mib, diff);
  return 0;
}

// ---------------------------------------------------------------------------
// BLAS: dgemm, C = A B, in two shapes. The handle's workspace is the variable.
// ---------------------------------------------------------------------------

std::int32_t run_blas(const std::string &routine, const std::string &mode) {
  std::int32_t m = 0;
  std::int32_t n = 0;
  std::int32_t k = 0;
  if (routine == "dgemm_square") {
    m = n = k = 4096;
  } else if (routine == "dgemm_splitk") {
    m = n = 256;
    k = 1 << 16;
  } else {
    die("unknown routine " + routine);
  }
  if (mode != "default" && mode != "user" && mode != "tiny" && mode != "reset") {
    die("blas mode must be default, user, tiny or reset");
  }
  const auto a_host = random_matrix(std::size_t(m) * k, 1);
  const auto b_host = random_matrix(std::size_t(k) * n, 2);
  double *a = upload(a_host);
  double *b = upload(b_host);
  double *c = device_alloc<double>(std::size_t(m) * n);
  const double one = 1.0;
  const double zero = 0.0;

  wwr::wwrStream_t stream = nullptr;
  check_rt(wwr::wwrStreamCreate(&stream), "stream create");

  auto gemm = [&](wwr::wwrblasHandle_t h, double *out) {
    return wwr::wwrblasDgemm(h, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, k, &one, a, m, b, k,
                             &zero, out, m);
  };

  const std::int64_t f0 = free_bytes();
  wwr::wwrblasHandle_t h = nullptr;
  if (wwr::wwrblasCreate(&h) != wwr::WWRBLAS_STATUS_SUCCESS ||
      wwr::wwrblasSetStream(h, stream) != wwr::WWRBLAS_STATUS_SUCCESS) {
    die("blas handle");
  }
  const std::int64_t f1 = free_bytes();

  void *ws = nullptr;
  std::size_t ws_bytes = 0;
  if (mode == "user" || mode == "reset") {
    ws_bytes = kUserWorkspaceBytes;
  } else if (mode == "tiny") {
    ws_bytes = kTinyWorkspaceBytes;
  }
  wwr::wwrblasStatus_t set_status = wwr::WWRBLAS_STATUS_SUCCESS;
  if (ws_bytes > 0) {
    check_rt(wwr::wwrMalloc(&ws, ws_bytes), "workspace malloc");
    set_status = wwr::wwrblasSetWorkspace(h, ws, ws_bytes);
  }
  if (mode == "reset") {
    set_status = wwr::wwrblasSetWorkspace(h, nullptr, 0);
  }
  const std::int64_t f2 = free_bytes();

  const auto s1 = gemm(h, c);
  const std::int64_t f3 = free_bytes();
  const auto s2 = gemm(h, c);
  const std::int64_t f4 = free_bytes();

  // The reference: a second, default handle, after every reading.
  double *c_ref = device_alloc<double>(std::size_t(m) * n);
  wwr::wwrblasHandle_t href = nullptr;
  if (wwr::wwrblasCreate(&href) != wwr::WWRBLAS_STATUS_SUCCESS ||
      wwr::wwrblasSetStream(href, stream) != wwr::WWRBLAS_STATUS_SUCCESS ||
      gemm(href, c_ref) != wwr::WWRBLAS_STATUS_SUCCESS) {
    die("reference gemm");
  }
  check_rt(wwr::wwrDeviceSynchronize(), "reference gemm");
  const double diff =
      s1 == wwr::WWRBLAS_STATUS_SUCCESS
          ? max_abs_diff(download(c, std::size_t(m) * n), download(c_ref, std::size_t(m) * n))
          : -1.0;

  std::println("routine={} mode={} m={} n={} k={} handle_mib={:.2f} set_workspace={} "
               "set_mib={:.2f} call1_status={} call1_mib={:.2f} call2_status={} call2_mib={:.2f} "
               "max_diff_vs_default={:.3g}",
               routine, mode, m, n, k, mib(f0, f1), wwr::wwrblasGetStatusName(set_status),
               mib(f1, f2), wwr::wwrblasGetStatusName(s1), mib(f2, f3),
               wwr::wwrblasGetStatusName(s2), mib(f3, f4), diff);
  return 0;
}

} // namespace

// int-type-ok: main's signature is the standard's.
int main(int argc, char **argv) {
  if (argc != 3) {
    die("usage: <routine> <mode>");
  }
  const std::string routine = argv[1];
  const std::string mode = argv[2];
  check_rt(wwr::wwrSetDevice(0), "set device");
  // Create the context and load the runtime before any reading.
  check_rt(wwr::wwrFree(nullptr), "context");
  if (routine.starts_with("dgemm")) {
    return run_blas(routine, mode);
  }
  return run_solver(routine, mode);
}
