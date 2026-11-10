/// @file fb_gemm.h
/// @brief A C ABI over free_breakfast's DeviceResources, its device buffers
/// and an fp64 GEMM: the surface the Rust crate beside it binds.
///
/// Rust cannot import a C++ module or instantiate a template, so this is the
/// whole contract, in C types only. The crate's src/ffi.rs declares the same
/// functions by hand; keep the two in step.
///
/// Every function returns an fb_status. Handles are opaque. Matrices are
/// column-major, as BLAS's are. A failed GPU creation or allocation aborts the
/// process, as everywhere in free_breakfast (CreatePolicy). Nothing here
/// unwinds into the caller: every function is noexcept to C++, so an escaping
/// exception terminates the process rather than unwinding into a Rust frame.
#ifndef FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_GEMM_H
#define FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define FB_NOEXCEPT noexcept
extern "C" {
#else
#define FB_NOEXCEPT
#endif

/// Plain int32_t on the wire: an enum's width is not fixed in C.
typedef int32_t fb_status;
enum {
  FB_OK = 0,
  /// A null handle or pointer, a negative dimension, or a buffer too small.
  FB_INVALID_ARGUMENT = 1,
  /// The GPU runtime returned an error (a copy or a synchronize).
  FB_GPU_ERROR = 2,
  /// The BLAS call returned a status other than success.
  FB_BLAS_ERROR = 3,
  /// A free_breakfast routine refused or failed; fb_last_error() says why.
  FB_FAILED = 4,
};

typedef int32_t fb_op;
enum {
  FB_OP_N = 0,
  FB_OP_T = 1,
};

typedef struct fb_resources fb_resources;
typedef struct fb_buffer fb_buffer;

/// The message of the last FB_FAILED on this thread, or "" if there has been
/// none. Valid until the next call on this thread that fails.
const char *fb_last_error(void) FB_NOEXCEPT;

/// A DeviceResources on device `dev_idx`: a stream, its memory pool and a
/// BLAS handle bound to the stream.
fb_status fb_resources_new(int32_t dev_idx, fb_resources **out) FB_NOEXCEPT;
void fb_resources_free(fb_resources *res) FB_NOEXCEPT;
/// Block until everything queued on the stream has finished.
fb_status fb_resources_sync(fb_resources *res) FB_NOEXCEPT;

/// `len` doubles on the device, from `res`'s pool; uninitialised.
fb_status fb_buffer_new(fb_resources *res, size_t len, fb_buffer **out) FB_NOEXCEPT;
/// Free a buffer, stream-ordered after the work queued against it. `buf`
/// must not outlive the fb_resources it came from.
void fb_buffer_free(fb_buffer *buf) FB_NOEXCEPT;
size_t fb_buffer_len(const fb_buffer *buf) FB_NOEXCEPT;
/// Copy `len` doubles from host `src` into the buffer's start, and wait.
fb_status fb_buffer_upload(fb_resources *res, fb_buffer *buf, const double *src,
                           size_t len) FB_NOEXCEPT;
/// Copy `len` doubles from the buffer's start into host `dst`, and wait.
fb_status fb_buffer_download(fb_resources *res, const fb_buffer *buf, double *dst,
                             size_t len) FB_NOEXCEPT;

/// C = alpha op(A) op(B) + beta C, column-major, op(A) m x k, op(B) k x n,
/// C m x n, each with its leading dimension; queued on `res`'s stream, not
/// waited for. Each buffer is checked to hold its matrix.
fb_status fb_dgemm(fb_resources *res, fb_op op_a, fb_op op_b, int32_t m, int32_t n, int32_t k,
                   double alpha, const fb_buffer *a, int32_t lda, const fb_buffer *b, int32_t ldb,
                   double beta, fb_buffer *c, int32_t ldc) FB_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif // FREE_BREAKFAST_EXPERIMENTAL_RUST_GEMM_FB_GEMM_H
