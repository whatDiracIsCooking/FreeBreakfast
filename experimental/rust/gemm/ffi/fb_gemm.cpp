/// @file fb_gemm.cpp
/// @brief fb_gemm.h over DeviceResources, Buffers::device<double> and
/// wwr::gemm. Not a module unit: a plain TU that imports modules and exports
/// C symbols, built into a static archive the Rust crate links.
///
/// Every function is noexcept (FB_NOEXCEPT in the header): a C++ exception unwinding into
/// a Rust frame is undefined behaviour, so one that escapes -- in practice
/// only std::bad_alloc from a `new` -- calls std::terminate here instead, as a
/// failed GPU create or alloc aborts everywhere in free_breakfast.
#include "fb_gemm.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

namespace fb = free_breakfast;

struct fb_resources {
  fb::DeviceResources res;
};

struct fb_buffer {
  fb::Buffers::device<double> data;
};

static fb_status from_gpu(const wwr::wwrError_t err) noexcept {
  return err == wwr::wwrSuccess ? FB_OK : FB_GPU_ERROR;
}

/// Whether a column-major rows x cols matrix with leading dimension ld fits
/// in `buf`.
static bool holds(const fb_buffer *buf, const int32_t rows, const int32_t cols,
                  const int32_t ld) noexcept {
  if (buf == nullptr || rows < 0 || cols < 0 || ld < std::max(rows, 1)) {
    return false;
  }
  if (rows == 0 || cols == 0) {
    return true;
  }
  const auto need =
      static_cast<size_t>(ld) * static_cast<size_t>(cols - 1) + static_cast<size_t>(rows);
  return need <= buf->data.num_elements();
}

static bool valid_op(const fb_op op) noexcept {
  return op == FB_OP_N || op == FB_OP_T;
}

static wwr::wwrblasOperation_t to_blas(const fb_op op) noexcept {
  return op == FB_OP_T ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
}

/// Queue a copy of `bytes` on `res`'s stream, and wait for it.
static fb_status copy_and_wait(fb_resources *const res, void *const dst, const void *const src,
                               const size_t bytes, const wwr::wwrMemcpyKind kind) noexcept {
  if (const fb_status s = from_gpu(wwr::wwrMemcpyAsync(dst, src, bytes, kind, res->res.stream()));
      s != FB_OK) {
    return s;
  }
  return from_gpu(wwr::wwrStreamSynchronize(res->res.stream()));
}

extern "C" {

fb_status fb_resources_new(const int32_t dev_idx, fb_resources **const out) noexcept {
  if (out == nullptr || dev_idx < 0) {
    return FB_INVALID_ARGUMENT;
  }
  *out = new fb_resources{fb::DeviceResources(dev_idx)};
  return FB_OK;
}

void fb_resources_free(fb_resources *const res) noexcept {
  delete res;
}

fb_status fb_resources_sync(fb_resources *const res) noexcept {
  if (res == nullptr) {
    return FB_INVALID_ARGUMENT;
  }
  return from_gpu(wwr::wwrStreamSynchronize(res->res.stream()));
}

fb_status fb_buffer_new(fb_resources *const res, const size_t len, fb_buffer **const out) noexcept {
  if (res == nullptr || out == nullptr) {
    return FB_INVALID_ARGUMENT;
  }
  *out = new fb_buffer{fb::Buffers::device<double>(len, res->res.device())};
  return FB_OK;
}

void fb_buffer_free(fb_buffer *const buf) noexcept {
  delete buf;
}

size_t fb_buffer_len(const fb_buffer *const buf) noexcept {
  return buf == nullptr ? 0 : buf->data.num_elements();
}

fb_status fb_buffer_upload(fb_resources *const res, fb_buffer *const buf, const double *const src,
                           const size_t len) noexcept {
  if (res == nullptr || buf == nullptr || (src == nullptr && len != 0) ||
      len > buf->data.num_elements()) {
    return FB_INVALID_ARGUMENT;
  }
  return copy_and_wait(res, buf->data.data(), src, len * sizeof(double),
                       wwr::wwrMemcpyHostToDevice);
}

fb_status fb_buffer_download(fb_resources *const res, const fb_buffer *const buf, double *const dst,
                             const size_t len) noexcept {
  if (res == nullptr || buf == nullptr || (dst == nullptr && len != 0) ||
      len > buf->data.num_elements()) {
    return FB_INVALID_ARGUMENT;
  }
  return copy_and_wait(res, dst, buf->data.data(), len * sizeof(double),
                       wwr::wwrMemcpyDeviceToHost);
}

fb_status fb_dgemm(fb_resources *const res, const fb_op op_a, const fb_op op_b, const int32_t m,
                   const int32_t n, const int32_t k, const double alpha, const fb_buffer *const a,
                   const int32_t lda, const fb_buffer *const b, const int32_t ldb,
                   const double beta, fb_buffer *const c, const int32_t ldc) noexcept {
  if (res == nullptr || !valid_op(op_a) || !valid_op(op_b) || m < 0 || n < 0 || k < 0) {
    return FB_INVALID_ARGUMENT;
  }
  // A as stored: m x k under N, k x m under T; likewise B.
  const bool a_ok = op_a == FB_OP_N ? holds(a, m, k, lda) : holds(a, k, m, lda);
  const bool b_ok = op_b == FB_OP_N ? holds(b, k, n, ldb) : holds(b, n, k, ldb);
  if (!a_ok || !b_ok || !holds(c, m, n, ldc)) {
    return FB_INVALID_ARGUMENT;
  }
  const wwr::wwrblasHandle_t handle = res->res.blas().get();
  // alpha and beta live on the host; borrowed for this call, as gemm_fp16 does.
  const wwr::extension::ScopedPointerMode<wwr::wwrblasHandle_t,
                                          fb::CreatePolicy<wwr::wwrblasStatus_t>>
      host_scalars{handle, wwr::WWRBLAS_POINTER_MODE_HOST};
  const wwr::wwrblasStatus_t status = wwr::gemm<double, int32_t>(
      handle, to_blas(op_a), to_blas(op_b), m, n, k, &alpha, a->data.data(), lda, b->data.data(),
      ldb, &beta, c->data.data(), ldc);
  return status == wwr::WWRBLAS_STATUS_SUCCESS ? FB_OK : FB_BLAS_ERROR;
}

} // extern "C"
