/// @file
/// @brief Diis: the ring push, the incremental Gram, the device coefficient
/// solve and the extrapolation.

module;

// The solve launcher, defined in diis_solve.cu: declared in the GMF so it keeps
// external linkage (see the header).
#include "diis_solve_bridge.h"

module free_breakfast.recipes.diis;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

#include "diagnostics.h"

namespace free_breakfast {

// The solve's shared memory is sized for this history and no more.
static_assert(std::cmp_equal(kDiisMaxHistoryLimit + 1, device::kDiisSolveMaxDim));

Diis::Diis(DeviceResources &res, const size_t len, const size_t max_history)
    : len_(len), cap_(std::clamp<size_t>(max_history, 1, kDiisMaxHistoryLimit)),
      dev_idx_(res.dev_idx()), fock_(len_ * cap_, res.device()), error_(len_ * cap_, res.device()),
      gram_(cap_ * cap_, res.device()), coeff_(cap_, res.device()), singular_(1, res.device()) {}

Status Diis::push_and_extrapolate(DeviceResources &res, const double *cur_fock,
                                  const double *cur_residual, double *out_fock) {
  // Both checks come before any work, so a rejected call leaves the history as
  // it was.
  if (res.dev_idx() != dev_idx_) {
    return err_config("Diis: history is on device {}, but the call passed device {}", dev_idx_,
                      res.dev_idx());
  }
  if (len_ > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("Diis: vector length {} exceeds a 32-bit BLAS dimension", len_);
  }

  // Ring push: append while growing, then overwrite the oldest column and
  // advance head_. The valid columns are therefore always the prefix
  // [0, size) or the whole buffer, never a wrapped range, which is what lets
  // every gemv below see one base pointer and a fixed stride.
  const bool growing = size_ < cap_;
  const size_t slot = growing ? size_ : head_;
  const size_t m = growing ? size_ + 1 : cap_;
  const size_t next_head = growing ? head_ : (head_ + 1) % cap_;
  last_singular_ = false;

  if (len_ == 0) {
    size_ = m;
    head_ = next_head;
    return {};
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto len = static_cast<int32_t>(len_);
  const auto cap = static_cast<int32_t>(cap_);
  const size_t column_bytes = sizeof(double) * len_;

  // A failure part-way leaves a column or Gram entry half-written, so any
  // error drops the whole subspace rather than leave it inconsistent.
  const Status status = [&]() -> Status {
    double *const e_slot = error_.data() + (slot * len_);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(fock_.data() + (slot * len_), cur_fock, column_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(e_slot, cur_residual, column_bytes,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));

    // Incremental Gram: only <e_j, e_slot> changed, so recompute that row and
    // column straight into the upper triangle the solve reads, at
    // min(j, slot) * cap + max(j, slot). Two gemvs with trans = T:
    //   j = 0..slot   -> gram[j * cap + slot]  (base gram + slot, incy = cap)
    //   j = slot..m-1 -> gram[slot * cap + j]  (base gram + slot*cap + slot)
    // Both write the diagonal, with the same value. This runs on every push,
    // m == 1 included, so each entry [a][c] is rewritten by the push that
    // wrote column max(a, c) and reset() can never expose a stale one.
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_T, len, static_cast<int32_t>(slot + 1), &kOne<double>,
        error_.data(), len, e_slot, 1, &kZero<double>, gram_.data() + slot, cap)));
    if (m > slot + 1) {
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          res.blas().get(), wwr::WWRBLAS_OP_T, len, static_cast<int32_t>(m - slot), &kOne<double>,
          e_slot, len, e_slot, 1, &kZero<double>, gram_.data() + (slot * cap_) + slot, 1)));
    }
    size_ = m;
    head_ = next_head;

    if (m < 2) {
      return {};
    }

    FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
        device::diis_solve(stream, static_cast<int32_t>(m), cap, static_cast<int32_t>(slot),
                           gram_.data(), coeff_.data(), singular_.data()))));

    // F* = F_hist c over physical columns 0..m-1. The sum is order-invariant,
    // so the ring's rotated order needs no unscrambling. beta = 0 overwrites
    // out_fock, which may therefore alias cur_fock (already copied in).
    FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, len, static_cast<int32_t>(m), &kOne<double>,
        fock_.data(), len, coeff_.data(), 1, &kZero<double>, out_fock, 1)));

    // The one host round-trip, for last_extrapolation_singular(). The solve
    // itself needs none: a singular subspace already resolved to the newest
    // Fock on the device.
    int32_t singular_host = 0;
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&singular_host, singular_.data(), sizeof(int32_t),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));
    last_singular_ = singular_host != 0;
    return {};
  }();

  if (!status) {
    reset();
  }
  return status;
}

} // namespace free_breakfast
