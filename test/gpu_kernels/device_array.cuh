// Launcher plumbing shared by the .cu files in this directory: a device
// allocation that frees itself, and the post-launch error check. Device-only,
// like the launchers that use it: the wwr* runtime names come from runtime.h's
// device section.

#pragma once

#include "integral_types.h"

// wwr's runtime surface for a device TU -- wwr::wwrMalloc, wwrMemcpy, ... --
// so the launchers need no backend #if.
#include "runtime.h"

namespace free_breakfast::test_kernels {

/// @brief A device allocation that frees itself, so a launcher's early return on
/// a failed call does not leak the arrays already allocated
template <typename T>
class DeviceArray {
public:
  /// @brief Allocate @p count elements; ok() reports whether it succeeded
  explicit DeviceArray(const size_t count) : bytes_(count * sizeof(T)) {
    void *ptr = nullptr;
    ok_ = wwr::wwrMalloc(&ptr, bytes_) == wwr::wwrSuccess;
    data_ = static_cast<T *>(ptr);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (data_ != nullptr) {
      static_cast<void>(wwr::wwrFree(data_));
    }
  }

  /// @brief Whether the allocation succeeded
  [[nodiscard]] bool ok() const { return ok_; }
  /// @brief The device pointer
  [[nodiscard]] T *data() const { return data_; }

  /// @brief Copy the whole array in from @p host
  [[nodiscard]] bool upload(const T *host) const {
    return wwr::wwrMemcpy(data_, host, bytes_, wwr::wwrMemcpyHostToDevice) == wwr::wwrSuccess;
  }
  /// @brief Copy the whole array out to @p host
  [[nodiscard]] bool download(T *host) const {
    return wwr::wwrMemcpy(host, data_, bytes_, wwr::wwrMemcpyDeviceToHost) == wwr::wwrSuccess;
  }

private:
  size_t bytes_;
  T *data_ = nullptr;
  bool ok_ = false;
};

/// @brief Whether the last launch, and the kernel it ran, succeeded
///
/// The launch itself reports nothing; a bad configuration surfaces from
/// wwrGetLastError, and a fault inside the kernel from the synchronize.
inline bool launched_cleanly() {
  return wwr::wwrGetLastError() == wwr::wwrSuccess &&
         wwr::wwrDeviceSynchronize() == wwr::wwrSuccess;
}

} // namespace free_breakfast::test_kernels
