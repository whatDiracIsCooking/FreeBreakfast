// Device side of test/gtest/primitives_gpu.cpp: kernels that evaluate every
// free_breakfast::math function, and the host launchers primitives_kernels.h
// declares. This is the tree's first .cu to include src/core/primitives/
// primitives.h, so it is also what proves that header's device half compiles --
// under nvcc in the CUDA tier, and under clang -x hip in
// devtools/cross-backend-check.sh, which builds this library even with
// FREE_BREAKFAST_COMPILE_TIME_ONLY.

// primitives.h first: it includes its standard headers ahead of the vendor
// runtime header, which under HIP must not precede libc++ (see its comments).
#include "primitives.h"

// wwr's runtime surface for a device TU -- wwr::wwrMalloc, wwrMemcpy, ... --
// so the launchers below need no backend #if.
#include "runtime.h"

#include "primitives_kernels.h"

namespace free_breakfast::test_kernels {

namespace {

constexpr int32_t kThreadsPerBlock = 256;

// One switch over every function, so each launch exercises exactly the name the
// test asked for. Device-only by declaration: the host never evaluates it.
template <typename T>
__device__ T apply(const MathOp op, const T x, const T y, const T z) {
  switch (op) {
  case MathOp::Exp:
    return math::exp(x);
  case MathOp::Exp2:
    return math::exp2(x);
  case MathOp::Expm1:
    return math::expm1(x);
  case MathOp::Log:
    return math::log(x);
  case MathOp::Log2:
    return math::log2(x);
  case MathOp::Log10:
    return math::log10(x);
  case MathOp::Log1p:
    return math::log1p(x);
  case MathOp::Pow:
    return math::pow(x, y);
  case MathOp::Sqrt:
    return math::sqrt(x);
  case MathOp::Rsqrt:
    return math::rsqrt(x);
  case MathOp::Sin:
    return math::sin(x);
  case MathOp::Cos:
    return math::cos(x);
  case MathOp::Tan:
    return math::tan(x);
  case MathOp::Asin:
    return math::asin(x);
  case MathOp::Acos:
    return math::acos(x);
  case MathOp::Atan:
    return math::atan(x);
  case MathOp::Atan2:
    return math::atan2(x, y);
  case MathOp::Sinh:
    return math::sinh(x);
  case MathOp::Cosh:
    return math::cosh(x);
  case MathOp::Tanh:
    return math::tanh(x);
  case MathOp::Floor:
    return math::floor(x);
  case MathOp::Ceil:
    return math::ceil(x);
  case MathOp::Trunc:
    return math::trunc(x);
  case MathOp::Round:
    return math::round(x);
  case MathOp::Rint:
    return math::rint(x);
  case MathOp::Fabs:
    return math::fabs(x);
  case MathOp::Hypot:
    return math::hypot(x, y);
  case MathOp::Copysign:
    return math::copysign(x, y);
  case MathOp::Fmod:
    return math::fmod(x, y);
  case MathOp::Fmin:
    return math::fmin(x, y);
  case MathOp::Fmax:
    return math::fmax(x, y);
  case MathOp::Fma:
    return math::fma(x, y, z);
  case MathOp::IsNan:
    return math::isnan(x) ? T{1} : T{0};
  case MathOp::IsInf:
    return math::isinf(x) ? T{1} : T{0};
  case MathOp::IsFinite:
    return math::isfinite(x) ? T{1} : T{0};
  }
  return T{0};
}

template <typename T>
__global__ void evaluate_kernel(const MathOp op, const T *x, const T *y, const T *z, T *out,
                                const size_t count) {
  const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = apply(op, x[i], y[i], z[i]);
  }
}

__global__ void combinatoric_kernel(const CombinatoricOp op, const int32_t *n, const int32_t *k,
                                    double *out, const size_t count) {
  const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  switch (op) {
  case CombinatoricOp::DoubleFactorialOdd:
    out[i] = math::double_factorial_odd(n[i]);
    break;
  case CombinatoricOp::Factorial:
    out[i] = math::factorial(n[i]);
    break;
  case CombinatoricOp::Binomial:
    out[i] = math::binomial(n[i], k[i]);
    break;
  }
}

// A device allocation that frees itself, so a launcher's early return on a
// failed call does not leak the arrays already allocated.
template <typename T>
class DeviceArray {
public:
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

  [[nodiscard]] bool ok() const { return ok_; }
  [[nodiscard]] T *data() const { return data_; }

  [[nodiscard]] bool upload(const T *host) const {
    return wwr::wwrMemcpy(data_, host, bytes_, wwr::wwrMemcpyHostToDevice) == wwr::wwrSuccess;
  }
  [[nodiscard]] bool download(T *host) const {
    return wwr::wwrMemcpy(host, data_, bytes_, wwr::wwrMemcpyDeviceToHost) == wwr::wwrSuccess;
  }

private:
  size_t bytes_;
  T *data_ = nullptr;
  bool ok_ = false;
};

uint32_t blocks_for(const size_t count) {
  return static_cast<uint32_t>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

// The launch itself reports nothing; a bad configuration surfaces from
// wwrGetLastError, and a fault inside the kernel from the synchronize.
bool launched_cleanly() {
  return wwr::wwrGetLastError() == wwr::wwrSuccess &&
         wwr::wwrDeviceSynchronize() == wwr::wwrSuccess;
}

template <typename T>
bool evaluate(const MathOp op, const T *x, const T *y, const T *z, T *out, const size_t count) {
  if (count == 0) {
    return true;
  }
  const DeviceArray<T> dx(count);
  const DeviceArray<T> dy(count);
  const DeviceArray<T> dz(count);
  const DeviceArray<T> dout(count);
  if (!dx.ok() || !dy.ok() || !dz.ok() || !dout.ok() || !dx.upload(x) || !dy.upload(y) ||
      !dz.upload(z)) {
    return false;
  }
  evaluate_kernel<T><<<blocks_for(count), kThreadsPerBlock>>>(op, dx.data(), dy.data(), dz.data(),
                                                              dout.data(), count);
  return launched_cleanly() && dout.download(out);
}

} // namespace

bool evaluate_on_device(const MathOp op, const float *x, const float *y, const float *z, float *out,
                        const size_t count) {
  return evaluate(op, x, y, z, out, count);
}

bool evaluate_on_device(const MathOp op, const double *x, const double *y, const double *z,
                        double *out, const size_t count) {
  return evaluate(op, x, y, z, out, count);
}

bool combinatoric_on_device(const CombinatoricOp op, const int32_t *n, const int32_t *k,
                            double *out, const size_t count) {
  if (count == 0) {
    return true;
  }
  const DeviceArray<int32_t> dn(count);
  const DeviceArray<int32_t> dk(count);
  const DeviceArray<double> dout(count);
  if (!dn.ok() || !dk.ok() || !dout.ok() || !dn.upload(n) || !dk.upload(k)) {
    return false;
  }
  combinatoric_kernel<<<blocks_for(count), kThreadsPerBlock>>>(op, dn.data(), dk.data(),
                                                               dout.data(), count);
  return launched_cleanly() && dout.download(out);
}

} // namespace free_breakfast::test_kernels
