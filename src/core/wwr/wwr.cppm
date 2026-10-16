/// @file
/// @brief The interface between free_breakfast and WarpWraps: the wwr modules
/// this tree uses, the GPU error bridge and policies, and the RAII wrappers
/// bound to them

export module free_breakfast.core.wwr;

export import free_breakfast.core.error_handling;

// gpu_status (:status) and the wwr error policies over it (:policy). Here, not
// in error_handling, so that module never names wwr and the whole wwr binding
// sits in one place.
export import :status;
export import :policy;

// Stream, Event and ForkStream (:fork_stream). The stream-level wrappers live in
// the partition because ForkStream is built from them, and a partition cannot
// import the primary interface to reach an alias defined here.
export import :fork_stream;

// Device (:device): this tree's own pool-tier device handle, with a ForkStream
// as its work stream. A partition rather than a class here for the same reason:
// it is built from ForkStream.
export import :device;

// gemm (:gemm): wwr::gemm for n x n operands with every leading dimension n,
// transposed or not. A convenience over wwr's call, not a replacement: any
// other shape calls wwr::gemm directly.
export import :gemm;

// gemmwich (:gemmwich): the sandwich product op(X) F op(X)^T -- X^T F X by
// default -- as two square gemms through a caller-supplied work matrix.
export import :gemmwich;

// gmmwich (:gmmwich): the same sandwich with a diagonal middle, op(X)^T diag(d)
// op(X), as a dgmm and a gemm, so the diagonal is never formed.
export import :gmmwich;

// Re-exported so `import free_breakfast.core.wwr;` is the one import a module
// above core/types needs for anything WarpWraps provides: the runtime API, the
// vector types and scalar traits, and the wrappers' own members and free
// functions (copy, memset, views, ScopedRange, ...), not just the aliases below.
// wwr.wrappers.{blas,solver} are the typed dispatch over them (gemm<T>,
// syevd<T>, ...), which the handles above only create and bind.
export import wwr.runtime_api;
export import wwr.vector_types;
export import wwr.wrappers.common;
export import wwr.wrappers.blas;
export import wwr.wrappers.solver;
export import wwr.extension.common;
export import wwr.extension.handle;
export import wwr.extension.runtime;
export import wwr.extension.memory_buffer;
export import wwr.extension.blas;
export import wwr.extension.solver;
export import wwr.extension.tx;

import std;

// Not exported: inside the export block below it would leak into every
// importer as free_breakfast::ext.
namespace free_breakfast {
namespace ext = wwr::extension;
}

export namespace free_breakfast {

/// @brief Every buffer kind and its view, allocated from a Device's pool:
/// `Buffers::device<T>`, `::host<T>`, `::pinned<T>`, `::unified<T>`, `::device_view<T>`, ...
using Buffers = ext::kit::device_buffer_suite<PolicyMap, Device>;

/// @brief A BLAS handle bound to a Device's stream.
using BlasHandle =
    ext::BlasHandleWrapper<CreatePolicy<wwr::wwrblasStatus_t>, DestroyPolicy<wwr::wwrblasStatus_t>,
                           Device, CreatePolicy<wwr::wwrError_t>>;

/// @brief A dense-solver handle bound to a Device's stream.
using SolverHandle = ext::SolverDnHandleWrapper<CreatePolicy<wwr::wwrsolverStatus_t>,
                                                DestroyPolicy<wwr::wwrsolverStatus_t>, Device,
                                                CreatePolicy<wwr::wwrError_t>>;

/// @brief Scoped NVTX / rocTX range (wwr's, re-exported under free_breakfast).
using ScopedRange = ext::ScopedRange;

/// @brief Make a device current on this thread for the guard's lifetime, then
/// restore the device that was current before.
///
/// Library calls and kernel launches run on whatever device the calling thread
/// has current; only wwr's buffers select theirs. Both the switch and the
/// restore abort on failure: a restore that only logged would leave later work
/// silently running on the wrong device.
using ScopedDeviceIndex = ext::ScopedDeviceIndex<CreatePolicy<wwr::wwrError_t>>;

} // namespace free_breakfast
