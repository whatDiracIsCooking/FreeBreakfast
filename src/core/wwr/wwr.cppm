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

// Re-exported so `import free_breakfast.core.wwr;` is the one import a module
// above core/types needs for anything WarpWraps provides: the runtime API, the
// vector types and scalar traits, and the wrappers' own members and free
// functions (copy, memset, views, ScopedRange, ...), not just the aliases below.
export import wwr.runtime_api;
export import wwr.vector_types;
export import wwr.wrappers.common;
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

/// @brief A non-blocking stream on one device.
using Stream = ext::StreamWrapper<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                  CreatePolicy<wwr::wwrError_t>>;

/// @brief An event on one device; default-constructed it is sync-only (timing disabled).
using Event = ext::EventWrapper<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                CreatePolicy<wwr::wwrError_t>>;

/// @brief One device's context: its properties, an owned stream and memory pool.
///
/// Hold it as a std::shared_ptr -- buffers and library handles retain their
/// device so its stream and pool outlive them.
using Device = ext::kit::DeviceHandle<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                      CreatePolicy<wwr::wwrError_t>>;

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

} // namespace free_breakfast
