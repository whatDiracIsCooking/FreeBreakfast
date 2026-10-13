/// @file
/// @brief The wwr RAII wrappers, bound once to this build's GPU error policies

export module free_breakfast.core.gpu;

export import free_breakfast.core.error_handling;

// Re-exported so `import free_breakfast.core.gpu;` reaches the wrappers' own
// members and free functions (copy, memset, views, ScopedRange, ...), not just
// the aliases below.
export import wwr.runtime_api;
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
