# `src/`

Project source. Every target in here is declared with one of the three macros
from [`cmake/`](../cmake/README.md) and named with dots, aliased to `::` —
`free_breakfast.core` is consumable as `free_breakfast::core`.

```
core/      The starting set: one module library over WarpWraps and `import std`
atom/      The Element enum and Atom<T>, shared by #include and import
molecule/  Molecule, nuclear repulsion energy and gradient
basis/     Gaussian basis description: shells, AO layouts, cart->sph, BasisStamp/place
io/        File-level I/O: whole-file primitives, JSON / TOML / .npy, config, BSE basis and XYZ geometry readers
```

## Adding a module

1. Make the directory and write the `.cppm`.
2. Declare it with `free_breakfast_add_cxx_module_library` (or
   `free_breakfast_add_gpu_device_library` for `.cu` device code,
   `free_breakfast_add_interface_library` for header-only).
3. `add_subdirectory(<dir>)` in this file.
4. Add the target to the parent's `LINK_PUBLIC` list, and
   `export import free_breakfast.<name>;` to the aggregating module if it is
   part of the public surface.
5. Drop a `README.md` in the directory, matching the neighbours.

## What the GPU layer is

This project does not wrap the CUDA or HIP APIs itself — [WarpWraps](https://github.com/whatDiracIsCooking/WarpWraps)
does, and it is a submodule at `deps/WarpWraps`. Import its backend-neutral
modules rather than a vendor header:

```cpp
import wwr.extension.memory_buffer;   // DeviceBuffer, PinnedBuffer, ...
import wwr.extension.runtime;         // RAII stream, event, graph
import wwr.wrappers.blas;             // gemm<float>(...), either backend
```

and link the matching target (`wwr::extension::memory_buffer`, …). A build
targets exactly one backend, chosen by `FREE_BREAKFAST_GPU_BACKEND`; the `wwr*`
names resolve to that backend's own, so code written against them builds for
both. Reach for a vendor-only entry point through the low-level module
(`import wwr.cuda.cublas_v2;`) and accept that the file is then CUDA-only.
