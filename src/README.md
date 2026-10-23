# `src/`

Project source. Every target in here is declared with one of the three macros
from [`cmake/`](../cmake/README.md) and named with dots, aliased to `::` —
`free_breakfast.core` is consumable as `free_breakfast::core`.

```
core/      The starting set: one module library over WarpWraps and `import std`
atom/      The Element enum and Atom, shared by #include and import
molecule/  Molecule, nuclear repulsion energy and gradient
basis/     Gaussian basis description: shells, AO layouts, cart->sph, BasisStamp/place
io/        File-level I/O: whole-file primitives, JSON / TOML / .npy, config, BSE basis and XYZ geometry readers
sad/       The SAD initial-guess density from spherically averaged atomic SCFs over a placed basis, in the spherical layout
linalg/    Dense GPU linear algebra: square gemm and the sandwich products, the symmetric orthogonalizer S^{-1/2}, and DIIS extrapolation
integrals/ Gaussian integrals on the GPU over FelizJueves-generated kernels: MMD primitive-pair records, S, T, V, the core Hamiltonian H = T + V and the Coulomb matrix J
pipeline/  The calculation pipeline: the step contract, the setup steps (molecule through H), and Pipeline<Steps...>, which plans and runs them
main.cpp   The free-breakfast executable: a TOML config planned and run as a pipeline, and the one Error -> exit-code boundary
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
does, and it is a submodule at `deps/WarpWraps`. Code here reaches it through
**one module**, [`free_breakfast.core.wwr`](core/wwr/README.md), which
re-exports the backend-neutral wwr modules the tree uses and binds the RAII
wrappers to this tree's error policies:

```cpp
import free_breakfast.core.wwr;   // Device, Buffers, BlasHandle, gpu_status,
                                  // and wwr::wwrGetDeviceCount, ...
```

and link `free_breakfast::core::wwr`. Don't import `wwr.*` or link `wwr::*`
from a new module; to use a wwr module the tree does not use yet, add it to
`core/wwr`. The exceptions (`core/types`, header and `.cu` device code, a
deliberately CUDA-only `import wwr.cuda.*`) and the reasons for them are in
that README.

A build targets exactly one backend, chosen by `FREE_BREAKFAST_GPU_BACKEND`; the
`wwr*` names resolve to that backend's own, so code written against them builds
for both.
