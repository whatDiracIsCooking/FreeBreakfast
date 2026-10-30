# `src/`

Project source. Every target in here is declared with one of the three macros
from [`cmake/`](../cmake/README.md) and named with dots, aliased to `::` —
`free_breakfast.core` is consumable as `free_breakfast::core`.

```
core/      The starting set: one module library over WarpWraps and `import std`
atom/      The Element enum and Atom, shared by #include and import
molecule/  Molecule, nuclear repulsion energy and gradient
basis/     Gaussian basis description: shells, AO layouts, cart->sph, BasisStamp/place
io/        File-level I/O: whole-file primitives, JSON / TOML / .npy, config, BSE basis and XYZ geometry readers, the Molden orbital writer, the compiled-in SAD stamps
sad/       The SAD initial-guess density from spherically averaged atomic SCFs over a placed basis, in the spherical layout
recipes/   Dense GPU linear algebra: packed gemm and the sandwich products, the symmetric orthogonalizer S^{-1/2}, DIIS extrapolation, and the stable GPU merge sort
integrals/ Gaussian integrals on the GPU over FelizJueves-generated kernels: MMD primitive-pair records, S, T, V, the core Hamiltonian H = T + V and the Coulomb matrix J
localization/  Orbital localization of the occupied space on the GPU: Pipek-Mezey (p = 2 or Knizia's p = 4), and intrinsic atomic and bond orbitals over MINAO, and the valence virtual orbitals over the IAOs
analysis/  Wavefunction analysis: atomic populations and partial charges over the intrinsic atomic orbitals, and the bond-orbital reading of the IBOs
ci/        Determinant CI over plain arrays: alpha/beta occupation strings, their lexical addresses and single-replacement lists
response/  The energy's orbital-rotation derivatives at fixed orbitals and RDMs on the GPU: the energy, the inactive, active and generalized Fock matrices, the orbital gradient and its packed pair order, and the orbital Hessian
mcscf/     CASSCF on the GPU: the two-step state-averaged loop over response's gradient and Hessian
nevpt2/    SC- and PC-NEVPT2 on the GPU: the density-fitted MO blocks and the external-integral slabs, and dm3 and the integral-contracted 4-RDMs f3ca and f3ac of the reference state
cis/       CIS (TDA) excitations on the GPU: the sigma, integral-direct or density-fitted over B's MO blocks, the orbital-energy diagonal, the unit-vector Davidson guess on its lowest entries, and the r / (θ − Δε) preconditioner
geometry/  A Cartesian quasi-Newton minimizer on the host: BFGS under a trust radius over a caller's energy and gradient, with the rigid-body motions projected out
pipeline/  The calculation pipeline: the step contract, the setup steps (molecule through H, and the SAD guess), the ERIs (integral-direct or density-fitted, once per pipeline), the RHF SCF, Pipek-Mezey and IBO localization, CASCI, FCI, state-averaged CASSCF and SC- and PC-NEVPT2 over it, CIS excitations, and Pipeline<Steps...>, which plans and runs them
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
