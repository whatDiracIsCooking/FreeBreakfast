# Upload

What every one-electron build shares beyond `mmd`'s primitive-pair records: the
nuclei, the per-class launch over the records' streams, and the checks every
build runs before it launches anything.

## Module

`free_breakfast.integrals.one_electron.upload`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

The primitive-pair records are `mmd`'s: `primitive_pair_classes(res, basis)`
puts them on the device once per basis, each class on its own `DeviceStream`.
This module adds `upload_nuclei`, a molecule's nuclei (Z and position) on the
device once per geometry. It copies from pageable memory and synchronizes once.

**One stream per class.** `launch_each(classes, parent, launch)` is how a build
runs the classes: it forks every class's stream from `parent`, calls
`launch(stream, cls)` with the class's stream, and joins every stream back into
`parent`. After the first failed fork or launch it launches nothing more, but
still joins every stream it forked, so `parent` never runs ahead of a kernel in
flight. The nuclei stay on `res.device()`'s own stream: every class reads them,
and `upload_nuclei` has synchronized before any build can.

**One record set serves every family.** S reads `zeta`, `pa_{x,y,z}`,
`pb_{x,y,z}`, `s` and the offsets; T reads those and `eb`, the ket exponent; V
reads those and `p_{x,y,z}`, the product centre. `mmd` copies them all,
whichever matrix is wanted.

**Owners in modules, views in a header.** `DeviceNuclei` (here) and
`PrimitivePairClass` (`mmd`) own wwr buffers, which only a module can name.
Each hands out a non-owning `view()`: `NucleiView` and `PrimitivePairClassView`,
plain pointer aggregates. `device_views.h` defines `NucleiView` and includes
`mmd`'s `primitive_pair_class_view.h`, so it is the launchers' one include for
both. They `#include` it and take a view instead of a dozen loose pointers;
this module includes it in its global module fragment and re-exports the names,
so the two sides share one entity (the `atom` pattern). A view is valid while
its owner is alive.

**The shared checks.** `check_one_electron_build` is what every build
(`overlap`, `kinetic`, `potential_en`, `core_hamiltonian`) runs first: the
matrix and every class on the call's device, the matrix's order within the
kernels' 32-bit `m * n_ao + n` index (`n_ao² ≤ INT32_MAX`, about 46k Cartesian
AOs), and every class `la <= lb <= kMaxAngularMomentum`. That limit is
`basis`'s, one for the whole repo: every family is generated to it, and the
launchers' kernel tables are sized by it. `check_nuclei` is the extra one for
the builds that read the nuclei (V and H): the nuclei on the call's device.

No device code: this module copies and validates, and each family's module
launches.

## Public API

| Symbol | Role |
|---|---|
| `launch_each(classes, parent, launch)` | Each class's launch on its own stream, between a fork from and a join into `parent`; the first error of a fork, launch or join |
| `DeviceNuclei` | A molecule's nuclei on the device: `z`, `c` (bohr), `n_nuc`, and the `dev_idx` they are on; `view()` gives a `NucleiView` |
| `PrimitivePairClassView`, `NucleiView` | Non-owning pointer views, through `device_views.h`, for the launchers |
| `upload_nuclei(res, molecule)` | `Result<DeviceNuclei>`, in atom order. `InvalidConfig` for a molecule with no atoms. Synchronizes once |
| `check_one_electron_build(res, classes, m, what)` | `Status`: `InvalidConfig` naming the first refused input, `what` naming the build |
| `check_nuclei(res, nuclei, what)` | `Status`: `InvalidConfig` if the nuclei are on another device |

## Files

| File | Role |
|------|------|
| `device_views.h` | `NucleiView`'s single definition, and the launchers' one include for both views |
| `upload.cppm` | Module interface: `launch_each`, the nuclei, the checks, and the views re-exported |
| `upload.cpp` | Implementation |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.one_electron.upload.headers` and the module |
