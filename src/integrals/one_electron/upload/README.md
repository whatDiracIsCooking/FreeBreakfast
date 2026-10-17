# Upload

The device inputs every one-electron build reads, and the check every build runs
before it launches anything.

## Module

`free_breakfast.integrals.one_electron.upload`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

`upload_primitive_pairs` copies `mmd`'s records to the device once per basis,
and `upload_nuclei` a molecule's nuclei (Z and position) once per geometry. Both
synchronize the stream once, after the copies, so the host data may be released
on return.

**One record set serves every family.** S reads `zeta`, `xpa`, `xpb`, `s` and
the offsets; T reads those and `eb`, the ket exponent; V reads those and `p`,
the product centre. The upload copies them all, whichever matrix is wanted.

**Owners in the module, views in a header.** `DevicePrimitivePairClass` and
`DeviceNuclei` own wwr device buffers, which only a module can name, so they
stay here. Each hands out a non-owning `view()`: `PrimitivePairClassView` and
`NucleiView`, plain pointer aggregates defined once in `device_views.h`. The
launchers `#include` that header and take a view instead of a dozen loose
pointers; this module includes it in its global module fragment and re-exports
the names, so the two sides share one entity (the `atom` pattern). A view is
valid while its owner is alive and unmoved.

**The shared check.** `check_one_electron_build` is what each family's build
(`overlap`, and T, V and H in `one_electron`) runs first: the matrix on the
call's device, its order within the kernels' 32-bit `m * n_ao + n` index
(`n_ao² ≤ INT32_MAX`, about 46k Cartesian AOs), and every class `la <= lb <=
kMaxAngularMomentum`. That limit is `basis`'s, one for the whole repo: every
family is generated to it, and the launchers' kernel tables are sized by it.

No device code: this module copies and validates, and each family's module
launches.

## Public API

| Symbol | Role |
|---|---|
| `DevicePrimitivePairClass` | One class's records on the device, the same fields as `PrimitivePairClass`; `view()` gives a `PrimitivePairClassView` |
| `upload_primitive_pairs(res, classes)` | `Result<std::vector<DevicePrimitivePairClass>>`. `InvalidConfig` for an empty class. Synchronizes once |
| `DeviceNuclei` | A molecule's nuclei on the device: `z`, `c` (bohr), `n_nuc`, and the `dev_idx` they are on; `view()` gives a `NucleiView` |
| `PrimitivePairClassView`, `NucleiView` | Non-owning pointer views, from `device_views.h`, for the launchers |
| `upload_nuclei(res, molecule)` | `Result<DeviceNuclei>`, in atom order. `InvalidConfig` for a molecule with no atoms. Synchronizes once |
| `check_one_electron_build(res, classes, m, what)` | `Status`: `InvalidConfig` naming the first refused input, `what` naming the build |

## Files

| File | Role |
|------|------|
| `device_views.h` | The single definition of the two views, for `#include` and `import` alike |
| `upload.cppm` | Module interface: the owners, the uploads, the check, and the views re-exported |
| `upload.cpp` | Implementation |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.one_electron.upload.headers` and the module |
