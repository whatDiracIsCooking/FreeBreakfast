# Upload

The device inputs every one-electron build reads, and the checks every build runs
before it launches anything.

## Module

`free_breakfast.integrals.one_electron.upload`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

`upload_primitive_pairs` puts a basis's primitive-pair records on the device
once per basis, and `upload_nuclei` a molecule's nuclei (Z and position) once
per geometry.

**Records written straight into pinned memory.** Each class's records are one
`n_rec`-long `MirroredBuffer` per field, on the class's stream:
`PairFieldBuffers`, mmd's `PairFields` over `MirroredBuffer`s. The upload sizes
them from `primitive_pair_classes`, has `write_primitive_pairs` fill their
pinned sides in place (through `map_pair_fields` to spans), and queues every
field's `to_device()` on the class's stream, ahead of anything a build launches
there. It never synchronizes: the pinned side is kept for the class's lifetime
(a `MirroredBuffer`'s destructor waits out its copies), at the cost of a host
mirror of the records, which are small next to any matrix they build. One
buffer per field is fourteen small pinned allocations per class rather than
two large ones; that is a once-per-basis cost, chosen for the plainer code.
`upload_nuclei` still copies from pageable memory and synchronizes once.

**One stream per class.** `upload_primitive_pairs` returns a
`DevicePrimitivePairs`: every class with its own `DeviceStream` on
`res.device()`, its records allocated and copied on that stream (from the
Device's one pool). `launch_each(parent, launch)` is how a build uses them: it
forks every class's stream from `parent`, calls `launch(stream, cls)` with the
class's stream, and joins every stream back into `parent`. After the first
failed fork or launch it launches nothing more, but still joins every stream it
forked, so `parent` never runs ahead of a kernel in flight. The nuclei stay on
`res.device()`'s own stream: every class reads them, and `upload_nuclei` has
synchronized before any build can.

**One record set serves every family.** S reads `zeta`, `pa_{x,y,z}`,
`pb_{x,y,z}`, `s` and the offsets; T reads those and `eb`, the ket exponent; V
reads those and `p_{x,y,z}`, the product centre. The upload copies them all,
whichever matrix is wanted.

**Owners in the module, views in a header.** `DevicePrimitivePairClass` and
`DeviceNuclei` own wwr buffers, which only a module can name, so they
stay here. Each hands out a non-owning `view()`: `PrimitivePairClassView` and
`NucleiView`, plain pointer aggregates defined once in `device_views.h`
(`PrimitivePairClassView` is `PairFields` of `const` pointers plus the class's
`la`, `lb` and `n_rec`, so the launchers still read `cls.zeta`). The
launchers `#include` that header and take a view instead of a dozen loose
pointers; this module includes it in its global module fragment and re-exports
the names, so the two sides share one entity (the `atom` pattern). A view is
valid while its owner is alive and unmoved.

**The shared checks.** `check_one_electron_build` is what every build
(`overlap`, `kinetic`, `potential_en`, `core_hamiltonian`) runs first: the
matrix and the primitive pairs on the call's device, the matrix's order within
the kernels' 32-bit `m * n_ao + n` index (`n_ao² ≤ INT32_MAX`, about 46k Cartesian AOs), and every
class `la <= lb <= kMaxAngularMomentum`. That limit is `basis`'s, one for the
whole repo: every family is generated to it, and the launchers' kernel tables
are sized by it. `check_nuclei` is the extra one for the builds that read the
nuclei (V and H): the nuclei on the call's device.

No device code: this module copies and validates, and each family's module
launches.

## Public API

| Symbol | Role |
|---|---|
| `PairFieldBuffers` | `PairFields<MirroredBuffer<double>, MirroredBuffer<int32_t>>`: one buffer per field |
| `DevicePrimitivePairClass` | One class's records: `la`, `lb`, `n_rec`, `fields` (a `PairFieldBuffers`), `device_stream()`; `view()` gives a `PrimitivePairClassView` |
| `DevicePrimitivePairs` | Every class of a basis on one device: `dev_idx()`, `classes()`, and `launch_each(parent, launch)` |
| `upload_primitive_pairs(res, basis)` | `Result<DevicePrimitivePairs>`, one new `DeviceStream` per class (none for an empty basis). Queues each class's copy on its stream; no host synchronization |
| `DeviceNuclei` | A molecule's nuclei on the device: `z`, `c` (bohr), `n_nuc`, and the `dev_idx` they are on; `view()` gives a `NucleiView` |
| `PrimitivePairClassView`, `NucleiView` | Non-owning pointer views, from `device_views.h`, for the launchers |
| `upload_nuclei(res, molecule)` | `Result<DeviceNuclei>`, in atom order. `InvalidConfig` for a molecule with no atoms. Synchronizes once |
| `check_one_electron_build(res, pairs, m, what)` | `Status`: `InvalidConfig` naming the first refused input, `what` naming the build |
| `check_nuclei(res, nuclei, what)` | `Status`: `InvalidConfig` if the nuclei are on another device |

## Files

| File | Role |
|------|------|
| `device_views.h` | The single definition of the two views, for `#include` and `import` alike |
| `upload.cppm` | Module interface: the owners, the uploads, the checks, and the views re-exported |
| `upload.cpp` | Implementation |
| `CMakeLists.txt` | The interface library `free_breakfast.integrals.one_electron.upload.headers` and the module |
