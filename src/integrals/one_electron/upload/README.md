# Upload

What every one-electron build shares beyond `mmd`'s primitive-pair records and
the molecule's nuclei: the per-class launch over the records' streams, and the
checks every build runs before it launches anything.

## Module

`free_breakfast.integrals.one_electron.upload`, also re-exported by
`free_breakfast.integrals.one_electron` and `free_breakfast.integrals`.

## Design

The primitive-pair records are `mmd`'s: `primitive_pair_classes(res, basis)`
puts them on the device once per basis, each class on its own `DeviceStream`.
The nuclei are `molecule`'s: a `Molecule` copies its Z and positions to the
device once, at construction, on a `DeviceStream` of its own, and hands the
launchers a `NucleiView`.

**One stream per class.** `launch_each(classes, parent, launch)` is how a build
runs the classes: it forks every class's stream from `parent`, calls
`launch(stream, cls)` with the class's stream, and joins every stream back into
`parent`. After the first failed fork or launch it launches nothing more, but
still joins every stream it forked, so `parent` never runs ahead of a kernel in
flight. The nuclei are not forked: every class reads them, and none is
ordered after their stream, so the `Molecule` constructor synchronizes it once,
after the copies, before any build can launch.

**One record set serves every family.** S reads `zeta`, `pa_{x,y,z}`,
`pb_{x,y,z}`, `s` and the offsets; T reads those and `eb`, the ket exponent; V
reads those and `p_{x,y,z}`, the product centre. `mmd` copies them all,
whichever matrix is wanted.

**The shared checks.** `check_one_electron_build` is what every build
(`overlap`, `kinetic`, `potential_en`, `core_hamiltonian`) runs first: the
matrix and every class on the call's device, the matrix's order within the
kernels' 32-bit `m * n_ao + n` index (`n_ao² ≤ INT32_MAX`, about 46k Cartesian
AOs), and every class `la <= lb <= kMaxAngularMomentum`. That limit is
`basis`'s, one for the whole repo: every family is generated to it, and the
launchers' kernel tables are sized by it. `check_nuclei` is the extra one for
the builds that read the nuclei (V and H): the molecule's nuclei on the call's
device.

No device code: this module validates, and each family's module launches.

## Public API

| Symbol | Role |
|---|---|
| `launch_each(classes, parent, launch)` | Each class's launch on its own stream, between a fork from and a join into `parent`; the first error of a fork, launch or join |
| `check_one_electron_build(res, classes, m, what)` | `Status`: `InvalidConfig` naming the first refused input, `what` naming the build |
| `check_nuclei(res, molecule, what)` | `Status`: `InvalidConfig` if the molecule's nuclei are on another device |

## Files

| File | Role |
|------|------|
| `upload.cppm` | Module interface: `launch_each` and the checks |
| `upload.cpp` | Implementation |
| `CMakeLists.txt` | The module |
