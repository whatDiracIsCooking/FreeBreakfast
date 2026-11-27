# Rust rebuild: the GPU layer

The kernels stay in C++ and CUDA, hand-written and FelizJueves-generated, and
Rust owns everything above a launch. The device code a crate needs is forked
into `rust/`, beside the crate that calls it, and built by `rust/`'s own
CMake project: `rust/` shares no source file and no build with the C++ tree
("What the port changes"). This document is the contract between
the two: what a launch wrapper may do, what crosses the C ABI and in which
shape, where the launch shape and the angular-momentum dispatch live, how
scratch is sized, what a status means, and how the libraries are built and
bound. It leans on the programming model's terms, *step*, *`Output`* and
*scratch* ([`programming_model.md`](programming_model.md)), and on the test
tiers ([`testing.md`](testing.md), "Tiers: where a check lives").

## The layer already exists

The C++ tree has had this layer since its first kernel, under the name
*bridge*. A module's host code never includes a `.cu`: it includes a bridge
header, `<name>_bridge.h`, in its global module fragment, and the matching
`<name>_launch.cu` (or `<name>.cu`) defines what the header declares. The
header is the single definition of the launcher's signature and of the
plain-struct views that cross it; `primitive_pair_class_view.h` says so in as
many words, and the module re-exports the very same struct. Counted on
2026-11-12:

| | Count |
|---|---|
| Bridge headers (`*_bridge.h`) | 42 |
| Launcher translation units (a `.cu` outside a `kernels/` directory) | 48 |
| Launch functions the bridges declare | 114 |
| Launchers that allocate, synchronize, record or wait on an event, or set the device | 0 |
| Bridges that pass a struct by `const &` | 18 |
| Bridges that are templates over the element type | 7 (all in `recipes`) |
| Generated kernel units (`kernels/*.cu` under `integrals`) | 1205 |

The zero is the finding that matters. "C++ wraps kernels and nothing else" is
not a rule to impose on the port; it is what the launchers already do. Each
takes a stream first, then integers, device pointers and parameter structs,
launches, and returns the launch status, documented everywhere as "launch
errors only: the kernel's own faults surface at the next synchronization".
The port changes the ABI the bridge is spoken in, not what it says.

The C++ tree is where the layer's shape comes from, not what `rust/` builds
on. A port copies the bridges and kernels it needs into `rust/`, and the
copy is what the Rust tree compiles; the counts above are the C++ tree's,
and they say how much a full port forks. Forked so far: the inverse
square root, the symmetrization, the Cholesky diagonal tile, the DIIS solve
and the triangular solve's diagonal block (`rust/recipes/kernels/`), the
launchers and emitted units of the
primitive-pair records, the overlap, the kinetic energy and the nuclear
attraction (`rust/integrals/kernels/`, the last with the emitted Boys
routine, `eval_boys.h` and its `eval_boys_<m>.h`, in `shared/` there), and
the Pipek-Mezey localizer's kernels (`rust/localization/kernels/`), over a
trimmed fork of the device headers they include (`rust/device_headers/`,
the block and warp reductions among them).

## What the port changes

**The bridge header is the port line.** Everything above a module's bridge
is Rust's to port; everything below it is forked. In the C++ tree `mmd.cpp` builds
`PrimitivePairSource` on the host, packs the shells' exponents and centres
into device arrays, computes `rec_begin`, and then launches. Under this
design all of that is Rust: the wrapper receives device pointers only, and
the host preparation, the copies that fill them and the budget they come
from belong to the step that calls it. "Porting a module" therefore has a
definition in files, all of them under `rust/`:

- its bridge header is copied to `rust/<module>/kernels/` and rewritten as a
  C header ("From bridge to C ABI");
- its launcher `.cu` files are copied beside it, the kernels unchanged and
  the launcher rewritten as an `extern "C"` wrapper;
- its FelizJueves units are emitted again into `rust/` ("Build");
- the device headers they include are forked, trimmed to what is used, into
  `rust/device_headers/`;
- its `.cppm` and `.cpp` are rewritten as the Rust crate of the same name.

Nothing in the C++ tree changes. The C++ module is not deleted then; the
paragraph on the two trees, below, says when it goes.

**The wrapper lives in the `.cu`.** The layering diagram below shows the
launch wrapper as its own layer, and that is its role, not its file. Every
launcher is a `.cu` (`mmd_launch.cu`, `trsm.cu`), because the launch
syntax is compiled only by nvcc or `clang -x hip`, and the `extern "C"`
wrapper goes into that same unit. There is no `.cpp` that launches, and no
attempt to build one.

**The port is incremental, and neither tree can break the other.** A Rust
PR touches only `rust/`, and the C++ tree's sources are nothing `rust/`
compiles, so each tree is built and tested by its own gate half and is
green on its own. Only what the two share reaches both: the submodules, and
the emitters, whose output each tree commits. The order of the port is therefore free: one module at a time, its
device code forked with it. Nothing is cut over all at once.

**The fork has a cost, and it is paid by hand.** The hand-written device
code a port forks, the launchers and the shared device headers, exists twice, once in each tree, and nothing keeps the two copies
in step but review: a fix to one is a fix owed to the other, and the PR that
makes it says so. The emitted units carry no such cost. `tools/emit`'s
`--out-root` writes the same tree under `rust/` that its default writes for
the C++ tree, byte for byte, so after a `deps/FelizJueves` bump both are
emitted again from the one generator. The forked headers are trimmed to
what their users include, and a fork's launcher may differ from the C++ one
on purpose (one workspace for two scratch arrays, refusals of its own); the
kernels it launches do not. The cost ends with the C++ tree's copy.

**The two trees coexist until the pipeline is Rust.** The C++ `pipeline`
step imports every C++ module, and the C++ pipeline is the tree's one
working calculation until the Rust one runs end to end, so a ported module's
`.cppm` and `.cpp` stay. Each tree calls its own copy of the kernels, built
by its own CMake project, and each is tested by its own suite; the two are
compared at the regression bars ([`testing.md`](testing.md), "Regression"),
not by sharing a binary. Nothing crosses from Rust back into C++: the Rust
tree stands alone, as its README says, and a C ABI over the Rust modules for
the C++ pipeline to call would be a second boundary to keep, for a pipeline
the port replaces. The C++ modules go together, at the cut-over, when the
Rust pipeline runs every config under `examples/` that the C++ one does and
matches it at those bars, and the second copy of the device code goes with
them. Until then "done" for a module PR means its Rust crate passes its
tiers.

**Each shared object says what it is, and the crate checks at load.**
Committed bindings and a separately built `.so` can drift at run time, not
only in review. Each library exports two entries,
`fb_<module>_abi_version()` and `fb_<module>_backend()`, defined in its
`kernels/abi.cu`: the first returns the `FB_<MODULE>_ABI_VERSION` its C
header defines, bumped by hand with any change to a declaration the header
reaches (`FB_RECIPES_ABI_VERSION` is 8, bumped as the inverse square root,
the symmetrization, the Cholesky diagonal tile and the DIIS solve joined the
merge sort, then as the inverse square root took its floor's factor from the
device, then as the triangular solve's diagonal block joined, then as the
merge sort and that block's `float` entry left, which nothing called;
`FB_INTEGRALS_ABI_VERSION` is 13, bumped as the kinetic and
nuclear-attraction wrappers with `fb_nuclei_view`, then density fitting's
metric and three-center launchers, joined the overlap, then as the records'
order left (`fb_mmd_order` and its query removed, `fb_mmd_records` taking
no permutation), then as `fb_three_center_class` took a class view for its
bra records instead of a packed buffer, then as the conventional ERI's
launchers joined, then as the dense conventional ERI's, then as density
fitting's pack and gather over the packed shell-pair map, then as the
packed three-center launcher, then as the square three-center launchers,
their cell field count and the symmetrization left, then as the packed
three-center launcher's aux index became spherical, its declaration the
same but its `n_aux` and `dst` another basis's counts, which an older
library would overrun, then as the fully spherical
`fb_three_center_spherical_class` joined, then as the conventional ERI's
launchers, dense and packed, and density fitting's pack left;
`FB_LOCALIZATION_ABI_VERSION` is 1, the Pipek-Mezey launchers; `fb-doc-check`'s
`doc_lists.rs` holds these numbers to the headers). The bump is held:
`devtools/rust-bindings.py` records in each module's bindings, beside the
version, a digest of the declarations the header reaches (its own, its
inlined includes' and `fb_launch.h`'s, comments and the version's own
`#define` left out), and refuses, checking or writing, a header whose
declarations have moved from it while the version has not gone up; so
`fb-device-check`'s `bindings.rs` fails, and nothing regenerates past it.
The second entry returns `FB_BACKEND_CUDA` or `FB_BACKEND_HIP`
(`fb_launch.h`), chosen by the compiler's own `__CUDACC__` or `__HIP__`
rather than a definition the build passes. The `-sys` crate's
`check_library()` checks the version against the constant in its committed
bindings and the backend against the CMake build whose C ABI library
`wwr-sys` links. `rust/kernel_lib.rs`, which every kernel `-sys` crate's
`build.rs` includes, reads that backend once from the build's
`CMakeCache.txt`, since wwr's C ABI has no query naming its backend, and
hands it to the crate as `FB_KERNELS_BACKEND`, which the crate reads as a
`fb_kernel_abi::Backend` at compile time. It refuses a mismatch with a
message naming both values. A library built for
the other backend otherwise fails at its first launch, or deep inside wwr,
with nothing naming the cause.

"At load" is the first call, not the dynamic loader: Rust has no load-time
hook short of a linker section an `rlib` may drop, so `check_library()` is a
plain function that a caller runs once before its first launch. The safe
layer runs it at setup: `fb-steps`' bus calls each library's
`check_library()` that the run's steps launch from before the plan is
made, so a mismatch is setup's `GpuRuntime` error and nothing launches. Each `-sys` crate's tier 4 also runs it against the
library built for its card. The comparison itself is
`fb_kernel_abi::check`, a function of values passed in, in `fb-kernel-abi`,
a crate that links nothing, so its tests of a mismatched version and
backend run in tier 3.

## Layering

```
 Rust      steps on rustflow: the DAG, fork, join, wait, sync; streams, events, memory
   │
 Rust      safe wrapper: owns the arguments' types and lifetimes, checks contracts
   │
 Rust      -sys: FFI bindings, generated from the C header
   │  C ABI
 C++ .cu   launch wrapper: extern "C", picks the launch shape, launches, returns a status
   │       (forked: rust/<module>/kernels/)
 C++ .cu   kernels: hand-written and FelizJueves-generated
```

**A launch wrapper only enqueues.** It takes a stream, device pointers, sizes
and parameters, picks the grid and block, launches, and returns a status. It
never allocates, synchronizes, waits on or records an event, sets the current
device, reads a device value back to the host, opens a profiler range, or
holds state between calls. Fork, join, wait and sync belong to the Rust
layer above, and every allocation to its `Budget`. A wrapper is therefore safe to
call from any thread, and its whole cost is on the stream it was given. The
rule is cheap to keep: a tier-3 test, `fb-device-check`'s
`launch_wrappers.rs`, greps every forked `.cu` and `.cuh` under `rust/` for
the runtime's allocation, synchronization, event, copy and device calls, the
grep that produced the zero above.

**And so it is capture-safe by construction.** A function that only
enqueues on the stream it was given can be captured into a graph and
replayed. The two items the "never" list gained are the two that would
break it: a read-back to the host is a synchronization, and a profiler
range belongs to the caller ([`diagnostics.md`](diagnostics.md)). A wrapper
that launches several kernels returns the first failure; what it enqueued
before stays enqueued, which is harmless, because a launch error is not
sticky and a sticky fault stops everything after it anyway.

**The current device is the stream's, and the caller made it so.** A launch
on a stream of a device that is not current fails at the runtime. The
wrapper does not check, because it cannot without a runtime call of its
own; `fb-core-device`'s `Stream`, `Event` and `GpuResources` make their
device current around every call ("The v0.0.4 runtime"), and a safe launch
takes that stream.

**Library calls go straight from Rust through `wwr-sys`.** BLAS, the
solvers, the collectives: WarpWraps' C ABI already covers them, and the
v0.0.4 layer's `fb_core_device::GpuResources` calls them that way.
Only this tree's own kernels need a C++ wrapper.

**The library's triangular solves are not called.** rocBLAS's `trsm`
takes workspace that grows with the right-hand sides, about a copy of B
when the order is not a multiple of 128, so no fixed handle workspace
covers it and a call past it allocates outside the budget; the
solvers' `potrs` and `getrs` are that trsm underneath. So
`GpuResources` binds no library `trsm`, and a triangular solve is
`fb_core_device::GpuResources::trsm`, the C++ `src/recipes/trsm`: blocked, each
diagonal block by a shared-memory kernel of `libfb_recipes_kernels.so`
and the update between blocks a gemm, with one right-hand side the
library's `trsv`, which takes no workspace on either backend. It is a
plain call on the handle's stream that enqueues, takes no workspace of
its own, and has the library call's signature. One code path on both
backends, so CUDA runs it too. `fb-doc-check` holds it
(`tests/source_rules.rs`): no source under `rust/` names a `wwr-sys`
`trsm`, `potrs` or `getrs`.

**A GPU's library handles are its `GpuResources`, made in setup.** One
per GPU, behind a `Mutex` whose guard a step's `run` is handed, never made
inside a `run`; its stream is set once, at creation, and its BLAS workspace,
which cuBLAS and hipBLAS would otherwise allocate for themselves, is a
`DeviceBuf` the final node allocates and binds once, kept until the handles
are destroyed, so a run allocates nothing through a library either
([`memory.md`](memory.md), "The budget"; "The v0.0.4 runtime" below). A
step that changes a handle's state (its pointer mode) sets it back before
it releases the guard.

**Rust never launches a kernel itself.** The C ABI exports a launch call
(`wwr_launch_kernel_ex_c`), so Rust could take a kernel's address and launch
it. It does not, because the block and the dispatch belong with the kernel
(below), and moving them to Rust would put the generated tables into the ABI.
clippy's `disallowed_methods` holds it: `wwr_launch_kernel_ex_c` is listed
in `clippy.toml`, and no site allows it.

## From bridge to C ABI

What changes when a C++ bridge header is forked into `rust/` as the C header
the bindings are generated from. The forked `mmd.h`, `overlap.h`,
`kinetic.h`, `potential_en.h` and `trsm.h` follow this table:

| The C++ tree's bridge | The forked C header |
|---|---|
| `namespace free_breakfast::device { wwrError_t mmd_bound(...) }` | `extern "C" fb_launch_status fb_mmd_bound(...)`, the module's name as the prefix |
| `wwr::wwrStream_t`, a vendor type behind a wwr alias | `wwr_stream`, the C ABI's `void *`, which Rust already holds for every stream |
| `wwr::wwrError_t`, the vendor's error enum | `fb_launch_status`, "Statuses" below |
| `const PrimitivePairClassView &` | `const fb_primitive_pair_class_view *` |
| `template <typename Z, typename R> phase_scale(...)` | One entry per instantiation, the element named by a suffix: `fb_phase_scale_c128_f64` |
| `[[nodiscard]]` | Nothing at the ABI; the safe wrapper returns `Result`, which is `#[must_use]` |
| `bool` | `bool`: C99's `_Bool` and Rust's `bool` share an ABI |
| `int32_t` counts and indices | `int32_t`, unchanged; `size_t` is the one other integer, for bytes |
| Scratch arrays whose lengths the doc comment states | One scratch pointer, sized by a workspace query, "Scratch" below |

**One C header per module.** Each module has one header,
`rust/<module>/kernels/fb_<module>.h` (`fb_recipes.h`, `fb_integrals.h`),
that includes its wrappers' C headers, and that is the file
`devtools/rust-bindings.py` reads, so there is one bindings file per shared
object and the two are versioned together. The plain-C vocabulary every
library shares, `fb_launch_status`, its values, `FB_DEVICE_ALIGNMENT` and
`fb_backend`, is `rust/device_headers/fb_launch.h`, and it has one Rust
home: the script binds it once, into `fb-kernel-abi`'s generated
`bindings.rs`, and does not inline it into a module's bindings, which open
with a `pub use fb_kernel_abi::{…}` of its names instead, so each `-sys`
crate still offers them and a new library adds no copy. There
`fb_launch_status` is `LaunchStatus`, a newtype ("Statuses" below), and
`fb_backend` stays the integer a library returns, read by value as
`fb_kernel_abi::Backend`. The two integer types are the bridges'
own: every count, index and momentum is an `int32_t`, and bytes, which only
the workspace queries and the budget speak, are `size_t`.

The seven template bridges, none forked yet, are `recipes`'
element-generic kernels (`abs_at`, `complex_cast`, `expm`, `einsum`, `cg_unitary`, `inverse_sqrt`,
`polynomial_roots`). A C ABI cannot export a template, so each explicit
instantiation becomes its own entry, the pattern WarpWraps' C ABI set for its
own device templates (`extension_random_normal`, `wrappers_*`: one
monomorphised `extern "C"` thunk per instantiation, the type in a suffix).
The instantiation list is the source: the entries are generated from it, so
none is invented and none dropped.

The views and parameter structs cross as they are, by pointer from host
memory. `PrimitivePairClassView`, `PrimitivePairSource`,
`PrimitivePairRecords` and `src/molecule`'s `NucleiView` are forked as
`fb_primitive_pair_class_view`, `fb_primitive_pair_source`,
`fb_primitive_pair_records` and `fb_nuclei_view` (`nuclei_view.h`), the same
fields in the same order. The CI `Replacement` and `EinsumPlan` are integers and
pointers already, and `EinsumPlan` already carries the `static_assert` that
it is trivially copyable and standard-layout "because it crosses the bridge
by value". "Shared types" below makes that the rule.

## Launch shape and dispatch

**The launch shape stays with the kernel.** The wrapper picks the grid and
block from the problem's dimensions, because the block belongs next to the
kernel that assumes it: `kFourWarps` is `4 * WWR_WARP_SIZE`, and a
FelizJueves unit emitted with a fixed block has shared arrays sized for
exactly that block, its header saying "launch with exactly that block". The
warp width itself is a configure-time define on both sides of the kernel
(`WWR_WARP_SIZE`, mirrored into `FJ_WAVEFRONT`), chosen when the backend is.
Rust passes dimensions, never a block size, and never learns the warp width.

**Dispatch on angular momentum stays in C++.** FelizJueves emits, for each
kernel family, one unit per class and a header with the template's primary
declared but defined nowhere, each class's specialization declared, and a
dispatch table `<template>_kernels[la][lb]` over them. A class whose unit is
not compiled in fails to link, so the table cannot name a kernel that does
not exist. The launcher indexes the table with the run-time `la` and `lb`,
and refuses a pair outside it. That is the one refusal a wrapper owns
("Statuses"). The alternative, `const` generics in Rust selecting the kernel,
would need one C entry per class, 1205 of them in `integrals` alone, and
would move the emitter's table into the ABI. `const` generics still have a
place above the wrapper: a shell type carrying its `l` at tier 1 is lowered
to an integer at the call.

**The table's limit is a constant the header exports.** The C++ launchers
`static_assert` their table's extent against a constant in the bridge
(`kPairBoundMaxMomentum`, `kRecordBoundMaxMomentum`), because a host module
unit cannot include the `.cu`'s table. The forked C headers export it
instead (`FB_PAIR_BOUND_MAX_MOMENTUM`, `FB_RECORD_BOUND_MAX_MOMENTUM`,
`FB_OVERLAP_MAX_MOMENTUM`, `FB_KINETIC_MAX_MOMENTUM`,
`FB_POTENTIAL_EN_MAX_MOMENTUM`), each `.cu` asserts it against its table, the
bindings carry it as a Rust `const`, and the safe wrapper can refuse an
unsupported momentum before the call, at tier 2, with the same number the
kernel was built from.

## Scratch: the workspace query

**The C++ tree has no query.** A launcher that needs scratch takes it as
explicit arrays whose lengths live in its doc comment: the merge sort's
`keys_alt` and `values_alt` are "n long, touched only past one tile";
`pick_lowest`'s scratch is `kCiLowestScratch` 64-bit words; `sort_bounds`
takes four `n_rec`-long arrays. The sizes are correct and unenforced, which is
the footprint problem the programming model names, one level down.

**Every forked launcher that takes scratch gets a query,
`fb_<launcher>_workspace_bytes(dims)`.** None forked so far takes scratch.
The query is a plain host function in the same header, pure in its
dimensions, and it returns the bytes of one buffer the wrapper partitions
internally, with each piece aligned as the kernel
needs. The step's `mem_budget` calls it and declares that amount as
scratch, `run` carves it from its `Lease`, and the wrapper receives one pointer and its length. A size that is a fixed
constant is exported as a constant instead, and the bindings carry it across.

**Pure in dimensions means two things.** First, no runtime call inside: the
query runs on the host in budget mode, where a dry budget has no
device. Second, if a kernel's scratch depends on the device, say an SM count
for a persistent grid, that count is one of its dimensions, and Rust passes
it: a step's GPUs are fixed before budget mode. The query is
checked at tier 4 by handing the wrapper exactly the bytes it asked for under
`memcheck` ([`testing.md`](testing.md), "The C++ boundary").

**Alignment is part of the kernel's contract.** In the C++ tree every buffer a
launcher receives came from the pool, whose allocations are 256-byte
aligned, and nothing had to say so. Under a packed scratch region the
packer decides where a carve starts inside a slab ([`memory.md`](memory.md),
"Step scratch"), so the alignment a kernel assumes must be stated. The C
header exports it, one tree-wide `FB_DEVICE_ALIGNMENT` unless a kernel needs
more, and the safe wrapper checks each pointer against it at tier 2. The
workspace query's internal partitioning follows the same constant, so a
wrapper never assumes more of its scratch than it was given.

**A vendor library's query is not pure, so its answer is a dimension.**
cuSOLVER's and hipSOLVER's `*_bufferSize` take a handle, which a
dry budget does not have, and the answer can differ between
backends and library versions. So the orthogonalizer step does what the
C++ `diagonalize_footprint(lwork)` already does: it asks
`GpuResources::syevd_lwork` on GPU 0's handle at budgeting and lays out
its `syevd` work at exactly that many doubles. A number too small is the
solver's refused argument, a negative flag, which the call returns as
`GpuRuntime`.

## Statuses

A launch wrapper returns `fb_launch_status`, an `int32_t` on the wire (a C
enum's width is not fixed), in three ranges:

| Value | Meaning | On the Rust side |
|---|---|---|
| `0` | Success | `Ok(())` |
| `> 0` | The runtime's launch error, as the C ABI's `wwr_status` | The same `gpu_status` path as every other wwr call, with `wwr_status`'s names |
| `< 0` | The wrapper's own refusal: `FB_LAUNCH_UNSUPPORTED` (no kernel for these momenta), `FB_LAUNCH_INVALID_ARGUMENT` (a count not positive, a null pointer) | `ErrorKind::Unsupported`, `ErrorKind::InvalidConfig` |

**The vendor code never crosses.** After the launch the wrapper calls the C
ABI's own `wwr_get_last_error()`, which returns `wwr_status` already mapped,
rather than wwr's C++ `wwrGetLastError()`, which returns the vendor's enum.
The wrapper therefore holds no mapping table, and the names Rust sees are the
ones every other `wwr-sys` call produces.

**Refusals and launch errors are separated.** In the C++ tree both come back as
`wwrErrorInvalidValue`: a launcher refusing a non-positive count returns the
same code the runtime returns for a bad launch. The negative range splits
them, so that an unsupported momentum reaches Rust as `Unsupported` through
the whole stack and is tested as such from Rust.

**Out of memory at a launch names the kernel stack.** A launch allocates
nothing but the kernel's stack, which no budget counts
([`memory.md`](memory.md), "Caps and margins"), so when the positive range
holds `wwr_status`'s out-of-memory member the error says so after the
status's name, on either backend: the kernel stack is the likely cause, and
the remedy is a lower device cap or `[eri] mem_per_gpu`, or
`GPU_MAX_HW_QUEUES` on HIP. The text is `fb_kernel_abi::launch_status`'s,
pinned by its tier-3 test, and no other status carries it.

**One bridge names the positive range.** `fb_core_device::launch_status`
hands `fb_kernel_abi::launch_status` the runtime's statuses and its
out-of-memory code, and every safe launch calls it: `GpuResources`'
recipes and the `-sys` crates' safe calls.

**A kernel's own fault is not a launch status.** It surfaces at the next
synchronization, which the step's wrapper owns, and it is sticky: the device
accepts no more launches. [`diagnostics.md`](diagnostics.md) has how a sticky
fault is reported once and stops launches on its device.

On the Rust side the integer is `fb_kernel_abi::LaunchStatus`, a
`#[repr(transparent)]` newtype over it that the extern functions return, so
a launch status cannot be passed where a `wwr_status` or a count is wanted.
It is converted by value, in `fb_kernel_abi::launch_status`, never made into
a Rust `enum`, so a value Rust does not know is an error rather than
undefined behaviour. The Rust list is the C list: its constants are
generated from `fb_launch.h`, the conversion names them rather than their
numbers, and the tier-3 test that regenerates the bindings fails on a
difference.

## Shared types

A type crosses the boundary as data when both sides read its fields: a
primitive-pair view, a shell's exponents and coefficients, a pair list. Such
a type has one source, a C header:

- **The header is plain C.** No templates, no vendor types, no layout that
  depends on the backend. The crate learned the last rule the hard way:
  `wwrDeviceProp` differs between CUDA and ROCm and could not be mirrored,
  and `MemPoolProps` happens to match and had to be checked by hand against
  both vendors' headers.
- **The C++ bridge's view headers are already this,** in C++ spelling.
  `primitive_pair_class_view.h` is included by the `.cu` and by the module's
  global module fragment as "the single definition"; its fork in
  `rust/integrals/kernels/` is the same fields in the same order, as a C
  `typedef struct`, and loses nothing.
- **Rust's `#[repr(C)]` mirror is generated** by `devtools/rust-bindings.py`,
  never written by hand.
- **Layout is checked on both sides.** Rust gets `const` assertions of
  `size_of`, `align_of` and each field's `offset_of` (tier 1); C++ gets the
  `static_assert`s `EinsumPlan` already has.
- **Only some modules need this.** A module whose wrappers take only pointers
  and integers has no shared type and no compatibility layer.

**Constants are the one shared header that is not plain C.** The C++
tree's `constants.h` is included by eleven launchers, and it is templates
over the element type with explicit specializations for the two complex
types, so it cannot be mirrored as it stands. Its fork in
`rust/device_headers/` holds only `kPi<T>` today, which no wrapper passes
across, so nothing is mirrored yet. When a number must cross, the split is: a plain-C `fb_constants.h` of object-like
macros holds the numbers that both sides need (the physical constants, the
unit conversions, `π`), `constants.h` defines its templates from those
macros, and the bindings script turns the same macros into Rust `const`s. One source,
two spellings, no copy. The typed literals (`kOne<T>`, `kImaginaryUnit<T>`)
are a convenience for passing a scalar by address to BLAS and have no Rust
counterpart.

A `.cuh` that is C++ only (templates, device functions, `block_params.h`,
`primitives.h`) stays below the C ABI and is never mirrored.

## The safe wrapper

The `-sys` crate's functions are `unsafe` because the C side trusts its
pointers and lengths. The safe wrapper above it is where the bridges' doc
comments become checks, and which kind of check each becomes follows
[`testing.md`](testing.md), "Contracts are part of the spec": what the
launch's memory safety rests on is checked in every build, and what only its
correctness rests on is a contract.

- **Buffers are typed slices with lengths, and the lengths are checked in
  every build.** A device slice knows its length and
  element type, so "`bound` is `n_rec` long, written whole" is a check on
  the length, not a sentence. A release build that skipped it would let the
  kernel read or write past the buffer, so it is always on, as is an extent
  that must fit the kernel's 32-bit index (`n²` for an `n × n` matrix): a
  `#[requires]` on the arguments, written with `fb_core_log`'s value checks
  so a tripped one prints the lengths it compares
  (`#[requires(eq!(dst.len(), src.len()), "the two slices are one length")]`,
  `fits_i32!(n.checked_mul(n))`, checked so it never wraps), and an
  `assert!` named in the wrapper's `# Panics` only for state the body
  produces, a recomputation too long for an attribute, or a private
  helper called once per operand whose message names that operand
  ([`testing.md`](testing.md), "Contracts are part of the spec").
- **What the kernel trusts in device data travels as a type, or is the
  call's own upload.** A kernel that follows indices in device memory (a
  record's shells, a cell's record ranges and destination, a packed
  column's row and column, an atom's AO offsets) cannot have them checked
  there by the host. So the safe call takes them on the host, checks each
  against what it indexes, and uploads them itself before the launch
  (`fb_integrals_sys::mmd_records`, `three_center_spherical_class`,
  `packed_gather`, `fb_localization_sys::pm_populations`), or a type only
  the call that wrote the device bytes makes holds them, in private fields,
  for the launch that trusts them (`fb_localization_sys::Schedule`, from
  `pm_schedule`, which `pm_sweep`'s pair kernel reads; that kernel then
  writes the partners the rotations read, in the caller's slice, on the
  same stream). A raw `fb_primitive_pair_class_view`, which anyone can fill
  with any pointers, never reaches a safe call. A call's view carries the
  record fields its kernel reads and null for the rest.
- **Non-overlap is checked from offsets.** "No array may overlap another" is
  decidable from the slices' offsets at tier 2.
- **Correctness-only rules are `debug_requires`.** `la <= lb` for a basis's
  own class, a momentum within the exported table limit (the launcher
  refuses one outside it with a status, so it is never memory-unsafe), a
  count positive. Where such a contract bounds an integer argument or a
  slice's length and the wrapper's body holds the `unsafe` call, a
  `// RELEASE:` comment directly above it names what holds the bound in a
  release build: where a launcher refuses an argument out of range as
  `FB_LAUNCH_INVALID_ARGUMENT`, as the recipes' `fb_diis_solve` does `m`,
  `ld` and `newest` past `FB_DIIS_SOLVE_MAX_DIM`, the comment names that.
- **The stream's device is current**, a `debug_requires` too.

Nothing the safe wrapper checks costs a device call, and every check it makes
is one the C++ caller today makes by reading the header.

The wrapper lives in the `-sys` crate itself, beside the generated
bindings: that crate is already allowed `unsafe`, and none is published, so
the usual split of a `-sys` crate from its safe one buys nothing, with one
exception. The recipes' safe calls are `fb_core_device::GpuResources`'
methods, over `fb-recipes-sys`'s bindings, because they take the guard's
stream and handles ("The v0.0.4 runtime"); `fb-recipes-sys` keeps the
`links` key, so a change to the kernels still selects both. A type that
carries what a kernel trusts lives in the crate whose call makes it,
since a type is unforgeable only inside the crate that defines it.

## The v0.0.4 runtime

The v0.0.4 layer ([`programming_model.md`](programming_model.md), "Steps
on rustflow"; [`memory.md`](memory.md), "The budget") runs on the layer
above. Its runtime is `fb-core-device`, written over `wwr-sys`, the
recipes kernels' raw `extern` declarations (`fb-recipes-sys`'s generated
bindings and its tile constants, whose safe calls are its own), `fb-kernel-abi`'s
launch status and the device-free crates (`fb-core-error-handling`,
`fb-core-log`), and nothing else of `rust/`. It holds the current device (`ScopedDevice`),
`device_count` and `memory` (a device's free and total bytes, read for the
budget's default cap before anything runs), `Stream`, `Event`, the copies
and fills a stream enqueues, device slices, and each GPU's
`GpuResources`.

- **Each call makes its stream's device current.** A `Stream` and an
  `Event` keep their device, and every method holds a `ScopedDevice` on it
  for the call, so the layer above's "the caller made it so" is no
  precondition here: a step's thread, or a GPU's thread of a fan-out, may
  have any device current.
- **A create aborts; an enqueue, a sync or a query is a value.** Making a
  stream, an event or a device current aborts at the caller's line, as
  every create does ([`diagnostics.md`](diagnostics.md), "Allocation is a
  planning-time error"); the rest return the runtime's
  status as a `Status`, a `GpuRuntime` error naming it, out of memory
  included, since an `Allocation` error is a value only where the budget
  refuses a plan.
- **The slices are the layer above's, again.** `DeviceSlice` and
  `DeviceSliceMut` carry the same promise; `part` and `part_mut` cut
  runs (`Part`, `PartMut`) from them, and the crate's own recipes
  `split_at_mut`, each cut an always-on contract at the caller's line,
  as "The safe wrapper" has it.
  The copies' lengths are always-on contracts too: the device end sets a
  copy's length, so `upload` reads the head of a longer host end,
  `download` writes the head of a longer one, and `copy_device` writes the
  head of a longer destination.
- **The host end of a copy is a `HostSlice`, handed over by value, and
  the copy's token holds it until a sync.** A copy from or into pinned
  memory runs after the call returns, and a borrow that ends with the call
  says nothing about a copy that has not run. So `upload` and `download` take
  the host end by value and return a `Pending`, which holds it until a
  sync of the stream has covered the copy: `Pending::sync` hands it back,
  without blocking again when an earlier sync of the stream already
  covered the copy (a ticket the stream's syncs count), so a step's several
  copies and its one sync cost that one sync. Safe code cannot touch the
  bytes mid-copy: the value is the token's (a `compile_fail,E0382` doctest
  on `PinnedBuf`). A token dropped unsynced leaks its host end, never
  freeing memory a copy may use, which for a lease's tenant costs nothing,
  its bytes the step's until the step-end sync; `mem::forget` leaks it the
  same way. A reference is no host end, since its borrow ends when it is
  dropped or forgotten: the memory's owner (the budget's pinned and heap
  buffers, a lease's pinned tenant) promises in an `unsafe impl` that only
  the value reaches the bytes, and that a leaked one leaves them valid and
  unreached until the work enqueued while it lived has finished. A ring
  that rewrites a buffer's host end once the copy that read it has run
  gives each buffer a stream of its own, so the token's sync waits for
  that buffer's copies and no other's.
- **One host end goes to every GPU under one token.** A step that uploads
  the same bytes to each GPU (the metric) stages them once:
  `Pending::and_upload(stream, dst)` enqueues a second upload of the
  token's host end on another stream, any device's, records an event
  there after it, and has the token's stream wait for that event, so the
  token returned covers both copies and its one sync, or the step-end
  sync of its stream, waits for every copy that reads the bytes. An event
  is made and destroyed within the call (a create aborts, as above; the
  wait keeps the recorded point, not the event). One staging rather than
  one per GPU, because the bytes grow with the GPUs otherwise: the
  metric's is `n_aux² × 8`, 104 MB at gly10's 3606 aux, 312 MB more
  pinned on four GPUs. Only an upload's token takes it: the token carries
  whether its copies only read the host end (`Pending<'_, H>`, an
  upload's) or write it (`Pending<'_, H, false>`, a download's, which has
  no `and_upload`: a `compile_fail,E0599` doctest), since a second copy
  may read the bytes only while nothing writes them. The cost is an
  order: gpu0's stream waits for the later GPUs' copies before anything
  enqueued on it after the call, which a host step, whose copies are its
  last work, does not see.
- **`Stream::sync` counts and times itself.** `sync_meter` reads the syncs
  made so far and the time blocked in them, a difference over a span
  being one step's. In this layer a host sync is the step wrapper's, after
  `run`, a copy token's (`Pending::sync`), and a host's wait on one point
  of a stream (`Event::sync`); each call into the runtime's sync carries an
  `allow` of `disallowed_methods` naming that. Inside a step's `run` a
  host sync is allowed at one site, a step reading a solver's flag after
  its last enqueue (the orthogonalizer's syevd `info`, through its
  download's token): it waits for no part of the step still to come, and
  the wrapper's own sync follows at once. The call carries an `allow` of
  `disallowed_methods` naming that site. A manager's ring is the second:
  it waits on one slot's copy token before it reads or rewrites that
  slot's host end, each slot on a stream of its own, so the wait covers
  that slot's copies alone and never a GPU's stream or the device, and it
  joins its streams before `run` (or a method such as `build_jk`, called
  from a reader's `run`) returns (`fb_steps::EriManager`, its build's ring
  and a J/K pass's). Between a slot's stream and the GPU's the order is an
  event, never a sync: the build's fit waits on the event its slot's
  stream recorded after the batch's kernels, and the slot's download on
  the one the GPU's stream recorded after the fit. Each such call carries an `allow` naming it.
  The eri build's screen is another: once per build, before its first
  batch, each GPU reads its records' Schwarz bounds through their
  download's token on its first ring stream, which waits for the records
  and their bounds alone, never the GPU's stream, and decides on the host
  which shell pairs get cells (`fb_steps::EriManager`, `[eri]
  schwarz_threshold`); at a threshold of 0 it reads nothing. The call
  carries an `allow` naming it. A loop
  inside `run` that decides on the host is the third: once per turn,
  after the turn's last enqueue, it reads that turn's scalars through
  their download's token (`Pending::sync`), which waits for the turn it
  reads and for no part of a turn still to come, and enqueues nothing
  more before it has read them (`fb_steps::ScfStep`'s convergence read:
  its energy, residual norm, DIIS flag and syevd flag; and, before its
  first turn, the guess's eigenvalues and syevd flag, whose count above
  the floor is the first J/K pass's width). `[scf]
  stage_sync`, a measurement off by default, syncs gpu0's stream at the
  three ends where an iteration would end a step were it three. Each
  such call carries an `allow` naming it; the SCF's read of its converged
  orbitals after its last enqueue is the solver-flag site.
- **One handle set per GPU, under its lock.** `GpuResources` is a GPU's
  stream, a BLAS and a dense-solver handle made on that stream, and the
  handle workspace; `GpuResources::new` returns it inside a `Mutex`, so
  every call is a method reached through the guard, `&mut` for the length
  of the hold. `SetStream` is called once, when the handles are made, and
  the workspace, `HANDLE_WORKSPACE_BYTES` (128 MiB) of the budget's, is
  bound once and kept until the handles are destroyed, since the library
  keeps its pointer; no handle manages memory of its own. The order is
  cuBLAS's: a `SetStream` resets its handle to a workspace of its own,
  while hipBLAS keeps the caller's, and binding gives back rocBLAS's own
  pool (`experimental/rust/handle_ring`), so `SetStream` then the bind is
  right on both. Nothing the guard
  lends outlives it, a `compile_fail` doctest on the type.
- **A call takes slices and checks them before it enqueues.** The BLAS
  calls (`gemm`, `gemm_strided_batched`, `gemv`, `geam`, `dgmm`, `syrk`)
  hold each operand to its extent, the private `check_*` helpers' asserts,
  one per operand; a solver call repeats its size query at the same
  arguments and panics at the caller's line (`#[track_caller]`) on a
  shorter work slice, before anything is enqueued, since the vendor does
  not check `lwork` and writes past a short one. A work buffer is never
  null: a query of 0 is answered 1 (`syevd_lwork`), since on HIP a null
  `work` puts hipSOLVER's handle in library-managed mode.
- **The recipes take the guard.** `GpuResources::inverse_sqrt` is the
  recipes' kernel (each element `1 / sqrt(x)`, 0 at or below a floor) on
  the guard's stream, `GpuResources::diis_solve` the recipes' bordered
  DIIS solve there, and `GpuResources::mirror_triangle` the recipes' copy
  of one strict triangle onto the other, a `syrk`'s `fill` passed
  through. `GpuResources::trsm` is the tree's
  blocked triangular solve ("Layering"), and
  `GpuResources::cholesky` the blocked Cholesky in place of a vendor
  `potrf` (rocSOLVER's raced across processes, #410); their diagonal
  kernels and their gemms and syrks land on the one stream the guard has,
  so a recipe cannot be handed another (a `compile_fail` doctest hands
  `trsm` a stream). No vendor `trsm`, `potrs` or `getrs` is bound, as
  "Layering" has it.
- **A kernel launch on this layer is its library's `-sys` crate's.**
  `fb_integrals_sys::three_center_spherical_class` is the first: one class
  of the fully spherical (μν|P) (`ssperi3<la, lb, lP>`) over a `Stream`
  and device slices, each length an always-on contract naming the kernel.
  The cells, the only indices the kernel follows, are handed in on the host
  (a `HostSlice`, by value), each checked inside the slices it indexes, and
  uploaded by the call before it launches, as "The safe wrapper" asks of
  indices the safe layer forms; the upload's token comes back, so nothing
  rewrites a checked cell before the copy has run.
  `fb_integrals_sys::mmd_records` (one class's primitive-pair records)
  and `fb_integrals_sys::packed_gather` (a packed batch into square
  matrices) follow on the same terms: the records' shell and pair tables
  (each shell's primitive range, each pair's shells and record range) and
  the gather's per-column row and column come in on the host by value,
  are checked against what they index and are uploaded by the call, and
  each equals its old-layer counterpart (`add_record_tasks`,
  `PackedTables::gather`) bitwise, the same kernel on the same inputs.
- **A library call may differ by backend where one was measured
  faster.** K's update in a J/K pass from C is a syrk on CUDA (−13% at
  gly10, RTX 3080) and a gemm on HIP (rocBLAS's dsyrk +15%, gfx1200),
  chosen at compile time from the backend the kernel libraries were built
  for (`steps/src/steps/eri.rs`, `K_BY_SYRK`).

## Build

**`rust/` builds its own native libraries.** `rust/CMakeLists.txt` is a
CMake project of its own, which the root `CMakeLists.txt` does not add. It
configures `deps/WarpWraps` alone and builds what the crates link: WarpWraps'
C ABI library, `libwwr_capi_runtime_api.so`, and the kernel shared objects,
from the launchers and kernels forked under `rust/` and the device headers
in `rust/device_headers/`. It configures nothing of the C++ tree, and no
change to the C++ tree's CMake precedes a port. Its only inputs from outside
`rust/` are `deps/WarpWraps` and FelizJueves' `include/` from
`deps/FelizJueves` ([`README.md`](../README.md), "What it depends on").

**The backend is chosen once.** The project's two presets, `cuda` and `hip`,
build into `rust/build-cuda/` and `rust/build-hip/`, and
`FREE_BREAKFAST_GPU_BACKEND` has no default, for the C++ tree's reason. The
crates' `build.rs` include `rust/capi_dir.rs`, which links whichever of the
two holds the C ABI library and refuses to choose when both do, rather than
defaulting to one: `FB_RUST_BUILD_DIR` names the build then.
`devtools/cross-backend-check.sh` builds this project for its backend too,
because nvcc accepts code that `clang -x hip` rejects.

**One shared object per module.** `rust/<module>/kernels/` builds
`libfb_<module>_kernels.so` at `<build>/<module>/kernels`:
`libfb_recipes_kernels.so` (the inverse square root, the
symmetrization, the Cholesky diagonal tile, the DIIS solve and the
triangular solve's diagonal block:
`fb_inverse_sqrt`, `fb_symmetrize`, `fb_mirror_triangle`,
`fb_cholesky_diagonal`, `fb_diis_solve` and `fb_trsm_diagonal_f64`) and `libfb_integrals_kernels.so` (the
primitive-pair records, the overlap, the kinetic energy, the nuclear
attraction, density fitting's metric and packed three-center integrals,
the bra Cartesian or spherical, and density fitting's gather over the
packed shell-pair map: `fb_mmd_*`,
`fb_ordered_pair_bound`, `fb_overlap_class`, `fb_kinetic_class`,
`fb_potential_en_class`, `fb_metric_class`, `fb_three_center_packed_class`,
`fb_three_center_spherical_class` and `fb_packed_gather`) and
`libfb_localization_kernels.so` (Pipek-Mezey's populations, pair angles,
rotations, functional and identity: `fb_pm_*`) today, each beside
its two ABI entries; `fb-doc-check`'s `doc_lists.rs` fails a function a
library's bindings declare that this list does not name. Under CUDA each uses
separable compilation with the device link resolved inside the `.so`
(`CUDA_RESOLVE_DEVICE_SYMBOLS`), so nothing relocatable crosses the ABI, and
links the shared vendor runtime, not CMake's static default, so that the
launches inside it and the C ABI's `wwr_get_last_error()` read the same
last error. Under HIP the `.cu` compiles as C++ with `hip::device` private
to the library. A library links no other, and includes none of another's
sources (`libfb_integrals_kernels.so` linked the recipes library for the
records' sort until the fork left that order out), so a file under a
module's `kernels/` reaches that module's `-sys` crate and its dependents
alone. The generated units find
FelizJueves' runtime header with `FJ_WAVEFRONT` set to `WWR_WARP_SIZE`.

One Rust `-sys` crate per library, `fb-recipes-sys`, `fb-integrals-sys`
and `fb-localization-sys` (each declaring the library's `links` key, and
`fb-core-device` linking the recipes' through `fb-recipes-sys`),
which is the workspace graph the test selection runs on. Their `build.rs`
include `rust/kernel_lib.rs` after `capi_dir.rs`: `link_kernel_lib(module)`
finds the library in the same `<build>` the C ABI library came from, links
it and sets the rpaths. A file under a module's `kernels/` re-runs the tier
4 of that module's `-sys` crate and of every crate that depends on it
([`testing.md`](testing.md), "Selecting GPU tiers in cargo"); the cargo half
of the gates builds this project for its card's backend before it runs, so a
broken device link fails there.

Shared rather than static, because the static route is the harder one and
the reason for it does not apply. `experimental/rust/s_overlap` links a static
archive, and had to record CMake's whole link line with a probe executable
and replay it from `build.rs`, because `/opt/spdlog`'s archive is not
position-independent and nothing shared can carry it. A kernel library pulls
in none of that: its dependencies are the vendor runtime and the C ABI
library, which the crates load already.

**Bindings are committed.** `devtools/rust-bindings.py` generates each
library's `-sys` bindings from its one C header and the output is checked
in; `fb-device-check`'s `bindings.rs`, a tier-3 test, regenerates them and
diffs. The script is stdlib Python with no bindgen and no libclang: it reads
a small plain-C grammar (typed `#define`s, scalar `typedef`s, `typedef
struct`s, prototypes and `///` comments) as text, as WarpWraps' C ABI
generator does, and anything outside that grammar is an error, not a guess.
Two precedents in the tree made the same choice: WarpWraps' C ABI commits
its `gen/` snapshot "so the PR is reviewable without running anything", and
FelizJueves' kernels are committed. A reviewer then sees a changed signature
in the diff.

**The emitted units are emitted again, not copied by hand.** The
`tools/emit` generators take `--out-root DIR`, which writes the same tree
under `DIR` that their default writes for the C++ tree, and `rust/` carries
the families it has forked from such a run, each directory at the same
relative path. From one generator at one `deps/FelizJueves` they are byte
for byte the C++ tree's units, so after a bump both trees are emitted again
([`README.md`](../README.md), "Build and test", has the commands). A
family the C++ tree does not have ([`design.md`](design.md), "What already
exists": a kernel the C++ tree lacks too is added to `rust/` alone) has no
C++ twin to agree with: its generator writes `rust/` alone, by default,
and its own test emits it again under another root and compares byte for
byte. The packed three-center integrals' units,
`tools/emit/packed_three_center_kernels.py`'s (`speri3`, and FelizJueves
#745's fully spherical `ssperi3`), are the one such family.

**FelizJueves can emit part of the wrappers.** Its header already names the
block a unit was sized for and emits the dispatch table, and for a
fixed-block family the launcher is mechanical: index the table, refuse an
out-of-range pair, `idivup` the count by the block, launch, return
`wwr_get_last_error()`. The pair-bound, record-bound and one-electron
launchers are this. The launchers that do more stay hand-written: `mmd`'s
record formation with its own kernels, Coulomb's sort and screen, exchange's
helper kernels. Emitting the mechanical ones is a FelizJueves change, so it
is a request to the submodule; until it lands they are written by hand to
the same shape, and the shape is what the request asks for. Every launcher
emitted is one fewer hand-kept copy for the fork to carry ("What the port
changes").

## Open questions

- **Structs by pointer or by value: settled, by pointer.** A struct crosses
  as a `const` pointer to a plain-C `typedef struct` in host memory, a small
  POD such as `EinsumPlan` included, because one rule is uniform; the wrapper
  reads the fields on the host and launches the kernel with what it needs.
  The first three crossed that way when `mmd` and the overlap were forked,
  and `fb_nuclei_view` with the nuclear attraction ("From bridge to C
  ABI"), each layout asserted on both sides ("Shared types").
- **The C ABI's own evolution.** This layer leans on three C ABI entries
  beyond the runtime it already used: `wwr_stream`, `wwr_status` and
  `wwr_get_last_error()`. If WarpWraps' C ABI ever grows a launch-wrapper
  convention of its own (its device surface is one unit so far), this
  tree's wrappers should match it.
- **The rest of the workspace sweep**: upper solves, rectangular `syrk`
  and `gemv`, the batched solver calls, hipBLASLt. What a sentinel test of
  the handle workspace would run.
- **cuBLAS and rocBLAS handle thread-safety**: the lock answers it either
  way; verify what each documents.
- **The `cells` work list as a kernel's own**: a dense cross product a
  kernel could derive from two short lists (58,653 and 1,272 entries on
  gly10), a FelizJueves change. It lowers the ERI buffer a batch needs.
