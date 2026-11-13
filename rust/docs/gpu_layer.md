# Rust rebuild: the GPU layer

The kernels stay in C++ and CUDA, hand-written and FelizJueves-generated, and
Rust owns everything above a launch. The device code a crate needs is forked
into `rust/`, beside the crate that calls it, and built by `rust/`'s own
CMake project: `rust/` shares no source file and no build with the C++ tree
("What the port changes"). This document is the contract between
the two: what a launch wrapper may do, what crosses the C ABI and in which
shape, where the launch shape and the angular-momentum dispatch live, how
scratch is sized, what a status means, and how the libraries are built and
bound. It leans on the programming model's terms, *task*, *layout* and
*arena* ([`programming_model.md`](programming_model.md)), and on the test
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
and they say how much a full port forks. Forked so far: the merge sort
(`rust/recipes/kernels/`), and the primitive-pair records' and the
overlap's launchers and emitted units (`rust/integrals/kernels/`), over a
trimmed fork of the device headers they include (`rust/device_headers/`).

## What the port changes

**The bridge header is the port line.** Everything above a module's bridge
is Rust's to port; everything below it is forked. In the C++ tree `mmd.cpp` builds
`PrimitivePairSource` on the host, packs the shells' exponents and centres
into device arrays, computes `rec_begin`, and then launches. Under this
design all of that is Rust: the wrapper receives device pointers only, and
the host preparation, the copies that fill them and the arena they come from
belong to the task and the orchestrator. "Porting a module" therefore has a
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
launcher is a `.cu` (`mmd_launch.cu`, `merge_sort.cu`), because the launch
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
code a port forks, the launchers, the merge sort and the shared device
headers, exists twice, once in each tree, and nothing keeps the two copies
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
reaches, and the second `FB_BACKEND_CUDA` or `FB_BACKEND_HIP`
(`fb_launch.h`), chosen by the compiler's own `__CUDACC__` or `__HIP__`
rather than a definition the build passes. The `-sys` crate's
`check_library()` checks the version against the constant in its committed
bindings and the backend against the CMake build whose C ABI library
`wwr-sys` links, which its `build.rs` reads from that build's
`CMakeCache.txt`, since wwr's C ABI has no query naming its backend. It
refuses a mismatch with a message naming both values. A library built for
the other backend otherwise fails at its first launch, or deep inside wwr,
with nothing naming the cause.

"At load" is the first call, not the dynamic loader: Rust has no load-time
hook short of a linker section an `rlib` may drop, so `check_library()` is a
plain function that a caller runs once before its first launch, and each
`-sys` crate's tier 4 runs it against the library built for its card. The
comparison itself is a private function of values passed in, unit-tested
with a mismatched version and backend in the same crate; since a `-sys`
crate links its library, those unit tests build only where the library is
built, and need no card.

## Layering

```
 Rust      orchestrator: DAG, fork, join, wait, sync; streams, events, memory
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
holds state between calls. Fork, join, wait and sync belong to the
orchestrator, and so does every allocation. A wrapper is therefore safe to
call from any thread, and its whole cost is on the stream it was given. The
rule is cheap to keep: a tier-3 test, `fb-device-check`'s
`launch_wrappers.rs`, greps every forked `.cu` and `.cuh` under `rust/` for
the runtime's allocation, synchronization, event, copy and device calls, the
grep that produced the zero above.

**And so it is capture-safe by construction.** Plan-once-run-many leans on
stream capture into a replayed graph ([`programming_model.md`](programming_model.md),
"Plan once, run many"), and a function that only enqueues on the stream it
was given can be captured. The two items the "never" list gained are the two
that would break it: a read-back to the host is a synchronization, and a
profiler range belongs to the orchestrator, which brackets the call with the
task's scope ([`diagnostics.md`](diagnostics.md)). A wrapper that launches
several kernels returns the first failure; what it enqueued before stays
enqueued, which is harmless, because a launch error is not sticky and a
sticky fault stops everything after it anyway.

**The current device is the stream's, and the caller made it so.** A launch
on a stream of a device that is not current fails at the runtime. Today the
C++ pipeline runs on one device and never changes it; the orchestrator places
tasks on many, so each stream worker is a thread of its own with its device
made current for the life of the run (`ScopedDeviceIndex`, which the crate
already has, held by that thread; [`programming_model.md`](programming_model.md),
"CPU workers"), and every wrapper a task calls runs on it. The wrapper does
not check, because it cannot without a runtime call
of its own; the safe wrapper's contract (tier 2) can, in debug builds.

**Library calls go straight from Rust through `wwr-sys`.** BLAS, the
solvers, the collectives: WarpWraps' C ABI already covers them, and the crate
already drives a GEMM that way (`experimental/rust/gemm`). Only this tree's
own kernels need a C++ wrapper.

**A library's workspace is part of the task's layout.** The solvers' is
already a dimension: the pipeline treats `syevd_lwork` as one, and a layout
calls the query at planning ([`programming_model.md`](programming_model.md),
"Memory: layouts and arenas"). cuBLAS and hipBLAS allocate a workspace per
handle on their own unless they are handed one (`cublasSetWorkspace` and its
HIP twin), so the orchestrator hands every handle a buffer from the arena,
and a run allocates nothing through a library either.

**Rust never launches a kernel itself.** The C ABI exports a launch call
(`wwr_launch_kernel_ex_c`), so Rust could take a kernel's address and launch
it. It does not, because the block and the dispatch belong with the kernel
(below), and moving them to Rust would put the generated tables into the ABI.

## From bridge to C ABI

What changes when a C++ bridge header is forked into `rust/` as the C header
the bindings are generated from. The forked `mmd.h`, `overlap.h` and
`merge_sort.h` follow this table:

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
library shares, `fb_launch_status`, its values and `FB_DEVICE_ALIGNMENT`, is
`rust/device_headers/fb_launch.h`. The two integer types are the bridges'
own: every count, index and momentum is an `int32_t`, and bytes, which only
the workspace queries and the arena speak, are `size_t`.

The seven template bridges, none forked yet, are `recipes`'
element-generic kernels (`abs_at`, `complex_cast`, `expm`, `einsum`, `cg_unitary`, `inverse_sqrt`,
`polynomial_roots`). A C ABI cannot export a template, so each explicit
instantiation becomes its own entry, the pattern WarpWraps' C ABI set for its
own device templates (`extension_random_normal`, `wrappers_*`: one
monomorphised `extern "C"` thunk per instantiation, the type in a suffix).
The instantiation list is the source: the entries are generated from it, so
none is invented and none dropped.

The views and parameter structs cross as they are, by pointer from host
memory. `PrimitivePairClassView`, `PrimitivePairSource` and
`PrimitivePairRecords` are forked as `fb_primitive_pair_class_view`,
`fb_primitive_pair_source` and `fb_primitive_pair_records`, the same fields
in the same order. The CI `Replacement` and `EinsumPlan` are integers and
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
`FB_OVERLAP_MAX_MOMENTUM`), each `.cu` asserts it against its table, the
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
`fb_<launcher>_workspace_bytes(dims)`,** as `fb_merge_sort` and
`fb_mmd_order` have. It is a plain host function in the
same header, pure in its dimensions, and it returns the bytes of one buffer
the wrapper partitions internally, with each piece aligned as the kernel
needs. The task's layout calls it at planning, the arena carves that amount,
and the wrapper receives one pointer and its length. A size that is a fixed
constant is exported as a constant instead, and the bindings carry it across.

**Pure in dimensions means two things.** First, no runtime call inside: the
query runs on the host at planning, where the tier-3 planner tests have no
device. Second, if a kernel's scratch depends on the device, say an SM count
for a persistent grid, that count is one of its dimensions, and Rust passes
it: the orchestrator knows every worker's device at planning. The query is
checked at tier 4 by handing the wrapper exactly the bytes it asked for under
`memcheck` ([`testing.md`](testing.md), "The C++ boundary").

**Alignment is part of the kernel's contract.** In the C++ tree every buffer a
launcher receives came from the pool, whose allocations are 256-byte
aligned, and nothing had to say so. Under an arena the offset assigner decides
where a buffer starts inside a block ([`programming_model.md`](programming_model.md),
"The arena is a table and a block"), so the alignment a kernel assumes must
be stated. The C header exports it, one tree-wide `FB_DEVICE_ALIGNMENT`
unless a kernel needs more, the assigner honours it for every buffer it
places, scratch and output alike, which is enough because the block itself
comes from the plain device allocation and is at least that aligned on both
vendors, and the safe wrapper checks each pointer against it at tier 2. The workspace
query's internal partitioning follows the same constant, so a wrapper never
assumes more of its scratch than the arena gave.

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

**A kernel's own fault is not a launch status.** It surfaces at the next
synchronization, which the orchestrator owns, and it is sticky: the device
accepts no more launches. [`diagnostics.md`](diagnostics.md) has how a sticky
fault is reported once and stops launches on its device.

On the Rust side the integer is converted with `TryFrom`, never made into a
Rust `enum` directly, so a value Rust does not know is an error rather than
undefined behaviour. A tier-3 test compares the C list and the Rust list
member by member.

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
comments become checks:

- **Buffers are typed slices with lengths.** A device buffer carved from the
  arena knows its length and element type, so "`bound` is `n_rec` long,
  written whole" is a `debug_requires` on the length, not a sentence.
- **Non-overlap is checked from offsets.** "No array may overlap another" is
  decidable from arena offsets at tier 2.
- **Ordering rules are checked.** `la <= lb` for a basis's own class, a
  momentum within the exported table limit, a count positive.
- **The stream's device is current**, in debug builds.

Nothing the safe wrapper checks costs a device call, and every check it makes
is one the C++ caller today makes by reading the header.

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
`libfb_recipes_kernels.so` (the merge sort) and `libfb_integrals_kernels.so`
(the primitive-pair records and the overlap) today. Under CUDA each uses
separable compilation with the device link resolved inside the `.so`
(`CUDA_RESOLVE_DEVICE_SYMBOLS`), so nothing relocatable crosses the ABI, and
links the shared vendor runtime, not CMake's static default, so that the
launches inside it and the C ABI's `wwr_get_last_error()` read the same
last error. Under HIP the `.cu` compiles as C++ with `hip::device` private
to the library. A library may link another for a host call:
`libfb_integrals_kernels.so` links the recipes library for `fb_merge_sort`,
and no device symbol crosses between them. The generated units find
FelizJueves' runtime header with `FJ_WAVEFRONT` set to `WWR_WARP_SIZE`.

One Rust `-sys` crate per library, `fb-recipes-sys` and `fb-integrals-sys`,
which is the workspace graph the test selection runs on. Their `build.rs`
include `rust/kernel_lib.rs` after `capi_dir.rs`: `link_kernel_lib(module)`
finds the library in the same `<build>` the C ABI library came from, links
it and sets the rpaths. A file under a module's `kernels/` re-runs every
kernel `-sys` crate's tier 4, since one library links another
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
([`README.md`](../README.md), "Build and test", has the commands).

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
  The first three crossed that way when `mmd` and the overlap were forked
  ("From bridge to C ABI"), their layout asserted on both sides ("Shared
  types").
- **Where the workspace query runs for a planner with no library.** The query
  is a host function in the kernel's `.so`, so a tier-3 planner test that
  calls it links that `.so` even though it launches nothing. Loading it is
  safe on a device-less box, as WarpWraps' device surface notes, but the
  dependency is real. A pure-Rust mirror of the formula would remove it and
  reintroduce a second program about sizes, which the query exists to avoid.
- **The C ABI's own evolution.** This layer leans on three C ABI entries
  beyond the runtime it already used: `wwr_stream`, `wwr_status` and
  `wwr_get_last_error()`. If WarpWraps' C ABI ever grows a launch-wrapper
  convention of its own (its device surface is one unit so far), this
  tree's wrappers should match it.
