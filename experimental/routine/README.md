# Routine spike: check-before-invoke and an allocation key (#1246, #1324)

Does a routine shape whose `check` the compiler enforces, and whose permission
to allocate is checked at compile time, hold up on one real routine, under
named modules and on the GPU? This spike measures that on
`mcscf::root_rotation`. #1246 built it and came back **rework**; #1324
reworked the two parts that did not survive, the footprint over live buffers
and the place of the key's friend. Nothing under `src/` changes.

**Verdict: promote the base, `In`, `Shape` and `check`; drop the key.**

- The `Routine` base, the `In` struct and `check` cost nothing measurable,
  work across module boundaries, and the port still matches the original bit
  for bit on the gfx1200.
- `footprint(shape)` now takes a shape-only `Shape` built from sizes alone,
  so pre-flight can call it before any buffer exists. A `Shape` built from the
  fixture's JSON sizes equals `In::shape()` of the live call, and its footprint
  is the pool's peak to the byte (63,616 bytes, 22 allocations).
- `check`'s aborts now have a death test, through the base, before `invoke`.
- The key's friend has exactly one unforgeable home: the root defined in the
  key's own module, as a partition. The fifth must-not-compile case shows a TU
  outside that module cannot forge it. But in `src/` that placement is a
  module cycle (measured below), because the key sits in `core` under every
  routine and the root is the pipeline's planner above them. Every placement
  that compiles in `src/`'s layering is forgeable by any TU. Keying
  `root_rotation` would also first need 3 callees keyed, with about 87 call
  sites. A guard any TU can bypass is not worth spreading up the call graph
  like `const`, and the pool-peak tests already catch an allocation that a
  footprint misses.

## What is here

| File | What |
|------|------|
| `key.cppm` (`:key`) | `Alloc { No, Yes }`; `AllocKey` (private default constructor, deleted copy, user-provided `constexpr ~AllocKey() {}`, one friend: `RootRotationStep`, attached to this module); `KeyedBuffer<T>`, a `Buffers::device<T>` whose constructor takes `const AllocKey &` |
| `base.cppm` (`:base`) | `Routine<Self, Alloc>`: a static `operator()` over one `const typename S::In &` (`S = Self`) that calls `Self::check(in)` and then `Self::invoke`, taking and passing on the key for `Alloc::Yes` and taking none for `Alloc::No` |
| `root_rotation.cppm` (`:root_rotation`) | `RootRotationRoutine : Routine<…, Alloc::Yes>` with its `Shape`, its `In` (and `In::shape()`), `check`, `footprint(shape)` and `invoke`; the object `root_rotation`; `RootRotationStep`, the root that mints the key and calls the port as a step would |
| `root_rotation.cpp` | The implementation unit: `In::shape`, `check`, `footprint`, `invoke` and `RootRotationStep::run`, plus `FBF_ROUTINE_PRE`, a local stand-in for #1226's `FBF_PRE`, which has not landed |
| `routine.cppm` | The primary interface, re-exporting the three partitions |

Tests (`test/experimental/`): `routine_gpu.cpp` (`RoutineRootRotation`, two
GPU cases, and `RoutineRootRotationDeathTest`, one) and
`routine_reject_{alloc_up,key_buffer,bit_cast,misspelled_field,forged_root}.cpp`,
each an `EXCLUDE_FROM_ALL` object library whose ctest entry
(`RoutineMustNotCompile.*`) builds it and passes only on the expected
diagnostic.

## Naming

`mcscf::RootRotation` is already the result type, so the routine class is
**`RootRotationRoutine`**, and the object a caller calls is
**`root_rotation`**, the function's own name. The reasons:

- The call site keeps the name it has today. `root_rotation(…)` stays
  `root_rotation(…)`, and only the argument list changes (see "Call-site
  diff").
- The suffix names the role, the same word as the base, `Routine<…>`.
  Nobody spells the type at a call site, only in a `::In` or a `::footprint`,
  so its length costs little.
- The result keeps the bare noun. Renaming every result type (`RootRotation`
  → `RootRotationResult`) would touch far more lines than renaming the
  routines.

Tree-wide that would be `<layer>::<Name>Routine` with `inline constexpr
<Name>Routine <name>{}`, e.g. `mcscf::RootRotationRoutine` /
`mcscf::root_rotation`.

## Did it export cleanly across module boundaries?

| Piece | Across a module boundary | Measured |
|---|---|---|
| `Routine<Self, Alloc>`, `typename S::In` | yes | The test TU and the reject TUs (global module) call `root_rotation(key, {…})` through the exported object; the static `operator()` template instantiates in the importer |
| `In`, a nested aggregate with reference members and a default member initializer | yes | Designated initialization from another TU works; the designators must be in declaration order, and a missing reference member is an error |
| Statics defined in an implementation unit | yes | `In::shape`, `check`, `footprint`, `invoke` and `RootRotationStep::run` are declared in `:root_rotation` and defined in `root_rotation.cpp` (`module free_breakfast.experimental.routine;`), and called from other TUs through `Routine::operator()`, which is instantiated in the importer and links against them |
| A friend declared in one partition, defined in another | yes, within one module | `RootRotationStep` is forward-declared (exported) in `:key` before `AllocKey` names it as a friend, and defined in `:root_rotation` |
| `requires` | not used | `Routine` needs no constraint: a missing `In`, `check` or `invoke` is an error at the call that instantiates `operator()`. A `requires` clause would only move the diagnostic |

Where a friend of `AllocKey` can live is its own section, next.

Separately: the container's compiler is now **clang 23.1.6**, not the 20.1.8
the issue's sketches used. The `bit_cast` hole needs the user-provided
destructor on 23 as well (see (c)).

## Where the key's friend lives

A friend declaration attaches the name it introduces to the declaring module,
and only that module can define it. Each candidate was built as a scratch
CMake project of two or three plain modules (no `import std`), with clang
23.1.6 and CMake 4.2.3. `key` holds `AllocKey`, as `:key` does here.
Diagnostics are trimmed to the error line.

| Candidate | Compiles in `src/`'s layering? | Forgeable by another TU? |
|---|---|---|
| (1) The friend defined in another module | no | — |
| (2) The root as a partition of the key's module | no: a module cycle | no |
| (3) A key minted by a function exported from the key's module | yes | yes, by any importer, every routine included |
| (4) A global-module friend | yes | yes, by any TU that defines the class |
| (4′) A friend class template, specialized by the planner's module | yes | yes, by any TU that specializes it |

**(1) The friend defined in another module.** `export struct Root;` and
`friend struct Root;` in `key`, `struct Root { … }` in module `root`:

```
root.cppm:3:15: error: declaration of 'Root' in module root follows declaration in module key
```

**(2) The root as a partition of the key's module.** `key:token` declares
`Root` and `AllocKey`, and `key:root` defines `Root` and mints a key: it
compiles. A TU outside the module cannot forge it. Defining a global `Root`
is refused by the attachment rule, and a `Root` in another namespace is not
the friend:

```
forge.cpp:2:8: error: declaration of 'Root' in the global module follows declaration in module key:token
forge_ns.cpp:2:67: error: calling a private constructor of class 'AllocKey'
```

This is what the spike does, with `RootRotationStep` in `:root_rotation` and
must-not-compile case (e) below. But in `src/` the key sits in `core`, below
every routine, and the root is the pipeline's planner, which imports what it
runs. Its partition would have to import a module that imports the key's
module. Given that (`key:root` importing `pipeline`, which imports `key`),
CMake's module scan refuses it before clang runs:

```
CMake Error: Circular dependency detected in the C++ module import graph. See modules named: "key", "key:root", "pipeline"
```

A root in `core` that takes the planner's work as a callable avoids the cycle
but becomes (3): anyone who can call it can mint.

**(3) A key minted by an exported function.** `export AllocKey
mint_alloc_key();` in `key`, friended there: it compiles, and so does an
`Alloc::No` routine in another module that calls `mint_alloc_key()` and
holds a key. A function that is the key's friend must be attached to the
key's module (see (1)), and a module exports to every importer or to none.
So every routine that can name `AllocKey` can mint one. It guards nothing that
`grep mint_alloc_key` does not.

**(4) A global-module friend.** `extern "C++" { namespace pipeline { struct
Planner; } }` in `key`, friended, and defined in module `pipeline` inside
`extern "C++"`: it compiles. A plain TU that defines its own
`pipeline::Planner` and mints a key also compiles, with no diagnostic. If both
definitions are linked into one program, that is an ODR violation that needs
no diagnostic. This is what #1246's `RoutineKeyMinter` was, and #1324 removed
it: with it, any TU could define the minter. It also makes `core`'s key name
a `pipeline` class.

**(4′) A friend class template.** `template <class Tag> friend struct
Minter;` in `AllocKey`. `pipeline` specializes `Minter<PlannerTag>` and mints:
it compiles. So does a plain TU that specializes `Minter<MyTag>`. It is (4)
without the upward name, and just as open.

**Decision: (2) for the spike, and no key in `src/`.** (2) is the only
placement the compiler holds, and the spike now uses it alone. With
`RoutineKeyMinter` gone, the footprint test calls through the root, which
costs nothing because minting allocates nothing. `src/`'s layering rules (2)
out, and (3), (4) and (4′) are disciplines that any TU can step around. None
of them pays for keying every allocating function and every function above
one. See the verdict.

## The compiler rejects what it should: (a)–(e)

Each is a `RoutineMustNotCompile.<name>` ctest entry that builds the TU and
passes on the regex in `test/experimental/CMakeLists.txt`. Diagnostics from
the `hip-experimental` build, clang 23.1.6, paths shortened:

**(a) An `Alloc::No` routine calls an `Alloc::Yes` one** (`alloc_up`):

```
test/experimental/routine_reject_alloc_up.cpp:18:45: error: no matching function for call to object of type 'const RootRotationRoutine'
experimental/routine/base.cppm:29:25: note: candidate function template not viable: requires 2 arguments, but 1 was provided
```

**(b) An `Alloc::No` routine constructs the keyed buffer** (`key_buffer`). It
has no key, so it tries to make one:

```
test/experimental/routine_reject_key_buffer.cpp:22:51: error: calling a private constructor of class 'free_breakfast::routine::AllocKey'
experimental/routine/key.cppm:38:3: note: implicitly declared private here
```

**(c) The key forged with `std::bit_cast<AllocKey>(char{})`** (`bit_cast`):

```
test/experimental/routine_reject_bit_cast.cpp:18:37: error: no matching function for call to 'bit_cast'
.../c++/v1/__bit/bit_cast.h:34:55: note: candidate template ignored: constraints not satisfied [with _ToType = fb::routine::AllocKey, _FromType = char]
.../c++/v1/__bit/bit_cast.h:32:52: note: because 'is_trivially_copyable_v<free_breakfast::routine::AllocKey>' evaluated to false
.../c++/v1/__bit/bit_cast.h:32:52: note: 'free_breakfast::routine::AllocKey' is not trivially copyable
.../c++/v1/__bit/bit_cast.h:32:52: note: because it has a user-provided destructor
experimental/routine/key.cppm:37:7: note: 'AllocKey' defined here
```

`key.cppm` also `static_assert`s `!std::is_trivially_copyable_v<AllocKey>`,
so if someone defaults the destructor the module itself stops compiling.

**(d) A misspelled `In` field at a call site** (`.ncroe` for `.ncore`,
`misspelled_field`):

```
test/experimental/routine_reject_misspelled_field.cpp:23:10: error: no matching function for call to object of type 'const RootRotationRoutine'
experimental/routine/base.cppm:29:25: note: candidate function [with S = free_breakfast::routine::RootRotationRoutine] not viable: cannot convert initializer list argument to 'const typename free_breakfast::routine::RootRotationRoutine::In'
```

It is rejected, but **the field is not named**. This is not the routine's
doing. In a scratch TU, a plain non-template `int plain(const In &)` called as
`plain({.ncroe = 1, .ncas = 2})` gets the same "cannot convert initializer
list argument". `plain(In{.ncroe = 1, …})` gets `field designator 'ncroe'
does not refer to any field in type 'In'`. A non-template base
(`Routine<Self, In>`, with `In` declared before the routine) gives the same
unnamed diagnostic too. So the regex for (d) pins the conversion to
`RootRotationRoutine::In`, the narrowest text clang prints. A call site that
wants the field named spells `RootRotationRoutine::In{…}`.

**(e) A key forged outside the root's module** (`forged_root`, #1324). A TU
that imports the module defines its own `free_breakfast::routine::RootRotationStep`
to mint a key with:

```
test/experimental/routine_reject_forged_root.cpp:13:8: error: declaration of 'RootRotationStep' in the global module follows declaration in module free_breakfast.experimental.routine:root_rotation
experimental/routine/root_rotation.cppm:96:8: note: previous declaration is here
```

(a)–(d) are quoted from the reworked design's run. They are #1246's
diagnostics, with only the line numbers in `key.cppm` changed. All five
entries pass.

## Matches the original

`RoutineRootRotation.PortMatchesOriginal` runs `mcscf::root_rotation` and
`RootRotationStep::run` (the port, with a key minted at the root) on the same
inputs in one process. The inputs are `test/gtest/data/casscf_hessian/`'s
equal-weight two- and three-root averages of water in 6-31G, set up as
`mcscf_root_rotation_gpu.cpp` sets them up. The bar is 1e-12 absolute.
Measured on the gfx1200 (HIP):

| Case | coupling (max) | θ (max) | eq. (rot) (max) |
|---|---|---|---|
| `h2o_631g_sa2` | 0 (5.5e-2) | 0 (0.45) | 0 (4.9e-2) |
| `h2o_631g_sa3` | 0 (0.13) | 0 (1.9) | 0 (0.48) |

Every difference is exactly zero, as on #1246's design. The calls are the
same in the same order, and `ci::Rdm::rdm12`'s reductions happened to come out
in the same order.

`check` turns the original's `InvalidConfig` refusals into
`FBF_ROUTINE_PRE` aborts (#1226: a broken shape is the caller's bug). The two
degeneracy refusals are numerical outcomes that a caller branches on, so they
stay in `invoke` as `Numerical`. The original's `Refusals` case becomes
`RoutineRootRotationDeathTest.CheckAbortsBeforeInvoke`. It calls the root with
`max_rotation = 0`, then with unequal weights, and expects each to abort with
check's condition text: `precondition failed: in.max_rotation > 0.0`, then
`precondition failed: routine_weights_equal(in.weights)`. `invoke` never
aborts, so an abort with that text shows the base ran `check` first. Two
details:

- `log::critical` writes to stdout, and a death test reads only stderr. So the
  child points stdout at stderr (`dup2`) before the call.
- The threadsafe style re-executes the binary for each child, and each child
  loads the fixture again. That makes it the slowest routine case, at 18.6 s.
  It would break `test/support/slow_test.h`'s rule in a gated binary. The
  experimental binary is not gated.

`FBF_ROUTINE_PRE` is still the local stand-in: #1226's `FBF_PRE` has not
landed (`grep -rn FBF_PRE src/core` is empty). The migration is a rename.

## The footprint is tight, from sizes alone

#1246's `footprint(in)` read naux from `B`'s length and nroots from
`weights`, so it could run only once those buffers existed. Pre-flight sizes a
run before anything is allocated, so #1324 splits the sizes out:

```cpp
struct Shape {
  size_t nao, naux, natoms;
  int32_t ncore, ncas, nalpha, nbeta, nroots;
  bool operator==(const Shape &) const = default;
};
static MemoryFootprint footprint(const Shape &shape);
// and on In:
Shape shape() const; // naux off B's length, nroots off the weights
```

`In` derives its `Shape` rather than holding one. A held `Shape` would repeat
sizes that `In`'s buffers already carry, and the two could disagree. Pre-flight
builds a `Shape` from the bases' sizes and the active space, and a call in
hand passes `in.shape()`. This is the "separate argument list in another
form" #1246 predicted, but it carries only sizes, so it stays small: eight
counts here, against `In`'s sixteen members.

`RoutineRootRotation.FootprintIsSoundAndTight` does this on the three-root
case:

- It builds the `Shape` from the fixture JSON's sizes alone (`nao`, `naux`,
  the atom count, the active space, the number of weights), with no buffer in
  sight, and checks that it equals `In::shape()` of the loaded call.
- It checks that `footprint(shape)` equals `mcscf::root_rotation_footprint`'s
  persistent, scratch and workspace numbers.
- It calls the port through the root, `RootRotationStep::run`, twice on a
  fresh `DeviceResources`. (#1246 called it directly with the global-module
  minter, which is gone.) The first call grows the workspace to the
  footprint's 118,528 bytes. The second is measured by `measure_pool_peak`.
- It holds the peak to `expect_footprint` with 22 allocations live at the
  peak: the strings' 4, the 5 keyed buffers, the integrals' 4, and the Fock's
  6 plus its 3 of scratch.

**Measured 63,616 bytes, footprint 63,616 bytes**: exact, on the gfx1200.

## Compile time

Measured on #1246's design and not re-measured: #1324 adds one small struct
and moves no code between units. Each TU compiled on its own with its bare
command from `compile_commands.json` (no sccache). This is the minimum of three runs in
the `hip-experimental` tree, with the other agent's CUDA build sharing the
box:

| | Original (`src/mcscf/root_rotation/`) | Port (`experimental/routine/`) |
|---|---|---|
| Interface | `root_rotation.cppm` 0.27 s | `root_rotation.cppm` 0.27 s + `key.cppm` 0.05 s + `base.cppm` 0.04 s + `routine.cppm` 0.05 s |
| Implementation | `root_rotation.cpp` 1.71 s | `root_rotation.cpp` 1.61 s |
| Total | 1.98 s | 2.02 s |

The difference is +0.04 s, inside run-to-run noise. The base and the key
partitions cost about 0.05 s each, and the implementation unit is no slower.
A tree-wide base would be compiled once in `core`, not once per routine.

## Call-site diff

Before (`test/gtest/mcscf_root_rotation_gpu.cpp`):

```cpp
fb::mcscf::root_rotation(*res_, *molecule_, *basis_, h_->spherical(), *b_, *c_, ncore_, ncas_,
                         nalpha_, nbeta_, roots_->data(), weights, energies, d, kappa,
                         max_rotation);
```

After (`test/experimental/routine_gpu.cpp`):

```cpp
fb::routine::root_rotation(key, {.res = res, .molecule = *molecule_, .basis = *basis_,
                                 .h = h_->spherical(), .b = *b_, .c = *c_, .ncore = ncore_,
                                 .ncas = ncas_, .nalpha = nalpha_, .nbeta = nbeta_,
                                 .roots = roots_->data(), .weights = weights_,
                                 .energies = energies_, .derivatives = *derivatives_,
                                 .kappa = kappa_->data()});
```

The call takes one argument more (`key`), and every argument now has a
name. The four `int32_t`s (`ncore`, `ncas`, `nalpha`, `nbeta`) can no longer
be swapped silently, and `max_rotation` keeps its default. On the other hand,
the designators must follow declaration order. An omitted `int32_t` designator
is zero-initialized without an error (only a reference member must be
given). And a misspelled one is diagnosed without its name, as (d) shows.

## Callees outside the key

The port's own five buffers (`pm`, `rdms`, `a_full`, `a_packed`, `orbital`)
are `KeyedBuffer`s. Its callees allocate as they always have. These are
**allocating** callees, each with its device buffers:

| Callee | Allocates | Call sites in `src/` (`grep -F`, approximate) |
|---|---|---|
| `ci::DeviceStringSet::create` (called twice) | its strings and replacements, 2 buffers per set | 40 in 20 files |
| `casscf_integrals` (density-fitted) | `h_eff`, `eri`, `ppaa`, `papa`, plus 3 nao² of per-call scratch | 20 in 14 files |
| `response::fock` (density-fitted) | its 6 members, plus 3 nao² of per-call scratch | 27 in 14 files |

These borrow the **workspace** but do not allocate: `ci::Rdm::rdm12`, the
`df_coulomb`/`df_exchange` that `casscf_integrals` and `response::fock` call,
and `response::fock`'s and `casscf_integrals`' own regions. These allocate
nothing: `ci::Rdm::create`, `response::pack_orbital_pairs`,
`response::response_gradient` (host), and `geam`/`gemv`.

**Before the key could be enforced inside `root_rotation`, 3 callees would
need keys.** "Enforced" means no `Buffers::device` reachable from it without
one. Those 3 callees have about 87 call sites in `src/`, and each caller of
those would then need a key too. The workspace sits outside the scheme by
design: it is sized once by pre-flight, and borrowing from it does not
allocate. That is the honest cost of the key: it spreads up the call graph,
like `const` did, and keying 3 callees means keying every function above
them. `src/` has about 531 named `Buffers::device` constructions in
`.cpp` files.

## Could the base check the pool's peak on every call?

wwr's pool exposes enough. `Device::pool_usage()` returns `used_current` and
`used_high`, the pool's `UsedMemCurrent` and `UsedMemHigh`, and
`Device::reset_pool_high_water()` resets the high mark. `measure_pool_peak`
is exactly those three calls. A test build of `Routine::operator()` could
reset the mark, call `invoke`, and compare `used_high − used_current(start)`
with `Self::footprint(in.shape())`. Three things stand in the way:

- **Nesting.** The reset applies to the whole pool, so an inner routine's
  check wipes its caller's high mark. The base would need a `thread_local`
  stack: on entry, fold the current `used_high` into the parent frame's
  running maximum and reset; on exit, fold its own peak back into the parent.
  This is sound for work queued on one stream, which is how every routine
  here runs.
- **Reach.** The base can reach the pool only through a convention, such as
  every `In` carrying `res`. This port's does; not every routine's would.
- **Only the soundness bar is automatic.** The tightness bar needs the number
  of allocations live at the peak (22 here), which only the test knows. On
  ROCm the soundness bar is also only within 1/8 (`kPoolReuseDivisor`). A
  per-call check would catch an under-counted footprint, not a loose one.

## If it is promoted

1. **The base, `In`, `Shape` and `check` go to `core`.** One `Routine<Self>`
   with no `Alloc` parameter, compiled once. Each routine gets an `In`, a
   `Shape` that `In::shape()` derives, `check` over `FBF_PRE` (#1226), and
   `footprint(shape)` in place of today's separate `*_footprint` argument
   lists.
2. **The key is not promoted.** `AllocKey`, `KeyedBuffer` and the `Alloc`
   split stay here as the measured record of why. "Where the key's friend
   lives" has the reasons. An allocation that a footprint misses is caught by
   the pool-peak tests, which need no key.
3. **Migrate one routine first, then by layer.** `root_rotation` is ported
   here. A first PR in `src/` would move it, with its call sites and
   `mcscf_root_rotation_gpu.cpp`, before any other routine adopts the base.

## Running it

```sh
cmake --preset hip-experimental && cmake --build build-hip-experimental -j 8
ctest --test-dir build-hip-experimental -L experimental -R 'Routine' --output-on-failure
```

`devtools/test.py pr --preset hip-experimental` says PASS with no tests run,
so it is not evidence. Neither is `test.py push`: it never configures
`experimental/`.
