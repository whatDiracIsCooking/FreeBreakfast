# Routine spike: check-before-invoke and an allocation key (#1246)

Does a routine shape whose `check` the compiler enforces, and whose permission
to allocate is checked at compile time, hold up on one real routine, under
named modules and on the GPU? This spike measures that on
`mcscf::root_rotation`. Nothing under `src/` changes.

**Verdict: rework.** The base, the `In` struct and the check split work
across module boundaries. The port matches the original bit for bit, and its
`footprint(in)` is the pool's peak to the byte. The key blocks what it should,
but **it cannot be friended across modules**: the root that mints it must
either live in the key's own module or be a global-module class that any TU
can define. It also cannot be enforced inside `root_rotation` until three
allocating callees take it, and those have about 87 call sites in `src/`.
`footprint(in)` over an `In` of live buffers cannot run before those buffers
exist, which pre-flight needs. Details and the rework are under each heading.

## What is here

| File | What |
|------|------|
| `key.cppm` (`:key`) | `Alloc { No, Yes }`; `AllocKey` (private default constructor, deleted copy, user-provided `constexpr ~AllocKey() {}`, friends `RootRotationStep` and the test's `RoutineKeyMinter`); `KeyedBuffer<T>`, a `Buffers::device<T>` whose constructor takes `const AllocKey &` |
| `base.cppm` (`:base`) | `Routine<Self, Alloc>`: a static `operator()` over one `const typename S::In &` (`S = Self`) that calls `Self::check(in)` and then `Self::invoke`, taking and passing on the key for `Alloc::Yes` and taking none for `Alloc::No` |
| `root_rotation.cppm` (`:root_rotation`) | `RootRotationRoutine : Routine<…, Alloc::Yes>` with its `In`, `check`, `footprint` and `invoke`; the object `root_rotation`; `RootRotationStep`, the root that mints the key and calls the port as a step would |
| `root_rotation.cpp` | The implementation unit: `check`, `footprint`, `invoke` and `RootRotationStep::run`, plus `FBF_ROUTINE_PRE`, #1226's `FBF_PRE` (not landed yet) |
| `routine.cppm` | The primary interface, re-exporting the three partitions |

Tests (`test/experimental/`): `routine_gpu.cpp` (`RoutineRootRotation`, two
GPU cases) and `routine_reject_{alloc_up,key_buffer,bit_cast,misspelled_field}.cpp`,
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
| Statics defined in an implementation unit | yes | `check`, `footprint`, `invoke` and `RootRotationStep::run` are declared in `:root_rotation` and defined in `root_rotation.cpp` (`module free_breakfast.experimental.routine;`), and called from other TUs through `Routine::operator()`, which is instantiated in the importer and links against them |
| A friend declared in one partition, defined in another | yes, within one module | `RootRotationStep` is forward-declared (exported) in `:key` before `AllocKey` names it as a friend, and defined in `:root_rotation` |
| **A friend defined in another module** | **no** | `friend struct Root;` in module `a` and `struct Root {…}` in module `b`: clang 23.1.6 says `error: declaration of 'Root' in module b follows declaration in module a`. A friend declaration attaches the name to the declaring module, and only that module can define it |
| A global-module friend | yes, but forgeable | `extern "C++" { namespace free_breakfast::test { struct RoutineKeyMinter; } }` in `:key` attaches it to the global module, so `routine_gpu.cpp` defines it and mints keys. Any other TU could define it too |
| `requires` | not used | `Routine` needs no constraint: a missing `In`, `check` or `invoke` is an error at the call that instantiates `operator()`. A `requires` clause would only move the diagnostic |

So the **key's friends must be in the key's module**. In `src/` the key
would live in `core` (`KeyedBuffer` wraps `Buffers::device`), and the root,
the pipeline's planner, is in `pipeline`, which imports `core`. The planner
cannot be `AllocKey`'s friend unless it is a global-module class. That makes
it as forgeable as `RoutineKeyMinter` here: anyone who writes
`struct free_breakfast::pipeline::Planner { … }` in a TU can mint a key. The
key is still useful as a discipline that `grep` can audit, but it is not a
boundary the compiler holds.

Separately: the container's compiler is now **clang 23.1.6**, not the 20.1.8
the issue's sketches used. The `bit_cast` hole needs the user-provided
destructor on 23 as well (see (c)).

## The compiler rejects what it should: (a)–(d)

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
experimental/routine/key.cppm:45:3: note: implicitly declared private here
```

**(c) The key forged with `std::bit_cast<AllocKey>(char{})`** (`bit_cast`):

```
test/experimental/routine_reject_bit_cast.cpp:18:37: error: no matching function for call to 'bit_cast'
.../c++/v1/__bit/bit_cast.h:34:55: note: candidate template ignored: constraints not satisfied [with _ToType = fb::routine::AllocKey, _FromType = char]
.../c++/v1/__bit/bit_cast.h:32:52: note: because 'is_trivially_copyable_v<free_breakfast::routine::AllocKey>' evaluated to false
.../c++/v1/__bit/bit_cast.h:32:52: note: 'free_breakfast::routine::AllocKey' is not trivially copyable
.../c++/v1/__bit/bit_cast.h:32:52: note: because it has a user-provided destructor
experimental/routine/key.cppm:44:7: note: 'AllocKey' defined here
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

Every difference is exactly zero. The calls are the same in the same order,
and `ci::Rdm::rdm12`'s reductions happened to come out in the same order.

`check` turns the original's `InvalidConfig` refusals into
`FBF_ROUTINE_PRE` aborts (#1226: a broken shape is the caller's bug). The two
degeneracy refusals are numerical outcomes that a caller branches on, so they
stay in `invoke` as `Numerical`. The original's `Refusals` case therefore has
no direct port: an abort needs a death test, which this spike does not add.

## The footprint is tight

`RoutineRootRotation.FootprintIsSoundAndTight` does three things on the
three-root case:

- It checks that `footprint(in)` equals `mcscf::root_rotation_footprint`'s
  persistent, scratch and workspace numbers.
- It calls the port directly with the fixture's key, twice on a fresh
  `DeviceResources`. The first call grows the workspace to the footprint's
  118,528 bytes. The second is measured by `measure_pool_peak`.
- It holds the peak to `expect_footprint` with 22 allocations live at the
  peak: the strings' 4, the 5 keyed buffers, the integrals' 4, and the Fock's
  6 plus its 3 of scratch.

**Measured 63,616 bytes, footprint 63,616 bytes**: exact, on the gfx1200.

`footprint(in)` reads naux from `B`'s length and nroots from `weights`. That
is why it can take the same `In` as `invoke`, and also why it can run only
once `B`, the roots and `derivatives` exist. Pre-flight sizes a run before
any of those are allocated, so in `src/` a footprint over `In` would need a
shape-only struct beside it, which brings back the separate argument list in
another form.

## Compile time

Each TU compiled on its own with its bare command from
`compile_commands.json` (no sccache). This is the minimum of three runs in
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
with `Self::footprint(in)`. Three things stand in the way:

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

## The rework

1. **Key placement.** No cross-module friend can be defined in another
   module, so decide openly what the key is. Either it is a discipline that
   `grep` can audit, minted by a global-module `Planner` (or a `protected
   static mint()` on a base class only roots derive from), or it is dropped.
   It cannot be a compiler-held boundary between `core` and `pipeline`.
2. **Footprint over shapes.** Give each routine a shape-only struct (counts,
   bases) that both `In` and pre-flight can produce, and have
   `footprint(shape)` take it, so pre-flight keeps working before buffers
   exist.
3. **Keep the base and the `In`/`check` split as they are.** They cost
   nothing measurable, match the original exactly, and the
   `typename S::In` spelling works across modules. Adopt `FBF_PRE` (#1226)
   for `check`.
4. **Key the leaves first.** If the key stays, start where nothing below
   allocates (`DeviceStringSet::create`), then `casscf_integrals` and
   `response::fock`. `root_rotation` can be enforced only after all three.

## Running it

```sh
cmake --preset hip-experimental && cmake --build build-hip-experimental -j 8
ctest --test-dir build-hip-experimental -L experimental -R 'Routine' --output-on-failure
```

`devtools/test.py pr --preset hip-experimental` says PASS with no tests run,
so it is not evidence. Neither is `test.py push`: it never configures
`experimental/`.
