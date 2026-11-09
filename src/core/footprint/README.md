# Footprint

The vocabulary every layer reports its memory in: how many bytes a routine or a
pipeline step holds, of which memory kind, and for how long. A library routine
exports its own footprint next to the code that allocates; a step composes its
routines' footprints with `then` and `drop`; the pipeline folds the steps'
(`pipeline/footprint`).

## Module

`free_breakfast.core.footprint` — target `free_breakfast::core::footprint`,
also re-exported by `free_breakfast.core`.

It imports `core.types` and `core.wwr`, the latter only for wwr's
`MemoryKind` (`wwr::extension::MemoryKind`). It lives in `core`, not in
`pipeline`, because nothing below `pipeline` may import the pipeline, and the
library routines export footprints too. Templates and `constexpr` functions
only, so there is no implementation unit.

| Export | What |
|---|---|
| `Footprint<K>` | The bytes of memory kind `K`: `persistent`, `scratch`, `workspace`, and `peak()`, their sum |
| `MemoryFootprint` | What a routine or step reports: `Footprint<Device> device` and `Footprint<Pinned> pinned` |
| `then(a, b)` | `a` runs and its output is kept, then `b` runs |
| `drop(f)` | What `f` returned is freed before the enclosing routine returns |

## The three lifetimes

- **`persistent`**: what the routine or step returns and keeps. For a step,
  its product, held to the pipeline's end — on the run and the load path
  alike: a `Load`ed product is as resident as a computed one.
- **`scratch`**: what it allocates and frees before it returns. For an
  adaptive routine (a tile, batch or width chosen from free memory), the
  **least** it runs with. At run time it sizes its adaptive part from the
  memory budget (#1047), less its own allocations still to come and the
  fragmentation headroom (#1048): as much as is available, up to its natural
  full size. Pre-flight checks the least, so a plan is refused only when even
  that cannot fit.
- **`workspace`**: what it borrows from the `DeviceResources` workspaces
  (device and pinned). The workspace is held for the whole run, one block per
  device, sized to the plan's maximum. **Fixed-size uses only**: what goes in
  it may depend on the dimensions and options, never on a memory budget. An
  adaptive routine's adaptive part is scratch.

`peak()` is `persistent + scratch + workspace`: the most held at once.

**The kind is in the type.** There is no generic `Bytes`: the name is too
generic, and reads as `std::byte`. `Footprint<Device>` and `Footprint<Pinned>`
are distinct types, so `then` and `drop` are templates over the kind and a
device footprint cannot be summed into a pinned one by mistake. `Host` and
`Unified` get no member of `MemoryFootprint`: pageable host memory is not
modelled, and unified memory is unused.

## Composition is exact

The footprint tests hold every footprint *tight* against the measured pool
peak, so composing two routines must not overcount. Per kind, with
peak = p + s + w:

| | persistent | scratch | workspace |
|---|---|---|---|
| `then(a, b)` | `a.p + b.p` | `max(a.s − b.p, b.s, 0)` | `max(a.w, b.w)` |
| `drop(f)` | `0` | `f.p + f.s` | `f.w` |

`then`'s peak is `max(a.p + a.s, a.p + b.p + b.s) + max(a.w, b.w)`: while `a`
runs, its own product and scratch; while `b` runs, both products and `b`'s
scratch. The part of `a`'s scratch above `b`'s product is the residue that
keeps the first term, clamped at zero when `b`'s product is larger. `drop`
leaves the peak unchanged and only moves the product into scratch. `then` is
associative, so a chain reads left to right. `MemoryFootprint`'s overloads
apply both to `device` and `pinned` alike.

**A worked example.** A routine forms B (100 bytes, built with 40 of scratch),
then builds J and K from it (persistent 30, scratch 20), and frees B before it
returns J and K:

```cpp
using wwr::extension::MemoryKind;
const Footprint<MemoryKind::Device> b{.persistent = 100, .scratch = 40};
const Footprint<MemoryKind::Device> jk{.persistent = 30, .scratch = 20};
const auto routine = then(drop(b), jk);
// drop(b)   = {p 0,  s 140}           peak 140
// then(...) = {p 30, s max(140 - 30, 20) = 110}
// peak 140: B's build, before J and K exist.
```

A step's `footprint()` then reads like the routine it describes, e.g.
`then(drop(b_mo_fp), drop(jk_fp))`.

## The library pattern

A library routine exports its own footprint **next to the code that
allocates**, taking its own arguments:

```cpp
export namespace free_breakfast::mcscf {
[[nodiscard]] MemoryFootprint solve_footprint(size_t ncas, size_t nelecas, size_t nroots,
                                              const Options &options);
}
```

It never takes the pipeline's dimensions, which a library cannot see; the
step unpacks those into the routine's arguments. Inside, the routine and its
footprint share element counts however the author likes. A private `sizes`
struct, used by both the allocation and the footprint, is the suggested
spelling, not a contract. The lifetime tags are mechanical at this level: the
return value is `persistent`, what is freed before return is `scratch`, and
what is borrowed is `workspace`.
