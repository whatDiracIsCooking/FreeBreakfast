# Atom

The chemical element (`Element`) and a single positioned atom (`Atom`), defined once and offered to both CUDA / HIP `#include` and C++ `import`.

## Module

`free_breakfast.atom`

## Design

`element.h` and `atom.h` hold the single definitions. `atom.cppm` includes them in its global module fragment and re-exports the names, so `import`ers and `#include`rs share one entity rather than two:

- **CUDA / HIP / plain headers** — `#include <atom.h>` (or `<element.h>`), reached through the `free_breakfast::atom::headers` include path.
- **C++ modules** — `import free_breakfast.atom;`.

The re-export in `atom.cppm` is written with `using free_breakfast::Element;` / `using free_breakfast::Atom;` *inside* `namespace free_breakfast` (not `export using` at global scope): a name brought in from the global module fragment is only usable by importers when the re-exporting using-declaration sits in the namespace that owns the name.

`Element` is a strong enum over `std::uint8_t` whose enumerator value is exactly the nuclear charge Z, covering `H` (1) through `Og` (118). `Sentinel` (0) is the default "no element" value. The headers include `<cstdint>` rather than relying on `import std`, so they work unchanged in device code.

The symbols are host-only, so they live in `atom.cppm` rather than the headers: `element_symbol(Element)` gives `"Cl"` (empty for `Sentinel`), and `element_from_symbol(std::string_view)` maps a symbol back, case-insensitively (`"CL"`, `"cl"`), returning `Sentinel` for none. The XYZ reader and the Molden writer (`src/io`) both read this one table.

`Atom` is a plain aggregate — a public `position` (`Vec<double, 3>`, i.e. the backend's `double3`, in bohr) plus an `Element`, with no invariant — so callers brace-initialise directly. A default-constructed `Atom` sits at the origin carrying `Element::Sentinel`.

## Usage

```cpp
import free_breakfast.atom;

// Carbon at the origin; coordinates are in bohr.
free_breakfast::Atom c{{0.0, 0.0, 0.0}, free_breakfast::Element::C};

// The enumerator value is Z directly.
auto z = static_cast<std::uint8_t>(c.element);   // 6
```

## Tests

`test/gtest/atom_host.cpp`, which also includes `atom.h` beside the import to pin that the two paths name one entity.

## Files

| File | Role |
|------|------|
| `element.h` | Single definition of the `Element` strong enum (Z=0..118) |
| `atom.h` | Single definition of the `Atom` aggregate; includes `element.h` and `vec.h` |
| `atom.cppm` | Module interface: includes the headers, re-exports `Element` / `Atom`, and defines the symbol lookups |
| `CMakeLists.txt` | The `free_breakfast.atom.headers` interface library and the `free_breakfast.atom` module library |
