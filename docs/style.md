# Coding style

This page covers what a tool cannot check. Where a tool does check something,
this page links to its config instead of repeating it. If the two ever
disagree, the config is what gets enforced.

## What is enforced, and where

| Rule | Tool | Config | Runs at |
| --- | --- | --- | --- |
| C++ layout (100 columns, 2-space indent, LLVM base) | clang-format 20.1.8 | [`.clang-format`](../.clang-format) | commit hook, CI `lint` |
| Python layout (88 columns) | `ruff format` | `pyproject.toml` `[tool.ruff]` | commit hook, CI `lint` |
| Python lint | `ruff check` | `pyproject.toml` `[tool.ruff.lint]` | commit hook, CI `lint` |
| CMake lint (80 columns) | `cmake-lint` | [`.cmake-format.yaml`](../.cmake-format.yaml) | commit hook, CI `lint` |
| Fixed-width integer types only (see [Integer types](#integer-types)) | `devtools/check-int-types.py` | the script's `EXEMPT_FILES` | commit hook, CI `lint` |
| C++ naming, bug-prone patterns | clang-tidy | [`.clang-tidy`](../.clang-tidy) | **advisory** — `devtools/cpp-tier.sh --tidy` |
| Whitespace, line endings, final newline | editor + pre-commit | [`.editorconfig`](../.editorconfig) | commit hook |

The two formatters only ever run in check mode from a hook or in CI, and fail
when a file has drifted. To fix the drift, run them yourself:

```sh
devtools/format.sh              # format every tracked file in place
devtools/format.sh path/to/file # just these files
devtools/format.sh --check      # what the hook and CI run
```

No hook reformats a file. (The `ruff-check` hook does apply ruff's lint
autofixes with `--fix`; formatting is never applied.) A formatter running as a
hook rewrites whole files
without you seeing it, and the real change gets lost among the restyling.
Running it yourself means you see the diff before you commit it. The tool
versions come from `uv.lock`, so the host and the image produce identical
output. Bumping clang-format can reformat the whole tree, so do it as a commit
of its own.

clang-tidy is advisory because its support for C++23 named modules still
produces bogus diagnostics on `.cppm` files. Its naming rules are still the
naming rules (see below). Don't restructure modules just to silence it.

### Escaping the formatter

`// clang-format off` / `// clang-format on` is allowed in one case only: a
table whose column alignment carries meaning, such as the specializations in
`src/core/types/vec.h`. Don't use it to protect a hand-wrapped line you happen
to prefer.

## Naming

| Kind | Style | Example |
| --- | --- | --- |
| Namespace | `lower_case` | `free_breakfast::log` |
| Class, struct, alias, concept | `CamelCase` | `BasisSet`, `Result<T>` |
| Function, variable, parameter | `lower_case` | `element_count`, `ao_offset_cart` |
| Private data member | `lower_case_` | `outer_`, `aos_` |
| Constant | `kCamelCase` | `kMaxAngularMomentum`, `kPi<T>` |
| Enum | `enum class`, `CamelCase` | `ErrorKind`, `LogLevel` |
| Enumerator | `CamelCase`, no `k` | `ErrorKind::InvalidConfig`, `LogLevel::Warn`, `Element::Fe` |
| Macro | `FBF_UPPER_CASE` | `FBF_TRY`, `FBF_REQUIRE_GPU` |
| Module | dotted, mirroring `src/` | `free_breakfast.io.npy` |
| Module partition | `:lower_case` | `:basis_set`, `:format` |

**Constants** means every `constexpr` variable (at namespace scope, as a class
static, as a local, or as a variable template) plus every `const` variable at
namespace scope. If it is fixed at compile time, it gets the `k` prefix. The
rule is in `.clang-tidy` as `ConstexprVariable*` and `GlobalConstant*`.

**Enumerators** are the exception: no `k`. Every enum is an `enum class`, so
the enum's name already qualifies each enumerator and a prefix would add
nothing. Without the prefix, element symbols read as they should
(`Element::Fe`, not `Element::kFe`).

**Macros** all start with `FBF_`, wherever they are defined: `src/`
(`FBF_TRY`, `FBF_HOST_DEVICE`), `test/gtest/` (`FBF_REQUIRE_GPU`) or anywhere
else. The prefix keeps a macro clear of any name that an including file, or a
third-party header it pulls in, might also define, such as `TRY` or `MUST`.
A bare `UPPER_CASE` macro gets no such protection.

The prefix is for preprocessor macros only. CMake options and cache variables
(`FREE_BREAKFAST_GPU_BACKEND`) and environment variables
(`FREE_BREAKFAST_FAST_TESTS`) keep the full `FREE_BREAKFAST_` prefix. They live
in a global namespace shared with every other project, where the short prefix
would be ambiguous.

## Integer types

**Every integer states its width.** Use `int8_t` … `int64_t`, `uint8_t` …
`uint64_t` or `size_t`. Never use `int`, `unsigned`, `short`, `long` or
`long long`. Choose the width deliberately: `int32_t` where 32 bits is enough
and intended (angular momenta, charges, kernel-side counters), `int64_t` or
`size_t` for anything that can grow with the problem. On a GPU, element counts
of large buffers and pair indices pass 2³¹ sooner than they look like they will.
Container indices stay `size_t`.

On every target this tree builds for, `int32_t` *is* `int` and `uint32_t` *is*
`unsigned int`, so the rule changes no ABI, overload or template match. An API
declared with `int`, such as `wwrGetDeviceCount(int *)` or wwr's `int_type`
concept (exactly `int` or `int64_t`), accepts the fixed-width spelling as is.
What the rule buys is that each declaration says how wide it means to be.
`long` is the one that actually differs across platforms (64 bits on Linux, 32
on Windows), which is why it is banned too.

**Inside `namespace free_breakfast`, write them unqualified**: `int32_t`,
`uint64_t`, `size_t`. They are `free_breakfast::int32_t` and so on, declared by
`src/core/types/integral_types.h` as the very same types as `std::int32_t`. To
make them visible:

- A module unit adds `import free_breakfast.core.types;`, and its library
  links `free_breakfast::core::types`.
- A header shared with device code includes `"integral_types.h"` (not
  `<cstdint>`), and its headers target links
  `free_breakfast::core::types_headers`, as `atom::headers` and
  `basis::headers` do.

Outside the namespace, as in `test/gtest/` where code goes through an `fb`
alias, write `std::int32_t`.

Allowed:

- `unsigned char` and `signed char`: character and raw-byte types, and what
  `<cctype>` requires its argument to be cast to.
- `long double`, which is a floating-point type.
- `int main(`.
- `src/core/types/vec.h`, whose job is to map every fundamental scalar type,
  including `int`, `long` and `long long` as three distinct types.
- A line marked `// int-type-ok: <reason>`, or the line after a comment-only
  line carrying that marker, where code must name a distinct fundamental type
  that no fixed-width alias spells (`Vec<long long, 1>`: `int64_t` is `long`
  on LP64).

`devtools/check-int-types.py` enforces this. It ignores comments and string
literals, so prose and test data can still say "int".

## Vector types

**Spell a GPU vector type `Vec<T, N>`**, from `src/core/types/vec.h`: a
point in space is `Vec<double, 3>`, not `wwr::double3`, `double3` or
`make_double3`. Build one by brace initialisation or with `make_vec<T>(…)`.
Both are in `free_breakfast`, so code inside the namespace writes them
unqualified. To make them visible:

- A module unit adds `import free_breakfast.core.types;`, and its library
  links `free_breakfast::core::types`. Don't import `wwr.vector_types` or
  link `wwr::vector_types` for this.
- A header shared with device code, or a `.cu`, includes `"vec.h"`, and its
  target links `free_breakfast::core::types_headers`.

`Vec` names the same builtin types as wwr's aliases, so there is no layout or
ABI change. The reasons for the rule are that the scalar and width are
parameters, so generic code can write `Vec<T, 3>`, and that `Vec` covers the
64-bit 4-vectors (`double4`, `long4`, …) that wwr leaves out. The one place
wwr's aliases appear on purpose is `test/gtest/core_vocabulary_host.cpp`,
which checks that `Vec` maps onto them.

## Stamps

A **stamp** is a geometry-free definition, a type named `…Stamp`: a
`ShellStamp` is a `Shell` without its center, a `BasisStamp` is a basis-set
file's per-element table of them. Putting one somewhere follows one pattern:

- **The verb is `place`, a free function overloaded per stamp**, with the stamp
  first and where it goes second: `place(shell_stamp, center)`,
  `place(basis_stamp, molecule)`. Not a member such as `stamp.at(center)`. A
  free overload lets every level of the hierarchy read the same, and `.at()`
  already means a bounds-checked index to every C++ reader.
- **`place` returns what it builds, or writes into the target it is given.**
  Placing a shell or a basis makes a new object, so it is returned (wrapped in
  `Result` when placement can fail). Placing into a target that several stamps
  share writes into it instead: `place(atomic_density_stamp, offset, n, density)`
  in `src/sad/sad.cpp` writes one atom's block of the molecular density. The
  target comes last and is a `std::span`, so `place` cannot resize it.
- **`place` never computes.** It copies a stamp to where it goes. Work that
  solves or allocates for real, such as computing the stamps in the first
  place, gets a name of its own (`sad_density`), so that seeing `place` says
  the call is cheap.

## Module-private code

**There are no anonymous namespaces and no `detail` namespaces**, with three
exceptions where an anonymous namespace is allowed: `.cu` files,
`test/gtest/*.cpp` and `src/main.cpp`. None of those is a module unit, so the alternatives
below are not available there.

In a named module, `export` already decides what is visible. Anything not
exported cannot be named by an importer, so an anonymous namespace adds
nothing, and `detail` is only a naming convention that nothing enforces. Both
also hide duplication. Before this rule existed, `double_factorial_odd` was
defined twice in the basis module, once in `:harmonics`' `detail` namespace and
once in an anonymous namespace in `:basis_set`, and neither copy could see the
other. (It now lives once, in `free_breakfast.core.primitives`.)

Code that is private to a module goes in one of three places. Which one depends
on who needs it:

| Who uses it | Put it in | Example |
| --- | --- | --- |
| One implementation unit (`.cpp`) | An unexported declaration at namespace scope in that file | `step_table` in `src/pipeline/config/config.cpp` |
| Several implementation units of one module | An **internal partition**, `module M:name;`, imported by those units | `src/io/npy/npy_format.cppm` (`kMagic`, used by both the reader and the writer) |
| An exported `inline` function or template — the job `detail` used to do | Unexported in the interface partition that uses it, or, when several partitions share it, a **private interface partition** `export module M:name;` that exports nothing | None in the tree today; `src/basis/numeric.cppm` was one until its helpers moved to `free_breakfast.core.primitives` |

All partitions, of either kind, go under `PARTITIONS` in
`free_breakfast_add_cxx_module_library`.

**Why helpers for inline functions and templates can't go in an internal
partition.** An exported template is instantiated in the importer's translation
unit, and an exported inline function may be compiled there too. Whatever they
call has to be *reachable* from that unit. The standard guarantees reachability
only for interface units, and leaves internal partitions unspecified. A private
interface partition is reachable but exports nothing, so its contents stay
unnameable. The primary interface still has to `export import :name;` it,
because every interface partition must be re-exported. That line exposes
nothing, so add a comment saying so.

**An unexported name is shared across the whole module.** It has module linkage,
not internal linkage, so two units of the same module that each define a helper
called `parse` are an ODR violation. The linker reports this for ordinary
functions, but **nothing reports it** for `inline` functions, templates or
constants. To avoid it:

- Give helpers specific names (`capture_slot`, not `slot`).
- If two units need the same helper, move it into a partition rather than
  writing a second copy. Ending up with one shared copy is what this rule is
  for.

### Headers shared with device code

A `.h` that a `.cu` includes (such as `vec.h`, `constants.h` or
`angular_momentum.h`) is included as text, so no module boundary applies to it
and **everything it declares is public**. Keep helpers out of these headers.
Either spell out what the helper would have generated (`vec.h` lists all 48
`vector_type` specializations), or make the helper a deliberate, documented
part of the API.

### `.cu` files

A `.cu` is never a module unit. Use an anonymous namespace there for file-local
kernels and helpers, because it is the only way to keep them out of the
device-link namespace.

## Errors

A failure is handled in one of three ways. Pick the tier by what a caller can
do about the failure, not by where it was detected. The mechanics are in
[`src/core/error_handling/README.md`](../src/core/error_handling/README.md) and
[`src/core/diagnostics/README.md`](../src/core/diagnostics/README.md).

| Tier | Use it when | Spelling |
| --- | --- | --- |
| Value | The caller can act on it: bad input, a missing file, a solver that did not converge | Return `Result<T>` or `Status`. Build the error with `err_*` and pass it up with `FBF_TRY` |
| Abort | Nothing in the process can recover, and passing a `Status` up would only move the abort to the caller | `FBF_GPU_MUST(call)`, or `check_or_abort(…)` |
| Wrapper | A wwr RAII wrapper is created or destroyed | Nothing to write. `CreatePolicy` aborts on a failed create or alloc, and `DestroyPolicy` logs a failed destroy or free |

- **Log a value-tier error once,** at the boundary that turns it into an exit
  code with `to_exit_code`. Don't log it at each hop on the way up: the `err_*`
  helpers only construct the error, and `FBF_TRY` passes it along unchanged,
  with its original `source_location`.
- **There is no recoverable wrapper.** A wrapper that exists is valid, so don't
  check one after constructing it. If a site ever needs to recover, give that
  one site a reporting policy rather than adding a tree-wide mode.
- **Choose the `ErrorKind` by what the caller should do next**, not by which
  stage failed (a parse error is `IO`). The exit codes are pinned. A new kind
  adds a new code and never renumbers the existing ones.

### Exceptions

This code does not throw. An exception from a third-party library (toml++,
nlohmann_json) is caught where that library is called and turned into a
`Result`, as `src/io/toml/toml_reader.cpp` does. A `noexcept` function that
calls something that can throw catches with `catch (...)` and gives up the
lesser thing, such as a log line, rather than letting `std::terminate` run.
Say why in a comment there (see `LogPolicy` in
`src/core/wwr/policy.cppm`).

## Comments

- Use `///` Doxygen comments with `@brief` on everything exported, and `//` for
  everything else. Don't use `/** … */`. "Exported" includes the public members
  of an exported type: data members, member functions, member aliases and
  constructors (see `MoleculeStep` in `src/pipeline/molecule/molecule.cppm`). It also
  includes enumerators, which take a trailing `///<` (see `LogLevel`). The one
  exception is special members that are only `= default` or `= delete`.
- Explain *why*: the constraint, the failure the code avoids, or the
  alternative that was rejected. Don't restate what the code says.
- When a comment no longer matches the code, fix the comment in the same
  change.
- Suppress a clang-tidy diagnostic on the line it fires on, naming the check
  and giving a reason:
  `// NOLINT(google-explicit-constructor): converts at the call site`. Never
  write a bare `NOLINT` or a `NOLINTBEGIN` block.
