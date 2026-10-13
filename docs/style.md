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
| C++ naming, bug-prone patterns | clang-tidy | [`.clang-tidy`](../.clang-tidy) | **advisory** — `devtools/cpp-tier.sh --tidy` |
| Whitespace, line endings, final newline | editor + pre-commit | [`.editorconfig`](../.editorconfig) | commit hook |

The two formatters only ever run in check mode from a hook or in CI, and fail
when a file has drifted. To fix the drift, run them yourself:

```sh
devtools/format.sh              # format every tracked file in place
devtools/format.sh path/to/file # just these files
devtools/format.sh --check      # what the hook and CI run
```

No hook rewrites a file. A formatter running as a hook rewrites whole files
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
| Function, variable, parameter | `lower_case` | `element_count`, `ao_offset` |
| Private data member | `lower_case_` | `outer_`, `aos_` |
| Constant | `kCamelCase` | `kMaxAngularMomentum`, `kPi<T>` |
| Enum | `enum class`, `CamelCase` | `ErrorKind`, `LogLevel` |
| Enumerator | `CamelCase`, no `k` | `ErrorKind::InvalidConfig`, `LogLevel::Warn`, `Element::Fe` |
| Macro | `UPPER_CASE`, project-prefixed | `FREE_BREAKFAST_REQUIRE_GPU` |
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

## Module-private code

**There are no anonymous namespaces and no `detail` namespaces**, with two
exceptions where an anonymous namespace is allowed: `.cu` files and
`test/gtest/*.cpp`. Neither of those is a module unit, so the alternatives
below are not available there.

In a named module, `export` already decides what is visible. Anything not
exported cannot be named by an importer, so an anonymous namespace adds
nothing, and `detail` is only a naming convention that nothing enforces. Both
also hide duplication. Before this rule existed, `double_factorial_odd` was
defined twice in the basis module, once in `:harmonics`' `detail` namespace and
once in an anonymous namespace in `:basis_set`, and neither copy could see the
other.

Code that is private to a module goes in one of three places. Which one depends
on who needs it:

| Who uses it | Put it in | Example |
| --- | --- | --- |
| One implementation unit (`.cpp`) | An unexported declaration at namespace scope in that file | `table_int` in `src/io/config/config_reader.cpp` |
| Several implementation units of one module | An **internal partition**, `module M:name;`, imported by those units | `src/io/npy/npy_format.cppm` (`kMagic`, used by both the reader and the writer) |
| An exported `inline` function or template — the job `detail` used to do | Unexported in the interface partition that uses it, or, when several partitions share it, a **private interface partition** `export module M:name;` that exports nothing | `CaptureScope` in `:gpu_policy`; `src/basis/numeric.cppm` |

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
nothing, so add a comment saying so (see `src/basis/basis.cppm`).

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

## Comments

- Use `///` Doxygen comments with `@brief` on everything exported, and `//` for
  everything else.
- Explain *why*: the constraint, the failure the code avoids, or the
  alternative that was rejected. Don't restate what the code says.
- When a comment no longer matches the code, fix the comment in the same
  change.
