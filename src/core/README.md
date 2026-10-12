# `free_breakfast.core`

The starting set — one module library, here so a fresh clone has something that
exercises the full chain rather than a tree that configures and builds nothing.

| File | What it is |
| ---- | ---------- |
| `core.cppm` | The primary interface. Re-exports `:version` and the two WarpWraps modules. |
| `version.cppm` | A partition holding the project version and `version_string()`. |

```cpp
import free_breakfast.core;

auto v = free_breakfast::core::version_string();   // "0.1.0"
```

The two `wwr` modules it re-exports (`wwr.extension.common`,
`wwr.extension.memory_buffer`) come along with that import, so a consumer gets
the buffer and error-policy types without naming the submodule. Those types take
error-policy template arguments and a device handle rather than a bare size —
`deps/WarpWraps/example/buffer_suite/main.cpp` is the worked usage.

Link it as `free_breakfast::core`.

**What it proves when it builds.** A green build of this target means clang
resolved `import std` against libc++'s module manifest, `clang-scan-deps`
ordered a partition before its primary interface, and the WarpWraps submodule
configured and built for the selected backend. That is most of what can go
wrong in the toolchain, which is why this stays in the tree until real modules
replace it.
