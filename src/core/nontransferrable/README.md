# Nontransferrable

Base classes for restricting copy and move semantics via private inheritance.

## Module

`free_breakfast.core.nontransferrable` — target `free_breakfast::core::nontransferrable`,
also re-exported by `free_breakfast.core`.

Pure `import std;`. WarpWraps has its own `wwr::extension::NonCopyable`, but
not the other two, and a language-level mixin should not pull in the GPU
extension layer — so all three live here.

## Usage

Inherit **privately** to delete unwanted special members:

```cpp
import free_breakfast.core.nontransferrable;

// RAII handle: movable but not copyable
class MyHandle : private free_breakfast::NonCopyable {
public:
  MyHandle() = default;
};

// Singleton or pinned resource: neither copyable nor movable
class Registry : private free_breakfast::NonTransferable {
public:
  Registry() = default;
};
```

Protected destructors prevent polymorphic deletion through a base pointer.

## Files

| File | Role |
|------|------|
| `nontransferrable.cppm` | Module interface with `NonCopyable`, `NonMovable`, `NonTransferable` |
| `CMakeLists.txt` | Build definition |
