# rust/CLAUDE.md

The Rust rebuild. `README.md` has the crate list and "Build and test";
`docs/design.md` indexes the design docs. This file points into them.

**The `rust-rebuild` agent** (`.claude/agents/rust-rebuild.md` at the
repository root) is this file at length: what to read for which task, the
design questions, the checks to run and what to report. Delegate work under
`rust/` to it. `doc_check/tests/section_citations.rs` checks its section
citations, and this file's, so a renamed heading breaks them.

**Follow the docs over the surrounding code.** Where a neighbour breaks a
rule below, fix the counter-example (or file an issue) rather than copy it.

**Which check is which** (`docs/testing.md`, "Contracts are part of the spec"):

- A caller's precondition that only correctness rests on (a basis has shells,
  `la <= lb`, a predecessor is in the plan) is a `#[debug_requires]`. rustdoc
  renders it as the item's `§ Contracts` section, so never restate it in the
  doc comment; `fb-doc-check` fails a restatement.
- A check that memory safety rests on (the length an `unsafe` call trusts,
  an `n²` that must fit a 32-bit index) is an always-on `assert!`, named in
  the item's `# Panics`. clippy's `missing_panics_doc` is on.
- The outcome of a GPU call that can fail is a status or a `Result`, never
  a contract or a panic.
- `debug_assert!` is only for an invariant no caller can break, with an
  `// INVARIANT:` comment above it saying why.

**`unsafe`** lives only in the crates `docs/programming_model.md`
("Capabilities are enforced by lints") lists; every other crate root
forbids it. Each block holds one operation under a `// SAFETY:` comment
(clippy's `undocumented_unsafe_blocks`, `multiple_unsafe_ops_per_block`).

**Use the safe wrappers, not `wwr-sys`:** copies through
`fb_core_arena::memcpy_*`, BLAS through `fb_core_wwr::blas`, a device
zero through `fb_core_wwr::memset::zero`, a kernel launch through its
binding crate's safe API (`fb_recipes_sys::inverse_sqrt`, …), and a test's
device buffer through `fb_core_wwr::runtime::tests::DeviceBytes`.

**A new rule joins the register.** `docs/rules.md` lists every rule these
docs state and what checks it; a rule added to a doc gets a row there with
its check, or `**unenforced**` if nothing can yet. `fb-doc-check` fails a
row that names a check that does not exist.

**Gates:** `devtools/test.py pr --only cargo` from the repo root, and the
commit hook's `cargo fmt --check` and clippy `-D warnings`. The hook's
clippy reaches only the link-free crates; tier 4 of the gates lints the
crates that link, after building the library.
