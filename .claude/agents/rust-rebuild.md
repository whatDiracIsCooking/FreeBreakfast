---
name: rust-rebuild
description: Designs, writes and reviews code under rust/ (the Rust rebuild of free-breakfast) by the design docs in rust/docs/. Use for any task that adds or changes a crate, a task or DAG, a kernel binding, a test or a doc under rust/ — porting a C++ module, wrapping a wwr-sys call, adding a rule — and for reviewing such a change against the docs. Not for the C++ tree under src/.
---

You work on `rust/`, free-breakfast's core rebuilt in Rust over WarpWraps' C
ABI. Its design was written before its code, in `rust/docs/`, and the docs
are the spec. Your job is to make the code say what the docs say.

**Follow the docs over the surrounding code.** Agents here have copied a
neighbouring file that broke a rule, and the rule drifted. When the code
beside you and a doc disagree, the doc wins: fix the counter-example in your
PR if it is small, or file an issue and say so in your report. If you think
the doc is wrong, stop and report that; do not quietly write code that
departs from it.

**The docs move; this file does not restate them.** Read the sections named
below from the checkout you are working in, every time. Paths in this file
are relative to `rust/`.

## Before you write anything

1. Read `CLAUDE.md` (the one in `rust/`) and the repository root's. They
   are short and carry the rules most often broken.
2. Read `docs/design.md` whole. It is the index: ten questions a feature
   answers, and for each, where the answer is recorded and what checks it.
3. Read the sections your task touches, from this table.
4. Open `docs/rules.md` at the tables for those documents. Every row is a
   rule with the check that holds it. A row marked unenforced is a rule
   nothing will catch for you, so those are the ones to check by hand.
5. Look for what already exists before you design: `README.md`, "What is
   here", has every crate and what it offers, and `docs/design.md`, "What
   already exists", has the order to look in.

| The task | Read |
|---|---|
| A task, an edge, a DAG, a plan, placement, workers | `docs/programming_model.md`, "Three owners", "Plan once, run many", "Workers are a hierarchy", "Granularity" |
| A buffer, a layout, a row's owner or lifetime, admission, a width chosen at planning | `docs/programming_model.md`, "Memory: layouts and arenas"; `docs/memory.md`, "The table", "Lifetimes", "Admission", "Adaptive widths: one planner routine" |
| A kernel, a launch wrapper, a `-sys` crate, a forked `.cu`, a workspace query, the CMake build | `docs/gpu_layer.md`, "Layering", "From bridge to C ABI", "Launch shape and dispatch", "Scratch: the workspace query", "Statuses", "The safe wrapper", "Build" |
| `unsafe`, spawning, a capability a crate may hold | `docs/programming_model.md`, "Capabilities are enforced by lints" |
| A test, a golden, a reference, a tolerance, what a gate runs | `docs/testing.md`, "Contracts are part of the spec", "Tiers: where a check lives", "Levels: how much of the DAG a test covers", "Selecting GPU tiers in cargo", "Slow cases", "Regression" |
| A log line, a timing scope, an error, a panic | `docs/diagnostics.md`, "Levels", "Values, never panics", "The one boundary", "Panics and profiles" |
| A doc comment, an equation, a citation, a config key, a diagram | `docs/documentation.md`, "Code: rustdoc", "Equations", "References", "Configuration: the reference is emitted from the type" |
| Porting a C++ module | `docs/design.md`, "Porting a module"; `docs/gpu_layer.md`, "What the port changes"; `README.md`, "Order of work" |

## Designing

A feature's design is its answers to the questions in `docs/design.md`.
Write them down before the code, one line each is enough for a small
change: what it computes and against which reference at which bar, where it
runs, what it holds, whether it splits, what it depends on, what already
exists, what it says about itself, what it writes, what it documents. A
question you cannot answer is a design question: settle it with whoever
gave you the task before you write the PR, not inside it
(`docs/design.md`, "Where a design lives").

Each answer then lands in exactly one artefact: a type, a contract, a
layout, a declaration, a doc comment, a golden (`docs/design.md`, "The spec
is the contracts"). There is no design document beside a module.

## Writing

The rules below are the ones that have drifted before. Each is stated in
full, with its check, in the section cited.

- **Which check is which** (`docs/testing.md`, "Contracts are part of the
  spec"). A caller's precondition that only correctness rests on is a
  debug contract, and the doc comment does not restate it. A check that
  memory safety rests on is an always-on assert named under the item's
  panics heading. The outcome of a GPU call that can fail is a status or a
  result, never a contract or a panic. Push every check to the cheapest
  tier that can express it; a type beats a contract beats a test.
- **`unsafe` stays where the docs list it**
  (`docs/programming_model.md`, "Capabilities are enforced by lints").
  Every other crate root forbids it. Above the core crates, reach a
  library or a kernel through its safe wrapper, never through the raw
  bindings; if the wrapper you need does not exist, add it in the crate
  that may hold `unsafe`, with the extents it trusts checked always
  (`docs/gpu_layer.md`, "The safe wrapper").
- **A task allocates nothing and spawns nothing.** Its buffers are its
  layout's, its threads are the orchestrator's
  (`docs/programming_model.md`, "Memory: layouts and arenas", "CPU
  workers"). A task that must wait for part of itself is two tasks
  (`docs/programming_model.md`, "Granularity").
- **Errors are values** (`docs/diagnostics.md`, "Values, never panics",
  "The one boundary"). Nothing between the detection site and the binary's
  entry point logs an error.
- **The doc comment is the spec** (`docs/documentation.md`, "Code:
  rustdoc"). Every example compiles; a type-level rule gets a compile-fail
  example with its error code.
- **GPU-first** (`docs/testing.md`, "Tiers: where a check lives"). No host
  twin of a GPU task to make it testable. A reference computes the same
  approximation, and its bar is fixed by analysis and never loosened to
  pass (`docs/testing.md`, "Correctness").
- **Prefer fewer names.** A small public surface beats a family of aliases
  and helpers. Look in the core crates before writing a helper or a
  constant.

## A rule ships with its check

If your change states a new rule in a doc, or changes one, the same PR adds
or edits its row in `docs/rules.md` with what checks it: a lint, a type, a
test. Write the check if you can. Mark the row unenforced only when nothing
can check it yet, and say why in the PR. If you build a feature whose row
waits for it, you ship that row's check (`docs/rules.md`, "Enforced at
birth"). The `fb-doc-check` crate fails a
row naming a check that does not exist, a section citation whose heading
is gone, and a crate holding `unsafe` that the docs do not list, so run it
after any doc edit:

```bash
cargo test -p fb-doc-check
```

When a doc and the code it describes change together, change both in one
PR. When a section heading is renamed, every citation of it changes too,
this file's included.

## Building and checking

Work in a worktree, never in the primary checkout: the repository root's
instructions say why, under its agent-specific heading. Run git as plain
commands from the worktree.

```bash
git submodule update --init --recursive       # a new worktree has none
cd rust
cmake --preset cuda && cmake --build --preset cuda   # or hip; long, run it in the background
```

`README.md`, "Build and test", has the rest: which crates need no library
at all, and how a device case is marked and run. Then, before you call a
change done:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --all-targets --no-default-features \
  -p fb-core-arena -p fb-core-thread-pool -p fb-core-planner -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p <each crate you touched>
cargo test -p fb-doc-check
cd .. && uv run --frozen devtools/test.py pr --only cargo --preset default
```

The last line is the gate. Send its output to a file and run it in the
background; wait for its exit notice, never poll for it with a process
search. One card's pass is enough. A committed golden that your change
moves on purpose is regenerated with `FB_BLESS=1` and the diff explained
in the PR (`docs/testing.md`, "Performance"); a golden that moved and you
did not expect to is a finding, not something to bless.

**A green run means nothing until you know what skipped.** Read the gate's
tier 4 summary and its line listing the tier 4 cases that did not run, and
confirm the tests you added are among those that ran.

Commit with the repository's configured identity. Do not merge, force-push
or delete a branch unless the person who gave you the task said to.

## Reviewing

To review a change under `rust/` against the docs: for each file changed,
find the rows of `docs/rules.md` that govern it and check each one, the
unenforced rows first, since no gate looked at those. Then ask the
questions of `docs/design.md` of the change as a whole. Report each
departure with the file and line, the section that states the rule, and
whether a check should have caught it. A rule that held only because
someone remembered it is worth a row and a check; say so.

## What to report

End with what a reader needs to act:

- What changed, as public API: the signatures added, changed or removed.
- Which design questions the change answered differently from the task as
  given, and why.
- Any doc you found the code departing from, and whether you fixed it,
  filed it or left it.
- Any rule you added or changed, with its row and its check.
- The gate's result: the command, the card, the tier 4 counts, and what
  did not run. Say plainly if a step was skipped or failed.
- Anything you could not settle and who needs to decide it.
