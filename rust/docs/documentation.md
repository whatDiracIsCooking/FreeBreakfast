# Rust rebuild: documentation

Documentation is everything a reader learns about the rebuild without running
it: the API reference, the design and theory pages, the equations the code
implements, the sources those equations come from, and the pictures. The rule
that joins them is the one the testing document applies to checks: a
statement about the code lives where the build can hold it to the code. A
doc example compiles, a diagram is emitted from the structure it draws, an
equation names the line of the paper it transcribes, and a reference is one
entry that every citation keys into.

The C++ tree already has most of the habits. Every directory under `src/`
carries a README and the root README links them; the theory lives in two
LaTeX documents (`docs/nevpt2/nevpt2.tex`, `docs/hessian/hessian.tex`) whose
labelled equations the READMEs cite by label; the NEVPT2 code cites Park 2019
by equation number; two Python checkers hold `docs/style.md`'s rules, and a
pytest pins `docs/testing.md`'s record table. What the tree lacks is a
published rendering of any of it, a bibliography (its citations are inline
prose in a dozen formats), any rule about pictures, and a configuration
reference that cannot drift from the parser. The rebuild adds those four and
keeps the habits, and drops one: the README beside every module, which
rustdoc makes a second copy.

The design leans on the other documents' terms: the *tiers* of
[`testing.md`](testing.md), a *plan* and its *arena*
([`programming_model.md`](programming_model.md)), and the *timing record*
([`diagnostics.md`](diagnostics.md)).

## Code: rustdoc

**The API reference is rustdoc**, built with `cargo doc --no-deps`. A module's
doc comment carries its spec: what it does, its contracts, and, while the port
lasts, its C++-to-Rust table, as the crate's seven modules already do. The
crate's front page is the README, included with
`#![doc = include_str!("../README.md")]`, so the module list exists once and
the README stays readable on GitHub.

**Warnings are errors, and the manifest says so.** The crate has been checked
with `RUSTDOCFLAGS=-D warnings` by hand (its README records the runs), which
means a plain `cargo doc` still passes with a broken link. The lints move into
`Cargo.toml`, where every invocation sees them:

```toml
[lints.rust]
missing_docs = "deny"

[lints.rustdoc]
broken_intra_doc_links = "deny"
private_intra_doc_links = "deny"
```

`missing_docs` is the half `-D warnings` never covered: a public item with no
doc comment is an error, so "a module's doc comment carries its spec" is a
rule the build enforces rather than a habit. The same table carries the
capability lints, `unsafe_code` and clippy's `disallowed_methods` among them
([`programming_model.md`](programming_model.md), "Capabilities are enforced
by lints").

**A doc example is a test, so it is never `ignore`.** `cargo test` compiles and
runs every example, which is what keeps an example from drifting from the API
it shows. An ```` ```ignore ```` block opts out of both, and the crate's one
example today (`core::diagnostics`) is one. The three markers that stay are:

| Marker | What `cargo test` does | For |
|---|---|---|
| none | Compiles and runs | Host-only code: a `Result` chain, a layout's sizes, an error's exit code |
| `no_run` | Compiles, never runs | Anything that touches a device. The example is checked against the API; the card is not needed |
| `compile_fail` | Fails to compile, and the test fails if it compiles | A tier-1 contract: a view outliving its handle, a host buffer passed where a device one goes, two dimension newtypes swapped |

The last row makes the documentation the tier-1 test suite. A type-level
contract has no runtime test to write, so its `compile_fail` example is both
the place a reader learns the rule and the check that the rule still holds.
Each one carries the error code it expects (```` ```compile_fail,E0597 ````),
so a different compile error does not pass for the intended one.

**Doctests touch no device.** With `no_run` for every GPU example, the doctests
are tier 3 and run on every PR with the other device-free tiers
([`testing.md`](testing.md), "What runs when"). An example that needs a card
to be worth running is an integration test, not an example.

**No README beside a module.** The C++ tree has a README in every directory
under `src/`, 152 of them, beside the Doxygen comments, because Doxygen is not
built and the README is what a reader finds. rustdoc renders the module doc
comment, so a `README.md` beside a module would be a second copy of the same
spec, and the two would drift. The crate has one README, its front page, and
no other Markdown under `src/`. What the C++ README said about a module goes
in the module's doc comment, which is also where `missing_docs` can see it.

**A contract appears in the reference.** The testing document makes contracts
part of the spec ([`testing.md`](testing.md), "Contracts are part of the
spec"), so a reader of the reference must see them without opening the
source. Whether the `contracts` crate writes its conditions into the item's
rustdoc is an open question below; until it is answered, a doc comment
restates each `#[requires]` in a line of its own, as the crate's README does
for its two today.

**The verification record is the metrics branch, not the README.** The
crate's README carries two dated "Verified" paragraphs: card, test counts,
runs in a row, clippy and rustdoc clean. That is a changelog, and it will not
scale past a few modules. The nightly already writes what ran, where and how
long to the metrics branch ([`testing.md`](testing.md), "Performance"), so
that branch is the record and the site reads it. The README says what the
crate is, and the two paragraphs retire when the nightly covers the crate.

## The C header

The GPU layer makes a C header the single source for everything that crosses
the boundary: the launch wrappers' signatures, their status enum and the
shared types ([`gpu_layer.md`](gpu_layer.md), "From bridge to C ABI"). It is
the single source for their documentation too.

- **The header's comment is the wrapper's contract.** What a launch wrapper
  requires of its pointers and sizes, which stream it enqueues on, and which
  refusals it returns are written once, as a `///` comment in the header, the
  C++ style ([`docs/style.md`](../../docs/style.md), "Comments").
- **bindgen carries it across.** bindgen copies Doxygen-style comments onto
  the items it generates, so the header's text is the `-sys` crate's rustdoc
  with nothing retyped. It reads only `///` and `/** */` comments unless told
  otherwise, which the style already requires.
- **The safe wrapper links down, and adds only what is new.** Its doc comment
  says what the Rust type guarantees over the C call (lifetimes, the checked
  shapes) and links to the `-sys` item for the rest. The kernel's contract is
  therefore read in one place whichever side the reader starts from.
- **Device code is documented on the C++ side only.** A `.cuh` that never
  crosses is never mirrored, so its comments stay in the C++ style with no Rust
  counterpart.

## Equations

rustdoc does not render math. KaTeX, loaded through `--html-in-header`
(`RUSTDOCFLAGS`), renders `$...$` and `$$...$$` in doc comments. It is one HTML
file in the repo and adds no build dependency.

**Markdown reads the equation first.** KaTeX runs in the browser over the HTML
rustdoc produced, so every equation has already been through Markdown, which
treats `\\` as an escaped backslash and `*` as emphasis. A row break in a
matrix does not survive, and nothing reports it. Doc comments therefore carry
the equations that fit one line, and a derivation goes to the narrative pages,
where the KaTeX preprocessor runs before Markdown (below).

**Unicode for a name, KaTeX for a displayed equation.** A doc comment is read
in an editor more often than on the site, and the C++ READMEs are written for
that: `⟨Ψ_P|∇Ψ_Q⟩`, `Σ_k N_k / (Δ_k + H_k / N_k)`. An inline quantity keeps
that spelling, which also makes it greppable. `$$` is for an equation set
off on its own lines, where the source's readability matters less than the
rendering's.

**Cite a published equation by number, our own by label.** Park 2019's
equation numbers are fixed, so `Park 2019 Eq. 47` is a stable address, and the
NEVPT2 code uses it. The tree's own documents are edited, so their numbers
move, and the C++ READMEs already cite them by label (`hessian.tex` eq.
`(df2)`, `nevpt2.tex` eq. `koo`). The rebuild keeps both conventions, and a
label in a doc comment is a link to the equation on the site.

## References

References are tracked in one bibliography file (BibTeX) in the repo, with one
entry per source and a stable key. Docs cite by key, and the key links to a
generated references page. There are five kinds of entry:

| Kind | What the entry records |
|---|---|
| Article | DOI. The equations and sections the code uses. |
| Textbook | Edition, and the chapters and sections the code uses. |
| Other codebase | URL, the version or commit compared against, and its license. For example, PySCF as the reference for regression values. |
| Data | Source and version, such as the BSE basis sets |
| License obligation | Code or data derived from someone else's, and what it requires: a notice, attribution, or a compatible license |

Two rules come with the bibliography:

- **Consulted and derived are different.** Reading a paper or a codebase owes
  nothing. Porting code from one carries its license. The entry says which it
  was, so an obligation such as a BSD notice is recorded when the code arrives,
  not discovered later.
- **Metadata only.** The bibliography is public, so it holds no PDFs (they are
  copyrighted) and no references to private material.

**The tree already has one derived entry, kept in prose.** `data/basis/minao.json`
is PySCF's MINAO table, converted by `tools/minao_from_pyscf.py`, whose
docstring records PySCF's license and the version it was taken from. That is
exactly a License-obligation entry, written where one script could hold it.
The bibliography is that bookkeeping generalised, and MINAO is its first
entry of the kind. The tree's own license is MIT in one file, with no notice
file beside it; the first entry that requires a notice is what creates one.

**One file for the whole tree.** `docs/references.bib`, not a file under the
rebuild. The C++ tree has no bibliography at all, so the rebuild's would be the
tree's first, and a second one would follow the day the rebuild leaves
`experimental/`. The C++ READMEs adopt keys as they are touched, which turns
their dozen citation formats into one over time, with no sweep.

**A key is checked.** A tier-3 test reads every `[@key]` in the docs and every
key in the bibliography, and fails on a citation with no entry or an entry
with no citation, the way the C and Rust launch-status lists are checked
against each other ([`testing.md`](testing.md), "The C++ boundary"). A
reference that nothing cites is as stale as a comment that nothing matches.

## Configuration: the reference is emitted from the type

The reader who runs the binary reads none of the above. They read the
configuration reference and the examples. In the C++ tree the reference is
`src/pipeline/config/README.md`, 772 lines kept by hand against the TOML the
parser accepts, and `examples/` holds one config per step family, three of
which `test/examples/` runs through the binary.

**The reference is emitted from the config type.** The rebuild's config is a
serde type, so each key's documentation is the field's doc comment, with its
default and its unit, and the reference page is generated from the type by
the docs build. A key the parser accepts and the reference lacks, or the
reverse, cannot exist. The page is also the rustdoc of the type, so the two
readers see one text.

**Examples are checked, then linked.** `examples/` carries over as it is: one
small config per step family, each run through the binary by a test, so an
example that stops parsing fails a PR. The reference links each key to the
example that sets it, and the examples page is the generated list of them
with their first-line comments, not a hand-kept table.

**Error messages are documentation.** The reference says what a key takes;
the parser's refusal says what it got. The refusal names the key, the file
and the line, and a reference entry quotes its own refusal, so a reader
searching the error text lands on the key.

## Narrative: mdBook beside the rustdoc

rustdoc is for the API. The design pages (this one and its siblings), the
theory and the reproduction notes are Markdown, and they stay Markdown in the
repo, where they are written, reviewed in a diff and read on GitHub without a
build. The tree's design docs are already read that way: `docs/style.md`,
`docs/testing.md`, `docs/resources.md`, and a pytest pins one of them. Moving
them into a format only a site can show would lose the readers they have.

mdBook renders that Markdown as a site, and is the choice because it changes
nothing about the source:

- **The source stays readable raw.** mdBook's input is CommonMark with a
  `SUMMARY.md` table of contents. The design documents under `docs/` are its
  first chapters unchanged, `design.md` first.
- **Equations are preprocessed.** `mdbook-katex` renders math before the
  Markdown pass, so a matrix survives. A derivation belongs here, not in a
  doc comment, for that reason.
- **Diagrams render on GitHub and on the site alike.** A Mermaid block renders
  in GitHub's Markdown view natively and in mdBook through `mdbook-mermaid`.
  That is why a checked-in diagram is Mermaid rather than DOT when there is a
  choice.
- **It is a cargo install.** Rust's toolchain is in the image at `/opt/rust`
  (`docker/Dockerfile.base`), so `mdbook`, `mdbook-katex` and `mdbook-mermaid`
  are three more pinned installs in the same stage, not a new toolchain.

**Reproduction notes are a fourth kind of page.** `docs/reproduction/park2019/`
is one: a paper's result reoptimised here, with the configs, the geometries,
the numbers and what could not be reproduced. The rule for such a page is
that its numbers come from a committed run product (a `hessian.json`, an
energy file), read by the docs build or quoted with the product's path beside
them, never retyped from a terminal. A page whose numbers can be regenerated
by a command names the command; one whose numbers cannot says so.

**A section citation is checked.** These documents cite each other as a file
and a quoted section name, and a renamed heading breaks the citation
silently: two such references pointed at sections of `design.md` that had
moved to documents of their own, and were found by hand. A tier-3 check reads
every `` [`x.md`](x.md), "Section" `` and fails when `x.md` has no such
heading. It is the first tier-3 test to write
([`README.md`](../README.md), "Order of work"). Links between pages and into
the API get `mdbook-linkcheck` the same way, so a page that moves is a failed
build, not a dead link.

**The two LaTeX documents stay LaTeX.** `nevpt2.tex` and `hessian.tex` are
long derivations with TikZ figures and algorithm floats, and a conversion
would be a rewrite. Their committed PDFs are the one binary media the tree
carries (1.3 MB between them), and they move out: the site build renders the
`.tex` and publishes the PDF, so the repo holds the source and the site holds
the picture. Until the site exists they stay where they are.

**One site, two generators.** `cargo doc` writes the API under `/api/` and
mdBook writes the book at the root, with a link each way, published to GitHub
Pages by one workflow.

## Diagrams and animations

**Diagrams are source, not pictures.** They are checked in as text, Mermaid or
hand-written SVG, and rendered when the docs build. The tree already does the
second half: `experimental/cope_rearrangement`'s surfaces and scans are SVG
written by stdlib Python, committed as text, and shown by its READMEs.

**DAG diagrams are emitted from the plan.** The programming model derives a
plan from the DAG: the fork and sync points, the placement, the arena. The
diagram worth having shows the plan, not the DAG the author drew, so the
emitter is on the plan type and writes Mermaid (for the pages) or DOT (for a
reader's own tools). It cannot drift from what the orchestrator runs, because
it is produced from the same structure.

**A plan's diagram is a tier-3 golden.** The plan is a deterministic function
of the DAG and its dimensions, which is what makes the arena's size a committed
number compared bitwise ([`testing.md`](testing.md), "Performance"). The
emitted Mermaid is the same kind of number: committed beside the arena size,
compared as text, so a planning change that adds a sync point or moves a task
is a diagram diff in the PR, read in review.

**Animations are generated from real runs, never drawn by hand.** A task
timeline from a run's timing record, SCF convergence, an IRC path, orbitals
along a reaction coordinate. A run needs a card, and the docs build has none
(below), so the nightly makes them: it already runs on a card and already
writes the timing record's task times to the metrics branch
([`diagnostics.md`](diagnostics.md), "The timing record"). The media goes to
the same branch, and the site build reads it from there. Binary media stays
out of `main` either way, and the site shows last night's run.

## Publishing

**The Pages site needs no GPU, and must not need the toolchain.** `cargo doc`
runs `build.rs`, and the crate's `build.rs` panics when
`libwwr_capi_runtime_api.so` is missing. That library is CMake's, built with
the CUDA or HIP toolchain, which is the one thing a GitHub-hosted runner
cannot have: it is why `.github/workflows/ci.yml`'s `cpp` job is `if: false`.
A Pages workflow that builds the rustdoc therefore has the `cpp` job's
problem, unless `build.rs` is told it is documenting. A tree-chosen
environment variable (`FB_RUST_DOC_ONLY=1`) makes `build.rs` emit nothing and
return, and rustdoc, which never links, succeeds. Doctests do link, so they
stay on the card hosts with the PR's tiers; the runner builds the pages and
nothing else.

**Publish now, under `experimental/`.** The site is the only rendering of
these documents, the KaTeX and Mermaid conventions are only checked by
rendering them, and the `build.rs` change above is the whole cost. The site
says it documents an experimental tree, the way `experimental/README.md`
does.

**What the docs build checks, and where.** The checks are tier 3 and tier 1,
and run where the testing document puts those tiers:

| Check | Tier | Runs |
|---|---|---|
| `cargo doc` with the lints above | 1 | Every PR, on the runner |
| Doctests, including `compile_fail` | 1 and 3 | Every PR, on a card host (they link) |
| Bibliography keys match citations | 3 | Every PR, on the runner |
| Section citations name a heading that exists; `mdbook-linkcheck` | 3 | Every PR, on the runner |
| The config reference matches the config type | 3 | Every PR, on the runner; it is generated, so the check is that the committed page is the generated one |
| Plan diagrams match their goldens | 3 | Every PR, on a card host (the planner is in the crate) |
| The book builds, with every equation and diagram rendered | 3 | Every PR, on the runner; `mdbook-katex` and `mdbook-mermaid` fail on a construct they cannot render |

## Open questions

- **Private items.** `cargo doc --document-private-items` would put the
  orchestrator's internals on the site, which the design pages describe
  anyway. The question is whether `missing_docs` should extend to them, which
  it does not by default.
- **How a doc comment links to the book.** A rustdoc page can link to a book
  page by relative URL once both are on one site, but the link is a string
  rustdoc cannot check, unlike an intra-doc link. A tier-3 test that resolves
  every such URL against the built site is one answer.
- **Equation labels across the two generators.** A label in `hessian.tex` is
  LaTeX's; one in a book page is `mdbook-katex`'s; a doc comment's `$$` has
  none. Whether one labelling convention can span all three, or the doc
  comments only cite and never define, is open.
- **Contracts in rustdoc.** The `contracts` crate's documentation describes
  its attributes and its feature flags (`disable_contracts`, `override_debug`,
  `override_log`, `mirai_assertions`) and does not say whether a condition is
  written into the item's rustdoc. Reading one rendered page answers it. If
  it is, the restating rule above goes; if not, a tier-3 check that every
  `#[requires]` has a line in the doc comment holds the rule.
- **The site's version.** One site for `main`, or one per tag as well? The
  tree has no tags today, so one site, rebuilt on merge, is the answer until
  it has.
