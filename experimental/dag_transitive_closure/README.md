# `dag_transitive_closure/`

Does packing each step's scratch region from the transitive closure of the
step DAG behave as the Rust redesign says
([`docs/rust-redesign.html`](../../docs/rust-redesign.html#scratch), "Step
scratch")? Two regions may overlap only when the DAG orders their steps, so no
edge is added for memory.

A model over a fixed dummy DAG, sizes in MiB that are not measurements:

- [`dag_transitive_closure.html`](dag_transitive_closure.html): the question,
  the dummy DAG, the closure, the packed slab and what the model found, computed
  live in the page's JavaScript.
- [`model.py`](model.py): the reference. Standard library only; it checks the
  page's input against its own and pins every number the page prints.

```sh
uv run python experimental/dag_transitive_closure/model.py
```

It exits non-zero on the first failed check. Nothing is built; no CMake, no GPU.

What it found: on the dummy DAG, packing reaches the lower bound (2,049 MiB,
against a plain sum of 3,318 and a lock's max of 2,040). Over 3000 random DAGs
it is always safe and offsets never depend on declaration order. Largest-first
first fit reaches the bound on 98.1% of them, worst 1.281×, and on every miss
some other first-fit order reaches it.
