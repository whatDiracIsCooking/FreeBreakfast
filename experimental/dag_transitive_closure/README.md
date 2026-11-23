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

The second round checks the rule itself, not just the packer against it:

- The bitset closure equals plain DFS reachability (1000 DAGs, up to 60 steps).
- Simulated runs on a work-stealing pool never have two byte-sharing regions in
  use at once (3500 runs). Dropping the step-end sync, the negative control,
  collides in 2395 of 3000: the sync is what makes the closure the right rule.
- One slab per (device, kind), multi-GPU steps and pinned staging: safe.
- A manager's scratch declared by its callers is safe; owned by the manager it
  collides in every run.
- A condition's exclusive branches: the closure keeps them apart (safe,
  conservative); packing them as exclusive is also safe, and not adopted.
- First fit over a fixed set of four orders, keeping the lowest peak, reaches the
  bound on 99.2% (worst 1.174). On 4000 small DAGs solved exactly, the optimum
  always equals the bound. That is empirical, at most 8 regions: in general it
  cannot hold (the page's "Correctness and better packing" has why).
