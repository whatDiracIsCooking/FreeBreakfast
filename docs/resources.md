# Resources

What a feature says about the compute and memory it uses, before it is built.
This is the contract each new step or routine is designed against, and the
mechanisms that hold the tree to it. Each section says what exists **today**
and what is still **aspiration**, so a design can tell which of its answers
the tree will enforce and which stay prose for now. The mechanisms
themselves are documented where they live:
[`core/resources`](../src/core/resources/README.md) (the node, its devices
and the executor), [`core/footprint`](../src/core/footprint/README.md) (the
vocabulary of memory), [`pipeline/footprint`](../src/pipeline/footprint/README.md)
(the plan fold), [`pipeline/step`](../src/pipeline/step/README.md) (where a
step runs) and [`pipeline/pipeline`](../src/pipeline/pipeline/README.md) (the
scheduling). `src/pipeline/README.md` "Memory caps" is pre-flight end to end.

## The five questions

A feature's design answers these, in its issue or design note, before the
first line of code. A reviewer reads them against the code the same way the
footprint tests read a footprint against the pool.

1. **Which compute does it use?** The GPU, the CPU, or both: which parts run
   where, and whether the CPU part runs while the device is busy or instead
   of it.
2. **What memory does each worker need?** Device, pinned host and pageable
   host, each in the three lifetimes: persistent (outlives the step),
   scratch (the step's own, freed at its end) and workspace (borrowed from
   the device's shared block). Per worker, not per feature: a feature that
   fans out states the per-worker figure and the fan-out width.
3. **How is the footprint enforced?** Which of those figures the tree checks
   before the run, which it refuses at the allocation, and which nothing
   checks. "Nothing" is an acceptable answer when it is written down.
4. **What is shared with other workers?** Streams, handles, pools, workspaces,
   the executor, a file: every resource two concurrent parts of the run could
   contend for, and the rule that keeps them apart.
5. **Where are the fork and join points?** What runs beside what, what waits
   on what, and who joins: a Taskflow edge, a stream wait, an explicit subflow
   join, an event.

The rest of this document is the tree's standing answer to each, so a design
need only say where it differs.

## 1. GPU or CPU

**Today.** A step's `run` signature is the declaration. `pipeline/step`'s
`RunShape` reads it off the resource parameters, and the five shapes are the
only ones a step may take:

| `run(View, ...)` | Shape | Runs |
|---|---|---|
| nothing extra | `Host` | on any executor worker, as soon as its inputs exist |
| `tf::Subflow&` | `HostSubflow` | the same, host-parallel inside |
| `DeviceResources&` | `Device` | on the device chain, one step at a time |
| `DeviceResources&, tf::Subflow&` | `DeviceSubflow` | on the chain, host-parallel inside; one thread drives the device |
| `NodeResources&` | `Node` | holding every device's chain |

Taking `DeviceResources&` is the capability to touch the device; a step
without it cannot. The signature cannot disagree with the code, which is why
this half of the contract needs no prose.

Below the step the declaration stops. Several method layers (`mcscf`,
`nevpt2`, `localization`) take the executor directly for host-parallel work
inside a device step. That is legitimate, the `DeviceSubflow` row above is
its shape, but the executor is one node-wide pool of `physical cores − 2`
workers, and nothing says how many of them a routine takes or for how long.

**Aspiration.** A CPU half of the contract: a step states its host width
(how many workers it can keep busy) and whether it needs them while the
device chain runs. Until a feature needs that enforced it is a line in the
design, and the design says which routine takes the executor and why.

## 2. Memory per worker

**Today.** `MemoryFootprint` (`core/footprint`) has two kinds, device and
pinned, each with the three lifetimes. Every `PipelineStep` carries a
`footprint(dims)` member by requirement of the concept, so a step without one
is a build error; it composes the footprints its routines export with `then`
and `drop`, so it reads like the routine it describes. The workspace is one
block per device, sized before the first step at the plan's maximum and
never grown, so it is for what many steps use at similar sizes (a solver's
`lwork`); a one-step intermediate is scratch.

Pageable host memory is not modelled, by decision: every pageable allocation
in the tree is O(nao²), or O(ncas⁴) in the active space, and large host-side
data goes to pinned memory, where the footprint counts it. The exceptions
the decision waves through are small and named: the optimizer's history and
trajectory, the Hessian's 6N gradients, the CI strings.

"Per worker" is mostly moot today because there is one device chain, hence
one device worker: `Pipeline::run` chains every device step on `node.device(0)`.
A `HostSubflow` step beside the chain takes neither device nor pinned memory,
by rule, so its workers need no per-worker figure either.

**Aspiration.** Two things, in the order a feature is likely to force them:

- **A pageable kind.** The O(nao²) argument is a claim nobody checks, and the
  first feature that keeps something O(nao² · natoms) or O(ncas⁶) on the
  host should add a third kind to `MemoryFootprint` rather than argue it
  into the exceptions. The fold already composes per kind, so it is a field,
  a line in `then`, and a machine check against `MemAvailable` beside the
  pinned one.
- **Per-worker figures.** The day a `Node`-shaped step fans a build out over
  devices, its footprint is one device's figure times nothing: each device
  folds its own chain, and the step's design states what each device holds
  and what the host holds across them. The fold is already written as one
  device's chain, so the shape is right; the step is what is missing.

A design states its figures as expressions in the plan's dimensions
(`FootprintDimensions`: nao, naux, nocc, ncas, natoms, the bases), not as
numbers at one molecule. The footprint member is that expression in code.

## 3. Enforcement

**Today**, by kind and by moment:

| Kind | Before the run | At the allocation |
|---|---|---|
| Device | the fold's peak held to the pool's cap less 64 MiB headroom (and a HIP rounding margin); refused naming the step | the pool's `maxSize`, set from `max_device_mem_per_gpu` less the 256 MiB context reserve; a workspace borrow past the plan's maximum is refused at the step |
| Pinned | the fold's pinned peak held to `max_pinned_mem_per_gpu` exactly; the node's devices times the cap held to `MemAvailable` | `CountedPinned` past the cap is `Unsupported`; a pinned workspace past it aborts at construction |
| Pageable host | nothing | nothing |
| CPU workers | nothing | nothing: the executor hands out what it has |

Both caps live in a config's `[memory]` table and are optional. Without
either, pre-flight still sizes the workspaces and refuses a plan whose
workspace alone passes the device; with either, the fold runs, and a step
whose footprint is empty where it should not be is refused rather than
counted as zero (`src/pipeline/README.md` "Memory caps" says which steps a
cap still refuses).

An **adaptive routine**, one that picks a tile, batch or width from free
memory, states as its scratch the least it runs with and sizes its adaptive
part at run time from `memory_budget()`, the cap less what the pool holds,
less the headroom and its own allocations still to come. Some forty routines
do this today, across the DF integrals, the Hessians, NEVPT2 and the CI.
What none of them does yet is declare it to pre-flight: the optional
`adaptive_unit(dims)` member, the bytes of one more unit, gets a warning
when the cap leaves a step under one unit, so "fits, but runs at width 1"
shows before the run. No step carries one. FCI's sigma blocks and the NEVPT2
tiles are the first that should.

**Aspiration.** The two empty rows filled the way the full ones were, by
extending the footprint rather than adding a mechanism beside it: a pageable
field folded and checked against the host, and a declared host width the
executor could be asked to honour. The enforcement a feature cannot have yet
is written into its design as "unchecked", so the next reader knows it was a
choice.

## 4. Shared resources

**Today.** `DeviceResources` is the list, and it is explicit: one device with
its `ForkStream` work stream and memory pool, a `BlasHandle` and a
`SolverHandle` bound to that stream, the device and pinned workspaces, and a
pinned solver `info` slot. **One thread drives a `DeviceResources` at a
time**: the stream's events, cuBLAS and the solver are not safe from two
threads, and every task that drives one selects the device first with a
`ScopedDeviceIndex`, because an executor worker may last have run work for
any device. The integrals add per-class streams (mmd's, J's, K's) beside the
main one, allocating from the same pool, and a step that enqueues on streams
of its own waits on them itself.

Across steps the sharing is by construction: the device chain serializes
device steps, so scratch is one step's at a time and the workspace is never
borrowed twice at once. Host steps beside the chain may take neither device
nor pinned memory, so they contend with it for nothing but executor workers.
The executor itself is shared by every host task and every nested pipeline,
which is why a nested `run` on a worker co-runs its flow there instead of
blocking (`tf::Executor::corun`): a blocked worker is a stolen share.

**Aspiration.** The list that does not exist is what two routines may share
when they run *concurrently on a device*, because today nothing does. A
multi-device `Node` step, or a second stream that does real work beside the
chain's, needs that list before it needs code: which handles are per stream,
whether the workspace is carved or duplicated, and what the two-driver rule
becomes. A design that introduces concurrency on a device writes that list
into `core/resources`'s README as part of landing.

## 5. Fork and join

**Today.** The pipeline is one `tf::Taskflow`, one task per step, with two
kinds of edge: producer to consumer, and device step to next device step in
plan order. So host steps fork off the chain by their data edges and run as
soon as their inputs exist; device steps never fork from each other. The
joins are:

- **After every device step, a wait on `res.stream()`.** A product is
  finished when any consumer reads it, so a device-to-host edge needs no
  event. A `Node` step is followed by a wait on every device.
- **A subflow is joined explicitly** by the `run` that opened it, because it
  holds references into that frame.
- **A nested pipeline** (`[optimize]`, a semi-numerical `[hessian]`) joins by
  co-running on the worker that called it; only a `run` from outside the
  executor waits.
- **A step's own streams** are joined by the step, before it returns.

**Aspiration.** Two forks the planner cannot see. The first is inside method
layers: a routine that opens a subflow or takes the executor forks work the
pipeline's graph does not show, and a design that does so says where the
join is and that nothing device-side is touched past it. The second is
across devices: the `Node` shape is a row in `RunShape`'s table with no step
behind it, and "one device for now" is written into `Pipeline::run`.
Multi-device fan-out is undesigned, not merely unbuilt, and the first `Node`
step's design is where the fork (which device gets what) and the join (an
event per device, or a wait on each) are decided for every step after it.

## Writing the contract into a design

Five lines are enough when the tree's standing answer holds; a design says
only where it differs. A step that fits the pattern reads:

```
Compute:   Device. Host-parallel inside (DeviceSubflow) for the RDM contraction.
Memory:    device persistent nao²·nroots (the product); scratch 3·naux·nocc·nvir
           (the tiles, adaptive, least width 1); workspace syevd lwork at nao.
           pinned none. pageable O(ncas⁴) (the 2-RDM, host copy).
Enforced:  device and pinned by the fold and the pool; adaptive_unit declared
           (one tile = naux·nocc bytes); pageable unchecked.
Shared:    res.stream(), the BLAS handle; the executor for the subflow.
Fork/join: the subflow, joined before the first launch; no streams of its own.
```

The test that holds it is the one `test/gtest/pipeline_step_footprint_gpu.cpp`
sets the pattern for: the step run once on a fresh `DeviceResources` whose
workspaces are the footprint's, its `workspace_high_water()` then exactly the
footprint's workspace, and its pool's high-water against the rest. A design
whose figures cannot be tested that way says why, and what it tests instead.

## What this document is not

It is not the memory model, which is `core/footprint`'s README, and not the
pre-flight, which is `src/pipeline/README.md` "Memory caps". It is the list
of questions those answer, and the honest record of which they do not yet.
When an aspiration above lands, move it to the **Today** half of its section
and point at the README that now holds it.
