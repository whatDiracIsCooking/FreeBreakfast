"""Step scratch packed by the transitive closure of the step DAG: a model.

The Rust redesign (rust/docs/memory.md, "Step scratch") packs
each step's scratch region into one slab per device and kind, letting two
regions overlap only when the DAG orders their steps. This is that packing
over fixed dummy sizes, and the checks dag_transitive_closure.html lists.
Standard library only:

    uv run python experimental/dag_transitive_closure/model.py

It exits non-zero on the first failed check. The page computes the same
closure and packing in JavaScript from the same values; check_page holds the
two inputs equal, and every number the page quotes is pinned here.

Two kinds of check. Static ones hold the packing against the closure: no two
regions of unordered steps intersect. Those cannot show the closure is the
right rule, so execution checks simulate runs on a work-stealing pool and look
for two regions sharing bytes while both are in use. A negative control drops
the step-end sync and must find such pairs, or the simulation tests nothing.
"""

from __future__ import annotations

import itertools
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "dag_transitive_closure.html"


@dataclass(frozen=True)
class Step:
    id: str
    reads: tuple[str, ...]
    out: int  # MiB, summed; not packed
    scratch: int  # MiB, packed, in device 0's device memory
    # More regions, each (device, kind, MiB): a multi-GPU step's other cards,
    # pinned staging. Packing is per (device, kind); each slab is its own.
    extra: tuple[tuple[int, str, int], ...] = ()


DEFAULT = (0, "device")

# The dummy DAG, in MiB. Not measurements: loosely gly10-shaped.
DUMMY = (
    Step("S", (), 4, 1),
    Step("T", (), 4, 1),
    Step("V", (), 4, 2),
    Step("guess", (), 4, 0),
    Step("metric", (), 104, 20),
    Step("X", ("S",), 4, 6),
    Step("H", ("T", "V"), 4, 0),
    Step("factor", ("metric",), 104, 8),
    Step("eri", ("factor",), 7800, 2040),
    Step("scf", ("H", "X", "eri", "guess"), 8, 40),
    Step("casscf", ("scf", "eri"), 20, 300),
    Step("nevpt2", ("casscf", "eri"), 2, 900),
)


def regions_of(step):
    """{(device, kind): MiB} for every region the step declares."""
    out = {DEFAULT: step.scratch} if step.scratch > 0 else {}
    for dev, kind, size in step.extra:
        out[(dev, kind)] = out.get((dev, kind), 0) + size
    return out


def size_in(step, key):
    return regions_of(step).get(key, 0)


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


class Dag:
    """Steps, their edges and the closure, as bitmasks over step indices."""

    def __init__(self, steps, exclusive=()):
        self.steps = list(steps)
        self.n = len(self.steps)
        self.index = {s.id: i for i, s in enumerate(self.steps)}
        if len(self.index) != self.n:
            raise ValueError("duplicate step id")
        self.succ = [0] * self.n
        for j, s in enumerate(self.steps):
            for r in s.reads:
                self.succ[self.index[r]] |= 1 << j
        self.desc = [0] * self.n
        for i in reversed(self._topo()):
            m = self.succ[i]
            for j in _bits(self.succ[i]):
                m |= self.desc[j]
            self.desc[i] = m
        # ordered[i]: every step i reaches or is reached by, itself included.
        self.ordered = [0] * self.n
        for i in range(self.n):
            self.ordered[i] |= self.desc[i] | (1 << i)
            for j in _bits(self.desc[i]):
                self.ordered[j] |= 1 << i
        # Exclusive branches of one condition never both run, so steps on two
        # of them could share bytes. Off unless asked: see check_branches.
        for branches in exclusive:
            for x, y in itertools.permutations(branches, 2):
                for a in x:
                    for b in y:
                        self.ordered[self.index[a]] |= 1 << self.index[b]

    def _topo(self):
        indeg = [len(s.reads) for s in self.steps]
        ready = [i for i in range(self.n) if indeg[i] == 0]
        order = []
        while ready:
            i = ready.pop()
            order.append(i)
            for j in _bits(self.succ[i]):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
        if len(order) != self.n:
            raise ValueError("the step graph has a cycle")
        return order

    def topo_ids(self):
        """A topological order, ties by step id: fixed for a given DAG."""
        indeg = [len(s.reads) for s in self.steps]
        ready = sorted(s.id for s in self.steps if not s.reads)
        order = []
        while ready:
            i = ready.pop(0)
            order.append(i)
            for j in _bits(self.succ[self.index[i]]):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(self.steps[j].id)
                    ready.sort()
        return order

    def is_ordered(self, a, b):
        return bool(self.ordered[self.index[a]] >> self.index[b] & 1)

    def keys(self):
        return sorted({k for s in self.steps for k in regions_of(s)})


# ---- packing ----


def pack(dag, order=None, key=DEFAULT):
    """Greedy first fit, largest first, ties by step id. Returns {id: offset}.

    The order depends on sizes and ids only, never on declaration order, so the
    offsets are the same however the steps were listed. `order`, a list of the
    scratch-declaring steps, replaces it. One slab: the regions of `key`.
    """
    regions = order or sorted(
        (s for s in dag.steps if size_in(s, key) > 0),
        key=lambda s: (-size_in(s, key), s.id),
    )
    placed = []  # (id, offset, size)
    offset = {}
    for s in regions:
        size = size_in(s, key)
        conflicts = [p for p in placed if not dag.is_ordered(s.id, p[0])]
        for o in sorted({0, *(p[1] + p[2] for p in conflicts)}):
            if all(o + size <= po or po + ps <= o for _, po, ps in conflicts):
                break
        offset[s.id] = o
        placed.append((s.id, o, size))
    return offset


def pack_all(dag):
    """{(device, kind): {id: offset}}: one slab per key, packed independently."""
    return {k: pack(dag, key=k) for k in dag.keys()}


def peak(dag, offset, key=DEFAULT):
    return max(
        (offset[s.id] + size_in(s, key) for s in dag.steps if s.id in offset),
        default=0,
    )


def _unordered(dag, s):
    return [q for q in dag.steps if q.id != s.id and not dag.is_ordered(s.id, q.id)]


# Fixed orders for pack_best, tried in this order; the first lowest peak wins.
ORDERS = {
    "largest": lambda dag, s, key, topo: (-size_in(s, key), s.id),
    "constrained": lambda dag, s, key, topo: (
        -(size_in(s, key) + sum(size_in(q, key) for q in _unordered(dag, s))),
        s.id,
    ),
    "size x conflicts": lambda dag, s, key, topo: (
        -size_in(s, key) * (1 + len(_unordered(dag, s))),
        s.id,
    ),
    "topological": lambda dag, s, key, topo: (topo[s.id], s.id),
}


def pack_best(dag, key=DEFAULT):
    """First fit over each of ORDERS; the lowest peak, ties to the earlier
    order. Deterministic, since the set of orders is fixed. (name, offsets)."""
    topo = {i: k for k, i in enumerate(dag.topo_ids())}
    regs = [s for s in dag.steps if size_in(s, key) > 0]
    best = None
    for name, rank in ORDERS.items():
        order = sorted(regs, key=lambda s, r=rank: r(dag, s, key, topo))
        off = pack(dag, order, key)
        if best is None or peak(dag, off, key) < peak(dag, best[1], key):
            best = (name, off)
    return best


def optimum(dag, key=DEFAULT, limit=8):
    """The least peak of any packing that keeps unordered regions apart, by
    first fit over every order of the regions; None past `limit` regions.

    First fit over all orders reaches the optimum: take an optimal placement
    and first-fit its regions in increasing offset. By induction each lands at
    or below its optimal offset, since every conflict placed before it ended at
    or below that offset in the optimum and has only moved down.
    """
    regs = [s for s in dag.steps if size_in(s, key) > 0]
    if len(regs) > limit:
        return None
    return min(
        peak(dag, pack(dag, list(o), key), key) for o in itertools.permutations(regs)
    )


def lower_bound(dag, key=DEFAULT):
    """The heaviest antichain of scratch-declaring steps: no packing that keeps
    unordered regions apart can go below it. Exact, by branch and bound."""
    ids = [i for i, s in enumerate(dag.steps) if size_in(s, key) > 0]
    w = [size_in(dag.steps[i], key) for i in ids]
    best = [0, 0]  # weight, mask over positions in ids
    # Positions k whose steps are ordered with position j, as a mask over ids.
    clash = [
        sum(1 << k for k, b in enumerate(ids) if dag.ordered[a] >> b & 1) for a in ids
    ]

    def go(k, allowed, weight, chosen):
        if (
            weight + sum(w[j] for j in range(k, len(ids)) if allowed >> j & 1)
            <= best[0]
        ):
            return
        if k == len(ids):
            best[:] = [weight, chosen]
            return
        if allowed >> k & 1:
            go(k + 1, allowed & ~clash[k], weight + w[k], chosen | 1 << k)
        go(k + 1, allowed & ~(1 << k), weight, chosen)

    go(0, (1 << len(ids)) - 1, 0, 0)
    return best[0], [dag.steps[ids[k]].id for k in _bits(best[1])]


def unsafe_pairs(dag, offset, key=DEFAULT):
    """Pairs of steps whose regions intersect and which the DAG does not order."""
    sized = [s for s in dag.steps if s.id in offset]
    bad = []
    for a, b in itertools.combinations(sized, 2):
        oa, ob = offset[a.id], offset[b.id]
        intersect = oa < ob + size_in(b, key) and ob < oa + size_in(a, key)
        if intersect and not dag.is_ordered(a.id, b.id):
            bad.append((a.id, b.id))
    return bad


def reach_dfs(steps):
    """Reachability by plain depth-first search over the reads, independent of
    Dag's bitset closure: {id: set of ids it reaches}."""
    succ = {s.id: [] for s in steps}
    for s in steps:
        for r in s.reads:
            succ[r].append(s.id)
    out = {}
    for s in steps:
        seen, stack = set(), list(succ[s.id])
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(succ[x])
        out[s.id] = seen
    return out


# ---- execution ----


def simulate(dag, packing, rng, workers, sync=True, uses=None, branches=()):
    """One execution on a pool of `workers`, any ready step taken at random (as
    stealing would). Returns every (key, a, b) whose regions share bytes and
    were in use at the same time.

    A step runs its host part (enqueueing) for h, and its device work ends h + d
    after it starts. With `sync` the wrapper waits for the device, so the node
    finishes then; without, the node finishes when the host part does and the
    device work runs on past it (the negative control). A region is in use from
    its step's start to the end of its device work. `uses` gives a region more
    users ({owner: [ids]}): it is in use during theirs too. `branches` is a list
    of (condition, [branch, ...]), each branch a list of ids: one branch, at
    random, runs; the others' steps are skipped and use nothing.
    """
    succ = {s.id: [] for s in dag.steps}
    for s in dag.steps:
        for r in s.reads:
            succ[r].append(s.id)
    skipped = set()
    for _, alts in branches:
        keep = rng.randrange(len(alts))
        for k, alt in enumerate(alts):
            if k != keep:
                skipped.update(alt)
    left = {s.id: len(s.reads) for s in dag.steps}
    ready = sorted(i for i, v in left.items() if v == 0)
    running = []  # (node finish, id)
    span = {}  # id -> (start, device end)
    now, free = 0.0, workers

    def finish(i):
        for j in succ[i]:
            left[j] -= 1
            if left[j] == 0:
                ready.append(j)

    while ready or running:
        while ready and free:
            i = ready.pop(rng.randrange(len(ready)))
            if i in skipped:  # not run: finishes at once, uses nothing
                finish(i)
                continue
            h = rng.uniform(0.1, 1.0)
            d = rng.choice((0.0, rng.uniform(0.0, 2.0)))
            span[i] = (now, now + h + d)
            running.append((now + h + d if sync else now + h, i))
            free -= 1
        if not running:
            continue
        running.sort()
        now, i = running.pop(0)
        free += 1
        finish(i)
    intervals = {i: [span[i]] for i in span}
    for owner, users in (uses or {}).items():
        if owner in span:
            intervals[owner] += [span[u] for u in users if u in span]
    by = {s.id: s for s in dag.steps}
    bad = []
    for key, offsets in packing.items():
        live = [i for i in offsets if i in intervals]
        for a, b in itertools.combinations(live, 2):
            oa, ob = offsets[a], offsets[b]
            if not (oa < ob + size_in(by[b], key) and ob < oa + size_in(by[a], key)):
                continue
            if any(
                x0 < y1 and y0 < x1
                for x0, x1 in intervals[a]
                for y0, y1 in intervals[b]
            ):
                bad.append((key, a, b))
    return bad


# ---- random DAGs ----


def random_dag(rng, n, p, keys=(DEFAULT,)):
    """n steps, each earlier step read with probability p, 15% with no scratch,
    sizes log-uniform from 1 to 2048 MiB, declared in shuffled order. With more
    than one key, each step's region goes to a random key, and a fifth of the
    steps declare a second one (a multi-GPU step, or pinned staging)."""
    steps = []
    for j in range(n):
        reads = tuple(f"n{i}" for i in range(j) if rng.random() < p)
        size = 0 if rng.random() < 0.15 else round(2 ** rng.uniform(0, 11))
        if keys == (DEFAULT,):
            steps.append(Step(f"n{j}", reads, 0, size))
            continue
        regs = {}
        if size:
            regs[rng.choice(keys)] = size
            if rng.random() < 0.2:
                k = rng.choice(keys)
                regs[k] = regs.get(k, 0) + round(2 ** rng.uniform(0, 9))
        main = regs.pop(DEFAULT, 0)
        extra = tuple((d, kind, v) for (d, kind), v in sorted(regs.items()))
        steps.append(Step(f"n{j}", reads, 0, main, extra))
    rng.shuffle(steps)
    return steps


# ---- checks ----


def check(cond, what):
    if not cond:
        raise AssertionError(what)


def check_page():
    """The page's STEPS literal carries the same dummy values as DUMMY."""
    text = PAGE.read_text()
    block = re.search(r"const STEPS = \[(.*?)\];", text, re.S).group(1)
    row = re.compile(
        r'\{id: "([^"]+)",.*?reads: \[([^\]]*)\],\s*out: (\d+),\s*scratch: (\d+)\}'
    )
    page = tuple(
        Step(i, tuple(re.findall(r'"([^"]+)"', reads)), int(o), int(sc))
        for i, reads, o, sc in row.findall(block)
    )
    check(page == DUMMY, f"the page's STEPS differ from DUMMY:\n{page}\n{DUMMY}")


def check_dummy_dag():
    dag = Dag(DUMMY)
    off = pack(dag)
    total = sum(s.scratch for s in DUMMY)
    biggest = max(s.scratch for s in DUMMY)
    bound, chain = lower_bound(dag)
    check(unsafe_pairs(dag, off) == [], "the dummy DAG packs unsafely")
    # Every number the page prints.
    check(total == 3318, f"sum {total}")
    check(peak(dag, off) == 2049, f"peak {peak(dag, off)}")
    check(bound == 2049 and sorted(chain) == ["T", "V", "X", "eri"], f"bound {bound}")
    check(biggest == 2040, f"max {biggest}")
    check(sum(s.out for s in DUMMY) == 8062, "output sum")
    expect = {
        "eri": 0, "nevpt2": 0, "casscf": 0, "scf": 0, "metric": 0, "factor": 0,
        "X": 2040, "S": 2040, "V": 2046, "T": 2048,
    }  # fmt: skip
    check(off == expect, f"offsets {off}")
    sized = [s.id for s in DUMMY if s.scratch]
    lock = [p for p in itertools.combinations(sized, 2) if not dag.is_ordered(*p)]
    check(len(lock) == 17, f"lock pairs {len(lock)}")
    return dag, off, total, biggest, bound


def check_determinism():
    """Offsets do not depend on the order steps are declared in."""
    base = pack(Dag(DUMMY))
    best = pack_best(Dag(DUMMY))
    rng = random.Random(1)
    for _ in range(200):
        steps = list(DUMMY)
        rng.shuffle(steps)
        check(pack(Dag(steps)) == base, "offsets changed with declaration order")
        check(pack_best(Dag(steps)) == best, "pack_best changed with declaration order")
    small = DUMMY[:6]
    want = pack(Dag(small))
    for perm in itertools.permutations(small):
        check(pack(Dag(perm)) == want, "offsets changed on a permutation")


def check_limits():
    chain = [Step(f"c{k}", (f"c{k - 1}",) if k else (), 0, 10 * (k % 4 + 1))
             for k in range(8)]  # fmt: skip
    d = Dag(chain)
    check(peak(d, pack(d)) == 40, "a chain packs to its largest region")
    flat = [Step(f"f{k}", (), 0, k + 1) for k in range(8)]
    d = Dag(flat)
    check(peak(d, pack(d)) == sum(range(1, 9)), "no edges packs to the sum")


def check_closure():
    """The bitset closure against plain DFS, on random DAGs up to 60 steps."""
    rng = random.Random(7)
    for _ in range(1000):
        steps = random_dag(rng, rng.randint(1, 60), rng.uniform(0.0, 0.5))
        dag, dfs = Dag(steps), reach_dfs(steps)
        for s in steps:
            got = {dag.steps[j].id for j in _bits(dag.desc[dag.index[s.id]])}
            check(got == dfs[s.id], f"closure differs from DFS at {s.id}")
            for q in steps:
                want = q.id == s.id or q.id in dfs[s.id] or s.id in dfs[q.id]
                check(dag.is_ordered(s.id, q.id) == want, "ordered differs from DFS")


def check_random():
    """Safety and the bounds on random DAGs; how far first fit is from the bound;
    and how much of the gap a fixed set of orders closes."""
    rng = random.Random(2026_11_21)
    ratios, missed, best_ratios = [], [], []
    for _ in range(3000):
        dag = Dag(random_dag(rng, rng.randint(2, 16), rng.uniform(0.05, 0.6)))
        off = pack(dag)
        check(unsafe_pairs(dag, off) == [], "a random DAG packs unsafely")
        sizes = [s.scratch for s in dag.steps]
        pk, (bound, _) = peak(dag, off), lower_bound(dag)
        check(max(sizes) <= bound <= pk <= sum(sizes), "max <= bound <= peak <= sum")
        if not bound:
            continue
        _, boff = pack_best(dag)
        check(unsafe_pairs(dag, boff) == [], "pack_best packs unsafely")
        bpk = peak(dag, boff)
        check(bound <= bpk <= pk, "pack_best is never worse than largest first")
        ratios.append(pk / bound)
        best_ratios.append(bpk / bound)
        if pk > bound:
            # Is the bound reachable at all? Some other first-fit order says so.
            regions = [s for s in dag.steps if s.scratch > 0]
            shuffle = random.Random(0)
            best = pk
            for _ in range(500):
                shuffle.shuffle(regions)
                order_off = pack(dag, regions)
                check(unsafe_pairs(dag, order_off) == [], "a probe order is unsafe")
                best = min(best, peak(dag, order_off))
            missed.append(best == bound)
    # The seed is fixed, so these are pinned: the page quotes them.
    check(len(ratios) == 2994, f"{len(ratios)} random DAGs with scratch")
    check(sum(r == 1 for r in ratios) == 2938, "first fit at the bound")
    check(round(max(ratios), 3) == 1.281, f"worst ratio {max(ratios):.3f}")
    check(len(missed) == 56 and all(missed), "another order reaches the bound")
    check(sum(r == 1 for r in best_ratios) == 2970, "best of orders at the bound")
    check(round(max(best_ratios), 3) == 1.174, f"best worst {max(best_ratios):.3f}")
    return ratios, missed, best_ratios


def check_optimum():
    """The exact optimum on small DAGs (at most 8 regions), where first fit over
    every order is cheap: is the lower bound always reachable, and how often do
    largest first and the best of ORDERS reach the optimum? Small integer sizes,
    where ties and tight fits make gaps likelier."""
    rng = random.Random(29)
    stats = {"dags": 0, "gap": 0, "largest": 0, "best": 0}
    for _ in range(4000):
        steps = []
        for j in range(rng.randint(3, 8)):
            reads = tuple(f"n{i}" for i in range(j) if rng.random() < 0.35)
            steps.append(Step(f"n{j}", reads, 0, rng.choice((0, *range(1, 9)))))
        dag = Dag(steps)
        bound, _ = lower_bound(dag)
        if not bound:
            continue
        stats["dags"] += 1
        lpk = peak(dag, pack(dag))
        bpk = peak(dag, pack_best(dag)[1])
        opt = bound if bpk == bound else optimum(dag)
        check(bound <= opt <= bpk <= lpk, "bound <= optimum <= best <= largest")
        stats["gap"] += opt > bound
        stats["largest"] += lpk == opt
        stats["best"] += bpk == opt
    want = {"dags": 4000, "gap": 0, "largest": 3769, "best": 3953}
    check(stats == want, f"optimum study {stats}")
    return stats


def check_execution():
    """Simulated runs: with the step-end sync no two regions sharing bytes are
    in use at once; without it (the negative control) the simulation finds
    such pairs, so it is able to."""
    rng = random.Random(11)
    runs = sync_bad = nosync_runs = nosync_bad = 0
    for _ in range(600):
        dag = Dag(random_dag(rng, rng.randint(2, 24), rng.uniform(0.05, 0.6)))
        packing = pack_all(dag)
        for _ in range(5):
            runs += 1
            sync_bad += bool(simulate(dag, packing, rng, rng.randint(1, 8)))
            nosync_runs += 1
            nosync_bad += bool(simulate(dag, packing, rng, rng.randint(1, 8), False))
    dummy = Dag(DUMMY)
    dpack = pack_all(dummy)
    for _ in range(500):
        runs += 1
        sync_bad += bool(simulate(dummy, dpack, rng, rng.randint(1, 8)))
    check(sync_bad == 0, f"{sync_bad} synced runs used shared bytes at once")
    check(nosync_bad > 0, "the negative control found nothing: the simulation is blind")
    check((runs, nosync_runs, nosync_bad) == (3500, 3000, 2395), "execution counts")
    return runs, nosync_runs, nosync_bad


def check_devices():
    """Several devices and kinds: one slab each, packed independently; a
    multi-GPU step declaring on both cards; pinned staging beside device memory."""
    steps = [
        Step("p", (), 0, 100, ((0, "pinned", 50),)),  # p and q are unordered
        Step("q", (), 0, 0, ((1, "device", 80),)),  # q is on card 1 only
        Step("jk", ("p", "q"), 0, 60, ((1, "device", 60),)),  # spans both cards
        Step("r", ("jk",), 0, 30, ((0, "pinned", 20),)),
        Step("s", ("p",), 0, 40),  # unordered with q, jk and r
    ]
    dag = Dag(steps)
    packing = pack_all(dag)
    check(
        sorted(packing) == [(0, "device"), (0, "pinned"), (1, "device")],
        f"slabs {sorted(packing)}",
    )
    for key, off in packing.items():
        check(unsafe_pairs(dag, off, key) == [], f"unsafe in {key}")
    # The device-0 slab does not change when other slabs' regions change.
    bare = [Step(s.id, s.reads, s.out, s.scratch) for s in steps]
    check(
        pack(Dag(bare)) == packing[DEFAULT], "another slab's regions moved device 0's"
    )
    # Random DAGs over two devices and two kinds.
    keys = ((0, "device"), (0, "pinned"), (1, "device"), (1, "pinned"))
    rng = random.Random(13)
    slabs = 0
    for _ in range(500):
        d = Dag(random_dag(rng, rng.randint(2, 20), rng.uniform(0.05, 0.6), keys))
        pk = pack_all(d)
        for key, off in pk.items():
            slabs += 1
            check(unsafe_pairs(d, off, key) == [], f"random unsafe in {key}")
            check(lower_bound(d, key)[0] <= peak(d, off, key), "bound in a slab")
        for _ in range(3):
            check(simulate(d, pk, rng, rng.randint(1, 8)) == [], "multi-key run")
    check(slabs == 1777, f"{slabs} slabs")
    return slabs


def check_manager():
    """A manager's scratch: declared by its callers it is safe with no special
    case; owned by the manager (packed as its step's) it is not.

    eri -> scf -> casscf, and scf -> response; casscf and response both call
    eri.build_jk, and are unordered."""
    jk = 25

    def dag_with(caller_declares):
        extra = jk if caller_declares else 0
        return Dag(
            [
                Step("eri", (), 7800, 200 + (0 if caller_declares else jk)),
                Step("scf", ("eri",), 8, 40 + extra),
                Step("casscf", ("scf", "eri"), 20, 300 + extra),
                Step("response", ("scf", "eri"), 10, 150 + extra),
            ]
        )

    good = dag_with(True)
    gp = pack_all(good)
    off = gp[DEFAULT]
    check(
        off["casscf"] + 325 <= off["response"]
        or off["response"] + 175 <= off["casscf"],
        "unordered callers' regions stay apart",
    )
    rng = random.Random(17)
    for _ in range(300):
        check(simulate(good, gp, rng, rng.randint(1, 8)) == [], "caller-declared run")
    # The manager owns the scratch: it is eri's region, used during its callers.
    bad = dag_with(False)
    bp = pack_all(bad)
    uses = {"eri": ["scf", "casscf", "response"]}
    hits = sum(
        bool(simulate(bad, bp, rng, rng.randint(1, 8), uses=uses)) for _ in range(300)
    )
    check(hits == 300, f"manager-owned scratch collided in {hits} of 300 runs")
    check((peak(good, off), peak(bad, bp[DEFAULT])) == (500, 450), "manager peaks")
    return peak(good, off), peak(bad, bp[DEFAULT])


def check_branches():
    """A condition's exclusive branches: the closure calls their steps unordered
    and keeps them apart, which is safe and costs the sum; packing them as
    exclusive is also safe when only one branch runs."""
    steps = [
        Step("pre", (), 0, 10),
        Step("cond", ("pre",), 0, 0),
        Step("a1", ("cond",), 0, 100),
        Step("a2", ("a1",), 0, 50),
        Step("b1", ("cond",), 0, 120),
        Step("join", ("a2", "b1"), 0, 30),
    ]
    branches = [("cond", [["a1", "a2"], ["b1"]])]
    plain = Dag(steps)
    excl = Dag(steps, exclusive=[[["a1", "a2"], ["b1"]]])
    pp, pe = pack_all(plain), pack_all(excl)
    rng = random.Random(19)
    for _ in range(300):
        run = simulate(plain, pp, rng, rng.randint(1, 8), branches=branches)
        check(run == [], "closure packing of branches")
        run = simulate(excl, pe, rng, rng.randint(1, 8), branches=branches)
        check(run == [], "exclusive-aware packing of branches")
    check((peak(plain, pp[DEFAULT]), peak(excl, pe[DEFAULT])) == (220, 120), "peaks")
    return peak(plain, pp[DEFAULT]), peak(excl, pe[DEFAULT])


def check_replay():
    """A replayed loop body: the closure is over one iteration, computed once.

    pre -> [ geom -> (a, b unordered) -> c ] -> post, the bracket replayed. The
    offsets are packed once over one iteration. Unrolling two iterations, every
    node of the first ahead of the second's entry (a module never runs
    concurrently with itself), each step keeping its one region, must stay
    safe, statically and in simulated runs.
    """
    body = [
        Step("pre", (), 0, 50),
        Step("geom", ("pre",), 0, 5),
        Step("a", ("geom",), 0, 30),
        Step("b", ("geom",), 0, 20),
        Step("c", ("a", "b"), 0, 40),
        Step("post", ("c",), 0, 10),
    ]
    once = Dag(body)
    off = pack(once)
    check(not once.is_ordered("a", "b"), "a and b are unordered")
    check(off["a"] + 30 <= off["b"] or off["b"] + 20 <= off["a"], "a and b stay apart")
    loop = ["geom", "a", "b", "c"]
    unrolled = [Step("pre", (), 0, 50)]
    for k in (1, 2, 3):
        prev = ("pre",) if k == 1 else tuple(f"{x}#{k - 1}" for x in loop)
        for s in body[1:5]:
            reads = prev if s.id == "geom" else tuple(f"{r}#{k}" for r in s.reads)
            unrolled.append(Step(f"{s.id}#{k}", reads, 0, s.scratch))
    unrolled.append(Step("post", ("c#3",), 0, 10))
    u = Dag(unrolled)
    u_off = {s.id: off[s.id.split("#")[0]] for s in unrolled}
    check(unsafe_pairs(u, u_off) == [], "the once-packed offsets are unsafe unrolled")
    rng = random.Random(23)
    for _ in range(300):
        check(simulate(u, {DEFAULT: u_off}, rng, rng.randint(1, 8)) == [], "replay")


def main():
    check_page()
    dag, off, total, biggest, bound = check_dummy_dag()
    check_determinism()
    check_limits()
    check_closure()
    check_replay()
    ratios, missed, best_ratios = check_random()
    opt = check_optimum()
    runs, nosync_runs, nosync_bad = check_execution()
    slabs = check_devices()
    good_peak, bad_peak = check_manager()
    plain_peak, excl_peak = check_branches()
    pk = peak(dag, off)
    exact = sum(r == 1 for r in ratios)
    bexact = sum(r == 1 for r in best_ratios)
    n = len(ratios)
    print("dummy DAG (MiB)")
    print(f"  plain sum     {total:6d}")
    print(f"  packed peak   {pk:6d}")
    print(f"  lower bound   {bound:6d}")
    print(f"  lock's max    {biggest:6d}")
    print("  offsets       " + ", ".join(f"{k}@{v}" for k, v in off.items()))
    print("closure: equal to DFS reachability on 1000 random DAGs of 1 to 60 steps")
    print(f"random DAGs: {n} with scratch, all safe")
    print(f"  largest first at the bound: {exact} ({100 * exact / n:.1f}%)")
    print(f"    worst {max(ratios):.3f}, mean {sum(ratios) / n:.4f}")
    print(
        f"    above the bound: {len(missed)}; another order reaches it in {sum(missed)}"
    )
    share = 100 * bexact / n
    print(f"  best of {len(ORDERS)} orders at the bound: {bexact} ({share:.1f}%)")
    print(f"    worst {max(best_ratios):.3f}, mean {sum(best_ratios) / n:.4f}")
    print(f"exact optimum on {opt['dags']} small DAGs (at most 8 regions):")
    print(f"  above the lower bound: {opt['gap']}")
    print(f"  largest first optimal: {opt['largest']}; best of orders: {opt['best']}")
    print(f"execution: {runs} synced runs, none used shared bytes at once")
    print(f"  negative control (no sync): {nosync_bad} of {nosync_runs} runs did")
    print(f"devices and kinds: {slabs} slabs over 500 random DAGs, all safe")
    print(f"manager scratch: declared by callers safe (peak {good_peak});")
    print(f"  owned by the manager collided in every run (peak {bad_peak})")
    print(
        f"branches: closure peak {plain_peak}; exclusive-aware {excl_peak}; both safe"
    )
    print("all checks passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
