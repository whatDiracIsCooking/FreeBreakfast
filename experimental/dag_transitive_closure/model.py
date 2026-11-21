"""Step scratch packed by the transitive closure of the step DAG: a model.

The Rust redesign (docs/rust-redesign.html, "Step scratch") packs each step's
scratch region into one slab per device and kind, letting two regions overlap
only when the DAG orders their steps. This is that packing over fixed dummy
sizes, and the checks dag_transitive_closure.html lists. Standard library only:

    uv run python experimental/dag_transitive_closure/model.py

It exits non-zero on the first failed check. The page computes the same
closure and packing in JavaScript from the same values; check_page holds the
two inputs equal and check_dummy_dag pins every number the page prints.
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
    scratch: int  # MiB, packed


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


class Dag:
    """Steps, their edges and the closure, as bitmasks over step indices."""

    def __init__(self, steps):
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

    def is_ordered(self, a, b):
        return bool(self.ordered[self.index[a]] >> self.index[b] & 1)


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def pack(dag, order=None):
    """Greedy first fit, largest first, ties by step id. Returns {id: offset}.

    The order depends on sizes and ids only, never on declaration order, so the
    offsets are the same however the steps were listed. `order`, a list of the
    scratch-declaring steps, replaces it: the probe in check_random uses it.
    """
    regions = order or sorted(
        (s for s in dag.steps if s.scratch > 0), key=lambda s: (-s.scratch, s.id)
    )
    placed = []  # (id, offset, size)
    offset = {}
    for s in regions:
        conflicts = [p for p in placed if not dag.is_ordered(s.id, p[0])]
        for o in sorted({0, *(p[1] + p[2] for p in conflicts)}):
            if all(o + s.scratch <= po or po + ps <= o for _, po, ps in conflicts):
                break
        offset[s.id] = o
        placed.append((s.id, o, s.scratch))
    return offset


def peak(dag, offset):
    return max(
        (offset[s.id] + s.scratch for s in dag.steps if s.id in offset), default=0
    )


def lower_bound(dag):
    """The heaviest antichain of scratch-declaring steps: no packing that keeps
    unordered regions apart can go below it. Exact, by branch and bound."""
    ids = [i for i, s in enumerate(dag.steps) if s.scratch > 0]
    w = [dag.steps[i].scratch for i in ids]
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


def unsafe_pairs(dag, offset):
    """Pairs of steps whose regions intersect and which the DAG does not order."""
    sized = [s for s in dag.steps if s.id in offset]
    bad = []
    for a, b in itertools.combinations(sized, 2):
        oa, ob = offset[a.id], offset[b.id]
        intersect = oa < ob + b.scratch and ob < oa + a.scratch
        if intersect and not dag.is_ordered(a.id, b.id):
            bad.append((a.id, b.id))
    return bad


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
    rng = random.Random(1)
    for _ in range(200):
        steps = list(DUMMY)
        rng.shuffle(steps)
        check(pack(Dag(steps)) == base, "offsets changed with declaration order")
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


def random_dag(rng, n, p):
    steps = []
    for j in range(n):
        reads = tuple(f"n{i}" for i in range(j) if rng.random() < p)
        size = 0 if rng.random() < 0.15 else round(2 ** rng.uniform(0, 11))
        steps.append(Step(f"n{j}", reads, 0, size))
    rng.shuffle(steps)  # declaration order is not topological
    return steps


def check_random():
    """Safety and the bounds on random DAGs; how far first fit is from the bound."""
    rng = random.Random(2026_11_21)
    ratios, missed = [], []
    for _ in range(3000):
        dag = Dag(random_dag(rng, rng.randint(2, 16), rng.uniform(0.05, 0.6)))
        off = pack(dag)
        check(unsafe_pairs(dag, off) == [], "a random DAG packs unsafely")
        sizes = [s.scratch for s in dag.steps]
        pk, (bound, _) = peak(dag, off), lower_bound(dag)
        check(max(sizes) <= bound <= pk <= sum(sizes), "max <= bound <= peak <= sum")
        if not bound:
            continue
        ratios.append(pk / bound)
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
    return ratios, missed


def check_replay():
    """A replayed loop body: the closure is over one iteration, computed once.

    pre -> [ geom -> (a, b unordered) -> c ] -> post, the bracket replayed. The
    offsets are packed once over one iteration. Unrolling two iterations, every
    node of the first ahead of the second's entry (a module never runs
    concurrently with itself), each step keeping its one region, must stay
    safe: a step meets its own region again only across iterations, which are
    ordered.
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
    for k in (1, 2):
        prev = ("pre",) if k == 1 else tuple(f"{x}#1" for x in loop)
        for s in body[1:5]:
            reads = prev if s.id == "geom" else tuple(f"{r}#{k}" for r in s.reads)
            unrolled.append(Step(f"{s.id}#{k}", reads, 0, s.scratch))
    unrolled.append(Step("post", ("c#2",), 0, 10))
    u = Dag(unrolled)
    u_off = {s.id: off[s.id.split("#")[0]] for s in unrolled}
    check(unsafe_pairs(u, u_off) == [], "the once-packed offsets are unsafe unrolled")


def main():
    check_page()
    dag, off, total, biggest, bound = check_dummy_dag()
    check_determinism()
    check_limits()
    check_replay()
    ratios, missed = check_random()
    pk = peak(dag, off)
    exact = sum(r == 1 for r in ratios)
    print("dummy DAG (MiB)")
    print(f"  plain sum     {total:6d}")
    print(f"  packed peak   {pk:6d}")
    print(f"  lower bound   {bound:6d}")
    print(f"  lock's max    {biggest:6d}")
    print("  offsets       " + ", ".join(f"{k}@{v}" for k, v in off.items()))
    print(f"random DAGs: {len(ratios)} with scratch, all safe")
    print(f"  first fit at the lower bound: {exact} ({100 * exact / len(ratios):.1f}%)")
    print(f"  worst peak / bound: {max(ratios):.3f}")
    print(f"  mean peak / bound:  {sum(ratios) / len(ratios):.4f}")
    print(
        f"  above the bound: {len(missed)}; another first-fit order reaches it"
        f" in {sum(missed)}"
    )
    print("all checks passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
