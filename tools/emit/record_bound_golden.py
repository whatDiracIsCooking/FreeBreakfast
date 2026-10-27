#!/usr/bin/env python3
"""Write K's dressed record bounds, FelizJueves's ao.k.record_bounds, as gtest
golden data: test/gtest/data/record_bound/<molecule>.json.

free_breakfast.integrals.eri.integral_direct.exchange.pairs computes each
ordered record's bound

    I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)

on the device, through the record_bound kernels FelizJueves generates
(tools/emit/exchange_kernels.py). This writes the host reference those
kernels are held to: `ao.k.record_bounds(epair(sa, sb), pq=reference.pq.pq)`
for every ORDERED shell pair (sa, sb) of the molecule, the bound orientation-
dependent (the max runs over the first shell's Cartesians).

FelizJueves has since moved both: `ao.k` is `eri.k_exchange.k`, and `epair` is
`ao.density`'s. The files keep the names they were written with, so a run at
the new pin rewrites them unchanged but for the commit.

The shells are read from the exchange golden the K test reads,
test/gtest/data/exchange/<molecule>.json (written by
tools/extract_integral_golden.py from FelizJueves's fixture), so the bounds
are of exactly the basis the test builds. Run that tool first after bumping
deps/FelizJueves, then this.

Each file holds, per ordered shell pair, its first and second shell's index in
basis order and its records as [zeta, bound], in FelizJueves's record order
(first-shell primitive major). The test matches a record by its two shells and
its zeta, so it does not depend on either side's record order.

FelizJueves's reference Boys function needs numpy, so this runs in the `tools`
group (in the devcontainer, with UV_PROJECT_ENVIRONMENT=$PWD/.venv):

    uv run --group tools tools/emit/record_bound_golden.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FJ = ROOT / "deps/FelizJueves"
sys.path.insert(0, str(FJ / "src"))

from felizjueves.ao.density import epair  # noqa: E402
from felizjueves.ao.shell import Shell  # noqa: E402
from felizjueves.eri.k_exchange.k import record_bounds  # noqa: E402
from felizjueves.reference.pq import pq  # noqa: E402

SOURCE = ROOT / "test/gtest/data/exchange"
OUT = ROOT / "test/gtest/data/record_bound"
# H2O/cc-pVDZ reaches d, and has the records whose dressed bound is above
# mmd's undressed one; the H10 chain (five H2 4 bohr apart) has the far pairs
# whose bound decays toward zero.
MOLECULES = ("h2o_ccpvdz", "h10_chain_ccpvdz")


def felizjueves_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=FJ, check=True, capture_output=True, text=True
    ).stdout.strip()


def emit(stem: str, commit: str) -> dict:
    golden = json.loads((SOURCE / f"{stem}.json").read_text())
    shells = [
        Shell(
            s["l"], tuple(s["centre"]), tuple(s["exponents"]), tuple(s["coefficients"])
        )
        for s in golden["shells"]
    ]
    pairs = []
    n_rec = 0
    for a, sa in enumerate(shells):
        for b, sb in enumerate(shells):
            ep = epair(sa, sb)
            bounds = record_bounds(ep, pq=pq)
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "records": [
                        [zeta, bound]
                        for (_, zeta, _), bound in zip(ep.recs, bounds, strict=True)
                    ],
                }
            )
            n_rec += len(bounds)
    return {
        "description": (
            f"K's dressed record bound, ao.k.record_bounds, of every ordered shell "
            f"pair of {stem}'s basis -- written by tools/emit/record_bound_golden.py"
        ),
        "source": {
            "felizjueves": commit,
            "reference": "felizjueves.ao.k.record_bounds(epair(sa, sb), "
            "pq=felizjueves.reference.pq.pq)",
            "shells": f"test/gtest/data/exchange/{stem}.json",
        },
        "n_shells": len(shells),
        "n_records": n_rec,
        # One per ordered shell pair (a, b), shell indices in basis order; each
        # record [zeta, bound].
        "pairs": pairs,
    }


def main() -> int:
    commit = felizjueves_commit()
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in MOLECULES:
        path = OUT / f"{stem}.json"
        path.write_text(json.dumps(emit(stem, commit), indent=1) + "\n")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
