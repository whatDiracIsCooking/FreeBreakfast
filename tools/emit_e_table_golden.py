#!/usr/bin/env python3
"""Write the rolled K kernel's E tables, FelizJueves's launch_kblock.e_table, as
gtest golden data: test/gtest/data/e_table/<molecule>.json.

The exchange helpers' exchange_e_table
(src/integrals/eri/integral_direct/exchange/helpers/) writes each record's
undressed Hermite expansion E^{la lb}, its nonzero entries, on the device: the
`Eac` / `Ebd` that FelizJueves's rolled block kernel (`kbr`,
eri/k_exchange/kernel_kblock_rolled.py) reads from global memory. This writes
the host reference that launcher is held to:
`launch_kblock.e_table(la, lb, zeta, P, A, B)` of every record of every
ORDERED shell pair (sa, sb) of the molecule, the records `ao.density.epair`'s.

The E table is an intermediate with no libcint counterpart, so this is
FelizJueves checking itself; K's libcint `vk` check covers it end to end.

The shells are read from the exchange golden the K test reads,
test/gtest/data/exchange/<molecule>.json (written by
tools/extract_integral_golden.py from FelizJueves's fixture), so the tables
are of exactly the basis the test builds. Run that tool first after bumping
deps/FelizJueves, then this.

Each file holds, per ordered shell pair, its two shells' index in basis order
and momenta, and per record (FelizJueves's record order) its inputs as
[zeta, X_PA, Y_PA, Z_PA, X_PB, Y_PB, Z_PB], each P - A and P - B formed as
mmd.hermite.e_block forms them, and its unsigned table. Per class
(la, lb), the table's width and which entries are of odd |t|: the ket's table,
e_table(..., sign=True), is the unsigned one with exactly those negated, which
this asserts of every record before writing.

    uv run --group tools tools/emit_e_table_golden.py

(in the devcontainer, with UV_PROJECT_ENVIRONMENT=$PWD/.venv).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FJ = ROOT / "deps/FelizJueves"
sys.path.insert(0, str(FJ / "src"))

from felizjueves.ao.density import epair  # noqa: E402
from felizjueves.ao.shell import Shell, cartesians  # noqa: E402
from felizjueves.eri.k_exchange.launch_kblock import e_table  # noqa: E402
from felizjueves.mmd.hermite import hermite_order  # noqa: E402

SOURCE = ROOT / "test/gtest/data/exchange"
OUT = ROOT / "test/gtest/data/e_table"
# H2O/cc-pVDZ reaches d; CO/cc-pVTZ reaches f, every class through (ff).
MOLECULES = ("h2o_ccpvdz", "co_ccpvtz")


def felizjueves_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=FJ, check=True, capture_output=True, text=True
    ).stdout.strip()


def odd_entries(la: int, lb: int) -> list[int]:
    """Per entry of a record's table, 1 where its |t| is odd: e_table's
    pattern (rows a x b, the Hermite order within, structural zeros
    dropped)."""
    out = []
    for x in cartesians(la):
        for y in cartesians(lb):
            out += [
                sum(t) % 2
                for t in hermite_order(la + lb)
                if all(t[k] <= x[k] + y[k] for k in range(3))
            ]
    return out


def emit(stem: str, commit: str) -> dict:
    golden = json.loads((SOURCE / f"{stem}.json").read_text())
    shells = [
        Shell(
            s["l"], tuple(s["centre"]), tuple(s["exponents"]), tuple(s["coefficients"])
        )
        for s in golden["shells"]
    ]
    classes = {}
    pairs = []
    n_rec = 0
    for a, sa in enumerate(shells):
        for b, sb in enumerate(shells):
            ep = epair(sa, sb)
            odd = classes.setdefault((ep.la, ep.lb), odd_entries(ep.la, ep.lb))
            records = []
            tables = []
            for P, zeta, _ in ep.recs:
                unsigned = e_table(ep.la, ep.lb, zeta, P, ep.A, ep.B)
                signed = e_table(ep.la, ep.lb, zeta, P, ep.A, ep.B, sign=True)
                assert len(unsigned) == len(odd)
                assert signed == tuple(
                    -v if o else v for v, o in zip(unsigned, odd, strict=True)
                )
                records.append(
                    [zeta]
                    + [P[k] - ep.A[k] for k in range(3)]
                    + [P[k] - ep.B[k] for k in range(3)]
                )
                tables.append(list(unsigned))
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "la": ep.la,
                    "lb": ep.lb,
                    "records": records,
                    "e": tables,
                }
            )
            n_rec += len(records)
    return {
        "description": (
            f"The rolled K kernel's E table, launch_kblock.e_table, of every record "
            f"of every ordered shell pair of {stem}'s basis -- written by "
            f"tools/emit_e_table_golden.py"
        ),
        "source": {
            "felizjueves": commit,
            "reference": "felizjueves.eri.k_exchange.launch_kblock.e_table("
            "la, lb, zeta, P, A, B) of felizjueves.ao.density.epair(sa, sb).recs",
            "shells": f"test/gtest/data/exchange/{stem}.json",
        },
        "n_shells": len(shells),
        "n_records": n_rec,
        # One per class (la, lb) reached: the table's doubles per record, and
        # per entry 1 where |t| is odd (the entries the ket's table negates).
        "classes": [
            {"la": la, "lb": lb, "width": len(odd), "odd": odd}
            for (la, lb), odd in sorted(classes.items())
        ],
        # One per ordered shell pair (a, b), shell indices in basis order; each
        # record [zeta, P - A, P - B] and its unsigned table.
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
