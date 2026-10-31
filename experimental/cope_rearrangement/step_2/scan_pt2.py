#!/usr/bin/env python3
"""Step 2a: PC- and SC-NEVPT2 along step 1's relaxed C2h CASSCF scan.

At every step-1 geometry (R = R16 = R34, 1.50 to 2.60 A), DF-CASSCF(6,6) is
reconverged in C1 from step 1's orbitals (pt2.py says why C1), checked
against step 1's C2h energy, and PC-NEVPT2 (block2) and SC-NEVPT2 (PySCF)
are run on it. The geometries are step 1's: CASSCF-relaxed, not PT2-relaxed
(that is step 4).

Writes, beside this script:
  pt2_points/R<r>.json   one record per R
  pt2_scan.csv           every record, in R order

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        experimental/cope_rearrangement/step_2/scan_pt2.py [--threads N] [R ...]

A point whose record exists is skipped, so a killed run resumes.
`--collect` only rebuilds pt2_scan.csv.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STEP1 = HERE.parent / "step_1"
sys.path.insert(0, str(HERE))

import pt2  # noqa: E402

BOHR = 0.529177210903  # A
GRID = [round(1.50 + 0.05 * k, 2) for k in range(23)]
OUT = HERE / "pt2_points"
# The C1 CASSCF must land on step 1's C2h solution.
SAME_ENERGY = 1e-7


def tag(r: float) -> str:
    return f"R{r:.2f}"


def load_step1(r):
    rec = json.loads((STEP1 / "points" / f"{tag(r)}.json").read_text())
    atoms = [(s, np.array(x) / BOHR) for s, x in rec["geometry"]]
    mo = np.load(STEP1 / "orbitals" / f"{tag(r)}.npy")
    return rec, atoms, mo


def point(r):
    rec, atoms, mo = load_step1(r)
    t0 = time.perf_counter()
    mc = pt2.casscf(pt2.make_mol(atoms), mo)
    t1 = time.perf_counter()
    d = mc.e_tot - rec["e_casscf"]
    assert abs(d) < SAME_ENERGY, f"{tag(r)}: C1 CASSCF is {d:.2e} Eh off step 1's"
    res = pt2.nevpt2(mc)
    out = {
        "r": r,
        "e_casscf": float(mc.e_tot),
        "e_casscf_step1": rec["e_casscf"],
        "e_pc": float(mc.e_tot) + res["e_corr_pc"],
        "e_sc": float(mc.e_tot) + res["e_corr_sc"],
        "sc": "natorb",
        **res,
        "seconds_casscf": t1 - t0,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag(r)}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        f"[{tag(r)}] CASSCF {mc.e_tot:.10f}  PC {out['e_pc']:.10f}  "
        f"SC {out['e_sc']:.10f}  ({t1 - t0:.0f}s + {res['seconds_pc']:.0f}s "
        f"+ {res['seconds_sc']:.0f}s)",
        flush=True,
    )


def refresh_sc(r):
    """Recompute only the SC energy (on natural orbitals) of a stored point."""
    f = OUT / f"{tag(r)}.json"
    out = json.loads(f.read_text())
    if out.get("sc") == "natorb":
        return
    _rec, atoms, mo = load_step1(r)
    mc = pt2.casscf(pt2.make_mol(atoms), mo)
    assert abs(mc.e_tot - out["e_casscf"]) < SAME_ENERGY
    res = pt2.nevpt2(mc, pc=False)
    out["e_corr_sc"] = res["e_corr_sc"]
    out["e_sc"] = out["e_casscf"] + res["e_corr_sc"]
    out["seconds_sc"] = res["seconds_sc"]
    out["sc"] = "natorb"
    f.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[{tag(r)}] SC (natorb) {out['e_sc']:.10f}", flush=True)


def collect():
    recs = [
        json.loads((OUT / f"{tag(r)}.json").read_text())
        for r in GRID
        if (OUT / f"{tag(r)}.json").exists()
    ]
    if not recs:
        return
    kcal = 627.509474
    ref = {k: min(x[k] for x in recs) for k in ("e_casscf", "e_pc", "e_sc")}
    lines = ["R,E_casscf,E_pc,E_sc,dE_casscf,dE_pc,dE_sc"]
    for x in recs:
        lines.append(
            f"{x['r']:.2f},{x['e_casscf']:.10f},{x['e_pc']:.10f},{x['e_sc']:.10f},"
            + ",".join(f"{(x[k] - ref[k]) * kcal:.3f}" for k in ref)
        )
    (HERE / "pt2_scan.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument(
        "--refresh-sc",
        action="store_true",
        help="recompute SC on natural orbitals for points stored without it",
    )
    parser.add_argument("r", nargs="*", type=float)
    args = parser.parse_args()

    from pyscf import lib

    if args.threads:
        lib.num_threads(args.threads)
    if args.refresh_sc:
        for r in args.r or GRID:
            refresh_sc(r)
    elif not args.collect:
        for r in args.r or GRID:
            if not (OUT / f"{tag(r)}.json").exists():
                point(r)
    collect()


if __name__ == "__main__":
    main()
