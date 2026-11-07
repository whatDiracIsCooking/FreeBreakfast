#!/usr/bin/env python3
"""Write the SC- and PC-QD-NEVPT2 nuclear Hessian references the analytic QD
Hessian (#959, #965-#967) is gated on: test/gtest/data/qd_nevpt2_hessian/.

No program has a QD-NEVPT2 Hessian, so the oracle is docs/nevpt2/nevpt2.tex's
semi-numerical one (Table qdhesschecks): a fourth-order central stencil over
the analytic QD gradient, here this tree's own `nevpt2::qd_sc_gradient` and
`qd_pc_gradient` (milestone 20), which test/gtest/data/qd_nevpt2_gradient/
ladder-validates against the tree's own energy. No other program has a QD
gradient either, and Python cannot call it, so the C++ driver
`free-breakfast-qd-nevpt2-hessian` (tools/qd_nevpt2_hessian/main.cpp) runs
the points: at every displaced geometry the chain and continuity guard of
tools/qd_nevpt2_gradient (`ChainSampler`, every point polished; the CASSCF
roots matched by their overlaps, H_eff's eigenstates by u_K's, never by
rank), then the analytic G(W) at each W this tool asks for, in the
reference's root basis:

- **the state**, W^K = u_K u_K^T, u_K the eigenvector matched to the
  reference's state K: its ladder is the whole state Hessian of E_K,
  Eq. qd-hess, roots tracked by R_P's overlap;
- **the pair sum**, W^P + W^Q, both tracked: Eq. qd-pair's left side.

This tool writes the request, runs the driver (or reads its output back with
--points), and judges every ladder; standard library only, over stencil.py.

**The Hessian.** Column j is the fourth-order central stencil over the
gradients at R0 +- s q_j, R0 +- 2 s q_j for s = h, h/2 and h/4 (eight points,
`stencil.displacements`); element (i, j) is q_i^T dG/dq_j. Per element the
ratio of successive differences must lie in RATIO_WINDOW about 16 wherever
the finer difference is above the noise floor, and below it the coarser
difference must be one a settled ladder could have (`stencil.judge`): an
element that does not settle fails the generator (STUCK, with the numbers),
never a looser bar. The bar is the finer difference, or the floor below it.
The recorded Hessian is the finest step's, symmetrized; before that it must
be symmetric to its bar, the oracle's own check that the displaced solves lie
on one surface and that the states were tracked.

**The noise floor.** sigma_g per quantity is measured, not assumed: the
reference geometry re-converged from the orbitals of NOISE_STARTS displaced
points (the +-h points of the first columns), every gradient against the
reference's own, the RMS over components and starts. A stencil at s
multiplies independent noise by stencil.noise_gain(s), so the finer
difference carries at most sigma_g (gain(h/2) + gain(h/4)); the floor is
FLOOR_FACTOR = 20 times that, as #803's, so a one-sigma noise moves a judged
ratio by 5% and the 20% window holds four.

**The guard** is the driver's (its docstring); a refused point fails the run
and nothing is written. Its worst values are recorded.

    tools/nevpt2_hessian/qd_nevpt2_hessian.py [--only CASE ...] \\
        [--binary build/tools/qd_nevpt2_hessian/free-breakfast-qd-nevpt2-hessian]

on a GPU, from a build of the tree, writes
test/gtest/data/qd_nevpt2_hessian/<case>.json. Never hand-edited. The tree's
GPU sums reassociate run to run, so a rerun reproduces every number to its
noise, not byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# isort: split
import stencil  # noqa: E402

OUT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_hessian"
SOURCE = ROOT / "test" / "gtest" / "data" / "qd_nevpt2" / "qd_nevpt2.json"
BINARY = (
    ROOT / "build" / "tools" / "qd_nevpt2_hessian" / "free-breakfast-qd-nevpt2-hessian"
)
BASIS_FILES = {"cc-pvdz": "data/basis/cc-pVDZ.json"}
AUX = ("cc-pVDZ-JKFIT", "data/aux_basis/cc-pVDZ-JKFIT.json")

# The ladder's verdicts, #803's: the window about 16 a judged ratio must lie
# in, and the noise floor's multiple of the finer difference's noise.
RATIO_WINDOW = (0.8 * stencil.TARGET, 1.2 * stencil.TARGET)
FLOOR_FACTOR = 20.0
# Every gradient's Z-vector tolerance (nevpt2::qd_*_gradient's
# zvector_tolerance; its true residual is held to it).
ZVECTOR_TOL = 3e-12
# The noise measurement: the reference geometry from the orbitals of the
# +h and -h points of this many columns, and those points and the reference
# again with the Z-vectors solved to ZVECTOR_LOOSE.
NOISE_COLUMNS = 2
ZVECTOR_LOOSE = 10 * ZVECTOR_TOL

# Every case: qd_nevpt2.json's system (`source`), the ladder's h (bohr), the
# schemes, and the pairs whose sum is differenced. The coordinates are
# Cartesian (C1).
CASES = [
    {
        "name": "h2co_skew_ccpvdz_sa3",
        "source": "h2co_skew_ccpvdz_sa3",
        "h": {"sc": 0.0003125, "pc": 0.0025},
        "pairs": [[0, 1], [0, 2], [1, 2]],
    },
    {
        "name": "h2co_skew_ccpvdz_sa3_weighted",
        "source": "h2co_skew_ccpvdz_sa3_weighted",
        "h": {"pc": 0.0025},
        "pairs": [[0, 1], [0, 2], [1, 2]],
    },
]


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def system(case: dict) -> dict:
    """qd_nevpt2.json's entry for the case."""
    doc = json.loads(SOURCE.read_text())
    (entry,) = [c for c in doc["cases"] if c["name"] == case["source"]]
    return {
        "atoms": entry["atoms"],
        "basis": entry["basis"],
        "ncas": entry["ncas"],
        "nelecas": entry["nelecas"],
        "active": None,
        "weights": entry["weights"],
    }


def coordinates(natom: int, symbols: list[str]) -> dict:
    """The Cartesian coordinates: q_j = e_j."""
    n = 3 * natom
    return {
        "labels": [
            f"{symbols[a]}{a} {'xyz'[k]}" for a in range(natom) for k in range(3)
        ],
        "vectors": [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)],
    }


def plan(h: float, coords: dict) -> tuple[list[dict], dict]:
    """The request's points at the ladder's @p h, and where each lands: index
    0 the reference; then per column and displacement; then the noise
    restarts and the looser solves."""
    disps = stencil.displacements(h)
    vectors = coords["vectors"]
    points = [{"displacement": [0.0] * len(vectors[0]), "from": None}]
    # noise: (restart, original) pairs of the same geometry.
    where = {"reference": 0, "columns": [], "noise": []}
    for q in vectors:
        row = {}
        for d in disps:
            row[d] = len(points)
            points.append({"displacement": [d * x for x in q], "from": None})
        where["columns"].append(row)
    zero = [0.0] * len(vectors[0])
    for j in range(min(NOISE_COLUMNS, len(vectors))):
        for d in (-h, h):
            where["noise"].append((len(points), 0))
            points.append({"displacement": zero, "from": where["columns"][j][d]})
    loose = [0] + [
        where["columns"][j][d]
        for j in range(min(NOISE_COLUMNS, len(vectors)))
        for d in (-h, h)
    ]
    for original in loose:
        where["noise"].append((len(points), original))
        points.append(
            {
                "displacement": points[original]["displacement"],
                "from": None,
                "zvector_tolerance": ZVECTOR_LOOSE,
            }
        )
    return points, where


def run_driver(binary: pathlib.Path, request: dict, out: pathlib.Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(request, f)
        path = f.name
    subprocess.run([str(binary), path, str(out)], check=True)
    pathlib.Path(path).unlink()


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def quantities(case: dict, n: int) -> list[tuple[str, int]]:
    """Each differenced G(W): ("state", K) per state, then ("pair", i) per
    requested pair, in the driver's order."""
    return [("state", k) for k in range(n)] + [
        ("pair", i) for i in range(len(case["pairs"]))
    ]


def floor_of(sigma: float, h: float) -> float:
    """FLOOR_FACTOR times the finer difference's noise (see the docstring)."""
    _, half, quarter = stencil.steps(h)
    return (
        FLOOR_FACTOR * sigma * (stencil.noise_gain(half) + stencil.noise_gain(quarter))
    )


def analyse(
    case: dict, coords: dict, result: dict, where: dict, schemes: list[str], h: float
) -> dict:
    """Every quantity of @p schemes over one driver run at the ladder's h."""
    name = case["name"]
    disps = stencil.displacements(h)
    vectors = coords["vectors"]
    n = len(vectors)
    nstates = len(result["reference"]["casscf_energies"])
    pts = result["points"]
    failures = []
    out = {}
    for scheme in schemes:
        part = {"states": [], "pairs": []}
        for slot, (kind, index) in enumerate(quantities(case, nstates)):

            def grad(p: int, slot: int = slot, scheme: str = scheme) -> list[float]:
                return pts[p][scheme]["gradients"][slot]

            def projected(g: list[float]) -> list[float]:
                return [dot(q, g) for q in vectors]

            g0 = grad(where["reference"])
            spread = [
                [x - y for x, y in zip(grad(a), grad(b), strict=True)]
                for a, b in where["noise"]
            ]
            sigma = math.sqrt(
                math.fsum(x * x for d in spread for x in d) / (len(spread) * len(g0))
            )
            floor = floor_of(sigma, h)
            points = [
                {d: projected(grad(where["columns"][j][d])) for d in disps}
                for j in range(n)
            ]
            columns = [stencil.column_ladder(points[j], h) for j in range(n)]
            ladder = [
                [[columns[j][k][i] for j in range(n)] for i in range(n)]
                for k in range(3)
            ]
            verdicts = [
                [
                    stencil.judge(
                        [ladder[k][i][j] for k in range(3)], floor, RATIO_WINDOW
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
            bar = [[v["bar"] for v in row] for row in verdicts]
            finest = ladder[-1]
            asym = stencil.asymmetry(finest)
            label = f"{name} {scheme} {kind} {index}"
            judged = []
            for i, row in enumerate(verdicts):
                for j, v in enumerate(row):
                    if v["settled"] is False:
                        shown = (
                            "unjudged" if v["ratio"] is None else f"{v['ratio']:.3f}"
                        )
                        failures.append(
                            f"{label}: H[{i}][{j}] ratio {shown} (ladder "
                            f"{ladder[0][i][j]:.9e} {ladder[1][i][j]:.9e} "
                            f"{ladder[2][i][j]:.9e}; D(h)-D(h/2) {v['upper']:.3e}, "
                            f"D(h/2)-D(h/4) {v['lower']:.3e}, floor {floor:.1e})"
                        )
                    elif v["ratio"] is not None:
                        judged.append(v["ratio"])
            misses = stencil.symmetry_misses(finest, bar)
            for i, j, value, allowed in misses:
                failures.append(
                    f"{label}: |H[{i}][{j}] - H[{j}][{i}]| {value:.3e} "
                    f"> bar {allowed:.3e}"
                )
            print(
                f"{label}: sigma_g {sigma:.1e}, floor {floor:.1e}, "
                f"{len(judged)} ratios judged {min(judged, default=0):.3f}.."
                f"{max(judged, default=0):.3f}, max bar {max(map(max, bar)):.2e}, "
                f"max asymmetry {max(map(max, asym)):.2e}, "
                f"{len(misses)} asymmetric pairs",
                flush=True,
            )
            entry = {
                "gradient": g0,
                "hessian": stencil.symmetrized(finest),
                "bar": stencil.symmetric_bar(bar),
                "ladder": ladder,
                "ratios": [[v["ratio"] for v in row] for row in verdicts],
                "column_bar": bar,
                "asymmetry": asym,
                "noise": {
                    "sigma_g": sigma,
                    "floor": floor,
                    "pairs": where["noise"],
                    "differences": spread,
                },
                "ratio_range": [min(judged), max(judged)] if judged else None,
                "ratios_judged": len(judged),
                "worst_ratio_deviation": max(
                    (abs(r - stencil.TARGET) for r in judged), default=None
                ),
                "points": [
                    {"column": j, "gradients": [points[j][d] for d in disps]}
                    for j in range(n)
                ],
            }
            if kind == "pair":
                entry["pair"] = case["pairs"][index]
                part["pairs"].append(entry)
            else:
                entry["state"] = index
                part["states"].append(entry)
        out[scheme] = part
    return {"quantities": out, "failures": failures}


def guard(runs: list[tuple[dict, list[str]]]) -> dict:
    """The worst guard values over every point of every run (a run is the
    driver's output and the schemes taken from it)."""
    pts = [p for result, _ in runs for p in result["points"]]
    worst = {
        "prediction": max(p["prediction"] for p in pts),
        "subspace": max(p["subspace"] for p in pts),
        "ci": max(p["ci"] for p in pts),
        "gradient_norm": max(p["gradient_norm"] for p in pts),
        "rerun_gradient_norm": max(p["rerun"]["gradient_norm"] for p in pts),
        "rerun_energy": max(p["rerun"]["energy"] for p in pts),
        "core_gap": min(p["core_gap"] for p in pts),
        "virtual_gap": min(p["virtual_gap"] for p in pts),
        "points": len(pts),
    }
    for result, schemes in runs:
        for scheme in schemes:
            run = result["points"]
            worst[f"{scheme}_vectors"] = max(p[scheme]["vector_miss"] for p in run)
            # Each at its own tolerance: the noise's looser solves apart.
            for key, tol in (
                ("zvector_residual", ZVECTOR_TOL),
                ("loose_zvector_residual", ZVECTOR_LOOSE),
            ):
                worst[f"{scheme}_{key}"] = max(
                    max(p[scheme]["zvector_residuals"])
                    for p in run
                    if p["zvector_tolerance"] == tol
                )
    return worst


def gaps(values: list[float]) -> list[float]:
    return [b - a for a, b in zip(values, values[1:], strict=False)]


def agreement(case: dict, runs: list[tuple[dict, list[str]]]) -> dict:
    """The reference point's energies against qd_nevpt2.json's goldens (the
    CASSCF roots, SC's and PC's H_eff eigenvalues): a different CASSCF
    solution misses by millihartree."""
    doc = json.loads(SOURCE.read_text())
    (entry,) = [c for c in doc["cases"] if c["name"] == case["source"]]
    (pc,) = [c for c in doc["pc"]["cases"] if c["name"] == case["source"]]
    golden = {"sc": entry["eigenvalues"], "pc": pc["eigenvalues"]}
    out = {
        "casscf": max(
            abs(a - b)
            for result, _ in runs
            for a, b in zip(
                result["reference"]["casscf_energies"],
                entry["casscf_energies"],
                strict=True,
            )
        )
    }
    for result, schemes in runs:
        for scheme in schemes:
            out[scheme] = max(
                abs(a - b)
                for a, b in zip(
                    result["reference"][scheme]["energies"], golden[scheme], strict=True
                )
            )
    return out


def record(
    case: dict,
    sys_: dict,
    coords: dict,
    runs: list[tuple[dict, list[str]]],
    quantities_: dict,
) -> dict:
    ref = runs[0][0]["reference"]
    # Each scheme's H_eff from the run that differenced it.
    heff = {
        scheme: result["reference"][scheme]
        for result, schemes in runs
        for scheme in schemes
    }
    basis_file = BASIS_FILES[sys_["basis"]]
    return {
        "description": "Semi-numerical SC- and PC-QD-NEVPT2 nuclear Hessians "
        "(DF-SA-CASSCF reference): a fourth-order central stencil at steps h, "
        "h/2, h/4 over this tree's analytic QD gradient G(W) at the tracked "
        "W^K = u_K u_K^T (each state) and W^P + W^Q (each pair), every point "
        "re-converged through the continuity guard. Hartree, bohr; "
        "H[i][j] = q_i . dG/dq_j. tools/nevpt2_hessian/qd_nevpt2_hessian.py "
        "over free-breakfast-qd-nevpt2-hessian",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_nevpt2_hessian.py",
            "driver": "tools/qd_nevpt2_hessian/main.cpp",
            "source": f"test/gtest/data/qd_nevpt2/qd_nevpt2.json {case['source']}",
        },
        "name": case["name"],
        "basis": {
            "name": sys_["basis"],
            "file": basis_file,
            "sha256": sha256(basis_file),
        },
        "aux_basis": {"name": AUX[0], "file": AUX[1], "sha256": sha256(AUX[1])},
        "atoms": sys_["atoms"],
        "ncas": sys_["ncas"],
        "nelecas": sys_["nelecas"],
        "active": sys_["active"],
        "weights": sys_["weights"],
        "schemes": list(case["h"]),
        "pairs": case["pairs"],
        "reference": {
            "casscf_energies": ref["casscf_energies"],
            "casscf_gaps": gaps(ref["casscf_energies"]),
            "gradient_norm": ref["gradient_norm"],
            "inactive_gaps": {"core": ref["core_gap"], "virtual": ref["virtual_gap"]},
            "pc_dropped": ref["pc_dropped"],
            "agreement": agreement(case, runs),
            **{
                scheme: {
                    "energies": heff[scheme]["energies"],
                    "heff_gaps": gaps(heff[scheme]["energies"]),
                    "heff": heff[scheme]["heff"],
                    "vectors": heff[scheme]["vectors"],
                }
                for scheme in case["h"]
            },
        },
        "steps": {
            scheme: {
                "h": h,
                "ladder": stencil.steps(h),
                "displacements": stencil.displacements(h),
            }
            for scheme, h in case["h"].items()
        },
        "stencil": {"offsets": list(stencil.OFFSETS), "weights": list(stencil.WEIGHTS)},
        "coordinates": coords,
        "tolerances": {
            "ratio_window": list(RATIO_WINDOW),
            "floor_factor": FLOOR_FACTOR,
            "zvector": ZVECTOR_TOL,
            "zvector_loose": ZVECTOR_LOOSE,
            "noise_columns": NOISE_COLUMNS,
        },
        "guard": guard(runs),
        **quantities_,
    }


def dumps(value, depth: int = 0) -> str:
    """JSON with objects laid out key by key down to depth 3 and every array
    on one line: a file of 30 000 numbers stays readable and a third of
    json.dumps(indent=1)'s size."""
    if isinstance(value, dict) and depth < 3 and value:
        pad = " " * (depth + 1)
        body = ",\n".join(
            f"{pad}{json.dumps(k)}: {dumps(v, depth + 1)}" for k, v in value.items()
        )
        return "{\n" + body + "\n" + " " * depth + "}"
    if isinstance(value, list) and depth < 3 and value and isinstance(value[0], dict):
        pad = " " * (depth + 1)
        body = ",\n".join(f"{pad}{dumps(v, depth + 1)}" for v in value)
        return "[\n" + body + "\n" + " " * depth + "]"
    return json.dumps(value, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT, help="directory")
    parser.add_argument("--only", nargs="+", help="these cases alone")
    parser.add_argument("--binary", type=pathlib.Path, default=BINARY)
    parser.add_argument(
        "--points",
        type=pathlib.Path,
        help="a directory of the driver's <case>.h<h>.points.json: read them "
        "back instead of running it, or, with --keep, where the run leaves them",
    )
    parser.add_argument(
        "--keep", action="store_true", help="run, and keep the points in --points"
    )
    args = parser.parse_args()
    status = 0
    for case in CASES:
        if args.only and case["name"] not in args.only:
            continue
        started = time.perf_counter()
        sys_ = system(case)
        symbols = [a[0] for a in sys_["atoms"]]
        coords = coordinates(len(sys_["atoms"]), symbols)
        # One driver run per distinct h, over the schemes that take it.
        by_h: dict[float, list[str]] = {}
        for scheme, h in case["h"].items():
            by_h.setdefault(h, []).append(scheme)
        runs = []
        quantities_ = {}
        failures = []
        for h, schemes in by_h.items():
            points, where = plan(h, coords)
            file = f"{case['name']}.h{h:g}.points.json"
            if args.points and not args.keep:
                path = args.points / file
            else:
                directory = args.points or pathlib.Path(tempfile.mkdtemp())
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / file
                request = {
                    "system": sys_,
                    "schemes": schemes,
                    "gradients": True,
                    "zvector_tolerance": ZVECTOR_TOL,
                    "pairs": case["pairs"],
                    "seams": [],
                    "points": points,
                }
                run_driver(args.binary, request, path)
            result = json.loads(path.read_text())
            assert len(result["points"]) == len(points), path
            analysis = analyse(case, coords, result, where, schemes, h)
            runs.append((result, schemes))
            quantities_.update(analysis["quantities"])
            failures += analysis["failures"]
        print(f"{case['name']}: {time.perf_counter() - started:.0f} s", flush=True)
        if failures:
            print(f"STUCK: {case['name']}", flush=True)
            for line in failures:
                print(f"  {line}", flush=True)
            status = 1
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / f"{case['name']}.json"
        out.write_text(dumps(record(case, sys_, coords, runs, quantities_)) + "\n")
        print(f"wrote {out}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
