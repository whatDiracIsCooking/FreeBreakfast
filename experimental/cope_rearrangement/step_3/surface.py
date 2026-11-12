"""Step 3: the relaxed 2D (R16, R34) surface of the chair Cope, at
DF-PC-NEVPT2 and, for contrast, DF-CASSCF(6,6), both cc-pVDZ / cc-pVDZ-JKFIT.

Each grid point is an [optimize] with `constraints` holding R16 (atoms 0-5)
and R34 (atoms 2-3) at the point's values while everything else relaxes
(#1260). The surface is symmetric under R16 <-> R34 (the chair's inversion
swaps the two bonds), so only R16 <= R34 is run and the rest is its mirror.

Sweep order, each point from a converged neighbour's geometry and orbitals:
the diagonal outward from the grid point nearest step 4b's TS (started from
geometries/pc_ts.xyz with step 6's ts_hessian orbitals), then from each
diagonal point (R, R) along its row, (R, R + h), (R, R + 2h), ... Every
start is C2h on the diagonal and C2 off it, and the walks keep that
symmetry: the gradient of a C2 geometry is C2-symmetric, and both
constraints map to themselves under C2.

Phases, as step 4b's stationary.py and step 6's experiment.py:

  run <method> [names]  (GPU) prepare and run each point in sweep order, or
                        the named ones; skips one done
  names <method>        the sweep order
  analyse               (numpy) -> surface_<method>.csv, surface.json,
                        surface_<method>.svg, comparison.svg
  comparison            (stdlib) comparison.svg from the committed
                        surface_<method>.csv and surface.json alone

`method` is pc or casscf.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
STEP4 = STUDY / "step_4"
STEP6 = STUDY / "step_6"
sys.path.insert(0, str(STEP4))
sys.path.insert(0, str(STEP6))

import energies  # noqa: E402
import experiment as s6  # noqa: E402  the run loop: log, wall clock, card memory
import stationary as s4  # noqa: E402

INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
BOHR = s4.BOHR
KCAL = s4.HARTREE_KCAL

# The grid, Angstrom (README "Step 3" has how it was sized).
GRID = {
    "pc": [round(1.5 + 0.05 * i, 2) for i in range(27)],  # 1.50 .. 2.80
    "casscf": [round(1.5 + 0.1 * i, 2) for i in range(14)],  # 1.5 .. 2.8
}
TS_XYZ = STEP4 / "geometries" / "pc_ts.xyz"
TS_ORBITALS = STEP6 / "runs" / "ts_hessian" / "casscf_orbitals.npy"

# Gaussian's default Opt thresholds (hartree/bohr, bohr): the surface is read
# at ~0.1 kcal/mol, and a geometry this converged is within ~g^2/k ~ 1e-7 Eh
# of its constrained minimum along any mode stiffer than ~0.1 Eh/bohr^2.
OPT = {
    "max_force": 4.5e-4,
    "rms_force": 3.0e-4,
    "max_displacement": 1.8e-3,
    "rms_displacement": 1.2e-3,
}
MAX_STEPS = 200

HEAD = """\
# Step 3, {name}: {what}. Written by surface.py.
[molecule]
xyz = "geometry.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 6
nelecas = 6
orbitals = "orbitals.npy"
conv_tol = {conv_tol}
conv_tol_grad = {conv_tol_grad}
write_npy = true              # casscf_orbitals.npy and casscf_ci.npy
"""
GRADIENT = {
    "pc": '[nevpt2_gradient]\nvariant = "pc"\n',
    "casscf": "[casscf_gradient]\n",
}


def name_of(method: str, r16: float, r34: float) -> str:
    return f"{method}_{r16:.2f}_{r34:.2f}"


def parse(name: str) -> tuple[str, float, float]:
    method, a, b = name.split("_")
    return method, float(a), float(b)


def grid(method: str) -> list[float]:
    return GRID[method]


def seed(method: str) -> float:
    """The diagonal grid value nearest the PC TS's R (1.9223 A)."""
    return min(grid(method), key=lambda r: abs(r - 1.9223))


def sweep(method: str) -> list[str]:
    g = grid(method)
    r0 = seed(method)
    i0 = g.index(r0)
    diag = [g[i0]]
    for d in range(1, len(g)):
        if i0 + d < len(g):
            diag.append(g[i0 + d])
        if i0 - d >= 0:
            diag.append(g[i0 - d])
    names = [name_of(method, r, r) for r in diag]
    for r in sorted(diag):
        i = g.index(r)
        names += [name_of(method, r, g[j]) for j in range(i + 1, len(g))]
    return names


def parent(name: str) -> str | None:
    method, a, b = parse(name)
    g = grid(method)
    if a == b:
        if a == seed(method):
            return None
        i = g.index(a)
        i0 = g.index(seed(method))
        step = -1 if i > i0 else 1
        r = g[i + step]
        return name_of(method, r, r)
    j = g.index(b)
    return name_of(method, a, g[j - 1])


def start_from(name: str) -> str | None:
    """The nearest converged ancestor (parent, else its parent, ...): a point
    whose neighbour failed still starts from a converged geometry, one or
    more grid steps away, rather than the sweep losing the rest of the row."""
    p = parent(name)
    while p is not None and not converged(p):
        p = parent(p)
    return p


def prepare(name: str) -> None:
    import numpy

    method, a, b = parse(name)
    p = start_from(name)
    if p is None:
        (_, atoms), *_ = s4.read_xyz_frames(TS_XYZ)
        c = numpy.load(TS_ORBITALS)
        what = (
            "step 4b's PC TS (step_4/geometries/pc_ts.xyz), "
            "step 6 ts_hessian's orbitals"
        )
    else:
        (_, atoms), *_ = s4.read_xyz_frames(RUNS / p / "final.xyz")
        c = numpy.load(RUNS / p / "casscf_orbitals.npy")
        what = f"{p}'s final geometry and CASSCF orbitals"
    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    energies.write_xyz(out / "geometry.xyz", atoms, f"{name}: {what}")
    numpy.save(out / "orbitals.npy", numpy.asfortranarray(c), allow_pickle=False)
    text = HEAD.format(
        name=name,
        what=f"{method} R16 {a:.2f} R34 {b:.2f} from {what}",
        conv_tol=s4.CONV_TOL,
        conv_tol_grad=s4.CONV_TOL_GRAD_WALK,
    )
    text += GRADIENT[method]
    text += "[optimize]                    # Gaussian's default Opt thresholds\n"
    text += "".join(f"{k} = {v}\n" for k, v in OPT.items())
    text += f"max_steps = {MAX_STEPS}\n"
    text += (
        f"constraints = [{{ atoms = [0, 5], distance = {a:.2f} }}, "
        f"{{ atoms = [2, 3], distance = {b:.2f} }}]\n"
    )
    (out / "config.toml").write_text(text)


def done(name: str) -> bool:
    return (RUNS / name / "wall.txt").exists()


def converged(name: str) -> bool:
    """optimize.json records no flag: the last iteration under all four
    thresholds, with the run's final.xyz written, is what [optimize] stopped
    on."""
    p = RUNS / name / "optimize.json"
    if not p.exists() or not (RUNS / name / "final.xyz").exists():
        return False
    its = json.loads(p.read_text())["iterations"]
    return bool(its) and all(its[-1][k] <= v for k, v in OPT.items())


def run(names: list[str], binary: Path) -> None:
    s6.RUNS, s6.INPUTS = RUNS, INPUTS
    s6.done = done
    for name in names:
        if done(name):
            continue
        p = parent(name)
        if p is not None and not converged(p):
            print(
                f"{name}: its parent {p} did not converge; starting from "
                f"{start_from(name)}",
                flush=True,
            )
        prepare(name)
        s6.run([name], binary)


# --- analyse -------------------------------------------------------------


def point(name: str) -> dict:
    method, a, b = parse(name)
    row = {"method": method, "r16": a, "r34": b, "name": name}
    d = RUNS / name
    if not (d / "optimize.json").exists():
        row["status"] = "not run" if not done(name) else "failed"
        if (d / "log.txt").exists():
            tail = [
                ln
                for ln in (d / "log.txt").read_text().splitlines()
                if "error" in ln.lower()
            ]
            row["error"] = tail[-1][-300:] if tail else None
        return row
    h = json.loads((d / "optimize.json").read_text())
    its = h["iterations"]
    last = its[-1]
    row.update(
        {
            "status": "converged" if converged(name) else "not converged",
            "energy": last["energy"],
            "evaluations": len(its),
            "max_force": last.get("max_force"),
            "wall_s": s4.wall(d),
        }
    )
    cons = h.get("constraints") or last.get("constraints") or []
    row["multipliers"] = [c.get("multiplier") for c in cons]
    if (d / "final.xyz").exists():
        (_, atoms), *_ = s4.read_xyz_frames(d / "final.xyz")
        r16, r34 = s4.distances(atoms)
        row["r16_final"], row["r34_final"] = r16, r34
        x = s4.coords_bohr(atoms)
        row["c2_breaking_bohr"] = c2_breaking(x, atoms)
    if (d / "casscf_ci.npy").exists():
        row["occupations"] = occupations(d)
    return row


def c2_permutation():
    """The TS's C2 as an atom permutation (C1<->C6, C2<->C5, C3<->C4 and
    their hydrogens): perm[a] is the atom C2 takes atom a onto."""
    (_, atoms), *_ = s4.read_xyz_frames(TS_XYZ)
    ops = s4.c2h_operations(s4.coords_bohr(atoms))
    (perm,) = [p for k, _, p in ops if k == "C2"]
    return perm


def c2_breaking(x, atoms) -> float:
    """Half the Kabsch RMSD (bohr, per atom) between the geometry and its
    atoms relabelled by the C2 permutation, a proper rotation allowed: zero
    for a C2 geometry in any frame, and the size of its C2-odd part
    otherwise."""
    import numpy

    perm = c2_permutation()
    y = numpy.empty_like(x)
    y[perm] = x  # atom a's position goes to its image's label
    a = x - x.mean(0)
    b = y - y.mean(0)
    u, _, vt = numpy.linalg.svd(b.T @ a)
    d = numpy.sign(numpy.linalg.det(u @ vt))
    r = u @ numpy.diag([1.0, 1.0, d]) @ vt
    return float(numpy.sqrt(numpy.mean(numpy.sum((b @ r - a) ** 2, axis=1))) / 2)


def occupations(d: Path) -> list[float] | None:
    import numpy

    try:
        from pyscf.fci import direct_spin1
    except ImportError:
        return None
    ci = numpy.load(d / "casscf_ci.npy").reshape(20, 20)
    occ = numpy.linalg.eigvalsh(direct_spin1.make_rdm1(ci, 6, (3, 3)))
    return [float(v) for v in sorted(occ, reverse=True)]


def analyse() -> None:
    result = {"settings": {"grid": GRID, "optimize": OPT, "max_steps": MAX_STEPS}}
    for method in GRID:
        names = sweep(method)
        rows = [point(n) for n in names]
        if not any("energy" in r for r in rows):
            continue
        result[method] = summarise(method, rows)
        write_csv(method, rows)
        import contour

        contour.plot(
            HERE / f"surface_{method}.svg",
            method,
            full_grid(method, rows),
            contour.irc_path(STEP4 / "irc.csv"),
            result[method],
        )
    (HERE / "surface.json").write_text(json.dumps(result, indent=1) + "\n")
    comparison()


def comparison() -> None:
    """comparison.svg, from what analyse wrote: each method's grid from its
    surface_<method>.csv (energies to 1e-10 Eh, the mirrored half included)
    and its stationary points from surface.json."""
    import contour

    summary = json.loads((HERE / "surface.json").read_text())
    surfaces = {}
    for method in GRID:
        rows = csv.DictReader((HERE / f"surface_{method}.csv").open())
        e = {
            (float(r["r16"]), float(r["r34"])): float(r["energy"])
            for r in rows
            if r["status"] == "converged"
        }
        surfaces[method] = {"grid": grid(method), "energy": e}
    contour.comparison(
        HERE / "comparison.svg", surfaces, summary, contour.irc_path(STEP4 / "irc.csv")
    )


def full_grid(method: str, rows: list[dict]) -> dict:
    """{(r16, r34): energy} over the whole square, the half run mirrored."""
    e = {}
    for r in rows:
        if "energy" in r and r["status"] == "converged":
            e[(r["r16"], r["r34"])] = r["energy"]
            e[(r["r34"], r["r16"])] = r["energy"]
    return {"grid": grid(method), "energy": e}


def stationary_points(g: list[float], e: dict) -> list[dict]:
    """The grid surface's stationary points, from a quadratic fitted by least
    squares to each interior point's 3 x 3 patch: where the fit's own
    stationary point falls inside the central cell (within h/2 of it in both
    R), it is kept, located and classified by the fit's Hessian (minimum,
    saddle or maximum). Points within 0.6 h of one already found are the
    same one seen from a neighbouring patch, and the lower |gradient| at the
    patch centre wins. Over the full square, so off-diagonal saddles are
    found with their mirrors."""
    import numpy

    h = g[1] - g[0]
    emin = min(e.values())
    offsets = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    design = numpy.array([[1, i, j, i * i, i * j, j * j] for i, j in offsets], float)
    found = []
    for (a, b), v in e.items():
        keys = [(round(a + i * h, 2), round(b + j * h, 2)) for i, j in offsets]
        if not all(k in e for k in keys):
            continue
        z = numpy.array([e[k] for k in keys])
        c, *_ = numpy.linalg.lstsq(design, z, rcond=None)
        hess = numpy.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
        try:
            x = numpy.linalg.solve(hess, -c[1:3])
        except numpy.linalg.LinAlgError:
            continue
        if abs(x[0]) > 0.5 or abs(x[1]) > 0.5:
            continue
        w = numpy.linalg.eigvalsh(hess) / h**2
        kind = "minimum" if w[0] > 0 else "maximum" if w[1] < 0 else "saddle"
        fit = c[0] + c[1:3] @ x + 0.5 * x @ hess @ x
        found.append(
            {
                "kind": kind,
                "r16": round(a + x[0] * h, 3),
                "r34": round(b + x[1] * h, 3),
                "dE_kcal": (fit - emin) * KCAL,
                "curvatures_kcal_per_A2": [float(v) * KCAL for v in w],
                "patch_centre": [a, b],
                "offset_in_cells": [float(x[0]), float(x[1])],
            }
        )
    found.sort(key=lambda s: max(abs(v) for v in s["offset_in_cells"]))
    kept = []
    for s in found:
        if all(
            abs(s["r16"] - k["r16"]) > 0.6 * h or abs(s["r34"] - k["r34"]) > 0.6 * h
            for k in kept
        ):
            kept.append(s)
    return sorted(kept, key=lambda s: (s["r16"], s["r34"]))


def summarise(method: str, rows: list[dict]) -> dict:
    """The grid's run status, its stationary points (stationary_points) and
    the diagonal's energies."""
    g = grid(method)
    e = full_grid(method, rows)["energy"]
    emin = min(e.values())
    found = stationary_points(g, e)
    status = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    return {
        "points_run": len(rows),
        "status": status,
        "not_converged": [
            {
                k: r.get(k)
                for k in ("name", "status", "evaluations", "max_force", "error")
            }
            for r in rows
            if r["status"] != "converged"
        ],
        "e_min": emin,
        "grid_stationary_points": found,
        "diagonal": [
            {"r": a, "energy": e[(a, a)], "dE_kcal": (e[(a, a)] - emin) * KCAL}
            for a in g
            if (a, a) in e
        ],
        "wall_s_total": sum(r.get("wall_s") or 0 for r in rows),
        "evaluations_total": sum(r.get("evaluations") or 0 for r in rows),
    }


def write_csv(method: str, rows: list[dict]) -> None:
    fields = ["r16", "r34", "status", "energy", "dE_kcal", "evaluations", "max_force",
              "wall_s", "r16_final", "r34_final", "c2_breaking_bohr",
              "n1", "n2", "n3", "n4", "n5", "n6", "mirrored"]  # fmt: skip
    es = [r["energy"] for r in rows if "energy" in r]
    emin = min(es)
    out = []
    for r in rows:
        row = dict(r)
        if "energy" in r:
            row["dE_kcal"] = f"{(r['energy'] - emin) * KCAL:.4f}"
            row["energy"] = f"{r['energy']:.10f}"
        for i, n in enumerate(r.get("occupations") or []):
            row[f"n{i + 1}"] = f"{n:.5f}"
        row["mirrored"] = 0
        out.append(row)
        if r["r16"] != r["r34"]:
            m = dict(row)
            m["r16"], m["r34"] = r["r34"], r["r16"]
            if "r16_final" in r:
                m["r16_final"], m["r34_final"] = r["r34_final"], r["r16_final"]
            m["mirrored"] = 1
            out.append(m)
    out.sort(key=lambda r: (r["r16"], r["r34"]))
    with (HERE / f"surface_{method}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["run", "names", "analyse", "comparison"])
    parser.add_argument("method", nargs="?", choices=list(GRID))
    parser.add_argument("names", nargs="*")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "src" / "free-breakfast"
    )
    args = parser.parse_args()
    if args.phase == "names":
        print(" ".join(sweep(args.method)))
    elif args.phase == "run":
        run(args.names or sweep(args.method), args.binary)
    elif args.phase == "comparison":
        comparison()
    else:
        analyse()


if __name__ == "__main__":
    main()
