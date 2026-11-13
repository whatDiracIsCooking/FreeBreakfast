"""surface.html: the two relaxed surfaces, the diagonal cut and every grid
point's geometry on one interactive page. Point at a surface or the cut and
the third panel shows that point's converged geometry.

Built from committed files alone, no runs/ needed:

- surface_<method>.csv: each point's E, evaluations, wall clock, C2 breaking
  and natural occupations, the mirrored half included;
- surface.json: the grid and its stationary points (the grid fit's);
- geometries_<method>.xyz: each run point's final geometry (surface.py
  analyse writes it from runs/);
- step_4/irc.csv: the IRC overlaid on the PC surface;
- step_4/geometries and stationary.json: 4b's optimised stationary points,
  shown on the markers 4b located.

page.html.in is the page; the data goes in at its /*DATA*/ as one JSON
object. Geometries are rotated onto the PC TS grid point's carbon frame
(Kabsch over the six carbons) so neighbouring points line up on screen; the
xyz files keep the optimiser's own frames."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP4 = HERE.parent / "step_4"
KCAL = 627.5094740631
METHODS = ("pc", "casscf")
REFERENCE = "pc_1.90_1.90"  # the grid point nearest 4b's TS: the display frame
# 4b's optimised stationary points, by (method, kind, R to 0.1 A) of the grid
# fit's point on the diagonal. casscf_2.30.xyz is the walk that left C2h and
# fell to 1,5-hexadiene; the aromatic saddle is its restart. The off-diagonal
# CASSCF saddles and the maximum were not located in 4b.
OPTIMISED = {
    ("pc", "saddle", 1.9): "pc_ts",
    ("casscf", "minimum", 1.6): "casscf_1.65",
    ("casscf", "saddle", 2.3): "casscf_2.30_restart",
}


def frames(path: Path) -> dict[str, list]:
    """{point name: [(symbol, x, y, z), ...]} from a multi-frame xyz whose
    comment lines open with the point's name."""
    lines = path.read_text().splitlines()
    out, i = {}, 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        name = lines[i + 1].split()[0]
        atoms = []
        for ln in lines[i + 2 : i + 2 + n]:
            s, x, y, z = ln.split()[:4]
            atoms.append((s, float(x), float(y), float(z)))
        out[name] = atoms
        i += n + 2
    return out


def aligner(reference: list):
    """A function taking atoms to their coordinates rotated onto the
    reference's carbon frame, both centred on their six carbons."""
    import numpy

    def centred(atoms):
        x = numpy.array([a[1:] for a in atoms])
        return x - x[:6].mean(axis=0)

    ref = centred(reference)

    def align(atoms):
        x = centred(atoms)
        u, _, vt = numpy.linalg.svd(x[:6].T @ ref[:6])
        d = numpy.sign(numpy.linalg.det(u @ vt))
        return x @ (u @ numpy.diag([1.0, 1.0, d]) @ vt)

    return align


def flat(x) -> list[float]:
    return [round(float(v), 4) for v in x.flatten()]


def data() -> dict:
    import numpy

    summary = json.loads((HERE / "surface.json").read_text())
    geoms = {m: frames(HERE / f"geometries_{m}.xyz") for m in METHODS}
    reference = geoms["pc"][REFERENCE]
    align = aligner(reference)
    stationary = json.loads((STEP4 / "stationary.json").read_text())["runs"]
    out = {"symbols": [a[0] for a in reference], "methods": {}}
    for m in METHODS:
        g = summary["settings"]["grid"][m]
        rows = {
            (round(float(r["r16"]), 2), round(float(r["r34"]), 2)): r
            for r in csv.DictReader((HERE / f"surface_{m}.csv").open())
        }
        emin = min(float(r["energy"]) for r in rows.values())
        n = len(g)
        de = [[0.0] * n for _ in range(n)]  # de[i][j]: i over R16, j over R34
        info = [[None] * n for _ in range(n)]
        shown = {}
        for i, a in enumerate(g):
            for j, b in enumerate(g):
                r = rows[(a, b)]
                lo, hi = min(a, b), max(a, b)
                key = f"{lo:.2f}_{hi:.2f}"
                if key not in shown:
                    shown[key] = flat(align(geoms[m][f"{m}_{key}"]))
                de[i][j] = round((float(r["energy"]) - emin) * KCAL, 4)
                info[i][j] = {
                    "e": float(r["energy"]),
                    "ev": int(r["evaluations"]),
                    "w": float(r["wall_s"]),
                    "c2": float(r["c2_breaking_bohr"]),
                    "n": [float(r[f"n{k}"]) for k in range(1, 7)],
                    "src": key,
                    "mir": a > b,
                }
        points = []
        for s in summary[m]["grid_stationary_points"]:
            p = {
                "kind": s["kind"],
                "r16": s["r16"],
                "r34": s["r34"],
                "dE": round(s["dE_kcal"], 3),
            }
            stem = OPTIMISED.get((m, s["kind"], round(s["r16"], 1)))
            if stem and s["r16"] == s["r34"]:
                path = STEP4 / "geometries" / f"{stem}.xyz"
                (atoms,) = frames_plain(path)
                x = align(atoms)
                final = stationary[stem]["final"]
                p["opt"] = {
                    "file": f"step_4/geometries/{stem}.xyz",
                    "e": final["energy"],
                    "dE": round((final["energy"] - emin) * KCAL, 4),
                    "r16": float(numpy.linalg.norm(x[0] - x[5])),
                    "r34": float(numpy.linalg.norm(x[2] - x[3])),
                    "geom": flat(x),
                    "n": final["active"]["occupations"],
                }
            points.append(p)
        out["methods"][m] = {
            "grid": g,
            "emin": emin,
            "dE": de,
            "info": info,
            "geoms": shown,
            "stationary": points,
            "points_run": summary[m]["points_run"],
        }
    irc = list(csv.DictReader((STEP4 / "irc.csv").open()))
    path = sorted(
        (r for r in irc if r["side"] == "backward"), key=lambda r: float(r["s"])
    ) + sorted((r for r in irc if r["side"] == "forward"), key=lambda r: float(r["s"]))
    out["irc"] = [[round(float(r["r16"]), 4), round(float(r["r34"]), 4)] for r in path]
    return out


def frames_plain(path: Path) -> list[list]:
    """Every frame of an ordinary xyz, as [(symbol, x, y, z), ...]."""
    lines = path.read_text().splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        out.append(
            [
                (s, float(x), float(y), float(z))
                for s, x, y, z in (ln.split()[:4] for ln in lines[i + 2 : i + 2 + n])
            ]
        )
        i += n + 2
    return out


def build(path: Path) -> None:
    page = (HERE / "page.html.in").read_text()
    blob = json.dumps(data(), separators=(",", ":"))
    path.write_text(page.replace("/*DATA*/", blob, 1))
