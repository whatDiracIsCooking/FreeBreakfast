"""Step 4d: the intrinsic reaction coordinate from step 4b's PC-NEVPT2 TS.

Replaces step 4b's IRC stand-in (BFGS from the TS displaced along the
imaginary mode) with free-breakfast's [irc], Gonzalez-Schlegel GS2 in
mass-weighted Cartesians, over the same DF-PC-NEVPT2 gradient. Every run is
one free-breakfast config under inputs/<name>/, run into runs/<name>/, as
stationary.py's. Phases, each resumable:

  prepare <name>   writes inputs/<name>/ (numpy only; no PySCF)
  run <name> ...   (GPU) the binary on each prepared run, as step 6's
                   experiment.py runs it: log.txt, wall.txt, memory.json
  analyse          (PySCF, for the irrep labels) -> irc.json, irc.csv,
                   irc.svg and geometries/irc_*.xyz

The runs:

  irc              [irc] both sides from geometries/pc_ts.xyz, its Hessian
                   loaded from step 6's ts_hessian (the analytic PC-NEVPT2
                   Hessian at that geometry), the orbitals that run's CASSCF
                   converged there
  sp_<side>_<k>    a CASSCF + [nevpt2] single point at IRC point k of side
                   f(orward) or b(ackward), from the orbitals of the point
                   before it on that side (ts_hessian's at k = 0): the
                   natural occupations along the path, and a check that the
                   path stayed on one CASSCF solution (its PC energy must be
                   the IRC's at that point)
  opt_<side>       [optimize] (Opt=Tight, as step 4b) from that side's last
                   IRC point, with its last single point's orbitals
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import energies
import stationary as s4

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
STEP6 = STUDY / "step_6"
sys.path.insert(0, str(STEP6))

import experiment as s6  # noqa: E402  step 6's run(), card sampling and memory.json

INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
GEOMETRIES = HERE / "geometries"
BOHR = s4.BOHR
KCAL = s4.HARTREE_KCAL

TS_XYZ = GEOMETRIES / "pc_ts.xyz"
TS_HESSIAN = STEP6 / "runs" / "ts_hessian"

# The IRC's own options. step_size 0.1 amu^1/2 bohr is Gaussian's and the
# tree's default. The straight mass-weighted distance from the TS to step
# 4b's hexadiene end (pc_plus, Kabsch-aligned) is 12.4 amu^1/2 bohr, so a
# side reaching that basin needs well over 124 points; max_points 400 a side
# leaves the walk to stop on its own (a minimum, or the minimum within a
# step) rather than on the cap. The sphere thresholds and max_force are the
# defaults (README "Step 4d" has why).
IRC = {
    "step_size": 0.1,
    "max_points": 400,
    "max_force": 4.5e-4,
    "sphere_gradient": 1e-4,
    "sphere_displacement": 1e-3,
    "max_sphere_iterations": 20,
}

# Every SP_STRIDE-th point gets a single point, and each side's last.
SP_STRIDE = 5

HEAD = """\
# Step 4d, {name}: {what}. Written by irc.py prepare.
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
PC_GRADIENT = '[nevpt2_gradient]\nvariant = "pc"\n'


def irc_block() -> str:
    lines = ['[hessian]\nfrom = "hessian.json"\n[irc]']
    lines += [f"{k} = {v}" for k, v in IRC.items()]
    lines.append('direction = "both"')
    return "\n".join(lines) + "\n"


# --- the IRC's records ---------------------------------------------------


def irc_record() -> dict:
    return json.loads((RUNS / "irc" / "irc.json").read_text())


def side_key(side: str) -> str:
    return {"f": "forward", "b": "backward"}[side]


def irc_points(side: str) -> list[dict]:
    return irc_record().get(side_key(side), {}).get("points", [])


def point_atoms(point: dict, symbols: list[str]) -> list:
    return [
        (s, tuple(map(float, v)))
        for s, v in zip(symbols, point["geometry"], strict=True)
    ]


def ts_symbols() -> list[str]:
    (_, atoms), *_ = s4.read_xyz_frames(TS_XYZ)
    return [s for s, _ in atoms]


def sp_indices(side: str) -> list[int]:
    n = len(irc_points(side))
    ks = list(range(0, n, SP_STRIDE))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    return ks


def sp_name(side: str, k: int) -> str:
    return f"sp_{side}_{k:03d}"


def all_names() -> list[str]:
    names = ["irc"]
    if (RUNS / "irc" / "irc.json").exists():
        for side in "fb":
            if irc_points(side):
                names += [sp_name(side, k) for k in sp_indices(side)]
    return names + ["opt_f", "opt_b"]


# --- prepare -------------------------------------------------------------


def prepare(name: str) -> None:
    import numpy

    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    symbols = ts_symbols()
    if name == "irc":
        (_, atoms), *_ = s4.read_xyz_frames(TS_XYZ)
        c = numpy.load(TS_HESSIAN / "casscf_orbitals.npy")
        shutil.copyfile(TS_HESSIAN / "hessian.json", out / "hessian.json")
        what = (
            "[irc] both sides from step 4b's TS (geometries/pc_ts.xyz), step 6's "
            "ts_hessian Hessian and CASSCF orbitals"
        )
        conv = s4.CONV_TOL_GRAD_WALK
        tail = PC_GRADIENT + irc_block()
    elif name.startswith("sp_"):
        _, side, k = name.split("_")
        k = int(k)
        atoms = point_atoms(irc_points(side)[k], symbols)
        ks = sp_indices(side)
        i = ks.index(k)
        src = TS_HESSIAN if i == 0 else RUNS / sp_name(side, ks[i - 1])
        c = numpy.load(src / "casscf_orbitals.npy")
        what = (
            f"single point at {side_key(side)} IRC point {k}, "
            f"{src.name}'s CASSCF orbitals"
        )
        conv = s4.CONV_TOL_GRAD_WALK
        tail = "[nevpt2]\n"
    elif name.startswith("opt_"):
        side = name[-1]
        pts = irc_points(side)
        k = len(pts) - 1
        atoms = point_atoms(pts[k], symbols)
        c = numpy.load(RUNS / sp_name(side, k) / "casscf_orbitals.npy")
        what = (
            f"[optimize] from the {side_key(side)} IRC's last point ({k}), "
            f"{sp_name(side, k)}'s CASSCF orbitals"
        )
        conv = s4.CONV_TOL_GRAD_WALK
        tail = PC_GRADIENT + s4.OPTIMIZE.format(**s4.OPT_TIGHT, max_steps=s4.MAX_STEPS)
    else:
        raise ValueError(name)
    energies.write_xyz(out / "geometry.xyz", atoms, f"{name}: {what}")
    numpy.save(out / "orbitals.npy", numpy.asfortranarray(c), allow_pickle=False)
    text = HEAD.format(name=name, what=what, conv_tol=s4.CONV_TOL, conv_tol_grad=conv)
    (out / "config.toml").write_text(text + tail)
    r16, r34 = s4.distances(atoms)
    print(f"{name}: prepared, R16 {r16:.4f} R34 {r34:.4f} A")


# --- run -----------------------------------------------------------------


def done(name: str) -> bool:
    out = RUNS / name
    if name == "irc":
        return (out / "irc.json").exists() and (out / "wall.txt").exists()
    if name.startswith("sp_"):
        return s6.pc_energy(out) is not None
    return (out / "wall.txt").exists()


def run(names: list[str], binary: Path) -> None:
    # step 6's run() keys done() off its own RUNS_SPEC; borrow its loop with
    # this step's directories and done(). A single point or end walk reads
    # the run before it, so each is prepared just before it runs.
    s6.RUNS, s6.INPUTS = RUNS, INPUTS
    s6.done = done
    for name in names:
        if not done(name) and not (INPUTS / name / "config.toml").exists():
            prepare(name)
        s6.run([name], binary)


# --- analyse -------------------------------------------------------------


def c2_breaking(x_bohr, ops) -> dict:
    """RMS (bohr per coordinate) of the geometry's parts off C2h by irrep.
    A path along the a_u mode keeps C2 (E, C2: a_g + a_u), so its C2-breaking
    part is the b_g and b_u ones; its Ci-breaking part is a_u and b_u."""
    import math

    d = s4.deviation_by_irrep(x_bohr, ops)
    return {
        **{f"{k}_bohr": v for k, v in d.items()},
        "c2_breaking_bohr": math.hypot(d["bg"], d["bu"]),
    }


def occupations(run_dir: Path) -> list[float]:
    import numpy
    from pyscf.fci import direct_spin1

    ci = numpy.load(run_dir / "casscf_ci.npy").reshape(20, 20)
    occ = numpy.linalg.eigvalsh(direct_spin1.make_rdm1(ci, 6, (3, 3)))
    return [float(x) for x in sorted(occ, reverse=True)]


def side_rows(side: str, ops, e_ts: float) -> list[dict]:
    import numpy

    symbols = ts_symbols()
    rows = []
    for k, p in enumerate(irc_points(side)):
        atoms = point_atoms(p, symbols)
        x = s4.coords_bohr(atoms)
        r16, r34 = s4.distances(atoms)
        row = {
            "side": side_key(side),
            "k": k,
            "s": p["arc_length"],
            "energy": p["energy"],
            "dE_kcal": (p["energy"] - e_ts) * KCAL,
            "gradient_norm": p["gradient_norm"],
            "max_force": p["max_force"],
            "sphere_iterations": p.get("sphere_iterations"),
            "tangential_gradient": p.get("tangential_gradient"),
            "r16": r16,
            "r34": r34,
            **c2_breaking(x, ops),
        }
        sp = RUNS / sp_name(side, k)
        e_sp = s6.pc_energy(sp)
        if e_sp is not None:
            row["sp_energy"] = e_sp
            row["sp_minus_irc"] = e_sp - p["energy"]
            row["occupations"] = occupations(sp)
        rows.append(row)
    # the arc length and energy against the TS's, and the gradient's local
    # minima along the path (any interior one is a candidate intermediate)
    g = numpy.array([r["gradient_norm"] for r in rows])
    for k, r in enumerate(rows):
        r["gradient_local_minimum"] = bool(
            0 < k < len(rows) - 1 and g[k] < g[k - 1] and g[k] < g[k + 1]
        )
    return rows


def rmsd(a, b, masses=None, reflect=False) -> float:
    """Kabsch RMSD (Angstrom, per atom) of two geometries in the same atom
    order; `reflect` allows an improper rotation (a mirror image)."""
    import numpy

    a = numpy.asarray(a, float)
    b = numpy.asarray(b, float)
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = numpy.linalg.svd(b.T @ a)
    d = 1.0 if reflect else numpy.sign(numpy.linalg.det(u @ vt))
    r = u @ numpy.diag([1.0, 1.0, d]) @ vt
    return float(numpy.sqrt(numpy.mean(numpy.sum((b @ r - a) ** 2, axis=1))))


def canonical(atoms: list) -> list:
    """A hexadiene with C1-C6 bonded relabelled by the TS's inversion
    permutation (C1<->C4, C2<->C5, C3<->C6 and their hydrogens), so every
    end has C3-C4 bonded, as pc_plus and step 6's conformers do."""
    r16, r34 = s4.distances(atoms)
    if r34 <= r16:
        return atoms
    (_, ts), *_ = s4.read_xyz_frames(TS_XYZ)
    (perm,) = [p for k, _, p in s4.c2h_operations(s4.coords_bohr(ts)) if k == "i"]
    out = [None] * len(atoms)
    for a, b in enumerate(perm):
        out[b] = atoms[a]
    return out


def optimum(name: str) -> dict | None:
    run_dir = RUNS / name
    if not (run_dir / "wall.txt").exists() or not (run_dir / "final.xyz").exists():
        return None
    hist = json.loads((run_dir / "optimize.json").read_text())
    frames = s4.read_xyz_frames(run_dir / "final.xyz")
    (_, atoms), *_ = frames
    r16, r34 = s4.distances(atoms)
    its = hist["iterations"]
    return {
        # optimize.json records no flag: the last iteration under all four
        # Opt=Tight thresholds is what [optimize] stopped on
        "converged": bool(its)
        and all(its[-1][k] <= v for k, v in s4.OPT_TIGHT.items()),
        "evaluations": len(its),
        "energy": its[-1]["energy"] if its else None,
        "max_force": its[-1].get("max_force") if its else None,
        "r16": r16,
        "r34": r34,
        "dihedrals": [
            round(v, 2) for v in s6.dihedrals(s4.coords_bohr(canonical(atoms)))
        ],
        "atoms": atoms,
        "wall_s": s4.wall(run_dir),
    }


def compare_minima(end: dict) -> list[dict]:
    """The optimised IRC end against step 4b's two ends and step 6's
    conformers: energy difference and RMSD, proper and with a mirror."""
    refs = {}
    for n in ("pc_plus", "pc_minus"):
        refs[n] = GEOMETRIES / f"{n}.xyz"
    for p in sorted((STEP6 / "geometries").glob("conf_*.xyz")):
        refs[p.stem] = p
    s6res = json.loads((STEP6 / "results.json").read_text())
    s4res = json.loads((HERE / "stationary.json").read_text())
    out = []
    x = [v for _, v in canonical(end["atoms"])]
    for n, path in refs.items():
        (_, atoms), *_ = s4.read_xyz_frames(path)
        y = [v for _, v in canonical(atoms)]
        e = None
        if n in s4res.get("runs", {}):
            e = s4res["runs"][n].get("final", {}).get("energy")
        for c in s6res.get("conformers", []):
            if c.get("run") == n:
                e = c["energy"]
        out.append(
            {
                "reference": n,
                "energy": e,
                "dE_kcal": None if e is None else (end["energy"] - e) * KCAL,
                "rmsd": rmsd(x, y),
                "rmsd_mirror": rmsd(x, y, reflect=True),
            }
        )
    return out


def analyse() -> None:
    rec = irc_record()
    (_, ts_atoms), *_ = s4.read_xyz_frames(TS_XYZ)
    ops = s4.c2h_operations(s4.coords_bohr(ts_atoms))
    e_ts = rec["forward"]["points"][0]["energy"]
    result = {
        "settings": {
            "irc": IRC,
            "sp_stride": SP_STRIDE,
            "conv_tol": s4.CONV_TOL,
            "conv_tol_grad": s4.CONV_TOL_GRAD_WALK,
            "optimize": s4.OPT_TIGHT,
        },
        "imaginary_frequency": rec.get("imaginary_frequency"),
        "imaginary_count": rec.get("imaginary_count"),
        "ts_energy": e_ts,
        "wall_s": {n: s4.wall(RUNS / n) for n in all_names()},
        "sides": {},
    }
    rows_all = []
    for side in "fb":
        key = side_key(side)
        if not irc_points(side):
            continue
        rows = side_rows(side, ops, e_ts)
        rows_all += rows
        e = [r["energy"] for r in rows]
        rises = [k for k in range(1, len(e)) if e[k] >= e[k - 1]]
        diyl = [
            r["k"]
            for r in rows
            if abs(r["r16"] - 1.6) < 0.1 and abs(r["r34"] - 1.6) < 0.1
        ]
        interior_gmin = [r["k"] for r in rows if r["gradient_local_minimum"]]
        sp = [r for r in rows if "sp_minus_irc" in r]
        opt = optimum(f"opt_{side}")
        info = {
            "stop": rec[key].get("stop"),
            "error": rec[key].get("error"),
            "points": len(rows),
            "s_end": rows[-1]["s"],
            "energy_end": e[-1],
            "dE_end_kcal": rows[-1]["dE_kcal"],
            "r16_end": rows[-1]["r16"],
            "r34_end": rows[-1]["r34"],
            "max_force_end": rows[-1]["max_force"],
            "energy_rises": rises,
            "diyl_region_points": diyl,
            "gradient_interior_local_minima": [
                {"k": k, "s": rows[k]["s"], "gradient_norm": rows[k]["gradient_norm"]}
                for k in interior_gmin
            ],
            "min_gradient_norm_interior": min(
                (r["gradient_norm"] for r in rows[1:-1]), default=None
            ),
            "sphere_iterations": {
                "total": sum(r["sphere_iterations"] or 0 for r in rows),
                "max": max(r["sphere_iterations"] or 0 for r in rows),
                "histogram": histogram([r["sphere_iterations"] for r in rows[1:]]),
            },
            "max_c2_breaking_bohr": max(r["c2_breaking_bohr"] for r in rows),
            "max_sp_minus_irc": max((abs(r["sp_minus_irc"]) for r in sp), default=None),
        }
        if opt is not None:
            info["optimum"] = {k: v for k, v in opt.items() if k != "atoms"}
            info["optimum"]["dE_from_irc_end_kcal"] = (opt["energy"] - e[-1]) * KCAL
            info["optimum"]["vs_references"] = compare_minima(opt)
            write_frames(
                GEOMETRIES / f"irc_{key}_optimum.xyz", [opt["atoms"]], [f"opt_{side}"]
            )
        result["sides"][key] = info
        symbols = ts_symbols()
        write_frames(
            GEOMETRIES / f"irc_{key}.xyz",
            [point_atoms(p, symbols) for p in irc_points(side)],
            [f"s {p['arc_length']:.2f} E {p['energy']:.10f}" for p in irc_points(side)],
        )
    sides = result["sides"]
    if all(sides.get(k, {}).get("optimum") for k in ("forward", "backward")):
        fo = optimum("opt_f")
        bo = optimum("opt_b")
        result["ends_mirror"] = {
            "dE": fo["energy"] - bo["energy"],
            "rmsd": rmsd(
                [v for _, v in canonical(fo["atoms"])],
                [v for _, v in canonical(bo["atoms"])],
            ),
            "rmsd_mirror": rmsd(
                [v for _, v in canonical(fo["atoms"])],
                [v for _, v in canonical(bo["atoms"])],
                reflect=True,
            ),
        }
    (HERE / "irc.json").write_text(json.dumps(result, indent=1) + "\n")
    write_csv(rows_all)
    plot(rows_all)
    summarise(result, rows_all)


def histogram(xs) -> dict:
    h: dict[str, int] = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0]) if kv[0] != "None" else -1))


def write_frames(path: Path, frames: list, comments: list[str]) -> None:
    lines = []
    for atoms, c in zip(frames, comments, strict=True):
        lines += [str(len(atoms)), c]
        lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in atoms]
    path.write_text("\n".join(lines) + "\n")


CSV_FIELDS = [
    "side", "k", "s", "energy", "dE_kcal", "gradient_norm", "max_force",
    "sphere_iterations", "tangential_gradient", "r16", "r34",
    "c2_breaking_bohr", "au_bohr", "sp_minus_irc", "n1", "n2", "n3", "n4", "n5", "n6",
]  # fmt: skip


def write_csv(rows: list[dict]) -> None:
    # the backward side first, s ascending, so the file reads along the path
    ordered = sorted(rows, key=lambda r: r["s"])
    seen = set()
    with (HERE / "irc.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            if r["s"] == 0 and 0 in seen:
                continue  # the TS once
            seen.add(r["s"])
            out = dict(r)
            for i, n in enumerate(r.get("occupations", []) or []):
                out[f"n{i + 1}"] = f"{n:.6f}"
            for k in ("energy",):
                out[k] = f"{r[k]:.10f}"
            for k in ("dE_kcal", "r16", "r34"):
                out[k] = f"{r[k]:.6f}"
            w.writerow(out)


def el(tag: str, text: str | None = None, **attrs) -> str:
    """One SVG element; attribute names take '-' for '_'."""
    a = " ".join(f'{k.rstrip("_").replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag} {a}/>" if text is None else f"<{tag} {a}>{text}</{tag}>"


def plot(rows: list[dict]) -> None:
    """irc.svg: E(s) (kcal/mol from the TS) above, R16 and R34 (Angstrom)
    below, s in amu^1/2 bohr. Stdlib SVG, as step 2's plot.py."""
    rows = sorted(rows, key=lambda r: r["s"])
    s = [r["s"] for r in rows]
    width, height, left, right, top, gap, panel = 720, 560, 80, 30, 30, 50, 220
    s0, s1 = min(s), max(s)
    inner = width - left - right

    def sx(v):
        return left + (v - s0) / (s1 - s0) * inner

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="sans-serif" font-size="12">',
        el("rect", width=width, height=height, fill="white"),
    ]

    def axes(y0, lo, hi, label, ticks, series):
        def sy(v):
            return y0 + panel - (v - lo) / (hi - lo) * panel

        out.append(
            el("rect", x=left, y=y0, width=inner, height=panel, fill="none",
               stroke="black")
        )  # fmt: skip
        for t in ticks:
            y = f"{sy(t):.1f}"
            out.append(el("line", x1=left - 4, x2=left, y1=y, y2=y, stroke="black"))
            out.append(
                el("text", f"{t:g}", x=left - 8, y=f"{sy(t) + 4:.1f}",
                   text_anchor="end")
            )  # fmt: skip
        mid = y0 + panel / 2
        out.append(
            el("text", label, x=18, y=mid, transform=f"rotate(-90 18 {mid})",
               text_anchor="middle")
        )  # fmt: skip
        for key, colour, _ in series:
            pts = " ".join(f"{sx(r['s']):.1f},{sy(r[key]):.1f}" for r in rows)
            out.append(
                el("polyline", points=pts, fill="none", stroke=colour,
                   stroke_width=1.6)
            )  # fmt: skip
        for i, (_, colour, name) in enumerate(series):
            if name:
                out.append(
                    el("text", name, x=width - right - 10, y=y0 + 18 + 16 * i,
                       text_anchor="end", fill=colour)
                )  # fmt: skip

    e = [r["dE_kcal"] for r in rows]
    lo = 5 * int(min(e) // 5)
    energy = [("dE_kcal", "#d62728", "")]
    axes(top, lo, 0.0, "E - E(TS) / kcal/mol", range(lo, 1, 5), energy)
    y2 = top + panel + gap
    rmax = max(max(r["r16"] for r in rows), max(r["r34"] for r in rows))
    hi = 0.5 * (int(rmax / 0.5) + 1)
    ticks = [1.5 + 0.5 * i for i in range(int((hi - 1.5) / 0.5) + 1)]
    bonds = [("r16", "#1f77b4", "R16"), ("r34", "#2ca02c", "R34")]
    axes(y2, 1.5, hi, "R / Angstrom", ticks, bonds)
    tick = 1.0 if s1 - s0 < 30 else 5.0
    v = tick * int(s0 / tick)
    base = y2 + panel
    while v <= s1:
        if v >= s0:
            x = f"{sx(v):.1f}"
            out.append(el("line", x1=x, x2=x, y1=base, y2=base + 4, stroke="black"))
            out.append(el("text", f"{v:g}", x=x, y=base + 18, text_anchor="middle"))
        v += tick
    out.append(
        el("text", "s / amu^1/2 bohr (backward &lt; 0 &lt; forward)",
           x=left + inner / 2, y=height - 8, text_anchor="middle")
    )  # fmt: skip
    out.append("</svg>")
    (HERE / "irc.svg").write_text("\n".join(out) + "\n")


def summarise(result: dict, rows: list[dict]) -> None:
    print(f"TS E {result['ts_energy']:.10f}, imag {result['imaginary_frequency']}")
    for key, info in result["sides"].items():
        print(f"{key}:")
        for k in (
            "stop", "points", "s_end", "r16_end", "r34_end", "dE_end_kcal",
            "max_force_end", "energy_rises", "diyl_region_points",
            "gradient_interior_local_minima", "sphere_iterations",
            "max_c2_breaking_bohr", "max_sp_minus_irc",
        ):  # fmt: skip
            print(f"   {k}: {info[k]}")
        if "optimum" in info:
            o = info["optimum"]
            print(f"   optimum {o['energy']} converged {o['converged']}")
            print(f"     R16/R34 {o['r16']:.4f}/{o['r34']:.4f}")
            refs = sorted(
                o["vs_references"], key=lambda c: min(c["rmsd"], c["rmsd_mirror"])
            )
            for c in refs:
                print(f"     {c}")
    print(result.get("ends_mirror"))
    print(result["wall_s"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["prepare", "run", "analyse", "names"])
    parser.add_argument("names", nargs="*")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "src" / "free-breakfast"
    )
    args = parser.parse_args()
    pattern = re.compile(r"^(irc|sp_[fb]_\d{3}|opt_[fb])$")
    bad = [n for n in args.names if not pattern.match(n)]
    if bad:
        parser.error(f"unknown runs {bad}")
    if args.phase == "names":
        print(" ".join(all_names()))
    elif args.phase == "prepare":
        for name in args.names:
            prepare(name)
    elif args.phase == "run":
        run(args.names, args.binary)
    else:
        analyse()


if __name__ == "__main__":
    main()
