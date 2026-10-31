#!/usr/bin/env python3
"""Step 1: relaxed C2h DF-CASSCF(6,6)/cc-pVDZ scan of R = R16 = R34.

R runs from 1.50 to 2.60 A in 0.05 A steps. At each R, every other
coordinate is relaxed under C2h (PySCF symmetrises each geomeTRIC step;
geomeTRIC holds R16 and R34 with `$set distance` constraints).

1. Converge at R = 1.95 A first, from setup.py's B3LYP geometry and CASSCF
   orbitals.
2. Sweep outward in both directions (two processes), each point starting
   from its inner neighbour's converged geometry, orbitals and CI vector:
   one CASSCF gradient scanner per sweep carries them over.
3. At each point record the energy and the active natural occupations per
   irrep (the a_u/b_g pair is the diradical one), plus the checks that the
   active space stayed itself: the active irreps (C2h CASSCF cannot rotate
   across irreps, so this guards the start only) and the singular values of
   the active-space overlap with the neighbour across the two geometries.

Writes, beside this script:
  points/R<r>.json      one record per R (energy, R16/R34 actually reached,
                        occupations, overlap with neighbour, geometry)
  orbitals/R<r>.npy     its converged MO coefficients, for step 2
  scan.json, scan.csv   every record, in R order, and the jump check

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        experimental/cope_rearrangement/step_1/scan.py [--threads N]

`--sweep down|up` runs one direction alone (scan.py launches both itself);
`--collect` only rebuilds scan.json/scan.csv from points/.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cope  # noqa: E402

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-jkfit"
R_CENTER = 1.95
R_DOWN = [round(1.90 - 0.05 * k, 2) for k in range(9)]  # 1.90 .. 1.50
R_UP = [round(2.00 + 0.05 * k, 2) for k in range(13)]  # 2.00 .. 2.60

POINTS = HERE / "points"
ORBITALS = HERE / "orbitals"

# geomeTRIC's defaults, but 1e-7 Eh on the energy: neighbours differ by
# ~0.1-1 mEh, and the scan's shape is read off those differences.
CONV = {
    "convergence_energy": 1e-7,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}


def tag(r: float) -> str:
    return f"R{r:.2f}"


def make_mol(atoms):
    from pyscf import gto

    mol = gto.M(atom=cope.atom_string(atoms), basis=BASIS, symmetry=True, verbose=3)
    assert mol.topgroup == "C2h", mol.topgroup
    return mol


def make_scanner(mol, mo, ci=None):
    from pyscf import mcscf, scf

    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-10
    mc = mcscf.DFCASSCF(mf, cope.NCAS, cope.NELECAS)
    mc.fcisolver.wfnsym = "Ag"
    mc.fix_spin_(ss=0)
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-5
    mc.mo_coeff = mo
    mc.ci = ci
    scanner = mc.nuc_grad_method().as_scanner()
    return scanner


def load_point(r):
    rec = json.loads((POINTS / f"{tag(r)}.json").read_text())
    atoms = [(s, np.array(xyz)) for s, xyz in rec["geometry"]]
    mo = np.load(ORBITALS / f"{tag(r)}.npy")
    return rec, atoms, mo


def relax(scanner, r, neighbour=None):
    """Relax everything but R16 = R34 = r under C2h, starting from the
    scanner's current geometry, orbitals and CI; return the record."""
    from pyscf.geomopt import geometric_solver

    constraints = HERE / "logs" / f"constraints_{tag(r)}.txt"
    constraints.parent.mkdir(exist_ok=True)
    constraints.write_text(f"$set\ndistance 1 6 {r:.4f}\ndistance 3 4 {r:.4f}\n")

    steps = []

    def callback(env):
        steps.append(float(env["energy"]))

    conv, mol_eq = geometric_solver.kernel(
        scanner,
        constraints=str(constraints),
        callback=callback,
        maxsteps=100,
        **CONV,
    )
    # One more evaluation at exactly the converged geometry, so the energy,
    # orbitals and CI recorded belong to it.
    e_tot, grad = scanner(mol_eq)
    mc = scanner.base
    assert mc.converged, f"CASSCF not converged at {tag(r)}"
    counts = cope.active_irrep_counts(mc)
    assert counts == cope.CAS_IRREP_NCAS, (r, counts)
    cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]

    overlap = None
    if neighbour is not None:
        n_mol, n_cas = neighbour
        overlap = [float(x) for x in cope.active_overlap(n_mol, n_cas, mol_eq, cas)]

    r16, r34 = cope.interallyl(mol_eq)
    rec = {
        "r_target": r,
        "r16": r16,
        "r34": r34,
        "e_casscf": float(e_tot),
        "geom_converged": bool(conv),
        "opt_steps": len(steps),
        "max_abs_grad": float(np.abs(grad).max()),
        "active_irreps": counts,
        "natural_occupations": cope.active_natural(mc),
        "active_overlap_with_neighbour": overlap,
        "geometry": [(s, [float(v) for v in xyz]) for s, xyz in cope.mol_atoms(mol_eq)],
    }
    POINTS.mkdir(exist_ok=True)
    ORBITALS.mkdir(exist_ok=True)
    (POINTS / f"{tag(r)}.json").write_text(json.dumps(rec, indent=1) + "\n")
    np.save(ORBITALS / f"{tag(r)}.npy", mc.mo_coeff)
    print(
        f"[{tag(r)}] E = {e_tot:.10f}  R16 = {r16:.4f}  R34 = {r34:.4f}  "
        f"steps = {len(steps)}  conv = {conv}",
        flush=True,
    )
    return rec, mol_eq, cas


def run_center():
    lines = (HERE / "b3lyp_c2h_chair.xyz").read_text().splitlines()[2:]
    atoms = [
        (ln.split()[0], np.array([float(v) for v in ln.split()[1:4]]))
        for ln in lines
        if ln.strip()
    ]
    mol = make_mol(atoms)
    mo = np.load(HERE / "setup_orbitals.npy")
    scanner = make_scanner(mol, mo)
    relax(scanner, R_CENTER)


def run_sweep(direction):
    grid = R_DOWN if direction == "down" else R_UP
    _rec, atoms, mo = load_point(R_CENTER)
    mol = make_mol(atoms)
    scanner = make_scanner(mol, mo)
    neighbour = (mol, mo[:, scanner.base.ncore : scanner.base.ncore + cope.NCAS])
    # Prime the scanner (orbitals + CI) at the converged centre geometry.
    scanner(mol)
    for r in grid:
        _rec, mol_eq, cas = relax(scanner, r, neighbour)
        neighbour = (mol_eq, cas)


def collect():
    grid = sorted([R_CENTER, *R_DOWN, *R_UP])
    recs = [json.loads((POINTS / f"{tag(r)}.json").read_text()) for r in grid]

    def occ(rec, irrep):
        return [d["occ"] for d in rec["natural_occupations"] if d["irrep"] == irrep]

    rows, flags = [], []
    for rec in recs:
        ag, bu = occ(rec, "Ag"), occ(rec, "Bu")
        (au,), (bg,) = occ(rec, "Au"), occ(rec, "Bg")
        ov = rec["active_overlap_with_neighbour"]
        rows.append(
            {
                "R": rec["r_target"],
                "R16": rec["r16"],
                "R34": rec["r34"],
                "E_casscf": rec["e_casscf"],
                "n_ag1": ag[0],
                "n_bu1": bu[0],
                "n_au": au,
                "n_bg": bg,
                "n_ag2": ag[1],
                "n_bu2": bu[1],
                "min_overlap_sv": min(ov) if ov else 1.0,
                "opt_steps": rec["opt_steps"],
                "geom_converged": rec["geom_converged"],
            }
        )
    # The jump check: an active-space change shows as a neighbour overlap
    # singular value well below 1, or an occupation step out of line with the
    # steps either side of it.
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        d_occ = max(abs(a[k] - b[k]) for k in ("n_ag1", "n_bu1", "n_au", "n_bg"))
        if d_occ > 0.1:
            flags.append(f"occupation step {d_occ:.3f} between R={a['R']} and {b['R']}")
        if min(a["min_overlap_sv"], b["min_overlap_sv"]) < 0.9:
            flags.append(f"active overlap < 0.9 near R={b['R']}")
        if not b["geom_converged"]:
            flags.append(f"geometry not converged at R={b['R']}")

    (HERE / "scan.json").write_text(
        json.dumps({"rows": rows, "flags": flags}, indent=1) + "\n"
    )
    keys = list(rows[0])
    lines = [",".join(keys)]
    for row in rows:
        lines.append(
            ",".join(
                f"{row[k]:.10f}"
                if k == "E_casscf"
                else (f"{row[k]:.5f}" if isinstance(row[k], float) else str(row[k]))
                for k in keys
            )
        )
    (HERE / "scan.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("flags:", flags or "none")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--sweep", choices=("down", "up"))
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()

    from pyscf import lib

    if args.threads:
        lib.num_threads(args.threads)

    if args.collect:
        collect()
        return
    if args.sweep:
        run_sweep(args.sweep)
        return

    run_center()
    each = max(1, (args.threads or lib.num_threads()) // 2)
    logs = HERE / "logs"
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--sweep", d, "--threads", str(each)],
            stdout=open(logs / f"sweep_{d}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        for d in ("down", "up")
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"sweep failed: exit codes {codes}; see logs/sweep_*.log")
    collect()


if __name__ == "__main__":
    main()
