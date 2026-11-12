"""Step 4a: free-breakfast's DF-CASSCF(6,6) and PC-NEVPT2 at step 1's geometries.

Three phases, each resumable:

  prepare   (PySCF)  per point, inputs/R<r>/: the geometry as XYZ, step 1's
                     orbitals projected onto this tree's cc-pVDZ, the config.
  run       (GPU)    the free-breakfast binary on each prepared point, its
                     log to runs/R<r>/log.txt. Skips a point already logged.
  collect            parses the logs into energies.json / energies.csv and
                     compares them with step 1's CASSCF and step 2's block2
                     PC-NEVPT2.

**Why project.** Steps 1-2 ask PySCF for "cc-pvdz" by name. That is a
different contraction of the same span as data/basis/cc-pVDZ.json (README
"Steps 1-2's basis is not this tree's"), so step 1's MO columns over this
tree's AOs are different orbitals. The projection C_tree = S_tt^-1 S_tp C_p is
exact because the spans agree. The CASSCF reconverges from it regardless, so
the projection only has to land in the right basin, and the CASSCF energy
check below says whether it did.

**What is compared.** E_CASSCF against step 1's (the same state, so equal to
convergence). E_PC = E_CASSCF + PC E_corr against step 2's block2 number, and
PC E_corr on its own. SC E_corr is recorded beside step 2's, but step 2's SC
was run on the active natural orbitals and SC is not invariant to active
rotations, so a difference there is not by itself a finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
STEP1 = STUDY / "step_1"
STEP2 = STUDY / "step_2"
INPUTS = HERE / "inputs"
RUNS = HERE / "runs"

# Step 2's own (pt2.py): tight enough that neither side's convergence limits
# the comparison.
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-6
NCAS, NELECAS = 6, 6

CONFIG = """\
# Step 4a, R = {r:.2f} A: step 1's DF-CASSCF(6,6)/cc-pVDZ geometry, CASSCF
# from step 1's orbitals (projected onto this tree's cc-pVDZ), then
# PC- and SC-NEVPT2. Written by energies.py prepare.
[molecule]
xyz = "geometry.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = {ncas}
nelecas = {nelecas}
orbitals = "orbitals.npy"
conv_tol = {conv_tol}
conv_tol_grad = {conv_tol_grad}
write_npy = true              # casscf_orbitals.npy, for crosscheck.py
[nevpt2]
"""


def points() -> list[float]:
    """Step 1's scan values of R, ascending."""
    return sorted(float(p.stem[1:]) for p in (STEP1 / "points").glob("R*.json"))


def tag(r: float) -> str:
    return f"R{r:.2f}"


def contraction_norms(mol, basis: dict):
    """Each AO's norm under the published coefficients, in `mol`'s AO order.

    free-breakfast folds only the primitive norm into a contraction and never
    renormalises it (#1220), so its AO i is PySCF's (normalised) AO i times
    this D_i, up to ~1e-6 from 1 in cc-pVDZ. An orbital's coefficients over
    this tree's AOs are therefore D^-1 times those over PySCF's.
    `basis` is pyscf_basis's form, [l, [exponent, c_1, ...], ...] per element;
    as experimental/rust/s_overlap/data/reference.py's contraction_norms.
    """
    import numpy
    from pyscf import gto

    per_element = {}
    for symbol, shells in basis.items():
        norms = []
        for ell, *rows in shells:
            exps = numpy.array([row[0] for row in rows])
            prim = numpy.array([gto.gto_norm(ell, a) for a in exps])
            pair = numpy.array(
                [[gto.gaussian_int(2 * ell + 2, a + b) for b in exps] for a in exps]
            )
            for col in range(1, len(rows[0])):
                c = numpy.array([row[col] for row in rows]) * prim
                norms += [float(numpy.sqrt(c @ pair @ c))] * (2 * ell + 1)
        per_element[symbol] = norms
    d = numpy.concatenate([per_element[mol.atom_symbol(a)] for a in range(mol.natm)])
    assert len(d) == mol.nao
    return d


def tree_mols(atoms: list) -> tuple:
    """PySCF's built-in cc-pVDZ, this tree's cc-pVDZ.json, and the D between."""
    from pyscf import gto

    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    from scf_reference import pyscf_basis

    basis = pyscf_basis("cc-pVDZ.json", {"C", "H"})
    builtin = gto.M(atom=atoms, basis="cc-pvdz", unit="Angstrom", verbose=0)
    tree = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    assert builtin.ao_labels() == tree.ao_labels()
    return builtin, tree, contraction_norms(tree, basis)


def projected(r: float) -> tuple:
    """Step 1's geometry at R (Angstrom atoms) and its orbitals projected onto
    this tree's cc-pVDZ (the contraction, then D), with the projection's
    orthonormality defect in this tree's overlap."""
    import numpy
    from pyscf import gto

    point = json.loads((STEP1 / "points" / f"{tag(r)}.json").read_text())
    atoms = [(s, tuple(xyz)) for s, xyz in point["geometry"]]
    mo = numpy.load(STEP1 / "orbitals" / f"{tag(r)}.npy")

    builtin, tree, d = tree_mols(atoms)
    s_tt = tree.intor("int1e_ovlp")
    s_tp = gto.intor_cross("int1e_ovlp", tree, builtin)
    c_json = numpy.linalg.solve(s_tt, s_tp @ mo)  # over PySCF's normalised JSON AOs
    c_tree = c_json / d[:, None]  # over this tree's AOs
    s_fb = d[:, None] * s_tt * d[None, :]  # this tree's own overlap
    defect = abs(c_tree.T @ s_fb @ c_tree - numpy.eye(mo.shape[1])).max()
    assert defect < 1e-10, f"{tag(r)}: projected orbitals off by {defect:.2e}"
    return atoms, c_tree, defect


def write_xyz(path: Path, atoms: list, comment: str) -> None:
    """`atoms` as (symbol, (x, y, z)) in Angstrom."""
    lines = [str(len(atoms)), comment]
    lines += [f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in atoms]
    path.write_text("\n".join(lines) + "\n")


def prepare(only: list[float]) -> None:
    import numpy

    for r in only:
        atoms, c_tree, defect = projected(r)
        out = INPUTS / tag(r)
        out.mkdir(parents=True, exist_ok=True)
        write_xyz(out / "geometry.xyz", atoms, f"step 1 {tag(r)}, Angstrom")
        numpy.save(
            out / "orbitals.npy", numpy.asfortranarray(c_tree), allow_pickle=False
        )
        (out / "config.toml").write_text(
            CONFIG.format(
                r=r,
                ncas=NCAS,
                nelecas=NELECAS,
                conv_tol=CONV_TOL,
                conv_tol_grad=CONV_TOL_GRAD,
            )
        )
        print(f"{tag(r)}: prepared (projection defect {defect:.1e})")


def run(only: list[float], binary: Path) -> None:
    for r in only:
        out = RUNS / tag(r)
        log = out / "log.txt"
        if log.exists() and "NEVPT2 root 0" in log.read_text():
            print(f"{tag(r)}: already run")
            continue
        out.mkdir(parents=True, exist_ok=True)
        # -i is where the config's own paths resolve; the config path itself
        # is the working directory's.
        inp = INPUTS / tag(r)
        cmd = [str(binary), "-i", str(inp), "-o", str(out), str(inp / "config.toml")]
        with log.open("w") as f:
            done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        status = "ok" if done.returncode == 0 else f"exit {done.returncode}"
        print(f"{tag(r)}: {status}", flush=True)


CASSCF_RE = re.compile(r"CASSCF macro\s+\d+: E (-?\d+\.\d+)")
NEVPT2_RE = re.compile(
    r"NEVPT2 root 0: SC E_corr (-?\d+\.\d+), PC E_corr (-?\d+\.\d+|refused)"
)


def parse(log: str) -> dict:
    cas = CASSCF_RE.findall(log)
    nev = NEVPT2_RE.search(log)
    if not cas or not nev or nev.group(2) == "refused":
        raise ValueError("no CASSCF energy, or no PC E_corr, in the log")
    return {
        "e_casscf": float(cas[-1]),
        "macro_iterations": len(cas),
        "e_corr_sc": float(nev.group(1)),
        "e_corr_pc": float(nev.group(2)),
    }


def collect(only: list[float]) -> None:
    rows = []
    for r in only:
        log = RUNS / tag(r) / "log.txt"
        if not log.exists():
            print(f"{tag(r)}: not run")
            continue
        fb = parse(log.read_text())
        s1 = json.loads((STEP1 / "points" / f"{tag(r)}.json").read_text())
        s2 = json.loads((STEP2 / "pt2_points" / f"{tag(r)}.json").read_text())
        e_pc = fb["e_casscf"] + fb["e_corr_pc"]
        rows.append(
            {
                "r": r,
                **fb,
                "e_pc": e_pc,
                "e_casscf_step1": s1["e_casscf"],
                "e_casscf_step2": s2["e_casscf"],
                "e_pc_block2": s2["e_pc"],
                "e_corr_pc_block2": s2["e_corr_pc"],
                "e_corr_sc_step2_natorb": s2["e_corr_sc"],
                "d_casscf": fb["e_casscf"] - s2["e_casscf"],
                "d_e_corr_pc": fb["e_corr_pc"] - s2["e_corr_pc"],
                "d_e_pc": e_pc - s2["e_pc"],
                "d_e_corr_sc": fb["e_corr_sc"] - s2["e_corr_sc"],
            }
        )
    (HERE / "energies.json").write_text(json.dumps(rows, indent=1) + "\n")
    if rows:
        with (HERE / "energies.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for row in rows:
        print(
            f"{tag(row['r'])}: dE_CAS {row['d_casscf']:+.2e}"
            f"  dEcorr_PC {row['d_e_corr_pc']:+.2e}"
            f"  dE_PC {row['d_e_pc']:+.2e}  dEcorr_SC {row['d_e_corr_sc']:+.2e}"
            f"  ({row['macro_iterations']} macro)"
        )
    if rows:
        for key in ("d_casscf", "d_e_corr_pc", "d_e_pc"):
            print(f"max |{key}| = {max(abs(row[key]) for row in rows):.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["prepare", "run", "collect"])
    parser.add_argument(
        "r", nargs="*", type=float, help="R values (default: every point)"
    )
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "src" / "free-breakfast"
    )
    args = parser.parse_args()
    only = args.r or points()
    if args.phase == "prepare":
        prepare(only)
    elif args.phase == "run":
        run(only, args.binary)
    else:
        collect(only)


if __name__ == "__main__":
    main()
