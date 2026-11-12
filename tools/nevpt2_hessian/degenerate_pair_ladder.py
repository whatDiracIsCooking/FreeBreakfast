#!/usr/bin/env python3
"""Write the degenerate-inactive-pair references of the analytic PC-NEVPT2
Hessian (#1268) and of the PC-QD-NEVPT2 state Hessian (#1277):
test/gtest/data/pc_nevpt2_degenerate/<case>.json.

Each case is this tree's own numerical `[hessian]` (method "numerical", the
h, h/2, h/4 ladder of central differences of the analytic PC gradient, the
step's default `pc_floor`, Cartesian coordinates) through the
`free-breakfast` binary: the config below, run in a work directory, and its
`hessian.json`'s finest-rung Hessian and per-element bar copied out beside
the case's start (atoms in bohr, basis, fitting set, active space), in the
start format of `nevpt2_hessian/`'s fixtures, which the C++ test reads.

The cases:

- `nh3_c3v_631g`: a C3v ammonia in 6-31G, CAS(8,7) over orbitals 1 to 7
  (df_nevpt2_tangents_gpu.cpp's), whose two virtual e pairs are exactly
  degenerate and split at first order by E displacements;
- `nh3_near_631g`: the same with one hydrogen moved 2e-6 bohr off C3v, so
  that both pairs are near-degenerate (about 1e-7 Eh) with no symmetry;
- `hexatriene_sto3g`: all-trans hexatriene (C2h) in STO-3G, CAS(6,6) over
  orbitals 19 to 24 (the pi space), whose terminal carbons' 1s pair is
  4.7e-7 Eh apart, the case of #1268's hexadiene;
- `qd_nh3_c3v_631g_sa2` and `qd_nh3_near_631g_sa2`: the two ammonias over
  an SA-2 CAS(8,7) at equal weights (df_qd_nevpt2_tangents_gpu.cpp's,
  both roots A1), the numerical `[hessian]` of PC-QD-NEVPT2 state
  `QD_STATE` (`quasi_degenerate`, the step tracking the state by its
  eigenvector), whose virtual e pairs are degenerate at f^SA.

A QD case's document carries `nroots`, `weights` and `state` beside the
start, and its `pc` the same keys.

Standard library only; needs a GPU and a build:

    tools/nevpt2_hessian/degenerate_pair_ladder.py \
        [--binary build/src/free-breakfast] [--only CASE ...] [--work DIR]

`--work` keeps each case's run directory (DIR/<case>); a directory that
already holds this config and its `hessian.json` is read, not rerun. About
four minutes for hexatriene and two for each ammonia on the gfx1200
(about fifteen for each QD ammonia). Not byte for byte: the GPU sums
reassociate run to run.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "pc_nevpt2_degenerate"
# The tree's kAngstromPerBohr (src/core/constants/constants.h).
ANGSTROM_PER_BOHR = 0.529177210903


# H1's shift along x off C3v for the near-degenerate ammonia (bohr).
NEAR_SHIFT = 2e-6
# The PC-QD-NEVPT2 state the QD cases differentiate (H_eff's eigenvalues
# ascending): the upper of the two, whose term (ii) is the eigenvector
# response to the lower.
QD_STATE = 1


def nh3(shift: float) -> list[tuple[str, tuple[float, float, float]]]:
    """df_nevpt2_tangents_gpu.cpp's C3v ammonia in bohr, H1 moved by
    @p shift along x."""
    rho = math.sqrt(1.9 * 1.9 - 0.7 * 0.7)
    atoms = [("N", (0.0, 0.0, 0.0))]
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        x = rho * math.cos(phi) + (shift if k == 0 else 0.0)
        atoms.append(("H", (x, rho * math.sin(phi), -0.7)))
    return atoms


# All-trans hexatriene, C2h in the xy plane (Angstrom): C=C 1.34/1.35,
# C-C 1.46, CCC 124 degrees, C-H 1.09.
HEXATRIENE_ANGSTROM = [
    ("C", (-3.06824289, -0.26055672, 0.0)),
    ("C", (-1.88509311, 0.36853518, 0.0)),
    ("C", (-0.59598963, -0.31689330, 0.0)),
    ("C", (0.59598963, 0.31689330, 0.0)),
    ("C", (1.88509311, -0.36853518, 0.0)),
    ("C", (3.06824289, 0.26055672, 0.0)),
    ("H", (-3.99261531, 0.31705528, 0.0)),
    ("H", (-3.10628334, -1.34989272, 0.0)),
    ("H", (-1.88509311, 1.45853518, 0.0)),
    ("H", (-0.59598963, -1.40689330, 0.0)),
    ("H", (0.59598963, 1.40689330, 0.0)),
    ("H", (1.88509311, -1.45853518, 0.0)),
    ("H", (3.99261531, -0.31705528, 0.0)),
    ("H", (3.10628334, 1.34989272, 0.0)),
]

CASES = {
    "nh3_c3v_631g": {
        "description": (
            "C3v ammonia, 6-31G, CAS(8,7): virtual e pairs exactly degenerate"
        ),
        "atoms_bohr": nh3(0.0),
        "basis": "6-31G",
        "ncas": 7,
        "nelecas": 8,
        "active": [1, 2, 3, 4, 5, 6, 7],
    },
    "nh3_near_631g": {
        "description": (
            "the C3v ammonia with H1 moved 2e-6 bohr along x, 6-31G, CAS(8,7): "
            "its virtual e pairs split, near-degenerate, no symmetry"
        ),
        "atoms_bohr": nh3(NEAR_SHIFT),
        "basis": "6-31G",
        "ncas": 7,
        "nelecas": 8,
        "active": [1, 2, 3, 4, 5, 6, 7],
    },
    "hexatriene_sto3g": {
        "description": (
            "all-trans hexatriene (C2h), STO-3G, CAS(6,6) pi: terminal C 1s "
            "pair 4.7e-7 Eh apart"
        ),
        "atoms_bohr": [
            (s, tuple(v / ANGSTROM_PER_BOHR for v in xyz))
            for s, xyz in HEXATRIENE_ANGSTROM
        ],
        "basis": "sto-3g",
        "ncas": 6,
        "nelecas": 6,
        "active": [19, 20, 21, 22, 23, 24],
    },
    "qd_nh3_c3v_631g_sa2": {
        "description": (
            "C3v ammonia, 6-31G, SA-2 CAS(8,7), PC-QD-NEVPT2 state 1: virtual e "
            "pairs exactly degenerate at f^SA"
        ),
        "atoms_bohr": nh3(0.0),
        "basis": "6-31G",
        "ncas": 7,
        "nelecas": 8,
        "active": [1, 2, 3, 4, 5, 6, 7],
        "nroots": 2,
    },
    "qd_nh3_near_631g_sa2": {
        "description": (
            "the C3v ammonia with H1 moved 2e-6 bohr along x, 6-31G, SA-2 "
            "CAS(8,7), PC-QD-NEVPT2 state 1: its virtual e pairs split at f^SA, "
            "near-degenerate, no symmetry"
        ),
        "atoms_bohr": nh3(NEAR_SHIFT),
        "basis": "6-31G",
        "ncas": 7,
        "nelecas": 8,
        "active": [1, 2, 3, 4, 5, 6, 7],
        "nroots": 2,
    },
}
AUX = "cc-pVDZ-JKFIT"


def config(case: dict) -> tuple[str, str]:
    """The case's molecule.xyz (Angstrom) and run.toml."""
    atoms = case["atoms_bohr"]
    xyz = f"{len(atoms)}\ndegenerate pair case\n" + "".join(
        f"{s} {x * ANGSTROM_PER_BOHR:.17g} {y * ANGSTROM_PER_BOHR:.17g} "
        f"{z * ANGSTROM_PER_BOHR:.17g}\n"
        for s, (x, y, z) in atoms
    )
    active = ", ".join(str(a) for a in case["active"])
    nroots = case.get("nroots", 1)
    toml = (
        '[molecule]\nxyz = "molecule.xyz"\n'
        f'[basis_set]\nbasis = "{case["basis"]}"\n'
        f'[eri]\nmode = "density_fitting"\naux_basis = "{AUX}"\n'
        "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n"
        f"[casscf]\nncas = {case['ncas']}\nnelecas = {case['nelecas']}\n"
        f"active = [{active}]\nconv_tol = 1e-12\nconv_tol_grad = 1e-10\n"
        + (f"nroots = {nroots}\n" if nroots > 1 else "")
        + '[nevpt2_gradient]\nvariant = "pc"\n'
        + (f"quasi_degenerate = true\nstate = {QD_STATE}\n" if nroots > 1 else "")
        + '[hessian]\nmethod = "numerical"\n'
    )
    return xyz, toml


def run(name: str, binary: pathlib.Path, work: pathlib.Path) -> dict:
    case = CASES[name]
    directory = work / name
    directory.mkdir(parents=True, exist_ok=True)
    xyz, toml = config(case)
    out = directory / "fbf_output" / "hessian.json"
    same = (
        (directory / "molecule.xyz").exists()
        and (directory / "molecule.xyz").read_text() == xyz
        and (directory / "run.toml").read_text() == toml
    )
    if not (same and out.exists()):
        (directory / "molecule.xyz").write_text(xyz)
        (directory / "run.toml").write_text(toml)
        with open(directory / "run.log", "w") as log:
            ran = subprocess.run(
                [str(binary), "run.toml"],
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if ran.returncode != 0:
            sys.exit(
                f"{name}: free-breakfast exited {ran.returncode}; "
                f"see {directory}/run.log"
            )
    hessian = json.loads(out.read_text())
    ladder = hessian["ladder"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    doc = {
        "description": case["description"],
        "provenance": {
            "tool": "tools/nevpt2_hessian/degenerate_pair_ladder.py",
            "binary": str(binary.name),
            "commit": commit,
            "date": datetime.date.today().isoformat(),
            "config": toml,
        },
        "name": name,
        "basis": {"name": case["basis"]},
        "aux_basis": {"name": AUX},
        "charge": 0,
        "atoms": [{"symbol": s, "coord": list(c)} for s, c in case["atoms_bohr"]],
        "ncas": case["ncas"],
        "nelecas": case["nelecas"],
        "active": case["active"],
        "pc_floor": hessian["pc_floor"],
        "pc": {
            "hessian": hessian["hessian"],
            "bar": ladder["bar"],
            "noise": ladder["noise"],
            "unsettled": ladder["unsettled"],
            "asymmetry": hessian.get("asymmetry"),
        },
    }
    if case.get("nroots", 1) > 1:
        doc["nroots"] = case["nroots"]
        doc["weights"] = [1.0 / case["nroots"]] * case["nroots"]
        doc["state"] = QD_STATE
        doc["qd"] = hessian.get("qd")
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--binary", type=pathlib.Path, default=ROOT / "build/src/free-breakfast"
    )
    parser.add_argument(
        "--only", nargs="*", choices=sorted(CASES), default=sorted(CASES)
    )
    parser.add_argument("--work", type=pathlib.Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = (args.work or pathlib.Path(tmp)).resolve()
        for name in args.only:
            doc = run(name, binary, work)
            (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
            print(f"{name}: wrote {OUT / (name + '.json')}")


if __name__ == "__main__":
    main()
