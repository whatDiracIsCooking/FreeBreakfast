#!/usr/bin/env python3
"""Write the PySCF single point the SC-NEVPT2 geometry-optimization test reads.

One case: skew H2O in cc-pVDZ over cc-pVDZ-JKFIT, the single-state
DF-CASSCF(4,4) over its sigma O-H orbitals ([2, 3, 5, 6]), as
nevpt2_gradient_reference.py's h2o_skew_ccpvdz, whose geometry is the
optimization's start.

**The geometry is the tree's.** No reference program has an SC-NEVPT2
nuclear gradient, so PySCF cannot find the minimum itself; and a minimum
found by differencing re-converged energies would cost a numerical gradient
per optimizer step. Instead OPTIMUM is the stationary point
`[optimize]` over `[nevpt2_gradient]` converges to (bohr, as
test/gtest/pipeline_optimize_gpu.cpp's
`WaterNevpt2ReachesAStationaryPointAtPyscfsEnergy` prints it), and this tool
checks it independently, in PySCF:

- **The single point.** At OPTIMUM, the exact RHF, sort_mo over the active
  list (the start the outer `[casscf]` takes there), then
  nevpt2_gradient_reference.py's polished DF-CASSCF and its SC-NEVPT2:
  PySCF's `mrpt.NEVPT` over a DFCASCI on the polished orbitals, its CI vector
  the dense singlet, core and virtuals canonicalized and the active orbitals
  left unrotated, DF throughout. Its total energy is the test's reference.
- **Stationary in PySCF too.** The numerical gradient of that energy at
  OPTIMUM, the CASSCF re-converged at every displaced geometry from the
  optimum's orbitals carried over (Lowdin-orthonormalized in the displaced
  overlap): central differences at STEPS (bohr, halving) and one Richardson
  step over them, every displaced energy recorded. Its largest component
  must be under FORCE_THRESHOLD, the test's `[optimize]` max_force, so PySCF
  agrees the point is one the optimizer may stop at. Every displaced CASSCF
  passes the continuity guard of nevpt2_gradient_reference.py's (its active
  space and CI vector overlap the optimum's past 1 - SUBSPACE_TOL and
  1 - CI_TOL).

BLAS and OpenMP run on one thread and nothing run-dependent is written, so a
rerun on the same PySCF and numpy is byte-identical:

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_optimize_reference.py

writes test/gtest/data/nevpt2_gradient/nevpt2_optimize.json. Never
hand-edited: a change to the optimizer that moves the optimum past the
test's tolerance re-runs this with the new OPTIMUM.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402

# isort: split
from casci_reference import aux_basis  # noqa: E402
from casscf_reference import casscf, check_aux, molecule  # noqa: E402
from df_reference import WATER_SKEW  # noqa: E402
from nevpt2_gradient_reference import (  # noqa: E402
    CI_TOL,
    SUBSPACE_TOL,
    carried,
    moved,
    polished,
    rows,
    run_rhf,
    singlet,
)
from nevpt2_reference import CASCI_AGREEMENT, SAME_STATE, nevpt  # noqa: E402
from pyscf import fci, gto, lib, mcscf  # noqa: E402
from pyscf.mcscf.df import _DFCAS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "nevpt2_gradient" / "nevpt2_optimize.json"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/nevpt2_optimize_reference.py"
)

# The tree's SC-NEVPT2 optimum from WATER_SKEW (bohr), as the GPU case
# prints it (RTX 3080, CUDA, the default preset).
OPTIMUM = [
    ("O", (-0.0051085081344577404, -0.0040699214988794826, 0.016752685740833585)),
    ("H", (1.4428944512863742, 0.21198320689015981, -1.079997348492322)),
    ("H", (-1.3877859431519166, 0.35208671460871954, -1.1267553372485117)),
]

CASE = {
    "name": "h2o_skew_ccpvdz",
    "atoms": OPTIMUM,
    "basis": "cc-pvdz",
    "cas": (4, 4),
    "active": [2, 3, 5, 6],
}

# The numerical gradient's steps (bohr, halving); Richardson over the two.
STEPS = (1e-3, 5e-4)
# The test's [optimize] max_force (hartree/bohr): PySCF's numerical gradient
# at OPTIMUM must be under it too.
FORCE_THRESHOLD = 1.5e-5


def sc_energy(name, mf, auxbasis, orbitals, ncas, nelecas, e_casscf):
    """The SC-NEVPT2 correlation energy at the polished orbitals, over a
    DFCASCI, as nevpt2_gradient_reference.nevpt2_at computes SC."""
    casci = mcscf.CASCI(mf.density_fit(auxbasis=auxbasis), ncas, nelecas)
    assert isinstance(casci, _DFCAS), f"{name}: the CASCI is not fitted"
    assert casci.with_df is casci._scf.with_df, f"{name}: two fitting objects"
    casci.fcisolver.conv_tol = 1e-14
    casci.fix_spin_(ss=0)
    casci.verbose = 0
    casci.kernel(orbitals)
    check_aux(name, casci, mf.mol, auxbasis)
    e_casci, ci = singlet(casci, casci.mo_coeff)
    overlap = abs(numpy.vdot(casci.ci, ci))
    assert overlap > 1 - SAME_STATE, f"{name}: Davidson overlap {overlap}"
    assert abs(e_casci - casci.e_tot) < CASCI_AGREEMENT, f"{name}: Davidson"
    assert abs(e_casci - e_casscf) < CASCI_AGREEMENT, f"{name}: CASCI"
    sc, _ = nevpt(casci, ci, 0, natorb=False)
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=OUT,
        help="where to write (default: the tree's)",
    )
    args = parser.parse_args()
    lib.num_threads(1)
    assert lib.num_threads() == 1
    name = CASE["name"]
    nelecas, ncas = CASE["cas"]
    coords = numpy.array([r for _, r in CASE["atoms"]], dtype=float)
    natom = len(coords)

    mol = molecule(CASE)
    auxbasis, provenance = aux_basis(CASE, mol)
    mf = run_rhf(name, mol)
    start = casscf(mf, ncas, nelecas, [1.0]).sort_mo(CASE["active"], base=0)
    mc, mo, e_casscf, ci = polished(name, mf, start, ncas, nelecas, auxbasis)
    ncore = mc.ncore
    e_corr = sc_energy(name, mf, auxbasis, mo, ncas, nelecas, e_casscf)
    energy = e_casscf + e_corr
    print(f"{name}: E_cas {e_casscf:.12f} E_corr {e_corr:.12f}", flush=True)

    act = slice(ncore, ncore + ncas)
    nelec = (nelecas // 2, nelecas // 2)
    worst = {"subspace": 0.0, "ci": 0.0}
    central = [numpy.zeros((natom, 3)) for _ in STEPS]
    displaced = []
    for atom in range(natom):
        for axis in range(3):
            entry = {"atom": atom, "axis": axis, "sc": []}
            for i, h in enumerate(STEPS):
                pair = {}
                for sign in (-1, 1):
                    displacement = numpy.zeros((natom, 3))
                    displacement[atom, axis] = sign * h
                    case = moved(CASE, coords + displacement)
                    dmol = molecule(case)
                    daux, _ = aux_basis(case, dmol)
                    dmf = run_rhf(name, dmol)
                    _, dmo, de, dci = polished(
                        name, dmf, carried(mo, dmol), ncas, nelecas, daux
                    )
                    cross = gto.intor_cross("int1e_ovlp", mol, dmol)
                    s = mo[:, act].T @ cross @ dmo[:, act]
                    subspace = 1 - numpy.linalg.svd(s, compute_uv=False).min()
                    assert subspace < SUBSPACE_TOL, f"{name}: subspace {subspace}"
                    overlap = float(fci.addons.overlap(ci, dci, ncas, nelec, s))
                    ci_miss = 1 - abs(overlap)
                    assert ci_miss < CI_TOL, f"{name}: CI overlap off 1 by {ci_miss}"
                    worst["subspace"] = max(worst["subspace"], float(subspace))
                    worst["ci"] = max(worst["ci"], float(ci_miss))
                    pair[sign] = de + sc_energy(name, dmf, daux, dmo, ncas, nelecas, de)
                entry["sc"].append([pair[-1], pair[1]])
                central[i][atom, axis] = (pair[1] - pair[-1]) / (2 * h)
            displaced.append(entry)
            print(f"{name}: atom {atom} axis {axis} done", flush=True)
    richardson = (4 * central[1] - central[0]) / 3
    max_gradient = float(numpy.abs(richardson).max())
    print(f"{name}: max |dE/dR| {max_gradient:.2e}", flush=True)
    assert max_gradient < FORCE_THRESHOLD, (
        f"{name}: PySCF's gradient {max_gradient} is not under {FORCE_THRESHOLD}"
    )

    doc = {
        "description": "PySCF's density-fitted SC-NEVPT2 (mrpt.NEVPT over a "
        "DFCASCI on the polished single-state DF-CASSCF orbitals) at the "
        "stationary point this tree's [optimize] over [nevpt2_gradient] "
        "reaches from nevpt2_gradient.json's h2o_skew_ccpvdz, over this "
        "tree's data/basis and data/aux_basis files, and its numerical "
        "gradient there. Hartree, bohr; G[A][k] = dE/dR_Ak",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "name": name,
        "basis": CASE["basis"],
        "aux_basis": provenance,
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in CASE["atoms"]
        ],
        "start": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in WATER_SKEW
        ],
        "ncore": int(ncore),
        "ncas": ncas,
        "nelecas": nelecas,
        "active": CASE["active"],
        "energies": {"casscf": e_casscf, "e_corr": e_corr, "sc": energy},
        "steps": list(STEPS),
        "gradient": {
            "central": [rows(d) for d in central],
            "richardson": rows(richardson),
        },
        "displaced": displaced,
        "max_gradient": max_gradient,
        "force_threshold": FORCE_THRESHOLD,
        "guard": {"subspace": worst["subspace"], "ci": worst["ci"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
