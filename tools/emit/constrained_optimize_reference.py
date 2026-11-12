#!/usr/bin/env python3
"""Write the reference the constrained `[optimize]` test reads (#1260).

One case: formaldehyde in cc-pVDZ over cc-pVDZ-JKFIT, DF-RHF, from
casscf_nac_reference.py's FORMALDEHYDE_SKEW (pyramidalized and twisted, C1),
minimized with two interatomic distances held fixed: C-O at 1.25 Angstrom and
C-H(2) at 1.15 Angstrom. Neither holds at the start (1.214 and 1.123), and
they share the carbon, so the two constraints are coupled.

**The optimizer** is geomeTRIC's, driven by PySCF's `geometric_solver` over
PySCF's analytic DF-RHF gradient (`df.grad.rhf`), the constraints a `$set
distance` block. It is an independent method from the tree's (internal
coordinates with its own constraint handling, against Cartesian BFGS with a
projection), so the two agree only where both reach the same constrained
minimum. Its criteria are tightened far past the tree test's thresholds
(CONVERGENCE), with `enforce` on so the constraints are met exactly at the
end, not to geomeTRIC's default 1e-2.

**What is recorded at its end point**, from one fresh evaluation there: the
energy; each constrained distance and its error; the gradient's projection
off the rigid motions and the constraint gradients (`max_force`, its largest
component, and `force_norm`, its 2-norm); each constraint's Lagrange
multiplier lambda (least squares, g = sum_k lambda_k dc_k/dR + the projected
part, dE/dd in hartree/bohr); and `k_min`, the least eigenvalue of the
Lagrangian's Hessian, d2E/dR2 - sum_k lambda_k d2c_k/dR2, over the free
coordinates (the orthonormal complement of the rigid motions and the
constraint gradients), d2E/dR2 a central difference of the analytic gradient
(HESSIAN_STEP bohr) and d2c/dR2 analytic. It is refused unless positive: a
constrained minimum, not a saddle. Those are what the test derives its bars
from (test/gtest/pipeline_optimize_gpu.cpp).

**Contraction norms (#1220).** PySCF reads this tree's own data/basis and
data/aux_basis files, and renormalizes each contraction, which the tree does
not: its AOs are PySCF's times a D_ii within ~1e-6 of 1. Only AO-basis arrays
see D. Energies, gradients and geometries are invariant to the AO scaling, and
nothing in the AO basis is written here, so nothing needs reconciling.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/constrained_optimize_reference.py

writes test/gtest/data/gradient/constrained_optimize.json. Re-run it to
regenerate; the file is never hand-edited. BLAS and OpenMP run at one thread;
nothing run-dependent (date, host) is written.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402

import geometric  # noqa: E402
import numpy  # noqa: E402
import pyscf  # noqa: E402
from casscf_nac_reference import FORMALDEHYDE_SKEW  # noqa: E402
from gradient_reference import (  # noqa: E402
    atoms_out,
    df_rhf,
    internal_projector,
    molecules,
)
from pyscf import lib  # noqa: E402
from pyscf.geomopt import geometric_solver  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "gradient" / "constrained_optimize.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/constrained_optimize_reference.py"
)

ANGSTROM_PER_BOHR = lib.param.BOHR

CASE = {
    "name": "h2co_skew_ccpvdz_constrained",
    "atoms": FORMALDEHYDE_SKEW,
    "basis": "cc-pvdz",
    "aux_basis": "cc-pvdz-jkfit",
    "charge": 0,
    # 0-based atoms, distance in Angstrom: as [optimize] constraints reads them.
    "constraints": [
        {"atoms": [0, 1], "distance": 1.25},
        {"atoms": [0, 2], "distance": 1.15},
    ],
}

# geomeTRIC's criteria (energy hartree, gradient hartree/bohr, displacement
# Angstrom), each well past the tree test's: its force bar is 1.5e-5.
CONVERGENCE = {
    "convergence_energy": 1e-11,
    "convergence_grms": 1e-7,
    "convergence_gmax": 1.5e-7,
    "convergence_drms": 1e-6,
    "convergence_dmax": 1.5e-6,
    # Constraint satisfaction, and the switch to exact enforcement near it.
    "convergence_cmax": 1e-10,
    "enforce": 0.1,
}
MAX_STEPS = 200
HESSIAN_STEP = 1e-3
# Each constrained distance at the end point to this (Angstrom).
CONSTRAINT_TOLERANCE = 1e-8


def distance(coords: numpy.ndarray, pair: list[int]) -> float:
    return float(numpy.linalg.norm(coords[pair[0]] - coords[pair[1]]))


def jacobian(coords: numpy.ndarray) -> numpy.ndarray:
    """dc_k/dR, one row per constraint: e at atom a, -e at atom b."""
    rows = []
    for c in CASE["constraints"]:
        a, b = c["atoms"]
        e = (coords[a] - coords[b]) / distance(coords, c["atoms"])
        row = numpy.zeros_like(coords)
        row[a], row[b] = e, -e
        rows.append(row.ravel())
    return numpy.array(rows)


def constraint_hessian(coords: numpy.ndarray, k: int) -> numpy.ndarray:
    """d2c_k/dR2: (I - e e^T) / r on the pair's blocks, + on aa and bb."""
    a, b = CASE["constraints"][k]["atoms"]
    r = distance(coords, [a, b])
    e = (coords[a] - coords[b]) / r
    block = (numpy.eye(3) - numpy.outer(e, e)) / r
    n = coords.size
    out = numpy.zeros((n, n))
    for i, si in ((a, 1.0), (b, -1.0)):
        for j, sj in ((a, 1.0), (b, -1.0)):
            out[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = si * sj * block
    return out


def energy_gradient(coords: numpy.ndarray):
    mf = df_rhf(*molecules(CASE, coords))
    return float(mf.e_tot), mf.nuc_grad_method().kernel()


def main() -> None:
    lib.num_threads(1)
    start = numpy.array([coord for _, coord in CASE["atoms"]])
    mol, auxmol = molecules(CASE, start)
    mf = df_rhf(mol, auxmol)

    steps = []
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "constraints.txt"
        lines = ["$set"]
        for c in CASE["constraints"]:
            a, b = c["atoms"]
            lines.append(f"distance {a + 1} {b + 1} {c['distance']:.10f}")
        path.write_text("\n".join(lines) + "\n")
        converged, mol_eq = geometric_solver.kernel(
            mf,
            constraints=str(path),
            callback=lambda env: steps.append(float(env["energy"])),
            maxsteps=MAX_STEPS,
            **CONVERGENCE,
        )
    assert converged, "geomeTRIC did not converge"

    coords = mol_eq.atom_coords(unit="Bohr")
    energy, gradient = energy_gradient(coords)
    g = gradient.ravel()

    a_rows = jacobian(coords)
    multipliers = numpy.linalg.lstsq(a_rows.T, g, rcond=None)[0]
    # The free coordinates: the complement of the rigid motions and the
    # constraint gradients.
    span = numpy.hstack(
        [
            numpy.linalg.svd(numpy.eye(coords.size) - internal_projector(coords))[0][
                :, :6
            ],
            a_rows.T,
        ]
    )
    u, s, _ = numpy.linalg.svd(span, full_matrices=True)
    rank = int(numpy.sum(s > 1e-8 * s[0]))
    assert rank == 6 + len(CASE["constraints"]), rank
    free = u[:, rank:]
    projected = free @ (free.T @ g)

    n = coords.size
    hessian = numpy.zeros((n, n))
    for i in range(n):
        for sign in (1, -1):
            moved = coords.ravel().copy()
            moved[i] += sign * HESSIAN_STEP
            hessian[i] += sign * energy_gradient(moved.reshape(coords.shape))[1].ravel()
    hessian = 0.5 * (hessian + hessian.T) / (2.0 * HESSIAN_STEP)
    lagrangian = hessian - sum(
        lam * constraint_hessian(coords, k) for k, lam in enumerate(multipliers)
    )
    curvatures = numpy.linalg.eigvalsh(free.T @ lagrangian @ free)
    k_min = float(curvatures.min())
    assert k_min > 0.0, f"not a constrained minimum: k_min {k_min}"

    distances = [
        distance(coords, c["atoms"]) * ANGSTROM_PER_BOHR for c in CASE["constraints"]
    ]
    errors = [
        d - c["distance"] for d, c in zip(distances, CASE["constraints"], strict=True)
    ]
    assert max(abs(e) for e in errors) <= CONSTRAINT_TOLERANCE, errors

    doc = {
        "description": "geomeTRIC's constrained DF-RHF minimum over PySCF, the "
        "reference [optimize]'s constraints test derives its bars from; "
        "coordinates in bohr, distances in Angstrom, energies in hartree, "
        "forces and multipliers in hartree/bohr, k_min in hartree/bohr^2",
        "provenance": {
            "pyscf": pyscf.__version__,
            "geometric": geometric.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "name": CASE["name"],
        "basis": CASE["basis"],
        "aux_basis": CASE["aux_basis"],
        "charge": CASE["charge"],
        "constraints": CASE["constraints"],
        "start": atoms_out(molecules(CASE, start)[0]),
        "atoms": atoms_out(mol_eq),
        "energy": energy,
        "distances": distances,
        "distance_errors": errors,
        "multipliers": [float(x) for x in multipliers],
        "max_force": float(numpy.abs(projected).max()),
        "force_norm": float(numpy.linalg.norm(projected)),
        "k_min": k_min,
        "free_curvatures": [float(x) for x in curvatures],
        "geometric_steps": len(steps),
        "optimizer": "geomeTRIC's optimizer (pyscf.geomopt.geometric_solver) "
        "over PySCF's analytic DF-RHF gradient, the constraints a $set distance "
        f"block, criteria {CONVERGENCE}; max_force, force_norm, multipliers and "
        "k_min from one fresh evaluation at its end point, k_min over the free "
        f"coordinates of the Lagrangian's Hessian (central difference, step "
        f"{HESSIAN_STEP} bohr)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        f"wrote {OUT}: E {energy:.12f}, {len(steps)} steps, max force "
        f"{doc['max_force']:.2e}, k_min {k_min:.4f}, distance errors {errors}"
    )


if __name__ == "__main__":
    main()
