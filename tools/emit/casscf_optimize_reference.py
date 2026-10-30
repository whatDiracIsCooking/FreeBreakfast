#!/usr/bin/env python3
"""Write the reference minimum the DF-CASSCF geometry-optimization test reads.

One case: H2O in cc-pVDZ over cc-pVDZ-JKFIT, the state-specific CAS(4,4)
over its sigma O-H orbitals ([2, 3, 5, 6], as casscf_gradient_reference.py's
h2o_skew_ccpvdz), optimized from that case's off-symmetry geometry
(df_reference.py's WATER_SKEW) to PySCF's DF-CASSCF minimum.

**The optimizer.** geomeTRIC is not in the tools environment, so the minimum
is Newton's method over PySCF's analytic DF-CASSCF gradient
(`df.grad.casscf`), its Hessian a central difference of that gradient
(HESSIAN_STEP bohr) recomputed every step, rigid translations and rotations
projected out, as gradient_reference.py's DF-RHF minima are found; converged
to max |dE/dR| < FORCE_TOLERANCE hartree/bohr. Newton's path is not
geometry::optimize's, so only the minimum is a reference.

**Each energy and gradient.** The CASSCF is built as
casscf_gradient_reference.py builds it (over a DF-RHF _scf sharing its fit,
so the gradient's W is fitted too), converged by mc1step and polished by
Newton steps to |g| < 1e-10, and differentiated there, at the polished
orbitals and their dense CI. At the start its orbitals are the exact RHF's,
sort_mo over the active list; at every later geometry the previous
geometry's converged orbitals, Lowdin-orthonormalized in the new overlap, so
the active space stays continuous. At the minimum the CASSCF is rerun from a
fresh RHF and sort_mo, and must reach the same energy (to 1e-9): the minimum
is the one a calculation started there finds.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casscf_optimize_reference.py

writes test/gtest/data/casscf_gradient/casscf_optimize.json. Re-run it to
regenerate; the file is never hand-edited. BLAS and OpenMP run at one thread
and nothing run-dependent is written.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so the output is reproducible.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from casci_reference import aux_basis  # noqa: E402
from casscf_gradient_reference import build, polish  # noqa: E402
from casscf_reference import casscf, molecule, optimize, spin_roots  # noqa: E402
from df_reference import WATER_SKEW  # noqa: E402
from gradient_reference import internal_projector  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_gradient" / "casscf_optimize.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/casscf_optimize_reference.py"
)

CASE = {
    "name": "h2o_ccpvdz",
    "atoms": WATER_SKEW,
    "basis": "cc-pvdz",
    "cas": (4, 4),
    "active": [2, 3, 5, 6],
}

FORCE_TOLERANCE = 1e-7
HESSIAN_STEP = 1e-3
# Newton steps before the tool gives up, and the cap on one step (bohr per
# coordinate).
MAX_NEWTON_STEPS = 20
MAX_STEP = 0.3
# The CASSCF rerun from a fresh start at the minimum must reach the carried
# one's energy to this (hartree).
FRESH_AGREEMENT = 1e-9


def placed(coords: numpy.ndarray) -> dict:
    return dict(
        CASE,
        atoms=[(s, tuple(r)) for (s, _), r in zip(CASE["atoms"], coords, strict=True)],
    )


def fresh_start(case: dict, mol: gto.Mole) -> numpy.ndarray:
    """The exact RHF's orbitals, sort_mo over the active list."""
    nelecas, ncas = case["cas"]
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    return casscf(mf, ncas, nelecas, [1.0]).sort_mo(case["active"], base=0)


def carried(mol: gto.Mole, mo: numpy.ndarray) -> numpy.ndarray:
    """mo Lowdin-orthonormalized in mol's overlap."""
    overlap = mo.T @ mol.intor("int1e_ovlp") @ mo
    values, vectors = numpy.linalg.eigh(overlap)
    return mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T


def energy_gradient(coords: numpy.ndarray, mo: numpy.ndarray | None):
    """The polished DF-CASSCF's energy, gradient and orbitals at coords,
    from mo carried over (a fresh start when None)."""
    case = placed(coords)
    mol = molecule(case)
    auxbasis, _ = aux_basis(case, mol)
    start = fresh_start(case, mol) if mo is None else carried(mol, mo)
    mc = build(case, mol, auxbasis, start)
    optimize(mc, mc.kernel, start)
    assert mc.converged, "mc1step did not converge"
    mo = polish(case["name"], mc, mc.mo_coeff, [1.0])
    # The gradient at the polished orbitals and their dense CI, not at
    # mc1step's own stop (|g| near conv_tol_grad), which is noise at 1e-7.
    energies, vectors, _ = spin_roots(mc, mo, 1)
    mc.mo_coeff, mc.ci, mc.e_tot = mo, vectors[0], energies[0]
    grad = mc.nuc_grad_method()
    grad.verbose = 0
    return float(energies[0]), grad.kernel(mo_coeff=mo, ci=vectors[0]), mo


def atoms_out(coords: numpy.ndarray) -> list[dict]:
    return [
        {"symbol": symbol, "z": gto.charge(symbol), "coord": [float(x) for x in r]}
        for (symbol, _), r in zip(CASE["atoms"], coords, strict=True)
    ]


def main() -> None:
    lib.num_threads(1)
    start = numpy.array([coord for _, coord in CASE["atoms"]], dtype=float)
    coords = start.copy()
    energy, gradient, mo = energy_gradient(coords, None)
    start_energy = energy
    steps = 0
    while numpy.abs(gradient).max() >= FORCE_TOLERANCE:
        assert steps < MAX_NEWTON_STEPS, "Newton did not converge"
        n = coords.size
        hessian = numpy.zeros((n, n))
        for i in range(n):
            for s in (1, -1):
                moved = coords.ravel().copy()
                moved[i] += s * HESSIAN_STEP
                _, g, _ = energy_gradient(moved.reshape(coords.shape), mo)
                hessian[i] += s * g.ravel()
        hessian = 0.5 * (hessian + hessian.T) / (2.0 * HESSIAN_STEP)
        projector = internal_projector(coords)
        values, vectors = numpy.linalg.eigh(projector @ hessian @ projector)
        internal = numpy.abs(values) > 1e-6
        assert internal.sum() == n - 6 and values[internal].min() > 0.0
        inverse = (vectors[:, internal] / values[internal]) @ vectors[:, internal].T
        step = -inverse @ (projector @ gradient.ravel())
        step *= min(1.0, MAX_STEP / numpy.abs(step).max())
        coords = coords + step.reshape(coords.shape)
        energy, gradient, mo = energy_gradient(coords, mo)
        steps += 1
        force = numpy.abs(gradient).max()
        print(f"newton {steps}: E {energy:.12f} max |g| {force:.2e}")
    fresh_energy, fresh_gradient, _ = energy_gradient(coords, None)
    assert abs(fresh_energy - energy) < FRESH_AGREEMENT, (fresh_energy, energy)
    assert numpy.abs(fresh_gradient).max() < FORCE_TOLERANCE

    mol = molecule(placed(coords))
    _, aux_provenance = aux_basis(CASE, mol)
    nelecas, ncas = CASE["cas"]
    doc = {
        "description": "PySCF's state-specific DF-CASSCF minimum (df.grad.casscf "
        "by Newton's method) over this tree's data/basis and data/aux_basis "
        "files, from an off-symmetry start. Hartree, bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "command": COMMAND,
            "threads": 1,
        },
        "cases": [
            {
                "name": CASE["name"],
                "basis": CASE["basis"],
                "aux_basis": aux_provenance,
                "charge": 0,
                "ncas": ncas,
                "nelecas": nelecas,
                "active": CASE["active"],
                "start": atoms_out(start),
                "start_energy": start_energy,
                "atoms": atoms_out(coords),
                "energy": energy,
                "max_force": float(numpy.abs(gradient).max()),
                "fresh_start_energy": fresh_energy,
                "newton_steps": steps,
                "optimizer": "Newton's method over PySCF's analytic DF-CASSCF "
                "gradient (df.grad.casscf), the Hessian a central difference of it "
                f"(step {HESSIAN_STEP} bohr) recomputed every step, rigid "
                "translations and rotations projected out, steps capped at "
                f"{MAX_STEP} bohr per coordinate; converged to max |dE/dR| < "
                f"{FORCE_TOLERANCE}. geomeTRIC is not in the tools environment",
            }
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
