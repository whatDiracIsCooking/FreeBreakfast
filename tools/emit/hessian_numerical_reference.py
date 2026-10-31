#!/usr/bin/env python3
"""Write the reference the `[hessian]` step's numerical Hessian is tested
against.

The step (`src/pipeline/hessian/`) differentiates the DF-RHF analytic
gradient by central differences; its oracle is PySCF's analytic DF-RHF
nuclear Hessian, `pyscf.df.hessian.rhf` (auxbasis response 2), over the same
aux set. The CPHF is solved one column at a time to 1e-12, as
`rhf_hessian_reference.py` solves it (PySCF's block Krylov stalls near 1e-7),
and the Hessian written is `hess_elec` over those solutions plus `hess_nuc`.
It is checked here against:

- PySCF's unmodified `Hessian().kernel()`, to what its stalled solve reaches;
- a central difference of PySCF's analytic DF-RHF gradient (`df.grad.rhf`)
  at +-FD_STEP bohr, each displaced SCF re-converged to `conv_tol` 1e-12;
- its own symmetry and translational sum rule, sum_B H[A k][B l] = 0.

Case, over this tree's own basis files (as `gradient_reference.py` hands them
to PySCF), spherical throughout:

- `h2o_ccpvdz`: the skew water of `density_fitting/` (C1, no symmetry
  plane) in cc-pVDZ / cc-pVDZ-JKFIT.

The Hessian is written as 3 n_atom rows of 3 n_atom, element (x, y) =
d^2E/dx dy with x = 3 A + k, hartree / bohr^2. BLAS and OpenMP are pinned to
one thread, and nothing run-dependent is written, so a rerun regenerates the
file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/hessian_numerical_reference.py

writes test/gtest/data/hessian_numerical/hessian_numerical.json. Re-run it to
regenerate; the file is never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import functools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from cphf_reference import KRYLOV_LINDEP  # noqa: E402
from df_reference import WATER_SKEW, libcint_version  # noqa: E402
from gradient_reference import atoms_out, df_rhf, molecules  # noqa: E402
from pyscf import lib  # noqa: E402
from pyscf.df.hessian import rhf as df_hessian  # noqa: E402
from rhf_hessian_reference import CPHF_TOLERANCE, column_mo1, flat  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "hessian_numerical" / "hessian_numerical.json"

CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
    },
]

# The central difference of PySCF's own gradient, and how far it may sit from
# the analytic Hessian: the step's h^2 error and the SCF's noise over 2h.
FD_STEP = 1e-3
FD_TOLERANCE = 2e-6
# PySCF's unmodified kernel: its block Krylov stalls near 1e-7 in U.
STALLED_TOLERANCE = 1e-6
# The analytic Hessian's own symmetry and translational sum, to the CPHF
# tolerance's reach.
INVARIANCE_TOLERANCE = 1e-8

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/hessian_numerical_reference.py"
)


def finite_difference(case: dict, mf) -> numpy.ndarray:
    """Row x: (g(+h e_x) - g(-h e_x)) / 2h, each SCF started from mf's
    density."""
    coords = numpy.array([coord for _, coord in case["atoms"]], dtype=float)
    n = coords.size
    dm0 = mf.make_rdm1()
    out = numpy.zeros((n, n))
    for x in range(n):
        sides = []
        for sign in (1.0, -1.0):
            displaced = coords.copy()
            displaced.flat[x] += sign * FD_STEP
            mol, auxmol = molecules(case, displaced)
            sides.append(df_rhf(mol, auxmol, dm0).nuc_grad_method().kernel().ravel())
        out[x] = (sides[0] - sides[1]) / (2 * FD_STEP)
    return out


def hessian_case(case: dict) -> dict:
    mol, auxmol = molecules(case)
    mf = df_rhf(mol, auxmol)
    mf.conv_tol_cpscf = CPHF_TOLERANCE
    hobj = mf.Hessian()
    assert isinstance(hobj, df_hessian.Hessian)
    assert hobj.auxbasis_response == 2
    hobj.max_cycle = 200
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ

    h1ao = hobj.make_h1(c, occ)
    mo1, mo_e1 = column_mo1(mf, h1ao)
    elec = hobj.hess_elec(eps, c, occ, mo1=mo1, mo_e1=mo_e1, h1ao=h1ao)
    hessian = flat(elec + hobj.hess_nuc())
    scale = numpy.abs(hessian).max()

    asymmetry = numpy.abs(hessian - hessian.T).max()
    assert asymmetry < INVARIANCE_TOLERANCE, (case["name"], "asymmetry", asymmetry)
    n_atom = mol.natm
    translation = numpy.abs(hessian.reshape(3 * n_atom, n_atom, 3).sum(axis=1)).max()
    assert translation < INVARIANCE_TOLERANCE, (case["name"], "sum rule", translation)
    stalled = numpy.abs(flat(hobj.kernel()) - hessian).max()
    assert stalled < STALLED_TOLERANCE * scale, (case["name"], "kernel", stalled)
    numerical = finite_difference(case, mf)
    fd = numpy.abs(numerical - hessian).max()
    assert fd < FD_TOLERANCE, (case["name"], "finite difference", fd)
    print(
        f"{case['name']}: max |H| {scale:.4f}, asymmetry {asymmetry:.2e}, "
        f"sum rule {translation:.2e}, kernel {stalled:.2e}, "
        f"central difference {fd:.2e}"
    )
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": atoms_out(mol),
        "energy": float(mf.e_tot),
        "hessian": [[float(v) for v in row] for row in hessian],
        "checks": {
            "asymmetry": float(asymmetry),
            "translational_sum": float(translation),
            "kernel": float(stalled),
            "central_difference": float(fd),
            "central_difference_step": FD_STEP,
        },
    }


def main() -> None:
    lib.num_threads(1)
    lib.krylov = functools.partial(lib.krylov, lindep=KRYLOV_LINDEP)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "PySCF's analytic DF-RHF nuclear Hessian "
        "(pyscf.df.hessian.rhf, auxbasis response 2, hess_elec over CPHF "
        "solutions to 1e-12 plus hess_nuc), checked against a central "
        "difference of df.grad.rhf, over this tree's data/basis and "
        "data/aux_basis files, spherical; coordinates in bohr, the Hessian "
        "in hartree / bohr^2, 3 n_atom rows of 3 n_atom, x = 3 A + k",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": [hessian_case(case) for case in CASES],
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
