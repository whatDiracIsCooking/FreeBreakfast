#!/usr/bin/env python3
"""Write the reference data the overlap half-gradient test reads.

Over `gradient_reference.py`'s `integrals` cases, skew H2O in cc-pVDZ (s, p,
d) and cc-pVTZ (oxygen's f shells, so every class through (ff)), over this
tree's own `data/basis` files, spherical: the overlap's derivative on the
first index's centre only, contracted per atom with a fixed AO matrix A,

    G[B, k] = sum_{m on B} sum_n A_mn dS_mn/dB_k,  only m's centre moved
            = -sum_{m on B} sum_n A_mn (d_k m | n)
            =  sum_{m on B} sum_n A_mn (m | d_k n)

with d_k the electron-coordinate derivative. -int1e_ipovlp is
`pyscf.grad.rhf.get_ovlp` (s1[k, m, n] = -(d_k m | n), the nabla on the
FIRST, bra index, whatever the comment in `pyscf.nac.sacasscf._nac_csf`
says: that comment's <p|dq/dR> is the electron derivative on the ket, the
same number by parts). So G is twice `_nac_csf(mol, mf_grad, A, None)`, which
the tool checks, along with a fourth-order central difference (step 1e-3
bohr) of the cross overlap <m(B moved)|n> on the bra's centre alone, and for
symmetric A that 2 G is the full overlap gradient sum_mn A_mn dS_mn/dR.

The three matrices are a recipe, not a file, each index 0-based in PySCF's
spherical AO order, element (m, n):

    symmetric      1 / (m + n + 1)
    antisymmetric  (m - n) / (m + n + 1)
    general        (2 m + 1) / ((n + 1) (m + n + 1))

a few IEEE operations each, so a C++ test rebuilds them bit for bit. BLAS and
OpenMP are pinned to one thread and nothing run-dependent is written, so a
rerun is byte-identical.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/overlap_half_gradient_reference.py

writes test/gtest/data/overlap_half_gradient/overlap_half_gradient.json.
Re-run it to regenerate; the file is never hand-edited.
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
from df_reference import libcint_version  # noqa: E402
from gradient_reference import (  # noqa: E402
    INTEGRAL_CASES,
    atoms_out,
    molecules,
    rows,
)
from pyscf import gto, lib  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402
from pyscf.nac import sacasscf as nac_sacasscf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "overlap_half_gradient"
MANIFEST = OUT / "overlap_half_gradient.json"

# The finite-difference check: fourth-order central difference at this step
# (bohr), and the agreement it must reach, relative to max(1, max |G|).
FD_STEP = 1e-3
FD_TOLERANCE = 1e-8
# Against _nac_csf and the full gradient, relative to max(1, max |G|).
TOLERANCE = 1e-13

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/overlap_half_gradient_reference.py"
)


def matrices(n: int) -> dict:
    """The three recipe matrices, element (m, n) as the module docstring
    writes it."""
    m, c = numpy.meshgrid(
        numpy.arange(n, dtype=float), numpy.arange(n, dtype=float), indexing="ij"
    )
    return {
        "symmetric": 1.0 / (m + c + 1.0),
        "antisymmetric": (m - c) / (m + c + 1.0),
        "general": (2.0 * m + 1.0) / ((c + 1.0) * (m + c + 1.0)),
    }


def half_gradient(mol: gto.Mole, a: numpy.ndarray) -> numpy.ndarray:
    """G[B, k] = -sum_{m on B, n} A_mn (d_k m | n)."""
    ip = mol.intor("int1e_ipovlp")
    out = numpy.zeros((mol.natm, 3))
    for b, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[b] = -numpy.einsum("xmn,mn->x", ip[:, p0:p1], a[p0:p1])
    return out


def check_fd(case: dict, a: numpy.ndarray, got: numpy.ndarray, what: str) -> None:
    """G against a central difference of sum_{m on B} A_mn <m(B moved)|n>."""
    mol, _ = molecules(case)
    coords = numpy.array([coord for _, coord in case["atoms"]])
    slices = mol.aoslice_by_atom()
    numeric = numpy.zeros_like(got)
    for b in range(mol.natm):
        p0, p1 = slices[b][2], slices[b][3]
        for k in range(3):
            values = {}
            for s in (-2, -1, 1, 2):
                moved = coords.copy()
                moved[b, k] += s * FD_STEP
                bra = molecules(case, moved)[0]
                cross = gto.intor_cross("int1e_ovlp", bra, mol)
                values[s] = numpy.sum(a[p0:p1] * cross[p0:p1])
            numeric[b, k] = (
                8.0 * (values[1] - values[-1]) - (values[2] - values[-2])
            ) / (12.0 * FD_STEP)
    error = numpy.abs(got - numeric).max()
    assert error < FD_TOLERANCE * max(1.0, numpy.abs(got).max()), (
        case["name"],
        what,
        error,
    )


def case_entry(case: dict) -> dict:
    mol, _ = molecules(case)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == case["max_l"]
    mf_grad = rhf_grad.Gradients(mol.RHF())
    s1 = mf_grad.get_ovlp(mol)
    out = {}
    for what, a in matrices(mol.nao).items():
        got = half_gradient(mol, a)
        scale = max(1.0, numpy.abs(got).max())
        csf = nac_sacasscf._nac_csf(mol, mf_grad, a, None)
        assert numpy.abs(got - 2.0 * csf).max() < TOLERANCE * scale, (what, "csf")
        if what == "symmetric":
            # The full gradient, both centres differentiated.
            full = numpy.zeros_like(got)
            for b, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                block = numpy.zeros_like(s1)
                block[:, p0:p1] = s1[:, p0:p1]
                block += block.transpose(0, 2, 1)
                full[b] = numpy.einsum("xmn,mn->x", block, a)
            assert numpy.abs(2.0 * got - full).max() < TOLERANCE * scale, (what, "full")
        check_fd(case, a, got, what)
        out[what] = rows(got)
    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "gradients": out,
    }


def main() -> None:
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "PySCF/libcint overlap half-gradients over this tree's "
        "data/basis files, spherical, PySCF's unit-normalised functions: "
        "G[B][k] = -sum_{m on B, n} A_mn int1e_ipovlp[k, m, n], the overlap's "
        "derivative on the first index's centre only, for three recipe matrices "
        "A (symmetric 1/(m+n+1), antisymmetric (m-n)/(m+n+1), general "
        "(2m+1)/((n+1)(m+n+1)), 0-based PySCF spherical AO order); twice "
        "pyscf.nac.sacasscf._nac_csf; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": [case_entry(case) for case in INTEGRAL_CASES],
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
