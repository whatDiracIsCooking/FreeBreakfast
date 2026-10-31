#!/usr/bin/env python3
"""Write the reference data the one-electron derivative-matrix tests read.

Over `gradient_reference.py`'s `integrals` cases, skew H2O in cc-pVDZ (s, p,
d) and cc-pVTZ (oxygen's f shells, so every class through (ff)), over this
tree's own `data/basis` files, spherical: the first derivatives of the
overlap, kinetic and nuclear-attraction matrices with respect to every
nuclear coordinate, as AO matrices,

    S^x_mn = dS_mn/dR_{A,k},  T^x_mn = dT_mn/dR_{A,k},  V^x_mn = dV_mn/dR_{A,k}

for x = 3 A + k, from libcint's `int1e_ipovlp`, `int1e_ipkin` and
`int1e_ipnuc` (each nabla on the bra's electron coordinate, so -d/dR of its
centre: the bra rows on A, plus their transpose) and, for V's operator at
nucleus A, `-Z_A int1e_iprinv` with the rinv origin at A plus its transpose,
as `pyscf.hessian.rhf.make_h1` assembles them through
`pyscf.grad.rhf.get_hcore`'s generator. The tool checks T^x + V^x against
that generator's own matrices, every matrix against a fourth-order central
difference of `int1e_ovlp`, `int1e_kin` and `int1e_nuc` (step 1e-3 bohr), and
each sum over atoms against zero.

Each matrix set is (nao, nao, 3 n_atom) in Fortran order, element (m, n, x)
at m + nao (n + nao x): matrix x at x nao^2, the layout
`overlap_derivative` and its two siblings return. Arrays are '<f8'
Fortran-order .npy, the layout `src/io/npy` reads; the manifest
`one_electron_derivative.json` names them. Functions are PySCF's
unit-normalised ones. BLAS and OpenMP are pinned to one thread and nothing
run-dependent is written, so a rerun is byte-identical.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/one_electron_derivative_reference.py

writes test/gtest/data/one_electron_derivative/. Re-run it to regenerate; the
files are never hand-edited.
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
)
from pyscf import lib  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "one_electron_derivative"
MANIFEST = OUT / "one_electron_derivative.json"

# The finite-difference check: fourth-order central difference at this step
# (bohr), and the agreement it must reach, relative to max(1, max |X^x|).
FD_STEP = 1e-3
FD_TOLERANCE = 1e-7
# T^x + V^x against PySCF's own hcore generator, relative to max |h^x|.
HCORE_TOLERANCE = 1e-13

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/one_electron_derivative_reference.py"
)

INTEGRALS = {
    "overlap": "int1e_ovlp",
    "kinetic": "int1e_kin",
    "potential": "int1e_nuc",
}


def centre_terms(mol, ip: numpy.ndarray) -> numpy.ndarray:
    """[3 n_atom, nao, nao]: for each atom A and axis k, -ip[k] on the bra
    rows of A's functions plus that block's transpose, ip libcint's ip1."""
    out = numpy.zeros((3 * mol.natm, mol.nao, mol.nao))
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        block = numpy.zeros((3, mol.nao, mol.nao))
        block[:, p0:p1] = -ip[:, p0:p1]
        out[3 * a : 3 * a + 3] = block + block.transpose(0, 2, 1)
    return out


def derivatives(mol) -> dict:
    """S^x, T^x, V^x as [3 n_atom, nao, nao]."""
    potential = centre_terms(mol, mol.intor("int1e_ipnuc"))
    for a in range(mol.natm):
        with mol.with_rinv_at_nucleus(a):
            rinv = -mol.atom_charge(a) * mol.intor("int1e_iprinv")
        potential[3 * a : 3 * a + 3] += rinv + rinv.transpose(0, 2, 1)
    return {
        "overlap": centre_terms(mol, mol.intor("int1e_ipovlp")),
        "kinetic": centre_terms(mol, mol.intor("int1e_ipkin")),
        "potential": potential,
    }


def check_hcore(mol, got: dict) -> None:
    """T^x + V^x against pyscf.grad.rhf's hcore generator, the one-electron
    part pyscf.hessian.rhf.make_h1 builds on."""
    generator = rhf_grad.Gradients(mol.RHF()).hcore_generator(mol)
    h = got["kinetic"] + got["potential"]
    want = numpy.concatenate([generator(a) for a in range(mol.natm)])
    error = numpy.abs(h - want).max()
    assert error < HCORE_TOLERANCE * numpy.abs(want).max(), error


def check_fd(case: dict, got: dict) -> None:
    """Every matrix against a fourth-order central difference of its
    integral matrix over each nuclear coordinate."""
    coords = numpy.array([coord for _, coord in case["atoms"]])
    for key, intor in INTEGRALS.items():
        numeric = numpy.zeros_like(got[key])
        for a in range(coords.shape[0]):
            for k in range(3):
                values = {}
                for s in (-2, -1, 1, 2):
                    moved = coords.copy()
                    moved[a, k] += s * FD_STEP
                    values[s] = molecules(case, moved)[0].intor(intor)
                numeric[3 * a + k] = (
                    8.0 * (values[1] - values[-1]) - (values[2] - values[-2])
                ) / (12.0 * FD_STEP)
        error = numpy.abs(got[key] - numeric).max()
        scale = max(1.0, numpy.abs(got[key]).max())
        assert error < FD_TOLERANCE * scale, (case["name"], key, error)


def check_translation(name: str, matrices: numpy.ndarray) -> None:
    natm = matrices.shape[0] // 3
    drift = numpy.abs(matrices.reshape(natm, 3, *matrices.shape[1:]).sum(axis=0))
    assert drift.max() < 1e-12 * max(1.0, numpy.abs(matrices).max()), (
        name,
        drift.max(),
    )


def save(name: str, matrices: numpy.ndarray) -> str:
    """Write [3 n_atom, nao, nao] as (nao, nao, 3 n_atom), '<f8' Fortran
    order; its file name."""
    path = OUT / f"{name}.npy"
    array = numpy.asfortranarray(matrices.transpose(1, 2, 0), dtype="<f8")
    numpy.save(path, array, allow_pickle=False)
    return path.name


def case_entry(case: dict) -> dict:
    mol, _ = molecules(case)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == case["max_l"]
    got = derivatives(mol)
    for key, value in got.items():
        assert numpy.array_equal(value, value.transpose(0, 2, 1)), key
        check_translation(key, value)
    check_hcore(mol, got)
    check_fd(case, got)
    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "matrices": {
            key: save(f"{case['name']}_{key}", value) for key, value in got.items()
        },
    }


def main() -> None:
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.npy"):
        stale.unlink()
    doc = {
        "description": "PySCF/libcint one-electron derivative matrices over this "
        "tree's data/basis files, spherical, PySCF's unit-normalised functions: "
        "dS/dR, dT/dR and dV/dR (V's operator term included) for every atom A and "
        "axis k, matrix x = 3 A + k, each array (nao, nao, 3 n_atom), element "
        "(m, n, x) at m + nao (n + nao x); coordinates in bohr. Arrays are '<f8' "
        "Fortran-order .npy files beside this one",
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
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
