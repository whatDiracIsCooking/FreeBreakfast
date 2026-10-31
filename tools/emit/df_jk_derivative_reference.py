#!/usr/bin/env python3
"""Write the reference data the DF J/K derivative tests read.

The two-electron half of the DF-RHF skeleton Fock derivatives: per nuclear
coordinate x = 3 A + k, J^(x) = dJ/dx and K^(x) = dK/dx at a fixed density
D = 2 C_occ C_occ^T, the integrals and the Coulomb metric both differentiated
(PySCF's auxbasis response at its default, 2), so that

    make_h1[A][k] = hcore_deriv(A)[k] + J^(x) - 1/2 K^(x)

with J_mn = sum (mn|ls) D_ls and K_mn = sum (ml|ns) D_ls over the fit
(mn|ls) = sum (mn|P) (P|Q)^-1 (Q|ls). J^(x) and K^(x) come from
`pyscf.df.hessian.rhf._gen_jk`, the generator `make_h1` itself sums, and the
tool asserts that sum against `make_h1` minus `hcore_deriv`, and J^(x) =
T^x d + T M^-1 (u^x - M^x d) (u = (P|mn) D_mn, d = M^-1 u, M = (P|Q)) over
libcint's `int3c2e_ip1`, `int3c2e_ip2` and `int2c2e_ip1`.

Nothing is written of u^x - M^x d itself, nor of any contraction of it short
of J^(x): those carry d, which is only as well determined as M is
conditioned. A 1e-15 relative perturbation of (mn|P) and (P|Q) moves the
Gram matrix (u^x - M^x d)^T M^-1 (u^y - M^y d) by 2.6e-10 relative in
cc-pVDZ-JKFIT and 6.6e-9 in cc-pVTZ-acCD-6-6 (and T M^-1 (u^x - M^x d) alike),
so no independent integral code can be held to 1e-10 there; J^(x) as a whole
is stable to round-off.

Every J^(x) and K^(x) is also checked against a fourth-order central
difference of PySCF's own DF `get_jk` at fixed D (step 1e-3 bohr), and each
slab's symmetry.

Cases, over this tree's own basis files (as gradient_reference.py hands them
to PySCF), spherical throughout, at PySCF's converged DF-RHF orbitals
(`conv_tol` 1e-12):

- `h2o_ccpvdz`: `density_fitting/`'s skew water in cc-pVDZ / cc-pVDZ-JKFIT
  (orbitals to d, aux to f);
- `h2o_ccpvtz`: the same water in cc-pVTZ / cc-pVTZ-JKFIT (orbitals to f,
  aux to g);
- `h2o_ccpvtz_accd66`: the same water in cc-pVTZ / cc-pVTZ-acCD-6-6 (aux to
  i, so (ff|i) and (i|i)).

Arrays are .npy, '<f8' in Fortran order (the layout `src/io/npy` reads, first
axis fastest): `c` (nao x nao, the orbitals, columns ascending in energy), `jx`
and `kx` (nao x nao x 3 n_atom, slab x the coordinate's matrix). The manifest
`df_jk_derivative.json` names the files and carries the atoms, `nocc`, the
orbital energies. Every function is PySCF's, unit-normalised: a test whose
function i is N_i times PySCF's divides C's rows by N, and compares
J^(x)_mn N_m N_n. BLAS and OpenMP are
pinned to one thread, and nothing run-dependent (date, host) is written, so
a rerun regenerates every file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/df_jk_derivative_reference.py

writes test/gtest/data/df_jk_derivative/. Re-run it to regenerate; the files
are never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from df_reference import WATER_SKEW, libcint_version  # noqa: E402
from gradient_reference import df_rhf, molecules  # noqa: E402
from pyscf import df, lib, scf  # noqa: E402
from pyscf.df.hessian import rhf as df_hessian  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "df_jk_derivative"
MANIFEST = OUT / "df_jk_derivative.json"

# A file over this many bytes would need splitting; none of these cases does.
LIMIT = 512 * 1024

CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
    },
    {
        "name": "h2o_ccpvtz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvtz",
        "aux_basis": "cc-pvtz-jkfit",
    },
    {
        "name": "h2o_ccpvtz_accd66",
        "atoms": WATER_SKEW,
        "basis": "cc-pvtz",
        "aux_basis": "cc-pvtz-accd-6-6",
    },
]

# The checks: make_h1 against the generator's sum, and the J identity, both
# exact up to round-off; the finite difference (fourth order, step in bohr)
# relative to max(1, max |J^(x)|).
SAME_TOLERANCE = 1e-11
FD_STEP = 1e-3
FD_TOLERANCE = 1e-7

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/df_jk_derivative_reference.py"
)


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy, under the limit; its file name."""
    array = numpy.asfortranarray(array, dtype="<f8")
    path = OUT / f"{name}.npy"
    numpy.save(path, array, allow_pickle=False)
    assert path.stat().st_size <= LIMIT, f"{path.name} is over {LIMIT} bytes"
    return path.name


def jk_derivatives(mf: scf.hf.SCF) -> tuple[numpy.ndarray, numpy.ndarray]:
    """J^(x) and K^(x), (3 n_atom, nao, nao), from the generator make_h1
    sums, checked against make_h1 minus the core Hamiltonian's derivative."""
    mol = mf.mol
    hessian = df_hessian.Hessian(mf)
    assert hessian.auxbasis_response == 2
    jx = numpy.zeros((3 * mol.natm, mol.nao, mol.nao))
    kx = numpy.zeros_like(jx)
    for atom, _, vj1, vk1 in df_hessian._gen_jk(hessian, mf.mo_coeff, mf.mo_occ):
        jx[3 * atom : 3 * atom + 3] = vj1
        kx[3 * atom : 3 * atom + 3] = vk1
    h1 = hessian.make_h1(mf.mo_coeff, mf.mo_occ)
    hcore = mf.nuc_grad_method().hcore_generator(mol)
    for atom in range(mol.natm):
        two = numpy.asarray(h1[atom]) - hcore(atom)
        diff = numpy.abs(
            two - (jx[3 * atom : 3 * atom + 3] - 0.5 * kx[3 * atom : 3 * atom + 3])
        )
        assert diff.max() < SAME_TOLERANCE, diff.max()
    for x in (jx, kx):
        assert numpy.abs(x - x.transpose(0, 2, 1)).max() < SAME_TOLERANCE
    return jx, kx


def fitted_jk(case: dict, coords: numpy.ndarray, density: numpy.ndarray):
    """PySCF's DF J and K of a fixed density at coords (bohr)."""
    mol, auxmol = molecules(case, coords)
    mf = scf.RHF(mol).density_fit(auxbasis=auxmol._basis)
    return mf.get_jk(mol, density, hermi=1)


def check_fd(
    case: dict, density: numpy.ndarray, jx: numpy.ndarray, kx: numpy.ndarray
) -> None:
    """Each J^(x) and K^(x) against a fourth-order central difference of
    the fitted J and K at fixed D."""
    coords = numpy.array([coord for _, coord in case["atoms"]], dtype=float)
    for x in range(coords.size):
        sides = {}
        for step in (-2, -1, 1, 2):
            moved = coords.copy()
            moved.flat[x] += step * FD_STEP
            sides[step] = fitted_jk(case, moved, density)
        for i, analytic in enumerate((jx[x], kx[x])):
            numeric = (
                sides[-2][i] - 8 * sides[-1][i] + 8 * sides[1][i] - sides[2][i]
            ) / (12 * FD_STEP)
            scale = max(1.0, numpy.abs(analytic).max())
            err = numpy.abs(numeric - analytic).max()
            assert err < FD_TOLERANCE * scale, (case["name"], x, i, err)


def check_fit_identity(mol, auxmol, density: numpy.ndarray, jx: numpy.ndarray) -> None:
    """J^(x) = T^x d + T M^-1 (u^x - M^x d) over libcint's derivative
    integrals, for every coordinate."""
    t = df.incore.aux_e2(mol, auxmol, "int3c2e")
    ip1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip1", comp=3)
    ip2 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip2", comp=3)
    metric = auxmol.intor("int2c2e")
    metric_ip1 = auxmol.intor("int2c2e_ip1")
    d = numpy.linalg.solve(metric, numpy.einsum("mnP,mn->P", t, density))
    aos = mol.aoslice_by_atom()[:, 2:]
    aux = auxmol.aoslice_by_atom()[:, 2:]
    for atom in range(mol.natm):
        m0, m1 = aos[atom]
        p0, p1 = aux[atom]
        for k in range(3):
            # d/dR_Ak of (mn|P): libcint's ip1 and ip2 are -d/d of their centre.
            t_x = numpy.zeros_like(t)
            t_x[m0:m1] -= ip1[k, m0:m1]
            t_x[:, m0:m1] -= ip1[k, m0:m1].transpose(1, 0, 2)
            t_x[:, :, p0:p1] -= ip2[k, :, :, p0:p1]
            m_x = numpy.zeros_like(metric)
            m_x[p0:p1] -= metric_ip1[k, p0:p1]
            m_x[:, p0:p1] -= metric_ip1[k, p0:p1].T
            x = 3 * atom + k
            g = numpy.einsum("mnP,mn->P", t_x, density) - m_x @ d
            rebuilt = t_x @ d + t @ numpy.linalg.solve(metric, g)
            err = numpy.abs(rebuilt - jx[x]).max()
            assert err < SAME_TOLERANCE * max(1.0, numpy.abs(jx[x]).max()), err


def jk_case(case: dict) -> dict:
    mol, auxmol = molecules(case)
    mf = df_rhf(mol, auxmol)
    nocc = int((mf.mo_occ > 0).sum())
    density = mf.make_rdm1()
    jx, kx = jk_derivatives(mf)
    check_fd(case, density, jx, kx)
    check_fit_identity(mol, auxmol, density, jx)
    name = case["name"]
    return {
        "name": name,
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": [
            {
                "symbol": mol.atom_pure_symbol(a),
                "z": int(mol.atom_charge(a)),
                "coord": [float(x) for x in mol.atom_coord(a)],
            }
            for a in range(mol.natm)
        ],
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "nocc": nocc,
        "mo_energies": [float(e) for e in mf.mo_energy],
        "c": save(f"{name}_c", mf.mo_coeff),
        # (nao, nao, 3 n_atom): slab x the matrix of coordinate x.
        "jx": save(f"{name}_jx", jx.transpose(1, 2, 0)),
        "kx": save(f"{name}_kx", kx.transpose(1, 2, 0)),
    }


def main() -> None:
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "PySCF's DF-RHF skeleton Fock derivatives' two-electron "
        "half, J^(x) and K^(x) at fixed D = 2 C_occ C_occ^T (make_h1 = "
        "hcore_deriv + J^(x) - 1/2 K^(x), auxbasis response 2), over this "
        "tree's data/basis and data/aux_basis files, spherical, "
        "unit-normalised functions; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": [jk_case(case) for case in CASES],
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
