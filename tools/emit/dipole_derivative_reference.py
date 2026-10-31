#!/usr/bin/env python3
"""Write the reference data the dipole derivative's tests read: the dipole
integrals' nuclear derivative contracted with a matrix, and the DF-RHF dipole
derivatives and IR intensities.

Two kinds of case, over this tree's own basis files (as `gradient_reference.py`
hands them to PySCF), spherical throughout:

- `integrals`: G[c, A, k] = SUM_mn M_mn d mu^c_mn / dR_{A,k}, mu^c = <m| r_c |n>
  about the origin (libcint's `int1e_r`), M the recipe 1 / (m + n + 1), for
  the skew water in cc-pVDZ and cc-pVTZ (f shells on the oxygen, so every
  class through (ff)): from libcint's `int1e_irp` (<m| r nabla |n>, its nine
  components r-major), checked here against central differences of `int1e_r`
  with the atom moved and against the sum rule SUM_A G[c, A, k] =
  delta_ck SUM_mn M_mn S_mn (moving every atom by t is moving the origin by -t).
- `rhf`: the DF-RHF dipole derivative at PySCF's converged orbitals
  (`conv_tol` 1e-12), dmu_c / dR_{A,k} with the electrons' charge -1 and the
  nuclei's Z_A,

      dmu/dx = -tr(D mu^x) - tr(D^x mu) + Z_A e_k,
      D^x    = 2 (C1^x C_o^T + C_o C1^x^T),

  C1^x = C U^x PySCF's mo1, the CPHF solved column by column to 1e-12 as
  `rhf_hessian_reference.py` solves it (its occupied block -S^x_oo / 2, so
  -tr(D^x mu) is the design's -4 U_ai mu_ai + 2 S^x_ij mu_ij), checked against
  central differences of PySCF's own DF-RHF dipole (h = 1e-3 bohr, each a
  converged SCF) to FD_TOLERANCE, and against the sum rule (a neutral
  molecule's dipole does not change under a translation). The cases are
  `rhf_hessian_reference.py`'s cc-pVDZ / cc-pVDZ-JKFIT skew water and skew
  formaldehyde, and its water at PySCF's DF-RHF minimum (its atoms read from
  `test/gtest/data/rhf_hessian/rhf_hessian.json`, so the two fixtures share
  one geometry), for which the harmonic frequencies of PySCF's Hessian
  (`thermo.harmonic_analysis`) and the IR intensities are written too:

      I_q = K |SUM_x L_qx dmu/dx|^2,   K = N_A / (12 eps_0 c^2) e^2 / u,

  L the normal modes as `harmonic_analysis`'s `norm_mode` (each unit
  mass-weighted eigenvector over sqrt(m_A / amu)), so dmu/dQ is in e / amu^1/2
  and K = 974.88 km/mol per e^2 / amu (CODATA 2018 / 2022 values, as
  `src/hessian/infrared.cpp` takes them).

Everything is written into one JSON manifest: matrices as rows, a 3 n_atom x 3
dipole derivative row x = 3 A + k, columns x, y, z; G as [c][A] rows of three.
The C and U the tree's unit test feeds its call are
`test/gtest/data/rhf_hessian/`'s, PySCF's same orbitals. BLAS and OpenMP are
pinned to one thread and nothing run-dependent is written, so a rerun
regenerates the file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/dipole_derivative_reference.py

writes test/gtest/data/dipole_derivative/dipole_derivative.json. Re-run it to
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
from gradient_reference import atoms_out, df_rhf, molecules, recipe  # noqa: E402
from pyscf import lib  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402
from rhf_hessian_reference import (  # noqa: E402
    CASES as HESSIAN_CASES,
)
from rhf_hessian_reference import (  # noqa: E402
    CPHF_TOLERANCE,
    column_mo1,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "dipole_derivative" / "dipole_derivative.json"
HESSIAN_MANIFEST = ROOT / "test" / "gtest" / "data" / "rhf_hessian" / "rhf_hessian.json"

INTEGRAL_CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER_SKEW, "basis": "cc-pvdz"},
    {"name": "h2o_ccpvtz", "atoms": WATER_SKEW, "basis": "cc-pvtz"},
]
RHF_CASES = ["h2o_ccpvdz", "h2co_ccpvdz", "h2o_ccpvdz_opt"]

# int1e_irp against central differences of int1e_r, of max |G|: the step's
# h^2 error.
INTEGRAL_STEP = 1e-4
INTEGRAL_FD_TOLERANCE = 1e-8
# The analytic dipole derivative against central differences of PySCF's own
# SCF dipole, absolute (e): h^2 and the SCF's convergence over 2h.
FD_STEP = 1e-3
FD_TOLERANCE = 1e-5
# SUM_A dmu/dR_{A,k}, absolute: the CPHF's tolerance.
SUM_RULE_TOLERANCE = 1e-8

# CODATA: the Avogadro constant (mol^-1), the vacuum permittivity (F / m), the
# speed of light (m / s), the elementary charge (C) and the dalton (kg).
AVOGADRO = 6.02214076e23
EPSILON_0 = 8.8541878188e-12
LIGHT_SPEED = 299792458.0
ELEMENTARY_CHARGE = 1.602176634e-19
DALTON = 1.66053906660e-27
# km / mol per (e / amu^1/2)^2.
KM_PER_MOL = (
    AVOGADRO
    / (12.0 * EPSILON_0 * LIGHT_SPEED**2)
    * ELEMENTARY_CHARGE**2
    / DALTON
    / 1000.0
)

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/dipole_derivative_reference.py"
)


def rows(a: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in a]


def dipole_integral_gradient(mol, matrix: numpy.ndarray) -> numpy.ndarray:
    """G[c, A, k] = SUM_mn M_mn d mu^c_mn / dR_{A,k}, (3, n_atom, 3), from
    int1e_irp: <m| r_c nabla_k |n> at 3 c + k, the nabla on the ket, so
    <nabla_k m| r_c |n> is its element (n, m); nabla is -d/dR of the
    function's centre."""
    irp = mol.intor("int1e_irp", comp=9).reshape(3, 3, mol.nao, mol.nao)
    g = numpy.zeros((3, mol.natm, 3))
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        # The derivative on m's centre, then on n's.
        g[:, a, :] -= numpy.einsum("cknm,mn->ck", irp[:, :, :, p0:p1], matrix[p0:p1])
        g[:, a, :] -= numpy.einsum("ckmn,mn->ck", irp[:, :, :, p0:p1], matrix[:, p0:p1])
    return g


def moved(mol, atom: int, k: int, shift: float):
    coords = mol.atom_coords().copy()
    coords[atom, k] += shift
    return mol.set_geom_(coords, unit="Bohr", inplace=False)


def integral_case(case: dict) -> dict:
    mol, _ = molecules(case)
    matrix = recipe(mol.nao, mol.nao)
    g = dipole_integral_gradient(mol, matrix)
    scale = numpy.abs(g).max()
    fd = numpy.zeros_like(g)
    for a in range(mol.natm):
        for k in range(3):
            plus = moved(mol, a, k, INTEGRAL_STEP).intor("int1e_r", comp=3)
            minus = moved(mol, a, k, -INTEGRAL_STEP).intor("int1e_r", comp=3)
            fd[:, a, k] = numpy.einsum("cmn,mn->c", plus - minus, matrix) / (
                2 * INTEGRAL_STEP
            )
    err = numpy.abs(fd - g).max()
    assert err < INTEGRAL_FD_TOLERANCE * scale, (case["name"], "fd", err, scale)
    trace = numpy.einsum("mn,mn->", matrix, mol.intor("int1e_ovlp"))
    drift = numpy.abs(g.sum(axis=1) - trace * numpy.eye(3)).max()
    assert drift < 1e-12 * scale, (case["name"], "sum rule", drift)
    print(f"integrals {case['name']}: max |G| {scale:.4f}, |G - FD| {err:.2e}")
    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "gradient": [rows(g[c]) for c in range(3)],
    }


def dipole(mf) -> numpy.ndarray:
    """The SCF dipole about the origin (e bohr), electrons -1, nuclei Z."""
    mol = mf.mol
    electronic = -numpy.einsum(
        "cmn,nm->c", mol.intor("int1e_r", comp=3), mf.make_rdm1()
    )
    return electronic + mol.atom_charges() @ mol.atom_coords()


def dipole_derivative(mf, mo1) -> numpy.ndarray:
    """dmu_c / dR_{A,k}, (3 n_atom, 3), from PySCF's mo1 (per atom,
    (3, nao, nocc) in the AO basis)."""
    mol = mf.mol
    c = mf.mo_coeff
    nocc = int((mf.mo_occ > 0).sum())
    c_occ = c[:, :nocc]
    mu = mol.intor("int1e_r", comp=3)
    explicit = dipole_integral_gradient(mol, mf.make_rdm1())
    out = numpy.zeros((3 * mol.natm, 3))
    for x in range(3 * mol.natm):
        a, k = divmod(x, 3)
        c1 = mo1[a][k]
        d1 = 2.0 * (c1 @ c_occ.T + c_occ @ c1.T)
        out[x] = -explicit[:, a, k] - numpy.einsum("cmn,nm->c", mu, d1)
        out[x, k] += mol.atom_charge(a)
    return out


def finite_difference(case: dict, atoms: list) -> numpy.ndarray:
    coords = numpy.array([coord for _, coord in atoms], dtype=float)
    out = numpy.zeros((coords.size, 3))
    dm0 = None
    for x in range(coords.size):
        values = []
        for sign in (1.0, -1.0):
            shifted = coords.copy()
            shifted.reshape(-1)[x] += sign * FD_STEP
            mol, auxmol = molecules(case, shifted)
            mf = df_rhf(mol, auxmol, dm0)
            values.append(dipole(mf))
        out[x] = (values[0] - values[1]) / (2 * FD_STEP)
    return out


def rhf_case(name: str, hessian_doc: dict) -> dict:
    case = next(c for c in HESSIAN_CASES if c["name"] == name)
    if case.get("optimize"):
        entry = next(c for c in hessian_doc["cases"] if c["name"] == name)
        atoms = [(a["symbol"], tuple(a["coord"])) for a in entry["atoms"]]
        case = {**case, "atoms": atoms}
    mol, auxmol = molecules(case)
    mf = df_rhf(mol, auxmol)
    mf.conv_tol_cpscf = CPHF_TOLERANCE
    hobj = mf.Hessian()
    h1ao = hobj.make_h1(mf.mo_coeff, mf.mo_occ)
    mo1, mo_e1 = column_mo1(mf, h1ao)
    analytic = dipole_derivative(mf, mo1)
    fd = finite_difference(case, case["atoms"])
    err = numpy.abs(fd - analytic).max()
    assert err < FD_TOLERANCE, (name, "fd", err)
    drift = numpy.abs(analytic.reshape(mol.natm, 3, 3).sum(axis=0)).max()
    assert drift < SUM_RULE_TOLERANCE, (name, "sum rule", drift)
    print(
        f"rhf {name}: max |dmu/dx| {numpy.abs(analytic).max():.4f}, |FD| {err:.2e}, "
        f"sum rule {drift:.2e}"
    )
    out = {
        "name": name,
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": atoms_out(mol),
        "dipole": [float(v) for v in dipole(mf)],
        "dipole_derivative": rows(analytic),
        "finite_difference": rows(fd),
    }
    if case.get("optimize"):
        eps, c, occ = mf.mo_energy, mf.mo_coeff, mf.mo_occ
        hessian = (
            hobj.hess_elec(eps, c, occ, mo1=mo1, mo_e1=mo_e1, h1ao=h1ao)
            + hobj.hess_nuc()
        )
        modes = thermo.harmonic_analysis(mol, hessian, imaginary_freq=False)
        norm_mode = modes["norm_mode"].reshape(-1, 3 * mol.natm)
        dq = norm_mode @ analytic
        intensities = KM_PER_MOL * (dq**2).sum(axis=1)
        out["frequencies"] = [float(f) for f in modes["freq_wavenumber"]]
        out["intensities"] = [float(i) for i in intensities]
        print(
            f"  frequencies {out['frequencies']},"
            f" intensities {out['intensities']} km/mol"
        )
    return out


def main() -> None:
    lib.num_threads(1)
    lib.krylov = functools.partial(lib.krylov, lindep=KRYLOV_LINDEP)
    hessian_doc = json.loads(HESSIAN_MANIFEST.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "The dipole integrals' nuclear derivative contracted with "
        "the recipe matrix 1 / (m + n + 1) (libcint's int1e_irp), as [c][A] rows "
        "of three; and PySCF's DF-RHF dipole derivatives (electrons -1, nuclei "
        "Z, origin 0), 3 n_atom rows x = 3 A + k of three components, from its "
        "CPHF solved column by column to 1e-12, with their central differences, "
        "and at the minimum the harmonic frequencies (cm^-1) and IR intensities "
        "(km/mol); over this tree's data/basis and data/aux_basis files, "
        "spherical, unit-normalised functions; coordinates in bohr",
        "km_per_mol": KM_PER_MOL,
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "integrals": [integral_case(case) for case in INTEGRAL_CASES],
        "rhf": [rhf_case(name, hessian_doc) for name in RHF_CASES],
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
