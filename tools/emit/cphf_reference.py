#!/usr/bin/env python3
"""Write the reference data the CPHF solver's tests read.

The case is `df_reference.py`'s `h2o_ccpvdz` integrals case, skew H2O in
cc-pVDZ / cc-pVDZ-JKFIT over this tree's own `data/basis` and
`data/aux_basis` files: its (P|Q) and (mn|P) are already in
`test/gtest/data/density_fitting/`, so a test fits B from those, and this tool
refuses to write unless it recomputes them exactly. Over that molecule:

- PySCF's DF-RHF (`conv_tol` 1e-12) orbitals C and orbital energies.
- The nuclear-displacement CPHF of `pyscf.df.hessian.rhf`: the right-hand
  sides B^x_ai = -F^(x)_ai + eps_i S^x_ai + [C_v^T G(D_S^x) C_o]_ai,
  D_S^x = 2 C_o S^x_oo C_o^T, G = J - K/2 over the same fitting set, from its
  `make_h1` and `int1e_ipovlp`, one column per nuclear coordinate; and the
  virtual-occupied block of its `solve_mo1`'s `mo1` (`conv_tol_cpscf` 1e-12,
  its Krylov solver's `lindep` lowered so that tolerance is reached, and one
  column at a time, see `nuclear`), the solution U of M U^x = B^x. The tool
  checks U against a dense solve of
  M = diag(eps_a - eps_i) + 4 (ai|bj) - (ab|ij) - (aj|ib) over PySCF's DF MO
  integrals.
- The static dipole polarizability: right-hand sides -r^x_ai (the AO dipole
  `int1e_r` in the MO basis) and alpha_xy = -4 sum_ai r^x_ai U^y_ai, PySCF's
  `pyscf.prop.polarizability.rhf` formula over its own CPHF (`scf.cphf`) and
  the same response function, checked against a finite field: the central
  difference of the DF-RHF dipole moment.

Every right-hand side and solution is (n_vir, n_occ, ncol) in Fortran order,
element (a, i, x) at a + n_vir (i + n_occ x): the [n_vir n_occ] x ncol
column-major block `response::cphf` takes. Arrays are '<f8' Fortran-order
.npy, the layout `src/io/npy` reads; the manifest `cphf.json` names them. BLAS
and OpenMP are pinned to one thread and nothing run-dependent is written, so a
rerun is byte-identical.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/cphf_reference.py

writes test/gtest/data/cphf/. Re-run it to regenerate (after
`df_reference.py`, whose integrals it reads); the files are never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so the output is reproducible.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import functools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from df_reference import (  # noqa: E402
    INTEGRAL_CASES,
    df_scf,
    libcint_version,
    molecules,
)
from df_reference import OUT as DF_OUT  # noqa: E402
from pyscf import df, lib  # noqa: E402
from pyscf.df.hessian import rhf as df_hessian  # noqa: E402
from pyscf.hessian import rhf as rhf_hessian  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "cphf"
MANIFEST = OUT / "cphf.json"
CASE = "h2o_ccpvdz"
# The finite field's step, atomic units: its O(h^2) error is ~1e-8 of alpha,
# the SCF's 1e-12 energy convergence ~1e-6 of the dipole difference.
FIELD_STEP = 1e-3

# PySCF's Krylov solver stops once a new direction's squared norm is under
# `lindep`, 1e-13 by default, whatever `tol` asks: here that leaves mo1 with a
# residual of 7e-7, well above the 1e-8 the C++ solve is held to. Lowered so
# `tol` decides, for every CPHF this tool runs.
KRYLOV_LINDEP = 1e-30

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/cphf_reference.py"
)


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy; its file name."""
    path = OUT / f"{name}.npy"
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.name


def check_integrals(case: dict, mol, auxmol) -> dict:
    """The density_fitting/ entry of case, after checking its (P|Q) and (mn|P)
    are what this molecule gives, so a test's B is this molecule's."""
    manifest = json.loads((DF_OUT / "density_fitting.json").read_text())
    entry = next(e for e in manifest["integrals"] if e["name"] == case["name"])
    metric = numpy.load(DF_OUT / entry["int2c2e"])
    assert numpy.array_equal(metric, auxmol.intor("int2c2e_sph")), "stale int2c2e"
    eri3 = df.incore.aux_e2(mol, auxmol, "int3c2e_sph", aosym="s1")
    for piece in entry["int3c2e"]:
        begin, end = piece["aux"]
        stored = numpy.load(DF_OUT / piece["file"])
        assert numpy.array_equal(stored, eri3[..., begin:end]), "stale int3c2e"
    return entry


def response(mf):
    """G(D) = J[D] - K[D]/2 over mf's fitting set, symmetric D."""

    def g(density):
        vj, vk = mf.get_jk(mf.mol, density, hermi=1)
        return vj - 0.5 * vk

    return g


def dense_m(mf, nocc: int) -> numpy.ndarray:
    """M over PySCF's DF MO integrals, row and column a + n_vir i."""
    c = mf.mo_coeff
    o, v = c[:, :nocc], c[:, nocc:]
    nv = v.shape[1]
    vovo = mf.with_df.ao2mo((v, o, v, o), compact=False).reshape(nv, nocc, nv, nocc)
    vvoo = mf.with_df.ao2mo((v, v, o, o), compact=False).reshape(nv, nv, nocc, nocc)
    # (a, i, b, j) <- 4 (ai|bj) - (ab|ij) - (aj|ib)
    a = 4 * vovo - vvoo.transpose(0, 2, 1, 3) - vovo.transpose(0, 3, 2, 1)
    eps = mf.mo_energy
    a += numpy.einsum(
        "ab,ij->aibj", numpy.diag(eps[nocc:]), numpy.eye(nocc)
    ) - numpy.einsum("ab,ij->aibj", numpy.eye(nv), numpy.diag(eps[:nocc]))
    # Fortran order: index a + nv i.
    return a.transpose(1, 0, 3, 2).reshape(nv * nocc, nv * nocc)


def as_columns(block: numpy.ndarray) -> numpy.ndarray:
    """(ncol, n_vir, n_occ) to the (n_vir n_occ) x ncol matrix, a + n_vir i."""
    return numpy.stack([x.reshape(-1, order="F") for x in block], axis=1)


def nuclear(mf, nocc: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The nuclear-displacement right-hand sides and mo1's vo block, as
    (n_vir n_occ) x 3 natm.

    mo1 is solve_mo1's, column by column: the h1 and s1 it builds, through
    the scf.cphf.solve and response function it calls. solve_mo1 itself hands
    a block of columns to one Krylov solve, whose classical Gram-Schmidt
    stalls near 1e-7 here; one column at a time reaches the tolerance. Its
    block answer is checked against this one to what it does reach.
    """
    mol = mf.mol
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    o, v = c[:, :nocc], c[:, nocc:]
    hobj = mf.Hessian()
    assert isinstance(hobj, df_hessian.Hessian)
    hobj.max_cycle = 200
    mf.conv_tol_cpscf = 1e-12
    h1ao = hobj.make_h1(c, occ)
    block, _ = hobj.solve_mo1(eps, c, occ, h1ao)
    fx = rhf_hessian.gen_vind(mf, c, occ)

    s1 = -mol.intor("int1e_ipovlp", comp=3)
    g = response(mf)
    overlap = mol.intor("int1e_ovlp")
    rhs, solution = [], []
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        for k in range(3):
            sx = numpy.zeros_like(overlap)
            sx[p0:p1] = s1[k, p0:p1]
            sx = sx + sx.T
            s_mo = c.T @ sx @ c
            h_mo = c.T @ h1ao[atom][k] @ c
            d_s = 2 * o @ s_mo[:nocc, :nocc] @ o.T
            rhs.append(
                -h_mo[nocc:, :nocc]
                + s_mo[nocc:, :nocc] * eps[None, :nocc]
                + v.T @ g(d_s) @ o
            )
            mo1, _ = cphf.solve(
                fx,
                eps,
                occ,
                h_mo[None, :, :nocc],
                s_mo[None, :, :nocc],
                max_cycle=200,
                tol=mf.conv_tol_cpscf,
            )
            solution.append(mo1[0, nocc:])
            # solve_mo1's is C times the MO-basis solution: back with C^T S.
            stalled = (c.T @ overlap @ block[atom][k])[nocc:]
            assert (
                numpy.abs(stalled - solution[-1]).max()
                < 1e-6 * numpy.abs(solution[-1]).max()
            )
    return as_columns(numpy.array(rhs)), as_columns(numpy.array(solution))


def polarizability(mf, nocc: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """-r_ai as (n_vir n_occ) x 3, and alpha, pyscf.prop's formula over
    scf.cphf, checked against a finite field."""
    mol = mf.mol
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    o = c[:, :nocc]
    r = mol.intor_symmetric("int1e_r", comp=3)
    h1 = numpy.einsum("xpq,pi,qj->xij", r, c, o)
    # One column at a time, as nuclear's, past the block Krylov's stall.
    fx = rhf_hessian.gen_vind(mf, c, occ)
    mo1 = numpy.concatenate(
        [
            cphf.solve(
                fx,
                eps,
                occ,
                h1[x : x + 1],
                numpy.zeros_like(h1[x : x + 1]),
                max_cycle=200,
                tol=1e-12,
            )[0]
            for x in range(3)
        ]
    )
    e2 = numpy.einsum("xpi,ypi->xy", h1, mo1)
    alpha = -2 * (e2 + e2.T)

    # Finite field: H = h + F . r for an electron, mu = -tr(D r) + nuclear,
    # alpha_xy = d mu_x / d F_y.
    hcore = mf.get_hcore()
    numeric = numpy.zeros((3, 3))
    for y in range(3):
        dipoles = []
        for sign in (1, -1):
            field = mf.copy()
            field.get_hcore = lambda *_, s=sign, a=y: hcore + s * FIELD_STEP * r[a]
            field.conv_tol = 1e-13
            field.kernel(dm0=mf.make_rdm1())
            assert field.converged
            dipoles.append(-numpy.einsum("xmn,nm->x", r, field.make_rdm1()))
        numeric[:, y] = (dipoles[0] - dipoles[1]) / (2 * FIELD_STEP)
    assert numpy.abs(numeric - alpha).max() < 1e-5 * numpy.abs(alpha).max(), (
        numeric,
        alpha,
    )
    return as_columns(-h1[:, nocc:, :]), alpha


def main() -> None:
    lib.num_threads(1)
    lib.krylov = functools.partial(lib.krylov, lindep=KRYLOV_LINDEP)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.npy"):
        stale.unlink()
    case = next(c for c in INTEGRAL_CASES if c["name"] == CASE)
    mol, auxmol, aux_name = molecules(case)
    entry = check_integrals(case, mol, auxmol)
    mf = df_scf(mol, auxmol)
    nocc = mol.nelectron // 2

    rhs, solution = nuclear(mf, nocc)
    m = dense_m(mf, nocc)
    dense = numpy.linalg.solve(m, rhs)
    assert numpy.abs(dense - solution).max() < 1e-9 * numpy.abs(solution).max()
    dipole_rhs, alpha = polarizability(mf, nocc)
    assert (
        numpy.abs(4 * dipole_rhs.T @ numpy.linalg.solve(m, dipole_rhs) - alpha).max()
        < 1e-9 * numpy.abs(alpha).max()
    )

    doc = {
        "description": "PySCF DF-RHF CPHF over this tree's data/basis and "
        "data/aux_basis files, spherical: the orbitals, orbital energies "
        "(hartree), the nuclear-displacement right-hand sides and "
        "pyscf.df.hessian.rhf solve_mo1's vo block, and the dipole right-hand "
        "sides and static polarizability (bohr^3). Each right-hand side and "
        "solution is (n_vir n_occ) x ncol, element a + n_vir i of column x. "
        "Arrays are '<f8' Fortran-order .npy files beside this one",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "integrals": f"density_fitting/density_fitting.json: {entry['name']}",
        "basis": case["basis"],
        "aux_basis": aux_name,
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "occupied": nocc,
        "energy": float(mf.e_tot),
        "mo_coefficients": save(f"{CASE}_mo_coeff", mf.mo_coeff),
        "mo_energies": save(f"{CASE}_mo_energy", mf.mo_energy),
        "nuclear": {
            "rhs": save(f"{CASE}_nuclear_rhs", rhs),
            "solution": save(f"{CASE}_nuclear_mo1", solution),
        },
        "dipole": {
            "rhs": save(f"{CASE}_dipole_rhs", dipole_rhs),
            "polarizability": save(f"{CASE}_polarizability", alpha),
        },
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
