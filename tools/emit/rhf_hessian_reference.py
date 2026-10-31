#!/usr/bin/env python3
"""Write the reference data the DF-RHF Hessian's response-part and
two-electron explicit-part tests read.

The response part of the closed-shell DF-RHF nuclear Hessian is everything the
orbitals' first-order response contributes: PySCF's `pyscf.df.hessian.rhf`
total less its explicit part, `partial_hess_elec` + `hess_nuc` (the second
derivative integrals contracted with the converged D, W and D (x) D). The
design (`rhf_hessian.tex`'s hess-sym) writes it

    -4 B^x . U^y - 2 (Sigma + Sigma^T) + 2 Omega + Theta,

over the CPHF right-hand sides B^x and solutions U^x; `src/hessian/README.md`
has each term.

PySCF's own `hess_elec` is used with `mo1` and `mo_e1` handed to it: its
`solve_mo1` gives a block of right-hand sides to one Krylov solve, whose
Gram-Schmidt stalls near 1e-7 in the residual (`cphf_reference.py` has the
numbers), so each column is solved alone here through the same `scf.cphf.solve`
and response function, to 1e-12 with the Krylov `lindep` lowered so the
tolerance decides. The response part written is then that `hess_elec` minus
`partial_hess_elec`, which is the total minus (`partial_hess_elec` +
`hess_nuc`). It is checked against:

- PySCF's unmodified `Hessian().kernel()` less the same explicit part, to
  what its stalled solve reaches;
- the symmetric form above built from PySCF's pieces (`make_h1`,
  `int1e_ipovlp`, DF `get_jk`) with U from a dense solve of
  M = diag(eps_a - eps_i) + 4 (ai|bj) - (ab|ij) - (aj|ib) over PySCF's DF MO
  integrals, to 1e-9 of its largest element.

Cases, over this tree's own basis files (as `gradient_reference.py` hands them
to PySCF), spherical throughout, at PySCF's converged DF-RHF orbitals
(`conv_tol` 1e-12):

- `h2o_ccpvdz`: the skew water of `density_fitting/` in cc-pVDZ /
  cc-pVDZ-JKFIT;
- `h2co_ccpvdz`: a skew formaldehyde, no symmetry plane, in the same sets, two
  heavy atoms with d shells and twelve coordinates.

Arrays are .npy, '<f8' in Fortran order (the layout `src/io/npy` reads, first
axis fastest): `c` (nao x nao, the orbitals, columns ascending in energy),
`response` (3 n_atom x 3 n_atom, element (x, y) the response part's
d^2E/dx dy, x = 3 A + k), `u` ((n_vir n_occ) x 3 n_atom, U^x_ai at
a + n_vir i of column x, `response::cphf`'s layout), and `coulomb` and
`exchange` (3 n_atom x 3 n_atom): the explicit part's two-electron terms,
`_partial_hess_ejk`'s ej and -ek (partial_hess_elec is e1 + ej - ek), which
`hessian::df_rhf_two_electron` returns, each checked against
partial_hess_elec and for the translational sum rule. U is in the MO basis, so
neither it nor the Hessian depends on the AOs' normalisation; C is PySCF's,
over unit-normalised functions, and a test whose function i is N_i times
PySCF's divides C's rows by N. BLAS and OpenMP are pinned to one thread, and
nothing run-dependent is written, so a rerun regenerates every file byte for
byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/rhf_hessian_reference.py

writes test/gtest/data/rhf_hessian/. Re-run it to regenerate; the files are
never hand-edited.
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
from cphf_reference import KRYLOV_LINDEP, dense_m  # noqa: E402
from df_reference import WATER_SKEW, libcint_version  # noqa: E402
from gradient_reference import atoms_out, df_rhf, molecules  # noqa: E402
from pyscf import lib  # noqa: E402
from pyscf.df.hessian import rhf as df_hessian  # noqa: E402
from pyscf.hessian import rhf as rhf_hessian  # noqa: E402
from pyscf.scf import cphf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "rhf_hessian"
MANIFEST = OUT / "rhf_hessian.json"

# Formaldehyde bent out of its plane and stretched unevenly, in no symmetry
# plane, so no Hessian element is zero by symmetry. Bohr.
FORMALDEHYDE_SKEW = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.12, 0.08, 2.31)),
    ("H", (1.78, 0.25, -1.02)),
    ("H", (-1.69, -0.37, -1.18)),
]

CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
    },
    {
        "name": "h2co_ccpvdz",
        "atoms": FORMALDEHYDE_SKEW,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
    },
]

CPHF_TOLERANCE = 1e-12
# The symmetric form over a dense solve against the written response part,
# of its largest element.
DENSE_TOLERANCE = 1e-9
# PySCF's own kernel against it: its block Krylov stalls near 1e-7 in U,
# which leaves 3e-8 of the largest element on the water.
STALLED_TOLERANCE = 1e-6

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/rhf_hessian_reference.py"
)


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy; its file name."""
    path = OUT / f"{name}.npy"
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.name


def flat(hessian: numpy.ndarray) -> numpy.ndarray:
    """PySCF's (A, B, k, l) to (3 A + k, 3 B + l)."""
    n = 3 * hessian.shape[0]
    return hessian.transpose(0, 2, 1, 3).reshape(n, n)


def overlap_derivatives(mol) -> numpy.ndarray:
    """S^x, (3 n_atom, nao, nao), as solve_mo1 builds them."""
    s1 = -mol.intor("int1e_ipovlp", comp=3)
    out = numpy.zeros((3 * mol.natm, mol.nao, mol.nao))
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        for k in range(3):
            m = numpy.zeros((mol.nao, mol.nao))
            m[p0:p1] = s1[k, p0:p1]
            out[3 * atom + k] = m + m.T
    return out


def column_mo1(mf, h1ao):
    """solve_mo1's mo1 and mo_e1, one column per scf.cphf.solve, to
    CPHF_TOLERANCE: per atom, mo1 (3, nao, nocc) in the AO basis and mo_e1
    (3, nocc, nocc)."""
    mol = mf.mol
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    nocc = int((occ > 0).sum())
    fx = rhf_hessian.gen_vind(mf, c, occ)
    sx = overlap_derivatives(mol)
    mo1s, e1s = [], []
    for atom in range(mol.natm):
        mo1_atom, e1_atom = [], []
        for k in range(3):
            h1 = c.T @ h1ao[atom][k] @ c[:, :nocc]
            s1 = c.T @ sx[3 * atom + k] @ c[:, :nocc]
            mo1, e1 = cphf.solve(
                fx,
                eps,
                occ,
                h1[None],
                s1[None],
                max_cycle=200,
                tol=CPHF_TOLERANCE,
            )
            mo1_atom.append(c @ mo1[0])
            e1_atom.append(e1[0])
        mo1s.append(numpy.array(mo1_atom))
        e1s.append(numpy.array(e1_atom))
    return mo1s, e1s


def symmetric_form(mf, h1ao) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The design's hess-sym response part from PySCF's pieces, U from a
    dense solve; and U as (n_vir n_occ) x 3 n_atom."""
    mol = mf.mol
    c, eps = mf.mo_coeff, mf.mo_energy
    nocc = int((mf.mo_occ > 0).sum())
    o, v = slice(0, nocc), slice(nocc, None)
    n = 3 * mol.natm
    fx_ao = numpy.array([h1ao[x // 3][x % 3] for x in range(n)])
    sx_ao = overlap_derivatives(mol)
    f_mo = numpy.einsum("mp,xmn,nq->xpq", c, fx_ao, c)
    s_mo = numpy.einsum("mp,xmn,nq->xpq", c, sx_ao, c)

    def g(density):
        vj, vk = mf.get_jk(mol, density, hermi=1)
        return vj - 0.5 * vk

    d_s = numpy.array([2 * c[:, o] @ s_mo[x][o, o] @ c[:, o].T for x in range(n)])
    g_s = numpy.array([g(d_s[x]) for x in range(n)])
    b = (
        -f_mo[:, v, o]
        + s_mo[:, v, o] * eps[None, None, :nocc]
        + numpy.einsum("ma,xmn,ni->xai", c[:, v], g_s, c[:, o])
    )
    # Columns a + n_vir i, as dense_m's rows.
    b_cols = numpy.stack([x.reshape(-1, order="F") for x in b], axis=1)
    u_cols = numpy.linalg.solve(dense_m(mf, nocc), b_cols)
    h = -4 * b_cols.T @ u_cols
    soo, foo = s_mo[:, o, o], f_mo[:, o, o]
    sigma = numpy.einsum("xij,yij->xy", soo, foo)
    h -= 2 * (sigma + sigma.T)
    h += 2 * numpy.einsum("xij,yij,ij->xy", soo, soo, eps[o, None] + eps[None, o])
    h += numpy.einsum("xmn,ymn->xy", d_s, g_s)
    return h, u_cols


def hessian_case(case: dict) -> dict:
    mol, auxmol = molecules(case)
    mf = df_rhf(mol, auxmol)
    mf.conv_tol_cpscf = CPHF_TOLERANCE
    nocc = int((mf.mo_occ > 0).sum())
    hobj = mf.Hessian()
    assert isinstance(hobj, df_hessian.Hessian)
    assert hobj.auxbasis_response == 2
    hobj.max_cycle = 200
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ

    h1ao = hobj.make_h1(c, occ)
    mo1, mo_e1 = column_mo1(mf, h1ao)
    explicit = hobj.partial_hess_elec(eps, c, occ)
    # The explicit part's two-electron terms on their own: partial_hess_elec
    # is e1 + ej - ek, the one-electron e1 beside the fitted Coulomb ej and
    # exchange ek.
    e1, ej, ek = df_hessian._partial_hess_ejk(hobj, eps, c, occ)
    assert numpy.abs(e1 + ej - ek - explicit).max() < 1e-12, case["name"]
    coulomb, exchange = flat(ej), flat(-ek)
    for part, term in (("coulomb", coulomb), ("exchange", exchange)):
        # The two-electron energy at fixed orbitals is translationally
        # invariant: every row of each part sums to zero over the atoms.
        drift = numpy.abs(term.reshape(-1, mol.natm, 3).sum(axis=1)).max()
        assert drift < 1e-8 * numpy.abs(term).max(), (case["name"], part, drift)
    total = hobj.hess_elec(eps, c, occ, mo1=mo1, mo_e1=mo_e1, h1ao=h1ao)
    response = flat(total - explicit)
    scale = numpy.abs(response).max()

    dense, u = symmetric_form(mf, h1ao)
    err = numpy.abs(dense - response).max()
    assert err < DENSE_TOLERANCE * scale, (case["name"], "dense", err, scale)
    stalled = flat(hobj.kernel() - hobj.hess_nuc() - explicit)
    err = numpy.abs(stalled - response).max()
    assert err < STALLED_TOLERANCE * scale, (case["name"], "kernel", err, scale)
    # U against the columns solved alone, through C^T S.
    overlap = mol.intor("int1e_ovlp")
    for x in range(3 * mol.natm):
        alone = (c.T @ overlap @ mo1[x // 3][x % 3])[nocc:]
        err = numpy.abs(alone.reshape(-1, order="F") - u[:, x]).max()
        assert err < DENSE_TOLERANCE * numpy.abs(u).max(), (case["name"], x, err)
    print(
        f"{case['name']}: max |response| {scale:.4f}, asymmetry "
        f"{numpy.abs(response - response.T).max():.2e}"
    )

    name = case["name"]
    return {
        "name": name,
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "nocc": nocc,
        "energy": float(mf.e_tot),
        "mo_energies": [float(e) for e in eps],
        "c": save(f"{name}_c", c),
        "response": save(f"{name}_response", response),
        "u": save(f"{name}_u", u),
        "coulomb": save(f"{name}_coulomb", coulomb),
        "exchange": save(f"{name}_exchange", exchange),
    }


def main() -> None:
    lib.num_threads(1)
    lib.krylov = functools.partial(lib.krylov, lindep=KRYLOV_LINDEP)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.npy"):
        stale.unlink()
    doc = {
        "description": "The response part of PySCF's DF-RHF nuclear Hessian "
        "(pyscf.df.hessian.rhf, auxbasis response 2): its total less "
        "partial_hess_elec and hess_nuc, over CPHF solutions to 1e-12, with "
        "the orbitals, their energies and the CPHF solutions U^x, and the "
        "explicit part's two-electron terms (_partial_hess_ejk's ej and -ek), "
        "over this "
        "tree's data/basis and data/aux_basis files, spherical, "
        "unit-normalised functions; coordinates in bohr, the Hessian in "
        "hartree / bohr^2. Arrays are '<f8' Fortran-order .npy files beside "
        "this one",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": [hessian_case(case) for case in CASES],
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
