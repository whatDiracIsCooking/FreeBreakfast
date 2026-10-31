#!/usr/bin/env python3
"""Write the reference data the DF second-derivative integral tests read.

libcint's second-derivative density-fitting integrals, each contracted with a
fixed matrix into the 3N x 3N nuclear Hessian H[3 A + k][3 B + l]:

- `metric`: sum_PQ W_PQ d2(P|Q)/dR_Ak dR_Bl, from int2c2e_ipip1 (both
  derivatives on one function) and int2c2e_ip1ip2 (one on each);
- `three_center`: sum_mnP Gamma_mnP d2(mn|P)/dR_Ak dR_Bl, from int3c2e_ipip1
  (both on m, and on n through (mn|P) = (nm|P)), int3c2e_ipvip1 (one on m, one
  on n) and int3c2e_ip1ip2 (one on m or n, one on P); both on P by
  translational invariance, d_P d_P = -(d_m + d_n) d_P, and checked here
  against int3c2e_ipip2.

These are the pieces pyscf.df.hessian.rhf.partial_hess_elec contracts, here
with matrices of no symmetry at all, so every ordering of the centres is
exercised: W_PQ = 1 / (P + 2 Q + 1) and Gamma_mnP = 1 / (m + 2 n + 3 P + 1),
0-based in PySCF's spherical AO order, one IEEE division each so a C++ test
rebuilds them bit for bit. The cases are tools/emit/gradient_reference.py's
DF ones, skew H2O over this tree's data/basis and data/aux_basis files:
cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux to f), cc-pVTZ / cc-pVTZ-JKFIT (to
f, to g) and cc-pVTZ / cc-pVTZ-acCD-6-6 (to f, to i: (ff|i), (i|i)).

Every Hessian is checked here against a fourth-order central difference (step
1e-3 bohr) of the matching analytic gradient (int2c2e_ip1, int3c2e_ip1 / ip2
contracted with the same matrix), for symmetry, H = H^T, and for the
translational sum rule, sum_A H[A k][B l] = 0. BLAS and OpenMP are pinned to
one thread and nothing run-dependent is written, so a rerun regenerates the
file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/df_hessian_reference.py

writes test/gtest/data/hessian/df_hessian.json. Re-run it to regenerate; the
file is never hand-edited.
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
from df_reference import libcint_version, shells  # noqa: E402
from gradient_reference import DF_INTEGRAL_CASES, atoms_out, molecules  # noqa: E402
from pyscf import df, gto, lib  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "hessian" / "df_hessian.json"

FD_STEP = 1e-3
# Of max(1, max|H|): the fourth-order difference's O(h^4) error and its
# round-off sit well below this.
FD_TOLERANCE = 1e-7
# Of max(1, max|H|): symmetry, the sum rule and the d_P d_P cross-check.
EXACT_TOLERANCE = 1e-10

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/df_hessian_reference.py"
)


def recipe_w(n: int) -> numpy.ndarray:
    p, q = numpy.ix_(numpy.arange(n), numpy.arange(n))
    return 1.0 / (p + 2 * q + 1.0)


def recipe_gamma(nao: int, naux: int) -> numpy.ndarray:
    m, n, p = numpy.ix_(numpy.arange(nao), numpy.arange(nao), numpy.arange(naux))
    return 1.0 / (m + 2 * n + 3 * p + 1.0)


def ao_atoms(m: gto.Mole) -> numpy.ndarray:
    """The atom each AO sits on."""
    out = numpy.empty(m.nao, dtype=int)
    for a, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[p0:p1] = a
    return out


def scatter(h: numpy.ndarray, c: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray):
    """h[x(i), k, y(j), l] += c[k, l, i, j] for every function pair (i, j)."""
    for a in range(h.shape[0]):
        for b in range(h.shape[0]):
            h[a, :, b, :] += c[:, :, x == a][:, :, :, y == b].sum(axis=(2, 3))


def scatter_diagonal(h: numpy.ndarray, c: numpy.ndarray, x: numpy.ndarray):
    """h[x(i), k, x(i), l] += c[k, l, i] for every function i."""
    for a in range(h.shape[0]):
        h[a, :, a, :] += c[:, :, x == a].sum(axis=2)


def metric_hessian(auxmol: gto.Mole, w: numpy.ndarray) -> numpy.ndarray:
    """sum_PQ W_PQ d2(P|Q)/dR dR for any W. Each ip is nabla on a function,
    -d/dR of its centre, so two of them carry no sign."""
    at = ao_atoms(auxmol)
    ipip = auxmol.intor("int2c2e_ipip1").reshape(3, 3, auxmol.nao, auxmol.nao)
    ip1ip2 = auxmol.intor("int2c2e_ip1ip2").reshape(3, 3, auxmol.nao, auxmol.nao)
    h = numpy.zeros((auxmol.natm, 3, auxmol.natm, 3))
    # Both on P, then both on Q ((P|Q) = (Q|P), so ipip1 at [Q, P]).
    scatter_diagonal(h, numpy.einsum("klpq,pq->klp", ipip, w), at)
    scatter_diagonal(h, numpy.einsum("klqp,pq->klq", ipip, w), at)
    # One on P (k) and one on Q (l), and its mirror.
    pq = ip1ip2 * w
    scatter(h, pq, at, at)
    scatter(h, pq.transpose(1, 0, 3, 2), at, at)
    return h.reshape(3 * auxmol.natm, 3 * auxmol.natm)


def three_center_hessian(
    mol: gto.Mole, auxmol: gto.Mole, gamma: numpy.ndarray, check: bool = False
) -> numpy.ndarray:
    """sum_mnP Gamma_mnP d2(mn|P)/dR dR for any Gamma."""
    nao, naux = mol.nao, auxmol.nao

    def intor(name: str) -> numpy.ndarray:
        return df.incore.aux_e2(mol, auxmol, f"{name}_sph", aosym="s1", comp=9).reshape(
            3, 3, nao, nao, naux
        )

    x, z = ao_atoms(mol), ao_atoms(auxmol)
    gamma_t = gamma.transpose(1, 0, 2)  # the n-derivatives are m's with m, n swapped
    h = numpy.zeros((mol.natm, 3, mol.natm, 3))

    ipip = intor("int3c2e_ipip1")
    scatter_diagonal(h, numpy.einsum("klmnp,mnp->klm", ipip, gamma), x)
    scatter_diagonal(h, numpy.einsum("klmnp,mnp->klm", ipip, gamma_t), x)
    del ipip

    ipvip = intor("int3c2e_ipvip1")
    mn = numpy.einsum("klmnp,mnp->klmn", ipvip, gamma)
    del ipvip
    scatter(h, mn, x, x)
    scatter(h, mn.transpose(1, 0, 3, 2), x, x)

    ip1ip2 = intor("int3c2e_ip1ip2")
    # One on m (or on n, Gamma swapped) and one on P, and the mirrors.
    xp = numpy.einsum("klmnp,mnp->klmp", ip1ip2, gamma) + numpy.einsum(
        "klmnp,mnp->klmp", ip1ip2, gamma_t
    )
    del ip1ip2
    scatter(h, xp, x, z)
    scatter(h, xp.transpose(1, 0, 3, 2), z, x)
    # Both on P: d_P d_P = -(d_m + d_n) d_P, summed over m.
    pp = -xp.sum(axis=2)
    scatter_diagonal(h, pp, z)
    if check:
        direct = numpy.einsum("klmnp,mnp->klp", intor("int3c2e_ipip2"), gamma)
        drift = numpy.abs(direct - pp).max()
        assert drift < EXACT_TOLERANCE * max(1.0, numpy.abs(pp).max()), drift
    return h.reshape(3 * mol.natm, 3 * mol.natm)


def bra(m: gto.Mole, ip: numpy.ndarray, matrix: numpy.ndarray) -> numpy.ndarray:
    """sum matrix d/dR of the integral, the derivative on the first index."""
    out = numpy.zeros((m.natm, 3))
    for a, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[a] = -numpy.tensordot(ip[:, p0:p1], matrix[p0:p1], axes=matrix.ndim)
    return out


def metric_gradient(auxmol: gto.Mole, w: numpy.ndarray) -> numpy.ndarray:
    ip = auxmol.intor("int2c2e_ip1")
    return (bra(auxmol, ip, w) + bra(auxmol, ip, w.T)).ravel()


def three_center_gradient(
    mol: gto.Mole, auxmol: gto.Mole, gamma: numpy.ndarray
) -> numpy.ndarray:
    ip1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip1_sph", aosym="s1", comp=3)
    ip2 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip2_sph", aosym="s1", comp=3)
    orbital = bra(mol, ip1, gamma) + bra(mol, ip1, gamma.transpose(1, 0, 2))
    aux = bra(auxmol, ip2.transpose(0, 3, 1, 2), gamma.transpose(2, 0, 1))
    return (orbital + aux).ravel()


def finite_difference(case: dict, gradient) -> numpy.ndarray:
    """Fourth-order central difference of gradient(mol, auxmol) over every
    nuclear coordinate: column 3 A + k is d gradient / dR_Ak."""
    coords = numpy.array([coord for _, coord in case["atoms"]])
    n = coords.size
    out = numpy.zeros((n, n))
    for c in range(n):
        values = {}
        for s in (-2, -1, 1, 2):
            moved = coords.copy()
            moved.flat[c] += s * FD_STEP
            values[s] = gradient(*molecules(case, moved))
        out[:, c] = (8.0 * (values[1] - values[-1]) - (values[2] - values[-2])) / (
            12.0 * FD_STEP
        )
    return out


def check(name: str, h: numpy.ndarray, case: dict, gradient) -> None:
    scale = max(1.0, numpy.abs(h).max())
    numeric = finite_difference(case, gradient)
    assert numpy.abs(h - numeric).max() < FD_TOLERANCE * scale, (case["name"], name)
    assert numpy.abs(h - h.T).max() < EXACT_TOLERANCE * scale, (case["name"], name)
    n = h.shape[0]
    drift = numpy.abs(h.reshape(n // 3, 3, n).sum(axis=0)).max()
    assert drift < EXACT_TOLERANCE * scale, (case["name"], name, drift)


def rows(h: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in h]


def df_hessian_case(case: dict) -> dict:
    mol, auxmol = molecules(case)
    lmax = max(mol.bas_angular(i) for i in range(mol.nbas))
    aux_lmax = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))
    assert (lmax, aux_lmax) == case["max_l"], (lmax, aux_lmax)
    w = recipe_w(auxmol.nao)
    gamma = recipe_gamma(mol.nao, auxmol.nao)
    got_metric = metric_hessian(auxmol, w)
    got_three = three_center_hessian(mol, auxmol, gamma, check=True)
    check("metric", got_metric, case, lambda _, a: metric_gradient(a, w))
    check(
        "three_center",
        got_three,
        case,
        lambda m, a: three_center_gradient(m, a, gamma),
    )
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "shells": shells(mol),
        "aux_shells": shells(auxmol),
        "metric": rows(got_metric),
        "three_center": rows(got_three),
    }


def main() -> None:
    lib.num_threads(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "libcint's second-derivative DF integrals contracted "
        "with fixed recipe matrices, over this tree's data/basis and "
        "data/aux_basis files, spherical. Every Hessian is 3N x 3N rows, "
        "H[3 A + k][3 B + l] = d2/dR_Ak dR_Bl in hartree/bohr^2, in atoms "
        "order; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "recipe": {
            "W": "W_PQ = 1 / (P + 2 Q + 1), naux x naux",
            "Gamma": "Gamma_mnP = 1 / (m + 2 n + 3 P + 1), nao x nao x naux",
            "indices": "0-based, PySCF's spherical AO order (shells in "
            "`shells` order, m = -l..l, p as x, y, z), unit-normalised "
            "functions",
        },
        "density_fitting": [df_hessian_case(case) for case in DF_INTEGRAL_CASES],
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
