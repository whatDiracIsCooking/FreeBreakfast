#!/usr/bin/env python3
"""Write the transition-RDM reference data ci::Rdm's tests read.

For each case below, PySCF's closed-shell RHF is run in PySCF's own basis
library, CASCI's active-space integrals are taken in its MO basis, and
`fci.direct_spin1` is solved over them for the lowest NROOTS roots of the
M_s = (N_alpha - N_beta) / 2 sector. For each named pair of roots (bra, ket),
picked by spin and order (the k-th singlet, the k-th triplet), it records:

- the two CI vectors (alpha-major, as `direct_spin1` stores them), so a test
  reads the RDMs of exactly those vectors and no solver's sign or rotation
  enters;
- `trans_rdm1`: `direct_spin1.trans_rdm1(bra, ket)` as PySCF returns it,
  spin summed and not symmetric. Its [p, q] is <bra| q+ p |ket> (the tool
  asserts it on one electron in two orbitals), the transpose of
  ci::Rdm::trans_rdm1's <bra| E_pq |ket>;
- `rdm1` and `rdm2`: the symmetrized transition 1- and 2-RDMs,
  (trans_rdm12(bra, ket) + trans_rdm12(ket, bra)) / 2, the 2-RDM in
  `make_rdm12`'s convention dm2[p, q, r, s] = <p+ r+ s q>, n^4 row-major.

The tool asserts that trans_rdm12's 1-RDM is trans_rdm1's, that
trans_rdm12s summed over the spins is trans_rdm12, and that a singlet-triplet
pair's spin-summed transition RDMs vanish.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/trans_rdm_reference.py

writes test/gtest/data/ci/trans_rdm.json. Re-run it to regenerate; the file
is never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from pyscf import ao2mo, fci, gto, mcscf, scf
from scf_reference import WATER

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "ci" / "trans_rdm.json"

# Formaldehyde, as fci_reference.py's (bohr).
FORMALDEHYDE = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 2.2771)),
    ("H", (0.0, 1.7817, -1.1111)),
    ("H", (0.0, -1.7817, -1.1111)),
]

# A pair is ((spin, k), (spin, k)): the k-th root of that <S^2> (0 singlet,
# 2 triplet) in energy order, bra first. Three active-space sizes; singlets
# and triplets over M_s = 0, a singlet-triplet pair (which vanishes), and
# triplets over M_s = 1 (N_alpha > N_beta).
CASES = [
    {
        "name": "h2o_sto3g_cas44",
        "atoms": WATER,
        "basis": "sto-3g",
        "cas": (4, 4),
        "nelec": (2, 2),
        "pairs": [
            ((0, 0), (0, 1)),
            ((0, 2), (0, 1)),
            ((2, 0), (2, 1)),
            ((0, 0), (2, 0)),
        ],
    },
    {
        "name": "h2o_631g_cas66",
        "atoms": WATER,
        "basis": "6-31g",
        "cas": (6, 6),
        "nelec": (3, 3),
        "pairs": [((0, 0), (0, 1)), ((2, 1), (2, 0))],
    },
    {
        "name": "h2o_631g_cas46",
        "atoms": WATER,
        "basis": "6-31g",
        "cas": (6, 4),
        "nelec": (2, 2),
        "pairs": [((0, 0), (0, 2)), ((0, 1), (0, 2))],
    },
    {
        "name": "ch2o_sto3g_cas44_ms1",
        "atoms": FORMALDEHYDE,
        "basis": "sto-3g",
        "cas": (4, 4),
        "nelec": (3, 1),
        "pairs": [((2, 0), (2, 1)), ((2, 1), (2, 2))],
    },
]

NROOTS = 12
SPIN_TOL = 1e-6
# The cross-checks between PySCF's own routines.
CHECK_TOL = 1e-12


def values(array: numpy.ndarray) -> dict:
    """An array as the JSON's shape-and-values pair, row-major."""
    return {"shape": list(array.shape), "values": [float(v) for v in array.ravel()]}


def solve(case: dict) -> tuple[int, numpy.ndarray, list[numpy.ndarray], list[float]]:
    """The active space's size and direct_spin1's lowest roots over it."""
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=case["basis"],
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{case['name']}: RHF did not converge"

    n, n_electrons = case["cas"]
    nelec = case["nelec"]
    assert sum(nelec) == n_electrons
    mc = mcscf.CASCI(mf, n, n_electrons)
    h1, e_core = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), n)

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-13
    solver.max_cycle = 1000
    solver.max_space = 40
    solver.nroots = NROOTS
    solver.verbose = 0
    energies, vectors = solver.kernel(h1, eri, n, nelec, ecore=e_core)
    assert numpy.all(solver.converged), f"{case['name']}: Davidson did not converge"
    s2 = [float(fci.spin_op.spin_square0(v, n, nelec)[0]) for v in vectors]
    return n, numpy.asarray(energies), list(vectors), s2


def pair_reference(
    n: int, nelec: tuple[int, int], bra: numpy.ndarray, ket: numpy.ndarray
) -> dict:
    """The transition RDMs of (bra, ket), PySCF's, and the vectors."""
    t1 = fci.direct_spin1.trans_rdm1(bra, ket, n, nelec)
    one, two = fci.direct_spin1.trans_rdm12(bra, ket, n, nelec)
    one_t, two_t = fci.direct_spin1.trans_rdm12(ket, bra, n, nelec)
    _, (aa, ab, ba, bb) = fci.direct_spin1.trans_rdm12s(bra, ket, n, nelec)
    assert numpy.abs(one - t1).max() < CHECK_TOL, "trans_rdm12's dm1 != trans_rdm1"
    assert numpy.abs(one_t - t1.T).max() < CHECK_TOL, "trans_rdm1(ket, bra) != its T"
    assert numpy.abs(aa + ab + ba + bb - two).max() < CHECK_TOL, (
        "trans_rdm12s summed over the spins != trans_rdm12"
    )
    return {
        "bra": values(bra.ravel()),
        "ket": values(ket.ravel()),
        "trans_rdm1": values(t1),
        "rdm1": values(0.5 * (one + one_t)),
        "rdm2": values(0.5 * (two + two_t)),
    }


def check_trans_rdm1_order() -> None:
    """trans_rdm1(bra, ket)[p, q] is <bra| q+ p |ket>: one alpha electron,
    bra in orbital 0 and ket in orbital 1, so only <bra| 0+ 1 |ket> = 1."""
    bra = numpy.array([[1.0], [0.0]])
    ket = numpy.array([[0.0], [1.0]])
    dm1 = fci.direct_spin1.trans_rdm1(bra, ket, 2, (1, 0))
    assert numpy.array_equal(dm1, numpy.array([[0.0, 0.0], [1.0, 0.0]])), dm1


def case_reference(case: dict) -> dict:
    """The pairs of one active space."""
    n, energies, vectors, s2 = solve(case)
    nelec = case["nelec"]
    by_spin = {
        spin: [k for k, s in enumerate(s2) if abs(s - spin) < SPIN_TOL]
        for spin in (0, 2)
    }
    pairs = []
    for (bra_spin, bra_k), (ket_spin, ket_k) in case["pairs"]:
        bra = by_spin[bra_spin][bra_k]
        ket = by_spin[ket_spin][ket_k]
        entry = pair_reference(n, nelec, vectors[bra], vectors[ket])
        if bra_spin != ket_spin:
            worst = max(
                numpy.abs(entry[key]["values"]).max()
                for key in ("trans_rdm1", "rdm1", "rdm2")
            )
            assert worst < 1e-10, f"{case['name']}: singlet-triplet RDMs {worst}"
        pairs.append(
            {
                "roots": [bra, ket],
                "s2": [s2[bra], s2[ket]],
                "energies": [float(energies[bra]), float(energies[ket])],
                **entry,
            }
        )
    return {
        "name": case["name"],
        "basis": case["basis"],
        "n_orbitals": n,
        "n_alpha": nelec[0],
        "n_beta": nelec[1],
        "pairs": pairs,
    }


def main() -> None:
    check_trans_rdm1_order()
    doc = {
        "description": "PySCF RHF in PySCF's own basis, CASCI active-space "
        "integrals in its MO basis, and fci.direct_spin1's lowest roots over "
        "them (conv_tol 1e-13); per case, pairs of roots (bra, ket) with "
        "their CI vectors (alpha-major), trans_rdm1(bra, ket) (gamma[p, q] = "
        "<bra| q+ p |ket>, spin summed, n x n row-major), and the symmetrized "
        "(trans_rdm12(bra, ket) + trans_rdm12(ket, bra)) / 2 as rdm1 and rdm2 "
        "(2-RDM [p, q, r, s] = <p+ r+ s q>, n^4 row-major); s2 is <S^2> of "
        "each root",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit/trans_rdm_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
