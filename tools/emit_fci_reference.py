#!/usr/bin/env python3
"""Write the reference data the determinant CI's tests read.

For each case below, PySCF's closed-shell RHF is run in PySCF's own basis
library, and CASCI's active-space integrals are taken in PySCF's own MO basis:
E_core and h_eff from `get_h1eff`, (tu|vw) from `get_h2eff` restored to the
full n^4 array. Nothing crosses from this tree's AOs, so there is no AO order,
normalisation or MO sign to reconcile: a test reads the integrals and builds
its Hamiltonian from them alone.

Over those integrals PySCF's `fci.direct_spin1` (no spin constraint, the M_s =
(N_alpha - N_beta) / 2 sector) is solved for the lowest NROOTS roots, and each
root's <S^2> and spin-summed 1-RDM recorded. The roots are checked against a
dense diagonalisation of the same sector, H built column by column with
`contract_2e`, so a Davidson that skipped a root cannot be written out.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit_fci_reference.py

writes test/gtest/data/ci/direct_spin1.json. Re-run it to regenerate; the file
is never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from emit_scf_reference import WATER
from pyscf import ao2mo, fci, gto, mcscf, scf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "ci" / "direct_spin1.json"

# N2 at 1.0977 A, and formaldehyde (C2v in the yz plane: C=O 1.205 A,
# C-H 1.111 A, HCH 116.1 deg). Bohr.
NITROGEN = [
    ("N", (0.0, 0.0, -1.03715)),
    ("N", (0.0, 0.0, 1.03715)),
]
FORMALDEHYDE = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 2.2771)),
    ("H", (0.0, 1.7817, -1.1111)),
    ("H", (0.0, -1.7817, -1.1111)),
]

# Water's STO-3G has 7 MOs, 5 of them doubly occupied, so CAS(6,6) (2 core,
# 6 active) does not fit; its CAS(6,6) is in 6-31G. Formaldehyde is the case
# chosen for its triplet below the second singlet (n -> pi*, T1 under S1);
# the emitter asserts it has one.
CASES = [
    {"name": "h2o_sto3g_cas44", "atoms": WATER, "basis": "sto-3g", "cas": (4, 4)},
    {"name": "h2o_631g_cas66", "atoms": WATER, "basis": "6-31g", "cas": (6, 6)},
    {"name": "n2_ccpvdz_cas66", "atoms": NITROGEN, "basis": "cc-pvdz", "cas": (6, 6)},
    {
        "name": "ch2o_sto3g_cas44",
        "atoms": FORMALDEHYDE,
        "basis": "sto-3g",
        "cas": (4, 4),
        "triplet_below_second_singlet": True,
    },
]

# Enough for three singlets in every case, N2's included, whose 7th-9th roots
# are its first singlets past the ground state.
NROOTS = 12

# Two roots closer than this are one degenerate level: within it a root's
# 1-RDM depends on the solver's arbitrary rotation, so it is flagged.
DEGENERATE = 1e-8

# The Davidson roots must equal the dense spectrum to this, and the first root
# left out must lie this far above the last kept, so no level is split.
DENSE_TOL = 1e-10
GAP = 1e-6


def dense_hamiltonian(
    h2e: numpy.ndarray, n: int, nelec: tuple[int, int]
) -> numpy.ndarray:
    """The sector's H, one contract_2e per unit vector."""
    na = fci.cistring.num_strings(n, nelec[0])
    nb = fci.cistring.num_strings(n, nelec[1])
    dim = na * nb
    h = numpy.empty((dim, dim))
    for j in range(dim):
        e = numpy.zeros(dim)
        e[j] = 1.0
        h[:, j] = fci.direct_spin1.contract_2e(h2e, e, n, nelec).ravel()
    return h


def case_reference(case: dict) -> dict:
    """The integrals and direct_spin1 roots of one active space."""
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

    n_electrons, n = case["cas"]
    nelec = (n_electrons // 2, n_electrons - n_electrons // 2)
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

    h2e = fci.direct_spin1.absorb_h1e(h1, eri, n, nelec, 0.5)
    dense = numpy.linalg.eigvalsh(dense_hamiltonian(h2e, n, nelec)) + e_core
    assert len(dense) > NROOTS, f"{case['name']}: sector smaller than NROOTS"
    error = numpy.abs(dense[:NROOTS] - energies).max()
    assert error < DENSE_TOL, (
        f"{case['name']}: Davidson off the dense spectrum by {error}"
    )
    assert dense[NROOTS] - dense[NROOTS - 1] > GAP, (
        f"{case['name']}: NROOTS splits a level"
    )

    roots = []
    for k, (energy, vector) in enumerate(zip(energies, vectors, strict=True)):
        s2, _ = fci.spin_op.spin_square0(vector, n, nelec)
        rdm1 = solver.make_rdm1(vector, n, nelec)
        degenerate = any(
            abs(energy - other) < DEGENERATE
            for j, other in enumerate(energies)
            if j != k
        )
        roots.append(
            {
                "energy": float(energy),
                "s2": float(s2),
                "degenerate": degenerate,
                "rdm1": {"shape": [n, n], "values": [float(v) for v in rdm1.ravel()]},
            }
        )

    singlets = [k for k, root in enumerate(roots) if abs(root["s2"]) < 1e-6]
    assert len(singlets) >= 3, f"{case['name']}: fewer than three singlets"
    if case.get("triplet_below_second_singlet"):
        assert any(abs(roots[k]["s2"] - 2.0) < 1e-6 for k in range(singlets[1])), (
            f"{case['name']}: no triplet below the second singlet"
        )

    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "n_orbitals": n,
        "n_alpha": nelec[0],
        "n_beta": nelec[1],
        "e_core": float(e_core),
        "h1": {"shape": [n, n], "values": [float(v) for v in h1.ravel()]},
        "eri": {"shape": [n, n, n, n], "values": [float(v) for v in eri.ravel()]},
        "roots": roots,
    }


def main() -> None:
    doc = {
        "description": "PySCF RHF in PySCF's own basis, CASCI active-space "
        "integrals in its MO basis (E_core and h_eff from get_h1eff, (tu|vw) "
        "from get_h2eff restored to n^4, chemists' order, row-major), and "
        "fci.direct_spin1's lowest roots over them (conv_tol 1e-13, checked "
        "against the dense spectrum): total energy (hartree, E_core "
        "included), <S^2>, and the spin-summed 1-RDM; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit_fci_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
