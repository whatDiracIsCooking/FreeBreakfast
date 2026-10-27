#!/usr/bin/env python3
"""Write the reference data the Fci step's tests read.

For each case below, PySCF's closed-shell RHF (as casci_reference.py runs
it, over this tree's own basis files, converged tighter) is handed to PySCF's
`fci.FCI`: `fci.direct_spin1` over every orbital and every electron, with no
spin constraint, solved for the lowest NROOTS roots of the M_s = 0 sector; the
lowest NSINGLETS singlets among them are recorded.

So that a Davidson that skipped a root cannot be written out, a sector small
enough (DENSE_MAX determinants) is diagonalised densely instead, and a larger
one is solved twice, from PySCF's guesses spread by differently seeded noise,
and the two must agree.

Per singlet: the total energy, <S^2>, and the natural occupations (the 1-RDM's
eigenvalues, descending). An FCI energy and its natural occupations are
invariant under any rotation of the orbitals, so they depend on neither the
two SCFs' convergence nor the orbitals' signs. A root of a degenerate level is
flagged: its 1-RDM depends on the solver's rotation within the level.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/fci_step_reference.py

writes test/gtest/data/fci/fci.json. Re-run it to regenerate; the file is
never hand-edited. H2O 6-31G (1.66M determinants) takes some minutes.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from active_integrals_reference import NITROGEN
from fci_reference import dense_hamiltonian
from pyscf import ao2mo, fci, gto, scf
from scf_reference import BASES, WATER, pyscf_basis

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "fci" / "fci.json"

# This tree's builtin basis name, and its file.
BASIS_FILES = {**BASES, "6-31g": "6-31g.json"}

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g"},
    {"name": "n2_sto3g", "atoms": NITROGEN, "basis": "sto-3g"},
    {"name": "h2o_631g", "atoms": WATER, "basis": "6-31g"},
]

# Roots solved for, and the singlets kept among them.
NROOTS = 12
NSINGLETS = 3

# Two roots closer than this are one degenerate level.
ROOT_DEGENERACY = 1e-8
# The roots must equal the check's to this.
CHECK_TOL = 1e-10
# The largest sector diagonalised densely.
DENSE_MAX = 20000


def mo_integrals(mf: scf.hf.RHF):
    """h and (pq|rs) over every orbital of mf, and its orbital count."""
    c = mf.mo_coeff
    nmo = c.shape[1]
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), nmo)
    return h1, eri, nmo


def davidson(mf: scf.hf.RHF, nelec: tuple[int, int], seed: int):
    """direct_spin1's lowest NROOTS roots over every orbital of mf.

    From PySCF's own guesses, each spread over every determinant by seeded
    noise: from its unit guesses alone a root of a spatial symmetry they miss
    never appears (N2's are lost that way). The noise is 1e-3 in norm, not
    per element: spread wider over H2O 6-31G's 1.66M determinants, it buries
    the guesses, and PySCF's Davidson converged on O 1s-hole states near
    -56 hartree instead.
    """
    h1, eri, nmo = mo_integrals(mf)
    solver = fci.FCI(mf)
    solver.conv_tol = 1e-13
    solver.max_cycle = 1000
    solver.max_space = 40
    solver.nroots = NROOTS
    solver.verbose = 0
    hdiag = solver.make_hdiag(h1, eri, nmo, nelec)
    rng = numpy.random.default_rng(seed)
    guesses = []
    for g in solver.get_init_guess(nmo, nelec, NROOTS, hdiag):
        noise = rng.standard_normal(hdiag.size)
        g = numpy.asarray(g).ravel() + 1e-3 * noise / numpy.linalg.norm(noise)
        guesses.append(g / numpy.linalg.norm(g))
    energies, vectors = solver.kernel(
        h1, eri, nmo, nelec, ecore=mf.mol.energy_nuc(), ci0=guesses
    )
    assert numpy.all(solver.converged), "Davidson did not converge"
    return numpy.asarray(energies), [numpy.asarray(v).ravel() for v in vectors]


def dense(mf: scf.hf.RHF, nelec: tuple[int, int]):
    """The sector's lowest NROOTS roots, by a dense diagonalisation."""
    h1, eri, nmo = mo_integrals(mf)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, nmo, nelec, 0.5)
    values, vectors = numpy.linalg.eigh(dense_hamiltonian(h2e, nmo, nelec))
    energies = values[:NROOTS] + mf.mol.energy_nuc()
    return energies, [vectors[:, k].copy() for k in range(NROOTS)]


def roots_of(name: str, mf: scf.hf.RHF, nelec: tuple[int, int]):
    """The lowest NROOTS roots, checked: densely where the sector is small
    enough, otherwise by two Davidson solves from differently seeded guesses.
    """
    nmo = mf.mo_coeff.shape[1]
    na = fci.cistring.num_strings(nmo, nelec[0])
    if na * na <= DENSE_MAX:
        return dense(mf, nelec)
    energies, vectors = davidson(mf, nelec, seed=229)
    again, _ = davidson(mf, nelec, seed=230)
    error = numpy.abs(again - energies).max()
    assert error < CHECK_TOL, f"{name}: two Davidson solves part by {error}"
    return energies, vectors


def case_reference(case: dict) -> dict:
    """The FCI reference of one molecule."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASIS_FILES[case["basis"]], symbols),
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    name = case["name"]
    assert mf.converged, f"{name}: RHF did not converge"

    nmo = mf.mo_coeff.shape[1]
    nelec = (mol.nelectron // 2, mol.nelectron // 2)
    energies, vectors = roots_of(name, mf, nelec)
    na = fci.cistring.num_strings(nmo, nelec[0])

    roots = []
    for energy, vector in zip(energies, vectors, strict=True):
        vector = vector.reshape(na, na)
        s2, _ = fci.spin_op.spin_square0(vector, nmo, nelec)
        if abs(s2) < 1e-6:
            rdm1 = fci.direct_spin1.make_rdm1(vector, nmo, nelec)
            occupations = numpy.linalg.eigvalsh(rdm1)
            roots.append(
                {
                    "energy": float(energy),
                    "s2": float(s2),
                    "degenerate": bool(
                        numpy.sum(numpy.abs(energies - energy) < ROOT_DEGENERACY) > 1
                    ),
                    "natural_occupations": [float(v) for v in occupations[::-1]],
                }
            )
    assert len(roots) >= NSINGLETS, f"{name}: fewer than {NSINGLETS} singlets"

    return {
        "name": name,
        "basis": case["basis"],
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "nmo": int(nmo),
        "determinants": int(na * na),
        "rhf_energy": float(mf.e_tot),
        "roots": roots[:NSINGLETS],
    }


def main() -> None:
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-14, "
        "conv_tol_grad 1e-11) over this tree's data/basis files, and PySCF "
        "fci.FCI (direct_spin1, conv_tol 1e-13, a dense diagonalisation where "
        "small, else checked by a second, differently seeded solve) over its "
        "orbitals: the lowest singlets' total energy (hartree), <S^2> and "
        "natural occupations; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit/fci_step_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
