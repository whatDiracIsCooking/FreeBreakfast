#!/usr/bin/env python3
"""Write the reference data the Casci step's tests read.

For each case below, PySCF's closed-shell RHF (as emit_scf_reference.py runs
it, over this tree's own basis files, converged tighter) is handed to PySCF's
`mcscf.CASCI` over a core and an active set of its canonical orbitals: the
frontier ones (the step's default), or an explicit list put in place by
`sort_mo`, 0-based. The fcisolver is `fci.direct_spin1` with no spin
constraint, solved for the lowest NROOTS roots of the M_s = 0 sector and
checked against a dense diagonalisation of it, so a Davidson that skipped a
root cannot be written out; the lowest NSINGLETS singlets among them are
recorded.

An AVAS case names MINAO AO labels and a threshold instead of an active
space: PySCF's `mcscf.avas` (this tree's data/basis/minao.json as its MINAO,
canonicalized, as by default) picks ncas, nelecas and the orbitals, and the
CASCI runs over those. Its ncas and nelecas are recorded; the tool refuses a
case whose projection weights come within WEIGHT_MARGIN of the threshold, so
a selection cannot hinge on the last digits of two SCFs.

Per singlet: the total energy, <S^2>, and the natural occupations (the 1-RDM's
eigenvalues, descending), which unlike the 1-RDM itself do not depend on the
signs, or any rotation, of the active orbitals. A root of a degenerate level is
flagged: its 1-RDM depends on the solver's rotation within the level. The
energies are invariant under rotations within the core and within the active
space, so each active space keeps every degenerate set of orbitals whole; the
tool refuses one that splits a set.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit_casci_reference.py

writes test/gtest/data/casci/casci.json. Re-run it to regenerate; the file is
never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from emit_active_integrals_reference import NITROGEN
from emit_fci_reference import dense_hamiltonian
from emit_scf_reference import BASES, WATER, pyscf_basis
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.mcscf import avas

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "casci" / "casci.json"

# `active` is None for the frontier orbitals: the ncas above the
# (N - nelecas) / 2 lowest. N2's frontier CAS(6,6) is 3sigma_g, the 1pi_u and
# 1pi_g pairs whole, and 3sigma_u. The explicit water case skips orbital 3 and
# 6, so its core is not the lowest orbitals' prefix and its active set is not
# contiguous.
CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "cas": (4, 4)},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "cas": (4, 4)},
    {"name": "n2_ccpvdz", "atoms": NITROGEN, "basis": "cc-pvdz", "cas": (6, 6)},
    {
        "name": "h2o_ccpvdz_explicit",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "active": [2, 4, 5, 7],
    },
    # AVAS: water's O 2p and H 1s, CAS(8,6); N2's full valence, CAS(10,8);
    # and N2's 2p at a threshold that drops the 0.22 virtual, CAS(8,6). N2's
    # CAS(10,8) keeps its lowest singlet only: ci_solve, at its defaults (100
    # Davidson iterations, residual 1e-12 relative), converges the ground
    # state in 36 iterations but no block of 2 to 5 roots in 100.
    {
        "name": "h2o_ccpvdz_avas",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "avas": {"labels": ["O 2p", "H 1s"], "threshold": 0.2},
    },
    {
        "name": "n2_ccpvdz_avas",
        "atoms": NITROGEN,
        "basis": "cc-pvdz",
        "avas": {"labels": ["N 2s", "N 2p"], "threshold": 0.2},
        "nsinglets": 1,
    },
    {
        "name": "n2_ccpvdz_avas_2p",
        "atoms": NITROGEN,
        "basis": "cc-pvdz",
        "avas": {"labels": ["N 2p"], "threshold": 0.5},
    },
]

# Roots solved for, and the singlets kept among them, unless a case says
# "nsinglets".
NROOTS = 12
NSINGLETS = 3

# Two orbital energies closer than this are one degenerate set (hartree).
ORBITAL_DEGENERACY = 1e-6
# Two roots closer than this are one degenerate level.
ROOT_DEGENERACY = 1e-8
# The Davidson roots must equal the dense spectrum to this.
DENSE_TOL = 1e-10
# No AVAS projection weight may lie closer than this to the threshold.
WEIGHT_MARGIN = 1e-3


def check_degeneracies(name: str, mo_energy: numpy.ndarray, active: list[int]):
    """Refuse an active space that splits a degenerate set of orbitals."""
    start = 0
    for i in range(1, len(mo_energy) + 1):
        if i == len(mo_energy) or mo_energy[i] - mo_energy[i - 1] > ORBITAL_DEGENERACY:
            hits = sum(1 for k in range(start, i) if k in active)
            if 0 < hits < i - start:
                raise RuntimeError(f"{name}: the active space splits {start}..{i - 1}")
            start = i


def case_reference(case: dict) -> dict:
    """The CASCI reference of one active space."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    # As emit_active_integrals_reference.py: a CASCI energy moves to first
    # order in the orbitals.
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{case['name']}: RHF did not converge"

    name = case["name"]
    if "avas" in case:
        # AVAS is invariant to rotations within the occupied and within the
        # virtual orbitals, so no degenerate set can be split by it.
        selection = avas.AVAS(
            mf,
            case["avas"]["labels"],
            threshold=case["avas"]["threshold"],
            minao=pyscf_basis("minao.json", symbols),
            verbose=0,
        )
        ncas, nelecas, mo = selection.kernel()
        ncas, nelecas = int(ncas), int(nelecas)
        weights = numpy.concatenate([selection.occ_weights, selection.vir_weights])
        margin = numpy.abs(weights - case["avas"]["threshold"]).min()
        assert margin > WEIGHT_MARGIN, (
            f"{name}: a weight within {margin} of the threshold"
        )
        ncore = (mol.nelectron - nelecas) // 2
        mc = mcscf.CASCI(mf, ncas, nelecas)
    else:
        nelecas, ncas = case["cas"]
        ncore = (mol.nelectron - nelecas) // 2
        active = case.get("active") or list(range(ncore, ncore + ncas))
        check_degeneracies(name, mf.mo_energy, active)

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mo = mc.sort_mo(active, base=0) if "active" in case else mf.mo_coeff
    solver = fci.direct_spin1.FCI(mol)
    solver.conv_tol = 1e-13
    solver.max_cycle = 1000
    solver.max_space = 40
    solver.nroots = NROOTS
    solver.verbose = 0
    mc.fcisolver = solver
    mc.verbose = 0
    energies, _, vectors, _, _ = mc.kernel(mo)
    assert numpy.all(solver.converged), f"{name}: Davidson did not converge"

    nelec = (nelecas // 2, nelecas - nelecas // 2)
    h1, e_core = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelec, 0.5)
    dense = numpy.linalg.eigvalsh(dense_hamiltonian(h2e, ncas, nelec)) + e_core
    error = numpy.abs(dense[:NROOTS] - energies).max()
    assert error < DENSE_TOL, f"{name}: Davidson off the dense spectrum by {error}"

    roots = []
    for energy, vector in zip(energies, vectors, strict=True):
        s2, _ = fci.spin_op.spin_square0(vector, ncas, nelec)
        if abs(s2) < 1e-6:
            occupations = numpy.linalg.eigvalsh(solver.make_rdm1(vector, ncas, nelec))
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
    nsinglets = case.get("nsinglets", NSINGLETS)
    assert len(roots) >= nsinglets, f"{name}: fewer than {nsinglets} singlets"

    return {
        "name": name,
        "basis": case["basis"],
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "ncas": ncas,
        "nelecas": nelecas,
        "ncore": ncore,
        "active": case.get("active"),
        "avas": case.get("avas"),
        "rhf_energy": float(mf.e_tot),
        "mo_energies": [float(e) for e in mf.mo_energy],
        "e_core": float(e_core),
        "roots": roots[:nsinglets],
    }


def main() -> None:
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-14, "
        "conv_tol_grad 1e-11) over this tree's data/basis files, and PySCF "
        "mcscf.CASCI over its canonical orbitals (the frontier ones, or "
        "'active' by sort_mo, 0-based, or by mcscf.avas over "
        "data/basis/minao.json) with fci.direct_spin1 (conv_tol 1e-13, "
        "checked against the dense spectrum): the lowest singlets' total "
        "energy (hartree), <S^2> and natural occupations; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit_casci_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
