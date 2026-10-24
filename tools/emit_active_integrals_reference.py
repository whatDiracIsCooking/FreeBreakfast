#!/usr/bin/env python3
"""Write the reference data the active-space Hamiltonian's tests read.

For each case below, PySCF's closed-shell RHF (as emit_scf_reference.py runs
it, over this tree's own basis files, converged tighter) is handed to PySCF's
CASCI over a chosen core and active set of its canonical orbitals, and its
active-space Hamiltonian recorded: `get_h1eff`'s constant E_core and h_eff, and
`get_h2eff`'s (tu|vw).

Only the elements invariant under each orbital's sign are recorded: h_eff_tt,
(tt|uu) and (tu|tu). The rest depend on the signs, which the two codes pick
independently. A degenerate set would also be free to rotate within itself, so
each active space keeps every degenerate set whole and holds at most one: a
rotation within one set is then a symmetry operation of the molecule, which
leaves every recorded element unchanged. The tool refuses a case that breaks
either rule.

CASSCF's `mc_ao2mo` integrals over the same orbitals are recorded too, from
`CASSCF.ao2mo`'s ppaa, (pq|uv), and papa, (pu|qv), over every orbital p and
active u: their sign-invariant (pp|uu) and (pu|pu). p ranges over the whole
basis, so an orbital outside the active space may share a degenerate set with
others, rotated within it independently of the active one; each is therefore
summed over p's degenerate set and u's, which is invariant under both
rotations (a trace of J[C_u C_u^T] or K[C_u C_u^T] over p's set, summed over
u's), and is the element itself where both sets are single orbitals.

    uv run --group tools tools/emit_active_integrals_reference.py

writes test/gtest/data/active_space/active_space.json. Re-run it to regenerate;
the file is never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from emit_scf_reference import BASES, WATER, pyscf_basis
from pyscf import ao2mo, gto, mcscf, scf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "active_space" / "active_space.json"

# N2 near its equilibrium bond (2.074 bohr), along z. Bohr.
NITROGEN = [
    ("N", (0.0, 0.0, -1.037)),
    ("N", (0.0, 0.0, 1.037)),
]

# Each case's core and active orbitals, as indices into the canonical orbitals
# in ascending energy; each active space spans occupied and virtual ones. N2's
# 3sigma_g lies below its 1pi_u pair (HOMO), so its core takes the pair and
# skips 3sigma_g, which the active space takes with the 1pi_g pair and the
# sigma above it: no two degenerate sets in one active space.
CASES = [
    {
        "name": "h2o_sto3g",
        "atoms": WATER,
        "basis": "sto-3g",
        "core": [0],
        "active": [1, 2, 3, 4, 5],
    },
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "core": [0],
        "active": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    {
        "name": "n2_ccpvdz",
        "atoms": NITROGEN,
        "basis": "cc-pvdz",
        "core": [0, 1, 2, 3, 5, 6],
        "active": [4, 7, 8, 9],
    },
]

# Two orbital energies closer than this are one degenerate set (hartree).
DEGENERACY = 1e-6


def check_degeneracies(name: str, mo_energy: numpy.ndarray, active: list[int]):
    """Refuse an active space that splits a degenerate set or holds two."""
    sets = []
    start = 0
    for i in range(1, len(mo_energy) + 1):
        if i == len(mo_energy) or mo_energy[i] - mo_energy[i - 1] > DEGENERACY:
            if i - start > 1:
                sets.append(range(start, i))
            start = i
    inside = 0
    for degenerate in sets:
        hits = sum(1 for i in degenerate if i in active)
        if 0 < hits < len(degenerate):
            raise RuntimeError(f"{name}: the active space splits {list(degenerate)}")
        if hits == len(degenerate):
            inside += 1
    if inside > 1:
        raise RuntimeError(f"{name}: the active space holds {inside} degenerate sets")


def degenerate_sets(mo_energy: numpy.ndarray) -> list[list[int]]:
    """Every orbital's degenerate set, singletons included, ascending."""
    sets = [[0]]
    for i in range(1, len(mo_energy)):
        if mo_energy[i] - mo_energy[i - 1] > DEGENERACY:
            sets.append([])
        sets[-1].append(i)
    return sets


def case_reference(case: dict) -> dict:
    """The active-space reference of one molecule."""
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
    # Tighter than emit_scf_reference.py: h_eff_tt moves to first order in the
    # orbitals, and the comparison is at 1e-10.
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{case['name']} did not converge"
    core, active = case["core"], case["active"]
    ncore, ncas = len(core), len(active)
    nocc = mol.nelectron // 2
    name = case["name"]
    assert all(i < nocc for i in core), f"{name}: a core orbital is virtual"
    assert any(i < nocc for i in active), f"{name}: no active occupied orbital"
    assert any(i >= nocc for i in active), f"{name}: no active virtual orbital"
    check_degeneracies(name, mf.mo_energy, active)

    # Core, then active, then the rest: CASCI's own column order.
    rest = [i for i in range(mol.nao) if i not in core and i not in active]
    orbitals = core + active + rest
    mo = mf.mo_coeff[:, orbitals]
    mc = mcscf.CASCI(mf, ncas, mol.nelectron - 2 * ncore, ncore=ncore)
    h_eff, e_core = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), ncas)

    # CASSCF's ppaa (pq|uv) and papa (pu|qv) over the same columns, summed over
    # degenerate sets: the sets as column positions, and the active ones as
    # active indices.
    eris = mcscf.CASSCF(mf, ncas, mol.nelectron - 2 * ncore, ncore=ncore).ao2mo(mo)
    ppaa = numpy.asarray(eris.ppaa)
    papa = numpy.asarray(eris.papa)
    sets = [[orbitals.index(i) for i in d] for d in degenerate_sets(mf.mo_energy)]
    active_sets = [
        [col - ncore for col in d] for d in sets if ncore <= d[0] < ncore + ncas
    ]
    assert sorted(u for d in active_sets for u in d) == list(range(ncas))
    pp_uu = [
        [float(sum(ppaa[p, p, u, u] for p in ps for u in us)) for us in active_sets]
        for ps in sets
    ]
    pu_pu = [
        [float(sum(papa[p, u, p, u] for p in ps for u in us)) for us in active_sets]
        for ps in sets
    ]
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "ncore": ncore,
        "ncas": ncas,
        "orbitals": orbitals,
        "energy": float(mf.e_tot),
        "mo_energies": [float(e) for e in mf.mo_energy],
        "e_core": float(e_core),
        "h_eff_diagonal": [float(h_eff[t, t]) for t in range(ncas)],
        "coulomb": [[float(eri[t, t, u, u]) for u in range(ncas)] for t in range(ncas)],
        "exchange": [
            [float(eri[t, u, t, u]) for u in range(ncas)] for t in range(ncas)
        ],
        "orbital_sets": sets,
        "active_sets": active_sets,
        "ppaa_diagonal": pp_uu,
        "papa_diagonal": pu_pu,
    }


def main() -> None:
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-14, "
        "conv_tol_grad 1e-11) over this tree's data/basis files, and PySCF "
        "CASCI's active-space Hamiltonian over its canonical orbitals "
        "reordered by 'orbitals' (core, active, rest), columns "
        "[ncore, ncore + ncas) active: get_h1eff's constant E_core (nuclear repulsion "
        "included) and diagonal h_eff_tt, and get_h2eff's (tt|uu) ('coulomb') "
        "and (tu|tu) ('exchange'), t the row; and CASSCF.ao2mo's ppaa (pq|uv) "
        "and papa (pu|qv) over every column p and active u, as "
        "'ppaa_diagonal' = sum (pp|uu) and 'papa_diagonal' = sum (pu|pu) over "
        "p in each 'orbital_sets' entry (column positions, the row) and u in "
        "each 'active_sets' entry (active indices, the column): the degenerate "
        "sets, singletons included; hartree, coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit_active_integrals_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
