#!/usr/bin/env python3
"""Write the reference data the Casci step's tests read.

For each case below, PySCF's closed-shell RHF (as scf_reference.py runs
it, over this tree's own basis files, converged tighter) is handed to PySCF's
`mcscf.CASCI` over a core and an active set of its canonical orbitals: the
frontier ones (the step's default), or an explicit list put in place by
`sort_mo`, 0-based. The fcisolver is `fci.direct_spin1` with no spin
constraint, solved for the lowest NROOTS roots of the M_s = S sector, S from
the case's "multiplicity" 2S + 1 (default 1), and checked against a dense
diagonalisation of it, so a Davidson that skipped a root cannot be written
out; the lowest NKEEP roots of spin S among them are recorded. The RHF is
the closed-shell one unless the case says "rohf", so a triplet case is a
triplet CASCI over RHF orbitals, `mcscf.CASCI(mf, ncas, (n_alpha, n_beta))`;
its multiplicity is recorded, and only its.

An "rohf" case's SCF is instead PySCF's density-fitted ROHF at the case's
multiplicity, over this tree's data/aux_basis file AUX_FILES pairs with the
orbital basis (handed to PySCF as parsed shells, and checked to be the set it
fitted over), since that is the open shell the Scf step runs. Its orbitals
are handed to an exact ROHF object, so the CASCI's core J and K and its
active integrals are exact, as the Casci step builds them. The tool refuses
an ROHF whose occupation is not aufbau over its orbital energies (the doubly
occupied, then the singly occupied, then the empty), whose core reaches past
the doubly occupied orbitals, or whose active space leaves a singly occupied
one out; such a case records "scf": "rohf" and its "aux_basis".

An AVAS case names MINAO AO labels and a threshold instead of an active
space: PySCF's `mcscf.avas` (this tree's data/basis/minao.json as its MINAO,
canonicalized, as by default) picks ncas, nelecas and the orbitals, and the
CASCI runs over those. Its ncas and nelecas are recorded; the tool refuses a
case whose projection weights come within WEIGHT_MARGIN of the threshold, so
a selection cannot hinge on the last digits of two SCFs.

Per root: the total energy, <S^2>, and the natural occupations (the 1-RDM's
eigenvalues, descending), which unlike the 1-RDM itself do not depend on the
signs, or any rotation, of the active orbitals. A root of a degenerate level is
flagged: its 1-RDM depends on the solver's rotation within the level. The
energies are invariant under rotations within the core and within the active
space, so each active space keeps every degenerate set of orbitals whole; the
tool refuses one that splits a set.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casci_reference.py [--append]

writes test/gtest/data/casci/casci.json. Re-run it to regenerate; the file is
never hand-edited. A rerun moves the recorded cases in their last digits
(threaded BLAS), so `--append` keeps every case the file already has byte for
byte and computes only the ones CASES adds; the description and provenance
are the run's.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform

import numpy
import pyscf
from active_integrals_reference import NITROGEN
from fci_reference import dense_hamiltonian
from pyscf import ao2mo, df, fci, gto, mcscf, scf
from pyscf.mcscf import avas
from scf_reference import AMIDOGEN, BASES, METHYLENE, WATER, pyscf_basis

ROOT = pathlib.Path(__file__).resolve().parents[2]
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
    # Water's frontier CAS(4,4) at N_alpha - N_beta = 2: its lowest triplets
    # over the RHF orbitals.
    {
        "name": "h2o_ccpvdz_triplet",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "multiplicity": 3,
    },
    # AVAS: water's O 2p and H 1s, CAS(8,6); N2's full valence, CAS(10,8);
    # and N2's 2p at a threshold that drops the 0.22 virtual, CAS(8,6). N2's
    # CAS(10,8) keeps its lowest root only: ci_solve, at its defaults (100
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
        "nkeep": 1,
    },
    {
        "name": "n2_ccpvdz_avas_2p",
        "atoms": NITROGEN,
        "basis": "cc-pvdz",
        "avas": {"labels": ["N 2p"], "threshold": 0.5},
    },
    # Over DF-ROHF orbitals: NH2's doublet CAS(5,5), its core 1a1 and 2a1 and
    # its singly occupied 1b1 active; CH2's 3B1 triplet CAS(4,4), its core 1a1
    # and 2a1 and both singly occupied orbitals active.
    {
        "name": "nh2_ccpvdz_rohf",
        "atoms": AMIDOGEN,
        "basis": "cc-pvdz",
        "cas": (5, 5),
        "multiplicity": 2,
        "rohf": True,
    },
    {
        "name": "ch2_ccpvdz_rohf",
        "atoms": METHYLENE,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "multiplicity": 3,
        "rohf": True,
    },
]

# The fitting set of each orbital basis an "rohf" case's DF-ROHF is over,
# under data/aux_basis (pyscf_basis reads from data/basis, hence the relative
# path): df_reference.py's pairing. Not imported from there, because
# importing it pins BLAS and OpenMP to one thread.
AUX_FILES = {"cc-pvdz": ("cc-pVDZ-JKFIT", "../aux_basis/cc-pVDZ-JKFIT.json")}

# Roots solved for, and those of the case's spin kept among them, unless a
# case says "nkeep".
NROOTS = 12
NKEEP = 3

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


def aux_basis(case: dict, mol: gto.Mole) -> tuple[dict, dict]:
    """The case's fitting set as PySCF shells, and its provenance."""
    name, file = AUX_FILES[case["basis"]]
    symbols = {symbol for symbol, _ in case["atoms"]}
    path = (ROOT / "data" / "basis" / file).resolve()
    provenance = {
        "name": name,
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "naux": int(df.addons.make_auxmol(mol, pyscf_basis(file, symbols)).nao),
    }
    return pyscf_basis(file, symbols), provenance


def df_rohf(case: dict, mol: gto.Mole) -> tuple[scf.hf.SCF, dict]:
    """PySCF's converged DF-ROHF over exactly the case's fitting set, its
    occupation checked to be aufbau, and the set's provenance."""
    name = case["name"]
    auxbasis, provenance = aux_basis(case, mol)
    mf = scf.ROHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: DF-ROHF did not converge"
    expected = df.addons.make_auxmol(mol, auxbasis)
    used = mf.with_df.auxmol
    assert used.nao == expected.nao, f"{name}: aux size {used.nao}"
    assert numpy.array_equal(used._bas, expected._bas), f"{name}: aux shells"
    assert numpy.array_equal(used._env, expected._env), f"{name}: aux exponents"
    n_alpha, n_beta = mol.nelec
    aufbau = numpy.zeros(len(mf.mo_occ))
    aufbau[:n_alpha] = 1
    aufbau[:n_beta] = 2
    assert numpy.array_equal(mf.mo_occ, aufbau), f"{name}: ROHF not aufbau"
    return mf, provenance


def check_open_shell(name: str, mol: gto.Mole, ncore: int, active: list[int]):
    """Refuse an ROHF active space whose core, sort_mo's lowest ncore
    orbitals not active, reaches past the doubly occupied orbitals, or which
    leaves a singly occupied one out."""
    n_alpha, n_beta = mol.nelec
    core = [i for i in range(mol.nao) if i not in active][:ncore]
    assert all(i < n_beta for i in core), f"{name}: core past doubly occupied"
    assert all(i in active for i in range(n_beta, n_alpha)), (
        f"{name}: a singly occupied orbital is not active"
    )


def case_reference(case: dict) -> dict:
    """The CASCI reference of one active space."""
    name = case["name"]
    symbols = {symbol for symbol, _ in case["atoms"]}
    # N_alpha - N_beta = 2S.
    two_s = case.get("multiplicity", 1) - 1
    rohf = case.get("rohf", False)
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=0,
        spin=two_s if rohf else 0,
        cart=False,
        verbose=0,
    )
    if rohf:
        fitted, aux_provenance = df_rohf(case, mol)
        # The fitted orbitals, over exact integrals: the core's J and K are
        # the SCF object's get_jk, so it must not be the fitted one.
        mf = scf.ROHF(mol)
        mf.mo_coeff = fitted.mo_coeff
        mf.mo_occ = fitted.mo_occ
        mf.mo_energy = fitted.mo_energy
        mf.e_tot = fitted.e_tot
    else:
        mf = scf.RHF(mol)
        # As active_integrals_reference.py: a CASCI energy moves to first
        # order in the orbitals.
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, f"{name}: RHF did not converge"
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
    else:
        nelecas, ncas = case["cas"]
        ncore = (mol.nelectron - nelecas) // 2
        active = case.get("active") or list(range(ncore, ncore + ncas))
        check_degeneracies(name, mf.mo_energy, active)
        if rohf:
            check_open_shell(name, mol, ncore, active)
    nelec = ((nelecas + two_s) // 2, (nelecas - two_s) // 2)
    mc = mcscf.CASCI(mf, ncas, nelec)
    if "active" in case:
        mo = mc.sort_mo(case["active"], base=0)
    elif "avas" not in case:
        mo = mf.mo_coeff
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

    h1, e_core = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelec, 0.5)
    dense = numpy.linalg.eigvalsh(dense_hamiltonian(h2e, ncas, nelec)) + e_core
    error = numpy.abs(dense[:NROOTS] - energies).max()
    assert error < DENSE_TOL, f"{name}: Davidson off the dense spectrum by {error}"

    roots = []
    for energy, vector in zip(energies, vectors, strict=True):
        s2, _ = fci.spin_op.spin_square0(vector, ncas, nelec)
        if abs(s2 - two_s * (two_s + 2) / 4) < 1e-6:
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
    nkeep = case.get("nkeep", NKEEP)
    assert len(roots) >= nkeep, f"{name}: fewer than {nkeep} roots of its spin"

    out = {
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
        "roots": roots[:nkeep],
    }
    if "multiplicity" in case:
        out["multiplicity"] = case["multiplicity"]
    if rohf:
        out["scf"] = "rohf"
        out["aux_basis"] = aux_provenance
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--append",
        action="store_true",
        help="keep the file's cases as written; compute only the ones it lacks",
    )
    kept = json.loads(OUT.read_text())["cases"] if parser.parse_args().append else []
    written = {case["name"] for case in kept}
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-14, "
        "conv_tol_grad 1e-11) over this tree's data/basis files, and PySCF "
        "mcscf.CASCI over its canonical orbitals (the frontier ones, or "
        "'active' by sort_mo, 0-based, or by mcscf.avas over "
        "data/basis/minao.json) with fci.direct_spin1 (conv_tol 1e-13, "
        "checked against the dense spectrum) at N_alpha - N_beta = "
        "'multiplicity' - 1 (1 when absent): the lowest roots of that spin, "
        "their total energy (hartree), <S^2> and natural occupations. A case "
        "with 'scf' 'rohf' is over PySCF's DF-ROHF orbitals instead (at its "
        "multiplicity, its 'aux_basis' this tree's data/aux_basis file, "
        "conv_tol 1e-14, conv_tol_grad 1e-10), its 'rhf_energy' and "
        "'mo_energies' that ROHF's, and its CASCI integrals exact; "
        "coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit/casci_reference.py",
        },
        "cases": kept
        + [case_reference(case) for case in CASES if case["name"] not in written],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
