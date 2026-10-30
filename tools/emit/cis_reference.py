#!/usr/bin/env python3
"""Write the reference data the CIS (RHF/TDA) tests read.

Every case runs PySCF's closed-shell RHF (`conv_tol` 1e-12, `conv_tol_grad`
1e-9) and then its Tamm-Dancoff `TDA`, singlets and triplets, in the
spherical basis over this tree's own `data/basis/*.json` (and, density-fitted,
`data/aux_basis/cc-pVDZ-JKFIT.json`), handed to PySCF as explicit shells:

- H2O in STO-3G, 6-31G and cc-pVDZ, **exact** integrals: the lowest five
  singlets and triplets, the singlets' oscillator strengths; what the
  integral-direct sigma is held to.
- H2O in cc-pVDZ, **density-fitted** RHF + TDA over cc-pVDZ-JKFIT: the same;
  what the density-fitted sigma is held to (DF against DF over the same set).
- HF in cc-pVTZ, exact: the lowest singlets, energies only, an f-shell check
  of the sigma (the dipole stops at d).
- Benzene in cc-pVDZ, density-fitted over cc-pVDZ-JKFIT and exact: the lowest
  five singlets and their oscillator strengths.
- TDA-ris (Zhou & Parker 2024; gpu4pyscf's `tdscf/ris.py`), `method` "ris":
  the same RHF, then A's two-electron part over two minimal RIS fits instead,
  2 (ia|jb) over the J fit (`sp`: s and p on a heavy atom, s on H) and (ij|ab)
  over the K fit (`s`), each one uncontracted shell per l of exponent
  theta / R_A^2 (R_A the Ghosh radius in bohr, theta 0.2), handed to PySCF's
  `df.DF` as explicit aux bases: H2O in cc-pVDZ over the exact RHF, singlets
  and triplets, and benzene in cc-pVDZ over the cc-pVDZ-JKFIT RHF, singlets;
  what `[cis] method = "ris"` is held to (RIS against RIS over the same aux
  bases, never against full CIS).
- The spherical AO dipole matrices `int1e_r` (origin at 0) for H2O in STO-3G
  and cc-pVDZ, as .npy.

Before a case is written, its A matrix is built explicitly from PySCF's MO
integrals (exact `ao2mo`, or the DF object's own `ao2mo`) and diagonalized
densely; PySCF's Davidson roots must equal its lowest eigenvalues to 1e-8, so
no root is missed (measured: 3e-14 or better), and the oscillator strengths,
recomputed from `int1e_r` and the dense eigenvectors, PySCF's to 1e-6
(measured: 7e-8, benzene's bright pair). A disagreement is a refusal. PySCF
has no TDA-ris, so a RIS case's roots and oscillator strengths are the dense
A's own, from its eigenvalues and eigenvectors.

Degenerate roots (benzene's E states, HF's Pi) have no defined individual
oscillator strength or eigenvector, so roots are recorded in groups,
degenerate to 1e-6 hartree, each with its energies and its summed oscillator
strength, which is invariant. A recorded list of roots always ends at a group
boundary, so "the lowest five" can carry six.

Arrays are .npy, '<f8' in Fortran order, the layout `src/io/npy` reads. BLAS
and OpenMP are pinned to one thread, and nothing run-dependent (date, host,
timing) is written, so a rerun regenerates every file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/cis_reference.py

writes test/gtest/data/cis/. Re-run it to regenerate; the files are never
hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import ao2mo, df, gto, lib, scf  # noqa: E402
from scf_reference import WATER, pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "cis"
MANIFEST = OUT / "cis.json"

BASES = {
    "sto-3g": "sto-3g.json",
    "6-31g": "6-31g.json",
    "cc-pvdz": "cc-pVDZ.json",
    "cc-pvtz": "cc-pVTZ.json",
}
# pyscf_basis reads from data/basis, hence the relative path.
AUX_FILES = {"cc-pvdz-jkfit": "../aux_basis/cc-pVDZ-JKFIT.json"}

# Hydrogen fluoride, 0.917 A (1.7329 bohr), the bond along no axis so every
# f component of fluorine's cc-pVTZ f shell is reached. Bohr.
_HF_BOND = numpy.array([0.48, 0.62, 0.62]) / numpy.linalg.norm([0.48, 0.62, 0.62])
HYDROGEN_FLUORIDE = [
    ("F", (0.13, -0.21, 0.17)),
    (
        "H",
        tuple(float(x) for x in numpy.array([0.13, -0.21, 0.17]) + 1.7329 * _HF_BOND),
    ),
]

# Benzene, D6h, C-C 1.397 A and C-H 1.084 A, in the xy plane. Bohr.
_CC = 1.397 / 0.52917721092
_CH = 1.084 / 0.52917721092
BENZENE = [
    (
        symbol,
        (radius * math.cos(k * math.pi / 3), radius * math.sin(k * math.pi / 3), 0.0),
    )
    for symbol, radius in (("C", _CC), ("H", _CC + _CH))
    for k in range(6)
]

# Two roots closer than this (hartree) are one degenerate group; the gap to the
# next group must be at least GROUP_GAP, or the grouping is ambiguous.
DEGENERATE = 1e-6
GROUP_GAP = 1e-4

# PySCF's Davidson residual-norm tolerance (its eigenvalues are then good to
# ~its square: 1e-13 or better against the dense A) and the norm squared below
# which it drops a new vector as linearly dependent. At PySCF's default
# lindep, 1e-12, a residual under ~1e-6 cannot be expanded on and stalls.
TDA_CONV_TOL = 1e-7
TDA_LINDEP = 1e-14
DENSE_TOL = 1e-8
F_TOL = 1e-6

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "roots": 5},
    {"name": "h2o_631g", "atoms": WATER, "basis": "6-31g", "roots": 5},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "roots": 5},
    {
        "name": "h2o_ccpvdz_df",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "roots": 5,
    },
    {
        "name": "hf_ccpvtz",
        "atoms": HYDROGEN_FLUORIDE,
        "basis": "cc-pvtz",
        "roots": 3,
        "triplets": False,
        "oscillator_strengths": False,
    },
    {
        "name": "benzene_ccpvdz_df",
        "atoms": BENZENE,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "roots": 5,
        "triplets": False,
    },
    {
        "name": "benzene_ccpvdz",
        "atoms": BENZENE,
        "basis": "cc-pvdz",
        "roots": 5,
        "triplets": False,
    },
]

# TDA-ris's fits, gpu4pyscf's TDDFT_ris defaults: one theta for both, the J
# fit `sp` and the K fit `s` (a heavy atom's shells; hydrogen gets s alone).
RIS_THETA = 0.2
RIS_J_MAX_L = 1
RIS_K_MAX_L = 0
# The Ghosh (GB) radii in Angstrom of the elements the RIS cases carry:
# src/cis/ris_basis's kGhoshRadii (gpu4pyscf's parameter.py `radii`), and its
# kBohrPerAngstrom (CODATA 2018).
GHOSH_RADII = {"H": 0.5292, "C": 0.6513, "O": 0.4652}
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

CASES += [
    {
        "name": "h2o_ccpvdz_ris",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "method": "ris",
        "roots": 5,
    },
    {
        "name": "benzene_ccpvdz_df_ris",
        "atoms": BENZENE,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "method": "ris",
        "roots": 5,
        "triplets": False,
    },
]

# The cases whose int1e_r is written.
DIPOLE_CASES = ("h2o_sto3g", "h2o_ccpvdz")

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/cis_reference.py"
)


def molecule(case: dict) -> gto.Mole:
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        cart=False,
        verbose=0,
    )
    assert numpy.abs(numpy.diag(mol.intor("int1e_ovlp_sph")) - 1).max() < 1e-12
    return mol


def rhf(case: dict, mol: gto.Mole) -> scf.hf.SCF:
    """PySCF's converged RHF, density-fitted over exactly the case's aux set."""
    mf = scf.RHF(mol)
    if "aux_basis" in case:
        symbols = {symbol for symbol, _ in case["atoms"]}
        aux = pyscf_basis(AUX_FILES[case["aux_basis"]], symbols)
        auxmol = df.addons.make_auxmol(mol, aux)
        mf = mf.density_fit(auxbasis=aux)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{case['name']}: RHF did not converge"
    if "aux_basis" in case:
        used = mf.with_df.auxmol
        assert numpy.array_equal(used._bas, auxmol._bas)
        assert numpy.array_equal(used._env, auxmol._env)
    return mf


def ris_aux(symbols: set[str], max_l: int) -> dict[str, list]:
    """The RIS minimal aux basis in PySCF's form: per element one
    single-primitive shell per l up to max_l (s alone on H), all of exponent
    theta / R^2, R the Ghosh radius in bohr."""
    out = {}
    for symbol in symbols:
        radius = GHOSH_RADII[symbol] * BOHR_PER_ANGSTROM
        exponent = RIS_THETA / (radius * radius)
        top = 0 if symbol == "H" else max_l
        out[symbol] = [[ang, [exponent, 1.0]] for ang in range(top + 1)]
    return out


def ris_fit(mol: gto.Mole, max_l: int) -> df.DF:
    """PySCF's density fitting of mol over its RIS basis up to max_l, the aux
    molecule checked to carry exactly those exponents."""
    symbols = {mol.atom_symbol(k) for k in range(mol.natm)}
    fit = df.DF(mol, auxbasis=ris_aux(symbols, max_l))
    fit.build()
    auxmol = fit.auxmol
    for shell in range(auxmol.nbas):
        symbol = auxmol.atom_symbol(auxmol.bas_atom(shell))
        radius = GHOSH_RADII[symbol] * BOHR_PER_ANGSTROM
        assert auxmol.bas_nprim(shell) == 1
        assert auxmol.bas_exp(shell)[0] == RIS_THETA / (radius * radius)
    return fit


def dense_a(mf: scf.hf.SCF, singlet: bool, ris: bool = False) -> numpy.ndarray:
    """The TDA A matrix, (ia, jb), built from the MO integrals PySCF's own SCF
    object integrates with: A = delta (e_a - e_i) + 2 (ia|jb) - (ij|ab) for a
    singlet, without the 2 (ia|jb) for a triplet. With ris, TDA-ris's: (ia|jb)
    over the J fit and (ij|ab) over the K fit, whatever the SCF's integrals."""
    nocc = mf.mol.nelectron // 2
    mo = mf.mo_coeff
    occ, vir = mo[:, :nocc], mo[:, nocc:]
    nvir = vir.shape[1]
    if ris:
        coulomb = ris_fit(mf.mol, RIS_J_MAX_L).ao2mo
        exchange = ris_fit(mf.mol, RIS_K_MAX_L).ao2mo
    elif getattr(mf, "with_df", None) is not None:
        coulomb = exchange = mf.with_df.ao2mo
    else:

        def integrals(orbitals, compact=False):
            return ao2mo.general(mf.mol, orbitals, compact=compact)

        coulomb = exchange = integrals

    ovov = coulomb((occ, vir, occ, vir), compact=False)
    oovv = exchange((occ, occ, vir, vir), compact=False)
    ovov = ovov.reshape(nocc, nvir, nocc, nvir)
    oovv = oovv.reshape(nocc, nocc, nvir, nvir)
    a = -oovv.transpose(0, 2, 1, 3).copy()
    if singlet:
        a += 2.0 * ovov
    a = a.reshape(nocc * nvir, nocc * nvir)
    gap = (mf.mo_energy[nocc:][None, :] - mf.mo_energy[:nocc][:, None]).ravel()
    a[numpy.diag_indices_from(a)] += gap
    assert numpy.abs(a - a.T).max() < 1e-12
    return a


def groups(energies: numpy.ndarray) -> list[list[int]]:
    """The roots in degenerate groups, each a list of indices."""
    out = [[0]]
    for k in range(1, len(energies)):
        gap = energies[k] - energies[k - 1]
        assert gap < DEGENERATE or gap > GROUP_GAP, f"ambiguous gap {gap}"
        if gap < DEGENERATE:
            out[-1].append(k)
        else:
            out.append([k])
    return out


def tda(case: dict, mf: scf.hf.SCF, dipole: numpy.ndarray, singlet: bool) -> dict:
    """PySCF's TDA roots, checked against the dense A, in whole groups."""
    wanted = case["roots"]
    ris = case.get("method") == "ris"
    a = dense_a(mf, singlet, ris)
    dense, vectors = numpy.linalg.eigh(a)
    # Three roots past the wanted ones, more until a group boundary: a
    # Davidson space that cuts a degenerate pair stalls (or diverges, on HF).
    # The dense spectrum only sizes the run; the roots are PySCF's own.
    nstates = wanted + 3
    while dense[nstates] - dense[nstates - 1] < GROUP_GAP:
        nstates += 1
    if ris:
        # No PySCF TDA runs this operator: the dense A's roots are the
        # reference, and its eigenvectors the oscillator strengths'.
        energies = dense[:nstates]
    else:
        td = mf.TDA()
        td.singlet = singlet
        td.nstates = nstates
        td.conv_tol = TDA_CONV_TOL
        td.lindep = TDA_LINDEP
        td.max_cycle = 300
        td.kernel()
        assert td.converged.all(), f"{case['name']}: TDA did not converge"
        energies = numpy.asarray(td.e)
        assert len(energies) == nstates
        error = numpy.abs(energies - dense[:nstates]).max()
        assert error < DENSE_TOL, f"{case['name']}: TDA misses dense A by {error:.2e}"

    # Whole groups, up to the first boundary at or past the wanted count.
    kept = []
    for group in groups(energies):
        if sum(len(g) for g in kept) >= wanted:
            break
        kept.append(group)
    assert sum(len(g) for g in kept) >= wanted, f"{case['name']}: too few roots"

    out = {
        "energies": [float(energies[k]) for g in kept for k in g],
        "groups": [[int(k) for k in g] for g in kept],
    }
    if not ris:
        out["dense_max_error"] = float(error)
    if not singlet or not case.get("oscillator_strengths", True):
        return out

    # f = 2/3 w |<0|r|n>|^2, <0|r|n> = sqrt(2) sum_ia r_ia X_ia with X unit
    # normalised; per group, summed. PySCF's from its own X (normalised to 1/2)
    # and the dense A's eigenvectors must agree to F_TOL. f is quadratic in
    # the vector, whose error is the residual over the gap, so this is looser
    # than the energies' check: ~1e-7 for benzene's bright E1u pair.
    pyscf_f = None if ris else numpy.asarray(td.oscillator_strength(gauge="length"))
    nocc = mf.mol.nelectron // 2
    mo = mf.mo_coeff
    r_ov = numpy.einsum("xmn,mi,na->xia", dipole, mo[:, :nocc], mo[:, nocc:])
    r_ov = r_ov.reshape(3, -1)
    sums = []
    f_error = 0.0
    for group in kept:
        mine = sum(
            2.0 / 3.0 * dense[k] * 2.0 * float(numpy.sum((r_ov @ vectors[:, k]) ** 2))
            for k in group
        )
        if ris:
            sums.append(mine)
            continue
        theirs = float(sum(pyscf_f[k] for k in group))
        assert abs(mine - theirs) < F_TOL, f"{case['name']}: f {mine} vs {theirs}"
        f_error = max(f_error, abs(mine - theirs))
        sums.append(theirs)
    out["oscillator_strengths"] = sums
    if not ris:
        out["dense_max_oscillator_error"] = f_error
    return out


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy; its file name."""
    path = OUT / f"{name}.npy"
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.name


def cis_case(case: dict, write: bool = True) -> dict:
    """One case's RHF and TDA roots, and its int1e_r file when write is set."""
    mol = molecule(case)
    mf = rhf(case, mol)
    dipole = mol.intor("int1e_r_sph")
    assert dipole.shape == (3, mol.nao, mol.nao)
    out = {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case.get("aux_basis"),
        **(
            {
                "method": "ris",
                "ris_theta": RIS_THETA,
                "ris_j_fit": "sp",
                "ris_k_fit": "s",
            }
            if case.get("method") == "ris"
            else {}
        ),
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "occupied": int(mol.nelectron // 2),
        "electrons": int(mol.nelectron),
        "rhf_energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "singlets": tda(case, mf, dipole, singlet=True),
    }
    if case.get("triplets", True):
        out["triplets"] = tda(case, mf, dipole, singlet=False)
    if write and case["name"] in DIPOLE_CASES:
        out["int1e_r"] = save(f"{case['name']}_int1e_r", dipole)
    return out


def main() -> None:
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-12, "
        "conv_tol_grad 1e-9) and TDA (CIS) over this tree's data/basis and "
        "data/aux_basis files, exact or density-fitted (aux_basis): excitation "
        "energies (hartree) in degenerate groups (1e-6) and each singlet "
        "group's summed length-gauge oscillator strength; every root checked "
        "against the dense A to 1e-8; int1e_r is the spherical AO dipole "
        "(3, nao, nao), origin 0; coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "tda_conv_tol": TDA_CONV_TOL,
            "tda_lindep": TDA_LINDEP,
            "command": COMMAND,
        },
        "cases": [cis_case(case) for case in CASES],
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
