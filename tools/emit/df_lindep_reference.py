#!/usr/bin/env python3
"""Write the reference data for density_fit's lindep path (a near-singular
metric).

The fitting set is this tree's `data/aux_basis/cc-pVDZ-JKFIT.json` over H and
O with three of oxygen's shells (an s, a p and a d) appended a second time,
exactly: its Coulomb metric is singular in exact arithmetic, nine eigenvalues
at rounding level, so the default Cholesky fit has no meaningful factor, and
the eigen route drops exactly those nine. The set is written as a BSE JSON
document the tree reads by path (`resolve_aux_basis`).

Over it, for `scf_reference`'s water in this tree's cc-pVDZ: PySCF's DF-RHF
(`conv_tol` 1e-12) whose three-index tensor is PySCF's own eigen route,
`df.incore.cholesky_eri(..., decompose_j2c='eig', lindep=LINDEP)`, which is
`_eig_decompose`'s w > lindep cut, the cut `density_fit` takes; and the same
molecule's DF-RHF over the plain cc-pVDZ-JKFIT, which the duplicated set must
reproduce, since both span the same fitting space.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/df_lindep_reference.py

writes test/gtest/data/df_lindep/. BLAS and OpenMP are pinned to one thread
and nothing run-dependent is written, so a rerun regenerates every file byte
for byte. Re-run it to regenerate; the files are never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, one reduction order.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import copy  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
import scipy.linalg  # noqa: E402
from pyscf import df, gto, scf  # noqa: E402
from scf_reference import BASES, WATER, pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "df_lindep"
MANIFEST = OUT / "df_lindep.json"
AUX_OUT = OUT / "aux_basis.json"
JKFIT = ROOT / "data" / "aux_basis" / "cc-pVDZ-JKFIT.json"

# PySCF's own default (df.incore.LINEAR_DEP_THR), the threshold the test
# passes to density_fit.
LINDEP = 1e-7
# Oxygen's shells, by index in cc-pVDZ-JKFIT, appended again: its first s, p
# and d, 1 + 3 + 5 functions.
DUPLICATED = (0, 10, 17)
ELEMENTS = ("1", "8")
# The duplicated set spans the plain one's space, so their energies agree to
# the SCF's convergence and the eigen route's rounding.
SAME_SPACE_TOLERANCE = 1e-10

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/df_lindep_reference.py"
)


def duplicated_set() -> dict:
    """cc-pVDZ-JKFIT over H and O, oxygen's DUPLICATED shells appended."""
    doc = json.loads(JKFIT.read_text())
    doc["elements"] = {z: doc["elements"][z] for z in ELEMENTS}
    shells = doc["elements"]["8"]["electron_shells"]
    shells.extend(copy.deepcopy(shells[i]) for i in DUPLICATED)
    doc["name"] = "cc-pVDZ-JKFIT, three O shells duplicated"
    doc["description"] = (
        "Test fixture, not a fitting set: cc-pVDZ-JKFIT over H and O with O's "
        f"shells {list(DUPLICATED)} appended again, so its metric is singular. "
        f"Written by {COMMAND}"
    )
    return doc


def water(basis: dict) -> gto.Mole:
    return gto.M(
        atom=[[symbol, coord] for symbol, coord in WATER],
        unit="Bohr",
        basis=basis,
        cart=False,
        verbose=0,
    )


def converged(mf: scf.hf.SCF) -> float:
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    energy = mf.kernel()
    assert mf.converged
    return float(energy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    AUX_OUT.write_text(json.dumps(duplicated_set(), indent=1) + "\n")

    symbols = {symbol for symbol, _ in WATER}
    mol = water(pyscf_basis(BASES["cc-pvdz"], symbols))
    aux_file = os.path.relpath(AUX_OUT, ROOT / "data" / "basis")
    auxmol = df.addons.make_auxmol(mol, pyscf_basis(aux_file, symbols))
    plain_aux = df.addons.make_auxmol(
        mol, pyscf_basis("../aux_basis/cc-pVDZ-JKFIT.json", symbols)
    )
    dropped_expected = auxmol.nao - plain_aux.nao
    assert dropped_expected == 9, dropped_expected

    # The spectrum: the nine duplicates at rounding level, every other mode
    # far above the threshold, so any eigensolver drops exactly these.
    j2c = auxmol.intor("int2c2e", hermi=1)
    w = scipy.linalg.eigh(j2c, eigvals_only=True)
    dropped = int(numpy.count_nonzero(w <= LINDEP))
    assert dropped == dropped_expected, (dropped, w[:12])
    assert numpy.abs(w[:dropped]).max() < 1e-3 * LINDEP, w[:dropped]
    assert w[dropped] > 1e2 * LINDEP, w[dropped]

    cderi = df.incore.cholesky_eri(
        mol, auxmol=auxmol, decompose_j2c="eig", lindep=LINDEP
    )
    assert cderi.shape[0] == auxmol.nao - dropped
    mf = scf.RHF(mol).density_fit()
    mf.with_df.auxmol = auxmol
    mf.with_df._cderi = cderi
    energy = converged(mf)

    plain = scf.RHF(mol).density_fit(auxbasis=plain_aux._basis)
    plain_energy = converged(plain)
    assert abs(energy - plain_energy) < SAME_SPACE_TOLERANCE, energy - plain_energy

    doc = {
        "source": f"PySCF {pyscf.__version__}: df.incore.cholesky_eri "
        "decompose_j2c='eig', RHF conv_tol 1e-12",
        "command": COMMAND,
        "what": (
            "H2O in cc-pVDZ over cc-pVDZ-JKFIT with O's shells "
            f"{list(DUPLICATED)} duplicated (aux_basis.json): PySCF's DF-RHF "
            "over its eigen-decomposed metric, modes at or below lindep dropped, "
            "and the DF-RHF over the plain cc-pVDZ-JKFIT. Hartree, bohr"
        ),
        "atoms": [
            {"symbol": symbol, "z": int(gto.charge(symbol)), "coord": list(coord)}
            for symbol, coord in WATER
        ],
        "basis": "cc-pVDZ",
        "aux_basis": AUX_OUT.name,
        "naux": int(auxmol.nao),
        "lindep": LINDEP,
        "dropped": dropped,
        "energy": energy,
        "plain_aux_basis": "cc-pVDZ-JKFIT",
        "plain_energy": plain_energy,
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
