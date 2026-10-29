#!/usr/bin/env python3
"""Write the reference data the density-fitting tests read.

Everything runs over this tree's own basis files, `data/basis/*.json` for the
orbitals and `data/aux_basis/*.json` for the fitting sets, handed to PySCF as
explicit shells (not its library's copies), spherical throughout:

- libcint's `int2c2e_sph`, the Coulomb metric (P|Q), and `int3c2e_sph`, the
  three-centre (mn|P), for H2O in cc-pVDZ / cc-pVDZ-JKFIT (orbitals to d, aux
  to f) and in cc-pVTZ / cc-pVTZ-JKFIT (oxygen's f orbitals and g fitting
  functions, so (ff|g) is reached).
- PySCF's DF `vj` and `vk` (`RHF(mol).density_fit(auxbasis=...)`'s `get_jk`)
  of one fixed density on the cc-pVDZ case: D = 2 C C^T over the five
  occupied orbitals of that case's own converged DF-RHF, so a test can
  factor it again.
- On the same case, that DF-RHF's orbitals C and PySCF's density-fitted MO
  integrals over them (`with_df.ao2mo`, the same fitting set): the
  occ-virt-occ-virt, occ^4 and virt^4 blocks of (pq|rs).
- DF-RHF (`conv_tol` 1e-12): total energy and orbital energies for H2O and
  NH4+ in cc-pVDZ / cc-pVDZ-JKFIT, H2O in cc-pVTZ / cc-pVTZ-JKFIT and H2O in
  cc-pVDZ / cc-pVDZ-acCD (general-contracted fitting shells), at
  `scf/rhf.json`'s geometries.
- DF-ROHF (`conv_tol` 1e-12, internally stable), in `df_rohf.json`: total
  energy, orbital energies (PySCF's default Roothaan effective Fock), spin
  and n_alpha / n_beta for every `scf/rohf.json` case whose orbital basis
  has a JKFIT pairing here (all but STO-3G), at its geometry.

Arrays are .npy, '<f8' in Fortran order, the layout `src/io/npy` reads; the
manifest `density_fitting.json` says which file holds what. A file over the
512 KiB limit is split along the aux index into balanced pieces. BLAS and
OpenMP are pinned to one thread, and nothing run-dependent (date, host) is
written, so a rerun regenerates every file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/df_reference.py

writes test/gtest/data/density_fitting/. Re-run it to regenerate; the files
are never hand-edited.
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
import re  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import df, gto, lib, scf  # noqa: E402
from scf_reference import (  # noqa: E402
    AMMONIUM,
    BASES,
    ROHF_CASES,
    WATER,
    pyscf_basis,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "density_fitting"
MANIFEST = OUT / "density_fitting.json"
ROHF_MANIFEST = OUT / "df_rohf.json"

# Under FelizJueves's pre-commit large-file hook, which this tree's data
# follows.
LIMIT = 512 * 1024
# np.save's header for a 3-D '<f8' array, padded to 64 bytes: 128 bytes.
NPY_HEADER = 128

# Each fitting set's file under data/aux_basis (pyscf_basis reads from
# data/basis, hence the relative path), and the one each builtin orbital
# basis pairs with unless a case names its own "aux_basis".
AUX_FILES = {
    "cc-pvdz-jkfit": "../aux_basis/cc-pVDZ-JKFIT.json",
    "cc-pvtz-jkfit": "../aux_basis/cc-pVTZ-JKFIT.json",
    "cc-pvdz-accd": "../aux_basis/cc-pVDZ-acCD.json",
}
AUX_BASES = {"cc-pvdz": "cc-pvdz-jkfit", "cc-pvtz": "cc-pvtz-jkfit"}

# The integrals' water: FelizJueves's h2o_ccpvdz_j molecule (and its
# cc-pVTZ-JKFIT metric fixtures'), no atom on an axis past O and the molecule
# in no symmetry plane, so every component of every shell is reached. Bohr.
WATER_SKEW = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (1.43, 0.21, -1.07)),
    ("H", (-1.38, 0.35, -1.12)),
]

INTEGRAL_CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER_SKEW, "basis": "cc-pvdz", "max_l": (2, 3)},
    {"name": "h2o_ccpvtz", "atoms": WATER_SKEW, "basis": "cc-pvtz", "max_l": (3, 4)},
]
# The case vj and vk are recorded on.
JK_CASE = "h2o_ccpvdz"

SCF_CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
    {"name": "h2o_ccpvtz", "atoms": WATER, "basis": "cc-pvtz", "charge": 0},
    # tools/atomic_cd's set: general contractions of up to 22 primitives
    {
        "name": "h2o_ccpvdz_accd",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-accd",
        "charge": 0,
    },
]

# scf_reference's open-shell cases, wherever their basis has a fitting set.
ROHF_SCF_CASES = [case for case in ROHF_CASES if case["basis"] in AUX_BASES]

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/df_reference.py"
)


def libcint_version() -> str:
    """The libcint PySCF ships, read from its cint.h as FelizJueves does."""
    header = pathlib.Path(pyscf.__file__).parent / "lib" / "deps" / "include"
    match = re.search(
        r'#define\s+CINT_VERSION\s+"([^"]+)"', (header / "cint.h").read_text()
    )
    assert match, "no CINT_VERSION in PySCF's cint.h"
    return match.group(1)


def molecules(case: dict) -> tuple[gto.Mole, gto.Mole, str]:
    """The orbital molecule, its fitting molecule and the fitting set's name."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    aux_name = case.get("aux_basis", AUX_BASES[case["basis"]])
    aux_file = AUX_FILES[aux_name]
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=case.get("charge", 0),
        spin=case.get("spin", 0),
        cart=False,
        verbose=0,
    )
    auxmol = df.addons.make_auxmol(mol, pyscf_basis(aux_file, symbols))
    # Unit-normalised functions on both sides, which the README promises. To
    # 1e-11, not 1e-12: an acCD shell's contraction over ~20 primitives with
    # cancelling coefficients rounds to 1.3e-12 on H2O's oxygen.
    for m in (mol, auxmol):
        assert numpy.abs(numpy.diag(m.intor("int1e_ovlp_sph")) - 1).max() < 1e-11
    return mol, auxmol, aux_name


def df_scf(mol: gto.Mole, auxmol: gto.Mole, method: type = scf.RHF) -> scf.hf.SCF:
    """PySCF's converged DF-RHF (or method's, scf.ROHF) over exactly auxmol's
    fitting functions."""
    mf = method(mol).density_fit(auxbasis=auxmol._basis)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    used = mf.with_df.auxmol
    assert used.nao == auxmol.nao
    assert numpy.array_equal(used._bas, auxmol._bas)
    assert numpy.array_equal(used._env, auxmol._env)
    return mf


def shells(m: gto.Mole) -> list[dict]:
    """One per PySCF shell, in AO order: its atom, l and contraction count.

    A shell of nctr contractions holds nctr * (2l + 1) AOs, contraction-major
    (each contraction's m = -l..l together), which is the order this tree
    gets by taking each BSE coefficient column as a shell of its own.
    """
    return [
        {
            "atom": int(m.bas_atom(i)),
            "l": int(m.bas_angular(i)),
            "nctr": int(m.bas_nctr(i)),
        }
        for i in range(m.nbas)
    ]


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy, under the limit; its file name."""
    array = numpy.asfortranarray(array, dtype="<f8")
    path = OUT / f"{name}.npy"
    numpy.save(path, array, allow_pickle=False)
    assert path.stat().st_size <= LIMIT, f"{path.name} is over {LIMIT} bytes"
    return path.name


def save_by_last(name: str, array: numpy.ndarray, key: str) -> list[dict]:
    """Write array in balanced pieces along its last index, each under the
    limit and listed with its [begin, end) under key."""
    total = array.shape[-1]
    per_slab = math.prod(array.shape[:-1]) * 8
    pieces = math.ceil(total / ((LIMIT - NPY_HEADER) // per_slab))
    bounds = [total * k // pieces for k in range(pieces + 1)]
    out = []
    for begin, end in zip(bounds, bounds[1:], strict=False):
        stem = name if pieces == 1 else f"{name}_{key}{begin}_{end}"
        out.append({"file": save(stem, array[..., begin:end]), key: [begin, end]})
    return out


def save_by_aux(name: str, array: numpy.ndarray) -> list[dict]:
    """Write (nao, nao, naux) in balanced pieces along P, each under the limit."""
    return save_by_last(name, array, "aux")


def atoms(case: dict) -> list[dict]:
    return [
        {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
        for symbol, coord in case["atoms"]
    ]


def integral_case(case: dict) -> dict:
    """The metric and three-centre integrals of one case, and vj/vk on JK_CASE."""
    mol, auxmol, aux_name = molecules(case)
    lmax = max(mol.bas_angular(i) for i in range(mol.nbas))
    aux_lmax = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))
    assert (lmax, aux_lmax) == case["max_l"], (lmax, aux_lmax)

    metric = auxmol.intor("int2c2e_sph")
    assert numpy.abs(metric - metric.T).max() < 1e-14 * numpy.abs(metric).max()
    assert numpy.linalg.eigvalsh(metric).min() > 0.0
    eri3 = df.incore.aux_e2(mol, auxmol, "int3c2e_sph", aosym="s1")
    assert eri3.shape == (mol.nao, mol.nao, auxmol.nao)
    assert (
        numpy.abs(eri3 - eri3.transpose(1, 0, 2)).max() < 1e-14 * numpy.abs(eri3).max()
    )

    out = {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": aux_name,
        "atoms": atoms(case),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "shells": shells(mol),
        "aux_shells": shells(auxmol),
        "int2c2e": save(f"{case['name']}_int2c2e", metric),
        "int3c2e": save_by_aux(f"{case['name']}_int3c2e", eri3),
    }
    if case["name"] != JK_CASE:
        return out

    mf = df_scf(mol, auxmol)
    nocc = mol.nelectron // 2
    occupied = mf.mo_coeff[:, :nocc]
    density = 2.0 * occupied @ occupied.T
    vj, vk = mf.get_jk(mol, density, hermi=1)
    # The same from the integrals written above: J = (mn|P) (P|Q)^-1 (Q|ls) D_ls
    # and K_ml = sum (mn|P) (P|Q)^-1 (Q|ks) D_ns over n, k.
    fitted = numpy.linalg.solve(metric, eri3.reshape(-1, auxmol.nao).T)
    fitted = fitted.T.reshape(eri3.shape)
    vj_check = numpy.einsum("mnP,lsP,ls->mn", eri3, fitted, density)
    vk_check = numpy.einsum("mnP,ksP,ns->mk", eri3, fitted, density)
    assert numpy.abs(vj - vj_check).max() < 1e-10 * numpy.abs(vj).max()
    assert numpy.abs(vk - vk_check).max() < 1e-10 * numpy.abs(vk).max()
    out["jk"] = {
        "density": save(f"{case['name']}_density", density),
        "vj": save(f"{case['name']}_vj", vj),
        "vk": save(f"{case['name']}_vk", vk),
        "occupied": nocc,
        "df_rhf_energy": float(mf.e_tot),
    }

    # (pq|rs) = sum_Q B_Q,pq B_Q,rs over PySCF's own cderi, each block
    # (np, nq, nr, ns), and the same from the integrals written above.
    mo = mf.mo_coeff
    spaces = {"o": mo[:, :nocc], "v": mo[:, nocc:]}
    out["mo"] = {"coefficients": save(f"{case['name']}_mo_coeff", mo), "occupied": nocc}
    for block in ("ovov", "oooo", "vvvv"):
        cs = [spaces[x] for x in block]
        eri = mf.with_df.ao2mo(tuple(cs), compact=False)
        eri = eri.reshape([c.shape[1] for c in cs])
        check = numpy.einsum(
            "mnP,lsP,mp,nq,lr,st->pqrt", eri3, fitted, *cs, optimize=True
        )
        assert numpy.abs(eri - check).max() < 1e-10 * numpy.abs(eri).max(), block
        out["mo"][block] = save_by_last(f"{case['name']}_{block}", eri, "s")
    return out


def scf_case(case: dict) -> dict:
    """One DF-RHF reference: total energy and orbital energies."""
    mol, auxmol, aux_name = molecules(case)
    mf = df_scf(mol, auxmol)
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": aux_name,
        "charge": case["charge"],
        "atoms": atoms(case),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "electrons": int(mol.nelectron),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }


def rohf_case(case: dict) -> dict:
    """One DF-ROHF reference, internally stable: total energy, orbital
    energies and the spin."""
    mol, auxmol, aux_name = molecules(case)
    mf = df_scf(mol, auxmol, scf.ROHF)
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    n_alpha, n_beta = mol.nelec
    assert n_alpha - n_beta == case["spin"]
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": aux_name,
        "charge": case["charge"],
        "spin": case["spin"],
        "multiplicity": case["spin"] + 1,
        "atoms": atoms(case),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "electrons": int(mol.nelectron),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }


def main() -> None:
    lib.num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.npy"):
        stale.unlink()
    provenance = {
        "pyscf": pyscf.__version__,
        "libcint": libcint_version(),
        "numpy": numpy.__version__,
        "threads": 1,
        "command": COMMAND,
    }
    doc = {
        "description": "PySCF/libcint density fitting over this tree's data/basis "
        "and data/aux_basis files, spherical: (P|Q), (mn|P), DF vj/vk of one "
        "density, DF-RHF energies (hartree); coordinates in bohr. Arrays are "
        "'<f8' Fortran-order .npy files beside this one",
        "provenance": provenance,
        "integrals": [integral_case(case) for case in INTEGRAL_CASES],
        "df_rhf": [scf_case(case) for case in SCF_CASES],
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    rohf_doc = {
        "description": "PySCF DF-ROHF (spherical, conv_tol 1e-12, conv_tol_grad "
        "1e-9, internally stable) over this tree's data/basis and "
        "data/aux_basis files: total energy, nuclear repulsion and orbital "
        "energies, the eigenvalues of PySCF's default Roothaan effective Fock "
        "(hartree); spin is 2S = n_alpha - n_beta; coordinates in bohr",
        "provenance": provenance,
        "df_rohf": [rohf_case(case) for case in ROHF_SCF_CASES],
    }
    ROHF_MANIFEST.write_text(json.dumps(rohf_doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
