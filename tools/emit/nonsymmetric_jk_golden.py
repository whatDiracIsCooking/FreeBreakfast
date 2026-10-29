#!/usr/bin/env python3
"""Write libcint's J and K of a random NON-SYMMETRIC density as gtest golden
data: test/gtest/data/exchange/<molecule>_nonsymmetric.json.

free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk builds J[D] and
K[D] of a D that is neither symmetric nor antisymmetric (a CIS transition
density), and FelizJueves's oracle fixtures carry only symmetric densities, so
this records PySCF's `get_jk(mol, D, hermi=0)` of one directly:

    vj_kl = sum_ij (kl|ij) D_ji,   vk_il = sum_jk (ij|kl) D_jk

vk is K's convention, K_ab = sum_cd D_cd (ac|bd), and vj is J of D's
symmetric part, since (kl|ij) is symmetric in i and j. Both are checked
against an explicit einsum over libcint's `int2e` before anything is written.

Each molecule is the exchange golden of the same stem
(test/gtest/data/exchange/<molecule>.json, written by
tools/extract_integral_golden.py from FelizJueves's fixture): its molecule,
basis name and Cartesian flag rebuild the PySCF Mole, which must match the
golden's AO count and shells' momenta and exponents, and reproduce its
recorded vk of its recorded density to 1e-13 of max|vk|, so the Mole is the
basis the golden's shells describe. The shells are copied from that golden,
so the test builds exactly its basis. Cartesian, as every exchange golden is.

D is uniform on [-1, 1), numpy's default_rng(SEED), row-major in AO order:
non-symmetric and dense, with |D_cd| <= 1, inside the |D| <= 2 that J's
pruned classes assume. BLAS and OpenMP are pinned to one thread and nothing
run-dependent (date, host) is written, so a rerun regenerates every file byte
for byte at the same PySCF and numpy.

    uv run --group tools tools/emit/nonsymmetric_jk_golden.py

(in the devcontainer with UV_PROJECT_ENVIRONMENT=$PWD/.venv). Run
tools/extract_integral_golden.py first after bumping deps/FelizJueves.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one order.
if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, scf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "test/gtest/data/exchange"
# H2O in STO-3G (O's p shell before the H s shells, the la > lb swap) and in
# cc-pVDZ (every class through (dd|dd)), the issue's two (#194).
MOLECULES = ("h2o_sto3g", "h2o_ccpvdz")
# The issue's number: any fixed seed would do.
SEED = 194
# How far PySCF's vk of the golden's density may sit from the golden's,
# relative to max|vk|: the same code on the same inputs, so rounding only.
REPRODUCE_TOLERANCE = 1e-13
# get_jk against the einsum over int2e, relative to the larger's maximum.
EINSUM_TOLERANCE = 1e-13


def rows(array: numpy.ndarray) -> dict:
    """{shape, values}, values row-major, as the exchange goldens hold them."""
    return {"shape": list(array.shape), "values": array.ravel().tolist()}


def check_close(
    stem: str, name: str, got: numpy.ndarray, want: numpy.ndarray, tolerance: float
) -> None:
    bound = tolerance * numpy.abs(want).max()
    worst = numpy.abs(got - want).max()
    if not worst <= bound:
        raise SystemExit(f"{stem}: {name} differs by {worst!r} > {bound!r}")


def mole(stem: str, golden: dict) -> gto.Mole:
    """The golden's molecule as PySCF was given it, refused unless it is the
    basis the golden's shells describe."""
    p = golden["provenance"]
    if not p["cart"]:
        raise SystemExit(f"{stem}: not a Cartesian golden")
    m = p["mol"]
    mol = gto.M(
        atom=m["atom"],
        unit=m["unit"],
        charge=m["charge"],
        spin=m["spin"],
        basis=p["basis"],
        cart=True,
        verbose=0,
    )
    if mol.nao != golden["nao"]:
        raise SystemExit(f"{stem}: PySCF has {mol.nao} AOs, the golden {golden['nao']}")
    # PySCF holds a general contraction as one shell of bas_nctr contractions;
    # the golden, one shell per contraction.
    expanded = [
        (mol.bas_angular(i), mol.bas_exp(i).tolist())
        for i in range(mol.nbas)
        for _ in range(mol.bas_nctr(i))
    ]
    if expanded != [(s["l"], s["exponents"]) for s in golden["shells"]]:
        raise SystemExit(f"{stem}: the golden's shells are not PySCF's")
    n = golden["nao"]
    arrays = golden["arrays"]
    density = numpy.array(arrays["density"]["values"]).reshape(n, n)
    _, vk = scf.hf.get_jk(mol, density, hermi=1)
    check_close(
        stem,
        "PySCF's vk of the golden's density",
        vk,
        numpy.array(arrays["vk"]["values"]).reshape(n, n),
        REPRODUCE_TOLERANCE,
    )
    return mol


def emit(stem: str) -> dict:
    golden = json.loads((DIR / f"{stem}.json").read_text())
    mol = mole(stem, golden)
    n = mol.nao
    density = numpy.random.default_rng(SEED).uniform(-1.0, 1.0, (n, n))
    vj, vk = scf.hf.get_jk(mol, density, hermi=0)
    eri = mol.intor("int2e")
    check_close(
        stem, "vj", vj, numpy.einsum("ijkl,ji->kl", eri, density), EINSUM_TOLERANCE
    )
    check_close(
        stem, "vk", vk, numpy.einsum("ijkl,jk->il", eri, density), EINSUM_TOLERANCE
    )
    # Else the case could not tell K[D] from K of D's symmetric part.
    if numpy.abs(vk - vk.T).max() < 1e-3 * numpy.abs(vk).max():
        raise SystemExit(f"{stem}: vk is nearly symmetric")
    return {
        "description": (
            f"{stem}'s molecule and basis (the exchange golden's), a random "
            "non-symmetric AO density, and PySCF's get_jk(hermi=0) of it: "
            "vj_kl = sum_ij (kl|ij) D_ji and vk_il = sum_jk (ij|kl) D_jk, "
            "written by tools/emit/nonsymmetric_jk_golden.py"
        ),
        "source": {
            "shells": f"test/gtest/data/exchange/{stem}.json",
            "call": "pyscf.scf.hf.get_jk(mol, density, hermi=0)",
        },
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "mol": golden["provenance"]["mol"],
            "basis": golden["provenance"]["basis"],
            "cart": True,
        },
        "nao": n,
        "atoms": golden["atoms"],
        "shells": golden["shells"],
        "density_recipe": (
            f"numpy.random.default_rng({SEED}).uniform(-1, 1, (nao, nao)), "
            "row-major: non-symmetric, |D| < 1"
        ),
        # nao x nao, row-major, AO order.
        "arrays": {"density": rows(density), "vj": rows(vj), "vk": rows(vk)},
    }


def main() -> int:
    for stem in MOLECULES:
        path = DIR / f"{stem}_nonsymmetric.json"
        path.write_text(json.dumps(emit(stem), indent=1) + "\n")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
