#!/usr/bin/env python3
"""Write the active-space Hamiltonians bench/ci_sigma_precision times ci_solve over.

N2 in cc-pVDZ at its equilibrium bond (1.0977 A) and stretched to 2.0 A, the
second strongly multireference, each with a CAS(10,10) and a CAS(12,12) over
PySCF's closed-shell RHF orbitals, the lowest-energy ones above the frozen
core. As in fci_reference.py: E_core and h_eff from `get_h1eff`, (tu|vw)
from `get_h2eff` restored to n^4, chemists' order, row-major, in that file's
`cases` schema, so the bench reads either file. Nothing is solved here; the
bench compares its solves with each other, not with PySCF.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/ci_bench_integrals.py build-bench/ci_bench_integrals.json

The output is not committed: it is a few MB of benchmark input, regenerated in
seconds.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy
from pyscf import ao2mo, gto, mcscf, scf

ANGSTROM_TO_BOHR = 1.0 / 0.52917721092

CASES = [
    (f"n2-{bond}-cas({n},{n})", bond, n) for bond in (1.0977, 2.0) for n in (10, 12)
]


def values(array: numpy.ndarray) -> dict:
    """A JSON array with its shape, row-major."""
    return {"shape": list(array.shape), "values": [float(v) for v in array.ravel()]}


def case(name: str, bond: float, n: int) -> dict:
    half = 0.5 * bond * ANGSTROM_TO_BOHR
    mol = gto.M(
        atom=[("N", (0.0, 0.0, -half)), ("N", (0.0, 0.0, half))],
        basis="cc-pvdz",
        unit="Bohr",
        verbose=0,
    )
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(rhf, n, n)
    h1, e_core = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), n)
    return {
        "name": name,
        "n_orbitals": n,
        "n_alpha": n // 2,
        "n_beta": n // 2,
        "e_core": float(e_core),
        "h1": values(h1),
        "eri": values(eri),
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: ci_bench_integrals.py OUT.json")
    out = pathlib.Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"cases": [case(*c) for c in CASES]}
    out.write_text(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
