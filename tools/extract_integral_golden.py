#!/usr/bin/env python3
"""Copy the integral golden data the gtests read out of FelizJueves.

FelizJueves tracks PySCF / libcint fixtures (deps/FelizJueves/test/oracle/
fixtures/, provenance in each file). This takes what test/gtest needs of a few
of them -- the basis as Cartesian shells over unnormalised primitives (the
fixture's `converted` block, which is the convention free_breakfast's Shell and
the generated kernels use), the nuclei, and per integral family its arrays --
and writes one file per molecule and family under test/gtest/data/, so the
tests read a small file of their own and PySCF is needed by neither:

- one_electron/: libcint's `int1e_ovlp`, `int1e_kin` and `int1e_nuc`, and
  PySCF's `hcore`;
- coulomb/: the fixture's AO `density` (a Hilbert matrix, its
  `density_recipe`) and libcint's Coulomb matrix `vj` of it.

    tools/extract_integral_golden.py

Re-run it after bumping deps/FelizJueves. Standard library only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FJ = ROOT / "deps/FelizJueves"
SOURCE = FJ / "test/oracle/fixtures"
OUT = ROOT / "test/gtest/data"

# The molecule's fixture -> the golden file's stem. h2o_sto3g puts O's p shell
# before the H s shells (the la > lb swap); h2o_ccpvdz reaches d.
FIXTURES = {"h2o_sto3g_j": "h2o_sto3g", "h2o_ccpvdz_j": "h2o_ccpvdz"}
# Per family, its directory under OUT and the fixture arrays it keeps:
# libcint's S, T and V (point nuclei) and PySCF's hcore = T + V; and the AO
# density with libcint's J of it.
FAMILIES = {
    "one_electron": ("int1e_ovlp", "int1e_kin", "int1e_nuc", "hcore"),
    "coulomb": ("density", "vj"),
}
# How a file's description names its family.
LABEL = {"one_electron": "one-electron", "coulomb": "Coulomb"}


def felizjueves_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=FJ, check=True, capture_output=True, text=True
    ).stdout.strip()


def extract(
    fixture: dict, source: str, commit: str, family: str, arrays: tuple
) -> dict:
    if not fixture["provenance"]["cart"]:
        raise SystemExit(f"{source}: not a Cartesian fixture")
    out = {
        "description": (
            f"{fixture['description']} -- the basis and the {LABEL[family]} arrays "
            "only, copied by tools/extract_integral_golden.py"
        ),
        "source": {"felizjueves": commit, "fixture": source},
        "provenance": fixture["provenance"],
        "nao": fixture["nao"],
        # The nuclei V attracts to: symbol, charge Z, coord (bohr).
        "atoms": fixture["atoms"],
        # One shell per contraction, in AO order; coefficients multiply
        # unnormalised Cartesian primitives x^a y^b z^c exp(-alpha r^2).
        "shells": [
            {key: shell[key] for key in ("l", "centre", "exponents", "coefficients")}
            for shell in fixture["converted"]["shells"]
        ],
        # nao x nao, row-major, AO order.
        "arrays": {name: fixture["arrays"][name] for name in arrays},
    }
    if "density" in arrays:
        out["density_recipe"] = fixture["density_recipe"]
    return out


def main() -> int:
    commit = felizjueves_commit()
    for family, arrays in FAMILIES.items():
        out = OUT / family
        out.mkdir(parents=True, exist_ok=True)
        for name, stem in FIXTURES.items():
            source = f"test/oracle/fixtures/{name}.json"
            fixture = json.loads((FJ / source).read_text())
            doc = extract(fixture, source, commit, family, arrays)
            path = out / f"{stem}.json"
            path.write_text(json.dumps(doc, indent=1) + "\n")
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
