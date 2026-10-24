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
- coulomb/: the fixture's AO `density` (a Hilbert matrix, or for the H10
  chain a converged RHF density, recorded; its `density_recipe`) and
  libcint's Coulomb matrix `vj` of it;
- exchange/: the same `density` and libcint's exchange matrix `vk` of it
  (PySCF `get_jk`, vk_il = sum_jk (ij|kl) D_jk), from the molecule's `_k`
  fixture. That fixture was computed on its `_j` sibling's inputs, and this
  refuses to write it unless the two agree exactly -- basis, geometry and
  density, as FelizJueves's `oracle.manifest.check_pair` does -- so a K test
  and the J test read the same density. It also refuses a `vk` that is not
  symmetric to 1e-14 of its largest element (libcint computes both
  triangles);
- overlap_mixed/: libcint's cross-basis overlap `int1e_ovlp_cross:<rows>`
  (PySCF `gto.intor_cross`), with both bases: the row fixture's shells and
  the cross fixture's own as the columns.

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

# The molecules, each the golden files' stem and its fixtures' prefix.
# h2o_sto3g puts O's p shell before the H s shells (the la > lb swap);
# h2o_ccpvdz reaches d; h10_chain_ccpvdz (five H2 4 bohr apart) carries an RHF
# density, which decays with distance as the Hilbert one does not, so K's
# density-decay factor drops below max|D| there.
MOLECULES = ("h2o_sto3g", "h2o_ccpvdz", "h10_chain_ccpvdz")
# Per family, its directory under OUT, the fixture suffix it reads, and the
# fixture arrays it keeps: libcint's S, T and V (point nuclei) and PySCF's
# hcore = T + V; the AO density with libcint's J of it; and the same density
# with libcint's K of it.
FAMILIES = {
    "one_electron": ("_j", ("int1e_ovlp", "int1e_kin", "int1e_nuc", "hcore")),
    "coulomb": ("_j", ("density", "vj")),
    "exchange": ("_k", ("density", "vk")),
}
# How a file's description names its family.
LABEL = {"one_electron": "one-electron", "coulomb": "Coulomb", "exchange": "exchange"}
# The cross-basis overlap fixtures, each the golden file's stem: rows
# h2o_sto3g_j's AOs, columns 6-31G or cc-pVDZ at the same geometry, and 6-31G
# at another geometry. The key of the cross array names the row fixture.
CROSS_FIXTURES = ("h2o_sto3g_x_631g", "h2o_sto3g_x_ccpvdz", "h2o_sto3g_x_631g_moved")
CROSS_PREFIX = "int1e_ovlp_cross:"

# What a `_k` fixture must share with its `_j` sibling, exactly: every
# top-level input, and every provenance field but how it was regenerated.
PAIR_FIELDS = (
    "atoms",
    "shells",
    "nao",
    "ao_labels",
    "ao_loc",
    "converted",
    "density_recipe",
)
PAIR_PROVENANCE_EXEMPT = {"command", "date", "host"}
# vk's asymmetry bound, relative to max|vk|.
SYMMETRY_TOLERANCE = 1e-14


def felizjueves_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=FJ, check=True, capture_output=True, text=True
    ).stdout.strip()


def load(name: str) -> tuple[str, dict]:
    source = f"test/oracle/fixtures/{name}.json"
    return source, json.loads((FJ / source).read_text())


def check_pair(j_source: str, j: dict, k_source: str, k: dict) -> None:
    """Refuse, naming both files and the field, unless the `_k` fixture was
    computed on the `_j` fixture's inputs. Every comparison is exact."""

    def fail(field: str) -> SystemExit:
        return SystemExit(f"{k_source} and {j_source}: {field} differ")

    for field in PAIR_FIELDS:
        if j.get(field) != k.get(field):
            raise fail(field)
    pj, pk = j["provenance"], k["provenance"]
    for field in sorted((set(pj) | set(pk)) - PAIR_PROVENANCE_EXEMPT):
        if pj.get(field) != pk.get(field):
            raise fail(f"provenance.{field}")
    if j["arrays"]["density"] != k["arrays"]["density"]:
        raise fail("arrays.density")


def check_symmetric(source: str, name: str, array: dict) -> None:
    """Refuse an array that is not symmetric to SYMMETRY_TOLERANCE of its
    largest element."""
    rows, cols = array["shape"]
    values = array["values"]
    if rows != cols or len(values) != rows * cols:
        raise SystemExit(f"{source}: {name} is not square")
    bound = SYMMETRY_TOLERANCE * max(abs(v) for v in values)
    worst = max(
        abs(values[i * rows + c] - values[c * rows + i])
        for i in range(rows)
        for c in range(i + 1, rows)
    )
    if worst > bound:
        raise SystemExit(
            f"{source}: {name} is asymmetric by {worst!r} > {bound!r} "
            f"({SYMMETRY_TOLERANCE} * max|{name}|)"
        )


def shells(fixture: dict) -> list:
    """One shell per contraction, in AO order; coefficients multiply
    unnormalised Cartesian primitives x^a y^b z^c exp(-alpha r^2)."""
    return [
        {key: shell[key] for key in ("l", "centre", "exponents", "coefficients")}
        for shell in fixture["converted"]["shells"]
    ]


def check_cartesian(fixture: dict, source: str) -> None:
    if not fixture["provenance"]["cart"]:
        raise SystemExit(f"{source}: not a Cartesian fixture")


def extract_cross(stem: str, commit: str) -> dict:
    """The cross fixture `stem` and its row fixture: both bases and the
    rows x cols overlap, refused unless its shape is the two AO counts."""
    source, fixture = load(stem)
    check_cartesian(fixture, source)
    (key,) = [name for name in fixture["arrays"] if name.startswith(CROSS_PREFIX)]
    row_source, rows = load(key.removeprefix(CROSS_PREFIX))
    check_cartesian(rows, row_source)
    cross = fixture["arrays"][key]
    if cross["shape"] != [rows["nao"], fixture["nao"]]:
        raise SystemExit(f"{source}: {key} is {cross['shape']}, not rows x cols")
    return {
        "description": (
            f"{fixture['description']} -- both bases and the cross overlap only, "
            "copied by tools/extract_integral_golden.py"
        ),
        "source": {
            "felizjueves": commit,
            "fixture": source,
            "rows_fixture": row_source,
        },
        "provenance": fixture["provenance"],
        "rows": {"nao": rows["nao"], "shells": shells(rows)},
        "cols": {"nao": fixture["nao"], "shells": shells(fixture)},
        # rows.nao x cols.nao, row-major, each basis's AO order.
        "int1e_ovlp_cross": cross,
    }


def extract(
    fixture: dict, source: str, commit: str, family: str, arrays: tuple
) -> dict:
    check_cartesian(fixture, source)
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
        "shells": shells(fixture),
        # nao x nao, row-major, AO order.
        "arrays": {name: fixture["arrays"][name] for name in arrays},
    }
    if "density" in arrays:
        out["density_recipe"] = fixture["density_recipe"]
    return out


def main() -> int:
    commit = felizjueves_commit()
    for family, (suffix, arrays) in FAMILIES.items():
        out = OUT / family
        out.mkdir(parents=True, exist_ok=True)
        for stem in MOLECULES:
            source, fixture = load(stem + suffix)
            if suffix == "_k":
                check_pair(*load(stem + "_j"), source, fixture)
                check_symmetric(source, "vk", fixture["arrays"]["vk"])
            doc = extract(fixture, source, commit, family, arrays)
            path = out / f"{stem}.json"
            path.write_text(json.dumps(doc, indent=1) + "\n")
            print(path.relative_to(ROOT))
    out = OUT / "overlap_mixed"
    out.mkdir(parents=True, exist_ok=True)
    for stem in CROSS_FIXTURES:
        path = out / f"{stem}.json"
        path.write_text(json.dumps(extract_cross(stem, commit), indent=1) + "\n")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
