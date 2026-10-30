"""The committed nuclear-gradient references, test/gtest/data/gradient/.

Their invariants are checked with the standard library alone, so they run on
the push gate: the issue's cases are all there, every gradient is n_atom x 3
and sums to ~0 over atoms (translational invariance), each term set adds up
to its total, and each optimized geometry's max force is below 1e-6.
Re-running the cheapest case through tools/emit/gradient_reference.py needs
PySCF, which lives in the opt-in ``tools`` dependency group, so that test
skips (with a reason) wherever the group is not installed::

    uv run --group tools pytest -rs test/pytest/test_gradient_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "test" / "gtest" / "data" / "gradient" / "gradient.json"


def doc() -> dict:
    return json.loads(MANIFEST.read_text())


def section(name: str) -> dict[str, dict]:
    return {case["name"]: case for case in doc()[name]}


def gradients(case: dict) -> list[tuple[str, list]]:
    """Every gradient in one case, by its key path."""
    out = []

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}", inner)
        elif isinstance(value, list) and value and isinstance(value[0], list):
            out.append((prefix, value))

    for key, value in case.items():
        if key not in ("atoms", "start", "shells", "aux_shells"):
            walk(key, value)
    return out


def add(*gs: list) -> list:
    return [[sum(c) for c in zip(*rows, strict=True)] for rows in zip(*gs, strict=True)]


def assert_close(got: list, want: list, tol: float, what: str) -> None:
    for g, w in zip(got, want, strict=True):
        for x, y in zip(g, w, strict=True):
            assert abs(x - y) < tol, (what, x, y)


def test_the_issue_cases_are_covered():
    assert {(c["name"], c["basis"]) for c in doc()["integrals"]} == {
        ("h2o_ccpvdz", "cc-pvdz"),
        ("h2o_ccpvtz", "cc-pvtz"),
    }
    assert {c["aux_basis"] for c in doc()["density_fitting"]} == {
        "cc-pvdz-jkfit",
        "cc-pvtz-jkfit",
        "cc-pvtz-accd-6-6",
    }
    # the (ff|i) and (i|i) classes
    (high,) = [c for c in doc()["density_fitting"] if "accd" in c["aux_basis"]]
    assert max(s["l"] for s in high["aux_shells"]) == 6
    assert max(s["l"] for s in high["shells"]) == 3
    assert {(n, c["aux_basis"]) for n, c in section("df_rhf").items()} == {
        ("h2o_ccpvdz", "cc-pvdz-jkfit"),
        ("nh4_ccpvdz", "cc-pvdz-jkfit"),
        ("h2o_distorted_ccpvdz", "cc-pvdz-jkfit"),
        ("h2o_ccpvtz", "cc-pvtz-jkfit"),
        ("h2o_ccpvdz_accd", "cc-pvdz-accd"),
    }
    assert set(section("rhf")) == {
        "h2o_ccpvdz",
        "h2o_distorted_ccpvdz",
        "h2o_ccpvtz",
    }
    assert all(c["aux_basis"] is None for c in doc()["rhf"])
    assert set(section("optimized")) == {"h2o_ccpvdz", "ethanol_ccpvdz"}


@pytest.mark.parametrize(
    "where", ["integrals", "density_fitting", "df_rhf", "rhf", "optimized"]
)
def test_gradients_are_translation_invariant(where: str):
    for case in doc()[where]:
        natm = len(case["atoms"])
        for what, g in gradients(case):
            assert len(g) == natm and all(len(row) == 3 for row in g), what
            # the two halves of V and of d(mn|P): only their totals move
            # rigidly with the molecule
            if what.split(".")[-1] in ("functions", "operator", "orbital", "aux"):
                continue
            scale = max(1.0, max(abs(x) for row in g for x in row))
            for k in range(3):
                assert abs(sum(row[k] for row in g)) < 1e-9 * scale, (
                    case["name"],
                    what,
                )


def test_term_sets_add_up_to_their_totals():
    for case in doc()["integrals"]:
        p = case["potential"]
        assert_close(add(p["functions"], p["operator"]), p["total"], 1e-12, "V")
    for case in doc()["density_fitting"]:
        t = case["three_center"]
        assert_close(add(t["orbital"], t["aux"]), t["total"], 1e-12, "3c")
    for where in ("df_rhf", "rhf"):
        for case in doc()[where]:
            assert set(case["terms"]) == {
                "kinetic",
                "potential",
                "overlap",
                "coulomb",
                "exchange",
                "nuclear_repulsion",
            }
            summed = add(*case["terms"].values())
            assert_close(summed, case["total"], 1e-9, case["name"])


def test_distorted_water_is_far_from_a_minimum():
    for where in ("df_rhf", "rhf"):
        g = section(where)["h2o_distorted_ccpvdz"]["total"]
        assert min(abs(x) for row in g for x in row) > 1e-3


def test_optimized_geometries_are_converged():
    for case in doc()["optimized"]:
        assert case["max_force"] < 1e-6
        assert case["aux_basis"] == "cc-pvdz-jkfit"
        assert case["optimizer"]
        assert len(case["start"]) == len(case["atoms"])


def test_cheapest_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import gradient_reference

    (case,) = [
        c for c in gradient_reference.INTEGRAL_CASES if c["name"] == "h2o_ccpvdz"
    ]
    got = gradient_reference.integral_case(case)
    want = section("integrals")["h2o_ccpvdz"]
    for (what, g), (_, w) in zip(gradients(got), gradients(want), strict=True):
        assert_close(g, w, 1e-10, what)
