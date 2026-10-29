"""The committed ROHF references, scf/rohf.json and density_fitting/df_rohf.json.

Their shape is checked with the standard library alone, so it runs on the push
gate: every case's electron count, spin and orbital count agree, the issue's
kinds of case are all there (doublets, a triplet, a cation, f shells), and the
DF file carries the same geometries as the exact one. Re-running the cheapest
case through tools/emit/scf_reference.py needs PySCF, which lives in the
opt-in ``tools`` dependency group, so that test skips (with a reason) wherever
the group is not installed::

    uv run --group tools pytest -rs test/pytest/test_rohf_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test" / "gtest" / "data"
ROHF = DATA / "scf" / "rohf.json"
DF_ROHF = DATA / "density_fitting" / "df_rohf.json"


def cases(path: Path, key: str) -> dict[str, dict]:
    return {case["name"]: case for case in json.loads(path.read_text())[key]}


@pytest.mark.parametrize(("path", "key"), [(ROHF, "cases"), (DF_ROHF, "df_rohf")])
def test_cases_are_consistent(path: Path, key: str):
    for name, case in cases(path, key).items():
        assert case["n_alpha"] + case["n_beta"] == case["electrons"], name
        assert case["n_alpha"] - case["n_beta"] == case["spin"] > 0, name
        assert case["multiplicity"] == case["spin"] + 1, name
        protons = sum(atom["z"] for atom in case["atoms"])
        assert protons - case["charge"] == case["electrons"], name
        assert len(case["mo_energies"]) == case["nao"], name
        assert case["mo_energies"] == sorted(case["mo_energies"]), name


def test_the_issue_cases_are_covered():
    exact = cases(ROHF, "cases").values()
    assert any(case["multiplicity"] == 2 for case in exact)
    assert any(case["multiplicity"] == 3 for case in exact)
    assert any(case["charge"] > 0 for case in exact)
    assert any(case["basis"] == "cc-pvtz" for case in exact)  # f shells


def test_df_cases_match_the_exact_ones():
    exact = cases(ROHF, "cases")
    fitted = cases(DF_ROHF, "df_rohf")
    # Every case with a JKFIT pairing (all but STO-3G).
    assert set(fitted) == {n for n, c in exact.items() if c["basis"] != "sto-3g"}
    for name, case in fitted.items():
        for field in ("basis", "charge", "spin", "atoms", "nao", "n_alpha", "n_beta"):
            assert case[field] == exact[name][field], (name, field)
        assert case["aux_basis"] == f"{case['basis']}-jkfit", name
        # Fitting error, not a different state.
        assert abs(case["energy"] - exact[name]["energy"]) < 1e-4, name


def test_cheapest_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT))
    from tools.emit import scf_reference

    (case,) = [c for c in scf_reference.ROHF_CASES if c["name"] == "oh_sto3g"]
    got = scf_reference.rohf(case)
    want = cases(ROHF, "cases")["oh_sto3g"]
    assert abs(got["energy"] - want["energy"]) < 1e-10
    for g, w in zip(got["mo_energies"], want["mo_energies"], strict=True):
        assert abs(g - w) < 1e-8
