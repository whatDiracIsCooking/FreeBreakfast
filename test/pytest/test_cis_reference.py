"""The committed CIS references, test/gtest/data/cis/.

Their shape is checked with the standard library alone, so it runs on the push
gate: the issue's cases are all there, each root list ends at a whole
degenerate group, energies ascend, singlet groups carry a summed oscillator
strength, and the dipole files exist with the shape the manifest implies.
Re-running the cheapest case through tools/emit/cis_reference.py needs PySCF,
which lives in the opt-in ``tools`` dependency group, so that test skips (with
a reason) wherever the group is not installed::

    uv run --group tools pytest -rs test/pytest/test_cis_reference.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test" / "gtest" / "data" / "cis"
MANIFEST = DATA / "cis.json"


def cases() -> dict[str, dict]:
    return {case["name"]: case for case in json.loads(MANIFEST.read_text())["cases"]}


def test_the_issue_cases_are_covered():
    got = cases()
    for name in ("h2o_sto3g", "h2o_631g", "h2o_ccpvdz"):
        assert got[name]["aux_basis"] is None
        assert len(got[name]["singlets"]["energies"]) >= 5
        assert len(got[name]["triplets"]["energies"]) >= 5
    assert got["h2o_ccpvdz_df"]["aux_basis"] == "cc-pvdz-jkfit"
    assert "oscillator_strengths" not in got["hf_ccpvtz"]["singlets"]
    assert got["benzene_ccpvdz_df"]["aux_basis"] == "cc-pvdz-jkfit"
    for name in ("h2o_sto3g", "h2o_ccpvdz"):
        assert "int1e_r" in got[name]


@pytest.mark.parametrize("name", sorted(cases()))
def test_roots_are_whole_groups(name: str):
    case = cases()[name]
    protons = sum(atom["z"] for atom in case["atoms"])
    assert protons == case["electrons"] == 2 * case["occupied"]
    for spin in ("singlets", "triplets"):
        if spin not in case:
            continue
        roots = case[spin]
        energies = roots["energies"]
        assert energies == sorted(energies) and energies[0] > 0
        flat = [k for group in roots["groups"] for k in group]
        assert flat == list(range(len(energies)))
        for group in roots["groups"]:
            assert energies[group[-1]] - energies[group[0]] < 1e-6
        assert roots["dense_max_error"] < 1e-8
        if "oscillator_strengths" in roots:
            assert spin == "singlets"
            assert len(roots["oscillator_strengths"]) == len(roots["groups"])
            assert all(f >= 0 for f in roots["oscillator_strengths"])


def test_degenerate_groups_are_recorded():
    # benzene's E states and HF's Pi are the point of the grouping
    for name in ("benzene_ccpvdz_df", "benzene_ccpvdz", "hf_ccpvtz"):
        groups = cases()[name]["singlets"]["groups"]
        assert any(len(group) == 2 for group in groups), name


def test_dipole_files_have_the_manifest_shape():
    for case in cases().values():
        if "int1e_r" not in case:
            continue
        with (DATA / case["int1e_r"]).open("rb") as f:
            assert f.read(6) == b"\x93NUMPY"
            major = f.read(2)[0]
            size = int.from_bytes(f.read(2 if major == 1 else 4), "little")
            header = ast.literal_eval(f.read(size).decode("latin1"))
        assert header["descr"] == "<f8"
        assert header["fortran_order"] is True
        assert header["shape"] == (3, case["nao"], case["nao"])


def test_cheapest_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import cis_reference

    (case,) = [c for c in cis_reference.CASES if c["name"] == "h2o_sto3g"]
    got = cis_reference.cis_case(case, write=False)
    want = cases()["h2o_sto3g"]
    assert abs(got["rhf_energy"] - want["rhf_energy"]) < 1e-10
    for spin in ("singlets", "triplets"):
        assert got[spin]["groups"] == want[spin]["groups"]
        for g, w in zip(got[spin]["energies"], want[spin]["energies"], strict=True):
            assert abs(g - w) < 1e-9
    for g, w in zip(
        got["singlets"]["oscillator_strengths"],
        want["singlets"]["oscillator_strengths"],
        strict=True,
    ):
        assert abs(g - w) < 1e-8
