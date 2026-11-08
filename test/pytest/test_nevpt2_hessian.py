"""tools/nevpt2_hessian: the stencil helpers, the committed SC- and
PC-NEVPT2 Hessian references in test/gtest/data/nevpt2_hessian/ (#803), and
the SC- and PC-QD-NEVPT2 ones in test/gtest/data/qd_nevpt2_hessian/ (#953).

The helpers (tools/nevpt2_hessian/stencil.py) and the fixtures' invariants
are checked with the standard library alone, so they run on the push gate:
the stencil's order and its ratio of 16, the ladder's verdicts, the symmetry
and bilinearity checks; then every fixture's ladder recomputed from its
recorded gradients, its bars and ratios against the rule, the skew and
near-degenerate cases (and HCN, whose PC metric has null modes by symmetry)
symmetric to their bar and bilinear on every pair, and the symmetric case's
failing pairs. Re-running the reference point through
the generator needs PySCF and block2, in the opt-in ``tools`` dependency
group, so that test is slow and skips (with a reason) wherever the group is
not installed::

    uv run --group tools pytest -m slow -rs test/pytest/test_nevpt2_hessian.py
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "nevpt2_hessian"
DATA = ROOT / "test" / "gtest" / "data" / "nevpt2_hessian"
CASES = {
    "h2co_skew_sto3g": "skew",
    "nh3_cs_sto3g": "symmetric",
    "h2co_neardeg_sto3g": "near_degenerate",
    "hcn_sto3g": "symmetric_nulls",
}


def load_stencil():
    spec = importlib.util.spec_from_file_location(
        "nevpt2_hessian_stencil", TOOL / "stencil.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stencil = load_stencil()


def close(a: float, b: float) -> bool:
    # Recomputed from fixture inputs, not read back: the stencil's sum()
    # reorders (3.12 compensates float sums), so it may move the last ulp of
    # each gradient, which the 1/s of a step leaves near 1e-16 / s absolute
    # (1e-12 is far below any bar, 2.6e-8 at least).
    return math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-12)


@functools.cache
def fixture(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


# ------------------------------------------------------------------ helpers


def test_the_stencil_is_exact_through_quartics():
    # Fourth order: exact on every polynomial of degree four or less.
    for s in (0.3, 0.05):
        for coefficients in ((1.0,), (0.0, 2.0), (0.5, -1.0, 3.0, -2.0, 0.7)):
            values = {
                o * s: sum(c * (o * s) ** k for k, c in enumerate(coefficients))
                for o in stencil.OFFSETS
            }
            want = coefficients[1] if len(coefficients) > 1 else 0.0
            assert stencil.derivative(values, s) == pytest.approx(want, abs=1e-12)


def test_the_ladder_ratio_is_sixteen_on_a_smooth_function():
    h = 0.08
    values = {
        d: math.exp(0.7 * d) + math.sin(1.3 * d) for d in stencil.displacements(h)
    }
    coarse, middle, fine = (stencil.derivative(values, s) for s in stencil.steps(h))
    ratio = (coarse - middle) / (middle - fine)
    assert abs(ratio - stencil.TARGET) < 0.1
    assert stencil.TARGET == 16.0


def test_the_ladder_reads_eight_points():
    h = 0.04
    assert stencil.steps(h) == [0.04, 0.02, 0.01]
    points = stencil.displacements(h)
    assert len(points) == 8
    assert points == sorted(points)
    assert points == [-x for x in reversed(points)]
    assert max(points) == 2 * h and min(abs(x) for x in points) == h / 4


def test_noise_gain():
    norm = math.sqrt(1 / 144 + 64 / 144 + 64 / 144 + 1 / 144)
    assert stencil.noise_gain(0.5) == pytest.approx(norm / 0.5)


def test_column_ladder_differentiates_each_component():
    h = 0.1
    points = {d: [d**2, 3.0 * d, math.sin(d)] for d in stencil.displacements(h)}
    for column in stencil.column_ladder(points, h):
        assert column[0] == pytest.approx(0.0, abs=1e-12)
        assert column[1] == pytest.approx(3.0, abs=1e-12)
        assert column[2] == pytest.approx(1.0, abs=1e-5)


def test_judge():
    window = (12.8, 19.2)
    settled = stencil.judge([1.0 + 17e-6, 1.0 + 1e-6, 1.0], 1e-8, window)
    assert settled["settled"] is True and settled["bar"] == pytest.approx(1e-6)
    assert settled["ratio"] == pytest.approx(16.0)
    # Not fourth order: a ratio of 4 fails.
    assert (
        stencil.judge([1.0 + 5e-6, 1.0 + 1e-6, 1.0], 1e-8, window)["settled"] is False
    )
    # Below the floor: not judged, the bar is the floor.
    quiet = stencil.judge([1.0 + 1e-9, 1.0 + 2e-10, 1.0], 1e-8, window)
    assert quiet["ratio"] is None and quiet["settled"] is None
    assert quiet["bar"] == 1e-8
    # Below the floor at the fine rung but far above it at the coarse one:
    # a kink the fine rung steps over.
    kink = stencil.judge([1.0 + 1e-6, 1.0 + 2e-10, 1.0], 1e-8, window)
    assert kink["ratio"] is None and kink["settled"] is False


def test_symmetry_helpers():
    m = [[1.0, 2.0], [2.0 + 3e-8, 5.0]]
    bar = [[1e-8, 1e-8], [5e-8, 1e-8]]
    assert stencil.symmetry_misses(m, bar) == []
    assert stencil.symmetry_misses(m, [[1e-8] * 2] * 2) == [
        (0, 1, pytest.approx(3e-8), 1e-8)
    ]
    assert stencil.symmetrized(m)[0][1] == stencil.symmetrized(m)[1][0]
    assert stencil.symmetric_bar(bar) == [[1e-8, 5e-8], [5e-8, 1e-8]]
    assert stencil.asymmetry(m)[1][0] == pytest.approx(3e-8)


def test_bilinearity_passes_a_quadratic_form_and_fails_a_ratio_of_them():
    # E = x^T A x / 2: its second derivative along the unit
    # (e_0 + e_1) / sqrt 2 is (A00 + A11 + 2 A01) / 2.
    a = [[2.0, 0.5], [0.5, 1.0]]
    bar = [[1e-9] * 2] * 2
    along = 0.5 * (a[0][0] + a[1][1] + 2 * a[0][1])
    assert not stencil.bilinearity(a, bar, 0, 1, along, 1e-9)["fails"]

    # E = t^2 Q(q) with Q a ratio of quadratic forms (an empty perturber's
    # T / (Delta + H / N)): each direction is quadratic, but not bilinear.
    def q(x, y):
        return (x * x + y * y) * (x * x + 2 * y * y) / (x * x + 4 * y * y)

    m = [[2 * q(1, 0), 0.0], [0.0, 2 * q(0, 1)]]
    # Along the unit (1, 1) / sqrt 2: 2 Q(1, 1) / 2, Q being homogeneous.
    check = stencil.bilinearity(m, bar, 0, 1, q(1, 1), 1e-9)
    assert check["fails"] and check["pair"] == [0, 1]


# ----------------------------------------------------------------- fixtures


def test_the_issue_cases_are_there():
    assert sorted(p.stem for p in DATA.glob("*.json")) == sorted(CASES)
    for name, kind in CASES.items():
        record = fixture(name)
        assert record["name"] == name and record["kind"] == kind
        assert record["weights"] == [1.0] and record["frozen_core"] == 0
        assert 2 * record["ncore"] + record["nelecas"] == record["electrons"]
        assert len(record["active"]) == record["ncas"]
        for key in ("basis", "aux_basis"):
            path = ROOT / record[key]["file"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record[key]["sha256"]
            )
        assert record["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
        assert record["provenance"]["threads"] == 1
    # The near-degenerate case's pair: its smallest inactive gap is the
    # smallest of the three.
    gaps = {name: min(fixture(name)["inactive_gaps"].values()) for name in CASES}
    assert min(gaps, key=gaps.get) == "h2co_neardeg_sto3g"
    assert gaps["h2co_neardeg_sto3g"] < 0.01
    # The symmetric case has empty perturbers.
    symmetric = fixture("nh3_cs_sto3g")
    assert sum(c["sc_empty_perturbers"] for c in symmetric["classes"]) > 0
    # HCN's PC drops Sir's two null modes by symmetry (#815), and nothing
    # is expected to fail there.
    nulls = fixture("hcn_sto3g")
    sir = next(c for c in nulls["classes"] if c["name"] == "Sir")
    assert sir["pc_dropped_modes"] == 2
    assert nulls["may_fail"] == []


def gated(record: dict, q: str, i: int, j: int) -> bool:
    """The generator's rule: a quantity the case expects to fail is held
    only on the totally symmetric block."""
    irreps = record["coordinates"]["irreps"]
    total = record["coordinates"]["irreps_order"][0]
    return q not in record["may_fail"] or irreps[i] == irreps[j] == total


def schemes(record: dict) -> list[str]:
    return [q for q in ("casscf", "sc", "pc") if q in record]


def test_every_record_holds_what_the_issue_asks():
    for name in CASES:
        record = fixture(name)
        n = 3 * len(record["atoms"])
        assert "pc" in schemes(record) and "casscf" in schemes(record)
        for q in schemes(record):
            part = record[q]
            assert len(part["gradient"]) == len(record["atoms"])
            for key in ("hessian", "bar", "ratios", "column_bar", "asymmetry"):
                assert len(part[key]) == n and all(len(r) == n for r in part[key]), key
            assert len(part["ladder"]) == 3
            assert part["noise"]["floor"] > 0
            assert q == "casscf" or q in record["energies"]
    assert "sc" in fixture("h2co_skew_sto3g") and "sc" in fixture("nh3_cs_sto3g")


def test_the_ladders_are_the_recorded_gradients():
    for name in CASES:
        record = fixture(name)
        h = record["steps"]["h"]
        assert record["steps"]["ladder"] == stencil.steps(h)
        assert record["steps"]["displacements"] == stencil.displacements(h)
        n = 3 * len(record["atoms"])
        for q in schemes(record):
            part = record[q]
            for entry in part["points"]:
                j = entry["column"]
                points = dict(
                    zip(
                        record["steps"]["displacements"],
                        entry["gradients"],
                        strict=True,
                    )
                )
                ladder = stencil.column_ladder(points, h)
                for k in range(3):
                    for i in range(n):
                        assert close(ladder[k][i], part["ladder"][k][i][j]), (
                            name,
                            q,
                            k,
                            i,
                            j,
                        )
            finest = part["ladder"][-1]
            assert part["hessian"] == stencil.symmetrized(finest)
            assert part["asymmetry"] == stencil.asymmetry(finest)


def test_bars_and_ratios_follow_the_rule():
    for name in CASES:
        record = fixture(name)
        window = record["tolerances"]["ratio_window"]
        assert window == [0.8 * 16, 1.2 * 16]
        for q in schemes(record):
            part = record[q]
            floor = part["noise"]["floor"]
            n = len(part["hessian"])
            judged = []
            for i in range(n):
                for j in range(n):
                    verdict = stencil.judge(
                        [part["ladder"][k][i][j] for k in range(3)], floor, window
                    )
                    assert verdict["bar"] == part["column_bar"][i][j]
                    assert verdict["ratio"] == part["ratios"][i][j]
                    if not gated(record, q, i, j):
                        continue
                    assert verdict["settled"] is not False, (name, q, i, j)
                    if verdict["ratio"] is not None:
                        judged.append(verdict["ratio"])
            assert part["bar"] == stencil.symmetric_bar(part["column_bar"])
            # Every case shows its ladder in the asymptotic range somewhere.
            assert judged, (name, q)
            assert all(window[0] < r < window[1] for r in judged)


def test_every_held_block_is_symmetric_and_bilinear():
    for name in CASES:
        record = fixture(name)
        for q in schemes(record):
            part = record[q]
            n = len(part["hessian"])
            assert len(part["bilinearity"]) == n * (n - 1) // 2
            failing = [p["pair"] for p in part["bilinearity"] if p["fails"]]
            assert failing == part["failing_pairs"]
            misses = stencil.symmetry_misses(part["ladder"][-1], part["column_bar"])
            for i, j, _, _ in misses:
                assert not gated(record, q, i, j), (name, q, i, j)
            for i, j in failing:
                assert not gated(record, q, i, j), (name, q, i, j)
            for p in part["bilinearity"]:
                i, j = p["pair"]
                if gated(record, q, i, j):
                    assert p["settled"] is not False, (name, q, i, j)
            if q not in record["may_fail"]:
                assert part["failed_blocks"] == [], (name, q)


def test_the_symmetric_case_records_its_failing_block():
    record = fixture("nh3_cs_sto3g")
    assert record["may_fail"] == ["sc"]
    irreps = record["coordinates"]["irreps"]
    assert record["coordinates"]["irreps_order"] == ["A'", "A''"]
    assert irreps.count("A'") == 7 and irreps.count("A''") == 5
    # SC's non-totally-symmetric block has no Hessian: recorded, and never
    # in the totally symmetric block.
    blocks = record["sc"]["failed_blocks"]
    assert blocks and ["A'", "A'"] not in blocks
    failing = record["sc"]["failing_pairs"]
    assert failing
    for i, j in failing:
        assert (irreps[i], irreps[j]) != ("A'", "A'")
    # PC is defined on every block there.
    assert record["pc"]["failing_pairs"] == [] and record["pc"]["failed_blocks"] == []
    h = record["steps"]["h"]
    for p in record["sc"]["bilinearity"]:
        derivatives = {
            d: [x]
            for d, x in zip(
                record["steps"]["displacements"], p["derivatives"], strict=True
            )
        }
        recomputed = [v[0] for v in stencil.column_ladder(derivatives, h)]
        assert len(recomputed) == len(p["ladder"])
        assert all(map(close, recomputed, p["ladder"])), p["pair"]


def test_the_guard_and_the_checks():
    for name in CASES:
        record = fixture(name)
        tol, guard = record["tolerances"], record["guard"]
        assert guard["prediction"] < tol["prediction_bound"]
        assert guard["subspace"] < tol["subspace_bound"]
        assert guard["ci"] < tol["ci_bound"]
        assert guard["inactive_gap"] > tol["degenerate"]
        checks = record["checks"]
        assert checks["sc_invariance"] < 1e-10
        assert checks["block2_sc"] < 1e-10
        assert checks["block2_pc"] < tol["block2_pc"]
        assert checks["casscf_gradient_vs_analytic"] < tol["analytic"]
        e = record["energies"]
        assert abs(e["sc"] - e["casscf"] - record["e_corr"]["sc"]) < 1e-12
        assert abs(e["pc"] - e["casscf"] - record["e_corr"]["pc"]) < 1e-12
        assert record["polish_grad"] < tol["polish_grad"]


@pytest.mark.slow
def test_the_reference_point_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("pyblock2", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(TOOL))
    import nevpt2_hessian as tool

    (spec,) = [c for c in tool.CASES if c["name"] == "h2co_skew_sto3g"]
    want = fixture("h2co_skew_sto3g")
    ref = tool.Reference(spec)
    assert abs(ref.energy - want["energies"]["casscf"]) < 1e-11
    assert abs(ref.nevpt2["sc"] - want["e_corr"]["sc"]) < 1e-11
    assert abs(ref.nevpt2["pc"] - want["e_corr"]["pc"]) < 1e-11


# ------------------------------------------------- the QD references (#953)

QD_DATA = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_hessian"
QD_CASES = ("h2co_skew_ccpvdz_sa3", "h2co_skew_ccpvdz_sa3_weighted")
# The C2v A1/B2 coupling cases (#1058): C_MN's ladders, and their analysis.
COUPLING_CASES = ("h2nb_sto3g_sa2",)


def load_qd_tool():
    spec = importlib.util.spec_from_file_location(
        "qd_nevpt2_hessian", TOOL / "qd_nevpt2_hessian.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qd_tool = load_qd_tool()


@functools.cache
def qd_fixture(name: str) -> dict:
    return json.loads((QD_DATA / f"{name}.json").read_text())


def qd_quantities(record: dict):
    """Every differenced G(W) of a QD record: (scheme, part)."""
    for scheme in record["schemes"]:
        for part in record[scheme]["states"] + record[scheme]["pairs"]:
            yield scheme, part


def test_the_qd_cases_are_there():
    assert sorted(p.stem for p in QD_DATA.glob("*.json")) == sorted(
        list(QD_CASES)
        + list(COUPLING_CASES)
        + [f"{name}.analysis" for name in COUPLING_CASES]
    )
    assert [c["name"] for c in qd_tool.CASES] == list(QD_CASES)
    for name in QD_CASES:
        record = qd_fixture(name)
        assert record["name"] == name
        for key in ("basis", "aux_basis"):
            path = ROOT / record[key]["file"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record[key]["sha256"]
            )
        n = len(record["weights"])
        assert n == 3
        for scheme in record["schemes"]:
            assert [p["state"] for p in record[scheme]["states"]] == list(range(n))
            assert [p["pair"] for p in record[scheme]["pairs"]] == record["pairs"]
    # Equal and unequal weights: theta in and out of the solve.
    weights = {tuple(qd_fixture(name)["weights"]) for name in QD_CASES}
    assert len(weights) == 2


def test_the_qd_ladders_are_the_recorded_gradients():
    for name in QD_CASES:
        record = qd_fixture(name)
        n = 3 * len(record["atoms"])
        assert record["coordinates"]["vectors"] == [
            [1.0 if i == j else 0.0 for i in range(n)] for j in range(n)
        ]
        for scheme, part in qd_quantities(record):
            steps = record["steps"][scheme]
            h = steps["h"]
            assert steps["ladder"] == stencil.steps(h)
            assert steps["displacements"] == stencil.displacements(h)
            assert len(part["points"]) == n
            for entry in part["points"]:
                j = entry["column"]
                points = dict(
                    zip(steps["displacements"], entry["gradients"], strict=True)
                )
                ladder = stencil.column_ladder(points, h)
                for k in range(3):
                    for i in range(n):
                        assert close(ladder[k][i], part["ladder"][k][i][j]), (
                            name,
                            scheme,
                            k,
                            i,
                            j,
                        )
            finest = part["ladder"][-1]
            assert part["hessian"] == stencil.symmetrized(finest)
            assert part["asymmetry"] == stencil.asymmetry(finest)


def test_the_qd_floor_is_the_measured_noise():
    for name in QD_CASES:
        record = qd_fixture(name)
        for scheme, part in qd_quantities(record):
            noise = part["noise"]
            d = noise["differences"]
            sigma = math.sqrt(
                math.fsum(x * x for row in d for x in row) / (len(d) * len(d[0]))
            )
            assert math.isclose(sigma, noise["sigma_g"], rel_tol=1e-12)
            assert len(d) == len(noise["pairs"]) > 0
            h = record["steps"][scheme]["h"]
            assert math.isclose(
                noise["floor"], qd_tool.floor_of(sigma, h), rel_tol=1e-12
            )


def test_the_qd_bars_and_ratios_follow_the_rule():
    for name in QD_CASES:
        record = qd_fixture(name)
        window = record["tolerances"]["ratio_window"]
        assert window == [0.8 * 16, 1.2 * 16]
        for scheme, part in qd_quantities(record):
            floor = part["noise"]["floor"]
            n = len(part["hessian"])
            judged = []
            for i in range(n):
                for j in range(n):
                    verdict = stencil.judge(
                        [part["ladder"][k][i][j] for k in range(3)], floor, window
                    )
                    assert verdict["bar"] == part["column_bar"][i][j]
                    assert verdict["ratio"] == part["ratios"][i][j]
                    assert verdict["settled"] is not False, (name, scheme, i, j)
                    if verdict["ratio"] is not None:
                        judged.append(verdict["ratio"])
            assert part["bar"] == stencil.symmetric_bar(part["column_bar"])
            assert part["ratios_judged"] == len(judged)
            assert all(window[0] < r < window[1] for r in judged)


def test_every_qd_hessian_is_symmetric_to_its_bar():
    for name in QD_CASES:
        record = qd_fixture(name)
        for scheme, part in qd_quantities(record):
            assert (
                stencil.symmetry_misses(part["ladder"][-1], part["column_bar"]) == []
            ), (name, scheme)


def test_each_pair_sum_is_its_states_sum():
    # G is linear in W, so the pair's Hessian is its two states' sum, to the
    # three bars (each point's Z-vectors are solved per W).
    for name in QD_CASES:
        record = qd_fixture(name)
        for scheme in record["schemes"]:
            states = record[scheme]["states"]
            for part in record[scheme]["pairs"]:
                p, q = (states[k] for k in part["pair"])
                n = len(part["hessian"])
                for i in range(n):
                    for j in range(n):
                        miss = abs(
                            part["hessian"][i][j]
                            - p["hessian"][i][j]
                            - q["hessian"][i][j]
                        )
                        allowed = part["bar"][i][j] + p["bar"][i][j] + q["bar"][i][j]
                        assert miss <= allowed, (name, scheme, part["pair"], i, j)


def test_the_qd_guard_and_the_gaps():
    for name in QD_CASES:
        record = qd_fixture(name)
        g = record["guard"]
        assert g["prediction"] < 2.0
        assert g["subspace"] < 1e-3 and g["ci"] < 1e-3
        assert g["gradient_norm"] < 1e-10 and g["rerun_gradient_norm"] < 1e-10
        assert g["rerun_energy"] < 1e-9
        for scheme in record["schemes"]:
            assert g[f"{scheme}_vectors"] < 1e-3
            assert g[f"{scheme}_zvector_residual"] <= record["tolerances"]["zvector"]
            # Every state's Hessian is defined: H_eff's gaps stay open.
            assert min(record["reference"][scheme]["heff_gaps"]) > 0.1
        assert min(record["reference"]["casscf_gaps"]) > 0.1
        # The reference point is qd_nevpt2/'s: another CASSCF solution misses
        # by millihartree.
        assert max(record["reference"]["agreement"].values()) < 1e-10


# ------------------------------------------- the C2v A1/B2 coupling cases


@functools.cache
def coupling(name: str) -> tuple[dict, dict, dict]:
    """The record, its analysis and the tool's case."""
    (case,) = [c for c in qd_tool.COUPLING_CASES if c["name"] == name]
    record = json.loads((QD_DATA / f"{name}.json").read_text())
    analysis = json.loads((QD_DATA / f"{name}.analysis.json").read_text())
    return record, analysis, case


def test_the_coupling_cases_are_what_the_issue_asks():
    assert [c["name"] for c in qd_tool.COUPLING_CASES] == list(COUPLING_CASES)
    for name in COUPLING_CASES:
        record, analysis, case = coupling(name)
        assert record["name"] == analysis["name"] == name
        for key in ("basis", "aux_basis"):
            path = ROOT / record[key]["file"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record[key]["sha256"]
            )
        for key in ("atoms", "ncas", "nelecas", "active", "weights"):
            assert record[key] == case["system"][key]
        # A C2v SA2 whose roots are A1 and B2, laddered along two b2
        # coordinates and their sum.
        assert record["irreps"] == ["A1", "B2"]
        assert len(record["weights"]) == 2
        assert record["state_pairs"] == [[0, 1]]
        coords = record["coordinates"]
        assert coords == analysis["coordinates"]
        assert coords == qd_tool.symmetry_coordinates(record["atoms"])
        irreps = dict(zip(coords["labels"], coords["irreps"], strict=True))
        assert [irreps[label] for label in record["coupled"]] == ["B2", "B2"]
        assert not any(label.startswith(("T_", "R_")) for label in record["coupled"])
        vectors = coords["vectors"]
        for i, a in enumerate(vectors):
            for j, b in enumerate(vectors):
                assert math.isclose(
                    qd_tool.dot(a, b), 1.0 if i == j else 0.0, abs_tol=1e-12
                )


def coupling_parts(record: dict):
    for scheme in record["schemes"]:
        for entry in record[scheme]:
            yield scheme, entry


def test_the_coupling_ladders_are_the_recorded_values():
    for name in COUPLING_CASES:
        record, _, case = coupling(name)
        directions = qd_tool.coupling_directions(case, record["coordinates"])
        for scheme, entry in coupling_parts(record):
            steps = record["steps"][scheme]
            h = steps["h"]
            assert steps["ladder"] == stencil.steps(h)
            assert steps["displacements"] == stencil.displacements(h)
            assert [line["direction"] for line in entry["directions"]] == [
                d for d, _ in directions
            ]
            for line, (_, q) in zip(entry["directions"], directions, strict=True):
                assert line["vector"] == q
                values = dict(zip(steps["displacements"], line["values"], strict=True))
                for k, s in enumerate(stencil.steps(h)):
                    assert close(stencil.derivative(values, s), line["ladder"][k])
                # C_MN is odd under the b2 coordinate (C2 maps R0 + s q to
                # R0 - s q and flips the B2 root's sign).
                for d in steps["displacements"]:
                    assert math.isclose(values[d], -values[-d], abs_tol=1e-12)


def test_the_coupling_floor_is_the_measured_noise():
    for name in COUPLING_CASES:
        record, _, _ = coupling(name)
        for scheme, entry in coupling_parts(record):
            noise = entry["noise"]
            d = noise["differences"]
            assert len(d) == len(noise["pairs"]) > 0
            sigma = math.sqrt(math.fsum(x * x for x in d) / len(d))
            assert math.isclose(sigma, noise["sigma"], rel_tol=1e-12)
            h = record["steps"][scheme]["h"]
            assert math.isclose(
                noise["floor"], qd_tool.floor_of(sigma, h), rel_tol=1e-12
            )


def test_every_coupling_ladder_settles_and_the_bilinearity_is_recorded():
    for name in COUPLING_CASES:
        record, _, _ = coupling(name)
        window = record["tolerances"]["ratio_window"]
        assert window == [0.8 * 16, 1.2 * 16]
        failing = []
        for scheme, entry in coupling_parts(record):
            floor = entry["noise"]["floor"]
            for line in entry["directions"]:
                verdict = stencil.judge(line["ladder"], floor, window)
                assert verdict["bar"] == line["bar"]
                assert verdict["ratio"] == line["ratio"]
                assert verdict["settled"] == line["settled"]
                assert verdict["settled"] is not False, (name, line["direction"])
            by_name = {line["direction"]: line for line in entry["directions"]}
            assert [c["coordinates"] for c in entry["bilinearity"]] == [
                record["coupled"]
            ]
            for check in entry["bilinearity"]:
                a, b = check["coordinates"]
                la, lb = by_name[a], by_name[b]
                lu = by_name[f"({a} + {b})/sqrt2"]
                residuals = [
                    lu["ladder"][k]
                    - (la["ladder"][k] + lb["ladder"][k]) / math.sqrt(2.0)
                    for k in range(3)
                ]
                assert all(
                    close(x, y)
                    for x, y in zip(residuals, check["residuals"], strict=True)
                )
                allowed = lu["bar"] + (la["bar"] + lb["bar"]) / math.sqrt(2.0)
                assert math.isclose(check["allowed"], allowed, rel_tol=1e-12)
                fails = abs(residuals[-1]) > allowed
                assert check["fails"] == fails
                if fails and scheme == "sc":
                    failing.append({"states": entry["pair"], "coordinates": [a, b]})
        # SC's expected refusals are exactly where its ladder is not bilinear.
        assert record["expected_refusals"]["sc"] == failing


def test_pc_expected_refusals_are_the_analysis_lifts():
    for name in COUPLING_CASES:
        record, analysis, _ = coupling(name)
        tol = analysis["tolerances"]
        pc = analysis["pc"]
        lifted = set()
        for e in pc["lifts"]:
            assert e["steps"] == analysis["steps"]
            assert e["second_order"] == qd_close_rates(e["lift"], tol["lift_agree"])
            assert e["bra_overlap_nonzero"] == (
                qd_close_rates(e["bra_overlap"], tol["overlap_agree"])
                and min(e["bra_overlap"]) > tol["overlap_floor"]
            )
            assert e["ket_overlap_first_order"] == qd_close_rates(
                e["ket_overlap_rate"], tol["overlap_agree"]
            )
            if (
                e["second_order"]
                and e["bra_overlap_nonzero"]
                and e["ket_overlap_first_order"]
            ):
                lifted.add(e["coordinate"])
            # A lifted mode had been cut at R0.
            (count,) = [
                d["count"]
                for d in pc["dropped"]
                if d["ket"] == e["ket"] and d["class"] == e["class"]
            ]
            assert 0 < max(e["modes"]) <= count
        b2 = [
            label
            for label, irrep in zip(
                analysis["coordinates"]["labels"],
                analysis["coordinates"]["irreps"],
                strict=True,
            )
            if irrep == "B2"
        ]
        assert pc["lifted_coordinates"] == [label for label in b2 if label in lifted]
        # Nothing lifts along the translation and the rotation.
        assert not lifted & {"T_y", "R_x"}
        assert record["expected_refusals"]["pc"] == qd_tool.pc_expected(analysis)
        # PC's computed coupling cannot be laddered there: the sampler refuses
        # the first displaced point as a rank change of the cut.
        probe = record["pc_probe"]
        assert probe["reference"]["returncode"] == 0
        assert probe["displaced"]["returncode"] != 0
        assert "rank change" in probe["displaced"]["message"]
        assert "pc" not in record["schemes"]


def qd_close_rates(rates: list[float], tol: float) -> bool:
    a, b = rates
    return abs(a - b) <= tol * max(abs(a), abs(b))


def test_the_one_sided_tuples():
    for name in COUPLING_CASES:
        record, analysis, case = coupling(name)
        tol = analysis["tolerances"]
        sc = analysis["sc"]
        assert sc["one_sided"], name
        for item in sc["one_sided"]:
            # Empty for its ket (the guard's), populated for the other root.
            assert abs(item["norm_ket"]) <= tol["guard"] < tol["populated"]
            assert abs(item["norm_bra"]) > tol["populated"]
            assert item["one_function"] == (
                (item["class"] == "Srs" and case["system"]["nelecas"] == 2)
                or (
                    item["class"] == "Sij"
                    and 2 * case["system"]["ncas"] - case["system"]["nelecas"] == 2
                )
            )
            first_order = all(
                qd_close_rates(rates, tol["tangent_agree"])
                or max(rates) < tol["tangent_zero"]
                for rates in item["tangent"].values()
            )
            assert item["first_order"] == first_order
            # The coordinate populates it at first order, and its transition
            # tangent to the other root is nonzero along both b2 vibrations.
            for label in record["coupled"]:
                assert min(item["tangent"][label]) > tol["tangent_zero"]
                assert min(item["population"][label]) > 0.0
            assert item["rho_defect"] == (
                not item["one_function"] and item["rho_spread"] > tol["rho_agree"]
            )
        candidates = [
            i for i in sc["one_sided"] if i["first_order"] and not i["one_function"]
        ]
        if candidates:
            assert sc["chosen"] in candidates
        else:
            assert sc["chosen"] is None
        assert record["guarded_tuple"] == sc["chosen"]
        # A one-function class's coupling is linear in T: a rho defect, the
        # analytic refusal's, appears only in a class of several functions,
        # and SC's ladder is bilinear exactly when there is none.
        assert bool(sc["rho_defects"]) == bool(record["expected_refusals"]["sc"])


def test_the_coupling_guard_and_the_agreement():
    for name in COUPLING_CASES:
        record, _, _ = coupling(name)
        for scheme in record["schemes"]:
            g = record["guard"][scheme]
            assert g["prediction"] < 2.0
            assert g["subspace"] < 1e-3 and g["ci"] < 1e-3
            assert g["gradient_norm"] < 1e-10
            assert g[f"{scheme}_vectors"] < 1e-3
            assert g["core_gap"] > 0.05 and g["virtual_gap"] > 0.05
        assert min(record["reference"]["casscf_gaps"]) > 0.1
        assert min(record["reference"]["sc"]["heff_gaps"]) > 0.1
        # The tree's reference point is the analysis's: another CASSCF
        # solution misses by millihartree.
        assert max(record["reference"]["agreement"].values()) < 1e-10
        # C_MN vanishes at R0: the roots' irreps differ.
        for key in ("sc", "pc"):
            assert abs(record["reference"][key]["heff"][1]) < 1e-12


@pytest.mark.slow
def test_the_coupling_reference_reruns():
    """The analysis's reference point again, over PySCF: its energies, both
    H_eff and the guarded tuples, against the recorded analysis."""
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    spec = importlib.util.spec_from_file_location(
        "qd_coupling_analysis", TOOL / "qd_coupling_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in COUPLING_CASES:
        _, analysis, case = coupling(name)
        _, _, reference, guarded, one_sided, dropped = module.at_reference(case)
        for key in ("casscf", "sc", "pc"):
            for a, b in zip(reference[key], analysis["reference"][key], strict=True):
                assert abs(a - b) < 1e-10, (name, key)
        assert guarded == analysis["sc"]["guarded"]
        assert [(i["class"], i["ket"], i["tuple"]) for i in one_sided] == [
            (i["class"], i["ket"], i["tuple"]) for i in analysis["sc"]["one_sided"]
        ]
        assert [
            {"ket": k, "class": c, "count": n} for (k, c), n in dropped.items()
        ] == analysis["pc"]["dropped"]
