"""devtools/slow_rule.py, the nightly's slow-test report (#987), over a
synthetic metrics series: a fast-set case over the line and a step are
flagged, and steady noise is not. Nothing here builds, needs a GPU, or reads
this checkout's history."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "devtools" / "slow_rule.py"
_spec = importlib.util.spec_from_file_location("devtools_slow_rule", _TOOL)
sr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sr
_spec.loader.exec_module(sr)
tf = sr._families_tool()

HOST = "box"
# Steady noise about 1 s, a few percent either way: its last night is the
# highest yet, and still not a step.
NOISE = [1.00, 1.02, 0.99, 1.01, 0.98, 1.03, 1.00, 0.97, 1.02, 1.01, 0.99, 1.00]
NOISE += [1.04, 0.98]


def _record(night: int, cases: list[dict], **extra) -> dict:
    return {
        "schema": 1,
        "kind": "nightly",
        "started": f"2026-11-{night + 1:02d}T02:30:00Z",
        "host": HOST,
        "commit": f"{night:040x}",
        "cases": cases,
        **extra,
    }


def _case(name: str, seconds: float, slow: bool | None = False, **extra) -> dict:
    c = {"name": name, "binary": "free_breakfast_tests_gpu", "status": "pass"}
    c["seconds"] = seconds
    if slow is not None:
        c["slow"] = slow
    return {**c, **extra}


def _series(last: dict[str, float], extra: list[dict] | None = None) -> str:
    """Fourteen nights of history, then the night under report."""
    lines = []
    for night, noise in enumerate(NOISE):
        cases = [
            _case("Steady.Noise", noise),
            _case("Steady.Stepped", 2 * noise),
            _case("Steady.Tiny", 0.002),
            _case("Big.Case", 10 * noise),
        ]
        if night >= len(NOISE) - 3:  # too short a history to judge
            cases.append(_case("Steady.New", 0.5))
        lines.append(json.dumps(_record(night, cases)))
    # The step cases are in the slow set, so the line leaves them alone.
    tonight = [_case(name, seconds, slow=True) for name, seconds in last.items()]
    lines.append(json.dumps(_record(len(NOISE), tonight + (extra or []))))
    return "\n".join(lines) + "\n"


def _fams() -> dict:
    F = tf.Family
    return {
        "Fast": F("Fast", ["Fast.Rep", "Fast.Over"], "Fast.Rep", "marked"),
        "Exempt": F("Exempt", ["Exempt.Timing"], by="exempt", reason="a timing"),
        "Slow": F("Slow", ["Slow.Big", "Slow.Small"], "Slow.Small", "cheapest"),
        "Host": F("Host", ["Host.Big", "Host.Small"], "Host.Small", "cheapest"),
        "Steady": F(
            "Steady",
            ["Steady.Noise", "Steady.Stepped", "Steady.Tiny", "Steady.New"],
            "Steady.Tiny",
            "cheapest",
        ),
        "Big": F("Big", ["Big.Case"], "Big.Case", "cheapest"),
    }


TONIGHT = {
    "Steady.Noise": 1.05,
    "Steady.Stepped": 2.60,
    "Steady.Tiny": 0.003,
    "Steady.New": 3.0,
    "Big.Case": 10.3,
}
OVER = [
    _case("Fast.Rep", 4.0),  # its family's representative: over its own line
    _case("Host.Small", 2.5, slow=None),  # a representative under its line
    _case("Fast.Over", 2.0),  # over the line, fast set: flagged
    _case("Exempt.Timing", 30.0),  # an exempt family
    _case("Slow.Big", 20.0, slow=True),  # under FBF_SKIP_IF_FAST_TESTS
    _case("Host.Big", 1.8, slow=None, binary="free_breakfast_io_tests_host"),
    _case("Fast.Failed", 9.0, status="fail"),  # not a pass, not judged
    {"name": "test/pytest/test_x.py::test_y", "binary": "pytest", "status": "pass"}
    | {"seconds": 40.0},  # pytest is outside slow_test.h's rule
]


def _findings() -> dict:
    return sr.check(_series(TONIGHT, OVER), HOST, _fams())


def test_a_fast_set_case_over_the_line_is_flagged() -> None:
    f = _findings()
    assert f["over_line"] == [{"name": "Fast.Over", "family": "Fast", "seconds": 2.0}]
    # Membership not recorded (ctest's report): listed apart, never dropped.
    assert [r["name"] for r in f["over_line_unrecorded"]] == ["Host.Big"]


def test_a_representative_over_its_line_is_flagged() -> None:
    # Fast.Rep, 4.0 s, is over the representatives' line (#1243) though the
    # fast-set line allows it. Host.Small at 2.5 s is under it; Big.Case, a
    # representative at 10.3 s, is one the record calls slow, which no gate
    # runs; Exempt.Timing's family has no representative.
    f = _findings()
    assert sr.REPRESENTATIVE_LINE_SECONDS == 2 * sr.LINE_SECONDS
    assert f["representative_line_seconds"] == sr.REPRESENTATIVE_LINE_SECONDS
    assert f["representatives_over_line"] == [
        {"name": "Fast.Rep", "family": "Fast", "seconds": 4.0}
    ]
    assert f["over_line"][0]["name"] == "Fast.Over"


def test_a_step_is_flagged_and_steady_noise_is_not() -> None:
    f = _findings()
    # Steady.Stepped: 2.0 s for two weeks, 2.6 s tonight, a 30% step against
    # a 2% spread. Steady.Noise's highest night yet, Big.Case's 3% and
    # Steady.Tiny's 1 ms are noise.
    assert [r["name"] for r in f["steps"]] == ["Steady.Stepped"]
    stepped = f["steps"][0]
    assert stepped["median"] == 2.0
    assert stepped["nights"] == len(NOISE)
    assert stepped["median"] < stepped["bar"] < stepped["seconds"]
    # Steady.New has three nights; OVER's seven passes have none.
    assert f["judged"] == 4
    assert f["too_short"] == 1 + 7


def test_the_bar_takes_the_largest_of_three_spreads() -> None:
    flat = [1.0] * sr.WINDOW
    # No spread of its own and none pooled: the resolution floors it.
    assert sr.step(flat, 1.0 + 4 * sr.RESOLUTION)["stepped"] is False
    assert sr.step(flat, 1.0 + 6 * sr.RESOLUTION)["stepped"] is True
    # The box's relative spread lifts a case whose own window is flat.
    assert sr.step(flat, 1.04, pooled=0.01)["stepped"] is False
    assert sr.step(flat, 1.06, pooled=0.01)["stepped"] is True
    # Too short a history is not judged at all.
    assert sr.step(flat[: sr.MIN_HISTORY - 1], 99.0) is None


def test_the_pooled_spread_ignores_quantized_cases() -> None:
    histories = {
        "a": [1.0, 1.01, 0.99] * 5,
        "tiny": [0.001, 0.002] * 7,  # 1 ms steps: quantization, not noise
        "short": [5.0, 9.0],
    }
    pooled = sr.pooled_spread(histories)
    window = histories["a"][-sr.WINDOW :]
    m = statistics.median(window)
    mad = statistics.median(abs(x - m) for x in window)
    assert pooled == sr.MAD_TO_SIGMA * mad / m > 0


def test_only_this_hosts_nightly_and_weekly_records_form_the_series() -> None:
    text = _series(TONIGHT, OVER)
    foreign = _record(20, [_case("Fast.Over", 99.0)], host="vm")
    manual = _record(21, [_case("Fast.Over", 99.0)], kind="manual")
    f = sr.check(text + json.dumps(foreign) + "\n" + json.dumps(manual), HOST, _fams())
    assert f["commit"] == f"{len(NOISE):040x}"
    assert f["over_line"][0]["seconds"] == 2.0
    assert sr.check(text, "elsewhere", _fams()) is None


def test_the_report_names_each_finding() -> None:
    f = _findings()
    text = "\n".join(sr.render(f))
    for name in ("Fast.Over", "Host.Big", "Steady.Stepped", "Fast.Rep"):
        assert name in text
    for name in ("Host.Small", "Exempt.Timing", "Slow.Big", "Steady.Noise"):
        assert name not in text
    assert sr.summary_field(f) == "slow-rule=1/1"
    assert sr.summary_field(None) == "slow-rule=none"


def test_a_gated_cheapest_member_is_not_the_representative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Gated.Small is the family's cheapest by time, but its record says it
    # reaches FBF_SKIP_IF_FAST_TESTS(): the representative is the cheapest
    # fast member, as test.py pr's is (#1125, #1225), so Gated.Mid over the
    # line is allowed and Gated.Big, also fast, is flagged.
    def case(name: str, line: int) -> tf.Case:
        return tf.Case(name, "x_gpu.cpp", line, name, None, None, "x_gpu")

    cases = [case("Gated.Small", 1), case("Gated.Mid", 2), case("Gated.Big", 3)]
    monkeypatch.setattr(tf, "build_cases", lambda build, source: (cases, []))
    monkeypatch.setattr(tf, "markers_in", lambda source, files: [])
    tonight = [
        _case("Gated.Small", 0.5, slow=True),
        _case("Gated.Mid", 2.0),
        _case("Gated.Big", 3.0),
    ]
    text = json.dumps(_record(0, tonight)) + "\n"
    fams = sr.families_of(tmp_path, tmp_path, text, HOST)
    assert (fams["Gated"].representative, fams["Gated"].by) == ("Gated.Mid", "cheapest")
    over = sr.check(text, HOST, fams)["over_line"]
    assert [r["name"] for r in over] == ["Gated.Big"]


def test_the_line_is_slow_test_hs() -> None:
    header = (_ROOT / "test" / "support" / "slow_test.h").read_text()
    m = re.search(r"more than about ([0-9.]+) s on an\s*///\s*IDLE RTX 3080", header)
    assert m, "slow_test.h no longer states the line where this test reads it"
    assert float(m.group(1)) == sr.LINE_SECONDS


def test_the_macro_records_the_property_metrics_reads() -> None:
    header = (_ROOT / "test" / "support" / "slow_test.h").read_text()
    assert 'RecordProperty("fbf_slow", "1")' in header
    metrics_py = (_ROOT / "devtools" / "metrics.py").read_text()
    assert 'case.get("fbf_slow") == "1"' in metrics_py


def test_imports_only_the_standard_library() -> None:
    tree = ast.parse(_TOOL.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    assert names - {"__future__"} <= sys.stdlib_module_names, names
