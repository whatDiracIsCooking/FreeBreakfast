"""devtools/nightly_issue.py and test.py nightly's issue step (#986): the
title dedup, the label, the recovery, and the body, with `gh` replaced by a
fake that keeps a repository's issues in memory. Nothing here calls GitHub,
builds, needs a GPU, or reads this checkout's history."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ni = _load("devtools_nightly_issue", _ROOT / "devtools" / "nightly_issue.py")
gates = _load("devtools_test_py_for_issue", _ROOT / "devtools" / "test.py")
slow_rule = _load("devtools_slow_rule_for_issue", _ROOT / "devtools" / "slow_rule.py")

TITLE = "Nightly fails on origin/main"


class FakeGh:
    """A repository's labels and issues, behind gh's command line as
    nightly_issue calls it. `calls` is every call, (args, stdin)."""

    def __init__(self, labels=("test-infra",), issues=()) -> None:
        self.labels = list(labels)
        # number -> {"title", "state", "labels", "body", "comments"}
        self.issues = {i["number"]: dict(i, comments=[]) for i in issues}
        self.calls: list[tuple[list[str], str | None]] = []

    def verbs(self) -> list[str]:
        return [" ".join(a[:2]) for a, _ in self.calls]

    def __call__(self, args: list[str], stdin: str | None = None) -> str:
        self.calls.append((list(args), stdin))
        verb = args[:2]
        if verb == ["label", "list"]:
            return json.dumps([{"name": n} for n in self.labels])
        if verb == ["label", "create"]:
            assert args[2] not in self.labels
            self.labels.append(args[2])
            return ""
        if verb == ["issue", "list"]:
            assert args[args.index("--state") + 1] == "open"
            label = args[args.index("--label") + 1]
            return json.dumps(
                [
                    {"number": n, "title": i["title"]}
                    for n, i in self.issues.items()
                    if i["state"] == "open" and label in i["labels"]
                ]
            )
        if verb == ["issue", "create"]:
            label = args[args.index("--label") + 1]
            assert label in self.labels, "the label must exist first"
            assert args[args.index("--body-file") + 1] == "-"
            n = max([100, *self.issues]) + 1
            self.issues[n] = {
                "title": args[args.index("--title") + 1],
                "state": "open",
                "labels": [label],
                "body": stdin,
                "comments": [],
            }
            return f"https://github.com/o/r/issues/{n}\n"
        if verb == ["issue", "comment"]:
            assert args[args.index("--body-file") + 1] == "-"
            self.issues[int(args[2])]["comments"].append(stdin)
            return ""
        if verb == ["issue", "close"]:
            self.issues[int(args[2])]["state"] = "closed"
            return ""
        raise AssertionError(f"unexpected gh call: {args}")


def _issue(number: int, title: str, state: str = "open", labels=("nightly",)):
    return {"number": number, "title": title, "state": state, "labels": list(labels)}


# --- the title dedup and the label ------------------------------------------


def test_a_first_failure_creates_the_label_and_opens_one_issue() -> None:
    gh = FakeGh()
    assert ni.report_failure(gh, TITLE, "body one") == ("opened", 101)
    assert "nightly" in gh.labels
    assert gh.verbs() == ["label list", "label create", "issue list", "issue create"]
    assert gh.issues[101]["title"] == TITLE and gh.issues[101]["body"] == "body one"
    create = gh.calls[1][0]
    assert create[create.index("--color") + 1] == ni.LABEL_COLOR


def test_a_second_failure_comments_instead_of_opening() -> None:
    gh = FakeGh()
    ni.report_failure(gh, TITLE, "night one")
    assert ni.report_failure(gh, TITLE, "night two") == ("commented", 101)
    assert list(gh.issues) == [101]
    assert gh.issues[101]["comments"] == ["night two"]
    assert gh.verbs().count("label create") == 1


def test_an_existing_label_is_not_recreated() -> None:
    gh = FakeGh(labels=("nightly",))
    ni.report_failure(gh, TITLE, "b")
    assert "label create" not in gh.verbs()


@pytest.mark.parametrize(
    "other",
    [
        TITLE + " (again)",
        "Re: " + TITLE,
        TITLE.lower(),
        "Nightly fails on some-branch",
    ],
)
def test_only_the_exact_title_is_the_same_issue(other: str) -> None:
    gh = FakeGh(labels=("nightly",), issues=[_issue(7, other)])
    assert ni.report_failure(gh, TITLE, "b") == ("opened", 101)
    assert gh.issues[7]["comments"] == []


def test_a_closed_issue_is_not_reopened() -> None:
    gh = FakeGh(labels=("nightly",), issues=[_issue(7, TITLE, state="closed")])
    assert ni.report_failure(gh, TITLE, "b") == ("opened", 101)


def test_an_unlabelled_issue_with_the_title_is_not_the_nightlys() -> None:
    gh = FakeGh(labels=("nightly",), issues=[_issue(7, TITLE, labels=("bug",))])
    assert ni.report_failure(gh, TITLE, "b") == ("opened", 101)


def test_of_two_open_issues_the_oldest_takes_the_comment() -> None:
    gh = FakeGh(labels=("nightly",), issues=[_issue(9, TITLE), _issue(4, TITLE)])
    assert ni.report_failure(gh, TITLE, "b") == ("commented", 4)


def test_issues_are_listed_not_searched() -> None:
    # The search index lags; the next night must find the last one's issue.
    gh = FakeGh()
    ni.find_open(gh, TITLE)
    ((args, _),) = gh.calls
    assert "--search" not in args and "-S" not in args


def test_a_create_without_a_url_is_an_error() -> None:
    def gh(args, stdin=None):
        if args[:2] == ["label", "list"]:
            return json.dumps([{"name": "nightly"}])
        if args[:2] == ["issue", "list"]:
            return "[]"
        return "something else\n"

    with pytest.raises(ni.IssueError):
        ni.report_failure(gh, TITLE, "b")


# --- the recovery -----------------------------------------------------------


def test_a_green_night_comments_and_closes_the_open_issue() -> None:
    gh = FakeGh()
    ni.report_failure(gh, TITLE, "night one")
    assert ni.report_recovery(gh, TITLE, "recovered") == ("recovered", 101)
    assert gh.issues[101]["comments"] == ["recovered"]
    assert gh.issues[101]["state"] == "closed"
    # The next failure is a new issue.
    assert ni.report_failure(gh, TITLE, "night three") == ("opened", 102)


def test_a_green_night_with_nothing_open_only_looks() -> None:
    gh = FakeGh()
    assert ni.report_recovery(gh, TITLE, "recovered") == ("none", None)
    assert gh.verbs() == ["issue list"]
    assert ni.summary_field("none", None) == ""


def test_the_summary_field() -> None:
    assert ni.summary_field("opened", 12) == "issue=opened#12"
    assert ni.summary_field("commented", 12) == "issue=commented#12"
    assert ni.summary_field("recovered", 12) == "issue=recovered#12"


# --- the body ---------------------------------------------------------------

NIGHT = {
    "summary": "test.py nightly FAIL(rc=1)  rev=abc1234  preset=default"
    "  mode=nightly  stamp=20261106-023000  metrics=0123456789ab",
    "commit": "abc1234" + "0" * 33,
    "host": "ASUSTeK PRIME X570",
    "preset": "default",
    "mode": "nightly",
    "metrics": "0123456789ab",
    "log": "/box/.slow-tier-reports/nightly-20261106-023000-abc1234-default.log",
}
CASES = [
    {"name": "Good.Case", "binary": "free_breakfast_tests_gpu", "status": "pass"},
    {"name": "Bad.Case", "binary": "free_breakfast_tests_gpu", "status": "fail"},
    {"name": "Host.Bad", "binary": "free_breakfast_io_tests_host", "status": "fail"},
    {"name": "Skip.Case", "binary": "free_breakfast_tests_gpu", "status": "skip"},
]
FINDINGS = {
    "commit": NIGHT["commit"],
    "kind": "nightly",
    "started": "2026-11-06T02:30:00Z",
    "host": NIGHT["host"],
    "line_seconds": 1.5,
    "over_line": [{"name": "Over.Line", "seconds": 2.25, "family": "OverFam"}],
    "over_line_unrecorded": [],
    "steps": [
        {
            "name": "Step.Case",
            "seconds": 3.0,
            "median": 1.0,
            "nights": 14,
            "bar": 1.2,
        }
    ],
    "k": 5.0,
    "window": 14,
    "pooled_relative_sigma": 0.02,
    "judged": 900,
    "too_short": 3,
    "min_history": 7,
}


def test_the_body_carries_the_failing_cases_and_the_metrics_commit() -> None:
    body = ni.failure_body(NIGHT, ["C++ tests (rc=1)"], CASES, ["line a", "line b"])
    assert NIGHT["summary"] in body and NIGHT["commit"] in body
    assert "`0123456789ab` on the box's local `metrics` branch" in body
    assert "### Failing cases (2)" in body
    assert "- `Bad.Case` (free_breakfast_tests_gpu)" in body
    assert "- `Host.Bad` (free_breakfast_io_tests_host)" in body
    assert "Good.Case" not in body and "Skip.Case" not in body
    assert "- C++ tests (rc=1)" in body
    assert "````text\nline a\nline b\n````" in body
    assert NIGHT["log"] in body
    assert "Slow-test rule" not in body


def test_the_body_carries_the_slow_rule_findings_when_present() -> None:
    lines = slow_rule.render(FINDINGS)
    body = ni.failure_body(NIGHT, [], CASES, [], lines)
    assert "### Slow-test rule (`slow-rule.json`)" in body
    assert "Over.Line" in body and "(family OverFam)" in body
    assert "Step.Case" in body and "bar 1.200 s" in body
    # Before the excerpt, which is what gives way to the size cap.
    assert body.index("Slow-test rule") < body.index("Log excerpt")


def test_a_night_with_no_record_says_so() -> None:
    body = ni.failure_body(dict(NIGHT, metrics=None), ["build (rc=1)"], [], ["x"])
    assert "**Metrics commit:** not recorded" in body
    assert "### Failing cases (0)" in body and "None recorded" in body


def test_a_long_case_list_is_capped() -> None:
    many = [
        {"name": f"S.C{i}", "binary": "b", "status": "fail"}
        for i in range(ni.MAX_CASES + 5)
    ]
    body = ni.failure_body(NIGHT, [], many, [])
    assert f"S.C{ni.MAX_CASES - 1}`" in body and f"S.C{ni.MAX_CASES}`" not in body
    assert "... and 5 more" in body


def test_the_excerpt_gives_way_to_the_size_cap() -> None:
    lines = [f"{i:06d} " + "x" * 200 for i in range(2000)]
    body = ni.failure_body(NIGHT, [], CASES, lines)
    assert len(body) <= ni.MAX_BODY
    assert "Bad.Case" in body and lines[-1] in body and lines[0] not in body


def test_a_fence_in_the_log_cannot_close_the_excerpt() -> None:
    body = ni.failure_body(NIGHT, [], [], ["```", "after"])
    assert "````text\n```\nafter\n````" in body


def test_the_excerpt_is_the_failed_output_else_the_tail() -> None:
    log = "\n".join(f"l{i}" for i in range(500))
    assert ni.excerpt(log, []) == [f"l{i}" for i in range(440, 500)]
    failed = [f"f{i}" for i in range(400)]
    assert ni.excerpt(log, failed) == failed[-ni.EXCERPT_LINES :]


def test_the_recovery_body() -> None:
    body = ni.recovery_body(dict(NIGHT, summary="test.py nightly PASS  rev=abc"))
    assert body.startswith("Recovered")
    assert "test.py nightly PASS  rev=abc" in body and "0123456789ab" in body


# --- test.py nightly's issue step -------------------------------------------


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    gh = FakeGh()
    monkeypatch.setattr(ni, "gh_runner", lambda cwd: gh)
    monkeypatch.setattr(gates, "load_nightly_issue", lambda: ni)
    return gh


def test_only_origin_main_files_by_default() -> None:
    args, _ = gates.parse(["nightly"])
    assert gates.issue_title(args) == ni.DEFAULT_TITLE
    args, _ = gates.parse(["nightly", "--rev", "some-branch"])
    assert gates.issue_title(args) is None
    args, _ = gates.parse(["nightly", "--rev", "b", "--issue-title", "[t] x"])
    assert gates.issue_title(args) == "[t] x"


def test_no_title_files_nothing(fake: FakeGh, tmp_path: Path) -> None:
    log = tmp_path / "n.log"
    assert gates.file_issue(None, False, NIGHT, lambda: "b", log, "nightly") == ""
    assert fake.calls == []


def test_a_failed_then_green_night_through_test_py(
    fake: FakeGh, tmp_path: Path
) -> None:
    log = tmp_path / "n.log"
    field = gates.file_issue(TITLE, False, NIGHT, lambda: "night one", log, "nightly")
    assert field == "issue=opened#101"
    field = gates.file_issue(TITLE, False, NIGHT, lambda: "night two", log, "nightly")
    assert field == "issue=commented#101"
    field = gates.file_issue(TITLE, True, NIGHT, lambda: "unused", log, "nightly")
    assert field == "issue=recovered#101"
    assert fake.issues[101]["state"] == "closed"
    assert "issue: opened #101" in log.read_text()


def test_a_gh_failure_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(args, stdin=None):
        raise ni.IssueError("gh label: HTTP 401")

    monkeypatch.setattr(ni, "gh_runner", lambda cwd: broken)
    monkeypatch.setattr(gates, "load_nightly_issue", lambda: ni)
    log = tmp_path / "n.log"
    assert gates.file_issue(TITLE, False, NIGHT, lambda: "b", log, "x") == (
        "issue=FAILED"
    )
    assert "HTTP 401" in log.read_text()


def test_a_missing_gh_is_an_issue_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ni.IssueError, match="not installed"):
        ni.gh_runner(Path("/"))(["issue", "list"], None)


def test_the_body_reads_the_log_and_slow_rule_json(tmp_path: Path) -> None:
    log = tmp_path / "nightly-x.log"
    log.write_text(
        "--- build ---\n"
        "[ RUN      ] Pass.Case\n[       OK ] Pass.Case (1 ms)\n"
        "[ RUN      ] Bad.Case\nexpected 1, got 2\n"
        "[  FAILED  ] Bad.Case (3 ms)\n"
        "nightly: done\n"
    )
    results = log.with_suffix(".results")
    results.mkdir()
    (results / "slow-rule.json").write_text(json.dumps(FINDINGS))
    body = gates.nightly_issue_body(NIGHT, ["C++"], {"cases": CASES}, log)
    excerpt = body[body.index("### Log excerpt") :]
    assert "expected 1, got 2" in excerpt and "Pass.Case" not in excerpt
    assert "Step.Case" in body

    (results / "slow-rule.json").unlink()
    body = gates.nightly_issue_body(NIGHT, ["build"], None, log)
    assert "Slow-test rule" not in body and "None recorded" in body


def test_nightly_takes_ctest_args_and_push_does_not() -> None:
    args, rest = gates.parse(["nightly", "--rev", "b", "--", "-R", "X"])
    assert args.gate == "nightly" and rest == ["-R", "X"]
    with pytest.raises(SystemExit):
        gates.parse(["push", "--", "-R", "X"])
