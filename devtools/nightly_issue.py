#!/usr/bin/env python3
"""The nightly's GitHub issue (#986; docs/testing.md "The nightly").

A failed night opens an issue labelled LABEL, creating the label when the
repository has none, or comments on the open one with the same title, so a
failure that persists stays one issue. Its body carries the failing cases, a
log excerpt, the metrics commit and, when the night wrote one, the slow-test
report (`<log>.results/slow-rule.json`, #987). A green night after a failure
comments that it recovered, and closes the issue.

The issue is found by its exact title among the open LABEL issues, listed
(not searched: the search index lags, and a second night must find the
first's issue). `test.py nightly` calls this module; it has no command line.

Every call to GitHub goes through `gh`, by subprocess, and through one
function (`gh_runner`), which the tests replace. Standard library only: it
runs under a bare python3, inside test.py nightly.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

LABEL = "nightly"
LABEL_COLOR = "B60205"
LABEL_DESCRIPTION = "A failed nightly run (devtools/test.py nightly)"
# The title of a night of origin/main. A night of any other rev files only
# under a title it is given (test.py nightly --issue-title).
DEFAULT_TITLE = "Nightly fails on origin/main"
# GitHub refuses a body over 65,536 characters; the excerpt gives way first.
MAX_BODY = 60_000
MAX_CASES = 50
EXCERPT_LINES = 150  # of the failed cases' output
TAIL_LINES = 60  # of the log, when no case's output names the failure
# The open issues listed, newest first; more than this many open nightly
# issues is a backlog no title dedup will help.
LIST_LIMIT = "200"

# gh's arguments, and its stdin when not None, to its stdout.
Gh = Callable[[list[str], str | None], str]


class IssueError(Exception):
    """gh is missing, or refused a call."""


def gh_runner(cwd: Path) -> Gh:
    """`gh` in `cwd`, whose `origin` names the repository."""

    def run(args: list[str], stdin: str | None = None) -> str:
        try:
            proc = subprocess.run(
                ["gh", *args],
                cwd=cwd,
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise IssueError("gh is not installed") from e
        if proc.returncode != 0:
            why = (proc.stderr.strip() or proc.stdout.strip()).splitlines()[-1:]
            raise IssueError(f"gh {' '.join(args[:2])}: {why[0] if why else 'failed'}")
        return proc.stdout

    return run


def ensure_label(gh: Gh) -> bool:
    """The LABEL label, created when missing. True when it was created."""
    labels = json.loads(
        gh(["label", "list", "--limit", "1000", "--json", "name"], None)
    )
    if any(label["name"] == LABEL for label in labels):
        return False
    gh(
        [
            "label",
            "create",
            LABEL,
            "--color",
            LABEL_COLOR,
            "--description",
            LABEL_DESCRIPTION,
        ],
        None,
    )
    return True


def find_open(gh: Gh, title: str) -> int | None:
    """The open LABEL issue titled exactly `title`; the oldest, should there
    be several."""
    issues = json.loads(
        gh(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                LABEL,
                "--limit",
                LIST_LIMIT,
                "--json",
                "number,title",
            ],
            None,
        )
    )
    numbers = [i["number"] for i in issues if i["title"] == title]
    return min(numbers) if numbers else None


def _number(url: str) -> int:
    m = re.search(r"/issues/(\d+)\s*$", url.strip())
    if m is None:
        raise IssueError(f"gh issue create printed no issue URL: {url.strip()!r}")
    return int(m.group(1))


def report_failure(gh: Gh, title: str, body: str) -> tuple[str, int]:
    """Open the night's issue, or comment on the open one with its title.
    Returns (`opened` or `commented`, the issue's number)."""
    ensure_label(gh)
    number = find_open(gh, title)
    if number is None:
        url = gh(
            ["issue", "create", "--title", title, "--label", LABEL, "--body-file", "-"],
            body,
        )
        return "opened", _number(url)
    gh(["issue", "comment", str(number), "--body-file", "-"], body)
    return "commented", number


def report_recovery(gh: Gh, title: str, body: str) -> tuple[str, int | None]:
    """A green night: comment on the open issue with its title, and close it.
    Returns (`recovered`, its number), or (`none`, None) with none open."""
    number = find_open(gh, title)
    if number is None:
        return "none", None
    gh(["issue", "comment", str(number), "--body-file", "-"], body)
    gh(["issue", "close", str(number), "--reason", "completed"], None)
    return "recovered", number


def summary_field(action: str, number: int | None) -> str:
    """The summary line's field: `issue=opened#N`, `issue=commented#N`,
    `issue=recovered#N`; empty when nothing was filed."""
    return f"issue={action}#{number}" if number is not None else ""


# --- the body ---------------------------------------------------------------


def excerpt(log_text: str, failed: list[str]) -> list[str]:
    """The log's lines worth reading: the failed cases' output (test.py's
    failed_output) when it names any, its last EXCERPT_LINES; else the log's
    last TAIL_LINES, where a build or a phase says why it stopped."""
    if failed:
        return failed[-EXCERPT_LINES:]
    return log_text.splitlines()[-TAIL_LINES:]


def _fence(lines: list[str]) -> list[str]:
    # Four backticks, so a ``` in the log cannot close the block.
    return ["````text", *lines, "````"]


def _facts(night: dict) -> list[str]:
    metrics = night.get("metrics")
    return [
        f"- **Result:** `{night['summary']}`",
        f"- **Commit:** `{night['commit']}`",
        f"- **Host:** {night['host']}, preset `{night['preset']}`,"
        f" mode `{night['mode']}`",
        "- **Metrics commit:** "
        + (
            f"`{metrics}` on the box's local `metrics` branch"
            if metrics
            else "not recorded"
        ),
        f"- **Log:** `{night['log']}`",
    ]


def failure_body(
    night: dict,
    phases: list[str],
    cases: list[dict],
    log_lines: list[str],
    slow_rule: list[str] | None = None,
) -> str:
    """A failed night's issue or comment, in Markdown.

    `night`: `summary` (the summary.log line), `commit`, `host`, `preset`,
    `mode`, `metrics` (the record's commit, or None) and `log` (its path).
    `phases`: the phases that failed, in order. `cases`: the record's cases;
    those with status `fail` are listed. `log_lines`: the excerpt
    (`excerpt`). `slow_rule`: slow_rule.render's lines of the night's
    slow-rule.json, or None when it wrote none."""
    failing = [c for c in cases if c.get("status") == "fail"]
    out = ["The nightly failed (`devtools/test.py nightly`, #986).", ""]
    out += _facts(night)
    out += ["", "### Failed phases", ""]
    out += [f"- {p}" for p in phases] or ["- (none named; see the log)"]
    out += ["", f"### Failing cases ({len(failing)})", ""]
    if failing:
        out += [f"- `{c['name']}` ({c['binary']})" for c in failing[:MAX_CASES]]
        if len(failing) > MAX_CASES:
            out.append(f"- ... and {len(failing) - MAX_CASES} more, in the record")
    else:
        out.append(
            "None recorded: the run failed outside a case (a build, a crash"
            " or a hung batch, a phase), or wrote no report."
        )
    if slow_rule:
        out += ["", "### Slow-test rule (`slow-rule.json`)", ""]
        out += _fence(slow_rule)
    head = "\n".join(out)
    lines = list(log_lines)
    while True:
        tail = ["", f"### Log excerpt ({len(lines)} lines)", "", *_fence(lines)]
        body = head + "\n" + "\n".join(tail) + "\n"
        if len(body) <= MAX_BODY or not lines:
            return body
        lines = lines[max(1, len(lines) // 10) :]


def recovery_body(night: dict) -> str:
    """A green night's comment on the open issue, which it closes."""
    out = ["Recovered: this nightly passed (`devtools/test.py nightly`, #986).", ""]
    out += _facts(night)
    return "\n".join(out) + "\n"
