#!/usr/bin/env python3
"""SessionStart hook: announce which half of this repo's split environment the
session is running in -- the devcontainer (container half) or the bare host.

This repo is deliberately split into a host half and a container half, and
several workflows diverge by half. The sharpest one: `devtools/worktree.sh add`
stands up a *sibling* devcontainer through the host docker daemon, so it cannot
run from inside the container -- the attempt fails with a misleading
`<root> is not writable by ubuntu (uid 1000)`, which reads like a permissions
bug rather than "wrong environment". In-container, the only worktree option is a
lightweight `.claude/worktrees/<name>` checkout. Surfacing the half up front
keeps that (and the git/token/build-preset differences CLAUDE.md documents) from
being rediscovered the hard way.

Detection: the standard Docker marker file `/.dockerenv`. The result is emitted
as SessionStart `additionalContext`, so it lands in the session's context rather
than being shown once and lost.

Wired from the committed `.claude/settings.json` as a SessionStart hook. Fails
OPEN (silent, exit 0) on any error: a hook bug must never break session startup.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if Path("/.dockerenv").exists():
        context = (
            "Environment: inside the devcontainer (the container half of this "
            "repo's split host/container setup). Worktree consequence: "
            "`devtools/worktree.sh add` is host-only and will NOT run here -- "
            "use a lightweight `.claude/worktrees/<name>` checkout instead "
            "(see CLAUDE.md 'Worktrees'). Other host/container differences "
            "(git identity, gh token reach, build presets) are documented "
            "there too."
        )
    else:
        context = (
            "Environment: the bare host (the host half of this repo's split "
            "host/container setup). Container-backed tooling "
            "(`devtools/worktree.sh add`, `devtools/devcontainer.sh`) runs from "
            "here; the C++ toolchain itself lives inside the container image."
        )

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Fail open: never let a hook bug break session startup.
        sys.exit(0)
