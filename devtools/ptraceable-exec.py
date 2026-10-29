#!/usr/bin/env python3
"""Let the calling shell's children trace us, then exec the command.

run_gpu_batch (devtools/lib.sh) starts the GPU batch through this, so that
when its watchdog finds the batch wedged (issue #477) it can attach a debugger
and record where every thread is before killing it. Yama's ptrace_scope=1,
the Ubuntu default and what the devcontainer inherits, lets a process trace
only its own descendants, and the debugger the watchdog runs is the batch's
sibling, not its ancestor; with no CAP_SYS_PTRACE in the container there is no
other way in. PR_SET_PTRACER names the launching shell as an allowed tracer,
and Yama extends that to the shell's descendants -- the debugger among them.

Usage: ptraceable-exec.py <tracer pid> <command> [args...]

exec keeps the pid, so the pid the shell holds is the command's own. Where
prctl is unavailable (no Yama, not Linux) the command is exec'd anyway: a batch
that cannot be traced still has to run.
"""

from __future__ import annotations

import ctypes
import os
import sys

PR_SET_PTRACER = 0x59616D61


def main() -> None:
    tracer, argv = int(sys.argv[1]), sys.argv[2:]
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(PR_SET_PTRACER, ctypes.c_ulong(tracer), 0, 0, 0)
    except (OSError, AttributeError):
        pass
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
