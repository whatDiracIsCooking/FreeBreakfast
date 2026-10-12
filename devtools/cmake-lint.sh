#!/usr/bin/env bash
# cmake-lint over the files pre-commit hands us, run at `git commit`.
#
# This exists because cmake-lint ran in CI and nowhere else, so the first push
# of a renamed tree failed on the server with 32 C0301 findings that were
# visible locally the whole time. It is a linter, not a formatter -- it reports
# and never rewrites -- so it does not violate this repo's no-formatter-hook
# rule (.pre-commit-config.yaml says why that rule exists).
#
# Filenames come from pre-commit, which passes only the files you staged, so a
# tree with pre-existing findings elsewhere does not block every commit. CI
# still lints the whole tree; that is the difference between the two, and the
# reason this hook is not the gate.
#
# find_tool, not `command -v`: cmakelang is a `uv sync` dependency, so on the
# host it lives in .venv/bin (not on PATH) and inside the image in
# /opt/venv/bin (on PATH). VENV_PATHS in devtools/config.sh covers both.
#
# Runnable by hand -- pass it the same file set CI lints, or just the ones you
# touched.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

[ "$#" -gt 0 ] || exit 0

CMAKE_LINT=$(find_tool cmake-lint) || {
  echo "cmake-lint: not found in ${VENV_PATHS} or on PATH" >&2
  echo "            run 'uv sync' (cmakelang is a dev dependency)" >&2
  exit 1
}

exec "$CMAKE_LINT" "$@"
