#!/usr/bin/env bash
# The formatters: clang-format over C++, ruff format over Python.
#
#   devtools/format.sh                 rewrite every tracked file in place
#   devtools/format.sh --check         report, rewrite nothing, exit 1 on drift
#   devtools/format.sh [--check] F...  the same, over just these files
#
# --check is what the pre-commit hook and CI's `lint` job run, so neither ever
# rewrites a file behind your back: they fail, and you run this script without
# --check to fix it. That keeps the rule .pre-commit-config.yaml states -- no
# hook rewrites files -- while the format itself is still enforced.
#
# Both tools are pinned by uv.lock (clang-format is the PyPI wheel, so host and
# image run the identical binary; two clang-format minors can disagree on the
# same file). find_tool, not `command -v`, resolves them: they live in .venv/bin
# on the host and /opt/venv/bin in the image, and only the second is on PATH.
# An image built before the wheel was locked falls back to the apt
# clang-format-20 on PATH -- `devcontainer.sh rebuild` to get the pinned one.
#
# cmake-format is deliberately not here: cmake-lint (devtools/cmake-lint.sh)
# covers CMake, and cmakelang's formatter is unmaintained. deps/ is never
# touched -- the submodule's style is its repository's business.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

check=0
if [ "${1:-}" = "--check" ]; then
  check=1
  shift
fi

CPP_RE='\.(cppm|cpp|cu|cuh|h|hpp)$'
PY_RE='\.py$'

if [ "$#" -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <(git ls-files ':!:deps/**')
fi

cpp=()
py=()
for f in "${files[@]}"; do
  [ -f "$f" ] || continue # a staged deletion
  case "$f" in deps/*) continue ;; esac
  if [[ "$f" =~ $CPP_RE ]]; then
    cpp+=("$f")
  elif [[ "$f" =~ $PY_RE ]]; then
    py+=("$f")
  fi
done

status=0

if [ "${#cpp[@]}" -gt 0 ]; then
  CLANG_FORMAT=$(find_tool clang-format) || {
    echo "format: clang-format not found in ${VENV_PATHS} or on PATH" >&2
    echo "        run 'uv sync' (it is a dev dependency)" >&2
    exit 1
  }
  if [ "$check" = 1 ]; then
    "$CLANG_FORMAT" --dry-run -Werror "${cpp[@]}" || status=1
  else
    "$CLANG_FORMAT" -i "${cpp[@]}"
  fi
fi

if [ "${#py[@]}" -gt 0 ]; then
  RUFF=$(find_tool ruff) || {
    echo "format: ruff not found in ${VENV_PATHS} or on PATH; run 'uv sync'" >&2
    exit 1
  }
  if [ "$check" = 1 ]; then
    "$RUFF" format --force-exclude --check "${py[@]}" || status=1
  else
    "$RUFF" format --force-exclude "${py[@]}"
  fi
fi

if [ "$status" != 0 ]; then
  echo >&2
  echo "format: files above are not formatted; run devtools/format.sh to fix" >&2
fi
exit "$status"
