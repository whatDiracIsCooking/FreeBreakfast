#!/usr/bin/env bash
# Report -- and bump -- the Antigravity CLI release the GPU images are pinned to.
#
#   devtools/agy-version.sh                     pinned vs upstream vs installed
#   devtools/agy-version.sh --apply             rewrite the pin to upstream's current release
#   devtools/agy-version.sh --apply 1.3.4-123   rewrite it to an explicit release
#   devtools/agy-version.sh --apply --rebuild       ... and rebuild the images
#                                                   that already exist locally
#   devtools/agy-version.sh --porcelain         one `key value` per line
#
# The twin of devtools/claude-version.sh, for the second agent the images ship.
# The pin is the PAIR of lines `ARG AGY_RELEASE=` and `ARG AGY_SHA512=` in each
# file named by AGY_PIN_FILES in devtools/config.sh -- the three GPU
# Dockerfiles, which each install docker/install-agy.sh as a leaf layer. That
# script's header has why the pin exists; this one keeps the copies in step,
# because a bump that reaches two files out of three is exactly the drift it
# was meant to end.
#
# WHY A RELEASE AND NOT A VERSION, which is the one real difference from
# claude-version.sh: npm can be asked for any version that was ever published,
# so that pin is a version. Antigravity's updater endpoint serves exactly ONE
# release -- the current one -- and the only addressable artifact is a storage
# URL whose path carries the version plus an opaque build id
# (`1.3.3-5524738307653632`). So the pin is that whole segment, plus the
# tarball's sha512, and `--apply` rewrites both. A consequence worth knowing:
# this script can tell you that a NEWER release exists, but upstream publishes
# no index of OLDER ones, so rolling back needs a release id you already have
# -- a previous value of this pin, which is in git.
#
# Exit status: 0 when every file agrees with every other (and, for --apply,
# when the rewrite landed), 1 otherwise. Being BEHIND upstream is not a
# failure -- a pin is a deliberate choice, and this script's job is to make the
# gap visible, not to decide it.
#:
# Reading the current release needs the network: one unauthenticated GET of the
# updater's per-platform manifest with a short timeout. When it fails, the
# upstream column reads `-` and everything else still works, which is what
# keeps doctor.sh usable on a box with no outbound route.
#
# `--apply <release>` additionally DOWNLOADS that release's tarball, because
# the sha512 half of the pin cannot be known any other way: the manifest
# carries a hash for the current release only. That is ~60MB and is the one
# path here that is not cheap.
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
cd "$REPO_ROOT"

usage() { usage_from_header "${BASH_SOURCE[0]}"; }

MANIFEST_BASE=${AGY_MANIFEST_BASE:-https://antigravity-cli-auto-updater-974169037036.us-central1.run.app}
MANIFEST_PLATFORM=${AGY_MANIFEST_PLATFORM:-linux_amd64}
URL_BASE=${AGY_URL_BASE:-https://storage.googleapis.com/antigravity-public/antigravity-cli}
PLATFORM=${AGY_PLATFORM:-linux-x64}
TIMEOUT=${AGY_REGISTRY_TIMEOUT:-8}

# --- the pin ---------------------------------------------------------------

# Every file that carries the ARG pair, as an array. A path that does not exist
# is a hard error rather than a skip: a renamed Dockerfile that silently drops
# out of this list is the drift, not a tidying-up.
pin_files() {
  config_lines "${AGY_PIN_FILES:-}"
  PIN_FILES=("${CONFIG_LINES[@]}")
  local f
  for f in "${PIN_FILES[@]}"; do
    [ -f "$f" ] || { echo "agy-version: no such file: $f (AGY_PIN_FILES)" >&2; exit 1; }
  done
  [ "${#PIN_FILES[@]}" -gt 0 ] || {
    echo "agy-version: AGY_PIN_FILES is empty -- nothing to report on" >&2
    exit 1
  }
}

release_in() { sed -n 's/^ARG AGY_RELEASE=//p' "$1" | head -1; }
sha_in()     { sed -n 's/^ARG AGY_SHA512=//p'  "$1" | head -1; }

# The pinned release, or nothing when the files disagree. Both halves are
# compared, because a release that agrees with a sha512 that does not is a
# build that fails the checksum -- and reads as a corrupt download.
#
# DISAGREE is left holding a `file=release` list for EVERY file either way --
# the whole list, not just up to the first clash, because a partial bump that
# reached one file of three is the case this exists to name, and naming two of
# them would send you back to grep for the third.
pinned_release() {
  local f r s first_r="" first_s=""
  DISAGREE=()
  PINNED=""
  PINNED_SHA=""
  for f in "${PIN_FILES[@]}"; do
    r=$(release_in "$f")
    s=$(sha_in "$f")
    DISAGREE+=("$f=${r:-NONE}")
    [ -n "$first_r" ] || first_r=$r
    [ -n "$first_s" ] || first_s=$s
    [ "$r" = "$first_r" ] || first_r=DISAGREE
    [ "$s" = "$first_s" ] || first_s=DISAGREE
  done
  { [ "$first_r" = DISAGREE ] || [ "$first_s" = DISAGREE ]; } && return 1
  PINNED=$first_r
  PINNED_SHA=$first_s
  [ -n "$PINNED" ] && [ -n "$PINNED_SHA" ]
}

# --- the other two numbers --------------------------------------------------

# Upstream's CURRENT release, as UP_VERSION / UP_RELEASE / UP_SHA. The release
# id is not a field of the manifest -- it is the path segment of the download
# URL, which is the only place the build id appears at all.
upstream_release() {
  UP_VERSION=""; UP_RELEASE=""; UP_SHA=""
  local json url
  json=$(curl -fsSL --max-time "$TIMEOUT" \
           "$MANIFEST_BASE/manifests/$MANIFEST_PLATFORM.json" 2>/dev/null) || return 0
  UP_VERSION=$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null) || true
  url=$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("url",""))' 2>/dev/null) || true
  UP_SHA=$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sha512",""))' 2>/dev/null) || true
  UP_RELEASE=$(printf '%s\n' "$url" | sed -n 's#.*/antigravity-cli/\([^/]*\)/.*#\1#p')
  [ -n "$UP_RELEASE" ] || UP_VERSION=""   # an unparseable url is not a release
  return 0
}

# What `agy` on THIS side actually is -- a VERSION, never a release: the binary
# does not report its build id. Host and container both answer, and the answer
# means different things: in a container it should equal the pin's version half,
# on the host it is just whatever the user installed and is reported for
# contrast rather than checked.
installed_version() {
  command -v agy >/dev/null 2>&1 || return 0
  agy --version 2>/dev/null | head -1
}

release_ok() { [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; }
sha_ok()     { [[ $1 =~ ^[0-9a-f]{128}$ ]]; }

tarball_url() {
  printf '%s/%s/%s/cli_%s.tar.gz\n' \
    "$URL_BASE" "$1" "$PLATFORM" "${PLATFORM//-/_}"
}

# The sha512 of an arbitrary release, by fetching it. The expensive path, and
# the only one there is: see the header.
sha_of_release() {
  local rel=$1 url tmp sha
  url=$(tarball_url "$rel")
  tmp=$(mktemp -d)
  # shellcheck disable=SC2064  # expand tmp now, not at trap time
  trap "rm -rf '$tmp'" RETURN
  echo "agy-version: hashing $url (~60MB)" >&2
  curl -fsSL --max-time 300 -o "$tmp/agy.tar.gz" "$url" || {
    echo "agy-version: could not download $url" >&2
    echo "  a release is version AND build id -- check the id you passed" >&2
    return 1
  }
  sha=$(sha512sum "$tmp/agy.tar.gz" | cut -d' ' -f1)
  printf '%s\n' "$sha"
}

# --- rewrite ----------------------------------------------------------------

apply_pin() {
  local rel=$1 sha=$2 f before
  release_ok "$rel" || { echo "agy-version: '$rel' is not a release id" >&2; exit 1; }
  sha_ok "$sha"     || { echo "agy-version: '$sha' is not a sha512" >&2; exit 1; }
  for f in "${PIN_FILES[@]}"; do
    before=$(release_in "$f")
    [ -n "$before" ] || { echo "agy-version: $f has no ARG AGY_RELEASE line" >&2; exit 1; }
    [ -n "$(sha_in "$f")" ] || { echo "agy-version: $f has no ARG AGY_SHA512 line" >&2; exit 1; }
    sed -i "s/^ARG AGY_RELEASE=.*/ARG AGY_RELEASE=${rel}/" "$f"
    sed -i "s/^ARG AGY_SHA512=.*/ARG AGY_SHA512=${sha}/" "$f"
    if [ "$(release_in "$f")" != "$rel" ] || [ "$(sha_in "$f")" != "$sha" ]; then
      echo "agy-version: rewrite of $f did not take" >&2; exit 1
    fi
    printf '  %s: %s -> %s\n' "$f" "$before" "$rel"
  done
}

# Rebuild only what is already built. Nothing here invents a ~40GB `combined`
# build on a box that has never had one; the images it skips are named, so the
# choice stays the caller's.
rebuild_built_images() {
  local f target
  for f in "${PIN_FILES[@]}"; do
    target=${f##*Dockerfile.}
    if docker image inspect "${PROJECT_NAME}:${target}" >/dev/null 2>&1; then
      echo "agy-version: + docker/build.sh $target"
      docker/build.sh "$target"
    else
      echo "agy-version: ${PROJECT_NAME}:${target} not built here -- skipping"
      echo "             build it with: docker/build.sh $target"
    fi
  done
}

# --- arguments --------------------------------------------------------------

mode=report
want=""
rebuild=
while [ $# -gt 0 ]; do
  case $1 in
    --apply)     mode=apply ;;
    --rebuild)   mode=apply; rebuild=1 ;;
    --porcelain) mode=porcelain ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "agy-version: unknown option $1" >&2; usage >&2; exit 2 ;;
    *)           want=$1 ;;
  esac
  shift
done

pin_files

case $mode in
  porcelain)
    pinned_release || true
    upstream_release
    printf 'pinned %s\n'         "${PINNED:--}"
    printf 'pinned_version %s\n' "${PINNED%%-*}"
    printf 'files %s\n'          "${DISAGREE[*]}"
    printf 'registry %s\n'       "${UP_RELEASE:--}"
    printf 'registry_version %s\n' "${UP_VERSION:--}"
    printf 'installed %s\n'      "$(installed_version || true)"
    [ -n "$PINNED" ]
    ;;

  report)
    upstream_release
    here=$(installed_version)
    if pinned_release; then
      printf 'pinned     %s   (%s)\n' "$PINNED" "${PIN_FILES[*]}"
    else
      printf 'pinned     DISAGREE: %s\n' "${DISAGREE[*]}"
      echo   '           (release or sha512 differs between files)'
      echo   '           fix with: devtools/agy-version.sh --apply'
    fi
    printf 'upstream   %s   (%s)\n' "${UP_RELEASE:-unreachable}" "$MANIFEST_PLATFORM"
    printf 'installed  %s   (%s)\n' "${here:-none}" "$(command -v agy || echo 'not on PATH here')"
    echo
    if [ -n "$PINNED" ] && [ -n "$UP_RELEASE" ] && [ "$PINNED" != "$UP_RELEASE" ]; then
      echo "the pin is behind upstream: ${PINNED%%-*} -> ${UP_VERSION}"
      echo "  bump it:  devtools/agy-version.sh --apply"
      echo "  then:     docker/build.sh cuda   (and hip / combined as you use them)"
    elif [ -n "$PINNED" ] && [ "$PINNED" = "$UP_RELEASE" ]; then
      echo "the pin is current"
    fi
    [ -n "$PINNED" ]
    ;;

  apply)
    if [ -z "$want" ]; then
      upstream_release
      [ -n "$UP_RELEASE" ] || { echo "agy-version: upstream unreachable and no release given" >&2; exit 1; }
      echo "agy-version: upstream's current release is $UP_RELEASE ($UP_VERSION)"
      want=$UP_RELEASE
      want_sha=$UP_SHA
    else
      # An explicit release has to be hashed, because the manifest only ever
      # carries the hash of the current one.
      upstream_release
      if [ "$want" = "$UP_RELEASE" ] && [ -n "$UP_SHA" ]; then
        want_sha=$UP_SHA
      else
        want_sha=$(sha_of_release "$want") || exit 1
      fi
    fi
    sha_ok "${want_sha:-}" || { echo "agy-version: no usable sha512 for $want" >&2; exit 1; }
    apply_pin "$want" "$want_sha"
    if [ -n "$rebuild" ]; then
      rebuild_built_images
    else
      echo "agy-version: rebuild to pick it up -- docker/build.sh cuda (and hip / combined)"
      echo "             or re-run with --rebuild to do the ones already built here"
    fi
    ;;
esac
