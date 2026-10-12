#!/usr/bin/env bash
# Install the Antigravity CLI (`agy`) AT AN EXPLICIT RELEASE.
#
# Called by docker/Dockerfile.{cuda,hip,combined} as a leaf layer beside
# install-claude-code.sh, and for the same reason that one is not in
# Dockerfile.base: base is the parent of every GPU image, so a bump there would
# invalidate the CUDA toolkit, the ~19GB ROCm install and the three /opt
# dependencies below it. At the leaves, a bump re-runs this layer and nothing
# else.
#
# WHY NOT upstream's `curl https://antigravity.google/cli/install.sh | bash`
# Three things it does are wrong for an image, and the fourth is the whole
# point of this file:
#
#   1. it installs to $HOME/.local/bin, which in a build is ROOT's home -- the
#      container runs as uid 1000 and would never see the binary;
#   2. it then runs `agy install`, which edits shell profiles and legacy
#      aliases. In an image that writes dotfiles for a user that is not the one
#      who will use them;
#   3. it refuses outright when the binary already exists ("delete the binary
#      first"), which is exactly the state Dockerfile.combined starts from --
#      it re-runs this script on top of the `cuda` image, which already ran it;
#   4. it installs whatever the auto-updater's manifest calls latest. That is
#      the unpinned-version drift docker/install-claude-code.sh's header
#      measured and ended, and it would be worse here: `agy` SELF-UPDATES in
#      the background during normal runs, so even one pinned install would walk
#      away from the image on its own.
#
# So this script does the same four steps upstream's does -- resolve, download,
# verify sha512, install -- with the version a reviewable line in git instead:
# ARG AGY_RELEASE and ARG AGY_SHA512 in each GPU Dockerfile, bumped together by
# devtools/agy-version.sh and reported by devtools/doctor.sh.
#
# THE PIN IS A RELEASE, NOT A VERSION, and it has to be. The manifest endpoint
# serves ONE release -- the current one -- so there is no way to ask it for an
# older version the way `npm install pkg@2.1.296` can. What is addressable is
# the storage URL, whose path segment carries the version AND an opaque build
# id (`1.3.3-5524738307653632`). Pinning that segment plus its sha512 is what
# makes this layer reproducible: the build needs no manifest at all, and an
# upstream release does not change what this image installs or break the build
# until someone bumps the pin on purpose.
#
# AGY_CLI_DISABLE_AUTO_UPDATE is set as ENV by the Dockerfile, not here -- an
# env var written in this script would last only as long as the RUN. It is the
# counterpart of `autoUpdates: false` in .devcontainer/seed-claude-config.sh:
# the binary lands in root-owned /usr/local/bin, so a self-update fails anyway,
# but failing quietly in the background is not the same as not being attempted.
# Bump with devtools/agy-version.sh, which is also what reports an old pin.
set -euo pipefail

: "${AGY_RELEASE:?set by the ARG of the same name in the Dockerfile}"
: "${AGY_SHA512:?set by the ARG of the same name in the Dockerfile}"

# The tarball's platform directory, and the base it hangs off. Overridable only
# so that an arm64 image is a build arg rather than a patch; the sha512 pin is
# per tarball, so changing either means changing AGY_SHA512 too.
AGY_PLATFORM=${AGY_PLATFORM:-linux-x64}
AGY_URL_BASE=${AGY_URL_BASE:-https://storage.googleapis.com/antigravity-public/antigravity-cli}

# /usr/local/bin, not /usr/bin: claude is in /usr/bin because the nodesource
# deb sets npm's global prefix there, which this has no reason to imitate.
# Both are on PATH for root and for ubuntu.
AGY_BIN=${AGY_BIN:-/usr/local/bin/agy}

# The version the binary reports is the release id up to the first dash. That
# is the half the pin shares with `agy --version`, so it is the half every
# check below compares.
version=${AGY_RELEASE%%-*}

# The arch guard. Without it a build on another machine downloads an x86_64
# tarball, fails the checksum, and reports a corrupt-or-compromised payload --
# which reads like a supply-chain problem rather than "this pin is for amd64".
case "$AGY_PLATFORM" in
  linux-x64)   want_arch=x86_64 ;;
  linux-arm64) want_arch=aarch64 ;;
  *) echo "install-agy: unknown AGY_PLATFORM '$AGY_PLATFORM'" >&2; exit 1 ;;
esac
have_arch=$(uname -m)
if [ "$have_arch" != "$want_arch" ]; then
  echo "install-agy: AGY_PLATFORM=$AGY_PLATFORM expects $want_arch, this is $have_arch" >&2
  echo "  the AGY_SHA512 pin is per-tarball: set both build args, not one" >&2
  exit 1
fi

# Dockerfile.combined re-runs this script on top of the `cuda` image, which
# already ran it, and the binary is 214MB -- a second copy is a second 214MB
# layer in an image that is already ~40GB. So inherit it when it is ALREADY the
# pinned release, and reinstall when it is not: the early exit is keyed on the
# version matching, so a `combined` that pins something else than its parent
# still gets its own, rather than silently keeping cuda's.
if [ -x "$AGY_BIN" ] && [ "$("$AGY_BIN" --version 2>/dev/null | head -1)" = "$version" ]; then
  echo "=== Antigravity CLI ${version} already at ${AGY_BIN} (inherited from the parent image) ==="
  exit 0
fi

url="${AGY_URL_BASE}/${AGY_RELEASE}/${AGY_PLATFORM}/cli_${AGY_PLATFORM//-/_}.tar.gz"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "install-agy: fetching $url"
if ! curl -fsSL -o "$tmp/agy.tar.gz" "$url"; then
  echo "install-agy: could not download $url" >&2
  echo "  a release is addressed by version AND build id; check ARG AGY_RELEASE" >&2
  echo "  against devtools/agy-version.sh" >&2
  exit 1
fi

# Checked, not merely recorded: this is a binary off a public bucket, and the
# sha512 in git is the only thing that says it is the one that was reviewed.
printf '%s  %s\n' "$AGY_SHA512" "$tmp/agy.tar.gz" | sha512sum -c - >/dev/null || {
  echo "install-agy: sha512 mismatch for $url" >&2
  echo "  expected $AGY_SHA512" >&2
  echo "  got      $(sha512sum "$tmp/agy.tar.gz" | cut -d' ' -f1)" >&2
  exit 1
}

# The archive holds one file, named `antigravity`. `agy` is the name upstream's
# installer gives it on disk and the name the docs use, so that is the name
# here; the binary's own usage line still says `antigravity`.
tar -xzf "$tmp/agy.tar.gz" -C "$tmp" antigravity
install -m 0755 "$tmp/antigravity" "$AGY_BIN"

# The guard that matters, and the reason this file ends rather than begins with
# the install: a download that 404s or a tarball whose member moved would both
# leave a container with no `agy` in it and a build that reported success.
installed=$("$AGY_BIN" --version 2>/dev/null | head -1)
if [ "$installed" != "$version" ]; then
  echo "install-agy: asked for ${version} (release ${AGY_RELEASE}), got ${installed:-nothing}" >&2
  exit 1
fi

# And that it is the one a shell will find. /usr/local/bin precedes /usr/bin in
# the default PATH, but this image prepends to PATH in three Dockerfiles, so
# checking beats assuming.
resolved=$(command -v agy || true)
if [ "$resolved" != "$AGY_BIN" ]; then
  echo "install-agy: installed $AGY_BIN but 'agy' on PATH resolves to ${resolved:-nothing}" >&2
  exit 1
fi

echo "=== Antigravity CLI ${version} at ${resolved} (release ${AGY_RELEASE}) ==="
