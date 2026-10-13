#!/usr/bin/env python3
"""Download basis sets from the Basis Set Exchange (https://www.basissetexchange.org/).

The C++ side (``src/io/basis``, ``free_breakfast.io.basis``) reads exactly one
on-disk format: the BSE JSON document, an object whose ``elements`` map each
atomic number to its ``electron_shells``. That is the only format this script fetches,
because it is the only one the reader understands -- there is deliberately no
``--format`` knob.

It talks to the BSE REST API with nothing but the standard library, so it adds
no entry to ``pyproject.toml`` / ``uv.lock`` (the Python side of this repo keeps
``dependencies = []`` on purpose).

Examples
--------
Download STO-3G for every element it defines, into the current directory::

    tools/bse_download.py sto-3g

Several basis sets at once, into a chosen directory::

    tools/bse_download.py sto-3g cc-pVDZ def2-SVP -o data/basis

Only a few elements (BSE accepts symbols, atomic numbers and hyphen ranges)::

    tools/bse_download.py cc-pVTZ --elements H,C,N,O
    tools/bse_download.py cc-pVTZ --elements 1,6-8

List the available basis-set names and exit::

    tools/bse_download.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_ROOT = "https://www.basissetexchange.org/api"

# Bump this if BSE ever rate-limits anonymous access; a descriptive agent is
# just good manners for a public, unauthenticated service.
USER_AGENT = "free-breakfast-bse-download/1.0"

TIMEOUT_SECONDS = 60


class BSEError(RuntimeError):
    """A request to the Basis Set Exchange failed in a way worth reporting."""


def _get(url: str) -> bytes:
    """GET ``url`` and return the raw body, mapping network failure to BSEError."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        # 404 here almost always means the basis name is unknown or misspelled;
        # say so rather than leaking a bare stack trace.
        detail = f"HTTP {exc.code} {exc.reason}"
        if exc.code == 404:
            detail += " (unknown basis set or element?)"
        raise BSEError(f"{url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BSEError(f"{url}: {exc.reason}") from exc


def fetch_basis(
    name: str,
    *,
    elements: str | None = None,
    version: str | None = None,
    header: bool = True,
    uncontract_general: bool = False,
    uncontract_spdf: bool = False,
    uncontract_segmented: bool = False,
    optimize_general: bool = False,
) -> dict:
    """Fetch one basis set as BSE complete-JSON and return it parsed.

    ``name`` is sent verbatim after URL-quoting, so the raw BSE name works even
    when it contains ``*`` or ``+`` (e.g. ``6-31g*``). The parsed document is
    checked for the ``elements`` key the C++ reader requires before it is
    handed back.
    """
    quoted = urllib.parse.quote(name, safe="")
    url = f"{API_ROOT}/basis/{quoted}/format/json/"

    # Only send the params the caller changed from the BSE default; an empty
    # query string keeps the URLs readable and the server's defaults in charge.
    params: dict[str, str] = {}
    if elements:
        params["elements"] = elements
    if version:
        params["version"] = version
    if not header:
        params["header"] = "false"
    if uncontract_general:
        params["uncontract_general"] = "true"
    if uncontract_spdf:
        params["uncontract_spdf"] = "true"
    if uncontract_segmented:
        params["uncontract_segmented"] = "true"
    if optimize_general:
        params["optimize_general"] = "true"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    body = _get(url)
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BSEError(f"{name}: response was not valid JSON: {exc}") from exc

    if not isinstance(document, dict) or "elements" not in document:
        raise BSEError(
            f"{name}: response has no 'elements' block; "
            "the basis reader would reject this file"
        )
    return document


def list_basis_names() -> list[str]:
    """Return the sorted list of basis-set names BSE currently serves."""
    body = _get(f"{API_ROOT}/metadata/")
    try:
        metadata = json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - server contract
        raise BSEError(f"metadata response was not valid JSON: {exc}") from exc
    return sorted(metadata)


def _safe_filename(name: str) -> str:
    """Turn a basis name into a portable ``.json`` filename.

    BSE names carry ``*``, ``+`` and ``/`` which are awkward or illegal in
    paths, so map them to the same spellings BSE itself uses in its filenames
    (``star`` / ``plus``) and flatten the rest.
    """
    stem = name.replace("*", "star").replace("+", "plus")
    stem = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
    return f"{stem}.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bse_download.py",
        description=(
            "Download basis sets from the Basis Set Exchange as the complete "
            "JSON that this project's basis reader consumes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="BASIS",
        help="one or more basis-set names, e.g. sto-3g cc-pVDZ def2-SVP",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        metavar="DIR",
        help="directory to write <basis>.json files into (default: current dir)",
    )
    parser.add_argument(
        "--elements",
        metavar="ELEMS",
        help="restrict to these elements; symbols or Z, ranges allowed "
        "(e.g. 'H,C,N,O' or '1,6-8')",
    )
    parser.add_argument(
        "--version",
        metavar="VER",
        help="a specific basis version (default: the latest BSE publishes)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="omit the descriptive header/metadata block from the JSON",
    )
    parser.add_argument(
        "--uncontract-general",
        action="store_true",
        help="split general contractions into separate shells",
    )
    parser.add_argument(
        "--uncontract-spdf",
        action="store_true",
        help="split sp/spd shells into separate single-momentum shells",
    )
    parser.add_argument(
        "--uncontract-segmented",
        action="store_true",
        help="fully uncontract to one primitive per shell",
    )
    parser.add_argument(
        "--optimize-general",
        action="store_true",
        help="BSE's general-contraction optimization",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing files instead of skipping them",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the available basis-set names and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list:
        try:
            for name in list_basis_names():
                print(name)
        except BSEError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args.names:
        print("error: give at least one basis-set name (or --list)", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in args.names:
        destination = args.output_dir / _safe_filename(name)
        if destination.exists() and not args.force:
            print(f"skip {name}: {destination} exists (use --force)")
            continue

        try:
            document = fetch_basis(
                name,
                elements=args.elements,
                version=args.version,
                header=not args.no_header,
                uncontract_general=args.uncontract_general,
                uncontract_spdf=args.uncontract_spdf,
                uncontract_segmented=args.uncontract_segmented,
                optimize_general=args.optimize_general,
            )
        except BSEError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue

        # Write via a temp sibling then rename, so an interrupted download never
        # leaves a half-written file that later looks cached.
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        tmp.replace(destination)
        n_elements = len(document.get("elements", {}))
        print(f"wrote {destination} ({n_elements} elements)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
