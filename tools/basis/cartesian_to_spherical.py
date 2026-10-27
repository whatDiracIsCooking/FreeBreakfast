#!/usr/bin/env python3
"""Relabel a BSE JSON basis set's Cartesian shells as spherical, in place.

``src/io/basis`` reads spherical shells only, and rejects ``gto_cartesian`` at
any l. Some sets the Basis Set Exchange serves are Cartesian from d up -- 6-31G
from potassium on, every Pople starred set from lithium -- and this is the
step that lets the tree carry them: each ``gto_cartesian`` shell becomes
``gto_spherical``, with its exponents and contraction coefficients untouched.

That is the spherical re-contraction quantum-chemistry codes do when told
"5D": BSE coefficients multiply normalized primitives, and a primitive's radial
part is the same in either convention, so one contraction serves both. What
changes is the space. A Cartesian d shell's six functions hold x^2+y^2+z^2,
an s-like combination; the five pure d drop it (an f shell's ten Cartesians
drop three p-like ones). The result is **a different basis from the one BSE
publishes**: energies differ from a 6D calculation in the same set, and from
any program that runs that set Cartesian by default. The document keeps its
name, and its ``revision_description`` says what was done, so the provenance
travels with the file.

Only the ``function_type`` of each converted shell, the top-level
``function_types`` and ``revision_description`` change. A document with no
Cartesian shell is rewritten unchanged, so running this twice gives the same
bytes. The output format is ``bse_download.py``'s, so download-then-convert
reproduces a file byte for byte. Standard library only.

    tools/basis/bse_download.py 6-31g -o data/basis --elements 1-36 --force
    tools/basis/cartesian_to_spherical.py data/basis/6-31g.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOTE = (
    "Cartesian shells relabelled gto_spherical (5D) by "
    "tools/basis/cartesian_to_spherical.py"
)


def to_spherical(document: dict) -> int:
    """Relabel ``document``'s Cartesian shells in place; return how many."""
    converted = 0
    function_types = set()
    for element in document["elements"].values():
        for shell in element.get("electron_shells", []):
            if shell.get("function_type") == "gto_cartesian":
                # BSE allows gto_cartesian only from d up; below it the two
                # conventions coincide, and bare gto is what BSE writes there.
                is_d_up = max(shell["angular_momentum"]) >= 2
                shell["function_type"] = "gto_spherical" if is_d_up else "gto"
                converted += 1
            if "function_type" in shell:
                function_types.add(shell["function_type"])
    if "function_types" in document:
        others = {t for t in document["function_types"] if not t.startswith("gto")}
        document["function_types"] = sorted(function_types | others)
    if converted:
        previous = document.get("revision_description", "")
        document["revision_description"] = f"{previous}; {NOTE}" if previous else NOTE
    return converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cartesian_to_spherical.py",
        description="Relabel a BSE JSON basis set's Cartesian shells as spherical.",
    )
    parser.add_argument("input", type=Path, metavar="FILE", help="BSE JSON document")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="OUT",
        help="where to write the result (default: FILE, in place)",
    )
    args = parser.parse_args(argv)

    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {args.input}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(document, dict) or "elements" not in document:
        print(f"error: {args.input}: no 'elements' block", file=sys.stderr)
        return 1

    converted = to_spherical(document)
    destination = args.output or args.input
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    tmp.replace(destination)
    print(f"wrote {destination} ({converted} Cartesian shells relabelled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
