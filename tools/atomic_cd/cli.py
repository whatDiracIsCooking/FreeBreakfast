#!/usr/bin/env python3
"""Run aCD and acCD for one atom's own basis, end to end (see this package's
``__init__.py``). It writes the acCD result out as a portable file, either
this tool's own schema (``--aux-basis-out``) or real Basis Set Exchange
spherical shells (``--bse-out``, see ``to_bse_spherical``'s doc comment for
why spherical and not this repo's Cartesian ``src/basis`` shape), and by
default it decomposes every target angular momentum the atom's own shells can
reach (``--max-l``, unbounded).

The ORBITAL basis can itself come from a Basis Set Exchange document
(``--bse-in``; ``tools/bse_download.py`` fetches one), not only a name PySCF's
own library resolves -- see ``eri_quartet.basis_from_bse``. Cartesian and
spherical BSE shells are equally acceptable: the orbital basis is always built
Cartesian here (``eri_quartet.build_mol``, matching ``src/basis``), and the
adapter reads only each shell's radial data (exponents/coefficients), which is
identical regardless of which of those two tags a shell carries.

Examples
--------
Run carbon/STO-3G end to end, printing a summary (from the repo root)::

    uv run --group tools python -m tools.atomic_cd.cli C sto-3g

or, from inside ``tools/``::

    python3 -m atomic_cd.cli C sto-3g

Tighter threshold, and write the acCD auxiliary shells to a file::

    python3 -m atomic_cd.cli C cc-pVDZ --threshold 1e-8 \
        --aux-basis-out c_ccpvdz_accd.json

Same, as a Basis Set Exchange spherical-shell document::

    python3 -m atomic_cd.cli C cc-pVDZ --threshold 1e-8 \
        --bse-out c_ccpvdz_accd.bse.json

Read the ORBITAL basis from a Basis Set Exchange document instead of PySCF's
own library (the positional `basis` becomes a display label only)::

    python3 -m atomic_cd.cli C my-basis-label --bse-in c_ccpvdz.bse.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from pyscf import gto

from . import cholesky, compact, eri_quartet
from .angular import primitive_norm


def _shells_from_mol(mol) -> list[compact.ShellSpec]:
    return [
        compact.ShellSpec(mol.bas_angular(ib), mol.bas_exp(ib).tolist())
        for ib in range(mol.nbas)
    ]


def run(
    element: str,
    basis_name: str,
    threshold: float,
    max_l: int | None,
    basis: eri_quartet.Basis | None = None,
) -> dict:
    """Run the full pipeline and return a JSON-serializable summary.
    `basis_name` is always the label recorded in the summary; `basis`, when
    given (a pre-parsed shell list, e.g. from ``eri_quartet.basis_from_bse``),
    is what actually builds the orbital basis instead of `basis_name` --
    see this module's docstring, ``--bse-in``."""
    atoms = [(element, (0.0, 0.0, 0.0))]
    eri, mol = eri_quartet.eri_ao(atoms, basis if basis is not None else basis_name)
    shells = _shells_from_mol(mol)
    if max_l is None:
        max_l = 2 * max(s.ell for s in shells)

    cd_result = cholesky.atomic_cholesky(eri, threshold)
    a_cd = {
        "nao": cd_result.nao,
        "vectors": len(cd_result.vectors),
        "max_reconstruction_error": cholesky.reconstruction_error(eri, cd_result),
    }

    accd = {}
    aux_shells: dict[int, list[compact.AuxShell]] = {}
    for target_l in range(max_l + 1):
        accd_result = compact.compact_cholesky_for_l(shells, target_l, threshold)
        if not accd_result.exponents:
            continue
        aux_shells[target_l] = compact.to_aux_shells(accd_result)
        accd[target_l] = {
            "candidates": len(accd_result.exponents),
            "vectors": len(accd_result.vectors),
            "max_reconstruction_error": compact.reconstruction_error(
                shells, accd_result
            ),
        }

    return {
        "element": element,
        "basis": basis_name,
        "threshold": threshold,
        "aCD": a_cd,
        "acCD": accd,
        # not printed/dumped verbatim -- see _write_aux_basis
        "_aux_shells": aux_shells,
    }


def _write_aux_basis(
    path: str, element: str, basis_name: str, threshold: float, aux_shells: dict
) -> None:
    """Write the acCD result as this tool's own JSON schema -- NOT Basis Set
    Exchange format (BSE assumes Cartesian shells with standard contraction
    normalization; these are spherical, unnormalized-primitive-derived
    shells, see compact.py's module docstring for why real solid harmonics)."""
    document = {
        "generator": "tools/atomic_cd",
        "convention": (
            "real solid harmonic (spherical) auxiliary shells -- one radial "
            "contraction of raw-monomial-embedded, unnormalized primitives "
            "applies identically to each of the 2l+1 components of that l. "
            "Not a Cartesian src/basis Shell."
        ),
        "element": element,
        "basis": basis_name,
        "threshold": threshold,
        "shells": [asdict(shell) for shells in aux_shells.values() for shell in shells],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


def to_bse_spherical(
    element: str, aux_shells: dict[int, list[compact.AuxShell]]
) -> dict:
    """The acCD auxiliary shells as a Basis Set Exchange document, schema_type
    "minimal" (only molssi_bse_schema/function_types/elements required --
    everything else BSE's "complete" schema demands, like a name or a
    reference, is metadata this tool has no basis to invent). One
    electron_shell per accepted Cholesky vector: ``AuxShell`` already IS the
    spherical (real solid harmonic) shape BSE's own "gto_spherical" means --
    one radial contraction shared by every 2l+1 component of that l (see
    compact.py's module docstring for why this, and not this repo's Cartesian
    src/basis reader, is the right target). BSE's validator rejects
    "gto_spherical"/"gto_cartesian" for ell<2 (spherical and Cartesian
    coincide there), so those shells are tagged plain "gto" instead.

    Coefficient convention: acCD's own vectors are raw, unnormalized
    primitive coefficients (compact.py's AuxPrimitive docstring); BSE's
    convention is coefficient-times-primitive-normalized-primitive, so each
    coefficient is divided by ``primitive_norm(exponent, ell)`` -- the same
    per-l PySCF normalization compact.py's own ``_same_center_raw_block``
    already divides out -- to become a real BSE coefficient, not raw numbers
    wearing a BSE-shaped envelope.

    A single acCD shell can carry the same exponent twice: distinct shell
    pairs' uncontracted primitives (module docstring in compact.py --
    exponent zeta = a_i+b_j) can sum to an identical zeta, and both survive
    as separate ``AuxPrimitive``s in the same Cholesky vector. BSE (like any
    real GTO contraction) requires one coefficient per DISTINCT exponent, so
    same-exponent primitives are merged by summing their raw coefficients --
    exact, since c1*exp(-z r^2) + c2*exp(-z r^2) is just (c1+c2)*exp(-z r^2)
    -- before dividing by ``primitive_norm``."""
    shells = []
    for shell_list in aux_shells.values():
        for shell in shell_list:
            function_type = "gto" if shell.ell <= 1 else "gto_spherical"
            merged: dict[float, float] = {}
            for p in shell.primitives:
                merged[p.exponent] = merged.get(p.exponent, 0.0) + p.coefficient
            exponents = sorted(merged, reverse=True)
            shells.append(
                {
                    "function_type": function_type,
                    "region": "",
                    "angular_momentum": [shell.ell],
                    "exponents": [repr(e) for e in exponents],
                    "coefficients": [
                        [
                            repr(merged[e] / primitive_norm(e, shell.ell))
                            for e in exponents
                        ]
                    ],
                }
            )
    return {
        "molssi_bse_schema": {"schema_type": "minimal", "schema_version": "0.1"},
        "function_types": sorted({s["function_type"] for s in shells}),
        "elements": {str(gto.charge(element)): {"electron_shells": shells}},
    }


def _write_bse_basis(
    path: str, element: str, aux_shells: dict[int, list[compact.AuxShell]]
) -> None:
    document = to_bse_spherical(element, aux_shells)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="atomic_cd.cli",
        description="Run aCD/acCD atomic Cholesky decomposition for one atom.",
    )
    parser.add_argument("element", help="element symbol, e.g. C")
    parser.add_argument(
        "basis",
        help=(
            "a PySCF-known basis name, e.g. sto-3g, cc-pVDZ -- or, with "
            "--bse-in, just a display label"
        ),
    )
    parser.add_argument(
        "--bse-in",
        metavar="PATH",
        help=(
            "read the ORBITAL basis from a Basis Set Exchange JSON document "
            "instead of PySCF's own library; `basis` above becomes a "
            "display label only"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="Cholesky threshold (default: 1e-6)",
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=None,
        metavar="L",
        help="highest acCD target angular momentum (default: 2x the highest shell ell)",
    )
    parser.add_argument(
        "--aux-basis-out",
        metavar="PATH",
        help="write the acCD auxiliary shells to this JSON file",
    )
    parser.add_argument(
        "--bse-out",
        metavar="PATH",
        help=(
            "write the acCD auxiliary shells as a Basis Set Exchange "
            "spherical-shell JSON file"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    basis = None
    if args.bse_in:
        with open(args.bse_in, encoding="utf-8") as f:
            document = json.load(f)
        basis = eri_quartet.basis_from_bse(document, args.element)

    summary = run(args.element, args.basis, args.threshold, args.max_l, basis=basis)
    aux_shells = summary.pop("_aux_shells")

    print(
        f"{summary['element']}/{summary['basis']}  threshold={summary['threshold']:g}"
    )
    a = summary["aCD"]
    print(
        f"  aCD: nao={a['nao']} vectors={a['vectors']} "
        f"max_err={a['max_reconstruction_error']:.2e}"
    )
    for target_l, c in summary["acCD"].items():
        print(
            f"  acCD l={target_l}: candidates={c['candidates']} "
            f"vectors={c['vectors']} "
            f"max_err={c['max_reconstruction_error']:.2e}"
        )

    if args.aux_basis_out:
        _write_aux_basis(
            args.aux_basis_out, args.element, args.basis, args.threshold, aux_shells
        )
        n_shells = sum(len(v) for v in aux_shells.values())
        print(f"wrote {args.aux_basis_out} ({n_shells} auxiliary shells)")

    if args.bse_out:
        _write_bse_basis(args.bse_out, args.element, aux_shells)
        n_shells = sum(len(v) for v in aux_shells.values())
        print(f"wrote {args.bse_out} ({n_shells} auxiliary shells, BSE spherical)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
