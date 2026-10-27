#!/usr/bin/env python3
"""Generate one atom's auxiliary basis end to end (see this package's
``__init__.py``): every primitive pair is a candidate (``--screen`` keeps
only the pairs aCD pivots on, ``eri_quartet.screen_pairs``), acCD picks the
candidates that
span them (``compact``), and the pivots are SVD-contracted against the
atom's own orbital products (``contract``, on by default; ``--no-contract``
keeps them as uncontracted primitives). One ``--threshold`` drives both
Cholesky stages, as in Lehtola's scheme. By default every target angular
momentum the atom's own shells can reach is decomposed (``--max-l``,
unbounded; ``--prune-l`` caps it per element by ``compact.pruned_max_l``,
off by default). The result is written as this tool's own schema
(``--aux-basis-out``) or real Basis Set Exchange spherical shells
(``--bse-out``, see ``to_bse_spherical``'s doc comment).

The ORBITAL basis can itself come from a Basis Set Exchange document
(``--bse-in``; ``tools/basis/bse_download.py`` fetches one), not only a name PySCF's
own library resolves -- see ``eri_quartet.basis_from_bse``. Cartesian and
spherical BSE shells are equally acceptable here, unlike in ``src/io/basis``,
which rejects Cartesian ones: the adapter reads only each shell's radial
data (exponents/coefficients), which is identical regardless of which of
those two tags a shell carries, and every Mole here is built spherical.

Examples
--------
Run carbon/STO-3G end to end, printing a summary (from the repo root)::

    uv run --group tools python -m tools.atomic_cd.cli C sto-3g

or, from inside ``tools/``::

    python3 -m atomic_cd.cli C sto-3g

A tighter contraction, and write the auxiliary shells to a file::

    python3 -m atomic_cd.cli C cc-pVDZ --contract-threshold 1e-6 \
        --aux-basis-out c_ccpvdz_accd.json

Same, as a Basis Set Exchange spherical-shell document::

    python3 -m atomic_cd.cli C cc-pVDZ --contract-threshold 1e-6 \
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

from . import compact, contract, df_check, eri_quartet

# the Cholesky threshold's default: tight under contraction, which then sizes
# the basis (Lehtola 2023 contracts a 1e-7 primitive set), looser without it
THRESHOLD = {True: 1e-7, False: 1e-4}
CONTRACT_THRESHOLD = 1e-5


def _max_target_l(
    element: str,
    prims: list[eri_quartet.Primitive],
    max_l: int | None,
    prune_l: int | None,
) -> int:
    """The highest acCD target l: `max_l`, else every l the primitives reach
    (2x their highest ell), capped by ``compact.pruned_max_l`` with
    l_inc = `prune_l` unless that is None."""
    l_obs = max(ell for ell, _e in prims)
    top = max_l if max_l is not None else 2 * l_obs
    if prune_l is not None:
        top = min(top, compact.pruned_max_l(element, l_obs, prune_l))
    return top


def build_aux_shells(
    element: str,
    basis: eri_quartet.Basis,
    threshold: float,
    max_l: int | None = None,
    prune_l: int | None = None,
    contract_threshold: float | None = None,
    screen: bool = False,
) -> tuple[dict[int, list[compact.AuxShell]], dict]:
    """One `element`'s auxiliary basis from its orbital `basis` (a name or a
    shell list): the primitive-pair pool (every pair, or aCD-screened at
    `threshold` if `screen`), acCD at `threshold` for each target l per
    ``_max_target_l``, then -- unless `contract_threshold` is None --
    ``contract.contract``. Returns (aux shells by l, summary with "aCD"
    (None unscreened) and per-l "acCD")."""
    prims = eri_quartet.primitives(element, basis)
    if screen:
        pairs, a_cd = eri_quartet.screen_pairs(element, prims, threshold)
    else:
        pairs, a_cd = eri_quartet.all_pairs(prims), None
    aux_shells: dict[int, list[compact.AuxShell]] = {}
    accd: dict[int, dict] = {}
    for target_l in range(_max_target_l(element, prims, max_l, prune_l) + 1):
        result = compact.compact_cholesky_for_l(pairs, target_l, threshold)
        if result.exponents.size == 0:
            continue
        aux_shells[target_l] = compact.to_aux_shells(result)
        accd[target_l] = {
            "candidates": int(result.exponents.size),
            "pivots": len(result.pivots),
            "max_reconstruction_error": compact.reconstruction_error(result),
        }
    if contract_threshold is not None:
        aux_shells = contract.contract(element, basis, aux_shells, contract_threshold)
        for ell, entry in accd.items():
            entry["contracted"] = sum(
                len(s.coefficients) for s in aux_shells.get(ell, [])
            )
    return aux_shells, {"aCD": a_cd, "acCD": accd}


def run(
    element: str,
    basis_name: str,
    threshold: float,
    max_l: int | None,
    basis: eri_quartet.Basis | None = None,
    prune_l: int | None = None,
    contract_threshold: float | None = None,
    screen: bool = False,
) -> dict:
    """Run the full pipeline and return a JSON-serializable summary.
    `basis_name` is always the label recorded in the summary; `basis`, when
    given (a pre-parsed shell list, e.g. from ``eri_quartet.basis_from_bse``),
    is what actually builds the orbital basis instead of `basis_name` --
    see this module's docstring, ``--bse-in``."""
    aux_shells, stages = build_aux_shells(
        element,
        basis if basis is not None else basis_name,
        threshold,
        max_l,
        prune_l,
        contract_threshold,
        screen,
    )
    return {
        "element": element,
        "basis": basis_name,
        "threshold": threshold,
        "prune_l": prune_l,
        "contract_threshold": contract_threshold,
        **stages,
        # not printed/dumped verbatim -- see _write_aux_basis
        "_aux_shells": aux_shells,
    }


def _write_aux_basis(
    path: str, element: str, basis_name: str, summary: dict, aux_shells: dict
) -> None:
    """Write the auxiliary basis as this tool's own JSON schema -- each
    shell's ell, exponents and coefficient rows (``compact.AuxShell``), NOT
    Basis Set Exchange format; ``to_bse_spherical`` is the BSE form."""
    document = {
        "generator": "tools/atomic_cd",
        "convention": (
            "real solid harmonic (spherical) auxiliary shells, each a "
            "general contraction: coefficients[f][k] is function f's "
            "coefficient on the normalized primitive r^l exp(-exponents[k] "
            "r^2), for each of the 2l+1 components of that l."
        ),
        "element": element,
        "basis": basis_name,
        "threshold": summary["threshold"],
        "screened": summary["aCD"] is not None,
        "contract_threshold": summary["contract_threshold"],
        "prune_l": summary["prune_l"],
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
    electron_shell per ``AuxShell``, which already IS the spherical
    (real solid harmonic) shape BSE's own "gto_spherical" means (compact.py's
    module docstring), and the only kind of shell this repo's src/io/basis
    reader accepts. BSE's validator rejects "gto_spherical"/"gto_cartesian"
    for ell<2 (spherical and Cartesian coincide there), so those shells are
    tagged plain "gto" instead.

    Coefficient convention: BSE coefficients multiply normalized primitives,
    which is what ``AuxShell`` holds, so they carry over as given: one
    coefficient list per contracted function (BSE's general-contraction
    shape, as cc-pVXZ uses)."""
    shells = [
        {
            "function_type": "gto" if shell.ell <= 1 else "gto_spherical",
            "region": "",
            "angular_momentum": [shell.ell],
            "exponents": [repr(e) for e in shell.exponents],
            "coefficients": [[repr(c) for c in row] for row in shell.coefficients],
        }
        for shell_list in aux_shells.values()
        for shell in shell_list
    ]
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
        description="Generate one atom's aCD/acCD auxiliary basis.",
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
        default=None,
        metavar="TAU",
        help=(
            "Cholesky threshold for both stages: aCD's, absolute over the "
            "normalized primitive pairs, and acCD's, relative to each "
            f"candidate's own self-Coulomb (default: {THRESHOLD[True]:g} with "
            f"contraction, {THRESHOLD[False]:g} without)"
        ),
    )
    parser.add_argument(
        "--screen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "draw acCD's candidates only from the primitive pairs aCD "
            "pivots on (eri_quartet.screen_pairs; default: off). ~28%% "
            "fewer primitives, ~18%% faster 3-index integrals, but never a "
            "smaller contracted set and up to ~10x the DF-RHF error at "
            "cc-pVTZ; far worse uncontracted"
        ),
    )
    parser.add_argument(
        "--contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SVD-contract the acCD pivots against the atom's own orbital "
            "products, Lehtola 2023 (contract.py; default: on)"
        ),
    )
    parser.add_argument(
        "--contract-threshold",
        type=float,
        default=None,
        metavar="EPS",
        help=(
            "keep contracted functions whose eigenvalue of J^T J is at least "
            f"EPS, the three-index integral precision (default: "
            f"{CONTRACT_THRESHOLD:g}); this is what sizes the auxiliary basis"
        ),
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=None,
        metavar="L",
        help="highest acCD target angular momentum (default: 2x the highest shell ell)",
    )
    parser.add_argument(
        "--prune-l",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="L_INC",
        help=(
            "drop target l above Lehtola's (2023) Eq. 9 cap, max(2 l_occ, "
            "l_occ + l_obs + L_INC), per element (default: off; bare flag: "
            "L_INC=1). Fine for global-fit HF/MP2 energies; it gives up the "
            "threshold's integral-error guarantee for the dropped l"
        ),
    )
    parser.add_argument(
        "--aux-basis-out",
        metavar="PATH",
        help="write the auxiliary shells to this JSON file",
    )
    parser.add_argument(
        "--bse-out",
        metavar="PATH",
        help=(
            "write the auxiliary shells as a Basis Set Exchange "
            "spherical-shell JSON file"
        ),
    )
    parser.add_argument(
        "--df-check",
        action="store_true",
        help=(
            "measure the generated auxiliary basis (df_check.py): one-centre "
            "ERI fitting error for this atom and, unless --bse-in, the DF-RHF "
            "energy error of each built-in molecule containing it, the other "
            "elements' aux sets generated with the same settings"
        ),
    )
    args = parser.parse_args(argv)
    # rejected here, by name, rather than as a ZeroDivisionError deep in the
    # Cholesky or an IndexError inside PySCF
    for flag, value in (
        ("--threshold", args.threshold),
        ("--contract-threshold", args.contract_threshold),
    ):
        if value is not None and not value > 0.0:
            parser.error(f"{flag} must be > 0, got {value:g}")
    for flag, value in (("--max-l", args.max_l), ("--prune-l", args.prune_l)):
        if value is not None and value < 0:
            parser.error(f"{flag} must be >= 0, got {value}")
    if not args.contract and args.contract_threshold is not None:
        parser.error("--contract-threshold has no effect with --no-contract")
    if args.threshold is None:
        args.threshold = THRESHOLD[args.contract]
    if args.contract and args.contract_threshold is None:
        args.contract_threshold = CONTRACT_THRESHOLD
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    basis = None
    if args.bse_in:
        with open(args.bse_in, encoding="utf-8") as f:
            document = json.load(f)
        basis = eri_quartet.basis_from_bse(document, args.element)

    summary = run(
        args.element,
        args.basis,
        args.threshold,
        args.max_l,
        basis=basis,
        prune_l=args.prune_l,
        contract_threshold=args.contract_threshold,
        screen=args.screen,
    )
    aux_shells = summary.pop("_aux_shells")

    print(
        f"{summary['element']}/{summary['basis']}  "
        f"threshold={summary['threshold']:g}"
        + (
            f"  prune-l l_inc={summary['prune_l']}"
            if summary["prune_l"] is not None
            else ""
        )
        + (
            f"  contract threshold={summary['contract_threshold']:g}"
            if summary["contract_threshold"] is not None
            else "  uncontracted"
        )
    )
    a = summary["aCD"]
    if a is None:
        print("  aCD: screening off, every primitive pair is a candidate")
    else:
        print(
            f"  aCD: nao={a['nao']} vectors={a['vectors']} "
            f"pairs={a['pairs']}/{a['of_pairs']} "
            f"max_err={a['max_reconstruction_error']:.2e}"
        )
    for target_l, c in summary["acCD"].items():
        print(
            f"  acCD l={target_l}: candidates={c['candidates']} "
            f"pivots={c['pivots']} "
            f"max_err={c['max_reconstruction_error']:.2e}"
            + (f" contracted={c['contracted']}" if "contracted" in c else "")
        )

    if args.aux_basis_out:
        _write_aux_basis(
            args.aux_basis_out,
            args.element,
            args.basis,
            summary,
            aux_shells,
        )
        n_shells = sum(len(v) for v in aux_shells.values())
        print(f"wrote {args.aux_basis_out} ({n_shells} auxiliary shells)")

    if args.bse_out:
        _write_bse_basis(args.bse_out, args.element, aux_shells)
        n_shells = sum(len(v) for v in aux_shells.values())
        print(f"wrote {args.bse_out} ({n_shells} auxiliary shells, BSE spherical)")

    if args.df_check:
        _df_check(args, basis, aux_shells)

    return 0


def _df_check(
    args: argparse.Namespace,
    basis: eri_quartet.Basis | None,
    aux_shells: dict[int, list[compact.AuxShell]],
) -> None:
    orbital = basis if basis is not None else args.basis
    aux = compact.to_pyscf_basis(aux_shells)
    n_aux = df_check.aux_function_count(args.element, aux)
    error = df_check.one_centre_error(args.element, orbital, aux)
    print(f"  DF one-centre: naux={n_aux} max|(ij|kl) error|={error:.2e}")
    if basis is not None:
        print("  DF-RHF: skipped (--bse-in carries only this element)")
        return
    for name, atoms in df_check.MOLECULES.items():
        symbols = {sym for sym, _xyz in atoms}
        if args.element not in symbols:
            continue
        auxbasis = {
            sym: aux
            if sym == args.element
            else compact.to_pyscf_basis(
                build_aux_shells(
                    sym,
                    args.basis,
                    args.threshold,
                    args.max_l,
                    args.prune_l,
                    args.contract_threshold,
                    args.screen,
                )[0]
            )
            for sym in symbols
        }
        e_rhf, delta = df_check.rhf_energy_error(atoms, args.basis, auxbasis)
        print(f"  DF-RHF {name}: E(RHF)={e_rhf:.10f}  E(DF)-E(RHF)={delta:+.3e}")


if __name__ == "__main__":
    sys.exit(main())
