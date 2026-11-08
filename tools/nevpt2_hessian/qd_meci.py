#!/usr/bin/env python3
"""Converge the SA3 crossing cases' geometries (#1059): the minimum-energy
conical intersection of ethylene's QD-NEVPT2 states 0 and 1 over its
DF-SA3-CASSCF(2,2) in cc-pVDZ / cc-pVDZ-JKFIT, once per scheme, by the
tree's own `[optimize]` MECI mode (src/pipeline/optimize/README.md, "The
QD-NEVPT2 MECI") through the `free-breakfast` binary.

The config is examples/qd_meci.toml's, one per `variant`: the start
(examples/ethylene.xyz, twisted 80 degrees and pyramidalized), Opt=Tight's
four thresholds, `max_gap` 1e-7 and `[casscf] conv_tol_grad` 1e-10. The step
writes `final.xyz` only when its own end-point criteria are met; this tool
checks them again on the last evaluation of `optimize.json`, and that the
third state stays more than mcscf::kRootGapFloor above the pair, and refuses
(STUCK, nothing written) otherwise.

It writes test/gtest/data/qd_nevpt2_hessian/ethylene_meci.json: per variant
the config, the step count, the criteria and the last evaluation's values,
the final geometry (bohr, from `final.xyz`'s twelve decimals in Angstrom by
the tree's kBohrPerAngstrom) and the request `system` qd_nevpt2_hessian.py's
crossing cases read. Standard library only; needs a GPU and a build:

    tools/nevpt2_hessian/qd_meci.py [--binary build/src/free-breakfast]

About a minute per variant on the RTX 3080. Not byte for byte: the GPU sums
reassociate run to run, and the walk with them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_hessian" / "ethylene_meci.json"
BINARY = ROOT / "build" / "src" / "free-breakfast"
START = ROOT / "examples" / "ethylene.xyz"
# src/core/constants/constants.h's kBohrPerAngstrom.
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
# mcscf::kRootGapFloor (src/mcscf/root_rotation).
ROOT_GAP_FLOOR = 1e-8

VARIANTS = ("sc", "pc")
# [optimize]'s end-point criteria, examples/qd_meci.toml's: the gap and
# Opt=Tight's four.
CRITERIA = {
    "max_gap": 1e-7,
    "max_force": 1.5e-5,
    "rms_force": 1e-5,
    "max_displacement": 6e-5,
    "rms_displacement": 4e-5,
}
ACTIVE = {"ncas": 2, "nelecas": 2, "weights": [1 / 3, 1 / 3, 1 / 3]}


def config(variant: str) -> str:
    """examples/qd_meci.toml at @p variant."""
    lines = [
        "[molecule]",
        'xyz = "ethylene.xyz"',
        "[basis_set]",
        'basis = "cc-pvdz"',
        "[eri]",
        'mode = "density_fitting"',
        'aux_basis = "cc-pvdz-jkfit"',
        "[scf]",
        "max_iterations = 400",
        "[casscf]",
        f"ncas = {ACTIVE['ncas']}",
        f"nelecas = {ACTIVE['nelecas']}",
        f"nroots = {len(ACTIVE['weights'])}",
        "conv_tol_grad = 1e-10",
        "[nevpt2_gradient]",
        "quasi_degenerate = true",
        f'variant = "{variant}"',
        "[optimize]",
        "meci = [0, 1]",
        *(f"{key} = {value}" for key, value in CRITERIA.items()),
        "max_steps = 200",
    ]
    return "\n".join(lines) + "\n"


def read_xyz(path: pathlib.Path) -> list:
    """An xyz file's atoms, in bohr."""
    rows = path.read_text().splitlines()
    count = int(rows[0])
    out = []
    for row in rows[2 : 2 + count]:
        symbol, *xyz = row.split()
        out.append([symbol, [float(x) * BOHR_PER_ANGSTROM for x in xyz]])
    return out


def met(last: dict) -> dict:
    """Each end-point criterion: the last evaluation's value and its bar."""
    values = {
        "max_gap": last["gap"],
        **{k: last[k] for k in CRITERIA if k != "max_gap"},
    }
    return {
        key: {"value": values[key], "bar": bar, "met": abs(values[key]) <= bar}
        for key, bar in CRITERIA.items()
    }


def run(binary: pathlib.Path, variant: str) -> dict | None:
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        (work / "ethylene.xyz").write_text(START.read_text())
        cfg = work / "meci.toml"
        cfg.write_text(config(variant))
        out = work / "out"
        started = time.perf_counter()
        done = subprocess.run(
            [str(binary), "-i", str(work), "-o", str(out), str(cfg)],
            capture_output=True,
            text=True,
            check=False,
        )
        seconds = time.perf_counter() - started
        final = out / "final.xyz"
        if done.returncode != 0 or not final.exists():
            print(f"STUCK: {variant}: free-breakfast exited {done.returncode}")
            print(done.stdout[-2000:], done.stderr[-2000:])
            return None
        history = json.loads((out / "optimize.json").read_text())
        atoms = read_xyz(final)
    last = history["iterations"][-1]
    criteria = met(last)
    margin = last["third_state_margin"]
    print(
        f"{variant}: {len(history['iterations'])} evaluations in {seconds:.0f} s, "
        f"gap {history['gap']:.2e}, third state {margin:.4f} above the pair, "
        + ", ".join(f"{k} {v['value']:.1e}" for k, v in criteria.items()),
        flush=True,
    )
    unmet = [k for k, v in criteria.items() if not v["met"]]
    if unmet or not margin > ROOT_GAP_FLOOR or "error" in history:
        print(f"STUCK: {variant}: unmet {unmet}, third-state margin {margin}")
        return None
    return {
        "variant": variant,
        "config": config(variant),
        "start": "examples/ethylene.xyz",
        "steps": len(history["iterations"]),
        "criteria": criteria,
        "final": {
            "energy": history["energy"],
            "lower": history["lower"],
            "upper": history["upper"],
            "gap": history["gap"],
            "third_state_margin": margin,
            "root_gap": last["root_gap"],
            "heff_gap": last["heff_gap"],
            "gradient_norm": last["gradient_norm"],
            "ci_residual_norm": last["ci_residual_norm"],
            "gap_sensitivity_orbital": last["gap_sensitivity_orbital"],
            "gap_sensitivity_ci": last["gap_sensitivity_ci"],
            "g_p": history["g_p"],
            "g_q": history["g_q"],
            "h_pq": history["h_pq"],
        },
        "system": {
            "atoms": atoms,
            "basis": "cc-pvdz",
            "ncas": ACTIVE["ncas"],
            "nelecas": ACTIVE["nelecas"],
            "active": None,
            "weights": ACTIVE["weights"],
        },
        "evaluations": [
            {
                k: e[k]
                for k in (
                    "energy",
                    "lower",
                    "upper",
                    "gap",
                    "max_force",
                    "rms_force",
                    "max_displacement",
                    "rms_displacement",
                    "third_state_margin",
                )
            }
            for e in history["iterations"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=pathlib.Path, default=BINARY)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    entries = []
    for variant in VARIANTS:
        entry = run(args.binary, variant)
        if entry is None:
            sys.exit(1)
        entries.append(entry)
    doc = {
        "description": "The minimum-energy conical intersections of ethylene's "
        "SC- and PC-QD-NEVPT2 states 0 and 1 over its DF-SA3-CASSCF(2,2) "
        "(equal weights), cc-pVDZ / cc-pVDZ-JKFIT, each converged by the "
        "tree's [optimize] MECI mode from examples/ethylene.xyz. Hartree, "
        "bohr. tools/nevpt2_hessian/qd_meci.py over free-breakfast",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_meci.py",
            "step": "src/pipeline/optimize (meci, quasi-degenerate)",
        },
        "root_gap_floor": ROOT_GAP_FLOOR,
        "meci": entries,
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
