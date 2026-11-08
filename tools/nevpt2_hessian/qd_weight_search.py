#!/usr/bin/env python3
"""The search for SC-QD-NEVPT2's unequal-weight C1 SA3 Hessian reference
(#1060): test/gtest/data/qd_nevpt2_hessian/qd_weight_search.json.

`h2co_skew_ccpvdz_sa3_weighted` (0.4/0.4/0.2) has no SC Hessian reference:
its SC ladder does not settle at any step the floor resolves (the data
README). Its SA-CASSCF is another stationary point from the equal-weight
case's: two SA-canonical core orbitals 2.6e-3 Eh apart, against 0.072 there,
and its roots 2 and 3 tens of millihartree from the equal-weight ones. A
ladder costs ten minutes of GPU, so the candidates are screened here first,
at their reference points alone, over tools/emit/qd_nevpt2_reference.py's
SC-QD-NEVPT2 (PySCF's class arithmetic, the state polished to |g| < 1e-10,
SA-canonical core and virtuals), on what makes a ladder slow to settle:

- the smallest gaps between SA-canonical core and between virtual orbitals,
  whose rotations the SC energy's canonical-orbital response divides by;
- per root and class, the smallest norm N^J_t the energy's guard keeps
  (|N| > 1e-14), and how many kept norms are under SMALL: a kept tuple of
  tiny norm amplifies round-off through H/N (#1055, skew's root 2 Si tuples
  at N = 2e-7 to 7e-7). Sijrs is unguarded and has no H/N; it is skipped;
- the CASSCF root gaps, the next root's gap and SC's H_eff spectrum.

Every candidate is the skew formaldehyde geometry of `CASES` (C1), cc-pVDZ
/ cc-pVDZ-JKFIT, CAS(4,3), three roots, at one weighting. `chosen` is the
case qd_nevpt2_hessian.py ladders (`h2co_skew_ccpvdz_sa3_w424`), and its
entry is the reference point that case's `reference.agreement` is held to.

Needs the `tools` dependency group (PySCF):

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/nevpt2_hessian/qd_weight_search.py

writes test/gtest/data/qd_nevpt2_hessian/qd_weight_search.json. One BLAS
thread, so a rerun is byte-identical. Never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

import numpy  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "emit"))

# isort: split
import qd_nevpt2_reference as qd  # noqa: E402


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "qd_nevpt2_hessian", HERE / "qd_nevpt2_hessian.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_generator()

# The energy's guard on a perturber's norm (PySCF's NUMERICAL_ZERO, the
# tree's kNumericalZero), and the kept norm counted as small.
GUARD = 1e-14
SMALL = 1e-6
# The weightings screened, in the order tried: the two with references
# (equal; 0.4/0.4/0.2, SC refused), then the rest.
WEIGHTS = [
    [1 / 3, 1 / 3, 1 / 3],
    [0.4, 0.4, 0.2],
    [0.4, 0.3, 0.3],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4],
    [0.5, 0.3, 0.2],
    [0.2, 0.4, 0.4],
    [0.4, 0.2, 0.4],
    [0.35, 0.35, 0.3],
    [0.3, 0.35, 0.35],
    [0.25, 0.25, 0.5],
    [0.5, 0.25, 0.25],
    [0.6, 0.2, 0.2],
    [0.2, 0.3, 0.5],
    [0.45, 0.35, 0.2],
    [0.38, 0.38, 0.24],
]
CHOSEN = "h2co_skew_ccpvdz_sa3_w424"


def chosen_case() -> dict:
    (case,) = [c for c in generator.CASES if c["name"] == CHOSEN]
    return case


def screen(weights: list[float]) -> dict:
    """One weighting's reference point: what the docstring lists."""
    sys_ = chosen_case()["system"]
    entry = {
        "from": None,
        "polish": True,
        "case": {
            "name": "/".join(f"{w:.4g}" for w in weights),
            "atoms": [(s, tuple(c)) for s, c in sys_["atoms"]],
            "basis": sys_["basis"],
            "cas": (sys_["nelecas"], sys_["ncas"]),
            "weights": weights,
        },
    }
    state = qd.build_state(entry)
    rdms, _, _ = qd.transition_rdms(state)
    sc, _ = qd.sc_qd(state, rdms)
    ncore, ncas = state.casci.ncore, state.casci.ncas
    energies = state.mo_energy
    smallest = {}
    for j in range(len(state.ci)):
        own = qd.class_norms(state.blocks, rdms[j, j], 1.0)
        for key in qd.CLASSES:
            if key == "Sijrs":
                continue
            norms = numpy.abs(numpy.ravel(own[key]))
            kept = norms[norms > GUARD]
            smallest.setdefault(key, []).append(
                {
                    "norm": float(kept.min()) if kept.size else None,
                    "small": int((kept < SMALL).sum()),
                }
            )
    casscf = [float(e) for e in state.energies]
    record = {
        "weights": weights,
        "casscf_energies": casscf,
        "casscf_gaps": generator.gaps(casscf),
        "next_root_gap": state.record["next_root_gap"],
        "inactive_gaps": {
            "core": float(numpy.diff(energies[:ncore]).min()),
            "virtual": float(numpy.diff(energies[ncore + ncas :]).min()),
        },
        "sc_energies": [float(e) for e in sc["eigenvalues"]],
        "smallest_kept": smallest,
    }
    gaps = record["inactive_gaps"]
    print(
        f"{entry['case']['name']}: CASSCF {casscf}, core gap {gaps['core']:.2e}, "
        f"virtual gap {gaps['virtual']:.2e}",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=generator.OUT)
    args = parser.parse_args()
    sys_ = chosen_case()["system"]
    record = {
        "description": "Screening of SC-QD-NEVPT2 unequal-weight C1 SA3 "
        "candidates at their reference points (PySCF over "
        "tools/emit/qd_nevpt2_reference.py): inactive gaps, the smallest kept "
        "tuple norms per root and class, root gaps and SC's spectrum. "
        "Hartree. tools/nevpt2_hessian/qd_weight_search.py",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_weight_search.py",
            "arithmetic": "tools/emit/qd_nevpt2_reference.py",
        },
        "atoms": sys_["atoms"],
        "basis": sys_["basis"],
        "ncas": sys_["ncas"],
        "nelecas": sys_["nelecas"],
        "tolerances": {"guard": GUARD, "small": SMALL},
        "chosen": CHOSEN,
        "candidates": [screen(w) for w in WEIGHTS],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "qd_weight_search.json"
    out.write_text(generator.dumps(record) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
