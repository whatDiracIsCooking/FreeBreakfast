#!/usr/bin/env python3
"""The PySCF reference point of each qd_nevpt2_hessian.py case that names
one (`"pyscf": True`, #1243): test/gtest/data/qd_nevpt2_hessian/
<case>.pyscf.json, which that case's `reference.agreement` is held to.

A case with its own `system` and no qd_nevpt2.json golden still needs its
reference point pinned to independent code, or the ladders could be the
derivatives of another stationary point than the one the tests converge to.
This writes, at that system, tools/emit/qd_nevpt2_reference.py's polished
SA-CASSCF (|g| < 1e-10, SA-canonical core and virtuals) and its SC- and
PC-QD-NEVPT2 (PySCF's class arithmetic): the CASSCF roots, H_eff's
eigenvalues per scheme and the inactive gaps. PC refuses nothing on a case
written here; one it would is not written.

Needs the `tools` dependency group (PySCF):

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/nevpt2_hessian/qd_pyscf_point.py [--only CASE ...]

One BLAS thread, so a rerun is byte-identical. Never hand-edited.
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


def point(case: dict) -> dict:
    """The case's reference point: what the docstring lists."""
    sys_ = case["system"]
    assert sys_["active"] is None, case["name"]
    state = qd.build_state(
        {
            "from": None,
            "polish": True,
            "case": {
                "name": case["name"],
                "atoms": [(s, tuple(c)) for s, c in sys_["atoms"]],
                "basis": sys_["basis"],
                "cas": (sys_["nelecas"], sys_["ncas"]),
                "weights": sys_["weights"],
            },
        }
    )
    rdms, _, _ = qd.transition_rdms(state)
    sc, _ = qd.sc_qd(state, rdms)
    pc, info = qd.pc_qd(state, rdms, sc)
    assert not info["refusals"], (case["name"], info["refusals"])
    ncore, ncas = state.casci.ncore, state.casci.ncas
    energies = state.mo_energy
    casscf = [float(e) for e in state.energies]
    return {
        "description": "PySCF reference point of a qd_nevpt2_hessian.py case "
        "(tools/emit/qd_nevpt2_reference.py's polished SA-CASSCF, SC- and "
        "PC-QD-NEVPT2). Hartree. tools/nevpt2_hessian/qd_pyscf_point.py",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_pyscf_point.py",
            "arithmetic": "tools/emit/qd_nevpt2_reference.py",
        },
        "name": case["name"],
        "atoms": sys_["atoms"],
        "basis": sys_["basis"],
        "aux_basis": state.record["aux_basis"],
        "ncas": sys_["ncas"],
        "nelecas": sys_["nelecas"],
        "weights": sys_["weights"],
        "ncore": int(ncore),
        "nvirt": state.record["nvirt"],
        "casscf_energies": casscf,
        "casscf_gaps": generator.gaps(casscf),
        "next_root_gap": state.record["next_root_gap"],
        "inactive_gaps": {
            "core": float(numpy.diff(energies[:ncore]).min()) if ncore > 1 else None,
            "virtual": float(numpy.diff(energies[ncore + ncas :]).min()),
        },
        "sc_energies": [float(e) for e in sc["eigenvalues"]],
        "pc_energies": [float(e) for e in pc["pc_eigenvalues"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=generator.OUT)
    parser.add_argument("--only", nargs="+", help="these cases alone")
    args = parser.parse_args()
    for case in generator.CASES:
        if not case.get("pyscf") or (args.only and case["name"] not in args.only):
            continue
        record = point(case)
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / f"{case['name']}.pyscf.json"
        out.write_text(generator.dumps(record) + "\n")
        print(
            f"{case['name']}: CASSCF {record['casscf_energies']}, SC "
            f"{record['sc_energies']}, PC {record['pc_energies']}; wrote {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
