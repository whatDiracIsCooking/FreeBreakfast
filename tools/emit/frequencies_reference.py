#!/usr/bin/env python3
"""Write the reference data the harmonic-analysis tests read.

For each case, PySCF's closed-shell RHF Hessian (`mf.Hessian()`, cc-pVDZ from
PySCF's own library, `conv_tol` 1e-12) at a fixed geometry in bohr, as the
3 n_atom x 3 n_atom matrix H[3 A + k][3 B + l] = d2E/dR_Ak dR_Bl in
hartree/bohr^2, and `pyscf.hessian.thermo.harmonic_analysis` of it with
`imaginary_freq=False`, so an imaginary frequency is minus its magnitude:

- `h2o`: bent water, 3 internal modes;
- `co2`: linear along z, 4 (the bend doubly degenerate);
- `hcn`: linear along (1, 2, 2) / 3, off every axis, 4;
- `h2o_linear`: water forced linear along x, a saddle point whose degenerate
  bend is imaginary, 4.

The Hessian is only harmonic_analysis's input: where it came from does not
matter to the test, so the geometries are not minima. Each case's mode count
is checked here (3N - 6, or 3N - 5 linear), and its mass-weighted modes for
orthonormality. BLAS and OpenMP are pinned to one thread, and nothing
run-dependent (date, host) is written, so a rerun regenerates the file byte
for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/frequencies_reference.py

writes test/gtest/data/frequencies/frequencies.json. Re-run it to regenerate;
the file is never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, scf  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "frequencies" / "frequencies.json"

THIRD = 1.0 / 3.0
CASES = [
    {
        "name": "h2o",
        "atoms": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 1.43, -1.1)),
            ("H", (0.0, -1.43, -1.1)),
        ],
        "internal": 3,
    },
    {
        "name": "co2",
        "atoms": [
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 2.2)),
            ("O", (0.0, 0.0, -2.2)),
        ],
        "internal": 4,
    },
    {
        "name": "hcn",
        "atoms": [
            ("H", tuple(-2.0 * c for c in (THIRD, 2 * THIRD, 2 * THIRD))),
            ("C", (0.0, 0.0, 0.0)),
            ("N", tuple(2.2 * c for c in (THIRD, 2 * THIRD, 2 * THIRD))),
        ],
        "internal": 4,
    },
    {
        "name": "h2o_linear",
        "atoms": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (1.8, 0.0, 0.0)),
            ("H", (-1.8, 0.0, 0.0)),
        ],
        "internal": 4,
    },
]

ORTHONORMAL_TOLERANCE = 1e-10

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/frequencies_reference.py"
)


def run_case(case: dict) -> dict:
    mol = gto.M(atom=case["atoms"], unit="Bohr", basis="cc-pvdz", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, case["name"]
    natm = mol.natm
    hess = mf.Hessian().kernel()
    # Symmetrized, so the input is exactly what both sides read.
    hess = 0.5 * (hess + hess.transpose(1, 0, 3, 2))
    flat = hess.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    result = thermo.harmonic_analysis(mol, hess, imaginary_freq=False)
    freq = numpy.asarray(result["freq_wavenumber"])
    modes = numpy.asarray(result["norm_mode"]).reshape(len(freq), 3 * natm)
    assert len(freq) == case["internal"], (case["name"], len(freq))

    # Mass-weighted, the modes are orthonormal.
    mass = mol.atom_mass_list(isotope_avg=True)
    weighted = modes * numpy.repeat(numpy.sqrt(mass), 3)[None, :]
    overlap = weighted @ weighted.T
    error = numpy.abs(overlap - numpy.eye(len(freq))).max()
    assert error < ORTHONORMAL_TOLERANCE, (case["name"], error)

    return {
        "name": case["name"],
        "atoms": [
            {"z": int(mol.atom_charge(a)), "coord": mol.atom_coord(a).tolist()}
            for a in range(natm)
        ],
        "masses": mass.tolist(),
        "hessian": flat.ravel().tolist(),
        "frequencies": freq.tolist(),
        "modes": modes.ravel().tolist(),
    }


def main() -> None:
    doc = {
        "source": (
            "PySCF RHF/cc-pVDZ Hessian and pyscf.hessian.thermo.harmonic_analysis"
            "(imaginary_freq=False)"
        ),
        "command": COMMAND,
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "cases": [run_case(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
