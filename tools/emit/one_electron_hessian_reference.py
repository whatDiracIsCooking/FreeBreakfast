#!/usr/bin/env python3
"""Write the reference data the one-electron Hessian tests read.

Everything runs over this tree's own basis files, `data/basis/*.json`, handed
to PySCF as explicit shells (not its library's copies), spherical throughout.
Every Hessian is 3 n_atom x 3 n_atom, H[3 A + k][3 B + l] = d2E/dR_Ak dR_Bl in
hartree/bohr^2, atoms in the order written.

For the skew water of tools/emit/gradient_reference.py in cc-pVDZ (s, p, d)
and cc-pVTZ (oxygen's f shells, so every pair class through (ff)), each of
libcint's second-derivative one-electron integrals contracted with one fixed
symmetric AO matrix M, sum_mn M_mn d2X_mn/dR dR, assembled as PySCF's
`hessian.rhf.partial_hess_elec` assembles them:

- `overlap`: int1e_ipipovlp on (A, A), int1e_ipovlpip on every (A, B);
- `kinetic`: int1e_ipipkin and int1e_ipkinip the same way;
- `potential`: PySCF's `hcore_generator` (int1e_ipipnuc and int1e_ipnucip,
  plus the operator's terms, int1e_ipiprinv and int1e_iprinvip at each
  nucleus) less the kinetic part above;

and `nuclear_repulsion`, PySCF's `hess_nuc`.

The fixed matrix is gradient_reference.py's recipe, M_mn = 1 / (m + n + 1),
0-based in PySCF's spherical AO order: one IEEE division each, so a C++ test
rebuilds it bit for bit.

Every Hessian written is checked here against a fourth-order central
difference of the matching analytic gradient at fixed M (gradient_reference's
`one_electron`, step 1e-3 bohr), for symmetry, and for the translational sum
rule. BLAS and OpenMP are pinned to one thread, and nothing run-dependent
(date, host) is written, so a rerun regenerates the file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/one_electron_hessian_reference.py

writes test/gtest/data/hessian/one_electron.json. Re-run it to regenerate; the
file is never hand-edited.
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
from df_reference import WATER_SKEW, libcint_version, shells  # noqa: E402
from gradient_reference import atoms_out, molecules, one_electron, recipe  # noqa: E402
from pyscf import gto, scf  # noqa: E402
from pyscf.hessian import rhf as rhf_hessian  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "hessian" / "one_electron.json"

CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER_SKEW, "basis": "cc-pvdz", "max_l": 2},
    {"name": "h2o_ccpvtz", "atoms": WATER_SKEW, "basis": "cc-pvtz", "max_l": 3},
]

# The finite-difference check: fourth-order central difference of the
# gradient at this step (bohr), and the agreement it must reach, relative to
# max(1, max |H|). The symmetry and the translational sum rule, relative to
# max |H|.
FD_STEP = 1e-3
FD_TOLERANCE = 1e-7
SYMMETRY_TOLERANCE = 1e-12
TRANSLATION_TOLERANCE = 1e-10

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/one_electron_hessian_reference.py"
)


def bra_pair(mol: gto.Mole, aa: numpy.ndarray, ab: numpy.ndarray, m: numpy.ndarray):
    """sum_mn M_mn d2X_mn/dR dR from libcint's ipip (both derivatives on the
    first function: d2/dA dA of its centre) and ipip-across (one on each:
    d2/dA dB), M symmetric, as partial_hess_elec assembles S: twice the
    first function's terms, the second's being their mirror."""
    natm = mol.natm
    out = numpy.zeros((natm, 3, natm, 3))
    slices = mol.aoslice_by_atom()
    for a in range(natm):
        p0, p1 = slices[a][2:]
        out[a, :, a, :] += 2.0 * numpy.einsum("xypq,pq->xy", aa[:, :, p0:p1], m[p0:p1])
        for b in range(natm):
            q0, q1 = slices[b][2:]
            out[a, :, b, :] += 2.0 * numpy.einsum(
                "xypq,pq->xy", ab[:, :, p0:p1, q0:q1], m[p0:p1, q0:q1]
            )
    return out


def one_electron_hessian(mol: gto.Mole, m: numpy.ndarray) -> dict:
    nao = mol.nao
    natm = mol.natm

    def ints(name: str) -> numpy.ndarray:
        return mol.intor(name, comp=9).reshape(3, 3, nao, nao)

    overlap = bra_pair(mol, ints("int1e_ipipovlp"), ints("int1e_ipovlpip"), m)
    kinetic = bra_pair(mol, ints("int1e_ipipkin"), ints("int1e_ipkinip"), m)
    hcore_deriv = rhf_hessian.Hessian(scf.RHF(mol)).hcore_generator(mol)
    hcore = numpy.zeros((natm, 3, natm, 3))
    for a in range(natm):
        for b in range(natm):
            hcore[a, :, b, :] = numpy.einsum("xypq,pq->xy", hcore_deriv(a, b), m)
    flat = 3 * natm
    return {
        "overlap": overlap.reshape(flat, flat),
        "kinetic": kinetic.reshape(flat, flat),
        "potential": (hcore - kinetic).reshape(flat, flat),
    }


def finite_difference(case: dict, key: str, m: numpy.ndarray) -> numpy.ndarray:
    """Fourth-order central difference of one_electron(...)[key], the
    gradient at fixed m, over every nuclear coordinate: row 3 A + k."""
    coords = numpy.array([coord for _, coord in case["atoms"]])
    rows = []
    for a in range(coords.shape[0]):
        for k in range(3):
            values = {}
            for s in (-2, -1, 1, 2):
                moved = coords.copy()
                moved[a, k] += s * FD_STEP
                g = one_electron(molecules(case, moved)[0], m)[key]
                values[s] = (g["total"] if isinstance(g, dict) else g).reshape(-1)
            rows.append(
                (8.0 * (values[1] - values[-1]) - (values[2] - values[-2]))
                / (12.0 * FD_STEP)
            )
    return numpy.array(rows)


def check(case: dict, key: str, h: numpy.ndarray, m: numpy.ndarray) -> None:
    scale = numpy.abs(h).max()
    error = numpy.abs(h - finite_difference(case, key, m)).max()
    assert error < FD_TOLERANCE * max(1.0, scale), (case["name"], key, error)
    assert numpy.abs(h - h.T).max() < SYMMETRY_TOLERANCE * scale, (case["name"], key)
    natm = h.shape[0] // 3
    drift = numpy.abs(h.reshape(natm, 3, -1).sum(axis=0)).max()
    assert drift < TRANSLATION_TOLERANCE * scale, (case["name"], key, drift)


def rows(h: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in h]


def hessian_case(case: dict) -> dict:
    mol, _ = molecules(case)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == case["max_l"]
    m = recipe(mol.nao, mol.nao)
    one = one_electron_hessian(mol, m)
    for key, h in one.items():
        check(case, key, h, m)
    nuclear = rhf_hessian.hess_nuc(mol).transpose(0, 2, 1, 3).reshape(3 * mol.natm, -1)
    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "shells": shells(mol),
        **{key: rows(h) for key, h in one.items()},
        "nuclear_repulsion": rows(nuclear),
    }


def main() -> None:
    doc = {
        "generator": COMMAND,
        "pyscf": pyscf.__version__,
        "libcint": libcint_version(),
        "matrix": "M_mn = 1 / (m + n + 1), 0-based in PySCF's spherical AO order",
        "layout": "H[3 A + k][3 B + l] = sum_mn M_mn d2X_mn / dR_Ak dR_Bl",
        "integrals": [hessian_case(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
