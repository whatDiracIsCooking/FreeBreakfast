#!/usr/bin/env python3
"""Write the reference data the high-aux density-fitting test reads.

tools/emit/df_reference.py records the two fitting sets this tree carries,
whose aux shells stop at g, so it reaches (g|g) and (ff|g) and no further.
This writes the fixture for the rest of the emitted ladder -- the aux h and i
rows, up to (i|i) and (ff|i) -- which no carried basis set reaches, because
a JKFIT set with i functions is not among them.

So the basis is not a library set: it is **one hand-built shell per angular
momentum**, a single primitive each, on two centres --

  orbitals  l = 0..3 (MAX_L)       on both atoms, 32 spherical functions
  auxiliary l = 0..6 (AUX_MAX_L)   on both atoms, 98 spherical functions

-- which is the smallest basis that reaches every class the two families
emit: all 28 metric classes lP <= lQ <= 6 and all 70 eri3 classes
la <= lb <= 3 against lP <= 6, each of them with both a one-centre pair
(R = 0, so the Boys argument is 0) and a two-centre one. The geometry is
skew, no atom on an axis and the pair axis in no plane, so no component of
any shell is missed by symmetry.

Each coefficient written is the primitive's own normaliser N(alpha, l), the
(l, 0, 0) Cartesian's, which is what `BasisSet` wants and what
`ShellStamp::primitive_normalized` would have produced: so the functions on
both sides are unit-normalised, every block of the two arrays is O(1), and
the test compares against libcint with no norm factor of its own (unlike
test/gtest/df_three_center_gpu.cpp, whose bases are the carried ones).

- libcint's `int2c2e_sph`, the metric (P|Q), 98 x 98.
- libcint's `int3c2e_sph`, the three-centre (mn|P), 32 x 32 x 98, split
  along the aux index into pieces under the 512 KiB hook, as
  df_reference.py's are.

Arrays are .npy, '<f8' in Fortran order, the layout `src/io/npy` reads; the
manifest `df_high_aux.json` carries the two bases shell by shell, so the
test builds exactly these shells rather than restating them. BLAS and OpenMP
are pinned to one thread and nothing run-dependent is written, so a rerun
regenerates every file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/df_high_aux_reference.py

writes test/gtest/data/df_high_aux/. Re-run it after changing the shells
here, or AUX_MAX_L in tools/emit/df_kernels.py; the files are never
hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import df, gto  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "df_high_aux"
MANIFEST = OUT / "df_high_aux.json"

# Under FelizJueves's pre-commit large-file hook, which this tree's data
# follows, and np.save's 64-byte-padded header for a 3-D '<f8' array.
# Spelled out rather than imported from df_reference.py: that module runs
# PySCF molecule setup at import, which this script has no use for.
LIMIT = 512 * 1024
NPY_HEADER = 128

# The two centres, bohr: skew, and the axis between them in no plane.
CENTRES = [(0.0, 0.0, 0.0), (1.07, 0.83, -1.21)]
# The element each centre is given. Nothing here is a nuclear integral, so
# only the basis attached to the symbol matters; two different symbols keep
# PySCF from sharing one basis between the centres.
SYMBOLS = ["He", "Ne"]

# One primitive per (centre, l): its exponent. Moderately tight at low l and
# diffuse at high l, as a real fitting set is, and different between the
# centres so no pair is accidentally degenerate.
ORBITAL_EXPONENTS = [
    {0: 1.10, 1: 0.90, 2: 0.75, 3: 0.60},
    {0: 1.30, 1: 1.00, 2: 0.80, 3: 0.65},
]
AUX_EXPONENTS = [
    {0: 2.00, 1: 1.60, 2: 1.30, 3: 1.00, 4: 0.85, 5: 0.70, 6: 0.60},
    {0: 2.40, 1: 1.80, 2: 1.40, 3: 1.10, 4: 0.90, 5: 0.75, 6: 0.65},
]

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/df_high_aux_reference.py"
)


def libcint_version() -> str:
    """The libcint PySCF ships, read from its cint.h as df_reference.py does."""
    header = pathlib.Path(pyscf.__file__).parent / "lib" / "deps" / "include"
    match = re.search(
        r'#define\s+CINT_VERSION\s+"([^"]+)"', (header / "cint.h").read_text()
    )
    assert match, "no CINT_VERSION in PySCF's cint.h"
    return match.group(1)


def primitive_norm(alpha: float, am: int) -> float:
    """N(alpha, am), the (l, 0, 0) Cartesian primitive's normaliser.

    (2 alpha / pi)^(3/4) (4 alpha)^(l/2) / sqrt((2l-1)!!) -- the one
    `ShellStamp::primitive_normalized` applies (src/basis), so a shell built
    with this coefficient has unit self-overlap and needs no further factor.
    """
    double_factorial = math.prod(range(2 * am - 1, 1, -2)) or 1
    return (
        (2.0 * alpha / math.pi) ** 0.75
        * (4.0 * alpha) ** (am / 2.0)
        / math.sqrt(double_factorial)
    )


def basis_shells(exponents: list[dict[int, float]]) -> list[dict]:
    """One manifest shell per (centre, l), in the AO order both sides use:
    centre-major, l ascending -- PySCF's shell order for a basis given per
    atom with its l ascending, and this tree's for the same `Shell` list."""
    shells = []
    for atom, per_am in enumerate(exponents):
        for am, alpha in sorted(per_am.items()):
            shells.append(
                {
                    "atom": atom,
                    "l": am,
                    "centre": list(CENTRES[atom]),
                    "exponent": alpha,
                    "coefficient": primitive_norm(alpha, am),
                }
            )
    return shells


def pyscf_basis(exponents: list[dict[int, float]]) -> dict:
    """The same shells as PySCF's per-symbol basis. The coefficient is 1.0:
    PySCF renormalises each contraction to unit self-overlap regardless, so
    its functions are the unit-normalised ones whatever is passed."""
    return {
        SYMBOLS[atom]: [[am, [alpha, 1.0]] for am, alpha in sorted(per_am.items())]
        for atom, per_am in enumerate(exponents)
    }


def molecules() -> tuple[gto.Mole, gto.Mole]:
    """The orbital molecule and the fitting one, over the same two centres."""
    atom = [[symbol, coord] for symbol, coord in zip(SYMBOLS, CENTRES, strict=True)]
    mol = gto.M(
        atom=atom,
        unit="Bohr",
        basis=pyscf_basis(ORBITAL_EXPONENTS),
        spin=0,
        cart=False,
        verbose=0,
    )
    auxmol = df.addons.make_auxmol(mol, pyscf_basis(AUX_EXPONENTS))
    # Unit-normalised functions on both sides, which the test relies on to
    # compare without a norm factor.
    for m in (mol, auxmol):
        assert numpy.abs(numpy.diag(m.intor("int1e_ovlp_sph")) - 1).max() < 1e-12
    return mol, auxmol


def check_order(m: gto.Mole, shells: list[dict]) -> None:
    """PySCF's shells are the manifest's, in the same order."""
    assert m.nbas == len(shells), (m.nbas, len(shells))
    for i, shell in enumerate(shells):
        assert int(m.bas_atom(i)) == shell["atom"], i
        assert int(m.bas_angular(i)) == shell["l"], i
        assert int(m.bas_nctr(i)) == 1, i
        assert abs(float(m.bas_exp(i)[0]) - shell["exponent"]) < 1e-15, i


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy, under the limit; its file name."""
    array = numpy.asfortranarray(array, dtype="<f8")
    path = OUT / f"{name}.npy"
    numpy.save(path, array, allow_pickle=False)
    assert path.stat().st_size <= LIMIT, f"{path.name} is over {LIMIT} bytes"
    return path.name


def save_by_aux(name: str, array: numpy.ndarray) -> list[dict]:
    """Write (nao, nao, naux) in balanced pieces along P, each under the
    limit -- df_reference.py's split, and the same manifest shape."""
    nao, _, naux = array.shape
    per_slab = nao * nao * 8
    pieces = math.ceil(naux / ((LIMIT - NPY_HEADER) // per_slab))
    bounds = [naux * k // pieces for k in range(pieces + 1)]
    out = []
    for begin, end in zip(bounds, bounds[1:], strict=False):
        stem = name if pieces == 1 else f"{name}_aux{begin}_{end}"
        out.append({"file": save(stem, array[:, :, begin:end]), "aux": [begin, end]})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    orbital_shells = basis_shells(ORBITAL_EXPONENTS)
    aux_shells = basis_shells(AUX_EXPONENTS)
    mol, auxmol = molecules()
    check_order(mol, orbital_shells)
    check_order(auxmol, aux_shells)

    metric = auxmol.intor("int2c2e_sph")
    assert metric.shape == (auxmol.nao, auxmol.nao)
    assert numpy.abs(metric - metric.T).max() < 1e-14 * numpy.abs(metric).max()
    assert numpy.linalg.eigvalsh(metric).min() > 0.0

    eri3 = df.incore.aux_e2(mol, auxmol, "int3c2e_sph", aosym="s1")
    assert eri3.shape == (mol.nao, mol.nao, auxmol.nao)
    assert (
        numpy.abs(eri3 - eri3.transpose(1, 0, 2)).max() < 1e-14 * numpy.abs(eri3).max()
    )

    manifest = {
        "written_by": "tools/emit/df_high_aux_reference.py",
        "command": COMMAND,
        "source": {
            "libcint": libcint_version(),
            "integrals": ["int2c2e_sph", "int3c2e_sph"],
        },
        "what": (
            "One hand-built shell per l on two skew centres -- orbitals to f, "
            "auxiliary to i -- the smallest basis reaching every emitted "
            "metric class (lP <= lQ <= 6) and eri3 class (la <= lb <= 3, "
            "lP <= 6). Coefficients are the primitives' own normalisers, so "
            "both sides are unit-normalised."
        ),
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in zip(SYMBOLS, CENTRES, strict=True)
        ],
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "orbital_max_l": max(shell["l"] for shell in orbital_shells),
        "aux_max_l": max(shell["l"] for shell in aux_shells),
        "orbital_shells": orbital_shells,
        "aux_shells": aux_shells,
        "int2c2e": save("int2c2e", metric),
        "int3c2e": save_by_aux("int3c2e", eri3),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(MANIFEST.relative_to(ROOT))
    print(
        f"  nao {mol.nao} naux {auxmol.nao}  max|M| {numpy.abs(metric).max():.4f}"
        f"  max|(mn|P)| {numpy.abs(eri3).max():.4f}"
    )


if __name__ == "__main__":
    main()
