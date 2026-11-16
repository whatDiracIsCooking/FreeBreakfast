"""PySCF's density-fitted ROHF orbitals, for the Rust SCF loop's published C:
`scf.ROHF(mol).density_fit(auxbasis=...)` for every case of df_rohf.json
(the C++ tree's energies and orbital energies, copied beside this script),
over the same orbital and fitting documents the tree compiles in and the
case's own geometry in bohr, converged to `conv_tol = 1e-12` and
`conv_tol_grad = 1e-9` as tools/emit/df_reference.py converges it.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/pipeline/scf/tests/data
    uv run --frozen --group tools python $D/rohf_reference.py

It writes df_rohf_orbitals.json beside this script: its provenance (the
versions, one thread, this command), and per case its name, PySCF's total
energy, `mo_energy`, `mo_coeff` and the overlap S, each matrix n x n,
column-major as a flat list, column j of `mo_coeff` the orbital of
`mo_energy[j]`. Both matrices are in the tree's functions, not PySCF's:
the tree never renormalizes a contraction, while PySCF rescales every one
to exactly 1, so a tree function is d_m times PySCF's, d_m its published
contraction's norm (the three-center crate's reference.py, whose
`contraction_norms` this reads), and

    S_fb = diag(d) S_pyscf diag(d),     C_fb = diag(d)^-1 C_pyscf,

so that C_fb^T S_fb C_fb = I and C_fb is directly the tree's published C.
The fitted result does not depend on the scale of a fitting function, so
the aux set needs no such step. Each case's energy is checked against
df_rohf.json's to 1e-10 hartree and each solution is internally stable,
as there; a case that does not converge writes nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
THREE_CENTER = (
    ROOT / "rust/integrals/eri/density_fitting/three_center/tests/data/reference.py"
)
COMMAND = (
    "uv run --frozen --group tools python "
    "rust/pipeline/scf/tests/data/rohf_reference.py"
)


def three_center_reference():
    """The three-center crate's reference.py, as a module."""
    spec = importlib.util.spec_from_file_location(
        "three_center_reference", THREE_CENTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orbitals(ref, case: dict) -> dict:
    """One case's converged DF-ROHF, its matrices in the tree's functions."""
    atoms = [(a["symbol"], tuple(a["coord"])) for a in case["atoms"]]
    symbols = sorted({symbol for symbol, _ in atoms})
    orb_doc = ref.bse_document(case["basis"], "basis")
    aux_doc = ref.bse_document(case["aux_basis"], "aux_basis")
    basis = {s: ref.basis_from_bse(orb_doc, s) for s in symbols}
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        charge=case["charge"],
        spin=case["spin"],
        cart=False,
        verbose=0,
    )
    d = np.array(
        [n for symbol, _ in atoms for n in ref.contraction_norms(basis[symbol])]
    )
    assert d.size == mol.nao == case["nao"]
    mf = scf.ROHF(mol).density_fit(
        auxbasis={s: ref.basis_from_bse(aux_doc, s) for s in symbols}
    )
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise SystemExit(f"{case['name']}: not converged, nothing written")
    assert mf.with_df.get_naoaux() == case["naux"]
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    miss = abs(mf.e_tot - case["energy"])
    assert miss < 1e-10, f"{case['name']}: |dE| = {miss:e} against df_rohf.json"
    s = mol.intor("int1e_ovlp_sph") * np.outer(d, d)
    c = mf.mo_coeff / d[:, None]
    assert np.abs(c.T @ s @ c - np.eye(mol.nao)).max() < 1e-10
    print(
        f"{case['name']}: nao {mol.nao}, E = {mf.e_tot:.12f}, |dE| = {miss:.1e},"
        f" max|d - 1| = {np.abs(d - 1).max():.1e}"
    )
    return {
        "name": case["name"],
        "energy": float(mf.e_tot),
        "mo_energy": [float(e) for e in mf.mo_energy],
        "mo_coeff": [float(v) for v in c.flatten(order="F")],
        "overlap": [float(v) for v in s.flatten(order="F")],
    }


def main() -> None:
    lib.num_threads(1)
    ref = three_center_reference()
    cases = json.loads((HERE / "df_rohf.json").read_text())["df_rohf"]
    doc = {
        "description": "PySCF DF-ROHF orbitals (spherical, conv_tol 1e-12, "
        "conv_tol_grad 1e-9, internally stable) for each df_rohf.json case: "
        "total energy (hartree), mo_energy, and mo_coeff and the overlap in "
        "the tree's unrenormalized functions, n x n column-major",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "df_rohf_orbitals": [orbitals(ref, case) for case in cases],
    }
    (HERE / "df_rohf_orbitals.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
