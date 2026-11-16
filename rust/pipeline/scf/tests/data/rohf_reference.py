"""PySCF's density-fitted ROHF orbitals, for the Rust SCF loop's published C:
`scf.ROHF(mol).density_fit(auxbasis=...)` for every case of df_rohf.json
(the C++ tree's energies and orbital energies, copied beside this script),
and for the cases of NO_BETA below, which the C++ reference does not
carry: one electron, so no beta electron (the H atom, and H2+, whose SAD
guess is rescaled from two electrons to one). Each over the same orbital
and fitting documents the tree compiles in and the case's own geometry in
bohr, converged to `conv_tol = 1e-12` and `conv_tol_grad = 1e-9` as
tools/emit/df_reference.py converges it.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/pipeline/scf/tests/data
    uv run --frozen --group tools python $D/rohf_reference.py

It writes df_rohf_orbitals.json beside this script: its provenance (the
versions, one thread, this command), and per case its name, PySCF's total
energy, `mo_energy`, `mo_coeff` and the overlap S, each matrix n x n,
column-major as a flat list, column j of `mo_coeff` the orbital of
`mo_energy[j]`; a NO_BETA case also carries every key a df_rohf.json
case does, so the test reads it as one. Both matrices are in the tree's
functions, not PySCF's:
the tree never renormalizes a contraction, while PySCF rescales every one
to exactly 1, so a tree function is d_m times PySCF's, d_m its published
contraction's norm (the three-center crate's reference.py, whose
`contraction_norms` this reads), and

    S_fb = diag(d) S_pyscf diag(d),     C_fb = diag(d)^-1 C_pyscf,

so that C_fb^T S_fb C_fb = I and C_fb is directly the tree's published C.
The fitted result does not depend on the scale of a fitting function, so
the aux set needs no such step. Each df_rohf.json case's energy is
checked against its own there to 1e-10 hartree, a NO_BETA case's against
PySCF's one-electron shortcut (`HF1e`, the lowest eigenvalue of h, which
`scf.ROHF` would have run in place of the ROHF iteration), and every
solution is internally stable, as there; a case that does not converge
writes nothing.
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
# One electron in a doublet: n_alpha = 1, n_beta = 0. H2+ at 2.0 bohr, its
# equilibrium bond length to a few 1e-3 bohr.
NO_BETA = [
    {
        "name": "h_ccpvdz",
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 0,
        "spin": 1,
        "atoms": [{"symbol": "H", "z": 1, "coord": [0.0, 0.0, 0.0]}],
    },
    {
        "name": "h2_cation_ccpvdz",
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 1,
        "spin": 1,
        "atoms": [
            {"symbol": "H", "z": 1, "coord": [0.0, 0.0, 0.0]},
            {"symbol": "H", "z": 1, "coord": [0.0, 0.0, 2.0]},
        ],
    },
]
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
    own = "energy" not in case
    assert d.size == mol.nao and (own or mol.nao == case["nao"])
    # scf.ROHF hands a one-electron molecule to HF1e, which diagonalizes the
    # core Hamiltonian and runs no ROHF: its energy is the same, the fitted
    # J and K of one electron cancelling, but its mo_energy is h's, not the
    # Roothaan matrix's the tree publishes. The ROHF class itself runs the
    # iteration, whose energy is checked against HF1e's below.
    method = scf.rohf.ROHF if mol.nelectron == 1 else scf.ROHF
    mf = method(mol).density_fit(
        auxbasis={s: ref.basis_from_bse(aux_doc, s) for s in symbols}
    )
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise SystemExit(f"{case['name']}: not converged, nothing written")
    assert own or mf.with_df.get_naoaux() == case["naux"]
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    if own:
        one = scf.ROHF(mol)
        assert isinstance(one, scf.rohf.HF1e)
        miss = abs(mf.e_tot - one.kernel())
    else:
        miss = abs(mf.e_tot - case["energy"])
    assert miss < 1e-10, f"{case['name']}: |dE| = {miss:e} against its check"
    s = mol.intor("int1e_ovlp_sph") * np.outer(d, d)
    c = mf.mo_coeff / d[:, None]
    assert np.abs(c.T @ s @ c - np.eye(mol.nao)).max() < 1e-10
    print(
        f"{case['name']}: nao {mol.nao}, E = {mf.e_tot:.12f}, |dE| = {miss:.1e},"
        f" max|d - 1| = {np.abs(d - 1).max():.1e}"
    )
    n_alpha, n_beta = mol.nelec
    meta = {
        **case,
        "multiplicity": case["spin"] + 1,
        "nao": int(mol.nao),
        "naux": int(mf.with_df.get_naoaux()),
        "electrons": int(mol.nelectron),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }
    return {
        **(meta if own else {}),
        "name": case["name"],
        "energy": float(mf.e_tot),
        "mo_energy": [float(e) for e in mf.mo_energy],
        "mo_coeff": [float(v) for v in c.flatten(order="F")],
        "overlap": [float(v) for v in s.flatten(order="F")],
    }


def main() -> None:
    lib.num_threads(1)
    ref = three_center_reference()
    cases = json.loads((HERE / "df_rohf.json").read_text())["df_rohf"] + NO_BETA
    doc = {
        "description": "PySCF DF-ROHF orbitals (spherical, conv_tol 1e-12, "
        "conv_tol_grad 1e-9, internally stable) for each df_rohf.json case "
        "and for one-electron cases of its own (no beta electron), which also "
        "carry df_rohf.json's keys: total energy (hartree), mo_energy, and "
        "mo_coeff and the overlap in the tree's unrenormalized functions, "
        "n x n column-major",
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
