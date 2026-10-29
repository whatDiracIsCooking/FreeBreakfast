#!/usr/bin/env python3
"""Write the reference data the Scf step's tests read.

For each case below, PySCF's closed-shell RHF, or for an open-shell case its
ROHF, is run to a tight tolerance in the spherical basis, over this tree's own
copy of the basis (`data/basis/*.json`, the file `resolve_basis` compiles in),
so a difference between PySCF's basis library and the Basis Set Exchange's
cannot pass for an SCF error. The geometry is recorded in bohr, the unit both
sides then read it in. Recorded: the total energy, the nuclear repulsion, and
the orbital energies, all invariant under the AO normalisation, which differs
between the two codes. An ROHF case also records its spin (2S, PySCF's
convention), its multiplicity and n_alpha / n_beta; its orbital energies are
the eigenvalues of PySCF's default Roothaan effective Fock, one
canonicalization of several, and it is checked internally stable before it
is written. A localized hole, whose minimum PySCF's ROHF cannot converge to,
is iterated on the Scf step's own Roothaan matrix instead and checked against
PySCF's second-order solver (localized_hole_rohf), in a file of its own.

BLAS and OpenMP are pinned to one thread, and nothing run-dependent (date,
host) is written, so a rerun regenerates every file byte for byte.

    uv run --group tools tools/emit/scf_reference.py

writes test/gtest/data/scf/rhf.json, rohf.json and rohf_localized_hole.json.
Re-run it to regenerate; the files are never hand-edited.
"""

from __future__ import annotations

import os

# Run as a script, before numpy loads its BLAS: one thread, so every reduction
# runs in one order and the output is reproducible bit for bit. Not on import:
# the other emitters that take this module's geometries pin their own.
if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402
from pyscf.scf import rohf as pyscf_rohf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "scf" / "rhf.json"
ROHF_OUT = OUT.with_name("rohf.json")
LOCALIZED_HOLE_OUT = OUT.with_name("rohf_localized_hole.json")

# The builtin basis name the C++ side resolves, and its file under data/basis.
BASES = {
    "sto-3g": "sto-3g.json",
    "cc-pvdz": "cc-pVDZ.json",
    "cc-pvtz": "cc-pVTZ.json",
}

# Water near its equilibrium geometry, no atom on an axis past O; and NH4+,
# which carries a charge, so the SAD guess (which places sum Z electrons)
# must be rescaled. Bohr.
WATER = [
    ("O", (0.0, 0.0, 0.2216)),
    ("H", (0.0, 1.4309, -0.8867)),
    ("H", (0.0, -1.4309, -0.8867)),
]
AMMONIUM = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (1.1, 1.1, 1.1)),
    ("H", (-1.1, -1.1, 1.1)),
    ("H", (-1.1, 1.1, -1.1)),
    ("H", (1.1, -1.1, -1.1)),
]
# CO, 2.10 bohr apart along no axis, so every f component of cc-pVTZ's f
# shells is reached; the geometry of test/gtest/data/coulomb/co_ccpvtz.json.
CARBON_MONOXIDE = [
    ("C", (0.13, -0.21, 0.17)),
    ("O", (0.77, 0.86, 1.86)),
]

# The open-shell molecules. OH (0.97 A) and NH2 (1.024 A, 103 deg) are
# doublets; CH2 (1.078 A, 134 deg) is the 3B1 triplet, tilted out of every
# coordinate plane so every f component of cc-pVTZ is reached. (O2's triplet
# is not here: its default-guess ROHF is internally unstable, and the stable
# one stalls short of conv_tol_grad along the soft rotation in its pi* pair.)
# Bohr.
HYDROXYL = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.05, 1.5)),
]
AMIDOGEN = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.514, -1.205)),
    ("H", (0.0, -1.514, -1.205)),
]
METHYLENE = [
    ("C", (0.13, -0.21, 0.17)),
    ("H", (-1.4343, 1.0313, 0.5717)),
    ("H", (0.9901, -0.8926, -1.5457)),
]

# Glycylglycine, H2N-CH2-CO-NH-CH2-COOH: the first two residues of a
# pentaglycine chain, the third's N and amide H made the acid's O and H. Its
# cation's hole sits on one carbonyl lone pair, where PySCF's Roothaan matrix
# puts the open orbital below three doubly occupied ones: the ROHF minimum is
# no aufbau state of that matrix, and PySCF's ROHF does not converge in 150
# cycles from either default guess (localized_hole_rohf, below). Bohr.
GLYCYLGLYCINE = [
    ("N", (0.0, 0.0, 0.0)),
    ("C", (2.7340, 0.0, 0.0)),
    ("C", (3.7195, 0.0, 2.7089)),
    ("O", (2.3405, 0.0, 4.5767)),
    ("H", (3.3994, -1.6932, -0.9872)),
    ("H", (3.4010, 1.6875, -0.9894)),
    ("H", (-0.8924, -0.0048, 1.7001)),
    ("H", (-0.9600, 0.1449, -1.6564)),
    ("N", (6.2131, 0.0, 3.1490)),
    ("C", (7.1478, 0.0004, 5.7183)),
    ("C", (10.0304, 0.0003, 5.7183)),
    ("O", (11.3142, 0.0, 3.7839)),
    ("H", (6.4477, 1.6938, 6.6808)),
    ("H", (6.4460, -1.6869, 6.6836)),
    ("H", (7.5057, 0.0045, 1.7292)),
    ("O", (11.2965, 0.0007, 7.9112)),
    ("H", (10.4041, -0.0035, 9.6113)),
]

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "charge": 0},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
    {"name": "co_ccpvtz", "atoms": CARBON_MONOXIDE, "basis": "cc-pvtz", "charge": 0},
]

# "spin" is 2S = n_alpha - n_beta. H2O+ is the cation, at the neutral's
# geometry.
ROHF_CASES = [
    {"name": "oh_sto3g", "atoms": HYDROXYL, "basis": "sto-3g", "charge": 0, "spin": 1},
    {
        "name": "oh_ccpvdz",
        "atoms": HYDROXYL,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 1,
    },
    {
        "name": "nh2_ccpvdz",
        "atoms": AMIDOGEN,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 1,
    },
    {
        "name": "ch2_ccpvdz",
        "atoms": METHYLENE,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 2,
    },
    {
        "name": "h2o_cation_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "charge": 1,
        "spin": 1,
    },
    {
        "name": "ch2_ccpvtz",
        "atoms": METHYLENE,
        "basis": "cc-pvtz",
        "charge": 0,
        "spin": 2,
    },
]

# ROHF minima PySCF's own iteration cannot reach (localized_hole_rohf), in a
# file of their own: their orbital energies, by occupation class, are not
# ascending.
LOCALIZED_HOLE_CASES = [
    {
        "name": "glygly_cation_sto3g",
        "atoms": GLYCYLGLYCINE,
        "basis": "sto-3g",
        "charge": 1,
        "spin": 1,
    },
]


def pyscf_basis(file: str, symbols: set[str]) -> dict[str, list]:
    """This tree's BSE JSON basis, in PySCF's form, for the given elements.

    A BSE shell of several momenta (sto-3g's sp) carries one coefficient
    column per momentum; PySCF wants one shell per momentum.
    """
    doc = json.loads((ROOT / "data" / "basis" / file).read_text())
    out: dict[str, list] = {}
    for symbol in symbols:
        shells = []
        for shell in doc["elements"][str(gto.charge(symbol))]["electron_shells"]:
            exponents = [float(e) for e in shell["exponents"]]
            momenta = shell["angular_momentum"]
            columns = [[float(c) for c in col] for col in shell["coefficients"]]
            if len(momenta) == 1:
                groups = [(momenta[0], columns)]
            else:
                assert len(momenta) == len(columns)
                groups = [
                    (ang, [col]) for ang, col in zip(momenta, columns, strict=True)
                ]
            for ang, cols in groups:
                rows = [[e, *(col[k] for col in cols)] for k, e in enumerate(exponents)]
                shells.append([ang, *rows])
        out[symbol] = shells
    return out


def molecule(case: dict) -> gto.Mole:
    """One case's molecule, over this tree's basis file."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    return gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=case["charge"],
        spin=case.get("spin", 0),
        cart=False,
        verbose=0,
    )


def converge(mf: scf.hf.SCF, name: str) -> None:
    """Run mf to this file's tolerances; it must converge."""
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name} did not converge"


def rhf(case: dict) -> dict:
    """PySCF's converged RHF for one case."""
    mol = molecule(case)
    mf = scf.RHF(mol)
    converge(mf, case["name"])
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }


class OpenBlockROHF(scf.rohf.ROHF):
    """ROHF iterated on the Scf step's Roothaan matrix: PySCF's, but with
    (Fa + 3 Fb)/4 on the open-open block in place of (Fa + Fb)/2, and the
    occupations aufbau in its spectrum.

    The diagonal blocks are a canonicalization: they move no stationary point,
    only which one the iteration can reach. PySCF's puts a localized hole's
    open orbital below doubly occupied ones, so it cannot reach that minimum.
    """

    def get_fock(
        self,
        h1e=None,
        s1e=None,
        vhf=None,
        dm=None,
        cycle=-1,
        diis=None,
        diis_start_cycle=None,
        level_shift_factor=None,
        damp_factor=None,
        fock_last=None,
    ):
        if h1e is None:
            h1e = self.get_hcore()
        if s1e is None:
            s1e = self.get_ovlp()
        if dm is None:
            dm = self.make_rdm1()
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        focka = h1e + vhf[0]
        fockb = h1e + vhf[1]
        fock = pyscf_rohf.get_roothaan_fock((focka, fockb), dm, s1e)
        po = (dm[0] - dm[1]) @ s1e
        fock = fock + po.T @ ((fockb - focka) * 0.25) @ po
        if diis and cycle >= self.diis_start_cycle:
            fock = diis.update(
                s1e, dm[0] + dm[1], fock, self, h1e, vhf, f_prev=fock_last
            )
        return lib.tag_array(fock, focka=focka, fockb=fockb)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        n_alpha, n_beta = self.nelec
        order = numpy.argsort(mo_energy, kind="stable")
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[order[:n_beta]] = 2
        mo_occ[order[n_beta:n_alpha]] = 1
        return mo_occ


def localized_hole_rohf(mol: gto.Mole, name: str) -> scf.rohf.ROHF:
    """The ROHF minimum of a localized hole, which PySCF's ROHF cannot reach.

    Converged by OpenBlockROHF to this file's tolerances, then checked against
    PySCF's own second-order solver from its default guess, which reaches the
    same energy but stalls at |g| ~ 1e-7, short of conv_tol_grad. Returned as
    a plain ROHF over those orbitals, canonicalized in PySCF's convention: each
    class's block of its Roothaan matrix diagonalized.
    """
    mf = OpenBlockROHF(mol)
    converge(mf, name)
    newton = scf.ROHF(mol).newton()
    newton.conv_tol = 1e-12
    newton.kernel()
    assert abs(newton.e_tot - mf.e_tot) < 1e-10, f"{name}: newton finds another state"
    out = scf.ROHF(mol)
    out.mo_occ = mf.mo_occ
    out.mo_energy, out.mo_coeff = out.canonicalize(mf.mo_coeff, mf.mo_occ)
    out.e_tot = mf.e_tot
    out.converged = True
    return out


def class_order(mo_occ: numpy.ndarray, mo_energy: numpy.ndarray) -> list[int]:
    """The orbitals doubly occupied, then singly, then empty, each ascending.

    The Scf step's columns, so its orbital energies compare by index. For an
    aufbau state that is PySCF's own order; a second-order solution's need not
    be, its open orbitals' energies falling among the doubly occupied ones.
    """
    return [
        int(i)
        for occ in (2, 1, 0)
        for i in sorted(numpy.flatnonzero(mo_occ == occ), key=lambda i: mo_energy[i])
    ]


def rohf(case: dict, localized_hole: bool = False) -> dict:
    """PySCF's converged, internally stable ROHF for one case."""
    mol = molecule(case)
    if localized_hole:
        mf = localized_hole_rohf(mol, case["name"])
    else:
        mf = scf.ROHF(mol)
        converge(mf, case["name"])
    # The default guess can land on a saddle; this file holds minima only.
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    n_alpha, n_beta = mol.nelec
    assert n_alpha - n_beta == case["spin"]
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "spin": case["spin"],
        "multiplicity": case["spin"] + 1,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [
            float(mf.mo_energy[i]) for i in class_order(mf.mo_occ, mf.mo_energy)
        ],
    }


def main() -> None:
    lib.num_threads(1)
    provenance = {
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "threads": 1,
        "command": "uv run --group tools tools/emit/scf_reference.py",
    }
    docs = {
        OUT: {
            "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-12, "
            "conv_tol_grad 1e-9) over this tree's data/basis files: total "
            "energy, nuclear repulsion and orbital energies (hartree); "
            "coordinates in bohr",
            "provenance": provenance,
            "cases": [rhf(case) for case in CASES],
        },
        ROHF_OUT: {
            "description": "PySCF ROHF (spherical, conv_tol 1e-12, conv_tol_grad "
            "1e-9, internally stable) over this tree's data/basis files: total "
            "energy, nuclear repulsion and orbital energies, the eigenvalues "
            "of PySCF's default Roothaan effective Fock (hartree); spin is 2S "
            "= n_alpha - n_beta; coordinates in bohr",
            "provenance": provenance,
            "cases": [rohf(case) for case in ROHF_CASES],
        },
        LOCALIZED_HOLE_OUT: {
            "description": "ROHF minima of a localized hole, which PySCF's ROHF "
            "does not converge to (spherical, conv_tol 1e-12, conv_tol_grad "
            "1e-9, internally stable, the energy PySCF's second-order solver's "
            "to 1e-10) over this tree's data/basis files: total energy, nuclear "
            "repulsion and orbital energies, PySCF's Roothaan effective Fock's "
            "eigenvalues in each occupation class, the doubly occupied, then the "
            "singly, then the empty, each ascending (hartree); spin is 2S = "
            "n_alpha - n_beta; coordinates in bohr",
            "provenance": provenance,
            "cases": [rohf(case, localized_hole=True) for case in LOCALIZED_HOLE_CASES],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, doc in docs.items():
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
