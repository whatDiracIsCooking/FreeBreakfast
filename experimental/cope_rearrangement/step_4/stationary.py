"""Step 4b: C2h stationary points, Hessians and mode-displaced walks.

Every run is one free-breakfast config under inputs/<name>/, run into
runs/<name>/. Phases, each resumable:

  prepare <name>   (PySCF)  writes inputs/<name>/: the geometry, the starting
                            orbitals over this tree's AOs, the config. What
                            each name is comes from RUNS below; a run that
                            starts where another ended reads that run's
                            files, so prepare it after that run finished.
  run <name> ...   (GPU)    the binary on each prepared run, its log to
                            runs/<name>/log.txt. Skips a run already done.
  analyse          (PySCF)  every finished run -> stationary.json, the final
                            geometries (geometries/*.xyz) and frequencies.csv.

The walks start from step 1's geometries with step 4a's projected orbitals
(energies.projected); a Hessian starts from its walk's final geometry and the
CASSCF orbitals the walk's outer [casscf] converged there; a displaced walk
from the TS Hessian's geometry plus or minus the imaginary mode, with the TS
CASSCF's orbitals.

No symmetry is imposed anywhere (study README, "Step 4 with free-breakfast").
The checks that stand in for it are computed here: per evaluation of a
symmetric walk, |R16 - R34| and the RMS distance of the geometry from its C2h
symmetrisation; at each stationary point, the a_u part of the gradient the
outer gradient step wrote there.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import energies

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
GEOMETRIES = HERE / "geometries"

BOHR = 0.529177210903  # Angstrom, CODATA 2018, this tree's
HARTREE_KCAL = 627.5094740631

# [casscf] for the walks: conv_tol_grad 1e-9 as the tree's own NEVPT2 and
# CASSCF optimisation tests converge it for Opt=Tight; for a Hessian, 1e-10,
# as the analytic Hessian's tests and examples (it assumes a stationary
# CASSCF).
CONV_TOL = 1e-12
CONV_TOL_GRAD_WALK = 1e-9
CONV_TOL_GRAD_HESSIAN = 1e-10

# Gaussian's Opt=Tight, the tightest set the tree's tests use; hartree/bohr
# and bohr.
OPT_TIGHT = {
    "max_force": 1.5e-5,
    "rms_force": 1.0e-5,
    "max_displacement": 6.0e-5,
    "rms_displacement": 4.0e-5,
}
MAX_STEPS = 300

# The displaced walks' start: the TS plus or minus the imaginary normal mode,
# scaled so its largest Cartesian component is this (bohr).
DISPLACEMENT = 0.2

HEAD = """\
# Step 4b, {name}: {what}. Written by stationary.py prepare.
[molecule]
xyz = "geometry.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 6
nelecas = 6
orbitals = "orbitals.npy"
conv_tol = {conv_tol}
conv_tol_grad = {conv_tol_grad}
write_npy = true              # casscf_orbitals.npy and casscf_ci.npy
"""
GRADIENT = {
    "pc": '[nevpt2_gradient]\nvariant = "pc"\n',
    "casscf": "[casscf_gradient]\n",
}
OPTIMIZE = """\
[optimize]                    # Gaussian's Opt=Tight
max_force = {max_force}
rms_force = {rms_force}
max_displacement = {max_displacement}
rms_displacement = {rms_displacement}
max_steps = {max_steps}
"""
HESSIAN = '[hessian]\nmethod = "analytic"\n'
# [memory] caps per run (per GPU, the context and libraries included). The
# run recorded here used 8 GiB and peaked at 8842 MiB on the 10 GB card,
# ~340 MiB short of what the card leaves free: the cap does not count ~2 GiB
# outside the pool (the driver's kernel stack reservation). A rerun is safer
# at 7 GiB, which still passes pre-flight (README, "Step 4b").
MEMORY = {"pc_ts_hessian_capped": "8GiB"}

# name -> (kind, method, start). kind: "walk" (an [optimize]), "hessian",
# "point" (one gradient). start: a step 1 R (float), or ("final", walk) for
# a walk's final geometry and orbitals, or ("mode", hessian, +1 | -1).
RUNS_SPEC = {
    "pc_point_1.95": ("point", "pc", 1.95),
    "pc_ts": ("walk", "pc", 1.95),
    "casscf_1.65": ("walk", "casscf", 1.65),
    "casscf_2.30": ("walk", "casscf", 2.30),
    "pc_ts_hessian": ("hessian", "pc", ("final", "pc_ts")),
    # pc_ts_hessian ran out of device memory on the 10 GB RTX 3080; the same
    # under a [memory] cap, which the Hessian's chunks size themselves to.
    "pc_ts_hessian_capped": ("hessian", "pc", ("final", "pc_ts")),
    "casscf_1.65_hessian": ("hessian", "casscf", ("final", "casscf_1.65")),
    # casscf_2.30's walk left C2h (its a_u part grew ~2x a step from noise)
    # and ended at 1,5-hexadiene; this one restarts it from its evaluation
    # 20, symmetrised by hand, with step 1's R = 2.30 orbitals (README).
    "casscf_2.30_restart": ("walk", "casscf", ("symmetrised", "casscf_2.30", 20, 2.30)),
    "casscf_2.30_hessian": ("hessian", "casscf", ("final", "casscf_2.30_restart")),
    "pc_plus": ("walk", "pc", ("mode", "pc_ts_hessian_capped", +1)),
    "pc_minus": ("walk", "pc", ("mode", "pc_ts_hessian_capped", -1)),
}


# --- geometry ------------------------------------------------------------


def read_xyz_frames(path: Path) -> list[tuple[str, list]]:
    """Every frame of an XYZ file: (comment, [(symbol, (x, y, z)), ...]) in
    Angstrom."""
    lines = path.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        comment = lines[i + 1]
        atoms = []
        for line in lines[i + 2 : i + 2 + n]:
            s, x, y, z = line.split()[:4]
            atoms.append((s, (float(x), float(y), float(z))))
        frames.append((comment, atoms))
        i += n + 2
    return frames


def coords_bohr(atoms: list):
    import numpy

    return numpy.array([xyz for _, xyz in atoms]) / BOHR


def distances(atoms: list) -> tuple[float, float]:
    """R16 and R34 (Angstrom): atoms 0-5 and 2-3."""
    import numpy

    x = numpy.array([xyz for _, xyz in atoms])
    return (
        float(numpy.linalg.norm(x[0] - x[5])),
        float(numpy.linalg.norm(x[2] - x[3])),
    )


# C2h characters over (E, C2, i, sigma_h).
CHARACTERS = {
    "ag": {"E": 1, "C2": 1, "i": 1, "sigma": 1},
    "bg": {"E": 1, "C2": -1, "i": 1, "sigma": -1},
    "au": {"E": 1, "C2": 1, "i": -1, "sigma": -1},
    "bu": {"E": 1, "C2": -1, "i": -1, "sigma": 1},
}


def c2h_operations(reference_bohr):
    """step 2's au.operations at a looser tolerance: the walks move the atoms
    by less than this off C2h, and the permutation is all that is taken."""
    sys.path.insert(0, str(STUDY / "step_2"))
    import au

    return au.operations(reference_bohr, tol=1e-3)


def representation(ops, natm: int) -> dict:
    """Each operation's 3N x 3N matrix, D(g) x moving atom a to perm[a]."""
    import numpy

    mats = {}
    for kind, r, perm in ops:
        d = numpy.zeros((3 * natm, 3 * natm))
        for a in range(natm):
            b = perm[a]
            d[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = r
        mats[kind] = d
    return mats


def projectors(ops, natm: int) -> dict:
    """The four irreps' projectors on Cartesian displacements (or on any
    vector that transforms as one, the gradient)."""
    mats = representation(ops, natm)
    return {
        irrep: sum(chi[k] * mats[k] for k in mats) / 4.0
        for irrep, chi in CHARACTERS.items()
    }


def c2h_deviation(x_bohr, ops) -> float:
    """RMS distance (bohr, per coordinate) of x from its C2h symmetrisation,
    (1/4) sum_g D(g) x. Zero for an exactly C2h geometry in this frame."""
    import numpy

    mats = representation(ops, len(x_bohr))
    v = x_bohr.ravel()
    sym = sum(m @ v for m in mats.values()) / 4.0
    return float(numpy.sqrt(numpy.mean((v - sym) ** 2)))


def deviation_by_irrep(x_bohr, ops) -> dict:
    """The RMS (bohr, per coordinate) of x's b_g, a_u and b_u parts: which
    symmetry a departure from C2h has. They add in quadrature to
    c2h_deviation."""
    import numpy

    proj = projectors(ops, len(x_bohr))
    v = x_bohr.ravel()
    return {
        irrep: float(numpy.sqrt(numpy.mean((proj[irrep] @ v) ** 2)))
        for irrep in ("bg", "au", "bu")
    }


def s_direction(x_bohr):
    """Unit Cartesian direction of R16 - R34 (step 2's s_bvector)."""
    import numpy

    sys.path.insert(0, str(STUDY / "step_2"))
    import au

    b = au.s_bvector(x_bohr)
    return b / numpy.linalg.norm(b)


# --- prepare -------------------------------------------------------------


def start_of(name: str) -> tuple[list, object, str]:
    """(atoms in Angstrom, orbitals over this tree's AOs, a description)."""
    import numpy

    kind, method, start = RUNS_SPEC[name]
    if isinstance(start, float):
        atoms, c_tree, _ = energies.projected(start)
        return atoms, c_tree, f"step 1's R = {start:.2f} A, its orbitals projected"
    if start[0] == "final":
        walk = RUNS / start[1]
        (_, atoms), *_ = read_xyz_frames(walk / "final.xyz")
        c = numpy.load(walk / "casscf_orbitals.npy")
        return atoms, c, f"{start[1]}'s final geometry and CASSCF orbitals"
    if start[0] == "symmetrised":
        _, walk, frame, r = start
        _, atoms = read_xyz_frames(RUNS / walk / "optimize.xyz")[frame]
        (_, ref), *_ = read_xyz_frames(INPUTS / walk / "geometry.xyz")
        x = coords_bohr(atoms)
        mats = representation(c2h_operations(coords_bohr(ref)), len(x))
        sym = (sum(m @ x.ravel() for m in mats.values()) / 4.0).reshape(-1, 3)
        atoms = [
            (s, tuple(float(v) for v in xyz * BOHR))
            for (s, _), xyz in zip(atoms, sym, strict=True)
        ]
        _, c_tree, _ = energies.projected(r)
        what = (
            f"{walk}'s evaluation {frame} symmetrised to C2h, step 1's "
            f"R = {r:.2f} A orbitals projected"
        )
        return atoms, c_tree, what
    _, hessian, sign = start
    h = json.loads((RUNS / hessian / "hessian.json").read_text())
    (_, atoms), *_ = read_xyz_frames(INPUTS / hessian / "geometry.xyz")
    k = int(numpy.argmin(h["frequencies"]))
    assert h["frequencies"][k] < 0, f"{hessian} has no imaginary frequency"
    mode = numpy.array(h["modes"][k]).reshape(-1, 3)
    # A fixed phase: the mode's R16 - R34 component positive, so "plus"
    # lengthens R16 against R34.
    x = coords_bohr(atoms)
    if s_direction(x) @ mode.ravel() < 0:
        mode = -mode
    mode *= sign * DISPLACEMENT / abs(mode).max()
    moved = (x + mode) * BOHR
    atoms = [
        (s, tuple(float(v) for v in xyz))
        for (s, _), xyz in zip(atoms, moved, strict=True)
    ]
    c = numpy.load(RUNS / hessian / "casscf_orbitals.npy")
    what = (
        f"{hessian}'s geometry {'+' if sign > 0 else '-'} its imaginary mode "
        f"({h['frequencies'][k]:.1f} cm-1), largest component {DISPLACEMENT} bohr"
    )
    return atoms, c, what


def prepare(name: str) -> None:
    import numpy

    kind, method, _ = RUNS_SPEC[name]
    atoms, c, what = start_of(name)
    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    energies.write_xyz(out / "geometry.xyz", atoms, f"{name}: {what}")
    numpy.save(out / "orbitals.npy", numpy.asfortranarray(c), allow_pickle=False)
    grad = CONV_TOL_GRAD_HESSIAN if kind != "walk" else CONV_TOL_GRAD_WALK
    text = HEAD.format(
        name=name,
        what=f"{method} {kind} from {what}",
        conv_tol=CONV_TOL,
        conv_tol_grad=grad,
    )
    text += GRADIENT[method]
    if kind == "walk":
        text += OPTIMIZE.format(**OPT_TIGHT, max_steps=MAX_STEPS)
    elif kind == "hessian":
        text += HESSIAN
    if name in MEMORY:
        text += f'[memory]\nmax_device_mem_per_gpu = "{MEMORY[name]}"\n'
    (out / "config.toml").write_text(text)
    r16, r34 = distances(atoms)
    print(f"{name}: prepared, R16 {r16:.4f} R34 {r34:.4f} A ({what})")


# --- run -----------------------------------------------------------------


def done(name: str) -> bool:
    out = RUNS / name
    kind, method, _ = RUNS_SPEC[name]
    gradient = "nevpt2_gradient.json" if method == "pc" else "casscf_gradient.json"
    if kind == "hessian":
        return (out / "hessian.json").exists()
    return (out / gradient).exists()


def run(names: list[str], binary: Path) -> None:
    import time

    for name in names:
        if done(name):
            print(f"{name}: already run")
            continue
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        inp = INPUTS / name
        cmd = [str(binary), "-i", str(inp), "-o", str(out), str(inp / "config.toml")]
        t0 = time.monotonic()
        with (out / "log.txt").open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        wall = time.monotonic() - t0
        (out / "wall.txt").write_text(f"{wall:.1f}\n")
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        print(f"{name}: {status} in {wall:.0f} s", flush=True)


# --- analyse -------------------------------------------------------------


def natural_occupations(run_dir: Path, atoms: list) -> dict:
    """The active natural occupations of the CASSCF the run wrote, and, where
    the geometry is C2h to 1e-4 bohr, each natural orbital's irrep."""
    import numpy
    from pyscf import gto, symm
    from pyscf.fci import direct_spin1

    ci = numpy.load(run_dir / "casscf_ci.npy")
    c = numpy.load(run_dir / "casscf_orbitals.npy")
    ci = ci.reshape(20, 20)
    rdm1 = direct_spin1.make_rdm1(ci, 6, (3, 3))
    occ, u = numpy.linalg.eigh(rdm1)
    order = numpy.argsort(-occ)
    occ, u = occ[order], u[:, order]
    natorbs = c[:, 20:26] @ u
    out = {"occupations": [float(x) for x in occ], "irreps": None}

    x = coords_bohr(atoms)
    try:
        ops = c2h_operations(x)
    except AssertionError:  # not C2h even to 1e-3 bohr: an end point
        return out
    if c2h_deviation(x, ops) > 1e-4:
        return out
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    from scf_reference import pyscf_basis

    basis = pyscf_basis("cc-pVDZ.json", {"C", "H"})
    # PySCF's symmetry detection wants C2h to ~1e-5 bohr, and a walk's
    # geometry is off it by up to ~1e-5: label at the symmetrised geometry,
    # within 1e-4 bohr of the orbitals' own.
    mats = representation(ops, len(x))
    sym = (sum(m @ x.ravel() for m in mats.values()) / 4.0).reshape(-1, 3)
    sym_atoms = [(s, tuple(v)) for (s, _), v in zip(atoms, sym, strict=True)]
    mol = gto.M(atom=sym_atoms, basis=basis, unit="Bohr", symmetry="C2h", verbose=0)
    if not numpy.allclose(mol.atom_coords(), sym, atol=1e-6):
        return out  # PySCF moved the frame; no labels rather than wrong ones
    plain = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    d = energies.contraction_norms(plain, basis)
    c_json = d[:, None] * natorbs  # over PySCF's normalised AOs
    s = plain.intor("int1e_ovlp")
    labels = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c_json, s=s)
    out["irreps"] = [str(lab).lower() for lab in labels]
    return out


def gradient_symmetry(run_dir: Path, method: str, atoms: list) -> dict:
    """The outer gradient step's vector at the run's geometry, split by
    irrep. A totally symmetric (a_g) gradient is what C2h allows; the a_u
    part is the reaction mode's, noise if symmetry held."""
    import numpy

    name = "nevpt2_gradient.json" if method == "pc" else "casscf_gradient.json"
    g = json.loads((run_dir / name).read_text())
    vec = numpy.array(g["total"]).ravel()
    x = coords_bohr(atoms)
    proj = projectors(c2h_operations(x), len(atoms))
    return {
        "energy": g["energy"],
        "norm": float(numpy.linalg.norm(vec)),
        "max_abs": float(abs(vec).max()),
        "by_irrep_norm": {
            k: float(numpy.linalg.norm(p @ vec)) for k, p in proj.items()
        },
        "s_component": float(s_direction(x) @ vec),
        "c2h_deviation_bohr": c2h_deviation(x, c2h_operations(x)),
    }


def gradient_from_geometry(run_dir: Path, method: str, h: dict, atoms: list):
    """Per non-totally-symmetric irrep: the gradient's part g, the geometry's
    displacement from C2h d (bohr), H d, and |g - H d|. Where g = H d to well
    below |g|, the gradient's symmetry breaking is the geometry's alone: the
    wavefunction kept C2h."""
    import numpy

    name = "nevpt2_gradient.json" if method == "pc" else "casscf_gradient.json"
    g = numpy.array(json.loads((run_dir / name).read_text())["total"]).ravel()
    x = coords_bohr(atoms)
    proj = projectors(c2h_operations(x), len(atoms))
    hess = numpy.array(h["hessian"])
    out = {}
    for irrep in ("bg", "au", "bu"):
        d = proj[irrep] @ x.ravel()
        hd = proj[irrep] @ (hess @ d)
        gi = proj[irrep] @ g
        out[irrep] = {
            "d_norm": float(numpy.linalg.norm(d)),
            "g_norm": float(numpy.linalg.norm(gi)),
            "hd_norm": float(numpy.linalg.norm(hd)),
            "g_minus_hd_norm": float(numpy.linalg.norm(gi - hd)),
        }
    return out


def walk_trajectory(run_dir: Path, reference: list) -> list[dict]:
    """Per evaluated geometry: energy, accepted, R16, R34, |R16 - R34|, and
    the RMS distance from C2h under `reference`'s operations."""
    frames = read_xyz_frames(run_dir / "optimize.xyz")
    ops = c2h_operations(coords_bohr(reference))
    hist = json.loads((run_dir / "optimize.json").read_text())["iterations"]
    rows = []
    for k, (comment, atoms) in enumerate(frames):
        r16, r34 = distances(atoms)
        it = hist[k] if k < len(hist) else {}
        rows.append(
            {
                "evaluation": k,
                "energy": it.get("energy"),
                "accepted": it.get("accepted"),
                "max_force": it.get("max_force"),
                "rms_force": it.get("rms_force"),
                "r16": r16,
                "r34": r34,
                "abs_r16_minus_r34": abs(r16 - r34),
                "c2h_deviation_bohr": c2h_deviation(coords_bohr(atoms), ops),
                "deviation_by_irrep_bohr": deviation_by_irrep(coords_bohr(atoms), ops),
                "comment": comment,
            }
        )
    return rows


def mode_symmetry(hessian: dict, atoms: list, count: int = 12) -> list[dict]:
    """The lowest `count` modes: frequency, each irrep's weight in the
    Cartesian displacement (the four sum to 1), and |cos| with R16 - R34."""
    import numpy

    x = coords_bohr(atoms)
    proj = projectors(c2h_operations(x), len(atoms))
    b = s_direction(x)
    rows = []
    order = numpy.argsort(hessian["frequencies"])[:count]
    for k in order:
        v = numpy.array(hessian["modes"][k])
        v = v / numpy.linalg.norm(v)
        weights = {irr: float(numpy.linalg.norm(p @ v) ** 2) for irr, p in proj.items()}
        rows.append(
            {
                "frequency": float(hessian["frequencies"][k]),
                "irrep": max(weights, key=weights.get),
                "weights": weights,
                "s_overlap": float(abs(b @ v)),
            }
        )
    return rows


def wall(run_dir: Path) -> float | None:
    p = run_dir / "wall.txt"
    return float(p.read_text()) if p.exists() else None


def analyse() -> None:
    GEOMETRIES.mkdir(exist_ok=True)
    result = {
        "settings": {
            "conv_tol": CONV_TOL,
            "conv_tol_grad_walk": CONV_TOL_GRAD_WALK,
            "conv_tol_grad_hessian": CONV_TOL_GRAD_HESSIAN,
            "optimize": {**OPT_TIGHT, "max_steps": MAX_STEPS},
            "displacement_bohr": DISPLACEMENT,
        },
        "runs": {},
    }
    freq_rows = []
    for name, (kind, method, start) in RUNS_SPEC.items():
        run_dir = RUNS / name
        if not (INPUTS / name / "geometry.xyz").exists() or not run_dir.exists():
            continue
        (_, start_atoms), *_ = read_xyz_frames(INPUTS / name / "geometry.xyz")
        entry = {"kind": kind, "method": method, "wall_s": wall(run_dir)}
        if kind == "walk":
            hist = json.loads((run_dir / "optimize.json").read_text())
            entry["error"] = hist.get("error")
            entry["evaluations"] = len(hist["iterations"])
            symmetric = isinstance(start, float) or start[0] == "symmetrised"
            ref = start_atoms if symmetric else None
            if symmetric:
                traj = walk_trajectory(run_dir, ref)
                entry["trajectory"] = traj
                entry["max_abs_r16_minus_r34"] = max(
                    t["abs_r16_minus_r34"] for t in traj
                )
                entry["max_c2h_deviation_bohr"] = max(
                    t["c2h_deviation_bohr"] for t in traj
                )
            else:
                frames = read_xyz_frames(run_dir / "optimize.xyz")
                entry["trajectory"] = [
                    {
                        "evaluation": k,
                        "energy": (
                            hist["iterations"][k]["energy"]
                            if k < len(hist["iterations"])
                            else None
                        ),
                        "r16": distances(a)[0],
                        "r34": distances(a)[1],
                    }
                    for k, (_, a) in enumerate(frames)
                ]
            if (run_dir / "final.xyz").exists():
                (_, final), *_ = read_xyz_frames(run_dir / "final.xyz")
                r16, r34 = distances(final)
                entry["final"] = {
                    "energy": hist["energy"],
                    "r16": r16,
                    "r34": r34,
                    "last": hist["iterations"][-1],
                }
                energies.write_xyz(
                    GEOMETRIES / f"{name}.xyz",
                    final,
                    f"{name}: {method} E {hist['energy']:.10f}",
                )
                left = symmetric and traj[-1]["c2h_deviation_bohr"] > 1e-3
                entry["left_c2h"] = left
                if symmetric and not left:
                    entry["final"]["gradient"] = gradient_symmetry(
                        run_dir, method, final
                    )
                entry["final"]["active"] = natural_occupations(run_dir, final)
        elif kind == "point":
            entry["gradient"] = gradient_symmetry(run_dir, method, start_atoms)
            entry["active"] = natural_occupations(run_dir, start_atoms)
        elif kind == "hessian" and (run_dir / "hessian.json").exists():
            h = json.loads((run_dir / "hessian.json").read_text())
            freqs = h["frequencies"]
            r16, r34 = distances(start_atoms)
            modes = mode_symmetry(h, start_atoms)
            entry.update(
                {
                    "r16": r16,
                    "r34": r34,
                    "asymmetry": h.get("asymmetry"),
                    "translational_sum": h.get("translational_sum"),
                    "n_imaginary": sum(f < 0 for f in freqs),
                    "frequencies": freqs,
                    "lowest_modes": modes,
                    "gradient": gradient_symmetry(run_dir, method, start_atoms),
                    "gradient_from_geometry": gradient_from_geometry(
                        run_dir, method, h, start_atoms
                    ),
                    "active": natural_occupations(run_dir, start_atoms),
                }
            )
            for k, f in enumerate(freqs):
                freq_rows.append({"run": name, "mode": k, "frequency_cm1": f})
        result["runs"][name] = entry

    runs = result["runs"]
    if "pc_ts_hessian_capped" in runs and all(
        "final" in runs.get(n, {}) for n in ("pc_plus", "pc_minus")
    ):
        e_ts = runs["pc_ts_hessian_capped"]["gradient"]["energy"]
        result["barrier"] = {
            side: {
                "eh": e_ts - runs[side]["final"]["energy"],
                "kcal_mol": (e_ts - runs[side]["final"]["energy"]) * HARTREE_KCAL,
            }
            for side in ("pc_plus", "pc_minus")
        }
        result["mirror"] = mirror_check()
    (HERE / "stationary.json").write_text(json.dumps(result, indent=1) + "\n")
    if freq_rows:
        with (HERE / "frequencies.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(freq_rows[0]))
            w.writeheader()
            w.writerows(freq_rows)
    summarise(result)


def mirror_check() -> dict:
    """Whether pc_plus's end is pc_minus's under the TS's inversion (and
    sigma_h): the RMS distance (bohr) between one end and the other's image,
    after the TS frame's operation. Both walks project out rigid-body motion,
    so the frames are comparable without alignment only if neither rotated;
    a Kabsch alignment is applied too, and both numbers kept."""
    import numpy

    (_, ts), *_ = read_xyz_frames(INPUTS / "pc_ts_hessian_capped" / "geometry.xyz")
    ops = c2h_operations(coords_bohr(ts))
    (_, a), *_ = read_xyz_frames(RUNS / "pc_plus" / "final.xyz")
    (_, b), *_ = read_xyz_frames(RUNS / "pc_minus" / "final.xyz")
    xa, xb = coords_bohr(a), coords_bohr(b)
    out = {}
    for kind, r, perm in ops:
        if kind not in ("i", "sigma"):
            continue
        image = numpy.empty_like(xb)
        image[perm] = xb @ r.T
        raw = float(numpy.sqrt(numpy.mean((xa - image) ** 2)))
        pa = xa - xa.mean(axis=0)
        pb = image - image.mean(axis=0)
        u, _, vt = numpy.linalg.svd(pb.T @ pa)
        dd = numpy.sign(numpy.linalg.det(u @ vt))
        rot = u @ numpy.diag([1.0, 1.0, dd]) @ vt
        aligned = float(numpy.sqrt(numpy.mean((pa - pb @ rot) ** 2)))
        out[kind] = {"rms_bohr": raw, "rms_aligned_bohr": aligned}
    return out


def summarise(result: dict) -> None:
    for name, e in result["runs"].items():
        if e["kind"] == "walk":
            fin = e.get("final")
            line = f"{name}: {e['evaluations']} evaluations"
            if e.get("error"):
                line += f", ERROR {e['error']}"
            if fin:
                line += (
                    f", E {fin['energy']:.10f}, R16 {fin['r16']:.4f} R34 "
                    f"{fin['r34']:.4f}"
                )
            if "max_c2h_deviation_bohr" in e:
                line += (
                    f", max|R16-R34| {e['max_abs_r16_minus_r34']:.2e} A,"
                    f" max C2h dev {e['max_c2h_deviation_bohr']:.2e} bohr"
                )
            print(line)
            if fin and "gradient" in fin:
                g = fin["gradient"]
                print(f"   gradient by irrep {g['by_irrep_norm']}")
            if fin:
                print(f"   active {fin['active']}")
        elif e["kind"] == "hessian" and "frequencies" in e:
            print(
                f"{name}: {e['n_imaginary']} imaginary, lowest "
                f"{[round(f, 1) for f in e['frequencies'][:4]]}"
            )
            for m in e["lowest_modes"][:3]:
                head = f"   {m['frequency']:.1f} {m['irrep']}"
                print(f"{head} {m['weights']} s {m['s_overlap']:.3f}")
            print(f"   gradient by irrep {e['gradient']['by_irrep_norm']}")
            print(f"   active {e['active']}")
        elif e["kind"] == "point":
            print(f"{name}: gradient {e['gradient']}")
            print(f"   active {e['active']}")
    if "barrier" in result:
        print(f"barrier {result['barrier']}")
        print(f"mirror {result['mirror']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["prepare", "run", "analyse"])
    parser.add_argument("names", nargs="*", help="runs (RUNS_SPEC's names)")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "src" / "free-breakfast"
    )
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in RUNS_SPEC]
    if unknown:
        parser.error(f"unknown runs {unknown}; known: {list(RUNS_SPEC)}")
    if args.phase == "prepare":
        for name in args.names:
            prepare(name)
    elif args.phase == "run":
        run(args.names, args.binary)
    else:
        analyse()


if __name__ == "__main__":
    main()
