"""Step 6's analyse phase (experiment.py analyse): every finished run ->
results.json, frequencies.csv and geometries/*.xyz, and a printed summary.

What it computes, per part of the step (README):
  1. at each Hessian: our own harmonic analysis against hessian.json's, the
     six rigid-body eigenvalues, the imaginary count; pc_plus against
     pc_minus, mode by mode.
  2. the conformer walks: energy, dihedrals, point group; distinct minima.
  3. ΔH‡, ΔS‡, ΔG‡ of the chair TS against the reference, at the
     experimental temperature and at 298.15 K.
  4. the boat walk's C2v drift, its Hessian's imaginary mode and irrep,
     ΔΔH‡(boat - chair).
"""

from __future__ import annotations

import csv
import itertools
import json

import experiment as ex
import numpy
import thermo

s4 = ex.s4
RUNS, INPUTS = ex.RUNS, ex.INPUTS

# Temperatures (K) for the comparisons (README "The experimental
# references"). Doering, Toscano & Beasley measured at four temperatures,
# 207-258 °C as the abstract gives them (480-531 K; the paper itself could
# not be read); 505 K is their middle. Goldstein & Benzon's boat values
# come from bicyclo[2.2.0]hexane's inversion at a temperature not read
# either, so the boat-chair gap is compared at the same 505 K.
T_EXPERIMENT = 505.0
T_ROOM = 298.15

# Point-group candidates in the principal-axes frame: the eight diagonal
# sign matrices (E, the three C2, i, the three sigma). Enough for the groups
# met here (C1, Cs, C2, Ci, C2v, C2h).
SIGNS = [numpy.diag(s) for s in itertools.product((1.0, -1.0), repeat=3)]


def load_atoms(path):
    (_, atoms), *_ = s4.read_xyz_frames(path)
    return atoms


def principal_frame(x, m):
    xc = x - (m @ x) / m.sum()
    i = numpy.einsum("a,ax,ay->xy", m, xc, xc)
    _, axes = numpy.linalg.eigh(numpy.eye(3) * numpy.trace(i) - i)
    return xc @ axes


def point_group(atoms, tol=0.02):
    """(name, operations as (r, perm), max displacement of the worst kept
    operation, bohr). An operation is kept when every atom lands within
    `tol` bohr of an atom of its own element. 0.02: conf_a_pm's Ci holds to
    0.011 bohr only, its walk having stopped (Opt=Tight met) on a flat
    Ci-breaking direction; every other structure here holds its group to
    0.005 bohr or better, or misses any group by more than 0.3."""
    symbols = [s for s, _ in atoms]
    m = thermo.masses(symbols)
    x = principal_frame(s4.coords_bohr(atoms), m)
    ops, worst = [], 0.0
    for r in SIGNS:
        moved = x @ r.T
        d = numpy.linalg.norm(moved[:, None] - x[None], axis=-1)
        same = numpy.array([[a == b for b in symbols] for a in symbols])
        d[~same] = numpy.inf
        perm = d.argmin(axis=1)
        dev = d[numpy.arange(len(x)), perm].max()
        if dev < tol and len(set(perm)) == len(x):
            ops.append((r, perm))
            worst = max(worst, float(dev))
    dets = [numpy.linalg.det(r) for r, _ in ops]
    proper = sum(d > 0 for d in dets)
    has_i = any(numpy.allclose(r, -numpy.eye(3)) for r, _ in ops)
    names = {(1, 1): "C1", (2, 2): "C2", (4, 2): None, (2, 1): None}
    name = names.get((len(ops), proper))
    if name is None:
        if len(ops) == 2:
            name = "Ci" if has_i else "Cs"
        else:
            name = "C2h" if has_i else ("C2v" if proper == 2 else "D2")
    return name, ops, worst


def bonds(atoms, cutoff=1.75):
    """Pairs (i, j) closer than `cutoff` A, C-C up to 1.75, C-H up to 1.2."""
    x = numpy.array([v for _, v in atoms])
    out = []
    for i, j in itertools.combinations(range(len(atoms)), 2):
        d = numpy.linalg.norm(x[i] - x[j])
        kinds = {atoms[i][0], atoms[j][0]}
        if (kinds == {"C"} and d < cutoff) or (kinds == {"C", "H"} and d < 1.2):
            out.append((i, j))
    return out


def check_connectivity(atoms, c34_bonded=True) -> dict:
    """The study's numbering, by connectivity: the hydrogens H_ON says sit
    on each carbon, and which interallylic bond is formed. Returns the
    carbon chain found and asserts the hydrogens."""
    b = set(bonds(atoms))
    for c, hs in ex.H_ON.items():
        for h in hs:
            assert (c, h) in b, f"H{h} not on C{c + 1}"
    chain = sorted((i, j) for i, j in b if i < 6 and j < 6)
    assert ((2, 3) in chain) == c34_bonded and ((0, 5) in chain) != c34_bonded, chain
    return {"cc_bonds": chain}


def active_character(run_dir, atoms) -> dict:
    """The CASSCF the run wrote: active natural occupations (descending)
    and, per natural orbital, its Mulliken population (this tree's overlap)
    on the two vinyl groups' carbons C1, C2, C5, C6 and on C3, C4. At a
    hexadiene with C3-C4 bonded, two pi bonds and a sigma bond show as two
    bonding/antibonding pairs on the vinyls (in- and out-of-phase
    combinations of the two pi bonds where the conformer is C2 or Ci) and
    one pair on C3-C4."""
    from pyscf.fci import direct_spin1

    ci = numpy.load(run_dir / "casscf_ci.npy").reshape(20, 20)
    c = numpy.load(run_dir / "casscf_orbitals.npy")
    occ, u = numpy.linalg.eigh(direct_spin1.make_rdm1(ci, 6, (3, 3)))
    order_ = numpy.argsort(-occ)
    occ, u = occ[order_], u[:, order_]
    nat = c[:, 20:26] @ u
    _, tree, d = ex.energies.tree_mols(atoms)
    s = d[:, None] * tree.intor("int1e_ovlp") * d[None, :]
    labels = tree.ao_labels(fmt=False)
    atom_of = numpy.array([lab[0] for lab in labels])
    rows = []
    for k in range(6):
        v = nat[:, k]
        pop = v * (s @ v)
        per_atom = numpy.array([pop[atom_of == a].sum() for a in range(len(atoms))])
        rows.append(
            {
                "occupation": float(occ[k]),
                "vinyl_c1c2c5c6": float(per_atom[[0, 1, 4, 5]].sum()),
                "c3c4": float(per_atom[[2, 3]].sum()),
            }
        )
    return {"occupations": [float(o) for o in occ], "natural_orbitals": rows}


def hessian_entry(name) -> dict | None:
    run = RUNS / name
    if not (run / "hessian.json").exists():
        return None
    h = json.loads((run / "hessian.json").read_text())
    atoms = load_atoms(INPUTS / name / "geometry.xyz")
    symbols = [s for s, _ in atoms]
    x = s4.coords_bohr(atoms)
    hess = numpy.array(h["hessian"])
    # Our analysis at the tree's masses reproduces its frequencies.
    mine_avg, _ = thermo.harmonic(
        hess, x, thermo.masses(symbols, table=thermo.MASS_AVERAGED)
    )
    freqs, modes = thermo.harmonic(hess, x, thermo.masses(symbols))
    g = json.loads((run / "nevpt2_gradient.json").read_text())
    pg, _, pg_dev = point_group(atoms)
    mem = (
        json.loads((run / "memory.json").read_text())
        if (run / "memory.json").exists()
        else {}
    )
    return {
        "energy": g["energy"],
        "gradient_max": float(numpy.abs(numpy.array(g["total"])).max()),
        "r16": s4.distances(atoms)[0],
        "r34": s4.distances(atoms)[1],
        "point_group": pg,
        "point_group_max_dev_bohr": pg_dev,
        "wall_s": s4.wall(run),
        "memory": mem,
        "method": h.get("method"),
        "ladder": ladder_summary(h),
        "min_inactive_gap": g.get("min_inactive_gap"),
        "max_zc": g.get("max_zc"),
        "asymmetry": h.get("asymmetry"),
        "translational_sum": h.get("translational_sum"),
        "frequencies_tree": h["frequencies"],
        "max_abs_mine_minus_tree_cm1": float(
            numpy.abs(mine_avg - numpy.array(h["frequencies"])).max()
        ),
        "frequencies": [float(f) for f in freqs],
        "n_imaginary": int((freqs < 0).sum()),
        "active": active_character(run, atoms),
        "rigid_body_cm1": [
            float(v)
            for v in thermo.rigid_body_eigenvalues(hess, x, thermo.masses(symbols))
        ],
        "_hessian": hess,
        "_x": x,
        "_symbols": symbols,
        "_modes": modes,
        "_atoms": atoms,
    }


def ladder_summary(h: dict) -> dict | None:
    """The numerical Hessian's ladder, if any: the noise floor and how
    many elements were judged and settled (README of pipeline/hessian,
    "The ladder and the bilinearity check")."""
    lad = h.get("ladder")
    if not lad:
        return None
    out = {k: v for k, v in lad.items() if not isinstance(v, list)}
    ratios = lad.get("ratios")
    if ratios is not None:
        flat = numpy.array(
            [numpy.nan if r is None else r for r in numpy.ravel(ratios)], dtype=float
        )
        judged = flat[~numpy.isnan(flat)]
        out["judged"] = int(judged.size)
        out["elements"] = int(flat.size)
        if judged.size:
            out["ratio_min"] = float(judged.min())
            out["ratio_max"] = float(judged.max())
            out["unsettled"] = int(((judged <= 12.8) | (judged >= 19.2)).sum())
    return out


def refusal(name) -> str | None:
    """The [error] line of a run that wrote no hessian.json, cut short."""
    for line in (RUNS / name / "log.txt").read_text().splitlines():
        if "[error]" in line:
            return line[line.index("[error]") :][:400]
    return None


def log_value(run_dir, prefix: str) -> float | None:
    """The first number after `prefix` at the start of a log line."""
    for line in (run_dir / "log.txt").read_text().splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :].split()[0])
    return None


def energy_entry(name) -> dict | None:
    run = RUNS / name
    e = ex.pc_energy(run)
    if e is None:
        return None
    atoms = load_atoms(INPUTS / name / "geometry.xyz")
    r16, r34 = s4.distances(atoms)
    return {
        "pc_energy": e,
        "casscf_energy": log_value(run, "CASSCF energy:"),
        "r16": r16,
        "r34": r34,
        "wall_s": s4.wall(run),
        "active": active_character(run, atoms),
    }


def walk_entry(name) -> dict | None:
    run = RUNS / name
    if not (run / "optimize.json").exists():
        return None
    hist = json.loads((run / "optimize.json").read_text())
    entry = {
        "evaluations": len(hist["iterations"]),
        "error": hist.get("error"),
        "wall_s": s4.wall(run),
    }
    if (run / "final.xyz").exists():
        atoms = load_atoms(run / "final.xyz")
        x = s4.coords_bohr(atoms)
        r16, r34 = s4.distances(atoms)
        pg, _, dev = point_group(atoms)
        entry.update(
            {
                "energy": hist["energy"],
                "r16": r16,
                "r34": r34,
                "last": hist["iterations"][-1],
                "point_group": pg,
                "point_group_max_dev_bohr": dev,
            }
        )
        entry["active"] = active_character(run, atoms)
        if r34 < 1.7:
            entry["dihedrals"] = [round(v, 2) for v in ex.dihedrals(x)]
        ex.GEOMETRIES.mkdir(exist_ok=True)
        ex.energies.write_xyz(
            ex.GEOMETRIES / f"{name}.xyz", atoms, f"{name}: pc E {hist['energy']:.10f}"
        )
    return entry


def modes_compare(a: dict, b: dict) -> dict:
    fa, fb = numpy.array(a["frequencies"]), numpy.array(b["frequencies"])
    d = numpy.abs(fa - fb)
    return {
        "max_abs_cm1": float(d.max()),
        "median_abs_cm1": float(numpy.median(d)),
        "at_mode": int(d.argmax()),
        "frequency_there": float(fa[d.argmax()]),
    }


def species(entry, temperature) -> dict:
    """RRHO for one Hessian entry at one temperature, sigma 1 (the
    structure counting is the ensemble's g)."""
    m = thermo.masses(entry["_symbols"])
    f, _ = thermo.harmonic(entry["_hessian"], entry["_x"], m)
    return thermo.rrho(entry["energy"], f, entry["_x"], m, temperature)


def order(pg: str) -> int:
    return {"C1": 1, "Cs": 2, "Ci": 2, "C2": 2, "C2v": 4, "C2h": 4, "D2": 4}[pg]


def members(entries, temperature):
    """Ensemble members (rrho at sigma 1, 2/|G|). 2/|G| is the number of
    optical isomers over the rotational symmetry number, for every group
    here: C1 2/1, C2 2/2, Ci and Cs 1/1, C2h and C2v 1/2."""
    return [(species(e, temperature), 2.0 / order(e["point_group"])) for e in entries]


def activation_at(ts, refs, temperature):
    t_ens = thermo.ensemble(members([ts], temperature))
    r_ens = thermo.ensemble(members(refs, temperature))
    act = thermo.activation(t_ens, r_ens)
    act["reference_weights"] = r_ens["weights"]
    return act


def symmetry_drift(run_name, ref_atoms):
    """Per evaluation of a walk from an exactly symmetric start: the RMS
    departure (bohr) from the start's point group, by projecting onto the
    totally symmetric part with the start's operations (in its own frame:
    the optimiser removes rigid-body motion, so the frame holds)."""
    x0 = s4.coords_bohr(ref_atoms)
    ops = []
    for r in SIGNS:
        moved = x0 @ r.T
        d = numpy.linalg.norm(moved[:, None] - x0[None], axis=-1)
        perm = d.argmin(axis=1)
        if d[numpy.arange(len(x0)), perm].max() < 1e-3 and len(set(perm)) == len(x0):
            ops.append((r, perm))
    rows = []
    for k, (_, atoms) in enumerate(
        s4.read_xyz_frames(RUNS / run_name / "optimize.xyz")
    ):
        x = s4.coords_bohr(atoms)
        sym = numpy.zeros_like(x)
        for r, perm in ops:
            img = numpy.empty_like(x)
            img[perm] = x @ r.T
            sym += img
        sym /= len(ops)
        r16, r34 = s4.distances(atoms)
        rows.append(
            {
                "evaluation": k,
                "r16": r16,
                "r34": r34,
                "deviation_bohr": float(numpy.sqrt(numpy.mean((x - sym) ** 2))),
            }
        )
    return len(ops), rows


def mode_characters(entry, mode_index, tol=0.01):
    """The characters of one normal mode under its geometry's own
    operations (in the principal frame): +1 or -1 each, with the overlap."""
    atoms = entry["_atoms"]
    m = thermo.masses(entry["_symbols"])
    x = s4.coords_bohr(atoms)
    xc = x - (m @ x) / m.sum()
    i = numpy.einsum("a,ax,ay->xy", m, xc, xc)
    _, axes = numpy.linalg.eigh(numpy.eye(3) * numpy.trace(i) - i)
    xp = xc @ axes
    v = entry["_modes"][mode_index].reshape(-1, 3) @ axes
    out = []
    for r in SIGNS:
        moved = xp @ r.T
        d = numpy.linalg.norm(moved[:, None] - xp[None], axis=-1)
        perm = d.argmin(axis=1)
        if d[numpy.arange(len(xp)), perm].max() > tol or len(set(perm)) != len(xp):
            continue
        img = numpy.empty_like(v)
        img[perm] = v @ r.T
        chi = float((img * v).sum() / (v * v).sum())
        # which atoms the operation holds fixed: names the element
        fixed = [int(a) for a in range(6) if perm[a] == a]
        out.append(
            {
                "op": numpy.diag(r).tolist(),
                "det": float(numpy.linalg.det(r)),
                "fixed_carbons": fixed,
                "character": chi,
            }
        )
    return out


def main() -> None:
    import experiment  # noqa: F401

    res: dict = {"runs": {}, "temperatures": {}}
    hess = {}
    for name, (kind, _) in ex.RUNS_SPEC.items():
        if not (INPUTS / name).exists():
            continue
        if kind in ("hessian", "numerical", "difference"):
            e = hessian_entry(name)
            if e:
                hess[name] = e
                res["runs"][name] = {
                    k: v for k, v in e.items() if not k.startswith("_")
                }
            elif (RUNS / name / "log.txt").exists():
                res["runs"][name] = {"refused": refusal(name)}
        elif kind == "energy":
            e = energy_entry(name)
            if e:
                res["runs"][name] = e
        else:
            e = walk_entry(name)
            if e:
                res["runs"][name] = e

    freq_rows = [
        {"run": n, "mode": k, "frequency_cm1": f}
        for n, e in hess.items()
        for k, f in enumerate(e["frequencies"])
    ]

    # 1. the end points
    if "plus_hessian" in hess and "minus_hessian" in hess:
        res["ends"] = {
            "plus_vs_minus": modes_compare(hess["plus_hessian"], hess["minus_hessian"]),
            "energy_difference": hess["plus_hessian"]["energy"]
            - hess["minus_hessian"]["energy"],
        }
        check_connectivity(hess["plus_hessian"]["_atoms"], c34_bonded=True)
        check_connectivity(hess["minus_hessian"]["_atoms"], c34_bonded=False)

    # 2. conformers: walks, then the distinct minima by energy and dihedrals
    confs = {
        n: res["runs"][n]
        for n in res["runs"]
        if n.startswith("conf_")
        and ex.RUNS_SPEC[n][0] == "walk"
        and "energy" in res["runs"][n]
    }
    e_plus = res["runs"]["plus_walk"]["energy"]
    if confs:
        table = []
        for n, e in sorted(confs.items(), key=lambda kv: kv[1]["energy"]):
            table.append(
                {
                    "run": n,
                    "start": ex.CONFORMERS[n[5:]],
                    "energy": e["energy"],
                    "rel_to_pc_plus_kcal": (e["energy"] - e_plus) * thermo.EH_KCAL,
                    "dihedrals": e.get("dihedrals"),
                    "r34": e["r34"],
                    "point_group": e["point_group"],
                }
            )
        res["conformers"] = table
        walks = {n: dict(e, _path=RUNS / n / "final.xyz") for n, e in confs.items()}
        walks["plus_walk"] = dict(
            res["runs"]["plus_walk"], _path=RUNS / "plus_walk" / "final.xyz"
        )
        res["conformer_minima"] = [
            {
                "runs": grp,
                "energy": walks[grp[0]]["energy"],
                "rel_to_pc_plus_kcal": (walks[grp[0]]["energy"] - e_plus)
                * thermo.EH_KCAL,
                "point_group": walks[grp[0]]["point_group"],
                "dihedrals": walks[grp[0]].get("dihedrals"),
            }
            for grp in distinct_minima(walks)
        ]

    # 3 and 4 need temperatures
    temps = {"experiment": T_EXPERIMENT, "room": T_ROOM}
    res["temperatures"] = temps
    # One Hessian per distinct minimum (energy within 1e-7 Eh): pc_plus's
    # first, then the conformers'.
    refs = []
    conf = sorted(n for n in hess if n.startswith("conf_") and n.endswith("_hessian"))
    for n in ["plus_hessian"] + conf:
        if n in hess and all(abs(hess[n]["energy"] - r["energy"]) > 1e-7 for r in refs):
            refs.append(hess[n])
    ts = hess.get("ts_hessian")
    if ts and refs:
        lowest = min(refs, key=lambda e: e["energy"])
        res["reference"] = {
            "lowest": next(n for n in hess if hess[n] is lowest),
            "ensemble": [next(n for n in hess if hess[n] is r) for r in refs],
            "ensemble_point_groups": [r["point_group"] for r in refs],
            "lowest_minus_pc_plus_kcal": (lowest["energy"] - e_plus) * thermo.EH_KCAL,
            "electronic_barrier_kcal": (ts["energy"] - lowest["energy"])
            * thermo.EH_KCAL,
        }
        act = {}
        for tkey, t in temps.items():
            act[tkey] = {
                "lowest": activation_at(ts, [lowest], t),
                "ensemble": activation_at(ts, refs, t),
            }
            r = thermo.rrho(
                lowest["energy"],
                numpy.array(lowest["frequencies"]),
                lowest["_x"],
                thermo.masses(lowest["_symbols"]),
                t,
            )
            tr = thermo.rrho(
                ts["energy"],
                numpy.array(ts["frequencies"]),
                ts["_x"],
                thermo.masses(ts["_symbols"]),
                t,
            )
            act[tkey]["zpe_kcal"] = {
                "ts": tr["zpe"] * thermo.EH_KCAL,
                "reference": r["zpe"] * thermo.EH_KCAL,
            }
            act[tkey]["h_thermal_kcal"] = {
                "ts": tr["h_thermal"] * thermo.EH_KCAL,
                "reference": r["h_thermal"] * thermo.EH_KCAL,
            }
        res["activation"] = act
        res["pyscf_check"] = pyscf_check(ts)

    # 4. the boat
    if "boat_walk" in res["runs"]:
        n_ops, drift = symmetry_drift(
            "boat_walk", load_atoms(INPUTS / "boat_walk" / "geometry.xyz")
        )
        res["boat_drift"] = {
            "start_operations": n_ops,
            "max_deviation_bohr": max(r["deviation_bohr"] for r in drift),
            "final_deviation_bohr": drift[-1]["deviation_bohr"],
            "trajectory": drift,
        }
    boat = next(
        (n for n in ("boat_hessian", "boat_hessian_difference") if n in hess), None
    )
    if boat and ts:
        b = hess[boat]
        k = int(numpy.argmin(b["frequencies"]))
        res["boat"] = {
            "run": boat,
            "imaginary": [f for f in b["frequencies"] if f < 0],
            "imaginary_mode_characters": mode_characters(b, k),
            "imaginary_mode_c2v": c2v_label(mode_characters(b, k)),
            "electronic_minus_chair_kcal": (b["energy"] - ts["energy"])
            * thermo.EH_KCAL,
        }
        for tkey, t in temps.items():
            bb = thermo.ensemble(members([b], t))
            cc = thermo.ensemble(members([ts], t))
            res["boat"][tkey] = thermo.activation(bb, cc)
    # The analytic Hessians against a difference Hessian at the same point.
    res["analytic_vs_difference"] = {
        a: modes_compare(hess[a], hess[d])
        | {
            "lowest_analytic": hess[a]["frequencies"][:3],
            "lowest_difference": hess[d]["frequencies"][:3],
        }
        for a, d in (
            ("boat_hessian", "boat_hessian_difference"),
            ("conf_g_pm_hessian_analytic", "conf_g_pm_hessian"),
        )
        if a in hess and d in hess
    }
    if ts:
        res["fd_check"] = fd_check(res, ts)
    res["casscf_points"] = casscf_points(res)
    if ts:
        k = int(numpy.argmin(ts["frequencies"]))
        res["ts_imaginary_mode_characters"] = mode_characters(ts, k)

    (ex.HERE / "results.json").write_text(
        json.dumps(res, indent=1, default=float) + "\n"
    )
    if freq_rows:
        with (ex.HERE / "frequencies.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(freq_rows[0]))
            w.writeheader()
            w.writerows(freq_rows)
    summary = {k: v for k, v in res.items() if k not in ("runs", "boat_drift")}
    print(json.dumps(summary, indent=1, default=float))
    for n, e in res["runs"].items():
        short = {
            k: e.get(k)
            for k in (
                "energy",
                "r16",
                "r34",
                "point_group",
                "n_imaginary",
                "wall_s",
                "evaluations",
                "dihedrals",
                "error",
                "max_abs_mine_minus_tree_cm1",
                "rigid_body_cm1",
            )
        }
        if "frequencies" in e:
            short["lowest"] = [round(f, 1) for f in e["frequencies"][:4]]
            short["memory"] = e["memory"]
        print(n, json.dumps(short, default=float))


# The energy-only check's noise: each PC energy converged at conv_tol_grad
# 1e-10 and printed to 10 decimals, so independent errors of about 1e-10 Eh.
SIGMA_E = 1e-10


def fd_check(res: dict, ts: dict) -> dict | None:
    """Part A: second and first differences of PC energies along the TS's
    lowest mode of each C2h irrep, against the analytic Hessian and
    gradient at the same geometry. q has largest Cartesian component 1;
    the points are x_TS +- delta q. Second difference d2 = E+ + E- - 2 E0,
    D(delta) = d2 / delta^2 -> q^T H q as delta -> 0 with D = c + a delta^2
    + O(delta^4): the truncation at delta is (D(2 delta) - D(delta)) / 3,
    and the Richardson value (4 D(delta) - D(2 delta)) / 3 removes it. The
    noise of D(delta) is sqrt(6) SIGMA_E / delta^2. The Hessian's own
    error along q is bounded by its asymmetry before symmetrizing (taken as
    each element's error) times (sum_i |q_i|)^2; the bars add it."""
    runs = res["runs"]
    if "fd_ts" not in runs:
        return None
    e0 = runs["fd_ts"]["pc_energy"]
    hmat = ts["_hessian"]
    g = numpy.array(
        json.loads((RUNS / "ts_hessian" / "nevpt2_gradient.json").read_text())["total"]
    ).ravel()
    out = {
        "e0": e0,
        "e0_minus_ts_hessian_energy": e0 - ts["energy"],
        "sigma_e": SIGMA_E,
        "modes": {},
    }
    for irrep in ex.FD_IRREPS:
        k, freq, q = ex.fd_direction(irrep)
        q = q.ravel()
        qhq = float(q @ hmat @ q)
        # For an eigenmode of H M^-1, q^T H q = lambda q^T M q (mass-weighted
        # eigenvalue, Eh/(bohr^2 amu)), the frequency's own number.
        m = numpy.repeat(thermo.masses(ts["_symbols"], table=thermo.MASS_AVERAGED), 3)
        lam = numpy.sign(freq) * (freq / thermo.CM1_PER_AU) ** 2
        rows = {}
        for d in ex.FD_DELTAS:
            tag = f"{round(d * 100):02d}"
            p, mm = runs.get(f"fd_{irrep}_p{tag}"), runs.get(f"fd_{irrep}_m{tag}")
            if not (p and mm):
                continue
            ep, em = p["pc_energy"], mm["pc_energy"]
            rows[d] = {
                "e_plus_minus_e0": ep - e0,
                "e_minus_minus_e0": em - e0,
                "d2": ep + em - 2 * e0,
                "D": (ep + em - 2 * e0) / d**2,
                "first_difference": ep - em,
                "first_difference_from_gradient": float(2 * d * (g @ q)),
                "noise_D": numpy.sqrt(6.0) * SIGMA_E / d**2,
            }
        hb = float(ts["asymmetry"] * numpy.abs(q).sum() ** 2)
        entry = {
            "hessian_bound": hb,
            "mode": k,
            "frequency": freq,
            "qHq": qhq,
            "lambda_qMq": float(lam * (q * m) @ q),
            "points": {str(d): v for d, v in rows.items()},
        }
        d1, d2 = ex.FD_DELTAS
        if d1 in rows and d2 in rows:
            a, b = rows[d1]["D"], rows[d2]["D"]
            trunc = (b - a) / 3.0
            rich = (4 * a - b) / 3.0
            noise_rich = float(
                numpy.hypot(4 / 3 * rows[d1]["noise_D"], 1 / 3 * rows[d2]["noise_D"])
            )
            entry.update(
                {
                    "truncation_at_d1": trunc,
                    "D_d1_minus_qHq": a - qhq,
                    "bar_D_d1_noise_only": abs(trunc) + 2 * rows[d1]["noise_D"],
                    "bar_D_d1": abs(trunc) + 2 * rows[d1]["noise_D"] + hb,
                    "richardson": rich,
                    "richardson_minus_qHq": rich - qhq,
                    "bar_richardson_noise_only": 2 * noise_rich,
                    "bar_richardson": 2 * noise_rich + hb,
                }
            )
        out["modes"][irrep] = entry
    return out


def kabsch_rms(xa, xb, m) -> float:
    """Mass-weighted-centroid Kabsch RMS distance (bohr) per coordinate."""
    a = xa - (m @ xa) / m.sum()
    b = xb - (m @ xb) / m.sum()
    u, _, vt = numpy.linalg.svd(a.T @ b)
    dd = numpy.sign(numpy.linalg.det(u @ vt))
    r = u @ numpy.diag([1.0, 1.0, dd]) @ vt
    return float(numpy.sqrt(numpy.mean((a @ r - b) ** 2)))


def casscf_points(res: dict) -> dict:
    """Part B: the PC walks from step 4b's two CASSCF C2h stationary
    points, against the PC TS (ts_walk's end)."""
    runs = res["runs"]
    out = {}
    ts = runs.get("ts_walk")
    ts_atoms = load_atoms(RUNS / "ts_walk" / "final.xyz")
    m = thermo.masses([sym for sym, _ in ts_atoms])
    x_ts = s4.coords_bohr(ts_atoms)
    step4 = {"diyl": -232.9967169074, "aromatic": -232.9954445232}
    for label in ("diyl", "aromatic"):
        sp, walk = runs.get(f"{label}_sp"), runs.get(f"{label}_walk")
        entry = {}
        if sp:
            entry["sp"] = {
                "casscf_energy": sp["casscf_energy"],
                "casscf_minus_step4b": sp["casscf_energy"] - step4[label],
                "pc_energy": sp["pc_energy"],
                "pc_minus_ts_kcal": (sp["pc_energy"] - ts["energy"]) * thermo.EH_KCAL,
                "occupations": sp["active"]["occupations"],
            }
        if walk and "energy" in walk:
            n_ops, drift = symmetry_drift(
                f"{label}_walk", load_atoms(INPUTS / f"{label}_walk" / "geometry.xyz")
            )
            x = s4.coords_bohr(load_atoms(RUNS / f"{label}_walk" / "final.xyz"))
            entry["walk"] = {
                "evaluations": walk["evaluations"],
                "r16": walk["r16"],
                "r34": walk["r34"],
                "energy": walk["energy"],
                "minus_ts_eh": walk["energy"] - ts["energy"],
                "rms_from_ts_bohr": kabsch_rms(x, x_ts, m),
                "point_group": walk["point_group"],
                "start_operations": n_ops,
                "max_c2h_deviation_bohr": max(r["deviation_bohr"] for r in drift),
                "max_at": max(drift, key=lambda r: r["deviation_bohr"])["evaluation"],
                "final_c2h_deviation_bohr": drift[-1]["deviation_bohr"],
                "max_abs_r16_minus_r34": max(abs(r["r16"] - r["r34"]) for r in drift),
                "wall_s": walk["wall_s"],
                "occupations": walk["active"]["occupations"],
            }
        if entry:
            out[label] = entry
    return out


def distinct_minima(entries: dict, e_tol=1e-7, x_tol=1e-3) -> list[list[str]]:
    """Groups of runs at the same minimum: energy within e_tol Eh and,
    under the best of a rotation and a rotation-inversion (the mirror
    image is the same energy, a distinct enantiomer only), Kabsch RMS
    within x_tol bohr."""
    groups: list[list[str]] = []
    for n in sorted(entries, key=lambda n: entries[n]["energy"]):
        a = load_atoms(entries[n]["_path"])
        xa = s4.coords_bohr(a)
        m = thermo.masses([sym for sym, _ in a])
        for grp in groups:
            ref = entries[grp[0]]
            if abs(ref["energy"] - entries[n]["energy"]) > e_tol:
                continue
            xb = s4.coords_bohr(load_atoms(ref["_path"]))
            d = min(kabsch_rms(xa, xb, m), kabsch_rms(-xa, xb, m))
            if d < x_tol:
                grp.append(n)
                break
        else:
            groups.append([n])
    return groups


def c2v_label(chars: list[dict]) -> dict:
    """The C2v irrep of a boat mode from its characters. Named by plane:
    sigma_allyl exchanges the two allyls (fixes no carbon), sigma_c2
    holds C2 and C5 (exchanges C1/C3 and C4/C6). Mulliken's B1/B2 is
    then a convention: here B1 is symmetric under sigma_allyl."""
    proper = [c for c in chars if c["det"] > 0 and c["op"] != [1.0, 1.0, 1.0]]
    improper = [c for c in chars if c["det"] < 0]
    if len(proper) != 1 or len(improper) != 2:
        return {"error": f"{len(chars)} operations, not C2v"}
    c2 = round(proper[0]["character"])
    s_c2 = next(c for c in improper if c["fixed_carbons"] == [1, 4])
    s_allyl = next(c for c in improper if c is not s_c2)
    key = (c2, round(s_allyl["character"]), round(s_c2["character"]))
    name = {(1, 1, 1): "A1", (1, -1, -1): "A2", (-1, 1, -1): "B1", (-1, -1, 1): "B2"}
    return {
        "C2": c2,
        "sigma_allyl": key[1],
        "sigma_c2": key[2],
        "irrep": name.get(key),
    }


def pyscf_check(entry) -> dict:
    """PySCF's thermo.thermo on one geometry (its isotope-averaged masses,
    its own symmetry number) against ours at the same masses and sigma."""
    from pyscf import gto
    from pyscf.hessian import thermo as pthermo

    atoms = entry["_atoms"]
    mol = gto.M(atom=atoms, unit="Angstrom", basis="sto-3g", verbose=0)
    m = thermo.masses(entry["_symbols"], table=thermo.MASS_AVERAGED)
    f, _ = thermo.harmonic(entry["_hessian"], entry["_x"], m)
    freq_au = numpy.where(f > 0, f, 1j * abs(f)) / thermo.CM1_PER_AU

    class Model:
        pass

    model = Model()
    model.mol = mol
    model.e_tot = entry["energy"]
    p = pthermo.thermo(model, freq_au, 298.15, 101325)
    sigma = int(p["sym_number"][0])
    ours = thermo.rrho(entry["energy"], f, entry["_x"], m, 298.15, sigma=sigma)
    return {
        "sigma": sigma,
        "zpe_diff": ours["zpe"] - p["ZPE"][0],
        "h_diff": ours["h"] - p["H_tot"][0],
        "s_diff": ours["s"] - p["S_tot"][0],
        "g_diff": ours["g"] - p["G_tot"][0],
        "pyscf_h": p["H_tot"][0],
    }
