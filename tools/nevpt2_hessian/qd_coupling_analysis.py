#!/usr/bin/env python3
"""The analysis half of the C2v A1/B2 coupling references
(test/gtest/data/qd_nevpt2_hessian/<case>.analysis.json, #1058): what
docs/nevpt2/nevpt2.tex's symmetric-geometries section says about a model
space whose two roots have different irreps that a vibration couples, read
off the case's own states.

qd_nevpt2_hessian.py ladders SC's C_MN = H_eff[M][N] over this tree's
driver and so marks SC's expected refusals itself. Two things a ladder
cannot show are measured here instead, over tools/emit/qd_nevpt2_reference.py's
SC- and PC-QD-NEVPT2 (PySCF's class arithmetic; the state polished to
|g| < 1e-12, SA-canonical core and virtuals):

- **The guarded tuples.** Per class and ket J, the tuples whose own norm
  N^J_t the energy's guard drops (|N| <= 1e-14), and among them those the
  other root I keeps (|N^I_t| > 1e-8): empty for one ket, populated for the
  other. Along each b2 coordinate q, at s and 2s, |T^t_IJ(R0 + s q)| / s
  estimates the first-order transition tangent T-dot (it is linear: the two
  must agree to TANGENT_AGREE) and N^J_t / s^2 the ket's population. The
  tuple of largest T-dot is `chosen`, the one #962 reads.
- **PC's lifted null modes** (Eq. qd-pcfail). Per class and ket N, the modes
  of the metric S_N cut at R0 (nevpt2_reference.pc_cut); along each b2
  coordinate, at s and 2s, those of the cut modes whose eigenvalue rises
  above LIFTED of the largest: lambda / s^2 (second order: the two must agree
  to LIFT_AGREE), and over every tuple t, with phi the normalized G v of the
  lifted modes, max_t |<phi_t|V|M>| (O(1): agreeing to OVERLAP_AGREE, and
  above OVERLAP_FLOOR) and max_t |<phi_t|V|N>| / s (O(h), as symmetry
  forces it). A coordinate along which such a mode lifts with a nonzero
  overlap to the other state is `lifted`; PC's expected-refused blocks are
  the pairs of lifted coordinates (qd_nevpt2_hessian.pc_expected).

The overlaps are <phi|V|M> = v^T b_{N<-M,t} / sqrt(lambda): pc_y over the
pair's (N, M) metrics at the lifted eigenvector, which is how
qd_nevpt2_reference.pc_qd forms its bra rows. Degenerate lifted modes mix
freely, so each tuple's overlap is its norm over the class's lifted modes.

Needs the `tools` dependency group (PySCF):

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/nevpt2_hessian/qd_coupling_analysis.py [--only CASE ...]

writes test/gtest/data/qd_nevpt2_hessian/<case>.analysis.json, then
qd_nevpt2_hessian.py (on a GPU) writes <case>.json from it. One BLAS
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
import nevpt2_reference  # noqa: E402
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
# tree's kNumericalZero), and the norm the other ket must keep for a tuple
# to count as populated there.
GUARD = 1e-14
POPULATED = 1e-8
# The displacements (bohr) the first- and second-order rates are read at.
STEPS = (1e-3, 2e-3)
# A cut mode is lifted when its eigenvalue passes this fraction of its
# metric's largest at the smaller step (a structural null stays at
# round-off, 1e-16 here).
LIFTED = 1e-12
# The rates at s and 2s agree to these, relative: lambda / s^2, |<phi|V|M>|,
# |<phi|V|N>| / s and |T| / s.
LIFT_AGREE = 0.02
OVERLAP_AGREE = 0.02
TANGENT_AGREE = 0.02
# |T| / s below this is zero: round-off over s.
TANGENT_ZERO = 1e-9
# Two coordinates' Rayleigh quotients rho_t = H^J_t / N^J_t of a populated
# empty tuple differ (a rho defect, Eq. qd-scfail) past this, relative.
RHO_AGREE = 1e-6
# |<phi|V|M>| above this is nonzero (the structural nulls' are round-off).
OVERLAP_FLOOR = 1e-8


def one_function(case: dict, key: str) -> bool:
    """A class whose perturbers span one function each: Srs at two active
    electrons, Sij at two active holes (src/nevpt2/energy/README.md,
    "One-function classes"). Its tuples keep the one function's Dyall energy
    whatever their norm, so the guard drops none of them (the tree's
    ScEffectiveHamiltonianTangent::empty lists none) and their coupling is
    linear in T."""
    sys_ = case["system"]
    if key == "Srs":
        return sys_["nelecas"] == 2
    if key == "Sij":
        return 2 * sys_["ncas"] - sys_["nelecas"] == 2
    return False


def state_at(case: dict, displacement: list[float]):
    """The polished SA-canonical state at R0 + displacement and its
    transition RDMs."""
    sys_ = case["system"]
    atoms = [
        (s, tuple(c[k] + displacement[3 * a + k] for k in range(3)))
        for a, (s, c) in enumerate(sys_["atoms"])
    ]
    entry = {
        "from": None,
        "polish": True,
        "case": {
            "name": case["name"],
            "atoms": atoms,
            "basis": sys_["basis"],
            "cas": (sys_["nelecas"], sys_["ncas"]),
            "weights": sys_["weights"],
        },
    }
    state = qd.build_state(entry)
    rdms, _, _ = qd.transition_rdms(state)
    return state, rdms


def own_norms(state, rdms) -> dict:
    n = len(state.ci)
    return {j: qd.class_norms(state.blocks, rdms[j, j], 1.0) for j in range(n)}


def transition_norms(state, rdms, i: int, j: int) -> dict:
    """T^t_IJ per class: the mean of the norm einsums over (I, J) and (J, I),
    as qd_nevpt2_reference.sc_qd forms it."""
    forward = qd.class_norms(state.blocks, rdms[i, j], 0.0)
    backward = qd.class_norms(state.blocks, rdms[j, i], 0.0)
    return {k: 0.5 * (forward[k] + backward[k]) for k in qd.CLASSES}


def symmetric(s):
    return 0.5 * (s + s.T)


def lifted_overlaps(state, rdms, ket: int, bra: int, key: str, dropped: int):
    """Class `key` of ket N at a displaced point: the cut modes (the first
    `dropped`, ascending) that rose past LIFTED, their eigenvalues over the
    largest, and over every tuple the norm over those modes of <phi|V|bra>
    and <phi|V|ket>."""
    tuples = qd.pc_tuples(state.blocks)
    x, h1, _, _ = tuples[key]
    s, s1 = qd.pc_metrics(state.blocks, rdms[ket, ket], 1.0)[key]
    sb, s1b = qd.pc_metrics(state.blocks, rdms[ket, bra], 0.0)[key]
    s = symmetric(s)
    sigma, v = numpy.linalg.eigh(s)
    modes = [k for k in range(dropped) if sigma[k] > LIFTED * sigma[-1]]
    if not modes:
        return None
    u = v[:, modes] / numpy.sqrt(sigma[modes])
    to_bra = qd.pc_y(x, h1, sb, s1b, u)
    to_ket = qd.pc_y(x, h1, s, s1, u)
    return {
        "modes": len(modes),
        "lambda": [float(sigma[k] / sigma[-1]) for k in modes],
        "bra": float(numpy.sqrt((to_bra**2).sum(axis=1)).max()),
        "ket": float(numpy.sqrt((to_ket**2).sum(axis=1)).max()),
    }


def agree(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def at_reference(case: dict) -> tuple:
    """R0's state and RDMs, its CASSCF energies, both H_eff (symmetrized,
    flat), its SA-canonical orbital energies, the guarded tuples per class
    and ket, the one-sided ones, and PC's cut modes per ket and class."""
    (pair,) = case["state_pairs"]
    state, rdms = state_at(case, [0.0] * 3 * len(case["system"]["atoms"]))
    sc, _ = qd.sc_qd(state, rdms)
    pc, info = qd.pc_qd(state, rdms, sc)
    assert not info["refusals"], info["refusals"]
    reference = {
        "casscf": [float(e) for e in state.energies],
        "sc": [float(x) for x in numpy.ravel(sc["heff_sym"])],
        "pc": [float(x) for x in numpy.ravel(pc["pc_heff_sym"])],
        "core_energies": [float(e) for e in state.mo_energy[: state.casci.ncore]],
        "virtual_energies": [
            float(e) for e in state.mo_energy[state.casci.ncore + state.casci.ncas :]
        ],
    }
    print(f"{case['name']}: R0 CASSCF {reference['casscf']}", flush=True)

    # The guarded tuples at R0, per class and ket.
    norms0 = own_norms(state, rdms)
    guarded = []
    one_sided = []
    for key in qd.CLASSES:
        shape = numpy.shape(norms0[0][key])
        flat = {j: numpy.ravel(norms0[j][key]) for j in pair}
        for ket in pair:
            (other,) = [i for i in pair if i != ket]
            empty = numpy.abs(flat[ket]) <= GUARD
            guarded.append({"class": key, "ket": ket, "count": int(empty.sum())})
            for t in numpy.where(empty & (numpy.abs(flat[other]) > POPULATED))[0]:
                one_sided.append(
                    {
                        "class": key,
                        "ket": ket,
                        "bra": other,
                        "tuple": int(t),
                        "index": [int(i) for i in numpy.unravel_index(t, shape)],
                        "shape": [int(n) for n in shape],
                        "one_function": one_function(case, key),
                        "norm_ket": float(flat[ket][t]),
                        "norm_bra": float(flat[other][t]),
                        "tangent": {},
                        "population": {},
                        "rho": {},
                    }
                )

    # PC: the cut modes at R0, per ket and class.
    dropped = {}
    for ket in pair:
        for key, (s, _) in qd.pc_metrics(state.blocks, rdms[ket, ket], 1.0).items():
            if key == "Sijrs":
                continue
            sigma = numpy.linalg.eigvalsh(symmetric(s))
            cut = nevpt2_reference.pc_cut(len(sigma)) * sigma[-1]
            dropped[ket, key] = int((sigma <= cut).sum())
    return state, rdms, reference, guarded, one_sided, dropped


def analyse(case: dict) -> dict:
    coords = generator.symmetry_coordinates(case["system"]["atoms"])
    b2 = [
        (label, v)
        for label, irrep, v in zip(
            coords["labels"], coords["irreps"], coords["vectors"], strict=True
        )
        if irrep == "B2"
    ]
    (pair,) = case["state_pairs"]
    _, _, reference, guarded, one_sided, dropped = at_reference(case)
    lifts = []
    for label, q in b2:
        found = {}
        for step in STEPS:
            st, r = state_at(case, [step * x for x in q])
            norms = own_norms(st, r)
            hams = {j: qd.class_hams(st.blocks, r[j, j]) for j in pair}
            t_norms = transition_norms(st, r, *pair)
            for item in one_sided:
                k, t = item["class"], item["tuple"]
                item["tangent"].setdefault(label, []).append(
                    float(abs(numpy.ravel(t_norms[k])[t]) / step)
                )
                n_ket = float(numpy.ravel(norms[item["ket"]][k])[t])
                item["population"].setdefault(label, []).append(abs(n_ket) / step**2)
                if label in case["coupled"]:
                    h_ket = float(numpy.ravel(hams[item["ket"]][k][0])[t])
                    item["rho"].setdefault(label, []).append(h_ket / n_ket)
            for ket in pair:
                (bra,) = [i for i in pair if i != ket]
                for key in qd.CLASSES:
                    if key == "Sijrs" or dropped[ket, key] == 0:
                        continue
                    got = lifted_overlaps(st, r, ket, bra, key, dropped[ket, key])
                    if got is not None:
                        found.setdefault((ket, key), []).append((step, got))
        for (ket, key), rows in found.items():
            entry = {
                "coordinate": label,
                "ket": ket,
                "bra": [i for i in pair if i != ket][0],
                "class": key,
                "steps": [s for s, _ in rows],
                "modes": [g["modes"] for _, g in rows],
                "lift": [max(g["lambda"]) / s**2 for s, g in rows],
                "bra_overlap": [g["bra"] for _, g in rows],
                "ket_overlap_rate": [g["ket"] / s for s, g in rows],
            }
            entry["second_order"] = len(rows) == len(STEPS) and agree(
                *entry["lift"], LIFT_AGREE
            )
            entry["bra_overlap_nonzero"] = (
                len(rows) == len(STEPS)
                and agree(*entry["bra_overlap"], OVERLAP_AGREE)
                and min(entry["bra_overlap"]) > OVERLAP_FLOOR
            )
            entry["ket_overlap_first_order"] = len(rows) == len(STEPS) and agree(
                *entry["ket_overlap_rate"], OVERLAP_AGREE
            )
            lifts.append(entry)
        print(
            f"{case['name']} along {label}: "
            + (
                ", ".join(
                    f"ket {e['ket']} {e['class']} lift {e['lift'][0]:.3e} "
                    f"<phi|V|{e['bra']}> {e['bra_overlap'][0]:.3e}"
                    for e in lifts
                    if e["coordinate"] == label
                )
                or "nothing lifts"
            ),
            flush=True,
        )

    for item in one_sided:
        # Linear in s along every coordinate: the two rates agree, or both
        # vanish (the translation and rotation leave T at round-off).
        item["first_order"] = all(
            agree(*rates, TANGENT_AGREE) or max(rates) < TANGENT_ZERO
            for rates in item["tangent"].values()
        )
    lifted = [
        label
        for label, _ in b2
        if any(
            e["coordinate"] == label
            and e["second_order"]
            and e["bra_overlap_nonzero"]
            and e["ket_overlap_first_order"]
            for e in lifts
        )
    ]
    # A rho defect: the populated empty tuple's Dyall quotient differs
    # between the coupled coordinates (at the smaller step), so its coupling
    # -s T-dot / (Delta + rho(q)) is homogeneous but not linear (Eq.
    # qd-scfail). A one-function class has none.
    for item in one_sided:
        rhos = [rates[0] for rates in item["rho"].values()]
        item["rho_spread"] = (max(rhos) - min(rhos)) / max(abs(x) for x in rhos)
        item["rho_defect"] = not item["one_function"] and item["rho_spread"] > RHO_AGREE
    # The tuple #962 reads: guarded (so in a class with more than one
    # function), linear in s, of largest T-dot; none if every one-sided
    # tuple is in a one-function class.
    candidates = [i for i in one_sided if i["first_order"] and not i["one_function"]]
    chosen = (
        max(
            candidates,
            key=lambda i: max(rates[0] for rates in i["tangent"].values()),
        )
        if candidates
        else None
    )
    return {
        "description": "The analysis of the C2v A1/B2 coupling case at R0 and "
        "along each b2 coordinate, over tools/emit/qd_nevpt2_reference.py's "
        "SC- and PC-QD-NEVPT2: the SC tuples the guard drops for one ket and "
        "the other keeps, with their first-order transition tangents, and "
        "PC's metric modes cut at R0 that lift at second order with a "
        "first-order overlap to the other state. "
        "tools/nevpt2_hessian/qd_coupling_analysis.py",
        "name": case["name"],
        "coordinates": coords,
        "steps": list(STEPS),
        "tolerances": {
            "guard": GUARD,
            "populated": POPULATED,
            "lifted": LIFTED,
            "lift_agree": LIFT_AGREE,
            "overlap_agree": OVERLAP_AGREE,
            "tangent_agree": TANGENT_AGREE,
            "tangent_zero": TANGENT_ZERO,
            "rho_agree": RHO_AGREE,
            "overlap_floor": OVERLAP_FLOOR,
        },
        "reference": reference,
        "sc": {
            "guarded": guarded,
            "one_sided": one_sided,
            "chosen": chosen,
            "rho_defects": [
                {"class": i["class"], "ket": i["ket"], "tuple": i["tuple"]}
                for i in one_sided
                if i["rho_defect"]
            ],
        },
        "pc": {
            "state_pair": pair,
            "dropped": [
                {"ket": ket, "class": key, "count": count}
                for (ket, key), count in dropped.items()
            ],
            "lifts": lifts,
            "lifted_coordinates": lifted,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=generator.OUT)
    parser.add_argument("--only", nargs="+", help="these cases alone")
    args = parser.parse_args()
    for case in generator.COUPLING_CASES:
        if args.only and case["name"] not in args.only:
            continue
        record = analyse(case)
        args.out.mkdir(parents=True, exist_ok=True)
        out = generator.analysis_path(args.out, case["name"])
        out.write_text(generator.dumps(record) + "\n")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
