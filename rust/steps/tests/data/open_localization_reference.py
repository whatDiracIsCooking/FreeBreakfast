"""The bars of `tests/open_localization_device.rs`: for each case of
`localization_rohf.json` (the C++ tree's `test/gtest/data/localization/rohf.json`,
copied byte for byte: PySCF's DF-ROHF, its doubly and singly occupied
spaces each localized by Pipek-Mezey over Mulliken populations) and each
exponent, how far the tree's localization of its own DF-ROHF may sit from
that reference, to first order, per unit of the SCF residual it stopped at.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    uv run --frozen --group tools python \\
        rust/steps/tests/data/open_localization_reference.py

It writes `localization_rohf_bars.json` beside itself: its provenance, and
per case and exponent `functional` (L summed over the two spaces, as the
reference has it), `sens_l`, `sens_q` and `conv_q`, below.

The SCF is redone here as the reference's emitter ran it (PySCF's DF-ROHF
over the same documents and geometry, `conv_tol` 1e-12, `conv_tol_grad`
1e-9), each space localized by `pm_reference.py`'s `localize` (PySCF's
kernel at `conv_tol` 1e-12 alternated with its Jacobi stability check), and
each space's L checked against the reference's to 1e-9: the reproduction
the analysis is made about. L and the populations are invariant under any
transformation within an atom's AOs, so the tree's unrenormalized
functions need no conversion.

Write the occupied spaces as the reference's rotated by `kappa`, whose
blocks rotate open into core, virtual into core and virtual into open (the
rotations ROHF's energy is not invariant under). The tree's SCF stops when
the residual `r = ||X^T (F D S - S D F) X||_F` of the densities it built
its last F from, `D(kappa)`, is under `commutator_tolerance`, and publishes
the eigenvectors of PySCF's Roothaan matrix at those densities (the
`publish open` of the tree; F's open block does not enter the residual),
whose spaces are `kappa' = T kappa`, with residual `R kappa`. g, T and R
are each formed by central differences at step 1e-4 about the reference,
one column per element of kappa, every Fock build PySCF's density-fitted
one.

- `sens_l = ||g^T T R^+||`, g the gradient of L (both spaces) in kappa at
  the reference's localization: L at a maximum moves by `g . kappa'` (the
  localization's own rotations are stationary there), so
  `|dL| <= sens_l r`.
- `sens_q`: the largest, over every orbital and atom, of `||J_q T R^+||`,
  `J_q` the gradient of that orbital's population on that atom in kappa
  with the localization following it, `dQ/dkappa - dQ/dtheta H^-1 G`, H
  the Hessian of L in each space's own rotations theta (its maximum's
  curvature) and G the mixed derivative of L's theta-gradient in kappa;
  so a population moves by at most `sens_q r`.
- `conv_q`: how far a population sits from the maximum's when the
  localization stops short of it by `tau = 2 (1e-10 + 1e-12)` in L (each
  space within the tree's `functional_tolerance` 1e-10 and PySCF's
  `conv_tol` 1e-12): `L_max - L >= lambda |theta|^2 / 2`, lambda the
  smallest curvature `-H`'s eigenvalue, so `|theta| <= (2 tau / lambda)^(1/2)`
  and a population moves by at most `max ||dQ/dtheta|| |theta|`. Zero for a
  case with no rotation within a space.

It prints each, with each space's L against the reference's.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyscf
import scipy.linalg
from pyscf import gto, lib, scf

HERE = Path(__file__).resolve().parent
MOLECULE = HERE / "molecule.py"
PM = HERE / "pm_reference.py"
REFERENCE = HERE / "localization_rohf.json"
OUT = HERE / "localization_rohf_bars.json"
COMMAND = (
    "uv run --frozen --group tools python "
    "rust/steps/tests/data/open_localization_reference.py"
)
STEP = 1e-4
TAU = 2 * (1e-10 + 1e-12)


def module(name: str, path: Path):
    """The script at `path`, as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def antisymmetric(blocks: list, params: np.ndarray, n: int) -> np.ndarray:
    """The n x n antisymmetric generator whose (rows, cols) blocks hold
    `params` in order, each block row-major."""
    k = np.zeros((n, n))
    at = 0
    for rows, cols in blocks:
        size = len(rows) * len(cols)
        k[np.ix_(rows, cols)] = params[at : at + size].reshape(len(rows), len(cols))
        at += size
    return k - k.T


def pm_gradient(q: np.ndarray, p: int) -> np.ndarray:
    """L_p's gradient in the rotations of one space's orbitals, pairs i < j,
    from the population matrices q[A, i, j]: rotating column i by theta
    toward j moves Q_ii by 2 theta Q_ij and Q_jj by -2 theta Q_ij."""
    m = q.shape[1]
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            d = q[:, i, i] ** (p - 1) - q[:, j, j] ** (p - 1)
            out.append(2 * p * float((q[:, i, j] * d).sum()))
    return np.array(out)


def pairs(m: int) -> list:
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def rotation(m: int, theta: np.ndarray) -> np.ndarray:
    """expm of the generator with theta at (j, i) and -theta at (i, j)."""
    k = np.zeros((m, m))
    for t, (i, j) in zip(theta, pairs(m), strict=True):
        k[j, i], k[i, j] = t, -t
    return scipy.linalg.expm(k)


def scf_case(ref, case: dict) -> dict:
    """The case's converged DF-ROHF, as the reference's emitter ran it, and
    what the analysis reads of it."""
    atoms = [(a["symbol"], tuple(a["coord"])) for a in case["atoms"]]
    symbols = sorted({s for s, _ in atoms})
    orb_doc = ref.bse_document(case["basis"], "basis")
    aux_doc = ref.bse_document(case["aux_basis"], "aux_basis")
    mol = gto.M(
        atom=atoms,
        basis={s: ref.basis_from_bse(orb_doc, s) for s in symbols},
        unit="Bohr",
        charge=case["charge"],
        spin=case["multiplicity"] - 1,
        cart=False,
        verbose=0,
    )
    mf = scf.ROHF(mol).density_fit(
        auxbasis={s: ref.basis_from_bse(aux_doc, s) for s in symbols}
    )
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise SystemExit(f"{case['name']}: not converged, nothing written")
    n_alpha, n_beta = mol.nelec
    assert (n_alpha, n_beta) == (case["n_alpha"], case["n_beta"])
    s = mol.intor_symmetric("int1e_ovlp")
    n = mf.mo_coeff.shape[1]
    core, open_, virt = (
        list(range(n_beta)),
        list(range(n_beta, n_alpha)),
        list(range(n_alpha, n)),
    )
    blocks = [(open_, core), (virt, core), (virt, open_)]
    w, v = np.linalg.eigh(s)
    return {
        "name": case["name"],
        "mol": mol,
        "mf": mf,
        "s": s,
        "c": mf.mo_coeff,
        "n": n,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "core": core,
        "open": open_,
        "virt": virt,
        "blocks": blocks,
        "n_par": sum(len(r) * len(cc) for r, cc in blocks),
        "x": v @ np.diag(w**-0.5) @ v.T,
        "offsets": [(p0, p1) for _, _, p0, p1 in mol.offset_nr_by_atom()],
    }


def rotated(ctx: dict, kappa: np.ndarray) -> np.ndarray:
    """The reference's orbitals rotated by kappa."""
    return ctx["c"] @ scipy.linalg.expm(antisymmetric(ctx["blocks"], kappa, ctx["n"]))


def residual_and_spaces(ctx: dict, kappa: np.ndarray) -> tuple:
    """The residual of D(kappa), and the spaces of PySCF's Roothaan matrix
    there, as rotations from the reference's."""
    c, s, n_alpha, n_beta = ctx["c"], ctx["s"], ctx["n_alpha"], ctx["n_beta"]
    core, open_, virt = ctx["core"], ctx["open"], ctx["virt"]
    rot = rotated(ctx, kappa)
    da = rot[:, :n_alpha] @ rot[:, :n_alpha].T
    db = rot[:, :n_beta] @ rot[:, :n_beta].T
    f = ctx["mf"].get_fock(dm=np.array([da, db]))
    d = da + db
    e = ctx["x"].T @ (f @ d @ s - s @ d @ f) @ ctx["x"]
    _, new = scipy.linalg.eigh(f, s)
    pc = new[:, :n_beta] @ new[:, :n_beta].T
    po = new[:, n_beta:n_alpha] @ new[:, n_beta:n_alpha].T
    moved = np.concatenate(
        [
            (c[:, open_].T @ s @ pc @ s @ c[:, core]).ravel(),
            (c[:, virt].T @ s @ pc @ s @ c[:, core]).ravel(),
            (c[:, virt].T @ s @ po @ s @ c[:, open_]).ravel(),
        ]
    )
    return e.ravel(), moved


def exponent_bars(pm, ctx: dict, record: dict) -> dict:
    """One exponent's functional and bars (the module's doc)."""
    p = record["exponent"]
    mol, s, c, n_par = ctx["mol"], ctx["s"], ctx["c"], ctx["n_par"]
    spaces = []
    for name in ("doubly", "singly"):
        cols = ctx["core"] if name == "doubly" else ctx["open"]
        orb = c[:, cols]
        local = pm.localize(mol, orb, p)[0] if len(cols) > 1 else orb
        l_space = float((pm.populations(mol, s, local) ** p).sum())
        miss = abs(l_space - record[name]["functional"])
        print(f"{ctx['name']} p{p} {name}: L {l_space:.12f}, |dL| {miss:.1e}")
        assert miss < 1e-9, f"{ctx['name']} p{p} {name}: |dL| = {miss:e}"
        spaces.append((cols, orb.T @ s @ local))
    n_theta = [len(pairs(len(cols))) for cols, _ in spaces]
    zero_t = [np.zeros(k) for k in n_theta]

    def orbitals(kappa: np.ndarray, thetas: list) -> list:
        rot = rotated(ctx, kappa)
        return [
            rot[:, cols] @ u @ rotation(len(cols), t)
            for (cols, u), t in zip(spaces, thetas, strict=True)
        ]

    def matrices(orb: np.ndarray) -> np.ndarray:
        wq = s @ orb
        return np.array(
            [
                0.5 * (orb[a:b].T @ wq[a:b] + wq[a:b].T @ orb[a:b])
                for a, b in ctx["offsets"]
            ]
        )

    def lfun(kappa: np.ndarray) -> float:
        return sum(
            float((pm.populations(mol, s, o) ** p).sum())
            for o in orbitals(kappa, zero_t)
        )

    def qvec(kappa: np.ndarray, thetas: list) -> np.ndarray:
        return np.concatenate(
            [pm.populations(mol, s, o).T.ravel() for o in orbitals(kappa, thetas)]
        )

    def gvec(kappa: np.ndarray, thetas: list) -> np.ndarray:
        return np.concatenate(
            [pm_gradient(matrices(o), p) for o in orbitals(kappa, thetas)]
        )

    def split(theta: np.ndarray) -> list:
        return np.split(theta, np.cumsum(n_theta)[:-1])

    def central(fun, k: int, size: int):
        step = np.zeros(size)
        step[k] = STEP
        return (fun(step) - fun(-step)) / (2 * STEP)

    n_t = sum(n_theta)
    zero = np.zeros(n_par)
    r = np.zeros((ctx["n"] ** 2, n_par))
    t = np.zeros((n_par, n_par))
    for k in range(n_par):
        step = np.zeros(n_par)
        step[k] = STEP
        plus, minus = residual_and_spaces(ctx, step), residual_and_spaces(ctx, -step)
        r[:, k] = (plus[0] - minus[0]) / (2 * STEP)
        t[:, k] = (plus[1] - minus[1]) / (2 * STEP)
    g = np.array([central(lfun, k, n_par) for k in range(n_par)])
    dq_dk = np.array(
        [central(lambda kap: qvec(kap, zero_t), k, n_par) for k in range(n_par)]
    ).T
    mixed = np.array(
        [central(lambda kap: gvec(kap, zero_t), k, n_par) for k in range(n_par)]
    ).T.reshape(n_t, n_par)
    hess = np.array(
        [central(lambda th: gvec(zero, split(th)), k, n_t) for k in range(n_t)]
    ).T.reshape(n_t, n_t)
    dq_dt = np.array(
        [central(lambda th: qvec(zero, split(th)), k, n_t) for k in range(n_t)]
    ).T.reshape(dq_dk.shape[0], n_t)
    r_pinv = np.linalg.pinv(r)
    sens_l = float(np.linalg.norm(g @ t @ r_pinv))
    if n_t:
        hess = 0.5 * (hess + hess.T)
        curvature = np.linalg.eigvalsh(-hess)
        assert curvature[0] > 0, f"{ctx['name']} p{p}: not a maximum"
        follow = dq_dk - dq_dt @ np.linalg.solve(hess, mixed)
        reach = np.sqrt(2 * TAU / curvature[0])
        conv_q = float(np.linalg.norm(dq_dt, axis=1).max() * reach)
        lam = float(curvature[0])
    else:
        follow, conv_q, lam = dq_dk, 0.0, float("nan")
    sens_q = float(np.linalg.norm(follow @ t @ r_pinv, axis=1).max())
    functional = record["doubly"]["functional"] + record["singly"]["functional"]
    print(
        f"{ctx['name']} p{p}: L {functional:.12f}, sens_l {sens_l:.3e}, "
        f"sens_q {sens_q:.3e}, lambda_min {lam:.3e}, conv_q {conv_q:.3e}"
    )
    return {
        "exponent": p,
        "functional": functional,
        "sens_l": sens_l,
        "sens_q": sens_q,
        "conv_q": conv_q,
    }


def main() -> None:
    lib.num_threads(1)
    ref = module("molecule_reference", MOLECULE)
    pm = module("pm_reference", PM)
    doc = json.loads(REFERENCE.read_text())
    cases = []
    for case in doc["cases"]:
        ctx = scf_case(ref, case)
        cases.append(
            {
                "name": case["name"],
                "exponents": [exponent_bars(pm, ctx, r) for r in case["pipek_mezey"]],
            }
        )
    out = {
        "description": "Per localization_rohf.json case and exponent: L summed "
        "over the spaces (the reference's), and the first-order bars of "
        "tests/open_localization_device.rs per unit of the SCF residual (sens_l on "
        "L, sens_q on any population) and conv_q, a population's reach from "
        "the localization stopping short of the maximum",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
