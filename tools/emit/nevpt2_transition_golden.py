#!/usr/bin/env python3
"""Write the transition goldens `nevpt2::rdms`' bra/ket overload is tested
against: PySCF's dm1, dm2, dm3, f3ca and f3ac between two different CI
vectors, test/gtest/data/nevpt2_transition/<case>/<array>.npy.

Each case takes its active (tu|vw) from a committed SC-NEVPT2 golden
(test/gtest/data/nevpt2/<source>/h2e.npy, written by nevpt2_golden.py),
optionally a block of its orbitals, and two different CI vectors: the
source's own CI vector where the electron count is the source's, otherwise
seeded uniform random ones, each normalised. Random vectors carry no spin
or spatial symmetry, so no element of any tensor vanishes by symmetry.

From PySCF's own entry points, between bra I and ket J:

- dm2[p,q,r,s] = <I|E_pq E_rs|J>, dm3[p,q,r,s,t,u] = <I|E_pq E_rs E_tu|J>:
  `fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, ...)`, raw (not
  reordered);
- dm1[p,q] = <I|E_pq|J>: **the transpose of make_dm1234's dm1**, which for
  bra != ket is <I|E_qp|J> (PySCF's `trans_rdm1` convention), so that dm1
  reads as dm2 and dm3 do;
- f3ca[p,q,r,s,t,u] = sum_ijk (ki|uj) dm4[p,q,r,s,t,j,k,i] and
  f3ac[p,q,r,s,t,u] = sum_ijk (ik|jt) dm4[p,q,r,s,j,u,i,k], over that call's
  raw transition dm4 and the chemists' (tu|vw): `mrpt.nevpt2.make_a16`'s
  commented dm4 forms of `f3ca` and `f3ac` (`_contract4pdm`'s
  'NEVPTkern_cedf_aedf' and 'NEVPTkern_aedf_ecdf'), which the transition
  energy reads with the transition dm4 in their place. PySCF's
  `_contract4pdm` takes one vector, so it cannot give these directly.

Before writing a case it checks, and on any miss writes nothing:

- that the two f3 contractions are `_contract4pdm`'s, at bra = ket = each
  of the two vectors, to CHECK_TOL of max|f3|;
- that dm1..dm4 are what the definitions above say, against a brute-force
  build over dense E_pq matrices in the determinant basis (from
  `cistring.gen_linkstr_index`), to CHECK_TOL of each max;
- that make_dm123(bra, ket)'s dm1, dm2 and dm3 are make_dm1234's.

Every array is a '<f8' Fortran-order .npy (the layout `src/io/npy` reads;
`np.load` gives back the same logical, C-indexed tensor). BLAS and OpenMP
are pinned to one thread, so a rerun regenerates every file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_transition_golden.py

writes test/gtest/data/nevpt2_transition/. Re-run it to regenerate; the
files are never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
from pyscf import fci  # noqa: E402
from pyscf.mrpt import nevpt2  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "test" / "gtest" / "data" / "nevpt2"
OUT = ROOT / "test" / "gtest" / "data" / "nevpt2_transition"

# Each case: the nevpt2/ golden whose h2e it reads, the active orbitals it
# keeps (a contiguous block), (n_alpha, n_beta), and the seed of its random
# vectors. A case at the source's own space takes the source's CI vector as
# its bra and one random ket; any other takes two random vectors.
CASES = {
    "n2_cas44": ("n2_ccpvdz_cas44_df", (0, 4), (2, 2), 672),
    "n2_cas6_3a2b": ("n2_ccpvdz_cas88_df", (1, 7), (3, 2), 673),
}

# Of max|tensor|: two builds of the same sums in different orders.
CHECK_TOL = 1e-12

# check-added-large-files' --maxkb in .pre-commit-config.yaml.
LIMIT = 3072 * 1024


def e_matrices(norb: int, nelec: tuple[int, int]) -> numpy.ndarray:
    """E[p, q] = sum_sigma a+_p,sigma a_q,sigma as a dense matrix over the
    alpha-major determinants: a link (p, q, J, s) of string I is
    a+_p a_q |I> = s |J>, so E[p, q][J, I] += s."""
    la, lb = (fci.cistring.gen_linkstr_index(range(norb), n) for n in nelec)
    na, nb = len(la), len(lb)
    e = numpy.zeros((norb, norb, na * nb, na * nb))
    for i, links in enumerate(la):
        for p, q, j, s in links:
            for b in range(nb):
                e[p, q, j * nb + b, i * nb + b] += s
    for i, links in enumerate(lb):
        for p, q, j, s in links:
            for a in range(na):
                e[p, q, a * nb + j, a * nb + i] += s
    return e


def brute(e: numpy.ndarray, bra, ket) -> tuple[numpy.ndarray, ...]:
    """<I|E_pq|J>, <I|E_pq E_rs|J>, <I|E_pq E_rs E_tu|J> and the 4-RDM, by
    applying the dense E matrices."""
    b, k = bra.ravel(), ket.ravel()
    t1k = numpy.einsum("pqKJ,J->pqK", e, k)  # E_pq |J>
    t2k = numpy.einsum("pqKM,rsM->pqrsK", e, t1k)  # E_pq E_rs |J>
    t1b = numpy.einsum("pqJK,J->pqK", e, b)  # <I| E_pq
    t2b = numpy.einsum("rsMK,pqM->pqrsK", e, t1b)  # <I| E_pq E_rs
    return (
        numpy.einsum("K,pqK->pq", b, t1k),
        numpy.einsum("K,pqrsK->pqrs", b, t2k),
        numpy.einsum("pqK,rstuK->pqrstu", t1b, t2k),
        numpy.einsum("pqrsK,tuvwK->pqrstuvw", t2b, t2k),
    )


def f3(aaaa: numpy.ndarray, dm4: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    """f3ca and f3ac of a raw (possibly transition) 4-RDM and the chemists'
    (tu|vw): make_a16's commented forms, `a16 -= einsum('kcij,rpqbajki',
    h2e, dm4)` and `a16 += einsum('ijka,rpqbjcik', h2e, dm4)`, with the
    physicist h2e written as aaaa and the f3 transposes undone."""
    return (
        numpy.einsum("kiuj,pqrstjki->pqrstu", aaaa, dm4),
        numpy.einsum("ikjt,pqrsjuik->pqrstu", aaaa, dm4),
    )


def require(what: str, got: numpy.ndarray, want: numpy.ndarray) -> float:
    """max|got - want| / max|want|, raising past CHECK_TOL."""
    rel = float(numpy.abs(got - want).max() / numpy.abs(want).max())
    if not rel <= CHECK_TOL:
        raise SystemExit(f"{what}: {rel:.3e} of max, over {CHECK_TOL}")
    return rel


def unit(rng: numpy.random.Generator, shape: tuple[int, int]) -> numpy.ndarray:
    v = rng.uniform(-1.0, 1.0, shape)
    return v / numpy.linalg.norm(v)


def save(path: pathlib.Path, array: numpy.ndarray) -> int:
    """One '<f8' Fortran-order .npy; its size in bytes."""
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.stat().st_size


def convert(name: str, out: pathlib.Path) -> int:
    """Check one case and write its directory; the bytes written."""
    source, (lo, hi), nelec, seed = CASES[name]
    h2e_all = numpy.load(SOURCE / source / "h2e.npy")
    h2e = numpy.ascontiguousarray(h2e_all[lo:hi, lo:hi, lo:hi, lo:hi])
    norb = hi - lo
    # h2e[p, q, r, s] = (pr|qs); _contract4pdm's aaaa is chemists'.
    aaaa = numpy.ascontiguousarray(h2e.transpose(0, 2, 1, 3))
    shape = tuple(fci.cistring.num_strings(norb, n) for n in nelec)
    rng = numpy.random.default_rng(seed)
    ci = numpy.load(SOURCE / source / "ci.npy")
    bra = (
        numpy.ascontiguousarray(ci)
        if ci.shape == shape and lo == 0
        else unit(rng, shape)
    )
    ket = unit(rng, shape)

    link = tuple(fci.cistring.gen_linkstr_index(range(norb), n) for n in nelec)
    worst = 0.0
    for c in (bra, ket):
        dm4 = fci.rdm.make_dm1234("FCI4pdm_kern_sf", c, c, norb, nelec)[3]
        for got, kern in zip(
            f3(aaaa, dm4), ("NEVPTkern_cedf_aedf", "NEVPTkern_aedf_ecdf"), strict=True
        ):
            want = nevpt2._contract4pdm(kern, aaaa, c, norb, nelec, link)
            worst = max(worst, require(f"{name} {kern} at bra = ket", got, want))

    dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234("FCI4pdm_kern_sf", bra, ket, norb, nelec)
    dm1 = dm1.T
    for what, got, want in zip(
        ("dm1", "dm2", "dm3", "dm4"),
        (dm1, dm2, dm3, dm4),
        brute(e_matrices(norb, nelec), bra, ket),
        strict=True,
    ):
        worst = max(worst, require(f"{name} {what} against brute force", got, want))
    for what, got, want in zip(
        ("dm1", "dm2", "dm3"),
        (dm1.T, dm2, dm3),
        fci.rdm.make_dm123("FCI3pdm_kern_sf", bra, ket, norb, nelec),
        strict=True,
    ):
        worst = max(worst, require(f"{name} make_dm123's {what}", got, want))
    f3ca, f3ac = f3(aaaa, dm4)

    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "bra": bra, "ket": ket, "h2e": h2e,
        "dm1": dm1, "dm2": dm2, "dm3": dm3, "f3ca": f3ca, "f3ac": f3ac,
    }  # fmt: skip
    written = sum(save(out / f"{a}.npy", arr) for a, arr in arrays.items())
    print(f"{name}: checks within {worst:.1e} of max, {written} bytes")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    total = sum(convert(name, args.out / name) for name in CASES)
    print(f"total {total} bytes")
    over = [p for p in args.out.rglob("*.npy") if p.stat().st_size > LIMIT]
    if over:
        raise SystemExit(f"over the {LIMIT}-byte limit: {over}")


if __name__ == "__main__":
    main()
