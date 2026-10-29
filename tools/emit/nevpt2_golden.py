#!/usr/bin/env python3
"""Convert nevpt2-gpu-demo's density-fitted goldens into the .npy files the
SC-NEVPT2 tests read, adding PySCF's dm3, f3ca and f3ac.

The demo (github.com/whatDiracIsCooking/nevpt2-gpu-demo) commits each golden
as one flat binary file, `golden/<case>.nevpt2gold`, written by its
`reference_data/generate_golden.py --df`: N2 at 1.1 A, a CASCI (on RHF or on
committed CASSCF orbitals), canonicalised as `pyscf.mrpt.NEVPT.kernel` does,
density-fitted throughout with `<basis>-jkfit`, and PySCF's own DF-NEVPT2
per-class answer. Its reader, `reference_data/golden_format.py`, is inlined
here (struct + numpy only) as `read_golden`; nothing from the demo runs.

The demo deliberately leaves out dm3, f3ca and f3ac (building them is what
its GPU code demonstrates). This script adds them from PySCF's own entry
points, exactly as `NEVPT.kernel` builds them:

- dm1, dm2, dm3 = `fci.rdm.make_dm123('FCI3pdm_kern_sf', ci, ci, ...)`, raw
  (not reordered);
- f3ca, f3ac = `mrpt.nevpt2._contract4pdm('NEVPTkern_cedf_aedf' /
  'NEVPTkern_aedf_ecdf', aaaa, ...)`, with aaaa = h2e.transpose(0, 2, 1, 3),
  the golden's physicist active block back in PySCF's ppaa layout.

They are written for the cases `CASES` marks (up to CAS(8,8)); at
CAS(10,10) each is 8 MB, over the tree's per-file limit, and `--all-rdms`
writes them anyway, for use outside the tree. Every case is checked either way.

Before writing a case it checks that dm1 and dm2 equal the golden's bit for
bit, and that the Sr and Si class norms and energies -- the two classes that
read dm3, f3ca and f3ac, through `make_a16` and `make_a22` -- rebuilt from
these tensors and the golden's own integrals match the golden's to
`CHECK_TOL`. On any miss it writes nothing.

Every array is a '<f8' Fortran-order .npy (the layout `src/io/npy` reads;
`np.load` gives back the same logical, C-indexed tensor), one file per
array: test/gtest/data/nevpt2/<case>/<array>.npy. BLAS and OpenMP are pinned
to one thread and nothing run-dependent is written, so a rerun regenerates
every file byte for byte. The data README has the conventions.

    git clone https://github.com/whatDiracIsCooking/nevpt2-gpu-demo /tmp/demo
    git -C /tmp/demo checkout <commit in test/gtest/data/nevpt2/README.md>
    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_golden.py /tmp/demo

writes test/gtest/data/nevpt2/. Re-run it to regenerate; the files are never
hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import pathlib  # noqa: E402
import struct  # noqa: E402

import numpy  # noqa: E402
from pyscf import fci  # noqa: E402
from pyscf.mrpt import nevpt2  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "nevpt2"

# Each converted case: its golden's sha256 (the file at the demo commit the
# data README names; a different file is refused), the aux basis it was
# fitted with, and whether dm3, f3ca and f3ac are written for it. They are
# n_act^6 doubles, 8 MB each at CAS(10,10), over the tree's 3 MB per-file
# limit; up to CAS(8,8) they are committed, and --all-rdms writes them for
# every case (to an --out outside the tree).
CASES = {
    "n2_ccpvdz_cas44_df": (
        "bdb42a749ec749b22e5cff75d75e2aa93b8ac86cad61655334105e29bba49eea",
        "cc-pVDZ-JKFIT",
        True,
    ),
    "n2_ccpvdz_cas44_casscf_df": (
        "8d36d59c15bfee529bd8587c632e97ae8a04b815049769d4631fd5c1472450b3",
        "cc-pVDZ-JKFIT",
        True,
    ),
    "n2_ccpvdz_cas88_df": (
        "8ce146787cfde3e062fa360cbcb4b5d1ee3e619b8d0e9d421d3c0aca49a7682b",
        "cc-pVDZ-JKFIT",
        True,
    ),
    "n2_ccpvdz_cas1010_df": (
        "b2414de49c6786716ec3d6a448c04f26d48e22ad5fa6f9a6099e02756e31726d",
        "cc-pVDZ-JKFIT",
        False,
    ),
    "n2_ccpvtz_cas1010_df": (
        "51ce5361bfee9dafae6c0584f5b8b85790a8aed06d03fd07d36f531a49c5e5c9",
        "cc-pVTZ-JKFIT",
        False,
    ),
}

# The golden arrays the NEVPT2 energies read, copied as they are, block2's
# PC-NEVPT2 answer (pc_class_energies, e_pc_total) among them: PySCF has no
# PC, so those two are the demo's block2 numbers, unchecked here.
COPIED = (
    "ci", "dm1", "dm2", "h1e", "h2e", "e_core", "e_virt",
    "h1e_v_Sr", "h1e_v_Si", "h1e_v_Sir",
    "h2e_v_Sr", "h2e_v_Si", "cvcv", "h2e_v_Sijr", "h2e_v_Srsi", "h2e_v_Srs",
    "h2e_v_Sij", "h2e_v1_Sir", "h2e_v2_Sir",
    "B_aa", "B_ca", "B_va", "B_cv",
    "class_norms", "class_energies", "pc_class_energies", "e_pc_total",
)  # fmt: skip

# check-added-large-files' --maxkb in .pre-commit-config.yaml: a default run
# refuses to leave a file over it in the tree.
LIMIT = 3072 * 1024

# Sr and Si, rebuilt here, against the golden's: measured <= 8e-16.
CHECK_TOL = 1e-12

# The golden scalar header after the 8-byte magic, per magic (golden_format.py's
# GOLDEN_HEADERS): ncas, nelec_a, nelec_b, ncore, ndet, e_casci,
# e_nevpt2_total, and in G2 reference_kind (0 casci, 1 casscf).
HEADERS = {b"NEVPT2G1": "<iiiiqdd", b"NEVPT2G2": "<iiiiqddi"}
HEADER_NAMES = (
    "ncas", "nelec_a", "nelec_b", "ncore", "ndet", "e_casci", "e_nevpt2_total",
    "reference_kind",
)  # fmt: skip


def read_golden(path: pathlib.Path) -> tuple[dict, dict[str, numpy.ndarray]]:
    """(header, arrays) of one `.nevpt2gold`: golden_format.read_golden's
    walk. After the header, an int32 count, then per array an int32 name
    length, the ASCII name, an int32 rank, int64 extents and C-order
    little-endian f64 data, ending exactly at the end of the file."""
    data = path.read_bytes()
    fmt = HEADERS.get(data[:8])
    if fmt is None:
        raise SystemExit(f"{path}: not a golden (magic {data[:8]!r})")
    at = 8 + struct.calcsize(fmt)
    header = dict(zip(HEADER_NAMES, struct.unpack_from(fmt, data, 8), strict=False))
    (count,) = struct.unpack_from("<i", data, at)
    at += 4
    arrays = {}
    for _ in range(count):
        (length,) = struct.unpack_from("<i", data, at)
        name = data[at + 4 : at + 4 + length].decode("ascii")
        at += 4 + length
        (rank,) = struct.unpack_from("<i", data, at)
        shape = struct.unpack_from(f"<{rank}q", data, at + 4)
        at += 4 + 8 * rank
        size = int(numpy.prod(shape, dtype=numpy.int64))
        arrays[name] = numpy.frombuffer(data, "<f8", size, at).reshape(shape).copy()
        at += 8 * size
    if at != len(data):
        raise SystemExit(f"{path}: {len(data) - at} bytes after the last array")
    if arrays["ci"].size != header["ndet"]:
        raise SystemExit(f"{path}: ci does not hold ndet elements")
    return header, arrays


def pyscf_rdms(ci, h2e, nelec) -> dict[str, numpy.ndarray]:
    """dm1, dm2, dm3, f3ca and f3ac of ci, as NEVPT.kernel builds them; h2e
    is the physicist active block, h2e[p, q, r, s] = (pr|qs)."""
    ncas = h2e.shape[0]
    ci = numpy.ascontiguousarray(ci)
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelec)
    aaaa = numpy.ascontiguousarray(h2e.transpose(0, 2, 1, 3))
    link = tuple(fci.cistring.gen_linkstr_index(range(ncas), n) for n in nelec)
    return {
        "dm1": dm1,
        "dm2": dm2,
        "dm3": dm3,
        "f3ca": nevpt2._contract4pdm(
            "NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelec, link
        ),
        "f3ac": nevpt2._contract4pdm(
            "NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelec, link
        ),
    }


def sr_si(a: dict[str, numpy.ndarray], rdms: dict, nelec) -> dict[str, tuple]:
    """(norm, energy) of the Sr and Si classes from the golden integrals and
    rdms: pyscf.mrpt.nevpt2.Sr's and Si's arithmetic on their `eris` branch,
    with the golden's blocks in place of the slices of eris it takes."""
    h1e, h2e, ci = a["h1e"], a["h2e"], numpy.ascontiguousarray(a["ci"])
    dm1, dm2, dm3 = rdms["dm1"], rdms["dm2"], rdms["dm3"]
    ncas = h1e.shape[0]
    dms = {"1": dm1, "2": dm2, "3": dm3, "4": None, **rdms}

    v, hv = a["h2e_v_Sr"], a["h1e_v_Sr"]
    a16 = nevpt2.make_a16(h1e, h2e, dms, ci, ncas, nelec)
    a17 = nevpt2.make_a17(h1e, h2e, dm2, dm3)
    a19 = nevpt2.make_a19(h1e, h2e, dm1, dm2)
    ener = (
        numpy.einsum("ipqr,pqrabc,iabc->i", v, a16, v)
        + numpy.einsum("ipqr,pqra,ia->i", v, a17, hv) * 2.0
        + numpy.einsum("ip,pa,ia->i", hv, a19, hv)
    )
    norm = (
        numpy.einsum("ipqr,rpqbac,iabc->i", v, dm3, v)
        + numpy.einsum("ipqr,rpqa,ia->i", v, dm2, hv) * 2.0
        + numpy.einsum("ip,pa,ia->i", hv, dm1, hv)
    )
    sr = nevpt2._norm_to_energy(norm, ener, a["e_virt"])

    v, hv = a["h2e_v_Si"], a["h1e_v_Si"]
    a22 = nevpt2.make_a22(h1e, h2e, dms, ci, ncas, nelec)
    a23 = nevpt2.make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = nevpt2.make_a25(h1e, h2e, dm1, dm2)
    delta = numpy.eye(ncas)
    dm3_h = numpy.einsum("abef,cd->abcdef", dm2, delta) * 2 - dm3.transpose(
        0, 1, 3, 2, 4, 5
    )
    dm2_h = numpy.einsum("ab,cd->abcd", dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2 * delta - dm1.transpose(1, 0)
    ener = (
        numpy.einsum("qpir,pqrabc,baic->i", v, a22, v)
        + numpy.einsum("qpir,pqra,ai->i", v, a23, hv) * 2.0
        + numpy.einsum("pi,pa,ai->i", hv, a25, hv)
    )
    norm = (
        numpy.einsum("qpir,rpqbac,baic->i", v, dm3_h, v)
        + numpy.einsum("qpir,rpqa,ai->i", v, dm2_h, hv) * 2.0
        + numpy.einsum("pi,pa,ai->i", hv, dm1_h, hv)
    )
    si = nevpt2._norm_to_energy(norm, ener, -a["e_core"])
    return {"Sr": sr, "Si": si}


def check(a: dict[str, numpy.ndarray], rdms: dict, nelec) -> float:
    """The largest |difference| of Sr's and Si's norm and energy from the
    golden's class_norms / class_energies (PySCF's class order: Sr first, Si
    second); raises on a miss."""
    worst = 0.0
    for k, (name, (norm, energy)) in enumerate(sr_si(a, rdms, nelec).items()):
        for got, want in (
            (norm, a["class_norms"][k]),
            (energy, a["class_energies"][k]),
        ):
            worst = max(worst, abs(got - want))
            if not abs(got - want) <= CHECK_TOL:
                raise SystemExit(f"{name}: {got!r} against the golden's {want!r}")
    return worst


def save(path: pathlib.Path, array: numpy.ndarray) -> int:
    """One '<f8' Fortran-order .npy; its size in bytes."""
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.stat().st_size


def convert(
    golden: pathlib.Path, sha256: str, out: pathlib.Path, with_rdms: bool
) -> int:
    """Check one golden and write its case directory; the bytes written."""
    digest = hashlib.sha256(golden.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{golden}: sha256 {digest}, expected {sha256}")
    header, a = read_golden(golden)
    nelec = (header["nelec_a"], header["nelec_b"])
    rdms = pyscf_rdms(a["ci"], a["h2e"], nelec)
    for name in ("dm1", "dm2"):
        if not numpy.array_equal(rdms[name], a[name]):
            raise SystemExit(f"{golden}: make_dm123's {name} is not the golden's")
    worst = check(a, rdms, nelec)

    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: a[name] for name in COPIED}
    if with_rdms:
        arrays |= {name: rdms[name] for name in ("dm3", "f3ca", "f3ac")}
    arrays["e_nevpt2_total"] = numpy.array(header["e_nevpt2_total"])
    written = sum(save(out / f"{name}.npy", arr) for name, arr in arrays.items())
    print(f"{out.name}: Sr/Si within {worst:.1e}, {written} bytes")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("demo", type=pathlib.Path, help="nevpt2-gpu-demo checkout")
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    parser.add_argument(
        "--all-rdms",
        action="store_true",
        help="write dm3, f3ca and f3ac for every case, not only the committed ones",
    )
    args = parser.parse_args()
    total = sum(
        convert(
            args.demo / "golden" / f"{case}.nevpt2gold",
            sha256,
            args.out / case,
            with_rdms or args.all_rdms,
        )
        for case, (sha256, _aux, with_rdms) in CASES.items()
    )
    print(f"total {total} bytes")
    if not args.all_rdms:
        over = [p for p in args.out.rglob("*.npy") if p.stat().st_size > LIMIT]
        if over:
            raise SystemExit(f"over the {LIMIT}-byte limit: {over}")


if __name__ == "__main__":
    main()
