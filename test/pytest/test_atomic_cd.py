"""tools/atomic_cd: the auxiliary-basis generator's math, checked against
PySCF and against its own guarantees.

PySCF lives in the opt-in ``tools`` dependency group, not ``dev``, so this
whole module skips (with a reason) wherever that group is not installed --
the push gate among them. Run it with the group::

    uv run --group tools pytest -rs test/pytest/test_atomic_cd.py

Every case here guards something that was once wrong or derived by hand:
Cholesky vectors used as contraction coefficients, s x d candidates at l=0,
the closed-form metric, the eight-fold packing, the per-l block structure
SVD contraction relies on, and the BSE writer's coefficient convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# before ANY third-party import: numpy is in the `tools` group too, not `dev`
pytest.importorskip("pyscf", reason="needs the `tools` dependency group")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from pyscf import df  # noqa: E402

from tools.atomic_cd import (  # noqa: E402
    cholesky,
    cli,
    compact,
    contract,
    df_check,
    eri_quartet,
)

_ORIGIN = (0.0, 0.0, 0.0)


@pytest.mark.parametrize("ell", range(9))
def test_coulomb_metric_matches_int2c2e(ell):
    """The closed form sech(ln(a/b)/2)^(l+1/2) is PySCF's unit-diagonal
    same-centre metric, over exponents six decades apart."""
    exponents = np.geomspace(1e-2, 1e4, 25)
    mol = eri_quartet.build_mol(
        [("H", _ORIGIN)], [[ell, [float(e), 1.0]] for e in exponents]
    )
    order = np.argsort([mol.bas_exp(ib)[0] for ib in range(mol.nbas)])
    first = mol.ao_loc[:-1][order]
    m = mol.intor("int2c2e")[np.ix_(first, first)]
    d = np.sqrt(np.diag(m))
    closed = compact.coulomb_metric(ell, exponents)
    assert np.abs(m / np.outer(d, d) - closed).max() < 1e-12
    assert np.all(np.diag(closed) == 1.0)


def test_effective_exponent():
    """Lehtola's Eq. 16: identity at l = L, and the paper's ~0.1375 for
    g x g -> s."""
    assert compact.effective_exponent(4, 4, 3.0) == pytest.approx(3.0, rel=1e-15)
    assert compact.effective_exponent(0, 8, 1.0) == pytest.approx(0.1375, abs=5e-4)


def test_build_candidates_obeys_gaunt_rule():
    """s x d couples to l=2 only (once wrongly to l=0); p x p to l=0 and 2,
    never 1."""
    sd = [((0, 1.0), (2, 2.0))]
    assert compact.build_candidates(sd, 0).size == 0
    assert compact.build_candidates(sd, 1).size == 0
    assert compact.build_candidates(sd, 2).tolist() == [3.0]
    pp = [((1, 1.5), (1, 1.5))]
    assert compact.build_candidates(pp, 1).size == 0
    assert compact.build_candidates(pp, 0).size == 1
    assert compact.build_candidates(pp, 2).tolist() == [3.0]


def test_pivoted_cholesky_bound_and_packed_equivalence():
    """Every residual below the threshold (Cauchy-Schwarz), each pivot used
    once, and the packed matrix decomposes exactly as the dense one."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((40, 25))
    base[:, 1] = base[:, 0] * (1 + 1e-9)  # a near-duplicate column
    dense = base.T @ base
    n = dense.shape[0]
    packed = eri_quartet.PackedSymmetric(dense[np.tril_indices(n)], n)
    threshold = 1e-8
    pivots, vectors = cholesky.pivoted_cholesky(dense, threshold)
    p_pivots, p_vectors = cholesky.pivoted_cholesky(packed, threshold)
    assert len(set(pivots)) == len(pivots)
    assert pivots == p_pivots
    np.testing.assert_allclose(vectors, p_vectors, rtol=0, atol=1e-12)
    assert cholesky.reconstruction_error(dense, vectors) <= threshold
    assert cholesky.reconstruction_error(packed, vectors) <= threshold


def test_packed_symmetric_matches_s4_eri():
    """s8 packing read three ways -- diagonal, full rows, lower-row blocks --
    against PySCF's s4 pair matrix."""
    mol = eri_quartet.build_mol([("C", _ORIGIN)], "cc-pvdz")
    dense = mol.intor("int2e", aosym="s4")
    n = dense.shape[0]
    packed = eri_quartet.PackedSymmetric(mol.intor("int2e", aosym="s8"), n)
    np.testing.assert_allclose(packed.diagonal(), np.diag(dense), atol=1e-14)
    for r in (0, 7, n // 2, n - 1):
        np.testing.assert_allclose(packed[r], dense[r], atol=1e-14)
    np.testing.assert_allclose(packed.lower_rows(0, n), np.tril(dense), atol=1e-14)


def test_screen_pairs_is_a_bounded_subset():
    prims = eri_quartet.primitives("C", "cc-pvdz")
    threshold = 1e-6
    pairs, summary = eri_quartet.screen_pairs("C", prims, threshold)
    assert pairs and set(pairs) <= set(eri_quartet.all_pairs(prims))
    assert summary["pairs"] == len(pairs) < summary["of_pairs"]
    assert summary["max_reconstruction_error"] <= threshold


@pytest.mark.parametrize(
    "element,l_obs,l_inc,expected",
    [("H", 2, 1, 3), ("C", 3, 1, 5), ("C", 3, 0, 4), ("Fe", 3, 1, 6), ("Cs", 1, 0, 6)],
)
def test_pruned_max_l(element, l_obs, l_inc, expected):
    """Eq. 9, max(2 l_occ, l_occ + l_obs + l_inc), with Yang's l_occ."""
    assert compact.pruned_max_l(element, l_obs, l_inc) == expected


def test_to_aux_shells_emits_pivots():
    """One single-primitive, unit-coefficient shell per pivot -- never the
    Cholesky vectors, which are Coulomb overlaps, not coefficients."""
    pairs = eri_quartet.all_pairs(eri_quartet.primitives("C", "cc-pvdz"))
    result = compact.compact_cholesky_for_l(pairs, 1, 1e-4)
    shells = compact.to_aux_shells(result)
    assert len(shells) == len(result.pivots)
    assert sorted(s.exponents[0] for s in shells) == sorted(
        result.exponents[result.pivots].tolist()
    )
    assert all(s.coefficients == ((1.0,),) for s in shells)
    assert compact.reconstruction_error(result) <= 1e-4


def test_contraction_blocks_are_per_l_and_m_independent():
    """W = J^T J couples only equal (l, m), and every m of one l carries the
    same block -- what lets ``contract`` solve one component per l."""
    aux = compact.to_pyscf_basis(
        cli.build_aux_shells("C", "cc-pvdz", 1e-4, contract_threshold=None)[0]
    )
    mol = eri_quartet.build_mol([("C", _ORIGIN)], "cc-pvdz")
    auxmol = df.make_auxmol(mol, aux)
    three = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1")
    j = three.reshape(-1, auxmol.nao) @ contract._inverse_sqrt(auxmol.intor("int2c2e"))
    w = j.T @ j
    ell = np.concatenate(
        [
            [auxmol.bas_angular(b)] * (2 * auxmol.bas_angular(b) + 1)
            for b in range(auxmol.nbas)
        ]
    )
    m = np.concatenate(
        [np.arange(2 * auxmol.bas_angular(b) + 1) for b in range(auxmol.nbas)]
    )
    same = (ell[:, None] == ell[None, :]) & (m[:, None] == m[None, :])
    scale = np.abs(w).max()
    assert np.abs(w[~same]).max() < 1e-10 * scale
    for l_value in set(ell):
        blocks = [
            w[np.ix_((ell == l_value) & (m == k), (ell == l_value) & (m == k))]
            for k in range(2 * l_value + 1)
        ]
        assert max(np.abs(b - blocks[0]).max() for b in blocks) < 1e-10 * scale


def test_bse_round_trip_reproduces_the_fit():
    """Written as BSE and read back, the contracted set fits exactly as the
    shells it came from: the coefficient convention survives the trip."""
    shells, _ = cli.build_aux_shells(
        "C", "cc-pvdz", cli.THRESHOLD[True], contract_threshold=cli.CONTRACT_THRESHOLD
    )
    direct = compact.to_pyscf_basis(shells)
    read_back = eri_quartet.basis_from_bse(cli.to_bse_spherical("C", shells), "C")
    assert df_check.aux_function_count("C", read_back) == df_check.aux_function_count(
        "C", direct
    )
    assert df_check.one_centre_error("C", "cc-pvdz", read_back) == pytest.approx(
        df_check.one_centre_error("C", "cc-pvdz", direct), rel=1e-8
    )


def test_default_sets_fit_co_and_ch4():
    """The default pipeline (contracted, unscreened) at cc-pVDZ: DF-RHF
    within 1e-4 Eh of conventional for CO and CH4 (measured: 2.6e-5 and
    5e-6, the one-centre floor), at no more than 80 functions on C (71)."""
    aux = {
        el: compact.to_pyscf_basis(
            cli.build_aux_shells(
                el,
                "cc-pvdz",
                cli.THRESHOLD[True],
                contract_threshold=cli.CONTRACT_THRESHOLD,
            )[0]
        )
        for el in ("C", "H", "O")
    }
    assert df_check.aux_function_count("C", aux["C"]) <= 80
    for atoms in df_check.MOLECULES.values():
        _e, delta = df_check.rhf_energy_error(
            atoms, "cc-pvdz", {s: aux[s] for s, _ in atoms}
        )
        assert abs(delta) < 1e-4


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--threshold", "0"], "--threshold must be > 0"),
        (["--threshold=-1e-6"], "--threshold must be > 0"),
        (["--threshold", "nan"], "--threshold must be > 0"),
        (["--contract-threshold", "0"], "--contract-threshold must be > 0"),
        (["--max-l", "-1"], "--max-l must be >= 0"),
        (["--prune-l", "-5"], "--prune-l must be >= 0"),
        (
            ["--no-contract", "--contract-threshold", "1e-3"],
            "--contract-threshold has no effect with --no-contract",
        ),
    ],
)
def test_cli_rejects_bad_arguments(argv, message, capsys):
    """Each of these once crashed deep inside the pipeline or was silently
    ignored; now argparse names the flag."""
    with pytest.raises(SystemExit) as exit_info:
        cli._parse_args(["C", "cc-pvdz", *argv])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_defaults():
    """Contraction on at 1e-5 over a 1e-7 primitive pass; without it, a
    1e-4 primitive set and no contraction threshold."""
    args = cli._parse_args(["C", "cc-pvdz"])
    assert (args.contract, args.threshold, args.contract_threshold) == (
        True,
        1e-7,
        1e-5,
    )
    assert not args.screen and args.prune_l is None and args.max_l is None
    args = cli._parse_args(["C", "cc-pvdz", "--no-contract"])
    assert (args.threshold, args.contract_threshold) == (1e-4, None)
    assert cli._parse_args(["C", "cc-pvdz", "--prune-l"]).prune_l == 1
