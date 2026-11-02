# nevpt2_transition golden data

PySCF's transition dm1, dm2, dm3, f3ca and f3ac between two different CI
vectors, the oracle of `nevpt2::rdms`' bra/ket overload
(`test/gtest/nevpt2_rdm_gpu.cpp`, `NevptRdm.Transition*`). Written by
`tools/emit/nevpt2_transition_golden.py`; no file here is hand-edited.

## Cases

Each takes its active (tu\|vw) from a committed `../nevpt2/<source>/h2e.npy`
(a contiguous block of its active orbitals) and two normalised CI vectors:
the source's own CI vector where the space is the source's, otherwise
uniform random ones from numpy's `default_rng(seed)`. A random vector has no
spin or spatial symmetry, so no element of any tensor vanishes by symmetry.

| case | source | orbitals kept | (n_α, n_β) | determinants | bra | ket | seed |
|---|---|---|---|---:|---|---|---:|
| `n2_cas44` | `n2_ccpvdz_cas44_df` | all 4 | (2, 2) | 36 | the source's CI vector | random | 672 |
| `n2_cas6_3a2b` | `n2_ccpvdz_cas88_df` | 1–6 of 0–7 | (3, 2) | 300 | random | random | 673 |

The second has n_α ≠ n_β, so its alpha and beta string sets differ.

## Files

One `<case>/<array>.npy` per array, `'<f8'`, Fortran order (the layout
`src/io/npy` reads; `np.load` returns the logically indexed C-order tensor).
I is the bra and J the ket.

| array | shape | what |
|---|---|---|
| `bra`, `ket` | (na, nb) | the two CI vectors, PySCF `cistring` order |
| `h2e` | a⁴ | physicist: `h2e[p,q,r,s] = (pr\|qs)`, the source's block |
| `dm1` | a² | ⟨I\|E_pq\|J⟩: **the transpose** of `fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, ...)`'s dm1, which is ⟨I\|E_qp\|J⟩ |
| `dm2`, `dm3` | a⁴, a⁶ | ⟨I\|E_pq E_rs\|J⟩, ⟨I\|E_pq E_rs E_tu\|J⟩: that call's, raw (not reordered) |
| `f3ca` | a⁶ | Σ_ijk (ki\|uj) dm4[p,q,r,s,t,j,k,i] over that call's raw transition dm4 |
| `f3ac` | a⁶ | Σ_ijk (ik\|jt) dm4[p,q,r,s,j,u,i,k], the same |

The two f3 forms are those `pyscf.mrpt.nevpt2.make_a16` comments `f3ca` and
`f3ac` with: `_contract4pdm`'s 'NEVPTkern_cedf_aedf' and
'NEVPTkern_aedf_ecdf' of one vector, with the transition 4-RDM in its place.
PySCF's `_contract4pdm` takes one vector, so it cannot give these directly.

## The checks before writing

For every case the script requires, to 1e-12 of each tensor's max:

- the two f3 forms to be `_contract4pdm`'s at bra = ket, for each of the
  two vectors;
- dm1..dm4 to be the definitions above, against a brute-force build over
  dense E_pq matrices in the determinant basis;
- `make_dm123(bra, ket)`'s dm1, dm2 and dm3 to be `make_dm1234`'s.

Measured: within 6.3e-16. On a miss it writes nothing for that case.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/nevpt2_transition_golden.py
```

PySCF 2.14.0. BLAS and OpenMP are pinned to one thread, so a re-run is
byte-identical. It takes about a second.
