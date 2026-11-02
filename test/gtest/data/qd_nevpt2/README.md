# qd_nevpt2 golden data

SC-QD-NEVPT2 references, written by `tools/emit/qd_nevpt2_reference.py`.
No file here is hand-edited. Neither PySCF 2.14 nor block2 0.5.4 implements
QD-NEVPT2, so this is this tree's own, built over PySCF's `mrpt.nevpt2`
class arithmetic and `dfnevpt2`'s fitted integrals and checked two ways
(below). The C++ follow-up is `nevpt2::sc_effective_hamiltonian` (#673); the
PC-QD-NEVPT2 references (#671) reuse these cases, orbitals and transition
RDMs, and add their arrays beside these ones.

## Cases

All density-fitted throughout over this tree's `data/basis` and
`data/aux_basis` files (cc-pVDZ-JKFIT for every case), singlets,
n_alpha = n_beta.

| case | basis | ncore | CAS (e, o) | nvirt | naux | roots, weights | why |
|---|---|---:|---|---:|---:|---|---|
| `h2co_ccpvdz_sa2` | cc-pVDZ | 6 | (4, 3) | 29 | 186 | 2, [0.5, 0.5] | `casscf/casscf_df.json`'s; C2v, roots 1A1 and 1A2, so every off-diagonal is zero by symmetry (≤ 8e-16 here) |
| `h2co_ccpvdz_sa2_weighted` | cc-pVDZ | 6 | (4, 3) | 29 | 186 | 2, [0.7, 0.3] | the same at unequal weights: the SA Fock, and so the orbitals and every number, move |
| `h2co_skew_ccpvdz_sa3` | cc-pVDZ | 6 | (4, 3) | 29 | 186 | 3, equal | `casscf_nac/casscf_nac.json`'s; C1, no coupling vanishes (max \|H_IJ\| 4.4e-3, roots 0.11 and 0.19 Eh apart) |
| `h6_631g_sa2` | 6-31G | 2 | (2, 3) | 7 | 138 | 2, [0.5, 0.5] | brute-force: H₆, C1, 48 400 determinants over all 12 orbitals |
| `h8_sto3g_sa3` | STO-3G | 2 | (4, 4) | 2 | 184 | 3, equal | brute-force and **near-degenerate**: roots 1 and 2 are 19 mEh apart and coupled by 4–5 mEh (a quarter of their gap), which moves their SC-QD energies 1.7 mEh off the diagonal's; 4 900 determinants over all 8 orbitals |

No existing DF SA-CASSCF case has a near-degeneracy, so `h8_sto3g_sa3` is
new, and it is also small enough for the brute force. The formaldehyde
cases start, polish and gate as `tools/emit/nevpt2_reference.py` does
(mc1step polished to |g| < 1e-10, each root within 1e-6 and the weighted
energy within 1e-9 of the source file's); the hydrogen cases are mc1step DF
SA-CASSCFs, converged, not polished. Geometries, sources and energies are in
`qd_nevpt2.json`.

## The states and orbitals

Per case: a multi-root DFCASCI over the SA-CASSCF orbitals, its CI vectors
replaced by the dense singlets of the same Hamiltonian, each **signed** so
that its first coefficient (row-major, alpha string then beta string)
within 1e-8 of its largest is positive (`casscf_nac/`'s rule). The core and
virtual orbitals are canonicalized over the **state-averaged** Fock at the
case's weights (what `CasscfOutput::mo_energies` holds) and are shared by
every state; the active orbitals are **not** rotated, so the CI vectors are
those of the SA-CASSCF's active space. Every integral and RDM is over these
orbitals.

## Conventions

- **Bra and ket.** For an ordered pair (I, J), I is the bra and J the ket:
  `pair_I_J/dm1[p, q]` = ⟨I|E_pq|J⟩, `dm2[p, q, r, s]` = ⟨I|E_pq E_rs|J⟩,
  `dm3[p, q, r, s, t, u]` = ⟨I|E_pq E_rs E_tu|J⟩ (`make_dm123`'s ordering,
  **not** reordered). `pair_I_I` is state I's own.
- **dm1 is the transpose of `make_dm123`'s.** At I ≠ J, PySCF's
  `fci.rdm.make_dm123(..., cibra=I, ciket=J)` returns dm1 as ⟨I|E_qp|J⟩ at
  [p, q] (as `trans_rdm1` does), while its dm2 and dm3 are in the order
  above. Here dm1 is in dm2's order, which the norm einsums read. All of
  dm1–dm4 are checked against explicit products of the spin-summed
  excitation operators on the CI vectors (≤ 8.1e-15).
- **f3ca, f3ac** are `NEVPTkern_cedf_aedf` / `NEVPTkern_aedf_ecdf`'s
  contractions of the transition 4-RDM (`make_dm1234('FCI4pdm_kern_sf', I,
  J)`, raw) with h2e, written as the einsums `make_a16`'s comments give:
  `f3ca = einsum('kcij,rpqbajki->rpqbac', h2e, dm4)`,
  `f3ac = einsum('ijka,rpqbjcik->rpqbac', h2e, dm4)`, h2e physicist. At
  I = J they equal PySCF's `_contract4pdm` (≤ 1.4e-14). (`make_a16`'s third
  comment, `'kbij,rpqjkiac'` = `f3ca.transpose(1,4,0,2,5,3)`, is not an
  identity even at I = J.)
- **Classes** are PySCF's order: Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir.
  A perturber t is one of PySCF's tuples in a class (one external
  occupation pattern).
- **Transition norms.** `t_norms[I, J, k]` = Σ_{t ∈ k} ⟨I|V P_t V|J⟩,
  symmetric in I and J. Per t it is the class's PySCF norm einsum over the
  transition RDMs with two changes, which the brute force found and #673
  needs:
  1. The zero-body terms are ⟨I|J⟩ = δ_IJ times a number: the 2δ of hdm1
     (Sijr) and dm1_h (Si), the δδ terms of hdm2 (Sij), Sir's
     h1e_v·h1e_v, and the whole of Sijrs. At I ≠ J they are dropped.
  2. Sr's, Si's and Sir's one-electron cross terms fold two Hermitian
     conjugates into one (PySCF's `* 2.0`, Sir's `* 4.0` and `* 2.0`),
     which holds only at I = J. So per t the norm is the **mean** of the
     einsum over (I, J) and over (J, I). The other five classes' einsums are
     symmetric already. `qd_nevpt2.json`'s `norm_einsum_asymmetry` gives,
     per case and class, the largest |einsum(I, J) − einsum(J, I)| per
     perturber (up to 1.2e-2 for Si on the skew case; round-off where
     the symmetry zeroes it).
- **Couplings (the sign).** `couplings[I, J, k]` = ⟨I|V|Ψ_J⁽¹⁾⟩ from class
  k = −Σ_{t ∈ k} T^t_IJ / (Δ_t + H^J_t / N^J_t). Δ_t, H^J_t and N^J_t are
  **J's own**, the ket's, as in `class_energies`: N^J_t its norm, H^J_t
  PySCF's Dyall part (`ener`/`h`, from J's dm1–dm3, f3ca, f3ac), Δ_t the
  orbital-energy gap. The sum runs over the t with |N^J_t| > 1e-14
  (`_norm_to_energy`'s guard); Sijrs has no guard and H = 0, as in PySCF.
  So a coupling is negative-definite in the sense of a second-order energy
  (the diagonal is each root's SC-NEVPT2 correlation energy), and
  `couplings` is **not** symmetric.
- `class_norms[I, k]`, `class_energies[I, k]` are root I's own N and
  energy per class: exactly PySCF's `mrpt.NEVPT(casci, root=I)` on these
  orbitals (`class_energies[I, k]` = `couplings[I, I, k]`).
- `heff2[I, J]` = Σ_k `couplings[I, J, k]`; `heff` = diag(`casci_energies`)
  + `heff2`, row = bra, column = ket, unsymmetrized; `heff_sym` =
  ½(`heff` + `heff`ᵀ). `eigenvalues` (ascending) and `eigenvectors`
  (columns, each signed so its largest-magnitude component is positive) are
  `heff_sym`'s: the SC-QD-NEVPT2 total energies and states in the CASCI
  basis.

## Files

One `<case>/<array>.npy` per array, `'<f8'`, Fortran order (the layout
`src/io/npy` reads; `np.load` returns the logical, C-indexed tensor). Index
names: `c` core, `a` active, `v` virtual, `L` auxiliary, `n` roots.

| array | shape | what |
|---|---|---|
| `B_aa`, `B_ca`, `B_va`, `B_cv` | (L, ·, ·) | MO three-index blocks, (pq\|rs) = Σ_L B[L,p,q] B[L,r,s] (metric folded in), checked against `dfnevpt2._ERIS` to 1e-12 |
| `h1e` | (a, a) | `h1eff`'s active block |
| `h2e` | a⁴ | physicist, `h2e[p,q,r,s]` = (pr\|qs) |
| `e_core`, `e_virt` | (c,), (v,) | SA-canonical orbital energies |
| `h1e_v_Sr`, `h1e_v_Si`, `h1e_v_Sir` | (v, a), (a, c), (v, c) | as in `../nevpt2/README.md` |
| `ci` | (n, na, nb) | the signed CI vectors, PySCF `cistring` order |
| `casci_energies` | (n,) | the roots' total CASCI energies |
| `weights` | (n,) | the SA weights |
| `pair_I_J/dm1`, `dm2`, `dm3` | a², a⁴, a⁶ | transition RDMs, bra I (above) |
| `pair_I_J/f3ca`, `f3ac` | a⁶ | transition f3ca, f3ac, bra I |
| `t_norms`, `couplings` | (n, n, 8) | [I, J, k], above |
| `class_norms`, `class_energies` | (n, 8) | each root's own |
| `heff2`, `heff`, `heff_sym` | (n, n) | above |
| `eigenvalues`, `eigenvectors` | (n,), (n, n) | `heff_sym`'s |

The inputs `nevpt2::EnergyInputs` reads are the first five rows plus a
pair's RDMs; the external four-index blocks are not written (the slabs come
from B). `qd_nevpt2.json` has the provenance, each case's geometry,
dimensions, CASSCF and CASCI energies, `heff`, `eigenvalues`, the largest
off-diagonal and its ratio to its roots' gap, and every check's achieved
error.

## The checks before writing

Every case, or nothing is written:

1. **Diagonal vs PySCF.** Each root's per-class norm and energy against
   PySCF's `mrpt.NEVPT(casci, root=I)` with `canonicalized=True` on the
   SA-canonical orbitals, its DF branch, to 1e-10. Measured ≤ 5.8e-16.
2. **Transition RDMs** dm1–dm4 against the explicit operator products, to
   1e-12 (≤ 8.1e-15); f3ca/f3ac at I = J against `_contract4pdm`, to 1e-12
   (≤ 1.4e-14).

The brute-force cases (`h6_631g_sa2`, `h8_sto3g_sa3`) also:

3. **Full determinant space.** Over all orbitals, with H built from the
   same h1 and the fitted (pq\|rs) = Σ B B, V|J⟩ restricted to each external
   occupation pattern t is H|J⟩ restricted to it (H_D keeps every pattern),
   and the Dyall Hamiltonian H_D = Σ_c ε_c n_c + Σ_r ε_r n_r + H_cas
   (h1eff's active block and (aa\|aa)) is applied in the full space. From
   those, per pattern, N^J_t, ⟨I|V P_t V|J⟩ and ⟨H_D⟩_t − ⟨J|H_D|J⟩, so every
   element of every class's `t_norms` and `couplings`, and `heff2`, to
   1e-9. Measured ≤ 2.7e-16.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/qd_nevpt2_reference.py
```

BLAS and OpenMP are pinned to one thread and nothing run-dependent is
written, so a rerun is byte-identical. It takes about 80 s.
