# Compressing the 3-RDM: two exact factorizations

A design note, not an implementation. It records what was measured on this
directory's case, and two ways to shrink what SC-NEVPT2 needs from dm3 that
introduce **no error beyond density fitting itself**. Each claim is marked
*measured* (on `glycolaldehyde_ccpvdz_cas1010_casscf_df/`), *exact*
(follows from the algebra), or *unverified* (derived here, not yet checked
numerically).

## What the case says about dm3

n = 10 active orbitals, so dm3 has n⁶ = 10⁶ elements. Γ3 below is the
normal-ordered 3-RDM, `Γ3[p,q,r,s,t,u] = ⟨p† r† t† u s q⟩`
(`fci.rdm.reorder_dm123` of the raw dm3, which is `⟨E_pq E_rs E_tu⟩`).

**Element-wise it is dense** (*measured*). Fraction of elements above a
threshold:

| \|x\| > | 1e-2 | 1e-4 | 1e-6 | 1e-8 | 1e-12 |
|---|---|---|---|---|---|
| raw dm3 | 2.4% | 27% | 88% | 98.3% | 98.4% |
| Γ3 | 2.1% | 26% | 88% | 97.9% | 98.0% |

The magnitudes fall off smoothly over six decades, with no gap to cut at, and
the molecule is C1, so there are no point-group zeros. Thresholding elements is
not a compression worth having.

**As an n³ × n³ matrix, Γ3 is low-rank** (*measured*). Rows (p,r,t), columns
(q,s,u). It is symmetric to 3e-15 and PSD, smallest eigenvalue −1e-14:

| | 1e-2 | 1e-3 | 1e-4 | 1e-6 | exact |
|---|---|---|---|---|---|
| rank for that relative Frobenius error | 133 | 277 | 406 | 649 | **780** |

The top eigenvalues are about 23; eigenvalue 100 is 0.39, 400 is 1.6e-3,
600 is 6e-5. The raw dm3 is not symmetric under this pairing (it is off by
3.9), so its eigenvalues do not give its rank.

**Not low-rank along a single pair mode** (*measured*). Unfolded as
(pq) × (rstu), Γ3 needs 69 of 100 directions for 1e-2 and 98 for 1e-4. Γ3
on its own is no more Tucker-compressible along a pair than a density matrix
is. In DF, the low rank comes from the operator, not from D, and that is what
idea 1 below uses.

## Idea 2: the S3 irreps of the triple index — exact, any state

Γ3 is unchanged when its three (creation, annihilation) pairs are permuted
together: `Γ3[πP, πQ] = Γ3[P, Q]` for π ∈ S3 acting on the triples
P = (p,r,t) and Q = (q,s,u). By Schur's lemma the n³-dimensional triple
space splits into S3 isotypic components, and Γ3 is block-diagonal across
them:

| irrep | copies of the irrep (n = 10) | Γ3 there |
|---|---|---|
| symmetric | C(n+2,3) = 220 | **exactly zero** |
| antisymmetric | C(n,3) = 120 | one 120 × 120 block |
| mixed (2-dim) | n(n²−1)/3 = 330 | one 330 × 330 block, repeated twice identically |

(220 + 120 + 2·330 = 1000.)

**The symmetric irrep is null** (*exact*; *measured* as the rank 780 =
1000 − 220). The spin-summed creation triple a†_p a†_r a†_t is antisymmetric
overall. A part symmetric in all three orbitals would need a fully
antisymmetric spin state of three spin-½ electrons, and none exists.

So Γ3 is determined by 120² + 330² = 123 300 numbers instead of 10⁶, about
8× smaller before Hermiticity, with zero error, for any state and any
integrals.

To use it, a consumer projects each integral triple onto the antisymmetric
and mixed irreps (n³ work per external index), then runs plain GEMMs at
dimensions 120 and 330. It is a constant factor, not a change in scaling.

It applies to the normal-ordered Γ3. The raw dm3 is Γ3 plus terms made of
dm1 and dm2 (`dm3 − reorder_dm123(dm3)` depends on dm1 and dm2 only), which
are kept separately.

## Idea 1: fit the perturber operators with the integrals' own DF factor

### The structure

With DF integrals, `(pq|rs) = Σ_L B[L,pq] B[L,rs]`, the two-electron
Hamiltonian over all orbitals is

    Ĥ₂ = ½ Σ_L ρ̂_L ρ̂_L − (one-body exchange term),   ρ̂_L = Σ_pq B[L,pq] E_pq.

Every SC-NEVPT2 class has a perturber V_k, the part of Ĥ|Ψ⟩ with given
external indices. Its norm is ⟨V†V⟩, and its energy ⟨V†[Ĥ_act, V]⟩.
PySCF expands both into RDMs; dm3 and the 4-RDM (through f3ca and f3ac) are
what that expansion costs. Keep the operators instead, and dm3 can enter in
only two roles, both of which DF factors exactly:

- **A, the perturber role.** One ρ̂_L carries the external indices and the
  other an active–active density ρ̂_L^{aa}, which lands on one E_pr of the
  RDM.
- **B, the Hamiltonian role.** dm3 or the 4-RDM appears only from expanding
  [Ĥ_act, ·], and Ĥ_act's two-electron part is
  Σ_K ρ̂_K^{aa} ρ̂_K^{aa} (B_aa reproduces (tu|vw) here to 8.9e-16,
  *measured* by `generate.py`).

### Sr, worked through

From `pyscf.mrpt.nevpt2.Sr`: `h2e_v[i,p,q,r] = ppaa[i,q,p,r] = (iq|pr)`, with
i virtual, and the norm's dm3 term is
`einsum('ipqr,rpqbac,iabc->i', h2e_v, dm3, h2e_v)`. Raw dm3 `[r,p,q,b,a,c]`
is ⟨E_rp E_qb E_ac⟩. So the two active–active densities (pr) and (ac)
fall exactly on the outer E's, and both external–active densities on the
middle one:

    norm_i ∋ Σ_{L,M,q,b} B[L,iq] B[M,ib] G[L,M,qb],
    G[L,M,qb] = ⟨Ψ| ρ̂_L E_qb ρ̂_M |Ψ⟩,   ρ̂_L = Σ_pr B_aa[L,pr] E_rp.

Since raw dm3 is a plain product of E's, this needs no normal-ordering
correction (*exact*). The energy is the same object with Ĥ_act between the
two sides. With φ_{M,b} = a_b ρ̂_M|Ψ⟩ in the N−1 space,

    ener_i ∋ Σ B[L,iq] B[M,ib] ⟨φ_{L,q}| [Ĥ_act, ·] |φ_{M,b}⟩,

computed by applying Ĥ_act to the vectors φ (the `ci` module's sigma).
**f3ca and f3ac never appear**: they exist only because PySCF folds Ĥ_act in
at the RDM level.

### Every class

Read off the PySCF code. "Norm" and "energy" say where dm3 or f3 enter.

| class | active part of V | dm3 in norm | dm3 / f3 in energy | role |
|---|---|---|---|---|
| Sr | a_b ρ̂^{aa}_M (N−1) | yes, ⟨E_rp E_qb E_ac⟩ | a16 (with f3ca, f3ac) | A, and B inside the energy |
| Si | a†_b ρ̂^{aa}_M (N+1) | yes, dm3_h | a22 (with f3ca, f3ac) | A, and B inside the energy |
| Srs | a_t a_u (N−2) | no (dm2) | a7 | B only |
| Sij | a†_t a†_u (N+2) | no (dm2) | a9, via hdm3 | B only |
| Sir | E_tu and ρ̂^{aa}_L | no (dm2) | a12, a13 | B only (A appears, but at dm2 level) |
| Sijr, Srsi | one a or a† | no | no | — |
| Sijrs | none | no | no | — |

**No class contracts dm3 with a non-DF two-electron integral** (*exact*,
from the code). The non-DF inputs, h1eff and the orbital energies, are
one-body or diagonal and enter unchanged. So for a DF-NEVPT2 (this case is
one, and the tree's ERI step is density-fitted), idea 1 reproduces every
class with no error beyond DF.

What the fitted objects buy, and who shares them:

- ρ̂_L^{aa}, the aux one-body operators: every class.
- ρ̂_L|Ψ⟩: Sr, Si and Sir.
- a_b ρ̂_M|Ψ⟩ (Sr) and a†_b ρ̂_M|Ψ⟩ (Si), each with its Ĥ_act image: one
  class each. L and M batch as aux batches do in DF-K.

### Where B may be truncated, and where it may not

B_aa's exact rank is min(naux, n(n+1)/2), 55 here (*measured*: 36 / 49 / 55
directions for 1e-2 / 1e-3 / 1e-4). Compressing to the exact rank costs
nothing.

**In role A, truncating below that rank is an added, controllable error.**

**In role B it is not allowed.** The rewriting of ⟨V†ĤV⟩ − E₀⟨V†V⟩ as a
commutator (a16, a22, a7, a9, a12, a13 all rest on it) uses
Ĥ_act|Ψ⟩ = E₀|Ψ⟩. That holds only for the Hamiltonian the CASCI
diagonalised. A truncated B there gives a different Ĥ, whose error does not
shrink like a DF error. Role B costs Ĥ applications, not n⁶ contractions,
so there is nothing to win by truncating it anyway.

### Scaling, honestly

- **At n = 10 there is no DF-specific gain.** G is 55 × 55 × 100 ≈ 3×10⁵
  against Γ3's 10⁶, which is just the integrals' (pr)↔(rp) symmetry. DF never
  paid off at small n either.
- **The gain needs naux_eff < n(n+1)/2.** If the active-pair rank grows like
  about c·n, as aux sets do, G grows like c²n⁴ against n⁶.
- **The cost moves to Ĥ applications.** It becomes Ĥ applications on about
  n·r_A vectors of length ndet_{N±1}, a few times 10⁴ each here. In an exact
  FCI space that grows faster than n⁸, so the clear win is where the CI is
  itself compressed (selected CI, DMRG — where one-body operators are cheap
  MPOs and 3- and 4-RDMs are the expensive part), or where r_A ≪ n².

## How the two combine

They do not fully stack. Idea 1 treats the outer two pairs of dm3 (fitted
on L and M) differently from the middle one (q,b), which breaks the
three-fold permutation. Only the swap of the outer two pairs (L ↔ M)
survives, worth about 2×. Idea 1 also wants the raw dm3, a product of E's,
while idea 2 wants the normal-ordered Γ3. Idea 2 is a fixed factor of about
8; idea 1's gain grows with n.

## Unverified, and the first checks

1. **The one-body exchange term.** The term from writing Ĥ₂ as ½ρ̂ρ̂ must
   match PySCF's own one-body folding (the `'mbbn->mn'` subtractions in its
   h1e_v) term for term. A mismatch would show as an O(1) error.
2. **Operator orderings.** Each class's V has a definite ordering (a_b ρ̂_M
   or ρ̂_M a_b, and so on). The orderings differ by exact one-body terms with
   deltas, but those terms must be carried.
3. **Reproduce Sr exactly.** Build G and the φ_{M,b} for this case and
   reproduce Sr's norm and energy to about 1e-14, with f3ca and f3ac unused.
   Then Si, then the role-B classes.
4. **Scaling needs a series.** One molecule with growing active spaces:
   glycolaldehyde CAS(12,12) and CAS(14,14) from `generate.py`, about 0.85M
   and 11.8M determinants. Measure B_aa's active-pair rank and Sr's error
   against r_A as n grows.
5. **Idea 2 directly.** Project Γ3 onto the three irreps and confirm the
   block structure: symmetric block zero, mixed blocks identical.

Prior art: avoiding the 4-RDM in NEVPT2 by working with N±1 perturber
vectors is known in spirit. Before building idea 1, check whether this form
— an a priori aux basis for the perturber operators — is already published.
