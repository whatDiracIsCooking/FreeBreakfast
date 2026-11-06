# NAC

The nonadiabatic (derivative) coupling between two roots of a converged
DF-SA-CASSCF at any weights, on the GPU: `mcscf::df_nac`,

    d_IJ = ⟨I|∂J/∂R⟩ = h_IJ / (E_J − E_I) + d^CSF_IJ

with its numerator h_IJ and CSF term returned on their own.

## Module

`free_breakfast.mcscf.nac`, target `free_breakfast::mcscf::nac`, re-exported
by `free_breakfast.mcscf`.

```cpp
Result<mcscf::Nac> mcscf::df_nac(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t bra,
    int32_t ket, const mcscf::NacOptions &options = {});
// d->numerator (h_IJ), d->csf (d^CSF), d->total: each n_atom x 3, row-major,
// element 3 A + k; d->iterations, d->residual: the Z-vector's
```

Every argument through `ncas` is `df_gradient`'s ([`../gradient/`](../gradient/README.md)),
so a caller (a MECI search) takes g_I, g_J and h_IJ from one SA solve.

| `NacOptions` | Default | Settles |
|---|---|---|
| `mult_ediff` | false | `total` is h_IJ + (E_J − E_I) d^CSF, finite at a degeneracy, rather than h_IJ / (E_J − E_I) + d^CSF |
| `use_etfs` | false | the CSF term left out of `total` |
| `solve_tolerance` | 1e-10 | the Z-vector's tolerance before the pair's gap scaling ("The Z-vector's tolerance"); above 0. The combination overload takes no options and solves to the default |

## Conventions

**Which state is the bra.** `bra` is I and `ket` is J, and the result is
⟨I|∂J/∂R⟩, PySCF's `NonAdiabaticCouplings.kernel(state=(I, J))` (whose
docstring has the order backwards; `test/gtest/data/casscf_nac/README.md`
shows it by a finite difference). So d_IJ = −d_JI, h_IJ = h_JI, and
d^CSF_IJ = −d^CSF_JI.

**Signs.** d_IJ changes sign with either root's CI phase, as `solution`
carries it; the call fixes no phase. The orbitals' signs do not enter.

**Options**, as PySCF's of the same names. `total` is

| `mult_ediff` | `use_etfs` | `total` |
|---|---|---|
| false | false | h_IJ / (E_J − E_I) + d^CSF (the default) |
| false | true | h_IJ / (E_J − E_I) |
| true | false | h_IJ + (E_J − E_I) d^CSF |
| true | true | h_IJ |

`numerator` and `csf` are the same whatever the options: h_IJ and d^CSF,
unscaled. Without `mult_ediff` a gap below 1e-10 hartree is refused
(`Unsupported`).

## The numerator

At fixed orbitals ⟨I|H|J⟩ is the CASSCF energy functional at the transition
RDMs, less E_nuc and the core's own energy (both multiply ⟨I|J⟩ = 0); since H
is real and the functional reads only the symmetric parts, the RDMs are the
symmetrized ones, `ci::Rdm::trans_rdm12(I, J)`. h_IJ is its total derivative
with the SA orbitals and CI relaxed, so it is the derivative of the Lagrangian

    L = E_act(γ^IJ, Γ^IJ) + z_o · g_o + Σ_r z_c,r · g_c,r,

g the SA stationarity conditions, exactly as a root's gradient
([`../gradient/README.md`](../gradient/README.md) "The state-averaged
Lagrangian") with E_I replaced by the transition functional:

1. **The transition RDMs** γ, Γ (`trans_rdm12`).
2. **The right-hand side**, the transition functional's gradient in the SA
   parameters. Its orbital part is `mcscf::transition_orbital_gradient`
   ([`../transition/`](../transition/README.md)), `response::fock`'s
   gradient at (γ, Γ) less its gradient at zero RDMs
   (`../gradient/README.md` "A transition density"), this tree's
   g = 2 (F − Fᵀ), equal to PySCF's `gen_g_hop_active` vector. **Its CI part y and its part t along the
   rotations among the roots are zero**, at any weights ("Unequal
   weights" below).
3. **The Z-vector**, `response::zvector` (linked PRIVATE) over the
   CI-relaxed Hessian at the solution's orbitals, roots and weights, built
   as `df_gradient` builds it, with no CI or rotation right-hand side:
   (z_o, z_c, Θ), Θ zero at equal weights.
4. **The terms**, one `df_transition_gradient` at (γ, Γ) with those
   multipliers: the Hellmann–Feynman terms with no core–core part, the
   transition RDMs of z_c (with Θ folded in as its part along the roots)
   added, and the orbital response along κ(z_o) at the SA RDMs, in one
   Γ_mnP and one W_PQ. Its total is h_IJ.

h_IJ is translation-invariant (Σ_A h_IJ = 0), as any derivative of an
energy-like functional is.

### Unequal weights

When the weighted γ is the roots' plain average (to 1e-10) the weights are
taken as exactly 1 / nroots, as they always were; otherwise they are
`sa_gradient_weights`', as `df_gradient` takes them: the solution's own, or
recovered from the RDMs when it carries none
([`../gradient/README.md`](../gradient/README.md)).

**Why the right-hand side has no rotation part.** The CI half of d_IJ comes
from the roots' eigen-condition, which holds at every geometry and at any
weights, ⟨c_I|H₀|c_J⟩ = 0 for I ≠ J (H₀ the active-space Hamiltonian at the
SA orbitals). Write ∂c_J = Σ_s A_sJ c_s + (a part off the roots), A
antisymmetric by orthonormality; differentiating,

    A_JI E_J + c_I · (dH₀/dR) c_J + E_I A_IJ = 0,   so   ⟨c_I|∂c_J⟩ = A_IJ = h_IJ / (E_J − E_I),

h_IJ = c_I · (dH₀/dR) c_J with c_I and c_J **held**, dH₀/dR the total
derivative through the SA orbitals' response. So h_IJ is E_act(γ^IJ, Γ^IJ)
differentiated with the orbitals relaxed and the pair's CI vectors fixed: it
depends on none of the SA CI parameters, neither z_c's directions off the
roots nor the rotations Θ among them, and its gradient along them is zero,
y = 0 and t = 0 (`../../response/zvector/README.md` "The rotations among the roots").
Holding ⟨I|H|J⟩ a function of the responding roots instead, with
∂/∂Θ_JI = E_J − E_I, would differentiate a quantity that is zero at every
geometry, and give zero: the pair's own rotation A_IJ is the coupling being
computed, not a response to eliminate.

What the weights change is the orbitals' response, which the Lagrangian
carries. At unequal weights the rotations among the roots are parameters of
E_SA, coupled to the orbitals (∂²E_SA/∂Θ_sr∂κ is (w_r − w_s) times the r–s
transition orbital gradient), so the Z-vector over the SA Hessian at the
solution's weights returns a nonzero Θ_sr = −(c_s · H′(z_o) c_r) / (E_s − E_r)
for every unequal pair, the coupled pair's included, and
`df_transition_gradient` folds it into z_c as `df_gradient` does. A pair of
equal weights is redundant in E_SA and stays projected, its Θ zero: at
partly equal weights (0.4, 0.4, 0.2) only the unequal pairs rotate, whether
or not the coupled pair is one of them.

**At equal weights nothing changes**: the weights are exactly 1 / nroots as
before, no pair rotates, so `RelaxedHessian`, `zvector` and
`df_transition_gradient` run the same operations on the same numbers, the
code path bit for bit the one before unequal weights were taken. The
output is not bitwise reproducible from one call to the next even so
("Reproducibility" below): the old and new code on one solution differ by up
to 2.7e-14, within the run-to-run spread, with the same Z-vector products.

### The Z-vector's tolerance

`zvector` stops at ||H̃ z_o + g|| ≤ its tolerance, an absolute one, and the
pair divides h_IJ by E_J − E_I after the solve, so a solve to 1e-10 leaves
d_IJ an error up to 1e-10 / |E_J − E_I| in the solve's units. The pair's
solve is therefore held to `solve_tolerance` (1e-10) × |E_J − E_I|, the gap taken in
[1e-2, 1] hartree (1e-12 at the floor, still reachable in doubles), and the
combination's, whose right-hand side is already over its gaps, to 1e-10.
Both then carry d to the same precision: at the skewed SA-3's (0, 1), a gap
of 0.108 hartree, a pair solved to a flat 1e-10 differed from the
combination by 1.8e-10 of max |d|, the pair's own solve error.

## The CSF term

The orbitals move with the nuclei, and with them the determinants, so
⟨I|∂J⟩ has a part from the basis functions' own motion:

    d^CSF_IJ = Σ_pq γ^IJ_pq ⟨φ_p|∂φ_q⟩ → ½ Σ_mn (D − Dᵀ)_mn ⟨m|∂n⟩,

D = C_a γ^IJ C_aᵀ, γ^IJ_pq = ⟨I|E_pq|J⟩ (`ci::Rdm::trans_rdm1(I, J)`, the
transpose of PySCF's `trans_rdm1(I, J)`); only the antisymmetric part
survives once the orbitals' own response (in h_IJ) is separated off, as in
PySCF's `nac_csf`. With m on atom A, Σ_n (D − Dᵀ)_nm ⟨∂_A m|n⟩ moved onto the
first index,

    d^CSF_IJ[A, k] = ½ G[Dᵀ − D][A, k],   G[a][A, k] = Σ_{m on A} Σ_n a_mn ∂S_mn/∂A_k,

G `overlap_half_gradient` (libcint's −int1e_ipovlp, PySCF's
`mf_grad.get_ovlp`), so ½ G is PySCF's `_nac_csf` exactly, at its
tm1 = C_a (γᵀ − γ) C_aᵀ. The AO matrix is one `gemmwich`. The half
derivative takes only the bra function's centre, so d^CSF (and with it
d_IJ) is **not** translation-invariant; electron-translation factors
(`use_etfs`) drop it.

## A combination of couplings

QD-NEVPT2's coupling (#749) needs not one d_IJ but a combination over every
pair of reference roots, and a term from an orbital density of its own:

    d^T = Σ_MN T_MN d_MN + d[D^X],   d[D^X] = Σ_xy D^X_xy ⟨φ_x|∂φ_y⟩,

T an antisymmetric nroots × nroots matrix and D^X an MO density, both
d_MN and d[D^X] with the SA orbitals (and, for d_MN, the CI) relaxed. Its
overload of `df_nac` takes T and D^X in place of the bra and the ket, and
solves one Z-vector where the pairwise sum would solve n(n − 1)/2:

```cpp
Result<mcscf::NacCombination> mcscf::df_nac(
    /* res ... ncas as above */, std::span<const double> t,
    std::span<const double> d_x = {});
// d->response, d->csf, d->basis_motion, d->total: each n_atom x 3;
// d->rhs (the Z-vector's packed right-hand side), d->gap, d->iterations,
// d->residual
```

**The layouts.** T is row-major, T_MN at M nroots + N, and exactly
antisymmetric (T_MN == −T_NM, the diagonal zero), as ½(R_P R_Qᵀ − R_Q R_Pᵀ)
is in floating point. D^X is nmo × nmo row-major, D^X_xy at x nmo + y, in
the solution's orbitals, or empty. **Its index order is the transition
density's**: D^X takes γ^MN_xy = ⟨M|E_xy|N⟩'s place, so the bra's orbital
comes first, and d[D^X] at D^X = γ^MN would be d^CSF_MN plus γ^MN's own
orbital response. D^X must be zero within the core, the active and the
virtual blocks, where the orbital rotations are redundant and no Z-vector
fixes them (`InvalidConfig` otherwise); QD-NEVPT2's PT2 density lives on
(t, i), (a, t) and (a, i) only.

**By linearity**, every piece of the pair's derivation carries over:

- **The CI part.** Σ_MN T_MN h_MN / (E_N − E_M) is the Lagrangian of
  "The numerator" at the combined RDMs Σ_{M<N} 2 T_MN / (E_N − E_M)
  (γ^MN, Γ^MN), the symmetrized `trans_rdm12` (T_NM / (E_M − E_N) is
  T_MN / (E_N − E_M), and the symmetrized RDMs are symmetric in M and N):
  one `transition_orbital_gradient`, one `zvector` and one
  `df_transition_gradient` there, as for a pair.
- **The CSF part.** ½ G[Dᵀ − D] at D = C_a γ^T C_aᵀ,
  γ^T = Σ_MN T_MN γ^MN = Σ_{M<N} T_MN (γ^MN − γ^MNᵀ), as the pair's d^CSF.
- **D^X.** With C(R) = C₀ (C₀ᵀ S(R) C₀)^−½ exp(κ) (the connection the
  Lagrangian's overlap term is written in), ⟨φ_x(R₀)|φ_y(R)⟩ moves by
  ½ (X − Xᵀ) + κ, X_xy = ⟨φ_x|∂φ_y⟩ with the derivative on the ket's centre
  only. The first part is the **basis motion**, ½ G[C (D^Xᵀ − D^X) Cᵀ] over
  every orbital, the CSF term's formula at D^X in γ's place
  (`basis_motion`). The second is the **orbital response**: along the packed
  x_pq (p > q, κ_pq = x_pq, `../../response/README.md`), Σ D^X_xy κ_xy has the
  gradient D^X_pq − D^X_qp, added to the Z-vector's orbital right-hand side;
  the multipliers then carry it through the same `df_transition_gradient`.
  D^X has no CI part: f holds no CI vector.

So `response` is the CI part and D^X's orbital response together (one
Z-vector cannot part them), `csf` the CSF part and `basis_motion` D^X's
basis motion, each on its own, and `total` their sum. With T = ½(E_IJ − E_JI)
and no D^X, `total` is d_IJ, `csf` d^CSF_IJ, and `rhs` the pair's right-hand
side over E_J − E_I. There are no options: `use_etfs` drops `csf` and
`basis_motion`, which a caller can do from the parts, and `mult_ediff` has
no one gap to multiply by.

**The pieces outside the Z-vector.** `mcscf::nac_combination_terms`
takes the combination's arguments that they read (no `h`, no `l`) and
returns them without solving: the combined RDMs `gamma` and `gamma2`
(Σ_{M<N} 2 T_MN / (E_N − E_M) (γ^MN, Γ^MN), as `response::fock` takes
them), `csf`, `basis_motion` and `gap`, after every check and refusal of
the combination's. The combination is that call, then its Z-vector and
its `df_transition_gradient`. It exists so that a caller can fold the
combination's right-hand side into a Z-vector of its own (QD-NEVPT2's d_PQ
with h_PQ's, `../../nevpt2/nac/README.md` "The fold"): (γ, Γ) give the
transition orbital gradient and the Hellmann-Feynman term, and
D^X_pq − D^X_qp the rest of the right-hand side.

**Gaps.** A pair with T_MN ≠ 0 whose roots are closer than `kRootGapFloor`
(1e-8 hartree, `../root_rotation/`, the QD gradient's ζ rule) is refused as
`Numerical`, naming the pair, before anything is built; `gap` is the
smallest |E_N − E_M| over the pairs with T_MN ≠ 0 (infinity for T = 0). The
memory, the stream and the reproducibility are the pair's: one Z-vector, one
`df_transition_gradient`, and one `overlap_half_gradient` more with D^X.

## Reproducibility

Two calls on one solution are **bitwise the same up to the derivative
integrals, and not after them**.

- **Bitwise:** everything before them. The transition RDMs, the three Focks
  and the Z-vector's right-hand side, the CI-relaxed Hessian's products, z_o,
  z_c and Θ hash the same over six calls on each test case (RTX 3080), so the
  Z-vector takes the same products to the same residual, which the test
  checks to the bit.
- **Round-off only:** every derivative contraction after it. Each is a
  FelizJueves kernel that adds its threads' shares into one output with
  `atomicAdd`, so the order of the sum is the scheduler's: `gs_`, `gt_` and
  `gv_` (h_IJ's one-electron terms, `integrals/one_electron/gradient`),
  `geri3` and `gmetric` (its two-electron term,
  `integrals/eri/density_fitting/gradient`), and `ips_`, the overlap
  derivative matrix under `overlap_half_gradient`, so d^CSF too: that call's
  row sums and per-atom host sum are in a fixed order, but the matrix they
  read is not.

**The size**, over ten repeats of every ordered pair of both test cases on
the RTX 3080: h_IJ within 1.4e-14, d^CSF within 5.6e-17 and d_IJ within
4.6e-14 (the skewed SA-3's (0, 2) and (2, 0), after the division by its
gap). `mcscf::df_gradient` shares every one of those kernels but `ips_` and
varies the same way, up to 5.7e-14 between calls on the same solutions'
root 0, and so do `gradient::df_rhf` and the Hessians, whose one-electron
and DF Hessian kernels accumulate by the same atomics.

**Why it is not fixed here.** A fixed order needs each thread's share in a
slot of its own, summed afterwards in a fixed order, and that is a change
to the emitted kernels, not to their staging: `gv_` adds every record into
every nucleus's row, and an ip1 shell pair's records add into the same
matrix elements, so no relabelling of the atoms a caller stages gives each
thread its own slot. So a refactor upstream of the derivative integrals is
shown unchanged by the Z-vector (its products and residual, bitwise) and
the outputs within the spread above, not by the outputs' bits.

## Memory and stream

`df_transition_gradient`'s, which checks its own memory, plus the CI
objects, `response::Hessian` and `RelaxedHessian` at the solution (as
`df_gradient` holds them for a root of a state average), the averaged Fock,
the transition orbital gradient and the Z-vector, none counted. Queued on res.stream(); the call returns with
everything done.

## Tests

`test/gtest/mcscf_nac_gpu.cpp`, against `test/gtest/data/casscf_nac/`
(H₂CO SA-2 and the skewed H₂CO SA-3, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT):
this tree's `DensityFitting::Full` solve from the exact RHF, taken as it
converges, then for every ordered pair h_IJ, d^CSF and d_IJ against `h`,
`csf` and `total` to 1e-8 up to the sign s_I s_J of the two CI phases;
d_IJ = −d_JI, d^CSF_IJ = −d^CSF_JI, h_IJ = h_JI to 1e-10 and Σ_A h_IJ below
1e-9; on the first pair the other three option settings against `etf`,
h + ΔE csf and `h`; a repeated call on the first pair, its Z-vector's
products and residual bitwise the first call's and its results within
1e-10 ("Reproducibility"); and the refusals (I = J, a root past the
solution's).
Measured on the RTX 3080: h_IJ within 3.3e-10, d^CSF within 2.2e-10 and
d_IJ within 1.0e-9 (the skewed case's (0, 2)), the Z-vector in 11 to 20
products (its tolerance scaled by the gap, "The Z-vector's tolerance"); about 4 s a case. The same file, at the same solutions, holds
`transition_orbital_gradient`'s ([`../transition/`](../transition/README.md)
"Tests").

PySCF refuses unequal weights, so there the reference is a finite
difference of `mcscf::state_overlaps` ([`../overlap/`](../overlap/README.md)),
in `test/gtest/mcscf_overlap_gpu.cpp`: the SA-CASSCF re-converged at ±δ and
±2δ, the central difference of ⟨I(R₀)|J(R)⟩ against d_IJ to 1e-5 of
max |d|, with its Richardson error required below a tenth of that. At
`h2co_ccpvdz_sa2` [0.6, 0.4] (δ = 1e-3 bohr) and the skewed SA-3 at
[0.4, 0.4, 0.2] (partly equal, δ = 2e-4: at 1e-3 the truncation error of its
largest coupling is above the bar), measured on the RTX 3080: |D(δ) − d| at
most 6.5e-8 and 2.2e-7, each its Richardson estimate, and the extrapolated
difference within 3.9e-9 and 1.5e-8 of d. With the weights held at
1 / nroots the same check misses by 6.0e-3 and 2.2e-2.

**The combination** (`McscfNacCombination` in `mcscf_nac_gpu.cpp`, on both
cases): every ordered pair as T = ½(E_IJ − E_JI) against `df_nac(I, J)`'s
d_IJ and d^CSF, and its right-hand side times E_J − E_I against
`transition_orbital_gradient` at `trans_rdm12(I, J)`; a random
antisymmetric T on the skewed SA-3 against Σ_{M<N} 2 T_MN d_MN; each to
1e-10 of the largest element compared, once. And the refusals: two roots
1e-9 hartree apart that T couples (Numerical), a T not antisymmetric and a
D^X inside the active block (InvalidConfig). **D^X**, by finite differences
(`McscfOverlapDensity` in `mcscf_overlap_gpu.cpp`, a slow test): at T = 0
and a fixed random D^X on (t, i), (a, t) and (a, i) of `h2co_ccpvdz_sa2`,
every coordinate's ladder of Σ_xy D^X_xy ⟨φ_x(R₀)|φ_y(R)⟩ over the
re-converged solutions, columns phase-aligned, its steadiest Richardson
extrapolant against d[D^X] to 1e-7 of max |d[D^X]|; the basis motion and the
orbital response each over 10× that bar.
Measured on the RTX 3080: the pairs' total within 6.6e-12 of max |d| (the
skewed (2, 0)), d^CSF within 6.8e-16, the right-hand side within 4.2e-15,
each Z-vector in the pair's own number of products; the random T within
7.8e-12; about 2.4 s and 4.3 s a case. The ladder: every one of the 12
coordinates settled (a run of at least three ratios in (3.2, 4.8)), the
worst steadiest extrapolant 1.9e-8 from d[D^X] (1.3e-8 of
max |d[D^X]| = 1.50, the bar 1.5e-7); the basis motion's max 1.0e7 and the
response's 8.5e6 times the bar (min |O_yy| 0.9975, so no column swapped);
15 s.
