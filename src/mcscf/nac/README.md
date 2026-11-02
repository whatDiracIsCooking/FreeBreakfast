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
   parameters. Its orbital part is `response::fock`'s gradient at (γ, Γ)
   less its gradient at zero RDMs (`../gradient/README.md` "A transition
   density"), this tree's g = 2 (F − Fᵀ), equal to PySCF's
   `gen_g_hop_active` vector. **Its CI part y and its part t along the
   rotations among the roots are zero**, at any weights ("Unequal
   weights" below).
3. **The Z-vector**, `mcscf::zvector` (internal, linked PRIVATE) over the
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

The weights are not in a `Solution`. When the weighted γ is the roots'
plain average (to 1e-10) they are taken as exactly 1 / nroots, as they
always were; otherwise they are recovered from the RDMs by
`sa_gradient_weights`, as `df_gradient` recovers them
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
y = 0 and t = 0 (`../zvector/README.md` "The rotations among the roots").
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
output is not bitwise reproducible from one call to the next even so (the
derivative integrals accumulate in no fixed order): on the RTX 3080 two
calls on one solution differ by up to 2.6e-14, and the old and new code on
one solution by up to 2.7e-14, with the same Z-vector products.

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

## Memory and stream

`df_transition_gradient`'s, which checks its own memory, plus the CI
objects, `response::Hessian` and `RelaxedHessian` at the solution (as
`df_gradient` holds them for a root of a state average), three Focks and the
Z-vector, none counted. Queued on res.stream(); the call returns with
everything done.

## Tests

`test/gtest/mcscf_nac_gpu.cpp`, against `test/gtest/data/casscf_nac/`
(H₂CO SA-2 and the skewed H₂CO SA-3, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT):
this tree's `DensityFitting::Full` solve from the exact RHF, taken as it
converges, then for every ordered pair h_IJ, d^CSF and d_IJ against `h`,
`csf` and `total` to 1e-8 up to the sign s_I s_J of the two CI phases;
d_IJ = −d_JI, d^CSF_IJ = −d^CSF_JI, h_IJ = h_JI to 1e-10 and Σ_A h_IJ below
1e-9; on the first pair the other three option settings against `etf`,
h + ΔE csf and `h`; and the refusals (I = J, a root past the solution's).
Measured on the RTX 3080: h_IJ within 4e-10, d^CSF within 2.2e-10 and d_IJ
within 1.3e-9 (the skewed case's (0, 2)), the Z-vector in 11 to 19
products; about 4 s a case.

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
