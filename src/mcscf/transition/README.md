# Transition

The orbital gradient of a transition element between two roots of a
DF-CASSCF, the CI vectors held: `mcscf::transition_orbital_gradient`,

    g^(sr) = ∂⟨c_s|H|c_r⟩/∂κ = g_fock(γ^(sr), Γ^(sr)) − g_fock(0, 0),

g_fock `response::fock`'s orbital gradient. It is `df_nac`'s Z-vector
right-hand side ([`../nac/`](../nac/README.md) "The numerator"), hoisted out
of it so that a multiplier on the condition ⟨c_s|H|c_r⟩ = 0 can be built on
the same per-pair gradient.

## Module

`free_breakfast.mcscf.transition`, target `free_breakfast::mcscf::transition`,
re-exported by `free_breakfast.mcscf`. It imports `response.fock` and no
other mcscf module, so `nac` and `gradient` can each take it.

```cpp
Result<Buffers::device<double>> mcscf::transition_orbital_gradient(
    DeviceResources &res, const Molecule &molecule, const SphericalMatrixAO &h,
    const Buffers::device<double> &b, int32_t naux, const SphericalMatrixAO &c,
    int32_t ncore, int32_t ncas, const Buffers::device<double> &ppaa,
    const Buffers::device<double> &papa, std::span<const double> gamma,
    std::span<const double> gamma2);
// the non-redundant pairs p > q in response::pack_orbital_pairs' order
```

Every argument is the density-fitted `response::fock`'s: `ppaa` and `papa`
are `casscf_integrals`' over `c`, and `gamma`, `gamma2` the pair's
`ci::Rdm::trans_rdm12(c_s, c_r)` read back to the host (ncas² and ncas⁴,
row-major). Those RDMs are symmetric in s and r, so g^(sr) = g^(rs). Per pair
(s, r) of a solution's roots, a caller forms the RDMs, then calls this with
one `casscf_integrals` over the solution's orbitals shared by every pair.

## What it is

At fixed orbitals the CASSCF energy is affine in the active RDMs, and
⟨c_s|H|c_r⟩ is that functional at the transition RDMs less the parts that
multiply ⟨c_s|c_r⟩ = 0, E_nuc and the core's own energy E_cc:

    ⟨c_s|H|c_r⟩ = E(γ^(sr), Γ^(sr)) − E(0, 0) = E_act(γ^(sr), Γ^(sr)).

(`../gradient/README.md` "A transition density" has E_act term by term.) The
orbital gradient g = 2 (F − Fᵀ) is linear in F, and F is affine in the RDMs,
so g^(sr) is `response::fock`'s gradient at (γ^(sr), Γ^(sr)) less its
gradient at zero RDMs. At zero RDMs F is 2 F^I in its core columns and zero
elsewhere, so the subtraction changes only the pairs (p, i), i core and p
not, each by −4 F^I_pi. It equals PySCF's `gen_g_hop_active` orbital vector
element for element (`gen_g_hop` returns 2 pack(F − Fᵀ), this tree's g), as
`test/gtest/mcscf_transition_gradient_gpu.cpp` measures.

Its CI part and its part along the rotations among the roots are not
returned: for a coupling they are zero ([`../nac/`](../nac/README.md)
"Unequal weights"), and a caller that needs ⟨c_s|H|c_r⟩'s CI derivative has
the roots' sigma for it.

## Reproducibility

**Bitwise from call to call.** Two `response::fock`s and one axpy, none of
which accumulates with atomics: two calls on one solution agree to the bit,
and so does the inline difference `df_nac` took before the hoist (the same
two Focks, the same IEEE subtraction). Measured on the RTX 3080 over every
pair of both test cases: a spread of 0 between calls. So `df_nac` runs the
same operations on the same numbers it did, its Z-vector the same products
to the same residual.

## Memory and stream

Two `response::fock`s' matrices, the zero-RDM one freed on return, the
result the first's gradient. Queued on res.stream() with res.blas(), and
waited for (the axpy reads the second Fock's gradient, freed on return).

## Tests

`test/gtest/mcscf_nac_gpu.cpp`, `McscfTransitionOrbitalGradient`, at
`df_nac`'s converged solutions (H₂CO SA-2 and the skewed H₂CO SA-3, CAS(4,3),
cc-pVDZ / cc-pVDZ-JKFIT), every pair r < s:

- two calls bitwise the same, and bitwise the inline difference of two
  `response::fock`s that `df_nac` formed before;
- a directional ladder: ⟨c_s|H|c_r⟩ at C exp(t κ(x)), the RDMs held, as
  Σ h_eff γ + ½ Σ (tu|vw) Γ over `casscf_integrals` at the rotated orbitals
  (no `response::fock`), central differences at t = 10⁻²/2^k over ten rungs,
  at least three successive ratios in (3.2, 4.8), and the Richardson
  extrapolant against ⟨x, g^(sr)⟩ to 1e-7 relative, along two random unit x.
  The bar and the ladder are `test/gtest/nevpt2_gradient_gpu.cpp`'s.

Measured on the RTX 3080: 279 pairs a gradient, max |g^(sr)| 0.045 to
0.11; no element differs between two calls or from the inline difference;
the ladders' ratios 4.000 for at least five halvings and their extrapolants
within 1.8e-9 relative of ⟨x, g^(sr)⟩; 2.2 s and 3.4 s a case, the solve
included.
