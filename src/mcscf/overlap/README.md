# Overlap

The overlaps between the CAS states of two CASSCF solutions, each over its
own basis: `mcscf::state_overlaps`,

    S_IJ = ⟨I(R)|J(R′)⟩

for every root I of the bra solution and J of the ket. The bases may differ
in geometry, in basis set or in both. It is what MECI root tracking and
surface hopping (time-derivative couplings ⟨I(t)|J(t + Δt)⟩) read, and what a
finite-difference check of `df_nac` ([`../nac/`](../nac/README.md)) is
built from.

## Module

`free_breakfast.mcscf.overlap`, target `free_breakfast::mcscf::overlap`,
re-exported by `free_breakfast.mcscf`.

```cpp
Result<std::vector<double>> mcscf::state_overlaps(
    DeviceResources &res, const BasisSet &bra_basis, const mcscf::Solution &bra,
    const BasisSet &ket_basis, const mcscf::Solution &ket, int32_t ncore,
    int32_t ncas, int32_t nelecas, int32_t multiplicity);
// row-major n_bra x n_ket: element I * n_ket + J is <I(bra)|J(ket)>
```

Both solutions have the same `ncore`, `ncas` and active electrons (`nelecas`
of spin 2S + 1 = `multiplicity`, the molecule's), so the same determinant
space; each one's orbitals (`Solution::c`, columns [0, ncore) core and
[ncore, ncore + ncas) active) are over its own basis, and its roots are
`Solution::roots`. Nothing else of a `Solution` is read, so a CASCI's
orbitals and roots put in one serve as well.

## Conventions

**Which state is the bra.** `bra` is I, at the first basis, and `ket` is J,
at the second. S(bra, ket) is the transpose of S(ket, bra).

**Phases are not fixed.** S_IJ changes sign with either root's CI phase as
the solutions carry it, and that is all. A sign flip of an orbital, or any
rotation among the core or among the active orbitals, changes the CI
coefficients so that the state itself is unchanged, and S with it. A caller
tracking roots aligns them from S itself (the test re-signs each ket root so
S_JJ > 0, a maximum-overlap rule for small displacements); the call does no
such thing, since which assignment is right is the caller's question.

**Against `df_nac`.** With the bra at R₀ and the ket at R,
∂S_IJ(R₀, R)/∂R at R = R₀ is ⟨I|∂J/∂R⟩, `df_nac`'s d_IJ with the bra and ket
the same roots and the CSF term in (`use_etfs = false`): the overlap moves
the AOs with the nuclei, which is the CSF term.

## How it is computed

1. **The AO overlap** S^AO_mn = ⟨m(R)|n(R′)⟩, spherical,
   `overlap_mixed_spherical` ([`../../integrals/one_electron/overlap_mixed/`](../../integrals/one_electron/overlap_mixed/README.md)).
2. **The MO overlap** over the ncore + ncas occupied orbitals,
   S_pq = C_braᵀ S^AO C_ket, two gemms, downloaded.
3. **The core, once.** A determinant of the bra is
   a†(core α) a†(active α) a†(core β) a†(active β)|vac⟩, the creators
   of each string ascending ([`../../ci/strings/`](../../ci/strings/README.md)),
   and the ket's the same, so ⟨Φ|Φ′⟩ is the product of an α and a β
   determinant, each of the (ncore + N_σ)-square block over the core and the
   strings' occupied active orbitals. By the Schur complement on the core
   block,

       det [[S_cc, S_ca′], [S_ac, S_aa′]] = det(S_cc) det(S̃[a, a′]),
       S̃ = S_aa − S_ac S_cc⁻¹ S_ca   (ncas square),

   so det(S_cc) and S̃ are formed once, by one host LU with partial pivoting
   of the core block, and each string pair's determinant is that of an
   N_σ-square submatrix of S̃.
4. **The string overlaps** A_aa′ = det S̃[occ(a), occ(a′)] over the α strings
   and B over the β (B is A when N_α = N_β), on the host.
5. **The contraction**, S_IJ = det(S_cc)² Σ c^I_ab A_aa′ B_bb′ c^J_a′b′, on the
   device: a CI vector in `Layout::full` (α-major) read column-major is C_Jᵀ,
   so W_J = B C_Jᵀ Aᵀ is one gemm over every ket root and one per root, and S
   one gemm of the bra roots against W.

**The core condition.** If |det S_cc| is below 1e-6 the core orbitals at the
two geometries are not carried into each other (a core orbital swapped with
an active one), S̃ would be ill-conditioned, and the call refuses
(`Unsupported`) rather than return a number built on it. Between nearby
geometries det S_cc is ±1 to second order in the displacement.

**Why the determinants are on the host.** They are m_σ² determinants of
N_σ-square matrices (m_σ = C(ncas, N_σ) strings): for CAS(4,3) 3² of 2 × 2,
for CAS(12,12) 924² ≈ 8.5 × 10⁵ of 6 × 6, about 10⁸ flops, under a second
on one core, and only the ncas² of S̃ go to the host and m_α² + m_β² come
back. Everything that scales with the basis (the AO and MO overlaps) and
with the determinant space times the roots (the contraction) is on the
device. A kernel would buy nothing measurable at these sizes and would add a
device library and a second backend to keep portable.

## Memory and stream

The mixed AO overlap is a buffer of its own (`overlap_mixed_spherical` takes
one); the half-transformed overlap, the MO overlap, A, B and the contraction's
intermediates are `res`'s workspace. Queued on `res.stream()`, synchronized
twice (the MO overlap's download, the result's); the call returns with
everything done.

## Tests

`test/gtest/mcscf_overlap_gpu.cpp`, on `test/gtest/data/casscf_nac/`'s
`h2co_ccpvdz_sa2` (H₂CO SA-2, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT), set up as
`mcscf_nac_gpu.cpp`'s: S of the solution against itself is the identity to
1e-12; then, per coordinate, the solution re-converged at ±δ and ±2δ
(δ = 1e-3 bohr) from the reference's orbitals Löwdin-carried, each root
re-signed so S_JJ > 0, and the central difference of S_IJ against `df_nac`'s
total for both ordered pairs, to 1e-5 of max |d|, with the Richardson
estimate (D(2δ) − D(δ)) / 3 of the difference's own error required below a
tenth of that; and S(R, R₀) = S(R₀, R)ᵀ to 1e-12. Three coordinates (H1's
out-of-plane x, where d is largest, and two in-plane ones where it vanishes
by symmetry) are in the fast tier; every coordinate is under
`FBF_SKIP_IF_FAST_TESTS()`. Measured on the RTX 3080: the identity to
3.6e-14, the transpose to 2.7e-15; where d is nonzero (the hydrogens'
out-of-plane x, 0.302 / bohr) |D(δ) − d| is 5.1e-8 to 6.8e-8, each within
1e-8 of its Richardson estimate, and the extrapolated (4 D(δ) − D(2δ)) / 3
is within 1e-9 of d: what is left is the step's own error. Every other
component is below 2.2e-10. 1e-5 of max |d| is 3.0e-6. About 4 s for the
fast case, 6.4 s for every coordinate. The same check at unequal weights,
where it is `df_nac`'s only reference, is described in
[`../nac/README.md`](../nac/README.md) "Tests".
