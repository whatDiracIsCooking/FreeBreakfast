# NAC

The nonadiabatic (derivative) coupling between two roots of a converged
DF-SA-CASSCF at equal weights, on the GPU: `mcscf::df_nac`,

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
   `gen_g_hop_active` vector. **Its CI part is zero at equal weights**: the
   CI row of root r is the projection, off the root space, of
   ∂⟨I|H|J⟩/∂c_r, which is H c_J = E_J c_J for r = I (and E_I c_I for
   r = J); that lies in the space P = 1 − Σ_s c_s c_sᵀ projects out. At
   unequal weights the rotations among the roots are parameters too and
   the pair's own rotation is not redundant, which is why unequal weights
   are refused (`Unsupported`) here.
3. **The Z-vector**, `mcscf::zvector` (internal, linked PRIVATE) over the
   CI-relaxed Hessian at the solution's orbitals and roots, built as
   `df_gradient` builds it, with no CI right-hand side: (z_o, z_c).
4. **The terms**, one `df_transition_gradient` at (γ, Γ) with those
   multipliers: the Hellmann–Feynman terms with no core–core part, z_c's
   transition RDMs added, and the orbital response along κ(z_o) at the SA
   RDMs, in one Γ_mnP and one W_PQ. Its total is h_IJ.

h_IJ is translation-invariant (Σ_A h_IJ = 0), as any derivative of an
energy-like functional is.

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
h + ΔE csf and `h`; and the refusals (I = J, a root past the solution's,
unequal weights). Measured on the RTX 3080: h_IJ within 4e-10, d^CSF
within 2.2e-10 and d_IJ within 1.3e-9 (the skewed case's (0, 2)), the
Z-vector in 11 to 19 products; about 4 s a case.
