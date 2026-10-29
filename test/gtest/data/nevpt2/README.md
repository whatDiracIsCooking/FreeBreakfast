# nevpt2 golden data

The SC-NEVPT2 reference states and answers from
[nevpt2-gpu-demo](https://github.com/whatDiracIsCooking/nevpt2-gpu-demo)'s
density-fitted goldens, plus PySCF's dm3, f3ca and f3ac, written by
`tools/emit/nevpt2_golden.py`. No file here is hand-edited.

## Source

Demo commit `7c5a67408a3ac6e143e4f1b6026f0344e3c3ff5e`, files
`golden/<case>.nevpt2gold`, which are ordinary committed files there (not
git-lfs, not generated at checkout). The script checks each one's sha256
against its `CASES` table and refuses a different file. The demo built them
with its `reference_data/generate_golden.py --df` (PySCF 2.14.0,
single-threaded). Each one is N2 at 1.1 Å in the auto-detected D2h. The state is a CASCI,
canonicalised as `pyscf.mrpt.NEVPT.kernel` does (`cas_natorb`), and
density-fitted throughout: RHF, CASCI, `h1eff` and every integral block. The
answer is PySCF's DF-NEVPT2 (`pyscf.mrpt.dfnevpt2`), not the conventional
one.

| case | basis / aux basis | orbitals | ncore | CAS (e, o) | nvirt | naux | dm3/f3 |
|---|---|---|---:|---|---:|---:|---|
| `n2_ccpvdz_cas44_df` | cc-pVDZ / cc-pVDZ-JKFIT | RHF | 5 | (4, 4) | 19 | 140 | yes |
| `n2_ccpvdz_cas44_casscf_df` | cc-pVDZ / cc-pVDZ-JKFIT | DF-CASSCF | 5 | (4, 4) | 19 | 140 | yes |
| `n2_ccpvdz_cas88_df` | cc-pVDZ / cc-pVDZ-JKFIT | RHF | 3 | (8, 8) | 17 | 140 | yes |
| `n2_ccpvdz_cas1010_df` | cc-pVDZ / cc-pVDZ-JKFIT | RHF | 2 | (10, 10) | 16 | 140 | no |
| `n2_ccpvtz_cas1010_df` | cc-pVTZ / cc-pVTZ-JKFIT | RHF | 2 | (10, 10) | 48 | 158 | no |

Every case has n_alpha = n_beta = e/2. The aux basis is PySCF's library
`<basis>-jkfit`, the generator's default, not this tree's
`data/aux_basis` copies. The CASSCF-orbital case's orbitals are the demo's
committed `golden/n2_ccpvdz_cas44_casscf_df.mogold`, optimised once with DF
and then used as the input to the same deterministic CASCI. The demo's CAS(12,12)
golden is not converted: its CI vector alone is 6.8 MB, and it is slow on
the CPU.

**dm3, f3ca and f3ac are committed only up to CAS(8,8).** Each is n_act^6
doubles, which is 8 MB at CAS(10,10), over the 3 MB per-file
`check-added-large-files` limit, and would bring the directory from 14 MB to 62 MB.
The device RDM build (#416) checks them directly against these files up to
CAS(8,8). CAS(10,10) is covered through the energy's chain test (#418),
where the device RDMs feed the class energies, which are checked against the golden's.
`--all-rdms` writes them for every case, to an `--out` outside the tree.

## Files

One `<case>/<array>.npy` per array, `'<f8'`, Fortran order (the layout
`src/io/npy` reads; `np.load` returns the same logically indexed tensor as
the golden's C-order array). Index names: `c` core, `a`/`p q r s t u`
active, `v` virtual, `L` auxiliary.

| array | shape | what |
|---|---|---|
| `ci` | (na, nb) | CI vector, PySCF `cistring` order |
| `dm1`, `dm2` | a², a⁴ | `make_dm123` convention, the golden's (equal bit for bit to PySCF's recomputation) |
| `dm3` | a⁶ | `fci.rdm.make_dm123('FCI3pdm_kern_sf', ci, ci, ...)`, raw, **not** reordered (`reorder_dm123` not applied) |
| `f3ca`, `f3ac` | a⁶ | `mrpt.nevpt2._contract4pdm('NEVPTkern_cedf_aedf' / 'NEVPTkern_aedf_ecdf', aaaa, ...)` with `aaaa = h2e.transpose(0,2,1,3)` |
| `h1e` | (a, a) | `h1eff` active block |
| `h2e` | a⁴ | physicist: `h2e[p,q,r,s] = (pr\|qs)` (PySCF `ppaa[a,a].transpose(0,2,1,3)`) |
| `e_core`, `e_virt` | (c,), (v,) | canonical orbital energies |
| `h1e_v_Sr` | (v, a) | `h1eff[v,a] - einsum('mbbn->mn', h2e_v_Sr)` |
| `h1e_v_Si`, `h1e_v_Sir` | (a, c), (v, c) | `h1eff` blocks |
| `h2e_v_Sr`, `h2e_v_Si`, `cvcv`, `h2e_v_Sijr`, `h2e_v_Srsi`, `h2e_v_Srs`, `h2e_v_Sij`, `h2e_v1_Sir`, `h2e_v2_Sir` | 4-index | the external blocks, each exactly the slice and transpose PySCF's `Sxxx` takes from its `eris` (the demo's `_integral_blocks`) |
| `B_aa`, `B_ca`, `B_va`, `B_cv` | (L, ·, ·) | MO three-index blocks, `(pq\|rs) = Σ_L B[L,p,q] B[L,r,s]` (metric already folded in) |
| `class_norms`, `class_energies` | (8,) | PySCF's per-class answer, order Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir |
| `e_nevpt2_total` | () | the SC-NEVPT2 correlation energy, the sum of `class_energies` |
| `pc_class_energies` | (8,) | block2's PC-NEVPT2 per-class answer (`icnevpt2_full.WickICNEVPT2`), the same order |
| `e_pc_total` | (1,) | block2's PC-NEVPT2 correlation energy |

**The two PC arrays are block2's, copied unchecked**: PySCF has no PC-NEVPT2,
so nothing here recomputes them. The demo's generator wrote them only after
checking that block2's SC equals the golden's PySCF SC per class to 1e-7
(measured ≤ 9.8e-10), so block2 saw this state, these integrals and these
RDMs (the demo's `docs/reference-data.md`, "PC fields").

## The check before writing

For every case, the script recomputes dm1/dm2 and requires them to equal the
golden's bit for bit. It then rebuilds the Sr class (through `make_a16`, which reads
dm3, f3ca and f3ac) and the Si class (through `make_a22`) from these tensors and the
golden's integrals, and requires each norm and energy to be within 1e-12 of
the golden's (measured: ≤ 7.8e-16). On a miss it writes nothing.
`test/pytest/test_nevpt2_golden.py` repeats the round trip on
`n2_ccpvdz_cas44_df`.

## Regenerating

```sh
git clone https://github.com/whatDiracIsCooking/nevpt2-gpu-demo /tmp/demo
git -C /tmp/demo checkout 7c5a67408a3ac6e143e4f1b6026f0344e3c3ff5e
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/nevpt2_golden.py /tmp/demo
```

BLAS and OpenMP are pinned to one thread, so a re-run is byte-identical. The run takes
about half a minute.
