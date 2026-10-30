# `free_breakfast.nevpt2.density_fitting`

SC-NEVPT2's two-electron integrals over density fitting: the MO blocks of the
fitted B that its classes read, the active h2e, and the external four-index
blocks built **slab by slab**, so that no whole external block need ever exist
on the device. Link `free_breakfast::nevpt2::density_fitting`.

    mo_blocks(res, naux, nao, nmo, b, c, ncore, ncas) -> MoBlocks
        aa, ca, va, cv = b_mo over C's core / active / virtual columns
        vc             = cv with its orbital pair swapped (one permute)
    h2e(res, blocks, out)                       h2e[t,u,v,w] = (tv|uw)
    slab(res, blocks, block, b0, b1, out)       block, batch axis in [b0, b1)
    batch_axis(block), block_shape(blocks, block)

    h2e_adjoint(res, blocks, h2e_bar, lambda)                 lambda +=
    slab_adjoint(res, blocks, block, b0, b1, slab_bar, lambda) lambda +=
    mo_blocks_adjoint(res, bar, core_diag_bar, virt_diag_bar, lambda)

(pq|rs) = Σ_Q B_Q,pq B_Q,rs, the metric already folded into B
([`density_fitting.mo`](../../integrals/eri/density_fitting/mo/README.md)).
**No (Q|vv) block**: no SC class has two virtuals in one charge distribution,
which is what keeps the density-fitted variant cheap. Everything is on
`res.stream()`, with no host synchronization.

## Layouts

Every array is column-major, first index fastest. Each B block is
`[Q][p][q]`, element (Q, p, q) at Q + naux (p + n_p q), `b_mo`'s layout and
the golden `B_*.npy` files' (`test/gtest/data/nevpt2/README.md`). A slab is
the block's physicist layout below, with its batch axis b1 − b0 in length.

## The slab table

The demo's table, carried over unchanged: C[x, y, z, w] = (xy|zw) =
Σ_Q X_Q,xy Y_Q,zw over x in the batch, then slab axis k = C axis axes[k].
Each row was derived from the PySCF `eris[...]` slice the demo's
`generate_golden.py` `_integral_blocks` takes for that block.

| block | physicist slab | X | Y | C | axes | batch axis |
|---|---|---|---|---|---|---|
| Sr | [r,a,b,c] = (rb\|ac) | B_va | B_aa | [r,b,a,c] | 0 2 1 3 | 0 |
| Si | [t,u,i,v] = (it\|uv) | B_ca | B_aa | [i,t,u,v] | 1 2 0 3 | 2 |
| Sijrs | [i,a,j,b] = (ia\|jb) | B_cv | B_cv | [i,a,j,b] | 0 1 2 3 | 0 |
| Sijr | [r,p,j,i] = (rj\|ip) | B_vc | B_ca | [r,j,i,p] | 0 3 1 2 | 0 |
| Srsi | [x,y,i,t] = (xi\|yt) | B_vc | B_va | [x,i,y,t] | 0 2 1 3 | 0 |
| SrsiT | [y,x,i,t] = (xt\|yi) | B_va | B_vc | [x,t,y,i] | 2 0 3 1 | 1 |
| Srs | [r,s,t,u] = (rt\|su) | B_va | B_va | [r,t,s,u] | 0 2 1 3 | 0 |
| Sij | [t,u,i,j] = (it\|ju) | B_ca | B_ca | [i,t,j,u] | 1 3 0 2 | 2 |
| Sir1 | [r,t,i,u] = (ri\|tu) | B_vc | B_aa | [r,i,t,u] | 0 2 1 3 | 0 |
| Sir2 | [r,u,t,i] = (rt\|iu) | B_va | B_ca | [r,t,i,u] | 0 3 1 2 | 0 |

SrsiT is Srsi's whole block (`h2e_v_Srsi`) batched along its second axis.
The batched index is always the **first** orbital of X's pair, which is why
B_vc is kept beside B_cv: several blocks batch over a virtual paired with a
core.

## How a slab is built

X's element (Q, x, y) is at Q + naux (x + n_x y), so the batch's x for one y
is the naux × n_b matrix at naux (b0 + n_x y), and consecutive y are
naux n_x apart. **One `gemmStridedBatched` over y**, C_y = X_yᵀ Y, writes
C_y at n_b y with ldc n_b n_y: C's element (x, y, z, w) at
x + n_b (y + n_y (z + n_z w)), the demo's C read column-major, for any batch.
The demo is row-major, where the batch is a contiguous run of X's rows and one
plain gemm does; here the fast index is Q, so the batch is strided over y.

Then one permute (`nevpt2::device::permute`, `nevpt2_permute.cu`, one thread
per output element) into the slab. Sijrs needs none: its gemm writes the slab
directly. C, when a permute follows, is scratch from `res.device_workspace()`,
the slab's own size. h2e is the same build over B_aa × B_aa, axes 0 2 1 3.

**No `IntegralSource` interface.** The demo's virtual base let an
integral-direct source hand out views into whole blocks; a future
integral-direct path here is a second driver, not a virtual.

## The adjoints

The reverse of the three builders, for the NEVPT2 nuclear gradient
(`nuclear_gradient.tex` §5): bars on the blocks, on h2e and on each slab,
pulled back onto Λ = ∂E/∂B in the MO basis. **Λ is naux × nmo², B's own
layout as `b_mo` reads it with C the identity**: (Q, p, q) at
Q + naux (p + nmo q), p and q absolute, nmo = ncore + ncas + nvirt. Every
adjoint **adds** onto Λ, so the caller zeroes it once and walks the slabs
one at a time: a slab's bar is consumed by its call and never held with the
others.

A slab is C = X_bᵀ Y over the batch and a permute, so with C̄ the slab bar
permuted back (the inverse axes; none for Sijrs),

    X̄_y += Y C̄_yᵀ      one gemmStridedBatched over y
    Ȳ_w += X_b C̄_w     one gemmStridedBatched over w

both with β = 1, **straight into Λ**: a block's element (Q, x, y) lies in Λ
at at + x s_x + y s_y with Q at stride 1, so each gemm's ldc and batch stride
reach it, and no bar on a block is ever formed. B_vc is B_cv's pair swapped,
so its s_x is naux nmo and its s_y naux. X_b is X's columns b0..b1 for every
y, packed by the permute kernel into scratch when the batch is not the whole
extent (its columns are then not uniformly strided). Where X and Y are the
same block (Sijrs, Srs, Sij, h2e) the two gemms add into the same corner one
after the other on the stream. The slab is quadratic in B, and the adjoint is
its derivative at `blocks`' B, both terms.

`mo_blocks_adjoint` takes bars on the blocks themselves (a `MoBlocks` of
bars; an empty buffer adds nothing), each one in-place geam at its corner,
B_vc's permuted back onto B_cv's. It also takes the bars on **B_cc's and
B_vv's diagonals**, B_Q,ii and B_Q,aa (naux × ncore and naux × nvirt), which
the orbital energies and h1e_v_sr read beyond the five blocks ((ii|jj),
(aa|tu), ...; `nuclear_gradient.tex` §5.1): one geam each onto Λ's diagonal,
leading dimension naux (nmo + 1). `mo_blocks` never forms them; the bar is
the whole of what is needed here.

Λ is the adjoint of the maps **as written**: they read B_cv's pair, and so
B_vc's, from Λ's (core, virtual) corner only, never its (virtual, core) one.
For a symmetric B the symmetric adjoint is (Λ + Λᵀ) / 2 per Q, the consumer's
step.

## Limits

The gemm runs in the `int32_t` BLAS dispatch: n_b n_y and n_z n_w past
`INT32_MAX` are `InvalidConfig`, as is a batch outside [0, extent]. An empty
slab writes nothing; `naux = 0` writes zeros. The adjoints add the same
limits, plus Λ's leading dimension: naux nmo (a B_vc operand's) and, for
`mo_blocks_adjoint`, the diagonal's naux (nmo + 1) past `INT32_MAX` are
`InvalidConfig`. An empty slab, or `naux = 0`, adds nothing.

## Not done

- `mo_blocks` calls `b_mo` four times, so the core half transform runs twice
  (ca and cv). Sharing one `b_half` per range would save it; the virtual one,
  the largest, already runs once.
- Λ stays in the MO basis: taking it to the AO derivative integrals (or the
  orbital source term, `nuclear_gradient.tex` Eq. Xao) is the consumer's.
- Only B_cc's and B_vv's diagonals have an adjoint. The core orbital
  energies' exchange term, Σ_j (ij|ji), reads B_cc off its diagonal too; a
  bar there is a `geam` onto Λ's core-core corner the same way, not yet
  needed by anything in the tree.

## Tests

`test/gtest/nevpt2_density_fitting_gpu.cpp`: from the golden B blocks of every
case in `test/gtest/data/nevpt2/`, h2e and every slab against the golden
four-index block at 1e-10 absolute (the demo's `--check-blocks`), whole and in
several batch sizes, the last one partial. And `mo_blocks` over H₂O/cc-pVDZ's
fitted B and DF-RHF orbitals (`data/density_fitting/`): h2e and the slabs
inside (pq|uv) and (pu|qv) against `casscf_integrals`' DF ppaa and papa over
the same B, and B_vc against B_cv. The adjoints by duality, over a random,
unsymmetric B and Ḃ (3 core, 4 active, 5 virtual, naux 6): ⟨Λ, Ḃ⟩ against
Σ ⟨bar, ẋ⟩ at 1e-12 relative, ẋ the forward builders' tangent along Ḃ
((f(B + Ḃ) − f(B − Ḃ)) / 2, exact for a quadratic map; `mo_blocks` over Ḃ
for the linear blocks). Every block whole and walked in batches of 1, 2 and
3, the last one partial; h2e; each block bar alone, then all of them with
the two diagonals; and every adjoint accumulated onto one Λ.
