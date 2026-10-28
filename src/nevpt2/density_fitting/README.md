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

## Limits

The gemm runs in the `int32_t` BLAS dispatch: n_b n_y and n_z n_w past
`INT32_MAX` are `InvalidConfig`, as is a batch outside [0, extent]. An empty
slab writes nothing; `naux = 0` writes zeros.

## Not done

- `mo_blocks` calls `b_mo` four times, so the core half transform runs twice
  (ca and cv). Sharing one `b_half` per range would save it; the virtual one,
  the largest, already runs once.

## Tests

`test/gtest/nevpt2_density_fitting_gpu.cpp`: from the golden B blocks of every
case in `test/gtest/data/nevpt2/`, h2e and every slab against the golden
four-index block at 1e-10 absolute (the demo's `--check-blocks`), whole and in
several batch sizes, the last one partial. And `mo_blocks` over H₂O/cc-pVDZ's
fitted B and DF-RHF orbitals (`data/density_fitting/`): h2e and the slabs
inside (pq|uv) and (pu|qv) against `casscf_integrals`' DF ppaa and papa over
the same B, and B_vc against B_cv.
