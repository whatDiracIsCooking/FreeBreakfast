# RDM

dm3, f3ca and f3ac of one CI vector, on the device: the three n_act⁶ tensors
SC-NEVPT2's perturber classes read of the reference state; and the same
three between a bra and a different ket, with the transition dm1 and dm2,
which QD-NEVPT2 reads of each ordered pair of states. Ported from
nevpt2-gpu-demo's `src/rdm` (commit `7c5a674`), narrowed to its BLAS digest
and BLAS consume, over this tree's `ci` strings. Beside it, the build's
adjoint, which the NEVPT2 nuclear gradient takes the three tensors' bars
through, back onto the CI vector and (tu\|vw).

## Module

`free_breakfast.nevpt2.rdm`, also re-exported by `free_breakfast.nevpt2`;
target `free_breakfast::nevpt2::rdm`.

| Name | Role |
|---|---|
| `nevpt2::rdms(res, alpha, beta, vector, eri, dm3, f3ca, f3ac, max_bytes)` | The three tensors of the device CI `vector` over the two `ci::DeviceStringSet`s, from the device (tu\|vw) `eri`, into the device `dm3`, `f3ca`, `f3ac`; `Result<int64_t>`, the number of tiles it took |
| `nevpt2::rdms(res, alpha, beta, bra, ket, eri, dm1, dm2, dm3, f3ca, f3ac, max_bytes)` | The same between the device `bra` ⟨I\| and `ket` \|J⟩, with the transition `dm1` (n²) and `dm2` (n⁴), either of which may be null; `bra == ket` (one pointer) is the one-vector build, bit for bit |
| `nevpt2::rdms_workspace_bytes(n, width, transition = false)` | The bytes of `res.device_workspace()` a build over `n` active orbitals carves at tile width `width`; a `transition` (bra ≠ ket) build's |
| `nevpt2::rdms_adjoint(res, alpha, beta, vector, eri, dm3_bar, f3ca_bar, f3ac_bar, vector_bar, eri_bar, max_bytes)` | From device bars on the three tensors, `vector_bar` = ∂E/∂c and `eri_bar` = ∂E/∂(tu\|vw) of E = ⟨dm3̄, dm3⟩ + ⟨f3cā, f3ca⟩ + ⟨f3ac̄, f3ac⟩, both overwritten; `Result<int64_t>`, the number of tiles |
| `nevpt2::rdms_adjoint_workspace_bytes(n, width)` | The same, for the adjoint |

## Layouts

**One layout for all three outputs, PySCF's**: n⁶ row-major, (p, q, r, s, t,
u) at ((((p n + q) n + r) n + s) n + t) n + u, the C-order arrays the
energy's einsums are written against.

- dm3[p, q, r, s, t, u] = ⟨c\| E_pq E_rs E_tu \|c⟩: `fci.rdm.make_dm123`'s
  raw dm3, before `reorder_dm123`.
- f3ca and f3ac are `mrpt.nevpt2._contract4pdm`'s `'NEVPTkern_cedf_aedf'` and
  `'NEVPTkern_aedf_ecdf'` with `aaaa` the chemists'-order (tu\|vw): the
  4-RDM contracted with the integrals, which never exists here.

### Bra and ket

Between ⟨I\| = `bra` and \|J⟩ = `ket` all five outputs are non-symmetric, in
the same layout:

- dm1[p, q] = ⟨I\| E_pq \|J⟩, dm2[p, q, r, s] = ⟨I\| E_pq E_rs \|J⟩ and
  dm3[p, q, r, s, t, u] = ⟨I\| E_pq E_rs E_tu \|J⟩: `make_dm123(bra, ket)`'s
  dm2 and raw dm3, and **the transpose of its dm1**, which for bra ≠ ket is
  ⟨I\| E_qp \|J⟩ (PySCF's `trans_rdm1` convention; `ci::Rdm`'s README has the
  same note). Here dm1 reads as dm2 and dm3 do.
- f3ca[p, q, r, s, t, u] = Σ_ijk (ki\|uj) ⟨I\| E_pq E_rs E_tj E_ki \|J⟩ and
  f3ac[p, q, r, s, t, u] = Σ_ijk (ik\|jt) ⟨I\| E_pq E_rs E_ju E_ik \|J⟩: the
  forms `make_a16` comments `f3ca` and `f3ac` with (its `#:` dm4 lines), which
  are 'NEVPTkern_cedf_aedf' and 'NEVPTkern_aedf_ecdf' of one vector, with the
  transition 4-RDM in the place of the state's.

The bra is always the left state: I ← J and J ← I are two calls, with the
two vectors swapped. They are not each other's transposes beyond dm1 and dm2,
and QD-NEVPT2 needs both, over different denominators. These are not
`ci::Rdm`'s transition RDMs, which are symmetrized.

**The inputs are this tree's.** The CI vector is alpha-major, (K_α, K_β) at
K_α · beta.size() + K_β, which is PySCF's (na, nb) CI array: the strings are
already PySCF's `cistring` order and sign convention (`../../ci/strings/`).
(tu\|vw) is `ActiveSpaceHamiltonian::eri`'s, t + n (u + n (v + n w)); the
build converts it once, at entry, into the two row-major copies it reads
(chemists' `eri[a, b, c, d]`, and the consume's `[x, a, p, q] = (ax|qp)`).

## Design

Per tile of determinants K, one thread per determinant gathers

- R[tu, K] = ⟨K\| E_tu \|c⟩, and
- L2[pqrs, K] = ⟨c\| E_pq E_rs \|K⟩,

both row-major [·, K] at the tile's width. Then dm3 += L2 Rᵀ is one gemm; per
order, the consume W = (tu\|vw) · L2 is n strided batched gemms (one per
contracted index, batched over the free one, k = n²), and f3 += L2 Wᵀ one
gemm. After the last tile, fdm2 (a contraction of (tu\|vw) with the summed
dm3) and PySCF's wedge reconstruction run once per order, in place on f3.
Nothing is atomic and every sum has a fixed order, so the result does not
depend on the run.

**Between two vectors**, L2 is gathered over the bra and R over the ket, so
dm3 = L2 Rᵀ is ⟨I\| E E E \|J⟩ as it stands, and dm1 += R c_I and dm2 +=
L2 c_J are one gemv each per tile, over the tile's slice of the other
vector. Two places in the one-vector build read ⟨c\|…\|c⟩'s hermiticity,
and each is the transition's own read instead:

- **The consume reads the ket's L2.** f3 += L2 Wᵀ is ⟨I\| E E \|K⟩ ⟨K\| E E \|J⟩,
  and W's factor ⟨K\| E_fx E_qp \|J⟩ is the ket's ⟨J\| E_pq E_xf \|K⟩: of one
  vector, L2 itself. So a transition build gathers the ket's L2 too, by a
  second launch of the same kernel with bra = ket (which writes R again, the
  same numbers), into one more n⁴-per-determinant region.
- **The ca order's fdm2 reads dm3[i, j, k, C, B, A]**, where the one-vector
  build reads dm3[A, B, C, k, j, i], its equal by ⟨c\| E_pq E_rs E_tu \|c⟩ =
  ⟨c\| E_ut E_sr E_qp \|c⟩ and not otherwise. The ac order's read is the
  transition's already.

The wedge needs nothing: its copy and its two fdm2 corrections are the
commutator E_pq E_rs = E_rs E_pq + δ_qr E_ps − δ_ps E_rq, an operator
identity, which holds between any two states.

**One pointer, `bra == ket`, is the one-vector build, operation for
operation**: one L2, the consume reading it, fdm2's old read, and the plan
against `rdms_workspace_bytes(n, width)` (the ket's L2 is a region of no
bytes). The one-vector overload is that call. dm3, f3ca and f3ac are then
bit for bit what they were before the bra/ket overload existed (checked by
a dump of all three at one tile and many, before and after, on the RTX
3080). Two pointers that hold the same vector take the transition's path,
and agree with it to round-off.

**The link tables are the strings' own.** A replacement a†_p a_q \|K⟩ =
s \|J⟩ in K's list is ⟨J\| E_pq \|K⟩ = s, so L2's two walks are K's list
and then the intermediate string's, as the demo's forward Knowles-Handy table
walks. The same link is also ⟨K\| E_qp \|J⟩ = s, which is what the demo's
*reverse* table held for R: so R reads K's own list too, and no reverse table,
and no host table builder, exists here. Both read the one 8-byte
`Replacement` list of each `ci::DeviceStringSet`, the copy `ci::Sigma` and
`ci::Rdm` read.

**No GEMM temporary, no scatter.** The demo's BLAS digest wrote f3ca's GEMM
into an n⁶ temporary and scattered it with the last two axes of each row
transposed. Here the ca consume writes W (f, a)-major instead of (a, f)-major,
a different leading dimension and batch stride on the same gemm, so the
digest lands in f3ca's layout directly and accumulates with β = 1, as dm3's
and f3ac's do.

**The tile width is planned from bytes, not a count.** The scratch is carved
from `res.device_workspace()` in six regions: the two (tu\|vw) copies and fdm2
(n⁴ doubles each) and the tile's R, L2 and W ((2 n² + n⁴) doubles per
determinant), each rounded up to `kWorkspaceAlignment`, which is exactly
`rdms_workspace_bytes(n, width)`; a transition build a seventh, the ket's
L2 (n⁴ more per determinant), `rdms_workspace_bytes(n, width, true)`. The width is the widest whose bytes fit the
budget, at most every determinant; the tiles are that wide but for the last,
which takes the remainder. The budget is `max_bytes`, capped at what the
device can give: the workspace already held, or the free memory less 512 MiB
for the BLAS libraries' own workspaces and the pool's rounding. There is no
tile-count knob. The demo once saw an auto-sized unequal split at CAS(10,10)
corrupt memory instead of failing; here every buffer's extent is the one the
width was planned from, and `TinyBudgetsMatchOneTile` runs unequal and
one-determinant tiles under compute-sanitizer.

**A budget below one determinant is `Unsupported`, refused before anything
is queued or allocated**: a failed workspace growth aborts in this tree
(`CreatePolicy`), so the check has to come first.

Every launch and BLAS call is on `res.stream()`; it does not synchronize.

## The adjoint

`rdms_adjoint` is of the one-vector build only; it runs the build backwards, in this tree's own transposes rather
than the demo's sigma-like passes (`docs/gradients.md` §5.4, §6 there), over
the same tiles and the same kind of budget. E = ⟨dm3̄, dm3⟩ + ⟨f3cā, f3ca⟩ +
⟨f3ac̄, f3ac⟩, with each bar in the outputs' layout.

- **Once, first: the wedge and fdm2.** They ran last, so they are transposed
  first. Per order, the wedge's transpose gives the bar on f3 *before* the
  wedge (zero where the first index is below the third, which the wedge
  overwrote; the sum of the two places an element reached where it is above)
  and the bar on that order's fdm2. The two fdm2 bars, through (tu\|vw), add
  to dm3̄: the bar dm3 carries into the tiles.
- **Per tile of determinants K**, R and L2 are rebuilt exactly as `rdms`
  builds them. Each digest acc += L2 Bᵀ transposes to two gemms, L2̄ += acc̄ B
  and B̄ = acc̄ᵀ L2 (B = R for dm3, W for each f3). Each consume W = (tu\|vw)
  · L2 transposes to two loops of strided batched gemms: onto L2̄, batched
  over the free index as the forward is, and onto the consume's own bar,
  batched over the contracted one, since that product sums over the free
  index and a batch cannot. The produce's transpose then gathers R̄ and L2̄
  onto the CI bar: one thread per determinant J, over J's own replacement
  lists (a link J → M by (p, q) is ⟨M\| E_pq \|J⟩ and ⟨J\| E_qp \|M⟩ both),
  adding only the K in the tile. So every tile walks every J; nothing is
  atomic, and the result does not depend on the run.
- **dm3 is summed over the tiles** as `rdms` sums it, for the fdm2s' own
  (tu\|vw) dependence, which after the last tile joins the consume's bar in
  one kernel that writes `eri_bar` in `eri`'s layout.

**`eri_bar` is by each stored element, with no symmetry imposed**: the build
reads (tu\|vw) as it is given, so a caller whose integrals carry the
eightfold symmetry folds the bar itself. The 4-RDM is never formed.

**The plan is `rdms`' own**, the checks, the budget and the walk down to the
widest width, against `rdms_adjoint_workspace_bytes`: fifteen regions, each
rounded up to `kWorkspaceAlignment`: the two (tu\|vw) copies, the two fdm2
bars and the consume's bar (n⁴ doubles each); the two pre-wedge f3 bars,
dm3's total bar and dm3 (n⁶ each); and the tile's R, L2, W and their bars
((4 n² + 2 n⁴) doubles per determinant). A budget below one determinant is
`Unsupported` before anything is queued or allocated, as for `rdms`.

## Tests

`test/gtest/nevpt2_rdm_gpu.cpp` (`NevptRdm.*`), against
`test/gtest/data/nevpt2/`'s PySCF dm3, f3ca and f3ac. Those are committed only
for the CAS(4,4) cases (RHF and CASSCF orbitals) and CAS(8,8); CAS(10,10)'s
would be 8 MB each, and the energy's chain test covers them. On those three:

- `MatchesPyscf`: one tile at the default budget, each tensor within 1e-12 of
  max\|PySCF's\|;
- `TinyBudgetsMatchOneTile`: budgets of `rdms_workspace_bytes(n, w + 1) − 1`,
  which must give width w exactly (the tile count says so), for w = 1, 7, 10
  (36 determinants: every tile one determinant; 6 tiles, the last one; 4
  tiles, the last 6) and w = 23, 64 (4900: 213 tiles, the last one; 77, the
  last 36); each within 1e-13 of the one-tile build and 1e-12 of PySCF's;
- `BudgetBelowOneDeterminantIsUnsupported`: 0 and
  `rdms_workspace_bytes(n, 1) − 1` are `Unsupported`, the outputs keep what
  they held and the workspace is not grown;
- `Refusals`: sets over different orbital counts, or none.

The bra/ket overload, against `test/gtest/data/nevpt2_transition/`'s PySCF
dm1, dm2, dm3, f3ca and f3ac (`tools/emit/nevpt2_transition_golden.py`):
N₂'s CAS(4,4) (2, 2), the golden's CI vector as the bra and a random ket,
and a CAS(6) block of its CAS(8,8) at (3, 2) electrons, so the alpha and
beta sets differ, between two random vectors:

- `TransitionMatchesPyscf`: one tile, each tensor within 1e-12 of max\|PySCF's\|;
- `TransitionTinyBudgetsMatchOneTile`: budgets of
  `rdms_workspace_bytes(n, w + 1, true) − 1` for w = 1, 7, 10 (36
  determinants) and 7, 23 (300: 43 tiles, the last 6; 14, the last one),
  each within 1e-13 of the one-tile build and 1e-12 of PySCF's; and
  `rdms_workspace_bytes(n, 1, true) − 1` is `Unsupported` with the workspace
  not grown, where the one-vector build at the same budget runs;
- `BraEqualKetIsTheOneVectorBuild`: on the three one-vector cases, at one
  tile and at width 7 or 23, the bra/ket call with one pointer gives the
  one-vector build's dm3, f3ca and f3ac bit for bit, and the state's dm1 and
  dm2 within 1e-12 of the golden's; with two pointers to the same vector,
  within 1e-13 of the one-vector build.

And the adjoint, against `rdms` itself, since `rdms` is quadratic in c and
linear in (tu\|vw):

- `AdjointMatchesPolarization`: at CAS(4,4) and CAS(6,6), from a random vector,
  a random (tu\|vw) with no symmetry and random bars, ⟨y, u⟩ along a random u
  is the bars on (rdms(c + u) − rdms(c − u)) / 2, and ⟨h2ē, δ⟩ along a random
  δ the bars on (rdms(eri + δ) − rdms(eri − δ)) / 2, each within 1e-12 of
  Σ\|bar\| \|difference\|; at one tile and at widths 7 and 23 (6 and 18 tiles,
  the last unequal), which agree with the one tile within 1e-13;
- `AdjointRefusals`: the budget below one determinant, with nothing written
  or grown, and the sets `rdms` refuses.

`Cas1212TilesAtTheDefaultBudget` (slow, skipped under
`FREE_BREAKFAST_FAST_TESTS`) builds CAS(12,12), 853 776 determinants, from a
random unit vector at the default budget, prints the tile count and the
pool's peak, and checks Σ_pqr dm3[p, p, q, q, r, r] = N³.
On the 16 GiB RX 9060 XT (gfx1200, HIP) with the card otherwise idle it
took 9 tiles, a 16.1 GB workspace and a pool peak of 16.2 GB (79 MB of it the
inputs and outputs), in 42 s. It is registered RUN_SERIAL, since its default
budget is nearly the whole card.

## Files

| File | Role |
|------|------|
| `rdm.cppm` | Module interface: `nevpt2::rdms`, `nevpt2::rdms_adjoint` and their `_workspace_bytes` |
| `rdm.cpp` | The checks, the tile plan, the consume and digest gemms and their transposes, and the launches |
| `nevpt2_rdm_bridge.h` | The launchers, shared by the module (GMF) and the `.cu` |
| `nevpt2_rdm.cu` | The eri's conversion, R and L2 per tile, fdm2 and the wedge, and the transposes of each |
| `CMakeLists.txt` | The device library `free_breakfast.nevpt2.rdm.device` and the module `free_breakfast.nevpt2.rdm` |
