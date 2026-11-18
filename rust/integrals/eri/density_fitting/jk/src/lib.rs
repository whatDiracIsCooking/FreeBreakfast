//! Closed-shell J and K over the fitted factor B on the GPU, as tasks of a
//! [`Dag`](fb_core_planner::Dag), every J and K of a set from one pass over
//! a packed B that lives in a heap row:
//!
//! ```text
//! J_p   += B_bᵀ (B_b d_p)         (two gemvs per aux batch b, d_p the packed D)
//! X_q    = B_q C                  (one strided-batched gemm per batch)   D = 2 C Cᵀ
//! K     += 2 X Xᵀ                 (one syrk per batch, lower)
//! upper  = lowerᵀ                 (fb-recipes-symmetrize's mirror)
//! ```
//!
//! so that `J_mn = Σ_ls (mn|ls) D_ls` and `K_mn = Σ_ls (ml|ns) D_ls` with
//! `(mn|ls) ≈ Σ_Q B_Q,mn B_Q,ls`, B from
//! `fb_integrals_eri_density_fitting_three_center::add_packed_b`. The C++
//! `src/integrals/eri/density_fitting/jk` is the reference design; its
//! README has why K is built from a factor, never from D.
//!
//! [`add_streamed_jk`] adds **the streamed pass**: every J and K of a set
//! of contractions ([`Contraction`]) from one read of B, streamed through
//! a pinned window in aux batches of an adaptive width (at most
//! [`BATCH_MAX`]), as a counted loop of ring depth 2, so no device row
//! holds B whole. B's packed layout is
//! `fb-integrals-eri-density-fitting-pair-map`'s, and its pack and gather
//! `fb_integrals_sys::PackedTables`'. K's `(m, n)` and `(n, m)` carry the
//! same bits.
//!
//! | C++ | Here |
//! |---|---|
//! | `df_coulomb(res, nao, naux, b, density, j)` and `df_exchange(res, nao, naux, nocc, b, c, k)`, each a pass over the square, resident B | [`add_streamed_jk`], one pass over the packed B on the host for every J and K of the set, each an output row of its own; no resident B (#1721) |
//! | gamma and X carved from `res.device_workspace()` by `device_regions` | the pass's scratch rows, its ring's region, read off the plan's table |
//! | `df_coulomb_footprint(nao, naux)`, `df_exchange_footprint(nao, naux, nocc)` | the tasks' layouts; the batch an adaptive width |
//! | `wwr::gemv<double, int32_t>`, `wwr::gemmStridedBatched<…>`, `wwr::syrk<…>` with `kOne`, `kZero`, `kTwo` | `fb_core_wwr::blas`'s `gemv`, `gemm_strided_batched` and `syrk`, the same modes per batch, on the stream worker's handle |
//! | `symmetrize(res, WWRBLAS_FILL_MODE_LOWER, nao, k, nao)` | `fb_recipes_symmetrize::symmetrize_from`, the same fill mode, in `df jk finish` |
//! | `nao == 0`: an empty success; `naux` (or `nocc`) `== 0`: J (or K) zeroed; a negative count: `InvalidConfig` | contracts: every count is at least 1 |
//! | `nao²` (J) or `naux · nocc` (K) past `int32_t`: `InvalidConfig` | always-on contracts and a check, since the BLAS's counts are `i32`s |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | nothing: the C++ tree compiles its kernels into the binary | `fb_integrals_sys::check_library` and `fb_recipes_sys::check_library` first, at planning |
//! | `df_exchange(res, nao, naux, rank, b, left, right, k)`, the two-factor K | not ported: an SCF's density is `2 C Cᵀ`; a response density's port brings it |
//! | the block forms, `df_coulomb(…, count, …)` and `df_exchange(…, rank, count, …)`, and `df_exchange_symmetrized` | not ported: their callers (CPHF, the CASSCF response) are not yet |

#![forbid(unsafe_code)]

mod streamed;

pub use streamed::{BATCH_MAX, Contraction, add_streamed_jk};
