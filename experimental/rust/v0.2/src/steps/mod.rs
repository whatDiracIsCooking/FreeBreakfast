//! The steps of a DF-RHF, each in its crudest form: the full slab, no
//! batching, one card. They and the bus are one crate, so the bus names
//! every `Output` and a step's writes are private to its module.
//!
//! | Step | Reads | Produces |
//! |---|---|---|
//! | [`harmonics`] | | U, Cartesian to spherical, of the orbital basis |
//! | [`records`] | | the primitive pair records of each class |
//! | [`overlap`] | records, harmonics | S |
//! | [`core_hamiltonian`] | records, harmonics | H = T + V |
//! | [`orthogonalizer`] | overlap | X = S^(-1/2) |
//! | [`eri`] | records, harmonics | the `EriManager`: B, and J and K over it |
//! | [`guess`] | | the SAD density |
//! | [`scf`] | overlap, core_hamiltonian, orthogonalizer, eri, guess | E, C, ε, D |

pub mod core_hamiltonian;
pub mod eri;
pub mod guess;
pub mod harmonics;
pub mod orthogonalizer;
pub mod overlap;
pub mod records;
pub mod scf;

use fb_core_error_handling::Status;
use fb_core_wwr::blas::{BlasHandle, gemm};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
pub(crate) use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

use crate::bus::Shared;

/// The privileged card: every one-GPU step runs here.
pub(crate) const GPU0: usize = 0;

/// syevd's workspace at order `n` on gpu0, in bytes, for a `mem_budget`.
/// The query needs a live handle, so a dry budget, which has none, counts
/// no workspace.
pub(crate) fn syevd_workspace(shared: &Shared, n: usize) -> fb_core_error_handling::Result<usize> {
    match shared.resources.get(GPU0) {
        Some(library) => library
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .syevd_workspace(n),
        None => Ok(0),
    }
}

/// `out = Uᵀ cart U`: an `n_cart × n_cart` Cartesian matrix into the
/// `n_sph × n_sph` spherical one, through the `n_cart × n_sph` `work`.
pub(crate) fn to_spherical(
    blas: &BlasHandle,
    n_cart: usize,
    n_sph: usize,
    u: &impl DeviceSlice<f64>,
    cart: &impl DeviceSlice<f64>,
    work: &mut impl DeviceSliceMut<f64>,
    out: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let (c, s) = (n_cart as i32, n_sph as i32);
    gemm(blas, OP_N, OP_N, c, s, c, 1.0, cart, c, u, c, 0.0, work, c)?;
    gemm(blas, OP_T, OP_N, s, s, c, 1.0, u, c, &*work, c, 0.0, out, s)
}
