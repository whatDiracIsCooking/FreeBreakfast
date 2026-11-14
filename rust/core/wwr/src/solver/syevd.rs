//! The dense solver's symmetric eigendecomposition: wwr's typed `syevd<T>`
//! and `syevd_lwork<T>` over `wwrsolverDn{S,D}syevd` and its legacy work
//! query, as [`SolverScalar`]'s raw calls and the safe [`syevd`] and
//! [`syevd_lwork`] over them.
//!
//! As [`blas`](crate::blas) does for `gemm` and `dgmm`: the raw layer takes
//! pointers and is `unsafe`, as the C ABI is, and only this crate calls
//! it; the safe call takes [`DeviceSlice`]s and checks every extent the
//! call will touch in every build, a slice too short a panic before
//! anything is enqueued.
//!
//! **The work is checked against the library's own query.** What the
//! library does with an `lwork` shorter than it needs is the vendor's
//! choice, so [`syevd`] asks the query itself (a host computation) and
//! panics on a shorter work slice, and passes the queried `lwork`, not the
//! slice's length.
//!
//! **`f32` and `f64`**, the C++'s `float` and `double`: the two real types
//! the C ABI's syevd takes. The complex `heevd` is a different routine and
//! is not wrapped.

use wwr_sys::{wwr_solver_eig_mode, wwr_solver_fill_mode, wwr_solver_handle, wwr_solver_status};

use super::SolverHandle;
use crate::blas::{check_extent, check_vector};
use crate::device_slice::DeviceSliceMut;
use crate::gpu_status;
use fb_core_error_handling::{Error, Result, Status};

/// A type the dense solver computes in: `f32` and `f64`. Each method is the
/// C ABI call of the type's letter; a routine wrapped later adds its own.
/// Sealed, since [`syevd`] and [`syevd_lwork`] trust each method to be that
/// call.
pub trait SolverScalar: Copy + sealed::Sealed {
    /// `wwrsolverDn<t>syevd_bufferSize`, the legacy query: the work
    /// [`syevd_raw`](SolverScalar::syevd_raw) needs at `n`, `lda` and the
    /// two modes, in elements, into `lwork`.
    ///
    /// # Safety
    /// The library's: `handle` is live and `lwork` valid for one write. The
    /// legacy query reads neither matrix pointer, which may be null.
    #[allow(clippy::too_many_arguments, reason = "the solver signature")]
    unsafe fn syevd_lwork_raw(
        handle: wwr_solver_handle,
        jobz: wwr_solver_eig_mode,
        uplo: wwr_solver_fill_mode,
        n: i32,
        a: *const Self,
        lda: i32,
        w: *const Self,
        lwork: *mut i32,
    ) -> wwr_solver_status;

    /// `wwrsolverDn<t>syevd`: the eigenvalues of the symmetric n × n A
    /// (its `uplo` triangle read, at `lda`) into w ascending, and under
    /// `WWR_SOLVER_EIG_MODE_VECTOR` its eigenvectors over A, column `j`
    /// that of `w[j]`; the flag into `info`.
    ///
    /// # Safety
    /// The library's: `handle` is live; A, w, the work and `info` are
    /// device memory on its device covering n × n at `lda`, n, `lwork` and
    /// one element, apart from one another; `lwork` is at least the
    /// query's; all of it alive until the work enqueued on the handle's
    /// stream has finished.
    #[allow(clippy::too_many_arguments, reason = "the solver signature")]
    unsafe fn syevd_raw(
        handle: wwr_solver_handle,
        jobz: wwr_solver_eig_mode,
        uplo: wwr_solver_fill_mode,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) -> wwr_solver_status;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! solver_scalar {
    ($($t:ty => $lwork:ident, $syevd:ident);* $(;)?) => {$(
        impl sealed::Sealed for $t {}

        impl SolverScalar for $t {
            unsafe fn syevd_lwork_raw(
                handle: wwr_solver_handle,
                jobz: wwr_solver_eig_mode,
                uplo: wwr_solver_fill_mode,
                n: i32,
                a: *const Self,
                lda: i32,
                w: *const Self,
                lwork: *mut i32,
            ) -> wwr_solver_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$lwork(handle, jobz, uplo, n, a, lda, w, lwork) }
            }

            unsafe fn syevd_raw(
                handle: wwr_solver_handle,
                jobz: wwr_solver_eig_mode,
                uplo: wwr_solver_fill_mode,
                n: i32,
                a: *mut Self,
                lda: i32,
                w: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) -> wwr_solver_status {
                // SAFETY: the caller's, which are the call's.
                unsafe { wwr_sys::$syevd(handle, jobz, uplo, n, a, lda, w, work, lwork, info) }
            }
        }
    )*};
}

solver_scalar! {
    f32 => wwr_solver_dn_ssyevd_buffer_size, wwr_solver_dn_ssyevd;
    f64 => wwr_solver_dn_dsyevd_buffer_size, wwr_solver_dn_dsyevd;
}

/// The work [`syevd`] needs for order `n` at leading dimension `lda` and
/// the two modes, in elements: the solver's legacy query on `handle`, a
/// host computation. wwr's `syevd_lwork<T>`.
///
/// # Errors
/// The query's refusal, as a `GpuRuntime` error naming `syevd_lwork` and
/// `n`.
///
/// # Panics
/// If `n` is negative or `lda` is less than `max(1, n)`.
#[track_caller]
pub fn syevd_lwork<T: SolverScalar>(
    handle: &SolverHandle,
    jobz: wwr_solver_eig_mode,
    uplo: wwr_solver_fill_mode,
    n: i32,
    lda: i32,
) -> Result<usize> {
    assert!(n >= 0, "syevd_lwork: n is {n}");
    assert!(
        lda >= n.max(1),
        "syevd_lwork: A's leading dimension {lda} is less than max(1, {n} rows)"
    );
    let mut lwork = 0_i32;
    // SAFETY: the handle is live; the legacy query reads neither matrix
    // pointer, and writes one integer through a valid local.
    let status = unsafe {
        T::syevd_lwork_raw(
            handle.get(),
            jobz,
            uplo,
            n,
            std::ptr::null(),
            lda,
            std::ptr::null(),
            &mut lwork,
        )
    };
    if let Err(e) = gpu_status(status) {
        return Err(Error::gpu(format!("syevd_lwork (n = {n}): {}", e.message)));
    }
    Ok(usize::try_from(lwork).expect("the solver's work is not negative"))
}

/// The eigendecomposition of the symmetric n × n A on `handle`'s stream:
/// the eigenvalues into `w`, ascending, and under
/// `WWR_SOLVER_EIG_MODE_VECTOR` the eigenvectors over A, column `j` that of
/// `w[j]` (under `_NOVECTOR` A is destroyed); the solver's flag into
/// `info[0]`, 0 on success, `-i` for a refused argument `i`, positive for
/// no convergence. wwr's `syevd<T>`, with every operand a slice it checks.
///
/// Only A's `uplo` triangle is read, at leading dimension `lda`; its rows
/// past n are left as they were. `work` is at least [`syevd_lwork`] long at
/// the same `n`, `lda` and modes, which the call asks again to check.
/// The flag is written on the device, so reading it is the caller's, after
/// the stream: the call does not wait. The borrow checker keeps the four
/// slices apart.
///
/// # Errors
/// The library's refusal of the query or the call, as a `GpuRuntime` error
/// naming `syevd_lwork` or `syevd` and `n`.
///
/// # Panics
/// If `n` is negative, `lda` is less than `max(1, n)`, or a slice is
/// shorter than its operand: A n × n at `lda`, w n, the work the query's
/// `lwork`, `info` one; before anything is enqueued.
#[allow(clippy::too_many_arguments, reason = "the solver signature")]
#[track_caller]
pub fn syevd<T: SolverScalar>(
    handle: &SolverHandle,
    jobz: wwr_solver_eig_mode,
    uplo: wwr_solver_fill_mode,
    n: i32,
    a: &mut impl DeviceSliceMut<T>,
    lda: i32,
    w: &mut impl DeviceSliceMut<T>,
    work: &mut impl DeviceSliceMut<T>,
    info: &mut impl DeviceSliceMut<i32>,
) -> Status {
    assert!(n >= 0, "syevd: n is {n}");
    check_extent("syevd", "A", a.len(), n, n, lda);
    check_vector("syevd", "w", w.len(), n, 1);
    check_vector("syevd", "info", info.len(), 1, 1);
    let lwork = syevd_lwork::<T>(handle, jobz, uplo, n, lda)?;
    // The query's lwork came back as an i32.
    let lwork = lwork as i32;
    check_vector("syevd", "work", work.len(), lwork, 1);
    let (a, w, work, info) = (
        a.as_mut_ptr(),
        w.as_mut_ptr(),
        work.as_mut_ptr(),
        info.as_mut_ptr(),
    );
    // SAFETY: the handle is live; each slice is device memory covering its
    // extent, checked above, the work at least the query's lwork, and stays
    // so until the enqueued work finishes (DeviceSlice's promise); each is
    // its memory's only user (DeviceSliceMut's), so the four are apart.
    let status = unsafe { T::syevd_raw(handle.get(), jobz, uplo, n, a, lda, w, work, lwork, info) };
    if let Err(e) = gpu_status(status) {
        return Err(Error::gpu(format!("syevd (n = {n}): {}", e.message)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_slice::DeviceSlice;
    use crate::runtime::Stream;
    use crate::runtime::tests::DeviceBytes;
    use std::rc::Rc;

    const VECTOR: wwr_solver_eig_mode = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
    const LOWER: wwr_solver_fill_mode = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;

    fn download_f64(device: &DeviceBytes, owner: &Stream) -> Vec<f64> {
        device
            .read(owner)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_ne_bytes(*c))
            .collect()
    }

    /// Needs a GPU. A 3 × 3 symmetric A at a padded leading dimension, its
    /// upper triangle garbage: the eigenvalues of the lower one, each
    /// eigenvector A's (`A v = λ v`) and of unit length, and the padding
    /// rows left as they were.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn syevd_reads_the_lower_triangle_at_its_leading_dimension() {
        const N: usize = 3;
        const LDA: usize = 5;
        // The lower triangle of [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
        // eigenvalues 2 - sqrt(2), 2 and 2 + sqrt(2).
        let lower = |i: usize, j: usize| match i.abs_diff(j) {
            0 => 2.0,
            1 => -1.0,
            _ => 0.0,
        };
        let a: Vec<f64> = (0..LDA * N)
            .map(|e| {
                let (i, j) = (e % LDA, e / LDA);
                if i >= N {
                    7.0 // padding
                } else if i >= j {
                    lower(i, j)
                } else {
                    99.0 // the upper triangle, which must not be read
                }
            })
            .collect();
        let owner = Rc::new(Stream::new(0));
        let solver = SolverHandle::new(owner.clone());
        let n = N as i32;
        let lwork = syevd_lwork::<f64>(&solver, VECTOR, LOWER, n, LDA as i32).unwrap();
        assert!(lwork > 0);
        let mut da = DeviceBytes::upload(&a, &owner);
        let mut dw = DeviceBytes::new(N * size_of::<f64>());
        let mut work = DeviceBytes::new(lwork * size_of::<f64>());
        let mut info = DeviceBytes::new(size_of::<i32>());
        syevd::<f64>(
            &solver, VECTOR, LOWER, n, &mut da, LDA as i32, &mut dw, &mut work, &mut info,
        )
        .unwrap();
        let flag = i32::from_ne_bytes(info.read(&owner)[..4].try_into().unwrap());
        assert_eq!(flag, 0);
        let w = download_f64(&dw, &owner);
        let v = download_f64(&da, &owner);
        let s = 2.0_f64.sqrt();
        // A backward-stable solver's error here is about n eps |A|, 3 x
        // 2.2e-16 x 3.4 = 2.3e-15 in each eigenvalue and each residual
        // entry, and over the smallest gap (0.59) in each eigenvector:
        // 1e-14 holds it with room, and a wrong answer misses by O(1).
        let tol = 1e-14;
        for (got, want) in w.iter().zip([2.0 - s, 2.0, 2.0 + s]) {
            assert!((got - want).abs() < tol, "w = {w:?}");
        }
        for (j, &lambda) in w.iter().enumerate() {
            let col = |i: usize| v[i + j * LDA];
            let norm: f64 = (0..N).map(|i| col(i) * col(i)).sum();
            assert!((norm - 1.0).abs() < tol, "column {j}: |v|² = {norm}");
            for i in 0..N {
                let av: f64 = (0..N).map(|k| lower(i.max(k), i.min(k)) * col(k)).sum();
                assert!((av - lambda * col(i)).abs() < tol, "column {j}, row {i}");
            }
        }
        for j in 0..N {
            for i in N..LDA {
                assert_eq!(v[i + j * LDA], 7.0, "padding ({i}, {j})");
            }
        }
    }

    /// Needs a GPU. A work slice one element short of the query's lwork
    /// panics before anything is enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn syevd_refuses_a_work_shorter_than_the_query() {
        let owner = Rc::new(Stream::new(0));
        let solver = SolverHandle::new(owner);
        let lwork = syevd_lwork::<f64>(&solver, VECTOR, LOWER, 4, 4).unwrap();
        let mut a = DeviceBytes::new(16 * size_of::<f64>());
        let mut w = DeviceBytes::new(4 * size_of::<f64>());
        let mut work = DeviceBytes::new((lwork - 1) * size_of::<f64>());
        let mut info = DeviceBytes::new(size_of::<i32>());
        assert_eq!(DeviceSlice::<f64>::len(&work), lwork - 1);
        let want = format!(
            "syevd: work is {lwork} entries at stride 1, {lwork} elements, but its slice \
             holds {}",
            lwork - 1
        );
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = syevd::<f64>(
                &solver, VECTOR, LOWER, 4, &mut a, 4, &mut w, &mut work, &mut info,
            );
        }))
        .expect_err("a short work panics");
        assert_eq!(got.downcast_ref::<String>(), Some(&want));
    }

    /// Needs a GPU. An A one element short of n × n at its leading
    /// dimension panics, before the query.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(
        expected = "syevd: A is 4 x 4 at leading dimension 4, 16 elements, but its slice holds 15"
    )]
    fn syevd_refuses_a_short_matrix() {
        let owner = Rc::new(Stream::new(0));
        let solver = SolverHandle::new(owner);
        let mut a = DeviceBytes::new(15 * size_of::<f64>());
        let mut w = DeviceBytes::new(4 * size_of::<f64>());
        let mut work = DeviceBytes::new(8);
        let mut info = DeviceBytes::new(size_of::<i32>());
        let _ = syevd::<f64>(
            &solver, VECTOR, LOWER, 4, &mut a, 4, &mut w, &mut work, &mut info,
        );
    }
}
