//! The spherical AO overlap matrix S of a molecule in a basis, built on the
//! GPU by free_breakfast's pipeline steps (Molecule, BasisSet,
//! SphericalTransform, PrimitivePairClasses, Overlap) through the C ABI in
//! ffi/fb_overlap.h, and a reader for the .npy references it is checked
//! against.
//!
//! S comes back as an [`fb_gemm::Matrix`] on the caller's [`fb_gemm::Gpu`],
//! so it feeds [`fb_gemm::gemm`] like any other matrix. The steps run on
//! DeviceResources of the C ABI's own; S is then copied, device to device,
//! into a buffer the `Gpu` allocated.

pub mod npy;

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use fb_gemm::{Error, Gpu, Matrix, Result};

/// fb_overlap.h's C ABI, declared by hand: keep it in step with the header.
mod ffi {
    use std::ffi::c_char;

    pub type Status = i32;
    pub const FB_OK: Status = 0;
    pub const FB_INVALID_ARGUMENT: Status = 1;
    pub const FB_GPU_ERROR: Status = 2;
    pub const FB_FAILED: Status = 3;

    /// Opaque: only ever behind a pointer.
    #[repr(C)]
    pub struct Overlap {
        _opaque: [u8; 0],
    }

    unsafe extern "C" {
        pub fn fb_last_error() -> *const c_char;
        pub fn fb_overlap_new(
            dev_idx: i32,
            xyz_path: *const c_char,
            basis: *const c_char,
            out: *mut *mut Overlap,
            nao: *mut usize,
        ) -> Status;
        pub fn fb_overlap_copy(s: *const Overlap, dst: *mut f64, len: usize) -> Status;
        pub fn fb_overlap_free(s: *mut Overlap);
    }
}

/// A C status as a [`Result`]; FB_FAILED carries fb_last_error()'s message.
fn check(call: &str, status: ffi::Status) -> Result<()> {
    match status {
        ffi::FB_OK => Ok(()),
        ffi::FB_INVALID_ARGUMENT => Err(Error::InvalidArgument),
        ffi::FB_GPU_ERROR => Err(Error::Failed(format!("{call}: GPU runtime error"))),
        ffi::FB_FAILED => {
            // SAFETY: fb_last_error returns a NUL-terminated string that
            // stays valid until this thread's next failing call.
            let why = unsafe { CStr::from_ptr(ffi::fb_last_error()) };
            Err(Error::Failed(why.to_string_lossy().into_owned()))
        }
        other => Err(Error::Failed(format!("{call}: unknown status {other}"))),
    }
}

/// Frees an fb_overlap on every way out of [`overlap`].
struct Owned(*mut ffi::Overlap);

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: a live handle from fb_overlap_new, freed once.
        unsafe { ffi::fb_overlap_free(self.0) }
    }
}

/// S of the molecule in XYZ file `xyz` (Angstrom) in `basis` (a carried name
/// such as "cc-pvdz", or a path to a BSE JSON file): nao x nao, column-major,
/// in PySCF's spherical AO order.
pub fn overlap<'g>(gpu: &'g Gpu, xyz: &Path, basis: &str) -> Result<Matrix<'g>> {
    let xyz = xyz
        .to_str()
        .ok_or_else(|| Error::Failed(format!("{} is not UTF-8", xyz.display())))?;
    let c_string =
        |s: &str| CString::new(s).map_err(|_| Error::Failed(format!("{s:?} holds a NUL")));
    let (xyz, basis) = (c_string(xyz)?, c_string(basis)?);
    let mut raw = ptr::null_mut();
    let mut nao = 0usize;
    // SAFETY: both strings are NUL-terminated and outlive the call; `raw` and
    // `nao` are valid out-pointers, set only on FB_OK.
    check("fb_overlap_new", unsafe {
        ffi::fb_overlap_new(
            gpu.device(),
            xyz.as_ptr(),
            basis.as_ptr(),
            &mut raw,
            &mut nao,
        )
    })?;
    let s = Owned(raw);
    let mut buf = gpu.alloc(nao * nao)?;
    // The allocation is stream-ordered on gpu's stream and the copy runs on
    // the C ABI's own, so wait for the allocation first.
    gpu.sync()?;
    // SAFETY: `buf` is nao * nao doubles of device memory on the same device,
    // with nothing queued against it.
    check("fb_overlap_copy", unsafe {
        ffi::fb_overlap_copy(s.0, buf.as_mut_ptr(), buf.len())
    })?;
    Matrix::from_buffer(nao, nao, buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// PySCF's int1e_ovlp over the same BSE basis and the same bohr
    /// coordinates (data/reference.py), at FelizJueves' bar for its
    /// one-electron kernels against libcint: 1e-12 * max|S|.
    fn matches_pyscf(xyz: &str, basis: &str, reference: &str) {
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
        let gpu = Gpu::new(0).unwrap();
        let s = overlap(&gpu, &data.join(xyz), basis).unwrap();
        let (shape, want) = npy::read_f64_matrix(&data.join(reference)).unwrap();
        assert_eq!(shape, [s.rows(), s.cols()], "{reference}");
        let got = s.to_vec().unwrap();
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        assert!(
            worst <= 1e-12 * scale,
            "{reference}: max |S - S_pyscf| = {worst:e}"
        );
    }

    #[test]
    fn h2o_sto3g_matches_pyscf() {
        matches_pyscf("h2o.xyz", "sto-3g", "h2o_sto-3g.npy");
    }

    #[test]
    fn h2o_ccpvdz_matches_pyscf() {
        matches_pyscf("h2o.xyz", "cc-pvdz", "h2o_cc-pvdz.npy");
    }

    /// f on both atoms, the bond off every axis: every class through (f, f).
    #[test]
    fn co_ccpvtz_matches_pyscf() {
        matches_pyscf("co.xyz", "cc-pvtz", "co_cc-pvtz.npy");
    }

    #[test]
    fn unknown_basis_says_why() {
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
        let gpu = Gpu::new(0).unwrap();
        match overlap(&gpu, &data.join("h2o.xyz"), "no-such-basis") {
            Err(Error::Failed(why)) => assert!(why.contains("no-such-basis"), "{why}"),
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("an unknown basis built an S"),
        }
    }

    #[test]
    fn missing_xyz_says_why() {
        let gpu = Gpu::new(0).unwrap();
        match overlap(&gpu, Path::new("/nonexistent/mol.xyz"), "sto-3g") {
            Err(Error::Failed(why)) => assert!(why.contains("mol.xyz"), "{why}"),
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("a missing file built an S"),
        }
    }
}
