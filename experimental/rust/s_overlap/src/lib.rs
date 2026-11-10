//! The spherical AO overlap matrix S of a molecule in a basis, built on the
//! GPU by free_breakfast's pipeline steps (Molecule, BasisSet,
//! SphericalTransform, PrimitivePairClasses, Overlap) through the C ABI in
//! ffi/fb_overlap.h, and a reader for the .npy references it is checked
//! against.
//!
//! S comes back as an [`fb_gemm::Matrix`] on the caller's [`fb_gemm::Gpu`],
//! so it feeds [`fb_gemm::gemm`] like any other matrix.

pub mod npy;

use std::ffi::CString;
use std::path::Path;
use std::ptr;

use fb_gemm::ffi::{Buffer, Resources, Status};
use fb_gemm::{DeviceBuffer, Error, Gpu, Matrix, Result, check};

unsafe extern "C" {
    fn fb_overlap(
        res: *mut Resources,
        xyz_path: *const std::ffi::c_char,
        basis: *const std::ffi::c_char,
        out: *mut *mut Buffer,
        nao: *mut usize,
    ) -> Status;
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
    // SAFETY: the handle is live; both strings are NUL-terminated and outlive
    // the call; `raw` and `nao` are valid out-pointers, set only on FB_OK.
    check(unsafe {
        fb_overlap(
            gpu.as_raw(),
            xyz.as_ptr(),
            basis.as_ptr(),
            &mut raw,
            &mut nao,
        )
    })?;
    // SAFETY: on FB_OK `raw` is a fresh buffer from gpu's handle, owned by us.
    let s = unsafe { DeviceBuffer::from_raw(gpu, raw) };
    Matrix::from_buffer(nao, nao, s)
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
