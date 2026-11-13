//! The kernel libraries' load check, the comparison half
//! (`docs/gpu_layer.md`, "What the port changes"): what a
//! `libfb_<module>_kernels.so` reports, its ABI version and backend, against
//! what its `-sys` crate expects. Each `-sys` crate's `check_library()` reads
//! the two exports and calls [`check`]; this crate links nothing, so the
//! mismatch tests run in tier 3, where no library is built.
//!
//! The backend is the `fb_backend` of `device_headers/fb_launch.h`, found as
//! the library's number (`FB_BACKEND_CUDA` = 1, `FB_BACKEND_HIP` = 2, which
//! each `-sys` crate asserts against its bindings) and expected as the name
//! `rust/kernel_lib.rs` reads from the linked build's `CMakeCache.txt`.

#![forbid(unsafe_code)]

/// Checks a kernel library's `(ABI version, fb_backend)`, `found`, against
/// the `(ABI version, backend name)` its `-sys` crate expects, `expected`;
/// `module` names both, `recipes` for `libfb_recipes_kernels.so` and
/// `fb-recipes-sys`.
///
/// # Errors
///
/// A mismatch of either, with a message naming the found and the expected
/// values.
pub fn check(module: &str, found: (u32, i32), expected: (u32, &str)) -> Result<(), String> {
    let backend = match found.1 {
        1 => "CUDA".to_owned(),
        2 => "HIP".to_owned(),
        other => format!("unknown backend {other}"),
    };
    if (found.0, backend.as_str()) == expected {
        return Ok(());
    }
    Err(format!(
        "libfb_{module}_kernels.so is ABI version {} built for {backend}, but \
         fb-{module}-sys expects ABI version {} built for {}: rebuild the library \
         from this tree for this backend (README.md, \"Build and test\")",
        found.0, expected.0, expected.1
    ))
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn equal_values_pass() {
        assert_eq!(check("recipes", (3, 2), (3, "HIP")), Ok(()));
        assert_eq!(check("integrals", (1, 1), (1, "CUDA")), Ok(()));
    }

    #[test]
    fn another_version_is_refused_naming_both() {
        let e = check("recipes", (2, 1), (3, "CUDA")).unwrap_err();
        assert!(
            e.starts_with("libfb_recipes_kernels.so is ABI version 2 built for CUDA,"),
            "{e}"
        );
        assert!(
            e.contains("fb-recipes-sys expects ABI version 3 built for CUDA:"),
            "{e}"
        );
    }

    #[test]
    fn another_backend_is_refused_naming_both() {
        let e = check("integrals", (3, 1), (3, "HIP")).unwrap_err();
        assert!(
            e.starts_with("libfb_integrals_kernels.so is ABI version 3 built for CUDA,"),
            "{e}"
        );
        assert!(
            e.contains("fb-integrals-sys expects ABI version 3 built for HIP:"),
            "{e}"
        );
        let e = check("integrals", (3, 7), (3, "HIP")).unwrap_err();
        assert!(e.contains("built for unknown backend 7,"), "{e}");
    }
}
