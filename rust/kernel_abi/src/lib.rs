//! What every kernel library under `rust/` shares above its `-sys` crate,
//! as functions of values passed in, so this crate links nothing and its
//! tests are tier 3, where no library is built:
//!
//! - [`check`], the kernel libraries' load check, the comparison half
//!   (`docs/gpu_layer.md`, "What the port changes"): what a
//!   `libfb_<module>_kernels.so` reports, its ABI version and backend,
//!   against what its `-sys` crate expects. Each `-sys` crate's
//!   `check_library()` reads the two exports and calls it.
//! - [`launch_status`], a launch wrapper's `fb_launch_status` as a
//!   [`Status`] (`docs/gpu_layer.md`, "Statuses"), which every safe layer
//!   over a `-sys` crate calls on each wrapper's return.
//!
//! The backend is the `fb_backend` of `device_headers/fb_launch.h`, found as
//! the library's number (`FB_BACKEND_CUDA` = 1, `FB_BACKEND_HIP` = 2, which
//! each `-sys` crate asserts against its bindings) and expected as the name
//! `rust/kernel_lib.rs` reads from the linked build's `CMakeCache.txt`. The
//! launch status is the same header's `fb_launch_status`, read by its numbers
//! too (`FB_LAUNCH_SUCCESS` = 0, `FB_LAUNCH_UNSUPPORTED` = -1,
//! `FB_LAUNCH_INVALID_ARGUMENT` = -2, asserted the same way); its positive
//! range is the C ABI's `wwr_status`, which this crate cannot name without
//! linking wwr, so the caller hands that range over.

#![forbid(unsafe_code)]

use std::panic::Location;

use fb_core_error_handling::{Error, Status};

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

/// A launch wrapper's `fb_launch_status` as a [`Status`], naming `what`, the
/// wrapper and whatever the caller adds (`fb_overlap_class (1, 2)`):
///
/// | `status` | Here |
/// |---|---|
/// | `FB_LAUNCH_SUCCESS` (0) | `Ok(())` |
/// | `FB_LAUNCH_UNSUPPORTED` (-1) | `Unsupported`, "`what`: no kernel" |
/// | `FB_LAUNCH_INVALID_ARGUMENT` (-2) | `InvalidConfig`, "`what`: refused its arguments" |
/// | positive, a `wwr_status` | `runtime(status, at)`, with `what` as its context |
/// | anything else, or a positive value `runtime` does not know (`None`) | `GpuRuntime`, "`what`: unknown launch status `status`" |
///
/// `runtime` is `fb_core_wwr::gpu_status_at` over the `wwr_status` of that
/// value, so a launch error carries the same name and description as every
/// other wwr call's, at `at`, this function's caller.
///
/// # Errors
///
/// Every status but `FB_LAUNCH_SUCCESS`, as above.
#[track_caller]
pub fn launch_status(
    what: &str,
    status: i32,
    runtime: impl FnOnce(i32, &'static Location<'static>) -> Option<Status>,
) -> Status {
    match status {
        0 => Ok(()),
        -1 => Err(Error::unsupported(format!("{what}: no kernel"))),
        -2 => Err(Error::config(format!("{what}: refused its arguments"))),
        1.. if let Some(s) = runtime(status, Location::caller()) => {
            s.map_err(|e| e.with_context(what))
        }
        _ => Err(Error::gpu(format!(
            "{what}: unknown launch status {status}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::Location;

    use fb_core_error_handling::{Error, ErrorKind, Status};

    use super::{check, launch_status};

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

    /// A runtime that knows the codes 1 to 7, `wwr_status`'s, as
    /// `gpu_status_at` makes them: `GpuRuntime` at the location handed in.
    fn runtime(code: i32, at: &'static Location<'static>) -> Option<Status> {
        (1..=7).contains(&code).then(|| {
            Err(Error {
                origin: at,
                ..Error::gpu(format!("WWR_STATUS_{code}"))
            })
        })
    }

    fn kind(status: i32) -> Option<ErrorKind> {
        launch_status("fb_overlap_class (1, 2)", status, runtime)
            .err()
            .map(|e| e.kind)
    }

    #[test]
    fn every_status_maps_to_its_kind() {
        assert_eq!(kind(0), None);
        assert_eq!(kind(-1), Some(ErrorKind::Unsupported));
        assert_eq!(kind(-2), Some(ErrorKind::InvalidConfig));
        for code in 1..=7 {
            assert_eq!(kind(code), Some(ErrorKind::GpuRuntime), "{code}");
        }
        for code in [-3, i32::MIN, 8, i32::MAX] {
            assert_eq!(kind(code), Some(ErrorKind::GpuRuntime), "{code}");
        }
    }

    #[test]
    fn the_message_names_the_wrapper_and_the_status() {
        let what = "fb_kinetic_class (3, 3)";
        let msg = |status| launch_status(what, status, runtime).unwrap_err().message;
        assert_eq!(msg(-1), "fb_kinetic_class (3, 3): no kernel");
        assert_eq!(msg(-2), "fb_kinetic_class (3, 3): refused its arguments");
        assert_eq!(msg(-3), "fb_kinetic_class (3, 3): unknown launch status -3");
        assert_eq!(msg(8), "fb_kinetic_class (3, 3): unknown launch status 8");
        let e = launch_status(what, 5, runtime).unwrap_err();
        assert_eq!(e.message, "WWR_STATUS_5");
        assert_eq!(e.context, [what]);
    }

    #[test]
    fn a_negative_status_never_reaches_the_runtime() {
        for status in [-1, -2, -3, i32::MIN] {
            let _ = launch_status("w", status, |_, _| unreachable!("{status}"));
        }
        assert_eq!(launch_status("w", 0, |_, _| unreachable!()), Ok(()));
    }

    #[test]
    fn every_error_is_located_at_the_caller() {
        for status in [-1, -2, -3, 3, 8] {
            let here = Location::caller();
            let e = launch_status("w", status, runtime).unwrap_err();
            assert_eq!(
                (e.origin.file(), e.origin.line()),
                (here.file(), here.line() + 1)
            );
        }
    }
}
