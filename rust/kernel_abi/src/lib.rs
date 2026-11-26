//! What every kernel library under `rust/` shares above its `-sys` crate,
//! as types and functions of values passed in, so this crate links nothing
//! and its tests are tier 3, where no library is built:
//!
//! - The plain-C vocabulary of `device_headers/fb_launch.h`, generated from
//!   it by `devtools/rust-bindings.py` into this crate's one Rust copy
//!   (`docs/gpu_layer.md`, "From bridge to C ABI"): [`fb_launch_status`] and
//!   its values, [`FB_DEVICE_ALIGNMENT`], [`fb_backend`] and its values.
//!   Each `-sys` crate's bindings re-export them, so a test may still write
//!   `fb_recipes_sys::FB_LAUNCH_SUCCESS`.
//! - [`LaunchStatus`], the type a launch wrapper returns, and [`Backend`],
//!   a library's `fb_backend` read by value.
//! - [`check`], the kernel libraries' load check, the comparison half
//!   (`docs/gpu_layer.md`, "What the port changes"): what a
//!   `libfb_<module>_kernels.so` reports, its ABI version and backend,
//!   against what its `-sys` crate expects. Each `-sys` crate's
//!   `check_library()` reads the two exports and calls it.
//! - [`launch_status`], a launch wrapper's [`LaunchStatus`] as a [`Status`]
//!   (`docs/gpu_layer.md`, "Statuses"), which every safe layer over a `-sys`
//!   crate calls on each wrapper's return.
//!
//! Both functions read the header's values by the generated names, never by
//! number. A launch status's positive range is the C ABI's `wwr_status`,
//! which this crate cannot name without linking wwr, so the caller hands
//! that range over.

#![forbid(unsafe_code)]
#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

use std::fmt;
use std::panic::Location;

use fb_core_error_handling::{Error, Status};

/// What a launch wrapper returns, [`fb_launch_status`] in the C header: an
/// `int32_t` that is [`FB_LAUNCH_SUCCESS`], a positive `wwr_status` (the
/// runtime's launch error) or a negative refusal ([`FB_LAUNCH_UNSUPPORTED`],
/// [`FB_LAUNCH_INVALID_ARGUMENT`]). [`launch_status`] turns it into a
/// [`Status`].
///
/// A type of its own, `#[repr(transparent)]` over the `i32`, so it is the
/// extern functions' return type as it is and every value the C side can
/// return is one of it, while a launch status is neither a `wwr_status` nor
/// a count:
///
/// ```compile_fail,E0308
/// let code: i32 = fb_kernel_abi::FB_LAUNCH_SUCCESS;
/// ```
///
/// Nor is one made from a number outside this crate; the header's values
/// are the only ones a caller names:
///
/// ```compile_fail,E0423
/// let status = fb_kernel_abi::LaunchStatus(0);
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "a launch wrapper reports a refusal or a launch error only here"]
pub struct LaunchStatus(i32);

// launch_status reads FB_LAUNCH_SUCCESS as 0 and the refusals as negative,
// leaving the positive range to wwr_status: the header's values must sit
// there, each refusal its own.
const _: () = assert!(
    FB_LAUNCH_SUCCESS.0 == 0
        && FB_LAUNCH_UNSUPPORTED.0 < 0
        && FB_LAUNCH_INVALID_ARGUMENT.0 < 0
        && FB_LAUNCH_UNSUPPORTED.0 != FB_LAUNCH_INVALID_ARGUMENT.0
);

/// The GPU backend a kernel library was built for: the header's
/// [`fb_backend`], read by value with [`TryFrom`] (an unknown value is
/// handed back, never made a variant), and named by [`fmt::Display`] as the
/// CMake build names it, `CUDA` or `HIP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// [`FB_BACKEND_CUDA`].
    Cuda,
    /// [`FB_BACKEND_HIP`].
    Hip,
}

impl Backend {
    /// The backend `name`s, as its [`fmt::Display`] and the CMake build's
    /// `FREE_BREAKFAST_GPU_BACKEND` spell it. `const`, so a `-sys` crate
    /// reads the name `rust/kernel_lib.rs` hands it at compile time:
    ///
    /// ```
    /// use fb_kernel_abi::Backend;
    ///
    /// const BUILT_FOR: Backend = Backend::named("HIP");
    /// assert_eq!(BUILT_FOR, Backend::Hip);
    /// ```
    ///
    /// # Panics
    ///
    /// If `name` is neither `CUDA` nor `HIP`; at compile time, in a `const`.
    #[must_use]
    pub const fn named(name: &str) -> Self {
        match name.as_bytes() {
            b"CUDA" => Self::Cuda,
            b"HIP" => Self::Hip,
            _ => panic!("a backend is CUDA or HIP"),
        }
    }
}

impl TryFrom<fb_backend> for Backend {
    /// The value, which names no backend.
    type Error = fb_backend;

    fn try_from(value: fb_backend) -> Result<Self, fb_backend> {
        match value {
            FB_BACKEND_CUDA => Ok(Self::Cuda),
            FB_BACKEND_HIP => Ok(Self::Hip),
            other => Err(other),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cuda => "CUDA",
            Self::Hip => "HIP",
        })
    }
}

/// Checks a kernel library's `(ABI version, fb_backend)`, `found`, as its
/// two exports return them, against the `(ABI version, backend)` its `-sys`
/// crate expects, `expected`; `module` names both, `recipes` for
/// `libfb_recipes_kernels.so` and `fb-recipes-sys`.
///
/// # Errors
///
/// A mismatch of either, or a backend value [`Backend`] does not know,
/// with a message naming the found and the expected values.
pub fn check(
    module: &str,
    found: (u32, fb_backend),
    expected: (u32, Backend),
) -> Result<(), String> {
    let backend = Backend::try_from(found.1);
    if (found.0, backend) == (expected.0, Ok(expected.1)) {
        return Ok(());
    }
    let backend = match backend {
        Ok(b) => b.to_string(),
        Err(other) => format!("unknown backend {other}"),
    };
    Err(format!(
        "libfb_{module}_kernels.so is ABI version {} built for {backend}, but \
         fb-{module}-sys expects ABI version {} built for {}: rebuild the library \
         from this tree for this backend (README.md, \"Build and test\")",
        found.0, expected.0, expected.1
    ))
}

/// A launch wrapper's [`LaunchStatus`] as a [`Status`], naming `what`, the
/// wrapper and whatever the caller adds (`fb_overlap_class (1, 2)`):
///
/// | `status` | Here |
/// |---|---|
/// | [`FB_LAUNCH_SUCCESS`] | `Ok(())` |
/// | [`FB_LAUNCH_UNSUPPORTED`] | `Unsupported`, "`what`: no kernel" |
/// | [`FB_LAUNCH_INVALID_ARGUMENT`] | `InvalidConfig`, "`what`: refused its arguments" |
/// | positive, a `wwr_status` | `runtime(code, at)`, with `what` as its context |
/// | anything else, or a positive code `runtime` does not know (`None`) | `GpuRuntime`, "`what`: unknown launch status `code`" |
///
/// `runtime` is the caller's bridge from the `wwr_status` of that code
/// (`fb_core_device`'s), so a launch error carries the same name and
/// description as every other wwr call's, at `at`, this function's caller.
///
/// # Errors
///
/// Every status but [`FB_LAUNCH_SUCCESS`], as above.
#[track_caller]
pub fn launch_status(
    what: &str,
    status: LaunchStatus,
    runtime: impl FnOnce(i32, &'static Location<'static>) -> Option<Status>,
) -> Status {
    match status {
        FB_LAUNCH_SUCCESS => Ok(()),
        FB_LAUNCH_UNSUPPORTED => Err(Error::unsupported(format!("{what}: no kernel"))),
        FB_LAUNCH_INVALID_ARGUMENT => Err(Error::config(format!("{what}: refused its arguments"))),
        LaunchStatus(code)
            if code > 0
                && let Some(s) = runtime(code, Location::caller()) =>
        {
            s.map_err(|e| e.with_context(what))
        }
        LaunchStatus(code) => Err(Error::gpu(format!("{what}: unknown launch status {code}"))),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::Location;

    use fb_core_error_handling::{Error, ErrorKind, Status};

    use super::{
        Backend, FB_BACKEND_CUDA, FB_BACKEND_HIP, FB_LAUNCH_INVALID_ARGUMENT, FB_LAUNCH_SUCCESS,
        FB_LAUNCH_UNSUPPORTED, LaunchStatus, check, launch_status,
    };

    #[test]
    fn equal_values_pass() {
        assert_eq!(
            check("recipes", (3, FB_BACKEND_HIP), (3, Backend::Hip)),
            Ok(())
        );
        assert_eq!(
            check("integrals", (1, FB_BACKEND_CUDA), (1, Backend::Cuda)),
            Ok(())
        );
    }

    #[test]
    fn another_version_is_refused_naming_both() {
        let e = check("recipes", (2, FB_BACKEND_CUDA), (3, Backend::Cuda)).unwrap_err();
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
        let e = check("integrals", (3, FB_BACKEND_CUDA), (3, Backend::Hip)).unwrap_err();
        assert!(
            e.starts_with("libfb_integrals_kernels.so is ABI version 3 built for CUDA,"),
            "{e}"
        );
        assert!(
            e.contains("fb-integrals-sys expects ABI version 3 built for HIP:"),
            "{e}"
        );
        let e = check("integrals", (3, 7), (3, Backend::Hip)).unwrap_err();
        assert!(e.contains("built for unknown backend 7,"), "{e}");
    }

    /// The header's two values read back as the two variants and no other
    /// value as any, and each variant's name is the one `named` reads: what
    /// the load check's message and `rust/kernel_lib.rs`'s name rest on.
    #[test]
    fn every_backend_reads_back_from_its_value_and_its_name() {
        for (value, backend) in [
            (FB_BACKEND_CUDA, Backend::Cuda),
            (FB_BACKEND_HIP, Backend::Hip),
        ] {
            assert_eq!(Backend::try_from(value), Ok(backend));
            assert_eq!(Backend::named(&backend.to_string()), backend);
        }
        for value in [0, -1, 3, i32::MIN, i32::MAX] {
            assert_eq!(Backend::try_from(value), Err(value));
        }
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
        launch_status("fb_overlap_class (1, 2)", LaunchStatus(status), runtime)
            .err()
            .map(|e| e.kind)
    }

    #[test]
    fn every_status_maps_to_its_kind() {
        let kind_of = |s: LaunchStatus| kind(s.0);
        assert_eq!(kind_of(FB_LAUNCH_SUCCESS), None);
        assert_eq!(kind_of(FB_LAUNCH_UNSUPPORTED), Some(ErrorKind::Unsupported));
        assert_eq!(
            kind_of(FB_LAUNCH_INVALID_ARGUMENT),
            Some(ErrorKind::InvalidConfig)
        );
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
        let msg = |status| {
            launch_status(what, LaunchStatus(status), runtime)
                .unwrap_err()
                .message
        };
        assert_eq!(
            msg(FB_LAUNCH_UNSUPPORTED.0),
            "fb_kinetic_class (3, 3): no kernel"
        );
        assert_eq!(
            msg(FB_LAUNCH_INVALID_ARGUMENT.0),
            "fb_kinetic_class (3, 3): refused its arguments"
        );
        assert_eq!(msg(-3), "fb_kinetic_class (3, 3): unknown launch status -3");
        assert_eq!(msg(8), "fb_kinetic_class (3, 3): unknown launch status 8");
        let e = launch_status(what, LaunchStatus(5), runtime).unwrap_err();
        assert_eq!(e.message, "WWR_STATUS_5");
        assert_eq!(e.context, [what]);
    }

    #[test]
    fn a_negative_status_never_reaches_the_runtime() {
        for status in [
            FB_LAUNCH_UNSUPPORTED.0,
            FB_LAUNCH_INVALID_ARGUMENT.0,
            -3,
            i32::MIN,
        ] {
            let _ = launch_status("w", LaunchStatus(status), |_, _| unreachable!("{status}"));
        }
        assert_eq!(
            launch_status("w", FB_LAUNCH_SUCCESS, |_, _| unreachable!()),
            Ok(())
        );
    }

    #[test]
    fn every_error_is_located_at_the_caller() {
        for status in [-1, -2, -3, 3, 8] {
            let here = Location::caller();
            let e = launch_status("w", LaunchStatus(status), runtime).unwrap_err();
            assert_eq!(
                (e.origin.file(), e.origin.line()),
                (here.file(), here.line() + 1)
            );
        }
    }
}
