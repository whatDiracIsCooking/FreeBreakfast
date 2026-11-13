//! `src/core/types` (module `free_breakfast.core.types`): the point in space
//! that `atom`, `molecule` and `basis` share.
//!
//! Only what those modules, `io/xyz` and `io/basis` name is ported; the rest
//! waits for the module that first needs it.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `vec.h`: `Vec<double, 3>`, the backend's `double3` | [`Double3`] |
//! | `vec.h`: `vector_type`, `Vec<T, N>` for any other `T` or `N` | not ported yet: nothing above names one |
//! | `vec.h`: `make_vec` | nothing: a struct literal, `Double3 { x, y, z }`, converts nothing implicitly |
//! | `:integral_types`: `int8_t` ... `uint64_t`, `size_t` | nothing: `i8` ... `u64` and `usize` are the language's |
//! | `:concepts`: `real_fp`, `int_type` | not ported yet: nothing above constrains on them |
//!
//! **The fixed-width integers have no counterpart.** The C++ partition
//! exists to give host and device code one spelling of `std::int32_t`;
//! Rust's integer types already have one width each, on every target.
//!
//! **`make_vec` has no counterpart.** It exists because the CUDA builtin is
//! an aggregate and HIP's is a class with a broadcasting constructor, so one
//! brace has to be spelled for both. A Rust struct literal names each field
//! and converts none, which is what `make_vec`'s per-component cast is for.

#![forbid(unsafe_code)]

/// A point or displacement in 3D space, in bohr: `Vec<double, 3>`.
///
/// `#[repr(C)]` with the layout of the backend's `double3`, which CUDA and
/// HIP share: three `f64` at offsets 0, 8 and 16, 24 bytes, aligned to 8. So
/// a host-built position can be copied to a kernel that reads `double3`, as
/// the C++ `Atom::position` is. `Default` is the origin, as the C++ `{}` is.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Double3 {
    /// The first component.
    pub x: f64,
    /// The second component.
    pub y: f64,
    /// The third component.
    pub z: f64,
}

#[cfg(test)]
mod tests {
    use super::Double3;
    use std::mem::{align_of, offset_of, size_of};

    /// `core_vocabulary_host.cpp`'s layout asserts, for the one `Vec` ported:
    /// the order fixes the size, and each component is a `double`.
    #[test]
    fn double3_has_the_backend_layout() {
        assert_eq!(size_of::<Double3>(), 3 * size_of::<f64>());
        assert_eq!(align_of::<Double3>(), align_of::<f64>());
        assert_eq!(offset_of!(Double3, x), 0);
        assert_eq!(offset_of!(Double3, y), 8);
        assert_eq!(offset_of!(Double3, z), 16);
    }

    /// The C++ `Atom` default sits at the origin; so does `Double3`'s.
    #[test]
    fn default_is_the_origin() {
        assert_eq!(
            Double3::default(),
            Double3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
    }
}
