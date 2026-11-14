//! The one-electron class launchers over [`ClassRows`]: each adds one
//! class's contributions into an `n_cart × n_cart` matrix, on a stream.
//!
//! A class kernel writes at the AO offsets its records hold, which are
//! device data; [`ClassRows`] is what makes them trustworthy (its doc has
//! the argument). What is left for a launcher to check, in every build, is
//! the matrix: `n_cart²` long, the records' own `n_cart`, and within the
//! kernels' 32-bit index.

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device};
use fb_core_error_handling::Status;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;

use crate::records::ClassRows;
use crate::{
    FB_KINETIC_MAX_MOMENTUM, FB_OVERLAP_MAX_MOMENTUM, FB_POTENTIAL_EN_MAX_MOMENTUM,
    fb_kinetic_class, fb_nuclei_view, fb_overlap_class, fb_potential_en_class, launched,
};

/// Panic unless the matrix a launcher adds into, `len` doubles, is
/// `n_cart²` for the class's `n_cart`, and `n_cart²` fits the kernels'
/// 32-bit index; returns `n_cart` as the C ABI's `n_ao`.
#[track_caller]
fn matrix_extent(routine: &str, class: &ClassRows, len: usize) -> i32 {
    let n = class.n_cart();
    let n2 = n.checked_mul(n);
    assert!(
        n2 == Some(len),
        "{routine}: the matrix holds {len} doubles, but the records' basis has {n}² Cartesian AOs"
    );
    assert!(
        n2.is_some_and(|n2| i32::try_from(n2).is_ok()),
        "{routine}: {n}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    n as i32
}

/// Adds `class`'s overlap contributions into the `n_cart × n_cart` matrix
/// `s` on `stream`: [`fb_overlap_class`], over records only the records
/// task makes and a matrix it checks. Each value is added atomically at
/// `(m, n)` and `(n, m)`, once on a diagonal shell pair, so `s` is zeroed
/// before a build's first class and symmetric after its last.
///
/// `stream` is the stream of the task whose layout read the class
/// ([`ClassRows::read`]), and `device` that task's view.
///
/// # Errors
/// The launcher's refusal or launch error, as
/// [`fb_kernel_abi::launch_status`] maps it, naming the class. A fault
/// inside the kernel surfaces at the next synchronization.
///
/// # Panics
/// If `s` is not the class's `n_cart²` doubles, or `n_cart²` does not fit
/// an `i32`; or a field of the class was not declared by the task's layout
/// ([`Bufs::get`]); before anything is enqueued.
#[debug_requires(
    0 <= class.la() && class.la() <= class.lb() && class.lb() <= FB_OVERLAP_MAX_MOMENTUM,
    "the class is in the overlap kernels' table"
)]
#[track_caller]
pub fn overlap_class(
    stream: &Stream,
    device: &Bufs<'_, Device>,
    class: &ClassRows,
    s: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let n_ao = matrix_extent("overlap_class", class, s.len());
    let view = class.view(device);
    let s = s.as_mut_ptr();
    // SAFETY: the view's arrays are the class's n_rec-long rows, which the
    // records task wrote before this task's stream work (ClassRows's
    // argument), so every offset they hold opens a block inside n_cart;
    // `s` is n_cart² doubles, checked above, with no other user until the
    // enqueued work finishes (DeviceSliceMut's promise); the view is a host
    // struct that outlives the call, which only enqueues.
    let status = unsafe { fb_overlap_class(stream.stream(), &view, n_ao, s) };
    launched("fb_overlap_class", class, status)
}

/// Adds `class`'s kinetic-energy contributions into the `n_cart × n_cart`
/// matrix `t` on `stream`: [`fb_kinetic_class`], as [`overlap_class`] is
/// over the overlap's. Each value is added atomically at `(m, n)` and
/// `(n, m)`, so `t` is zeroed (or holds what T is added to) before a
/// build's first class and symmetric after its last when it was before.
///
/// # Errors
/// As [`overlap_class`].
///
/// # Panics
/// As [`overlap_class`], for `t`.
#[debug_requires(
    0 <= class.la() && class.la() <= class.lb() && class.lb() <= FB_KINETIC_MAX_MOMENTUM,
    "the class is in the kinetic kernels' table"
)]
#[track_caller]
pub fn kinetic_class(
    stream: &Stream,
    device: &Bufs<'_, Device>,
    class: &ClassRows,
    t: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let n_ao = matrix_extent("kinetic_class", class, t.len());
    let view = class.view(device);
    let t = t.as_mut_ptr();
    // SAFETY: as in overlap_class, for `t`.
    let status = unsafe { fb_kinetic_class(stream.stream(), &view, n_ao, t) };
    launched("fb_kinetic_class", class, status)
}

/// Adds `class`'s nuclear-attraction contributions over the nuclei of
/// charges `z` and positions `c` (bohr, x y z per nucleus) into the
/// `n_cart × n_cart` matrix `v` on `stream`: [`fb_potential_en_class`],
/// as [`overlap_class`] is over the overlap's, the [`fb_nuclei_view`]
/// formed from the two slices. Each value is added atomically at `(m, n)`
/// and `(n, m)`, so `v` is zeroed (or holds what V is added to) before a
/// build's first class and symmetric after its last when it was before.
///
/// # Errors
/// As [`overlap_class`]; the launcher refuses no nuclei as
/// `FB_LAUNCH_INVALID_ARGUMENT`.
///
/// # Panics
/// As [`overlap_class`], for `v`; and if `c` is not three doubles per
/// charge, or the nuclei do not fit an `i32` count.
#[debug_requires(
    0 <= class.la() && class.la() <= class.lb() && class.lb() <= FB_POTENTIAL_EN_MAX_MOMENTUM,
    "the class is in the nuclear-attraction kernels' table"
)]
#[track_caller]
pub fn potential_en_class(
    stream: &Stream,
    device: &Bufs<'_, Device>,
    class: &ClassRows,
    z: &impl DeviceSlice<f64>,
    c: &impl DeviceSlice<f64>,
    v: &mut impl DeviceSliceMut<f64>,
) -> Status {
    assert!(
        c.len() == 3 * z.len(),
        "potential_en_class: {} charges, but {} coordinates",
        z.len(),
        c.len()
    );
    let Ok(n_nuc) = i32::try_from(z.len()) else {
        panic!("potential_en_class: {} nuclei, past an i32 count", z.len());
    };
    let n_ao = matrix_extent("potential_en_class", class, v.len());
    let view = class.view(device);
    let nuclei = fb_nuclei_view {
        n_nuc,
        z: z.as_ptr(),
        c: c.as_ptr(),
    };
    let v = v.as_mut_ptr();
    // SAFETY: as in overlap_class, for `v`; and the nuclei's arrays are
    // n_nuc and 3 n_nuc doubles, checked above, unwritten until the
    // enqueued work finishes (DeviceSlice's promise).
    let status = unsafe { fb_potential_en_class(stream.stream(), &view, &nuclei, n_ao, v) };
    launched("fb_potential_en_class", class, status)
}
