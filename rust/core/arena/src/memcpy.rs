//! The copies between a plan's rows and the host: `h` a host slice (a
//! `Vec`'s, say), `p` a pinned row, `d` a device row.
//!
//! | Copy | Who calls it | How |
//! |---|---|---|
//! | [`memcpy_h2p`], [`memcpy_p2h`] | a CPU node | on the host, now |
//! | [`memcpy_p2d`], [`memcpy_d2p`], [`memcpy_d2d`] | a stream node | enqueued on its stream |
//!
//! They are safe because a row is: the enqueued ones are this crate's only
//! `wwr_memcpy_async`, and what makes each sound is what [`Bufs`] already
//! guarantees, so a task needs no `unsafe` to move its data.
//!
//! - **Lengths.** Both sides of a copy have the same length, checked always,
//!   not as a contract: a release build that skipped it would copy past a
//!   row.
//! - **Who else touches the bytes while the copy is in flight.** The rows'
//!   borrows are taken by the copy itself, on the view, and held until it
//!   drops: a pinned source as a reader, a pinned destination as its writer.
//!   So the node that enqueued the copy cannot write the bytes it reads, nor
//!   read or write the bytes it writes: asking panics. The next node to own
//!   them, the row's next reader or the bytes' next owner, is ordered after
//!   this node's event, by an edge or a handoff
//!   ([`Table::reads`](super::Table::reads),
//!   [`Table::handoffs`](super::Table::handoffs)); that is the rule [`Bufs`]
//!   states for a stream node, here enforced rather than trusted.
//!
//! The pinned side of a stream copy is named by its row, not passed as a
//! slice, for the second point: a slice could be any memory, and a borrow
//! that ends when the call returns says nothing about a copy that has not
//! started.

#[cfg(feature = "device")]
use fb_core_error_handling::Status;
#[cfg(feature = "device")]
use fb_core_wwr::gpu_status;
#[cfg(feature = "device")]
use fb_core_wwr::handle::DeviceHandleStream;
#[cfg(feature = "device")]
use fb_core_wwr::runtime::Stream;
#[cfg(feature = "device")]
use wwr_sys::wwr_memcpy_kind;

use super::bufs::Bufs;
#[cfg(feature = "device")]
use super::bufs::{DeviceMut, DeviceRef};
#[cfg(feature = "device")]
use super::kind::Pinned;
use super::kind::{Element, HostKind};
use super::table::RowId;

/// `src` copied into the host row `dst`, which the node writes.
///
/// # Panics
/// If the lengths differ, or as [`Bufs::take_mut`] does.
pub fn memcpy_h2p<K: HostKind, T: Element>(host: &Bufs<'_, K>, dst: RowId<K, T>, src: &[T]) {
    let dst = host.take_mut(dst);
    assert_eq!(dst.len(), src.len(), "memcpy_h2p: the lengths differ");
    dst.copy_from_slice(src);
}

/// The host row `src`, which the node reads, copied into a new `Vec`.
///
/// # Panics
/// As [`Bufs::get`] does.
pub fn memcpy_p2h<K: HostKind, T: Element>(host: &Bufs<'_, K>, src: RowId<K, T>) -> Vec<T> {
    host.get(src).to_vec()
}

/// The pinned row `src`, which the node reads, copied on `stream` into the
/// device row `dst`.
///
/// # Panics
/// If the lengths differ, or as [`Bufs::get`] does.
#[cfg(feature = "device")]
#[track_caller]
pub fn memcpy_p2d<T: Element>(
    dst: &DeviceMut<'_, T>,
    host: &Bufs<'_, Pinned>,
    src: RowId<Pinned, T>,
    stream: &Stream,
) -> Status {
    let (src, len) = host.acquire_ptr(src, false);
    copy(
        dst.as_mut_ptr(),
        dst.len(),
        src,
        len,
        wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
        stream,
    )
}

/// The device row `src` copied on `stream` into the pinned row `dst`, which
/// the node writes. A CPU node reads it after the executor waited for this
/// node's event.
///
/// # Panics
/// If the lengths differ, or as [`Bufs::take_mut`] does.
#[cfg(feature = "device")]
#[track_caller]
pub fn memcpy_d2p<'s, T: Element>(
    host: &Bufs<'_, Pinned>,
    dst: RowId<Pinned, T>,
    src: impl Into<DeviceRef<'s, T>>,
    stream: &Stream,
) -> Status {
    let src = src.into();
    let (dst, len) = host.acquire_ptr(dst, true);
    copy(
        dst,
        len,
        src.as_ptr(),
        src.len(),
        wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        stream,
    )
}

/// The device row `src` copied on `stream` into the device row `dst`.
///
/// # Panics
/// If the lengths differ.
#[cfg(feature = "device")]
#[track_caller]
pub fn memcpy_d2d<'s, T: Element>(
    dst: &DeviceMut<'_, T>,
    src: impl Into<DeviceRef<'s, T>>,
    stream: &Stream,
) -> Status {
    let src = src.into();
    copy(
        dst.as_mut_ptr(),
        dst.len(),
        src.as_ptr(),
        src.len(),
        wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
        stream,
    )
}

/// The one enqueue: `len` elements from `src` to `dst` on `stream`, once
/// the lengths are checked. An empty row enqueues nothing, since its
/// pointer may be the block's end.
#[cfg(feature = "device")]
#[track_caller]
fn copy<T: Element>(
    dst: *mut T,
    dst_len: usize,
    src: *const T,
    len: usize,
    kind: wwr_memcpy_kind,
    stream: &Stream,
) -> Status {
    assert_eq!(dst_len, len, "memcpy: the rows' lengths differ");
    if len == 0 {
        return Ok(());
    }
    // SAFETY: `dst` and `src` are rows of `len` elements of one block each,
    // of the kinds `kind` names, borrowed for the caller's view until it
    // drops (the module's doc has why that covers the copy in flight), and
    // `stream` is a live handle's stream.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.cast(),
            src.cast(),
            len * size_of::<T>(),
            kind,
            stream.stream(),
        )
    })
}
