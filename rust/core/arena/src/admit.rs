//! Admission's fit check, a table against its worker's cap, and with
//! `device` the reading of a card's free and total memory that a device
//! cap is bounded by and the device's reserve is measured from.

use super::kind::Kind;
use super::table::Table;
#[cfg(feature = "device")]
use fb_core_error_handling::Result;
use fb_core_error_handling::{Error, Status};

/// Whether `table`'s block fits in `cap` bytes on the worker named
/// `worker`.
///
/// The cap is what is left for the block: for a device block, the card's
/// cap less the reserve and the kernels' stack, which the caller works out
/// (`docs/memory.md`, "Admission"). Pure; no device.
///
/// # Errors
/// [`ErrorKind::Allocation`](fb_core_error_handling::ErrorKind::Allocation)
/// when the block is over the cap, naming the worker, the kind, the bytes
/// over; then every row, and every loop's region, larger than the cap on
/// its own, which no order of the plan could fit; then the peak with every
/// row live at it: the C++ refusal, which names the step that reaches the
/// peak. Each row is named with its bytes, `name (bytes B)`.
pub fn admit<K: Kind>(table: &Table<K>, cap: usize, worker: &str) -> Status {
    let total = table.total();
    if total <= cap {
        return Ok(());
    }
    let named = |(name, bytes): (&str, usize)| format!("{name} ({bytes} B)");
    let alone: Vec<String> = table
        .rows
        .iter()
        .map(|r| (r.name.as_str(), r.bytes))
        .chain(table.regions.iter().map(|g| (g.name.as_str(), g.bytes)))
        .filter(|&(_, bytes)| bytes > cap)
        .map(named)
        .collect();
    let alone = if alone.is_empty() {
        String::new()
    } else {
        format!("; over the cap alone: {}", alone.join(", "))
    };
    let peak = table.peak();
    let at = match peak.position {
        Some(p) => format!("node {p}"),
        None => "every node".to_owned(),
    };
    let live: Vec<String> = peak.rows.iter().map(|&row| named(row)).collect();
    Err(Error::allocation(format!(
        "{worker}: the {} block needs {total} bytes, {} over its cap of {cap}{alone}; \
         its peak is {} bytes at {at}: {}",
        K::NAME,
        total - cap,
        peak.bytes,
        live.join(", ")
    )))
}

/// Device `dev_idx`'s free and total memory, in bytes, read at planning:
/// one reading, from which admission takes the card's bound and the
/// device's measured reserve, its total less its free memory
/// (`docs/memory.md`, "Admission"). Every tenant of the card is counted,
/// so the reading moves with them; it is admission's, never a width's at
/// run time.
///
/// # Errors
/// The failed device switch or runtime call.
#[cfg(feature = "device")]
#[track_caller]
#[allow(
    clippy::disallowed_methods,
    reason = "admission bounds a device cap by the card and measures the device's reserve at planning (docs/memory.md, \"Admission\"); nothing is sized from it at run time"
)]
pub fn device_memory(dev_idx: i32) -> Result<(usize, usize)> {
    let location = std::panic::Location::caller();
    let _device = fb_core_wwr::common::ScopedDeviceIndex::try_new(dev_idx)?;
    let (mut free, mut total) = (0, 0);
    // SAFETY: a plain runtime call writing two locals.
    let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
    fb_core_wwr::gpu_status_at(status, location)?;
    Ok((free, total))
}
