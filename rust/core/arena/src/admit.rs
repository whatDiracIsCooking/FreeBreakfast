//! Admission's fit check: a table against its worker's cap.

use super::kind::Kind;
use super::table::Table;
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
