//! Admission's fit check: a table against its worker's cap.

use super::kind::Kind;
use super::table::Table;
use crate::core::error_handling::{Error, Status};

/// Whether `table`'s block fits in `cap` bytes on the worker named
/// `worker`.
///
/// The cap is what is left for the block: for a device block, the card's
/// cap less the reserve and the kernels' stack, which the caller works out
/// (`docs/memory.md`, "Admission"). Pure; no device.
///
/// # Errors
/// [`ErrorKind::Allocation`](crate::core::error_handling::ErrorKind::Allocation)
/// when the block is over the cap, naming the worker, the kind, the bytes
/// over, and the peak with every row live at it: the C++ refusal, which
/// names the step that reaches the peak.
pub fn admit<K: Kind>(table: &Table<K>, cap: usize, worker: &str) -> Status {
    let total = table.total();
    if total <= cap {
        return Ok(());
    }
    let peak = table.peak();
    let at = match peak.position {
        Some(p) => format!("node {p}"),
        None => "every node".to_owned(),
    };
    let live: Vec<String> = peak
        .rows
        .iter()
        .map(|(name, bytes)| format!("{name} ({bytes} B)"))
        .collect();
    Err(Error::allocation(format!(
        "{worker}: the {} block needs {total} bytes, {} over its cap of {cap}; \
         its peak is {} bytes at {at}: {}",
        K::NAME,
        total - cap,
        peak.bytes,
        live.join(", ")
    )))
}
