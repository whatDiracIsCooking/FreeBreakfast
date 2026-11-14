//! Admission's fit checks, a table against its worker's cap and a node's
//! host blocks against its available memory; the reading of that memory;
//! and with `device` the reading of a card's free and total memory that a
//! device cap is bounded by and the device's reserve is measured from.

use super::kind::Kind;
use super::table::Table;
use fb_core_error_handling::{Error, Result, Status};

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

/// Whether a node's host blocks fit together in `available` bytes: every
/// GPU worker's pinned block, `pinned`, and the plan's heap block, `heap`,
/// each a table's total (`docs/memory.md`, "Admission"). Neither kind is
/// paged out for the other: pinned is locked, and the heap block's pages
/// are committed by its fill when it is made. Pure; no device.
///
/// # Errors
/// [`ErrorKind::Allocation`](fb_core_error_handling::ErrorKind::Allocation)
/// when the sum is over `available`, naming each kind's bytes, the bytes
/// over and the available memory.
pub fn admit_host(pinned: &[usize], heap: usize, available: usize) -> Status {
    let locked = pinned.iter().fold(0usize, |sum, &b| sum.saturating_add(b));
    let total = locked.saturating_add(heap);
    if total <= available {
        return Ok(());
    }
    let each: Vec<String> = pinned.iter().map(usize::to_string).collect();
    Err(Error::allocation(format!(
        "node: the host blocks need {total} bytes, {} over the {available} bytes available: \
         the pinned blocks {locked} bytes ({} over the GPU workers) and the heap block {heap} \
         bytes",
        total - available,
        each.join(" + ")
    )))
}

/// The host memory the plan's host blocks may take together, in bytes,
/// read at planning ([`admit_host`]): the node's `MemAvailable` from
/// `/proc/meminfo`, or, under a cgroup v2 memory limit, the smaller of it
/// and what the limit leaves, `memory.max` less `memory.current`, the
/// cgroup's inactive page cache (`inactive_file` in `memory.stat`) counted
/// free, as `MemAvailable` counts the node's. A container's `MemAvailable`
/// is the host's; its limit is the cgroup's. The limit read is the
/// process's own cgroup's (`/proc/self/cgroup`), or the root of
/// `/sys/fs/cgroup` when that path is not mounted there, as inside a
/// container without a cgroup namespace; an ancestor's lower limit is not
/// read. A limit of `max`, or none, is no limit.
///
/// # Errors
/// [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io): `/proc/meminfo`
/// cannot be read or holds no `MemAvailable` line it can parse, or a cgroup
/// file present cannot be parsed.
pub fn host_available() -> Result<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error::io(format!("cannot read '/proc/meminfo': {e}")))?;
    let dir = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("0::"))
                .map(|p| format!("/sys/fs/cgroup{}", p.trim_end_matches('/')))
        })
        .filter(|d| std::path::Path::new(&format!("{d}/memory.max")).exists())
        .unwrap_or_else(|| "/sys/fs/cgroup".to_owned());
    let read = |file: &str| std::fs::read_to_string(format!("{dir}/{file}")).ok();
    available(
        &meminfo,
        read("memory.max").as_deref(),
        read("memory.current").as_deref(),
        read("memory.stat").as_deref(),
    )
}

/// [`host_available`] from the files' text: `MemAvailable` from `meminfo`,
/// and the cgroup's `memory.max`, `memory.current` and `memory.stat` where
/// each was read.
pub(super) fn available(
    meminfo: &str,
    max: Option<&str>,
    current: Option<&str>,
    stat: Option<&str>,
) -> Result<usize> {
    let unparsed = |what: &str| Error::io(format!("cannot parse {what}"));
    let line = meminfo
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .ok_or_else(|| Error::io("no MemAvailable in '/proc/meminfo'"))?;
    let node = match line["MemAvailable:".len()..]
        .split_whitespace()
        .collect::<Vec<_>>()[..]
    {
        [kib, "kB"] => kib
            .parse::<usize>()
            .ok()
            .and_then(|kib| kib.checked_mul(1024)),
        _ => None,
    }
    .ok_or_else(|| unparsed(&format!("'{line}' in '/proc/meminfo'")))?;
    let Some(max) = max.map(str::trim).filter(|m| *m != "max") else {
        return Ok(node);
    };
    let bytes = |text: &str, what: &str| {
        text.trim()
            .parse::<usize>()
            .map_err(|_| unparsed(&format!("the cgroup's {what} '{}'", text.trim())))
    };
    let max = bytes(max, "memory.max")?;
    let current = current.map_or(Ok(0), |c| bytes(c, "memory.current"))?;
    let inactive = match stat.and_then(|s| s.lines().find_map(|l| l.strip_prefix("inactive_file ")))
    {
        Some(v) => bytes(v, "memory.stat inactive_file")?,
        None => 0,
    };
    Ok(node.min(max.saturating_sub(current.saturating_sub(inactive))))
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
