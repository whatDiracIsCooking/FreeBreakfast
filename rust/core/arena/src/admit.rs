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
/// `/proc/meminfo`, or, under cgroup v2 memory limits, the smallest of it
/// and what each limit leaves: `memory.max` less the cgroup's usage,
/// `memory.current` less what it could reclaim, from `memory.stat`. A
/// container's `MemAvailable` is the host's; its limits are its cgroups'.
///
/// What counts as reclaimable is the kernel's own `MemAvailable` estimate
/// (`si_mem_available`, `mm/show_mem.c`) applied to the cgroup's counts
/// (`memory.stat`, `Documentation/admin-guide/cgroup-v2.rst`): the page
/// cache on the file LRU lists, `active_file` plus `inactive_file` (shmem
/// is on the anon lists, and counts used), and the reclaimable slab,
/// `slab_reclaimable`, each less the smaller of half of it and the node's
/// low watermark (the kernel's other reclaimable kernel memory has no
/// `memory.stat` field, and counts used). A cgroup has no watermark of its
/// own, so the half is taken: each term counts half of itself, rounded up,
/// which is what the kernel's estimate counts when the watermark is at
/// least that half and less than it counts otherwise, so a cgroup's figure
/// is never more hopeful than the node's rule (`docs/memory.md`,
/// "Admission").
///
/// Every cgroup from the process's own (`/proc/self/cgroup`) up to the
/// root of `/sys/fs/cgroup` is read, so a lower limit on an ancestor, a
/// systemd slice above a container's cgroup, wins. The walk starts at that
/// root when the process's path is not a directory under it, as inside a
/// container without a cgroup namespace, and stops at the first directory
/// it cannot list, or at the root, above which a cgroup namespace hides
/// the parents: a limit there is not read. A cgroup with no `memory.max`
/// file, or `max` in it, has no limit.
///
/// # Errors
/// [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io): `/proc/meminfo`
/// cannot be read or holds no `MemAvailable` line it can parse, or a cgroup
/// file present cannot be parsed.
pub fn host_available() -> Result<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error::io(format!("cannot read '/proc/meminfo': {e}")))?;
    let own = std::fs::read_to_string("/proc/self/cgroup").ok();
    available_under(
        &meminfo,
        std::path::Path::new("/sys/fs/cgroup"),
        own.as_deref(),
    )
}

/// [`host_available`] over the cgroup v2 tree mounted at `mount`, the
/// process's cgroup given as `/proc/self/cgroup`'s text, `own`: the
/// smallest of [`available`] at each cgroup on the walk.
pub(super) fn available_under(
    meminfo: &str,
    mount: &std::path::Path,
    own: Option<&str>,
) -> Result<usize> {
    use std::path::Component;
    let leaf = own
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("0::")))
        .map(|p| std::path::Path::new(p.trim()))
        .filter(|p| {
            p.components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
        })
        .map(|p| mount.join(p.strip_prefix("/").unwrap_or(p)))
        .filter(|d| d.is_dir())
        .unwrap_or_else(|| mount.to_owned());
    let mut least = available(meminfo, None, None, None)?;
    for dir in leaf.ancestors() {
        if !dir.starts_with(mount) || std::fs::read_dir(dir).is_err() {
            break;
        }
        let read = |file: &str| std::fs::read_to_string(dir.join(file)).ok();
        least = least.min(available(
            meminfo,
            read("memory.max").as_deref(),
            read("memory.current").as_deref(),
            read("memory.stat").as_deref(),
        )?);
    }
    Ok(least)
}

/// The host memory one cgroup leaves, from the files' text: `MemAvailable`
/// from `meminfo`, or, where the cgroup's `memory.max` is a number, the
/// smaller of it and the limit less the cgroup's usage, `memory.current`
/// less what it could reclaim, from `memory.stat`, by
/// [`host_available`]'s rule.
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
    let field = |name: &str| {
        let prefix = format!("{name} ");
        match stat.and_then(|s| s.lines().find_map(|l| l.strip_prefix(prefix.as_str()))) {
            Some(v) => bytes(v, &format!("memory.stat {name}")),
            None => Ok(0),
        }
    };
    let half = |x: usize| x - x / 2;
    let cache = field("active_file")?.saturating_add(field("inactive_file")?);
    let reclaimable = half(cache).saturating_add(half(field("slab_reclaimable")?));
    Ok(node.min(max.saturating_sub(current.saturating_sub(reclaimable))))
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
