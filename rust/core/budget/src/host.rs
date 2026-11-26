//! The host's available memory, read once in setup for the defaulted host
//! cap (`docs/memory.md`, "Caps and margins").
//!
//! Cgroup-aware: the smaller of the machine's `MemAvailable` and, for
//! every cgroup v2 from the process's own up to the mount, the cgroup's
//! limit less its use. Inside a container `MemAvailable` is the host's; the
//! container's limits are its cgroups'.

use std::path::{Component, Path};

use fb_core_error_handling::{Error, Result};

/// The bytes the host can still give this process.
///
/// # Errors
/// If `/proc/meminfo` cannot be read or has no `MemAvailable`.
pub fn host_available() -> Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error::io(format!("cannot read '/proc/meminfo': {e}")))?;
    let own = std::fs::read_to_string("/proc/self/cgroup").ok();
    available_under(&meminfo, Path::new("/sys/fs/cgroup"), own.as_deref())
}

/// [`host_available`] over its three sources: `/proc/meminfo`'s text, the
/// cgroup mount and `/proc/self/cgroup`'s text.
pub(crate) fn available_under(meminfo: &str, mount: &Path, own: Option<&str>) -> Result<u64> {
    let mut least = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kib| kib.trim().parse::<u64>().ok())
        .map(|kib| kib * 1024)
        .ok_or_else(|| Error::io("no MemAvailable in '/proc/meminfo'"))?;
    let leaf = own
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("0::")))
        .map(|p| Path::new(p.trim()))
        .filter(|p| {
            p.components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
        })
        .map(|p| mount.join(p.strip_prefix("/").unwrap_or(p)))
        .filter(|d| d.is_dir())
        .unwrap_or_else(|| mount.to_owned());
    for dir in leaf.ancestors() {
        if !dir.starts_with(mount) {
            break;
        }
        let read = |file: &str| {
            std::fs::read_to_string(dir.join(file))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
        };
        // "max" does not parse: an unlimited cgroup bounds nothing.
        if let (Some(max), Some(current)) = (read("memory.max"), read("memory.current")) {
            least = least.min(max.saturating_sub(current));
        }
    }
    Ok(least)
}
