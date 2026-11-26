//! What the crates' tests share: a dev-dependency only.

#![forbid(unsafe_code)]

/// This process's device memory on AMD cards, in bytes, from amdgpu's DRM
/// fdinfo: the `drm-memory-vram` of every open file of this process whose
/// `drm-driver` is `amdgpu`, counted once per `drm-client-id` (a file
/// duplicated into two descriptors is one client). The kernel keeps it per
/// file, so it is this process's alone: no other tenant of the card moves
/// it, as one moves the device's free memory, and it counts every buffer of
/// ours in VRAM, the budget's and a library's alike. `None` when the process
/// has no amdgpu file open: on CUDA, or before the runtime opened one.
pub fn amdgpu_process_vram() -> Option<u64> {
    let dir = std::fs::read_dir("/proc/self/fdinfo").ok()?;
    amdgpu_vram(
        dir.flatten()
            .filter_map(|entry| std::fs::read_to_string(entry.path()).ok()),
    )
}

/// [`amdgpu_process_vram`] over the fdinfo texts `files`.
fn amdgpu_vram(files: impl IntoIterator<Item = String>) -> Option<u64> {
    let mut clients = std::collections::BTreeMap::new();
    for text in files {
        let field = |key: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(':'))
                .map(str::trim)
        };
        if field("drm-driver") != Some("amdgpu") {
            continue;
        }
        let (Some(id), Some(vram)) = (field("drm-client-id"), field("drm-memory-vram")) else {
            continue;
        };
        clients.insert(id.to_owned(), bytes_of(vram));
    }
    (!clients.is_empty()).then(|| clients.values().sum())
}

/// A DRM fdinfo memory value, `<n>` with an optional ` KiB`, ` MiB` or
/// ` GiB` unit, in bytes.
///
/// # Panics
/// If the value is not a number with one of those units.
fn bytes_of(value: &str) -> u64 {
    let (n, unit) = value.split_once(' ').unwrap_or((value, ""));
    let n: u64 = n.parse().expect("a DRM memory value is a count");
    n * match unit {
        "" => 1,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        _ => panic!("a DRM memory unit: {value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// amdgpu's lines, as kernel 6.8 writes them for one render node.
    fn file(id: u32, vram_kib: u64) -> String {
        format!(
            "pos:\t0\nflags:\t02100002\nmnt_id:\t26\nino:\t1033\ndrm-driver:\tamdgpu\n\
             drm-pdev:\t0000:03:00.0\ndrm-client-id:\t{id}\npasid:\t32790\n\
             drm-memory-vram:\t{vram_kib} KiB\ndrm-memory-gtt: \t2048 KiB\n\
             drm-memory-cpu: \t0 KiB\n"
        )
    }

    /// Two clients summed, a client's second descriptor counted once, and
    /// every other file passed over.
    #[test]
    fn sums_each_amdgpu_client_once() {
        let other = "pos:\t0\nflags:\t02\ndrm-driver:\tnvidia\ndrm-client-id:\t3\n".to_owned();
        let files = [
            file(7, 100),
            file(7, 100),
            file(9, 28),
            other,
            "pos:\t0\n".to_owned(),
        ];
        assert_eq!(amdgpu_vram(files), Some(128 << 10));
    }

    /// No amdgpu file: no reading, not zero.
    #[test]
    fn no_amdgpu_file_is_none() {
        assert_eq!(amdgpu_vram(["pos:\t0\n".to_owned()]), None);
    }

    /// Each unit, and a bare count.
    #[test]
    fn units() {
        assert_eq!(bytes_of("5"), 5);
        assert_eq!(bytes_of("5 KiB"), 5 << 10);
        assert_eq!(bytes_of("5 MiB"), 5 << 20);
        assert_eq!(bytes_of("5 GiB"), 5 << 30);
    }
}
