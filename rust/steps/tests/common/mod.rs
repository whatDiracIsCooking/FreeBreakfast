//! What the steps' tests share: a config over a molecule of `tests/data`,
//! the chain's config, a file of `tests/data`, the energy bar, the bohr,
//! the gates' fast switch, a buffer brought back to the host, a packed
//! triangle unpacked, the check that the orthogonalizer, gpu0's lock
//! holder, shares no scratch byte with a step the DAG leaves unordered with
//! it, and this process's VRAM on an AMD card; and the allocation record,
//! installed in every test binary of the steps, so the wrapper fails a step
//! whose `run` makes one allocation past its threshold on the plain heap (`docs/memory.md`, "The budget").

#![allow(
    dead_code,
    reason = "each test binary uses its own part of what the tests share"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's host buffers, outside any step's run"
)]

use std::path::{Path, PathBuf};

use fb_core_budget::{Backing, Caps, DeviceBuf, Kind, Ledger};
use fb_core_device::{DeviceSlice, Stream};
use fb_steps::{Config, ScfTable};

/// The allocation record: every `run` the wrapper calls in this binary is
/// watched.
#[global_allocator]
static RECORD: fb_core_budget::AllocationRecord = fb_core_budget::AllocationRecord;

/// The bar on an energy, hartree, set by the loop's own convergence:
/// `[scf] energy_tolerance` is 1e-10 between two builds and the commutator
/// 1e-7, whose square bounds a converged energy's distance from the
/// stationary point, so two runs that converge along different paths agree
/// within a few 1e-11 (`chain_device.rs`).
pub const ENERGY_BAR: f64 = 1e-9;

/// Angstrom per bohr, CODATA 2018: the XYZ reader's.
pub const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

/// `tests/data/<file>`.
pub fn data(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The config of `xyz` (a file of `tests/data`) in `basis`, with `[eri]`
/// over `aux` when given, on device 0.
pub fn config(xyz: &str, basis: &str, aux: Option<&str>) -> Config {
    let mut text = format!("[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"{basis}\"\n");
    if let Some(aux) = aux {
        text += &format!("[eri]\naux_basis = \"{aux}\"\n");
    }
    Config::parse(&text, &data("")).unwrap()
}

/// The chain's config of `xyz`: cc-pVDZ, `[eri]` over cc-pVDZ-JKFIT, and
/// `[scf]` at its defaults.
pub fn chain(xyz: &str) -> Config {
    let mut config = config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    config
}

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0` (`docs/testing.md`, "Slow cases").
pub fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// `buf`, on `device`, brought back on a stream of the test's own into a
/// pinned buffer of a ledger of its own.
pub fn download(device: i32, buf: &DeviceBuf<f64>) -> Vec<f64> {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let host = ledger.budget("test").pinned::<f64>(0, buf.len());
    let host = Stream::new(device)
        .download(host, buf)
        .unwrap()
        .sync()
        .unwrap();
    host.to_vec()
}

/// The full symmetric matrix of the packed lower triangle `packed`.
pub fn unpack(packed: &[f64], n: usize) -> Vec<f64> {
    let mut full = vec![0.0; n * n];
    let mut at = 0;
    for p in 0..n {
        for q in 0..=p {
            full[p * n + q] = packed[at];
            full[q * n + p] = packed[at];
            at += 1;
        }
    }
    full
}

/// The steps the orthogonalizer is unordered with: all but the overlap,
/// which it reads, and itself.
const UNORDERED: [&str; 4] = ["core_hamiltonian", "guess", "metric", "eri"];

/// On each of gpu0's slabs the orthogonalizer's region meets no region of
/// a step unordered with it; returns how many such pairs were compared.
pub fn orthogonalizer_scratch_is_its_own(ledger: &Ledger) -> usize {
    let regions = ledger.regions();
    let mut compared = 0;
    for kind in [Kind::Device, Kind::Pinned] {
        let on = |step: &str| {
            regions
                .iter()
                .find(|r| r.gpu == 0 && r.kind == kind && r.step == step)
        };
        let Some(x) = on("orthogonalizer") else {
            continue;
        };
        for other in UNORDERED.iter().filter_map(|s| on(s)) {
            let apart =
                x.offset + x.bytes <= other.offset || other.offset + other.bytes <= x.offset;
            assert!(apart, "{kind:?}: {x:?} meets {other:?}");
            compared += 1;
        }
    }
    compared
}

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
pub fn amdgpu_vram(files: impl IntoIterator<Item = String>) -> Option<u64> {
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
pub fn bytes_of(value: &str) -> u64 {
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
