//! What the steps' tests share: a config over a molecule of `tests/data`,
//! and the check that the orthogonalizer, gpu0's lock holder, shares no
//! scratch byte with a step the DAG leaves unordered with it.

use std::path::Path;

use fb_core_budget::{Kind, Ledger};
use fb_steps::Config;

/// The config of `xyz` (a file of `tests/data`) in `basis`, with `[eri]`
/// over `aux` when given, on device 0.
pub fn config(xyz: &str, basis: &str, aux: Option<&str>) -> Config {
    let mut text = format!("[molecule]\nxyz = \"{xyz}\"\n[basis]\nname = \"{basis}\"\n");
    if let Some(aux) = aux {
        text += &format!("[eri]\naux_basis = \"{aux}\"\n");
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    Config::parse(&text, &dir).unwrap()
}

/// The steps the orthogonalizer is unordered with: all but the overlap,
/// which it reads, and itself.
const UNORDERED: [&str; 3] = ["core_hamiltonian", "guess", "metric"];

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
