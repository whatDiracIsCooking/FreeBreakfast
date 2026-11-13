//! Each fixture run once on a card (tier 4, device 0) against PySCF's S,
//! written by `tests/data/reference.py` with the tree's basis digits, its
//! CODATA 2018 bohr and its unrenormalized contractions (`D S D`), so the
//! comparison measures the integrals and nothing else.
//!
//! The bar is fixed by analysis, one run: `max |S - S_ref| <= 1e-12 * max
//! |S_ref|`, the rounding bound of the atomic sums and the two GEMMs at
//! these sizes (at most 60 spherical and 70 Cartesian AOs) with three
//! decades of margin. Symmetry is not gated: the reference is symmetric.
//!
//! The run's timing record has an entry per task, and every stream task's
//! entry has a stream scope.

mod common;

use std::path::Path;

use fb_core_thread_pool::Placement;

/// The bar, relative to `max |S_ref|`.
const BAR: f64 = 1e-12;

/// `(shape, values)` of the 2-D little-endian float64 array in the .npy file
/// `path`, values column-major: format versions 1 to 3, C or Fortran order.
fn read_npy(path: &Path) -> ([usize; 2], Vec<f64>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        &bytes[..6],
        b"\x93NUMPY",
        "{}: not a .npy file",
        path.display()
    );
    // Version 1 has a 2-byte header length, 2 and 3 a 4-byte one.
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => panic!("{}: .npy version {v}", path.display()),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    assert!(header.contains("'descr': '<f8'"), "{header}: not <f8");
    let fortran = header.contains("'fortran_order': True");
    let dims: Vec<usize> = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .unwrap()
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().unwrap())
        .collect();
    let [rows, cols] = dims[..] else {
        panic!("{header}: not 2-D")
    };
    let flat: Vec<f64> = bytes[start + header_len..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    assert_eq!(flat.len(), rows * cols, "{}: short data", path.display());
    let values = if fortran {
        flat
    } else {
        // Row-major on disk: transposed into column-major.
        let mut col = vec![0.0; flat.len()];
        for i in 0..rows {
            for j in 0..cols {
                col[i + j * rows] = flat[i * cols + j];
            }
        }
        col
    };
    ([rows, cols], values)
}

/// Each fixture planned, run once on device 0 and compared with PySCF's S
/// against the bar; its largest error and its task times are printed, and
/// every miss named before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn overlap_matches_pyscf() {
    let mut misses = Vec::new();
    for (stem, xyz, basis) in common::FIXTURES {
        let run = common::run(xyz, basis);
        let n = run.n_ao();
        let planned = run.planned();
        let stream_tasks: Vec<bool> = (0..planned.len())
            .map(|p| matches!(planned.placement(p), Placement::Stream(_)))
            .collect();
        let (record, s) = run.run(0);
        record.result.unwrap();

        // One entry per task, in rank order; a stream scope on each stream
        // task's, none on a CPU task's.
        assert_eq!(
            record.tasks.len(),
            stream_tasks.len(),
            "{stem}: tasks timed"
        );
        for (t, &on_stream) in record.tasks.iter().zip(&stream_tasks) {
            println!(
                "{stem} {}@{}: host {:?}, device {:?}",
                t.name, t.worker, t.host, t.device
            );
            assert_eq!(
                t.device.is_some(),
                on_stream,
                "{stem} {}: a stream scope iff on a stream",
                t.name
            );
        }

        let (shape, want) = read_npy(&common::data(&format!("{stem}.npy")));
        assert_eq!(shape, [n, n], "{stem}: the AO count");
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = s
            .iter()
            .zip(&want)
            .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{stem}: n_ao {n}, max |S - S_ref| = {worst:e}, bar {:e}",
            BAR * scale
        );
        if worst > BAR * scale {
            misses.push(format!(
                "{stem}: max |S - S_ref| = {worst:e} > {:e}",
                BAR * scale
            ));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}
