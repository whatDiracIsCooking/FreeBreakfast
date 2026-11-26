//! The host Coulomb metric (P|Q) against this crate's: the same matrix built on
//! a card by `fb_integrals_eri_density_fitting_metric` and downloaded, on water
//! and benzene in cc-pVDZ-JKFIT (tier 4, device 0). The water's XYZ file is
//! this crate's fixture, benzene's `fb-pipeline-scf`'s.
//!
//! The bar is fixed by analysis, one run: `max |M - M_gpu| <= 1e-12 max
//! |M_gpu|`, every element of the full matrix against the host's packed lower
//! triangle. Each element is a sum of tens of FP64 terms (the primitive pairs
//! of a contracted pair, then a harmonic sum), which the two paths order
//! differently, so they differ by a few units in the last place of the largest
//! element, well inside the bar.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;

use fb_core_primitives::lower_index;
use fb_integrals_df_metric::metric;
use fb_integrals_eri_density_fitting_metric::add_metric;

/// The bar, relative to `max |M_gpu|`.
const BAR: f64 = 1e-12;

/// Each fixture: a name, its XYZ file relative to `rust/`, and the set.
const FIXTURES: [(&str, &str, &str); 2] = [
    (
        "water",
        "integrals/eri/density_fitting/metric/tests/data/h2o.xyz",
        "cc-pvdz-jkfit",
    ),
    (
        "benzene",
        "pipeline/scf/tests/data/benzene.xyz",
        "cc-pvdz-jkfit",
    ),
];

/// A path relative to `rust/`, the nearest ancestor holding `Cargo.lock`.
fn rust(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|d| d.join("Cargo.lock").exists())
        .expect("a crate under rust/")
        .join(path)
}

/// `got`, a packed lower triangle, against `want`, the full `n × n` matrix
/// column-major: the largest error relative to `max |want|`, which must be
/// within [`BAR`].
fn compare(name: &str, got: &[f64], want: &[f64], n: usize) {
    assert_eq!(got.len(), lower_index(n, 0), "{name}: the host triangle");
    assert_eq!(want.len(), n * n, "{name}: the device matrix");
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for q in 0..n {
        for p in 0..n {
            let host = got[lower_index(p.max(q), p.min(q))];
            worst = worst.max((host - want[p + q * n]).abs());
        }
    }
    println!(
        "{name}: n {n}, max |M_gpu| {scale:.6}, max |M - M_gpu| / max |M_gpu| = {:e}, \
         bar {BAR:e}",
        worst / scale
    );
    assert!(
        worst <= BAR * scale,
        "{name}: max |M - M_gpu| = {worst:e} > {:e}",
        BAR * scale
    );
}

/// `set` placed on the molecule of `xyz`, primitive-normalized as the
/// pipeline places it.
fn setup(xyz: &str, set: &str) -> fb_basis::BasisSet {
    let molecule = fb_io_xyz::read_xyz(&rust(xyz), 0, 1).expect("the XYZ file reads");
    fb_io_basis::resolve_aux_basis(set)
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers the molecule")
}

/// M on the host against M on device 0, each fixture. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn metric_matches_the_device_build() {
    for (name, xyz, set) in FIXTURES {
        let aux = setup(xyz, set);
        let n = aux.nao_sphe() as usize;
        let mut dag = Dag::new();
        let m = add_metric(&mut dag, 0, &aux, &mut None).expect("the library loads");
        compare(name, &metric(&aux), &download(dag, m, n * n), n);
    }
}

/// GPU: the matrix, `len` doubles, copied into a pinned row on the stream.
struct Download {
    m: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.m);
        host.output("matrix", self.len)
    }

    fn run(
        &self,
        pinned: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(host, *pinned, device.get(self.m), stream)
    }
}

/// CPU: the pinned copy read out, after the executor waited for the
/// download.
struct Collect {
    pinned: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.pinned);
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = host.get(self.pinned).to_vec();
        Ok(())
    }
}

/// `m`, `len` doubles on worker 0, downloaded after `dag` runs once on
/// device 0.
fn download(mut dag: Dag<'static>, m: RowId<Device, f64>, len: usize) -> Vec<f64> {
    let pinned = dag.add_gpu("download", 0, Download { m, len });
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            pinned,
            out: Arc::clone(&out),
        },
    );
    let planned = dag.plan();
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    std::mem::take(&mut *out.lock().unwrap())
}
