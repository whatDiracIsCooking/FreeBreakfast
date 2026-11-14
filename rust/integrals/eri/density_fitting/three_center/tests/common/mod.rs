//! The two fixtures both test binaries run, each with PySCF's `int3c2e`
//! beside it (`tests/data/README.md`): water in cc-pVDZ / cc-pVDZ-JKFIT
//! (orbitals to d, aux to f), and the C++ tree's hand-built `df_high_aux`
//! (one shell per l on two skew centres, orbitals to f, aux to i, so every
//! one of the kernels' 70 classes); and the plan they run: the records
//! ([`fb_integrals_mmd::add_records`]), [`add_three_center`], and two tasks
//! that bring (mn|P) to the host.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_three_center::add_three_center;

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// Water in cc-pVDZ and cc-pVDZ-JKFIT, placed as the DF steps will place
/// them (charge and multiplicity do not enter (mn|P)).
pub fn water() -> (BasisSet, BasisSet) {
    let molecule = fb_io_xyz::read_xyz(&data("h2o.xyz"), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers water");
    let aux = fb_io_basis::resolve_aux_basis("cc-pvdz-jkfit")
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers water");
    (basis, aux)
}

/// `df_high_aux.json`'s two bases, built exactly as the manifest holds
/// them: one primitive per shell, its coefficient already the primitive's
/// own normaliser, so no `primitive_normalized` (the C++
/// `df_high_aux_fixture.h`).
#[allow(dead_code, reason = "the plan binary plans water alone")]
pub fn high_aux() -> (BasisSet, BasisSet) {
    let text = std::fs::read_to_string(data("df_high_aux.json")).expect("the manifest reads");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("the manifest parses");
    let shells = |key: &str| {
        let shells = doc[key].as_array().expect("a shell list");
        BasisSet::new(
            shells
                .iter()
                .map(|s| {
                    let f = |v: &serde_json::Value| v.as_f64().expect("a number");
                    let c: Vec<f64> = s["centre"]
                        .as_array()
                        .expect("a centre")
                        .iter()
                        .map(f)
                        .collect();
                    Shell::zipped(
                        s["l"].as_i64().expect("a momentum") as i32,
                        &[f(&s["exponent"])],
                        &[f(&s["coefficient"])],
                        Double3 {
                            x: c[0],
                            y: c[1],
                            z: c[2],
                        },
                    )
                })
                .collect(),
        )
    };
    (shells("orbital_shells"), shells("aux_shells"))
}

/// A fixture planned.
#[allow(dead_code, reason = "the plan binary reads the plan alone")]
pub struct Fixture {
    pub planned: Planned<'static>,
    pub n_ao: usize,
    pub n_aux: usize,
    pub out: Arc<Mutex<Vec<f64>>>,
}

/// The plan over `basis` and `aux`: the records, (mn|P)'s tasks, then
/// `download` (stream: (mn|P) copied into a pinned row) and `collect`
/// (CPU: the pinned row copied into `out`).
pub fn plan(basis: &BasisSet, aux: &BasisSet) -> Fixture {
    let mut dag = Dag::new();
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, basis).expect("the library loads");
    let b = add_three_center(&mut dag, 0, basis, &classes, aux).expect("the kernels cover both");
    let (n_ao, n_aux) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
    let pinned = dag.add(
        "download",
        0,
        Download {
            b,
            len: n_ao * n_ao * n_aux,
        },
    );
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add(
        "collect",
        0,
        Collect {
            pinned,
            out: Arc::clone(&out),
        },
    );
    Fixture {
        planned: dag.plan(),
        n_ao,
        n_aux,
        out,
    }
}

/// GPU: (mn|P), `len` doubles, copied into a pinned row on the stream.
struct Download {
    b: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.b);
        host.output("(mn|P) host", self.len)
    }

    fn run(
        &self,
        pinned: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(host, *pinned, device.get(self.b), stream)
    }
}

/// CPU: the pinned copy read out, after the executor waited for the
/// download.
struct Collect {
    pinned: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.pinned);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = host.get(self.pinned).to_vec();
        Ok(())
    }
}
