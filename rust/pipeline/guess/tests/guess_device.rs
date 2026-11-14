//! The SAD guess on a card (tier 4, device 0), one run per fixture: the
//! overlap crate's three (H2O in STO-3G and cc-pVDZ, CO in cc-pVTZ), each
//! planned with its overlap S in one DAG, the guess uploaded to the device
//! and copied back.
//!
//! Each atomic block holds Z electrons against its atom's overlap by
//! construction, so `tr(D S) = Σ_A Z_A`; the closed-form overlaps the SCF
//! used and the GPU overlap agree to rounding. The bar, fixed by analysis
//! with three decades of margin: `|tr(D S) - Σ_A Z_A| <= 1e-10 Σ_A Z_A`. The
//! device row copied back equals the pinned row bitwise: the copies move
//! bytes.

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule::nuclear_charge;
use fb_pipeline_guess::{add_guess_upload, add_sad_guess};
use wwr_sys::wwr_memcpy_kind;

/// The bar, relative to `Σ_A Z_A`.
const BAR: f64 = 1e-10;

/// The overlap crate's fixtures: their name, XYZ text (Angstrom) and basis.
const FIXTURES: [(&str, &str, &str); 3] = [
    ("h2o_sto-3g", H2O, "sto-3g"),
    ("h2o_cc-pvdz", H2O, "cc-pvdz"),
    ("co_cc-pvtz", CO, "cc-pvtz"),
];

const H2O: &str = "3\nwater, Angstrom\n\
                   O    0.000000    0.000000    0.117300\n\
                   H    0.000000    0.757200   -0.469200\n\
                   H    0.000000   -0.757200   -0.469200\n";
const CO: &str = "2\ncarbon monoxide, the bond off every axis, Angstrom\n\
                  C    0.100000    0.200000    0.300000\n\
                  O    0.751000    0.851000    0.951000\n";

/// GPU: a device row copied into a pinned one on the stream.
struct Download {
    row: RowId<Device, f64>,
    name: &'static str,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.row);
        host.output(self.name, self.len)
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
        let (src, dst) = (device.get(self.row), host.take_mut(*pinned));
        // SAFETY: two rows of the same length, on the task's stream.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size_of_val(dst),
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                stream.stream(),
            )
        })
    }
}

/// CPU: pinned rows read out after the run, in order.
struct Collect {
    rows: Vec<RowId<Pinned, f64>>,
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        for &row in &self.rows {
            host.read(row);
        }
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = self.rows.iter().map(|&r| host.get(r).to_vec()).collect();
        Ok(())
    }
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn trace_of_d_s_is_the_nuclear_charge_and_the_upload_moves_bytes() {
    for (stem, xyz, spec) in FIXTURES {
        let molecule = fb_io_xyz::parse_xyz(xyz, 0, 1).unwrap();
        let basis = fb_io_basis::resolve_basis(spec)
            .unwrap()
            .primitive_normalized()
            .place(&molecule)
            .unwrap();
        let n = basis.nao_sphe() as usize;

        let mut dag = Dag::new();
        let d = add_sad_guess(&mut dag, 0, &basis, molecule.atoms()).unwrap();
        let d_device = add_guess_upload(&mut dag, 0, &basis, d);
        let s = add_overlap(&mut dag, 0, &basis, &mut None).unwrap();
        let d_back = dag.add(
            "download D",
            0,
            Download {
                row: d_device,
                name: "D back",
                len: n * n,
            },
        );
        let s_host = dag.add(
            "download S",
            0,
            Download {
                row: s,
                name: "S",
                len: n * n,
            },
        );
        let out = Arc::new(Mutex::new(Vec::new()));
        dag.add(
            "collect",
            0,
            Collect {
                rows: vec![d, d_back, s_host],
                out: Arc::clone(&out),
            },
        );
        let instance = dag.plan().instantiate(&[0]);
        let backend = thread_pool::Device::new(&[0], instance.plan());
        instance.run(&backend, 2).result.unwrap();
        let out = std::mem::take(&mut *out.lock().unwrap());
        let [d, d_back, s] = &out[..] else {
            panic!("{stem}: collect ran")
        };

        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(bits(d_back), bits(d), "{stem}: the device D copied back");
        let mut trace = 0.0;
        for j in 0..n {
            for i in 0..n {
                trace += d[i + n * j] * s[j + n * i];
            }
        }
        let charge: f64 = molecule
            .atoms()
            .iter()
            .map(|a| f64::from(nuclear_charge(a)))
            .sum();
        assert!(
            (trace - charge).abs() <= BAR * charge,
            "{stem}: tr(D S) = {trace:.15}, Σ Z = {charge}"
        );
    }
}
