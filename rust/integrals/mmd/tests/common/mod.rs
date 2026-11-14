//! The two molecules every test binary plans: H2O/cc-pVDZ and CO/cc-pVTZ,
//! their bases as the BasisSet step makes them (the compiled-in set, its
//! primitives normalized, placed on the atoms).
//!
//! The geometries, in bohr, are the C++ tree's mmd fixtures
//! (`test/gtest/data/coulomb/{h2o_ccpvdz,co_ccpvtz}.json`), copied, and
//! the same literals `tools/mmd_records/main.cpp` dumps the reference
//! records from: no atom on an axis.
//!
//! Beside them, what the device tests share: [`run`], a plan run once with
//! every class of it downloaded ([`HostClass`]).

use std::sync::{Arc, Mutex};

use fb_atom::{Atom, Element};
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_sys::{ClassRows, MixedClassRows};
use fb_molecule::Molecule;

/// A molecule of `atoms` (bohr) in the compiled-in set `basis`.
fn placed(basis: &str, atoms: &[(Element, [f64; 3])]) -> BasisSet {
    let molecule = Molecule::new(
        atoms
            .iter()
            .map(|&(element, [x, y, z])| Atom {
                position: Double3 { x, y, z },
                element,
            })
            .collect(),
        0,
        1,
    );
    let stamp = fb_io_basis::resolve_basis(basis).expect("a compiled-in set");
    stamp
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers the molecule")
}

/// Each case: its file stem, beside the goldens and the reference data, and
/// its basis.
pub fn cases() -> [(&'static str, BasisSet); 2] {
    [
        (
            "h2o_ccpvdz",
            placed(
                "cc-pvdz",
                &[
                    (Element::O, [0.0, 0.0, 0.0]),
                    (Element::H, [1.43, 0.21, -1.07]),
                    (Element::H, [-1.38, 0.35, -1.12]),
                ],
            ),
        ),
        (
            "co_ccpvtz",
            placed(
                "cc-pvtz",
                &[
                    (Element::C, [0.13, -0.21, 0.17]),
                    (Element::O, [0.77, 0.86, 1.86]),
                ],
            ),
        ),
    ]
}

/// The two bases a two-basis plan pairs: water in cc-pVDZ, the first
/// case's, as the bra basis, and water in STO-3G with every atom moved
/// (bohr) as the ket basis, so the two differ in set and in geometry. The
/// same literals `tools/mmd_records/main.cpp` dumps the C++ tree's mixed
/// records from.
#[allow(
    dead_code,
    reason = "the plan and records binaries pair two bases; mixed.rs pairs one with itself"
)]
pub fn two_bases() -> (BasisSet, BasisSet) {
    let [(_, rows), _] = cases();
    let cols = placed(
        "sto-3g",
        &[
            (Element::O, [0.11, -0.07, 0.05]),
            (Element::H, [1.52, 0.18, -0.98]),
            (Element::H, [-1.31, 0.42, -1.19]),
        ],
    );
    (rows, cols)
}

/// One class's records on the host, every field in record order: the
/// double fields in [`ClassRows::f64_rows`]'s order, then the integer ones
/// in [`ClassRows::i32_rows`]'.
#[allow(dead_code, reason = "the plan binary downloads nothing")]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HostClass {
    pub la: i32,
    pub lb: i32,
    pub n_rec: i32,
    pub f64s: Vec<Vec<f64>>,
    pub i32s: Vec<Vec<i32>>,
}

/// One class's device rows, of either basis count.
#[allow(dead_code, reason = "the plan binary downloads nothing")]
#[derive(Clone)]
pub struct Class {
    la: i32,
    lb: i32,
    n_rec: i32,
    f64s: [RowId<Device, f64>; 16],
    i32s: [RowId<Device, i32>; 4],
}

impl From<&ClassRows> for Class {
    fn from(c: &ClassRows) -> Self {
        Class {
            la: c.la(),
            lb: c.lb(),
            n_rec: c.n_rec(),
            f64s: c.f64_rows(),
            i32s: c.i32_rows(),
        }
    }
}

impl From<&MixedClassRows> for Class {
    fn from(c: &MixedClassRows) -> Self {
        Class {
            la: c.la(),
            lb: c.lb(),
            n_rec: c.n_rec(),
            f64s: c.f64_rows(),
            i32s: c.i32_rows(),
        }
    }
}

/// `dag`, whose tasks write `classes`, with `download` (stream: every
/// field into a pinned row) and `collect` (CPU: the pinned rows read out)
/// after it, planned and run once on device 0: each class on the host.
#[allow(dead_code, reason = "the plan binary downloads nothing")]
pub fn run(mut dag: Dag<'static>, classes: Vec<Class>) -> Vec<HostClass> {
    let out = Arc::new(Mutex::new(Vec::new()));
    let down = dag.add_gpu("download", 0, Download(classes.clone()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            classes,
            down,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    std::mem::take(&mut *out.lock().unwrap())
}

/// A class's pinned copies.
#[derive(Clone)]
struct PinnedClass {
    f64s: Vec<RowId<Pinned, f64>>,
    i32s: Vec<RowId<Pinned, i32>>,
}

/// GPU: every field of every class copied down into pinned rows on the
/// stream.
struct Download(Vec<Class>);

impl GpuTask for Download {
    type Rows = Vec<PinnedClass>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        self.0
            .iter()
            .map(|c| {
                let n = c.n_rec as usize;
                PinnedClass {
                    f64s: c
                        .f64s
                        .iter()
                        .map(|&r| d.read(r))
                        .map(|_| h.output("down", n))
                        .collect(),
                    i32s: c
                        .i32s
                        .iter()
                        .map(|&r| d.read(r))
                        .map(|_| h.output("down", n))
                        .collect(),
                }
            })
            .collect()
    }

    fn run(
        &self,
        rows: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        for (c, down) in self.0.iter().zip(rows) {
            for (&src, &dst) in c.f64s.iter().zip(&down.f64s) {
                memcpy_d2p(h, dst, d.get(src), stream)?;
            }
            for (&src, &dst) in c.i32s.iter().zip(&down.i32s) {
                memcpy_d2p(h, dst, d.get(src), stream)?;
            }
        }
        Ok(())
    }
}

/// CPU: the pinned copies read into `out`, after the executor waited for
/// the download.
struct Collect {
    classes: Vec<Class>,
    down: Vec<PinnedClass>,
    out: Arc<Mutex<Vec<HostClass>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>) {
        for c in &self.down {
            c.f64s.iter().for_each(|&r| {
                h.read(r);
            });
            c.i32s.iter().for_each(|&r| {
                h.read(r);
            });
        }
    }

    fn run(&self, _: &(), h: &Bufs<'_, Pinned>) -> Status {
        *self.out.lock().unwrap() = self
            .classes
            .iter()
            .zip(&self.down)
            .map(|(c, down)| HostClass {
                la: c.la,
                lb: c.lb,
                n_rec: c.n_rec,
                f64s: down.f64s.iter().map(|&r| h.get(r).to_vec()).collect(),
                i32s: down.i32s.iter().map(|&r| h.get(r).to_vec()).collect(),
            })
            .collect();
        Ok(())
    }
}
