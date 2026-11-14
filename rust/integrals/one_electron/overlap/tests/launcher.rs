//! `fb_integrals_sys::overlap_class`'s own check (tier 4, device 0): a
//! matrix that is not the records' `n_cart²` panics before anything is
//! enqueued, since the kernel writes wherever the records' device-held AO
//! offsets say.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_mmd::add_records;
use fb_integrals_sys::{ClassRows, overlap_class};

/// GPU: every class launched into a matrix of `len` doubles.
struct Short {
    classes: Vec<ClassRows>,
    len: usize,
}

impl Task for Short {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device);
        }
        device.output("short", self.len)
    }

    fn run(
        &self,
        s: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut s = device.take_mut(*s);
        for c in &self.classes {
            overlap_class(stream, device, c, &mut s)?;
        }
        Ok(())
    }
}

/// Needs a GPU. H2O in STO-3G has 7 Cartesian AOs; a matrix of 48 doubles
/// is one short of 7².
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(
    expected = "overlap_class: the matrix holds 48 doubles, but the records' basis has 7² Cartesian AOs"
)]
fn a_matrix_short_of_n_cart_squared_panics() {
    let molecule = fb_io_xyz::read_xyz(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/h2o.xyz")),
        0,
        1,
    )
    .unwrap();
    let basis = fb_io_basis::resolve_basis("sto-3g")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let mut dag = Dag::new();
    let classes = add_records(&mut dag, 0, &basis).unwrap();
    assert!(classes.iter().all(|c| c.n_cart() == 7));
    dag.add("short", 0, Short { classes, len: 48 });
    let planned = dag.plan();
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let _ = instance.run(&mut backend, 1);
}
