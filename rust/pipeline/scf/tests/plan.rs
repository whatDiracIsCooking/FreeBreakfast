//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"),
//! at a fixed syevd work of 1234 doubles, since the vendor's number needs a
//! device: the nuclei, S and H over one staging of U, X and the step. And
//! the refusals planning passes on. And one SCF iteration's dense nodes as
//! a loop body, with the refusal of an open shell. Planning touches no
//! device, but the records' layout asks the kernel library for the sort's
//! workspace, so the library must be built.

mod common;

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_pipeline_scf::{Run, add_density, add_fock};
use fb_test_support::golden;

/// The work every plan here is laid out with.
const LWORK: usize = 1234;

#[test]
fn plan_goldens() {
    for (stem, xyz, basis) in common::FIXTURES {
        let run = common::run(xyz, basis, |_| Ok(LWORK));
        // The table's text ends an empty handoffs line in a space, which the
        // trailing-whitespace hook strips from the committed file.
        let text: String = run
            .planned()
            .to_string()
            .lines()
            .map(|l| format!("{}\n", l.trim_end()))
            .collect();
        golden(
            env!("CARGO_MANIFEST_DIR"),
            &format!("plan_{stem}.txt"),
            &text,
        );
    }
}

/// A file that is not there, a basis that is neither carried nor a file,
/// and `lwork`'s own error are each the run's error, before anything is
/// planned; `lwork` is asked at the number of spherical AOs.
#[test]
fn a_refused_input_is_an_error() {
    assert!(Run::new(&common::xyz("none.xyz"), "sto-3g", |_| Ok(LWORK)).is_err());
    assert!(Run::new(&common::xyz("h2o.xyz"), "no-such-basis", |_| Ok(LWORK)).is_err());
    let mut asked = None;
    let err = Run::new(&common::xyz("h2o.xyz"), "sto-3g", |n| {
        asked = Some(n);
        Err(Error::gpu("no device"))
    })
    .err()
    .expect("lwork's error is the run's");
    assert_eq!(err.kind, ErrorKind::GpuRuntime);
    assert_eq!(asked, Some(7), "asked at H2O/STO-3G's 7 AOs");
}

/// GPU: an input, declared and never computed.
struct Input {
    name: &'static str,
    len: usize,
}

impl Task for Input {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

/// A decision that reads the iteration's `E, |e|` row, as the loop's does.
struct Reads(RowId<Pinned, f64>);

impl Decision for Reads {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>) {
        host.read(self.0);
    }

    fn decide(&self, _: &(), _: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
        unreachable!("the plan is never run")
    }
}

/// One iteration's dense nodes as a loop body, at water in cc-pVDZ's
/// shape (n = 24, nocc = 5, from the XYZ file's electron count): H, J, K,
/// S, X and C as inputs, the carried D and occupied factor produced before
/// the loop, `fock`, `residual` and `energy and norm`, then `density`
/// writing both carried rows, and a decision reading `E, |e|`. The plan
/// text, the body's rows laid out in the loop's region and the carried
/// ones live across it, is the committed golden.
#[test]
fn iteration_plan_golden() {
    let molecule = fb_io_xyz::read_xyz(&common::xyz("h2o.xyz"), 0, 1).unwrap();
    let (n, nocc) = (24, fb_pipeline_scf::occupied_orbitals(&molecule).unwrap());
    assert_eq!(nocc, 5);
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let [h, j, k, s, x, c, d] = ["H", "J", "K", "S", "X", "C", "D"]
        .map(|name| dag.add(name, 0, Input { name, len: n * n }));
    let c_occ = dag.add(
        "C occ",
        0,
        Input {
            name: "C occ",
            len: n * nocc,
        },
    );
    let body = dag.len();
    let (_, _, scalars) = add_fock(&mut dag, 0, n, 9.0, h, j, k, d, s, x);
    add_density(&mut dag, 0, n, nocc, c, d, c_occ);
    dag.add_loop("converged", 0, body, 50, Reads(scalars));
    let text: String = dag
        .plan()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_iteration_h2o_cc-pvdz.txt",
        &text,
    );
}

/// An odd electron count, or none, is refused at planning as
/// `InvalidConfig`: water at charge +1 (9 electrons), -1 (11) and +10
/// (none); neutral water's 10 electrons are 5 orbitals.
#[test]
fn an_odd_electron_count_is_refused() {
    let water = |charge| fb_io_xyz::read_xyz(&common::xyz("h2o.xyz"), charge, 1).unwrap();
    assert_eq!(fb_pipeline_scf::occupied_orbitals(&water(0)).unwrap(), 5);
    for charge in [1, -1, 10] {
        let err =
            fb_pipeline_scf::occupied_orbitals(&water(charge)).expect_err("not a closed shell");
        assert_eq!(err.kind, ErrorKind::InvalidConfig, "charge {charge}");
    }
    let err = fb_pipeline_scf::occupied_orbitals(&water(1)).unwrap_err();
    assert!(err.message.contains("9 electrons"), "{}", err.message);
}
