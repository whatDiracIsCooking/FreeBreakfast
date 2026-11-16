//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"),
//! at a fixed syevd work of 1234 doubles, since the vendor's number needs a
//! device: the nuclei, S and H over one staging of U, X and the step. And
//! one SCF iteration's dense nodes as a loop body, with the occupations
//! and their refusals. And the whole DF-RHF run's plan for water in
//! cc-pVDZ over cc-pVDZ-JKFIT, and the DF-ROHF run's for its cation, with
//! the refusals planning passes on. Planning touches no
//! device, but the records' layout asks the kernel library for the sort's
//! workspace, so the library must be built.

#[allow(
    dead_code,
    reason = "the plans here are never run: `core_orbitals.rs` runs them"
)]
mod common;

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_core_planner::{Dag, Decision, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_molecule::Molecule;
use fb_pipeline_scf::{Run, ScfStep, add_density, add_fock, occupied_orbitals};
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

/// The text of `planned`, each line's trailing space dropped: the table's
/// text ends an empty handoffs line in a space, which the
/// trailing-whitespace hook strips from the committed file.
fn text(planned: &impl std::fmt::Display) -> String {
    planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect()
}

/// Water at `charge` and `multiplicity`, from the overlap's XYZ file.
fn water(charge: i32, multiplicity: i32) -> Molecule {
    fb_io_xyz::read_xyz(&common::xyz("h2o.xyz"), charge, multiplicity).unwrap()
}

/// Water's DF-RHF planned by [`Run`] with `step`.
fn scf_run(step: &ScfStep) -> Result<Run> {
    Run::new(&water(0, 1), "cc-pvdz", "cc-pvdz-jkfit", step, |_| {
        Ok(LWORK)
    })
}

/// The whole DF-RHF run of water in cc-pVDZ over cc-pVDZ-JKFIT, at the
/// default options: the setup, B, the SAD guess, the guess's factor and
/// the rest's K before the loop, the iteration as the loop's body (its rows
/// in the region, D, the factor, the rest's K, DIIS's rings and `last build`
/// carried, F read after it), and the converged F's orbitals and density
/// after it. The plan text, the arena's size among it, is the committed
/// golden.
#[test]
fn scf_plan_golden() {
    let run = scf_run(&ScfStep::default()).expect("water plans");
    assert_eq!(run.n_ao(), 24);
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_scf_h2o_cc-pvdz_cc-pvdz-jkfit.txt",
        &text(run.planned()),
    );
}

/// A basis that is neither carried nor a file, a fitting set that is not
/// carried, and `lwork`'s own error are each the run's error, before the
/// loop is planned; `lwork` is asked at the number of spherical AOs.
#[test]
fn a_refused_input_is_an_error() {
    let step = ScfStep::default();
    let plan = |basis: &str, aux: &str| Run::new(&water(0, 1), basis, aux, &step, |_| Ok(LWORK));
    assert!(plan("no-such-basis", "cc-pvdz-jkfit").is_err());
    let err = plan("cc-pvdz", "no-such-jkfit").err().expect("refused");
    assert_eq!(err.kind, ErrorKind::InvalidConfig);
    let mut asked = None;
    let err = Run::new(&water(0, 1), "sto-3g", "cc-pvdz-jkfit", &step, |n| {
        asked = Some(n);
        Err(Error::gpu("no device"))
    })
    .err()
    .expect("lwork's error is the run's");
    assert_eq!(err.kind, ErrorKind::GpuRuntime);
    assert_eq!(asked, Some(7), "asked at H2O/STO-3G's 7 AOs");
}

/// Each option out of range is refused at planning as `InvalidConfig`,
/// naming it, as the C++ step refuses it: no iteration, a negative or NaN
/// tolerance, and a DIIS history of none or past the limit.
#[test]
fn an_option_out_of_range_is_refused() {
    let default = ScfStep::default();
    for (key, step) in [
        (
            "max_iterations",
            ScfStep {
                max_iterations: 0,
                ..default.clone()
            },
        ),
        (
            "energy_tolerance",
            ScfStep {
                energy_tolerance: -1e-10,
                ..default.clone()
            },
        ),
        (
            "commutator_tolerance",
            ScfStep {
                commutator_tolerance: f64::NAN,
                ..default.clone()
            },
        ),
        (
            "diis_history",
            ScfStep {
                diis_history: 0,
                ..default.clone()
            },
        ),
        (
            "diis_history",
            ScfStep {
                diis_history: 65,
                ..default.clone()
            },
        ),
    ] {
        let err = scf_run(&step).err().expect("refused");
        assert_eq!(err.kind, ErrorKind::InvalidConfig, "{key}");
        assert!(err.message.contains(key), "{key}: {}", err.message);
    }
}

/// GPU: an input, declared and never computed.
struct Input {
    name: &'static str,
    len: usize,
}

impl GpuTask for Input {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

/// A decision that reads the iteration's `E, |e|` row, as the loop's does.
struct Reads(RowId<Pinned, f64>);

impl Decision for Reads {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.0);
    }

    fn decide(
        &self,
        _: &(),
        _: &Bufs<'_, Pinned>,
        _: &Bufs<'_, Heap>,
        _: usize,
    ) -> Result<ControlFlow<()>> {
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
    let (n, (nocc, n_beta)) = (24, occupied_orbitals(&water(0, 1)).unwrap());
    assert_eq!((nocc, n_beta), (5, 5));
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let [h, j, k, s, x, c, d] = ["H", "J", "K", "S", "X", "C", "D"]
        .map(|name| dag.add_gpu(name, 0, Input { name, len: n * n }));
    let c_occ = dag.add_gpu(
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

/// The occupations of water: neutral 10 electrons, a singlet's `(5, 5)`,
/// a triplet's `(6, 4)`; the cation's 9, a doublet's `(5, 4)`. An electron
/// count of the wrong parity for the multiplicity, a multiplicity under 1
/// or past the count plus one, or no electrons, is refused at planning as
/// `InvalidConfig` naming both, as the C++ step refuses them.
#[test]
fn the_occupations_follow_the_multiplicity() {
    for (charge, multiplicity, want) in [(0, 1, (5, 5)), (0, 3, (6, 4)), (1, 2, (5, 4))] {
        let got = occupied_orbitals(&water(charge, multiplicity)).unwrap();
        assert_eq!(got, want, "charge {charge}, multiplicity {multiplicity}");
    }
    for (charge, multiplicity) in [(1, 1), (-1, 1), (0, 2), (0, 0), (0, 13), (10, 1)] {
        let err = occupied_orbitals(&water(charge, multiplicity)).expect_err("refused");
        assert_eq!(
            err.kind,
            ErrorKind::InvalidConfig,
            "charge {charge}, multiplicity {multiplicity}"
        );
    }
    let err = occupied_orbitals(&water(1, 1)).unwrap_err();
    assert!(
        err.message.contains("9 electrons") && err.message.contains("multiplicity 1"),
        "{}",
        err.message
    );
}

/// The whole DF-ROHF run of the water cation, a doublet, in cc-pVDZ over
/// cc-pVDZ-JKFIT: the run above with the guess rescaled to nine electrons,
/// the guess's alpha and beta factors and their rests' K before the loop,
/// with the spin densities and the identity; the body's two K, the spin
/// Fock matrices, the Roothaan matrix, the built densities kept and the
/// open density; and after it the open orbitals canonicalized, PySCF's own
/// Roothaan matrix and its orbitals, and the order test that publishes
/// one. The plan text, the arena's size among it, is the committed golden.
#[test]
fn rohf_plan_golden() {
    let run = Run::new(
        &water(1, 2),
        "cc-pvdz",
        "cc-pvdz-jkfit",
        &ScfStep::default(),
        |_| Ok(LWORK),
    )
    .expect("the cation plans");
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_rohf_h2o_cation_cc-pvdz_cc-pvdz-jkfit.txt",
        &text(run.planned()),
    );
}

/// What planning refuses of an open shell before anything is added: more
/// alpha electrons than orbitals (water's nonet in STO-3G, 9 in 7 AOs),
/// `InvalidConfig`.
#[test]
fn an_open_shell_planning_cannot_hold_is_refused() {
    let step = ScfStep::default();
    let err = Run::new(&water(0, 9), "sto-3g", "cc-pvdz-jkfit", &step, |_| {
        Ok(LWORK)
    })
    .err()
    .expect("nine alpha electrons in seven orbitals");
    assert_eq!(err.kind, ErrorKind::InvalidConfig, "{}", err.message);
    assert!(err.message.contains("do not fit 7"), "{}", err.message);
}

/// An open shell with no beta electron (water at charge +9, one electron
/// in a doublet) plans with no beta factor: three `df exchange` tasks, the
/// two rests' before the loop and the alpha factor's in the body, where
/// one beta electron or more has a fourth, the beta factor's in the body.
#[test]
fn no_beta_electron_plans_without_a_beta_exchange() {
    let step = ScfStep::default();
    let exchanges = |charge: i32| {
        let run = Run::new(&water(charge, 2), "sto-3g", "cc-pvdz-jkfit", &step, |_| {
            Ok(LWORK)
        })
        .expect("the doublet plans");
        text(run.planned())
            .lines()
            .filter(|line| line.contains(" df exchange:"))
            .count()
    };
    assert_eq!(exchanges(9), 3, "one electron");
    assert_eq!(exchanges(1), 4, "nine electrons");
}
