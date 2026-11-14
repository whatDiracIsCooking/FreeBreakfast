//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"),
//! at a fixed syevd work of 1234 doubles, since the vendor's number needs a
//! device: the nuclei, S and H over one staging of U, X and the step. And
//! the refusals planning passes on. Planning touches no device, but the
//! records' layout asks the kernel library for the sort's workspace, so
//! the library must be built.

mod common;

use fb_core_error_handling::{Error, ErrorKind};
use fb_pipeline_scf::Run;
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
