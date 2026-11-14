//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance").
//! Planning touches no device, but the records' layout asks the kernel
//! library for the sort's workspace, so the library must be built.

mod common;

use fb_test_support::golden;

#[test]
fn plan_goldens() {
    for (stem, xyz, basis) in common::FIXTURES {
        let fixture = common::plan(xyz, basis);
        // The table's text ends an empty handoffs line in a space, which the
        // trailing-whitespace hook strips from the committed file.
        let text: String = fixture
            .planned
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
