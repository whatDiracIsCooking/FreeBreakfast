# The old binary's fixtures

`examples/` holds the old binary's example configs, moved here from
`rust/examples/` when the `free-breakfast` binary moved onto `fb-steps`
(#1872), with the two geometries they read. `tests/plan.rs` plans them
against the goldens in `tests/golden/`; nothing runs them. They leave with
this crate (#1873).
