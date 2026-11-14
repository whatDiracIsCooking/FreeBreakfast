//! The documentation checks that need nothing of the tree but its text:
//! `tests/section_citations.rs`, which checks that every section the
//! workspace's documents and comments cite by name is a heading of the file
//! cited, and `tests/unsafe_code.rs`, which checks that every crate root
//! forbids `unsafe_code` unless `docs/programming_model.md` lists its crate
//! as allowed it. A crate of its own so that it builds and runs without the wwr
//! library. The library target is empty; the checks are its tests.

#![forbid(unsafe_code)]
