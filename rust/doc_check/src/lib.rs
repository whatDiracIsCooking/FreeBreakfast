//! The documentation checks that need nothing of the tree but its text:
//! `tests/section_citations.rs`, which checks that every section the
//! workspace's documents and comments cite by name is a heading of the file
//! cited. A crate of its own so that it builds and runs without the wwr
//! library. The library target is empty; the checks are its tests.

#![forbid(unsafe_code)]
