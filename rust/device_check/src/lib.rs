//! The checks on rust/'s kernel libraries that need neither card nor
//! library, only the tree's text (tier 3): `tests/launch_wrappers.rs`, that
//! no launch wrapper calls the runtime for anything but a launch, and
//! `tests/bindings.rs`, that every `-sys` crate's committed bindings are
//! what `devtools/rust-bindings.py` generates from its C header today; and
//! `tests/kernel_sources.rs`, that no header the bindings read names a
//! launch shape, and that every emitted unit equals its C++ twin. A
//! crate of its own so that it builds and runs without any library; the
//! library target is empty, and the checks are its tests.

#![forbid(unsafe_code)]
