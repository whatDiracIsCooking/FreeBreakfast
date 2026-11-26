//! The clippy bans the rules lean on are configured, so a gate's clippy
//! `-D warnings` fails the code they name:
//!
//! - **Nothing large is allocated on the plain heap in a step's `run`**
//!   (`docs/memory.md`, "The budget"): `steps/clippy.toml`, which clippy
//!   reads for `fb-steps` in place of `rust/clippy.toml`, bans `vec!` and
//!   the presized and growing `Vec` and `String` calls ([`HEAP_GROWERS`]),
//!   and holds every entry of `rust/clippy.toml` besides, so the
//!   workspace's bans reach `fb-steps` too; `disallowed_macros` and
//!   `disallowed_methods` are denied workspace-wide, and `fb-steps` takes
//!   the workspace's lints.
//! - **Device, pinned and budgeted heap memory are allocated only in
//!   `mem_budget`** (`docs/memory.md`, "The budget"), and **a task has no
//!   internal host sync** (`docs/programming_model.md`, "Granularity"):
//!   `rust/clippy.toml` bans the raw `wwr-sys` allocators and the v0.0.4
//!   layer's syncs ([`WORKSPACE_BANS`]), each allowed site an `allow` with
//!   its reason.
//!
//! Tier 3: it reads files and nothing else. That a banned call fails
//! clippy is clippy's; this holds the lists and the lint levels.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use fb_doc_check::rust_root;

/// What `steps/clippy.toml` bans beyond the workspace's: macros, then
/// methods.
const HEAP_GROWERS: (&[&str], &[&str]) = (
    &["std::vec"],
    &[
        "std::vec::Vec::with_capacity",
        "std::vec::Vec::reserve",
        "std::vec::Vec::reserve_exact",
        "std::vec::Vec::resize",
        "std::vec::Vec::extend_from_slice",
        "std::string::String::with_capacity",
    ],
);

/// What `rust/clippy.toml` must ban, beside its older entries.
const WORKSPACE_BANS: &[&str] = &[
    "fb_core_device::Stream::sync",
    "fb_core_device::Pending::sync",
    "fb_core_device::Event::sync",
    "wwr_sys::wwr_malloc",
    "wwr_sys::wwr_malloc_async",
    "wwr_sys::wwr_malloc_from_pool_async",
    "wwr_sys::wwr_malloc_pitch",
    "wwr_sys::wwr_malloc_array",
    "wwr_sys::wwr_host_alloc",
    "wwr_sys::wwr_host_register",
];

fn toml_at(path: &str) -> toml::Table {
    let text =
        std::fs::read_to_string(rust_root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.parse().unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The `path` of every entry of `key` in `config`, each with a reason.
fn banned(config: &toml::Table, key: &str, file: &str) -> BTreeSet<String> {
    let Some(entries) = config.get(key).and_then(toml::Value::as_array) else {
        return BTreeSet::new();
    };
    entries
        .iter()
        .map(|entry| {
            let path = entry.get("path").and_then(toml::Value::as_str);
            let reason = entry.get("reason").and_then(toml::Value::as_str);
            match (path, reason) {
                (Some(path), Some(reason)) if !reason.is_empty() => path.to_owned(),
                _ => panic!("{file}: an entry of {key} without a path and a reason: {entry}"),
            }
        })
        .collect()
}

#[test]
fn the_steps_ban_the_plain_heap_and_keep_every_workspace_ban() {
    let workspace = toml_at("clippy.toml");
    let steps = toml_at("steps/clippy.toml");
    let methods = banned(&steps, "disallowed-methods", "steps/clippy.toml");
    let missing: Vec<_> = banned(&workspace, "disallowed-methods", "clippy.toml")
        .into_iter()
        .chain(HEAP_GROWERS.1.iter().map(|&m| m.to_owned()))
        .filter(|m| !methods.contains(m))
        .collect();
    assert!(
        missing.is_empty(),
        "steps/clippy.toml does not ban {missing:?}: it replaces rust/clippy.toml for fb-steps"
    );
    let macros = banned(&steps, "disallowed-macros", "steps/clippy.toml");
    for &m in HEAP_GROWERS.0 {
        assert!(macros.contains(m), "steps/clippy.toml does not ban {m}!");
    }

    let manifest = toml_at("Cargo.toml");
    for lint in ["disallowed_macros", "disallowed_methods"] {
        let level = manifest
            .get("workspace")
            .and_then(|w| w.get("lints"))
            .and_then(|l| l.get("clippy"))
            .and_then(|c| c.get(lint))
            .and_then(toml::Value::as_str);
        assert_eq!(level, Some("deny"), "the workspace's clippy::{lint}");
    }
    let steps_lints = toml_at("steps/Cargo.toml")
        .get("lints")
        .and_then(|l| l.get("workspace"))
        .and_then(toml::Value::as_bool);
    assert_eq!(
        steps_lints,
        Some(true),
        "fb-steps takes the workspace's lints"
    );
}

#[test]
fn the_workspace_bans_the_raw_allocators_and_the_v004_syncs() {
    let methods = banned(&toml_at("clippy.toml"), "disallowed-methods", "clippy.toml");
    let missing: Vec<_> = WORKSPACE_BANS
        .iter()
        .filter(|m| !methods.contains(**m))
        .collect();
    assert!(
        missing.is_empty(),
        "rust/clippy.toml does not ban {missing:?}"
    );
}
