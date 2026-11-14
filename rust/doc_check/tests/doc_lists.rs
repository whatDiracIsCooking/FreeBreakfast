//! The docs' lists of what the tree holds are the tree's. Each list below
//! is prose a reader acts on, and each drifted once when the code moved
//! without it (#1525):
//!
//! - **The ABI versions**: `docs/gpu_layer.md`, "What the port changes",
//!   names each kernel library's `FB_<MODULE>_ABI_VERSION` with the value
//!   its committed bindings carry (the bindings are the header's, which
//!   `fb-device-check`'s `bindings.rs` holds).
//! - **The libraries' functions**: `docs/gpu_layer.md`, "Build", names each
//!   `libfb_<module>_kernels.so` and every function its bindings declare
//!   but the two ABI entries, a trailing `*` naming a prefix.
//! - **The slow cases**: `docs/testing.md`, "Slow cases", names every
//!   function that calls `fast_tests()`, and only a function named
//!   `fast_tests` reads `FREE_BREAKFAST_FAST_TESTS`.
//! - **What links the library**: `README.md`, "Build and test", sorts every
//!   member into three lists, by whether its graph reaches a package that
//!   declares `links` (`wwr-sys`, a kernel `-sys` crate) never, only
//!   through default features, or always.
//!
//! Tier 3: it reads text and runs `cargo metadata`, and links nothing.

#![forbid(unsafe_code)]

use fb_doc_check::{code_only, is_ident, rust_root, without_comments, workspace_sources};
use guppy::graph::DependencyDirection;
use guppy::graph::feature::StandardFeatures;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The text of `## heading` in `doc`, up to the next `## `, with every run
/// of whitespace one space, so a phrase matches across a line break.
fn section(doc: &str, heading: &str) -> String {
    let start = doc
        .find(&format!("\n## {heading}\n"))
        .unwrap_or_else(|| panic!("no section \"{heading}\""));
    let body = &doc[start + 1..];
    let body = &body[3..];
    let end = body.find("\n## ").unwrap_or(body.len());
    body[..end].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every backticked span in `text`, in order.
fn backticked(text: &str) -> Vec<&str> {
    text.split('`').skip(1).step_by(2).collect()
}

/// Each kernel library: its module (`recipes`), and the functions and
/// constants its committed bindings declare, read from
/// `<module>/sys/src/bindings.rs`.
struct Library {
    module: String,
    functions: Vec<String>,
    abi_version: Option<String>,
}

/// The kernel libraries, one per member directory holding a
/// `sys/src/bindings.rs` beside a `kernels/`.
fn libraries(root: &Path) -> Vec<Library> {
    let mut out = Vec::new();
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dirs.sort();
    for dir in dirs {
        let bindings = dir.join("sys/src/bindings.rs");
        if !dir.join("kernels").is_dir() || !bindings.is_file() {
            continue;
        }
        let module = dir.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(bindings).unwrap();
        let functions = text
            .split("pub fn ")
            .skip(1)
            .map(|rest| rest.split('(').next().unwrap().to_owned())
            .collect();
        let constant = format!("pub const FB_{}_ABI_VERSION: u32 = ", module.to_uppercase());
        let abi_version = text
            .split_once(&constant)
            .map(|(_, rest)| rest.split(';').next().unwrap().trim().to_owned());
        out.push(Library {
            module,
            functions,
            abi_version,
        });
    }
    out
}

/// What [`libraries`]' entries fail against the two sections of
/// `gpu_layer.md`: a version not stated as the bindings carry it, a library
/// or function "Build" does not name.
fn check_libraries(libs: &[Library], port: &str, build: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let names: Vec<&str> = backticked(build);
    for lib in libs {
        let constant = format!("FB_{}_ABI_VERSION", lib.module.to_uppercase());
        match &lib.abi_version {
            None => failures.push(format!(
                "{}: the bindings define no `{constant}`",
                lib.module
            )),
            Some(v) => {
                let stated = format!("`{constant}` is {v}");
                if !port.contains(&stated) {
                    failures.push(format!(
                        "gpu_layer.md, \"What the port changes\": does not say {stated}, \
                         the value {}/sys/src/bindings.rs carries",
                        lib.module
                    ));
                }
            }
        }
        let so = format!("libfb_{}_kernels.so", lib.module);
        if !names.contains(&so.as_str()) {
            failures.push(format!("gpu_layer.md, \"Build\": does not name `{so}`"));
        }
        let abi = [
            format!("fb_{}_abi_version", lib.module),
            format!("fb_{}_backend", lib.module),
        ];
        for f in lib.functions.iter().filter(|f| !abi.contains(f)) {
            let named = names.iter().any(|n| match n.strip_suffix('*') {
                Some(prefix) => f.starts_with(prefix),
                None => n == f,
            });
            if !named {
                failures.push(format!(
                    "gpu_layer.md, \"Build\": `{so}`'s `{f}` is in its bindings but not named"
                ));
            }
        }
    }
    failures
}

#[test]
fn gpu_layer_states_each_library_version_and_function() {
    let root = rust_root();
    let doc = std::fs::read_to_string(root.join("docs/gpu_layer.md")).unwrap();
    let libs = libraries(&root);
    assert!(
        libs.iter().any(|l| l.module == "recipes") && libs.iter().any(|l| l.module == "integrals"),
        "the scan found the kernel libraries: {:?}",
        libs.iter().map(|l| &l.module).collect::<Vec<_>>()
    );
    let failures = check_libraries(
        &libs,
        &section(&doc, "What the port changes"),
        &section(&doc, "Build"),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_stale_version_or_an_unnamed_function_fails() {
    let lib = Library {
        module: "recipes".to_owned(),
        functions: [
            "fb_merge_sort",
            "fb_merge_sort_workspace_bytes",
            "fb_diis_solve",
            "fb_recipes_backend",
        ]
        .map(str::to_owned)
        .to_vec(),
        abi_version: Some("5".to_owned()),
    };
    let build = "`libfb_recipes_kernels.so` (`fb_merge_sort*` and `fb_diis_solve`)";
    assert!(
        check_libraries(
            std::slice::from_ref(&lib),
            "`FB_RECIPES_ABI_VERSION` is 5",
            build
        )
        .is_empty()
    );
    let failures = check_libraries(
        std::slice::from_ref(&lib),
        "`FB_RECIPES_ABI_VERSION` is 4",
        "`libfb_recipes_kernels.so` (`fb_merge_sort`)",
    );
    assert_eq!(failures.len(), 3, "{failures:#?}");
    assert!(failures[0].contains("is 5"), "{}", failures[0]);
    assert!(
        failures[1].contains("fb_merge_sort_workspace_bytes"),
        "{}",
        failures[1]
    );
    assert!(failures[2].contains("fb_diis_solve"), "{}", failures[2]);
}

/// The name of the function enclosing byte `at` of `code`: the last
/// `fn name` before it.
fn enclosing_fn(code: &str, at: usize) -> Option<&str> {
    let head = &code[..at];
    let mut from = head.len();
    while let Some(i) = head[..from].rfind("fn ") {
        from = i;
        if i > 0 && is_ident(head.as_bytes()[i - 1]) {
            continue;
        }
        let name = &head[i + 3..];
        let end = name
            .bytes()
            .position(|c| !is_ident(c))
            .unwrap_or(name.len());
        if end > 0 {
            return Some(&name[..end]);
        }
    }
    None
}

/// The slow cases of `sources`: each function that calls `fast_tests()`,
/// as `path::name`; and each place outside a function named `fast_tests`
/// that reads the switch.
fn slow_cases(sources: &[(String, String)]) -> (BTreeSet<(String, String)>, Vec<String>) {
    let mut cases = BTreeSet::new();
    let mut stray = Vec::new();
    for (path, text) in sources {
        let code = code_only(text);
        for (at, _) in code.match_indices("fast_tests()") {
            if at > 0 && is_ident(code.as_bytes()[at - 1]) || code[..at].ends_with("fn ") {
                continue;
            }
            if let Some(name) = enclosing_fn(&code, at) {
                cases.insert((path.clone(), name.to_owned()));
            }
        }
        let literal = without_comments(text);
        for (at, _) in literal.match_indices("\"FREE_BREAKFAST_FAST_TESTS\"") {
            if enclosing_fn(&code, at) != Some("fast_tests") {
                let line = text[..at].lines().count();
                stray.push(format!(
                    "{path}:{line}: reads FREE_BREAKFAST_FAST_TESTS outside `fast_tests`"
                ));
            }
        }
    }
    (cases, stray)
}

#[test]
fn testing_names_every_slow_case() {
    let root = rust_root();
    let doc = std::fs::read_to_string(root.join("docs/testing.md")).unwrap();
    let slow = section(&doc, "Slow cases");
    let named: BTreeSet<&str> = backticked(&slow).into_iter().collect();
    let (cases, mut failures) = slow_cases(&workspace_sources());
    assert!(!cases.is_empty(), "the scan found no slow case");
    for (path, name) in &cases {
        if !named.contains(name.as_str()) {
            failures.push(format!(
                "{path}: `{name}` calls `fast_tests()`, and testing.md, \"Slow cases\", does not name it"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_slow_case_is_found_and_a_stray_read_fails() {
    let src = "fn fast_tests() -> bool {\n    std::env::var(\"FREE_BREAKFAST_FAST_TESTS\").is_ok()\n}\n\
               // fast_tests() in a comment\n\
               fn slow() {\n    if fast_tests() {\n        return;\n    }\n}\n\
               fn quick() {}\n\
               fn sneaky() {\n    let _ = std::env::var(\"FREE_BREAKFAST_FAST_TESTS\");\n}\n";
    let (cases, stray) = slow_cases(&[("t.rs".to_owned(), src.to_owned())]);
    assert_eq!(
        cases.into_iter().map(|(_, n)| n).collect::<Vec<_>>(),
        ["slow"]
    );
    assert_eq!(stray.len(), 1, "{stray:#?}");
    assert!(stray[0].starts_with("t.rs:12:"), "{}", stray[0]);
}

/// How a member reaches a package that declares `links`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Links {
    /// Never: it builds and its tests pass with the library missing.
    Never,
    /// Only through its default features.
    ByDefault,
    /// With `--no-default-features` too.
    Always,
}

/// Each member's [`Links`], from the package graph: its own normal, build
/// and dev edges, and its dependencies' normal and build edges, resolved
/// with the features each edge enables.
fn link_classes(root: &Path) -> BTreeMap<String, Links> {
    let graph = guppy::MetadataCommand::new()
        .manifest_path(root.join("Cargo.toml"))
        .other_options(["--locked"])
        .build_graph()
        .expect("cargo metadata over rust/Cargo.toml");
    let reaches = |id, features| {
        let query = graph
            .query_forward([id])
            .expect("a member's id")
            .to_feature_query(features);
        query
            .resolve_with_fn(|_, link| !link.dev_only() || link.from().package_id() == id)
            .to_package_set()
            .packages(DependencyDirection::Forward)
            .any(|p| p.links().is_some())
    };
    graph
        .workspace()
        .iter()
        .map(|p| {
            let class = if reaches(p.id(), StandardFeatures::None) {
                Links::Always
            } else if reaches(p.id(), StandardFeatures::Default) {
                Links::ByDefault
            } else {
                Links::Never
            };
            (p.name().to_owned(), class)
        })
        .collect()
}

/// The members `section` names in the sentence that ends at `marker`.
fn listed(section: &str, marker: &str, members: &BTreeMap<String, Links>) -> BTreeSet<String> {
    let end = section
        .find(marker)
        .unwrap_or_else(|| panic!("README.md, \"Build and test\": no \"{marker}\""));
    let start = section[..end].rfind(". ").map_or(0, |i| i + 2);
    backticked(&section[start..end])
        .into_iter()
        .filter(|n| members.contains_key(*n))
        .map(str::to_owned)
        .collect()
}

/// What the README's three lists fail against `members`: a member in no
/// list or in two, or in the wrong one.
fn check_link_lists(section: &str, members: &BTreeMap<String, Links>) -> Vec<String> {
    let lists = [
        (Links::Never, "do not link the library at all"),
        (Links::ByDefault, "link it through their `device` feature"),
        (Links::Always, "always link"),
    ];
    let mut failures = Vec::new();
    let mut seen: BTreeMap<String, Links> = BTreeMap::new();
    for (class, marker) in lists {
        for name in listed(section, marker, members) {
            if let Some(other) = seen.insert(name.clone(), class) {
                failures.push(format!("`{name}` is in two lists, {other:?} and {class:?}"));
            }
            if members[&name] != class {
                failures.push(format!(
                    "`{name}` is listed as {class:?} (\"{marker}\"), and the graph says {:?}",
                    members[&name]
                ));
            }
        }
    }
    for (name, class) in members {
        if !seen.contains_key(name) {
            failures.push(format!("`{name}` ({class:?}) is in none of the lists"));
        }
    }
    failures
}

#[test]
fn readme_sorts_every_member_by_what_it_links() {
    let root = rust_root();
    let readme = std::fs::read_to_string(root.join("README.md")).unwrap();
    let failures = check_link_lists(&section(&readme, "Build and test"), &link_classes(&root));
    assert!(
        failures.is_empty(),
        "README.md, \"Build and test\":\n{}",
        failures.join("\n")
    );
}

#[test]
fn a_member_missing_or_misfiled_fails() {
    let members: BTreeMap<String, Links> = [
        ("fb-a", Links::Never),
        ("fb-b", Links::ByDefault),
        ("fb-c", Links::Always),
        ("fb-d", Links::Always),
    ]
    .map(|(n, c)| (n.to_owned(), c))
    .into_iter()
    .collect();
    let good = "x. `fb-a` do not link the library at all. `fb-b` link it through their \
                `device` feature. Only `fb-c` and `fb-d` always link.";
    assert!(check_link_lists(good, &members).is_empty());
    let bad = "x. `fb-a` and `fb-c` do not link the library at all. `fb-b` link it through \
               their `device` feature. Only `fb-c` always link.";
    let failures = check_link_lists(bad, &members);
    assert_eq!(failures.len(), 3, "{failures:#?}");
    assert!(
        failures[0].contains("`fb-c` is listed as Never"),
        "{}",
        failures[0]
    );
    assert!(
        failures[1].contains("`fb-c` is in two lists"),
        "{}",
        failures[1]
    );
    assert!(
        failures[2].contains("`fb-d` (Always) is in none"),
        "{}",
        failures[2]
    );
}
