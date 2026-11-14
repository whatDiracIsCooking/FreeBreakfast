//! Every kernel library's committed `-sys` bindings, and the shared
//! vocabulary's in `fb-kernel-abi`, are what `devtools/rust-bindings.py`
//! generates from its C header now (`docs/gpu_layer.md`, "Build", "From
//! bridge to C ABI"): a header edited without regenerating fails here, with
//! the diff. And a declaration a module header reaches, changed without its
//! `FB_<MODULE>_ABI_VERSION` going up, is refused by the script in both its
//! modes (`docs/gpu_layer.md`, "What the port changes"), shown on a copy of
//! the recipes library's headers. Tier 3: it reads files and runs the
//! script, which needs python3 and nothing else.

use std::path::Path;
use std::process::Command;

#[test]
fn committed_bindings_match_their_headers() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("python3")
        .arg(repo.join("devtools/rust-bindings.py"))
        .arg("--check")
        .output()
        .expect("python3 runs devtools/rust-bindings.py");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}{stderr}");
    // The shared header's home and at least the recipes library, so a glob
    // that found nothing, or a shared header dropped from the script, fails.
    for file in [
        "rust/kernel_abi/src/bindings.rs",
        "rust/recipes/sys/src/bindings.rs",
    ] {
        assert!(stdout.contains(&format!("{file} is current")), "{stdout}");
    }
}

/// The names `bindings` defines at its top level: each `pub const`,
/// `pub type` and `pub struct`.
fn defined(bindings: &str) -> Vec<&str> {
    bindings
        .lines()
        .filter_map(|l| {
            ["pub const ", "pub type ", "pub struct "]
                .iter()
                .find_map(|p| l.strip_prefix(p))
        })
        .map(|rest| rest.split([':', ' ', '=']).next().unwrap())
        .collect()
}

/// `fb_launch.h` is bound once, in `fb-kernel-abi`: no `-sys` crate's
/// bindings define a name of it again, and each re-exports every one, so a
/// shared header the script inlined into a module, or a `-sys` crate that
/// lost one of the names its tests write, fails.
#[test]
fn the_shared_vocabulary_is_bound_once() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let shared = std::fs::read_to_string(rust.join("kernel_abi/src/bindings.rs")).unwrap();
    let names = defined(&shared);
    assert!(names.contains(&"FB_LAUNCH_SUCCESS"), "{names:?}");
    let mut libraries = 0;
    for entry in std::fs::read_dir(&rust).unwrap() {
        let file = entry.unwrap().path().join("sys/src/bindings.rs");
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        libraries += 1;
        let own = defined(&text);
        let reexport = text
            .split_once("pub use fb_kernel_abi::{")
            .and_then(|(_, rest)| rest.split_once("};"))
            .map(|(list, _)| list)
            .unwrap_or_else(|| panic!("{}: no pub use of fb_kernel_abi", file.display()));
        let reexported: Vec<&str> = reexport.split(',').map(str::trim).collect();
        for name in &names {
            assert!(
                !own.contains(name),
                "{} defines {name} again",
                file.display()
            );
            assert!(
                reexported.contains(name),
                "{} does not re-export {name}",
                file.display()
            );
        }
    }
    assert!(libraries >= 2, "found {libraries} -sys bindings");
}

/// A copy of the parts of the tree `devtools/rust-bindings.py` reads for the
/// recipes library, under a fresh directory named for `case`: the device
/// headers, the library's headers, and the two committed bindings files.
fn recipes_copy(case: &str) -> std::path::PathBuf {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let root = std::env::temp_dir().join(format!("fb-abi-bump-{}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut dirs = vec![rust.join("device_headers"), rust.join("recipes/kernels")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "h") {
                let to = root.join("rust").join(path.strip_prefix(&rust).unwrap());
                std::fs::create_dir_all(to.parent().unwrap()).unwrap();
                std::fs::copy(&path, to).unwrap();
            }
        }
    }
    for file in ["recipes/sys/src/bindings.rs", "kernel_abi/src/bindings.rs"] {
        let to = root.join("rust").join(file);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(rust.join(file), to).unwrap();
    }
    root
}

/// `devtools/rust-bindings.py --root root`, checking or writing: its exit
/// code and everything it printed.
fn bindings_script(root: &Path, check: bool) -> (i32, String) {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cmd = Command::new("python3");
    cmd.arg(repo.join("devtools/rust-bindings.py"))
        .arg("--root")
        .arg(root);
    if check {
        cmd.arg("--check");
    }
    let out = cmd
        .output()
        .expect("python3 runs devtools/rust-bindings.py");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

/// Append `line` to the copy's `file`.
fn append(root: &Path, file: &str, line: &str) {
    let path = root.join("rust").join(file);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{text}\n{line}\n")).unwrap();
}

const RECIPES_H: &str = "recipes/kernels/fb_recipes.h";
const RECIPES_BINDINGS: &str = "recipes/sys/src/bindings.rs";
const VERSION: &str = "#define FB_RECIPES_ABI_VERSION ((uint32_t)";

/// The copy's `FB_RECIPES_ABI_VERSION`.
fn version(root: &Path) -> u32 {
    let text = std::fs::read_to_string(root.join("rust").join(RECIPES_H)).unwrap();
    let rest = text
        .split_once(VERSION)
        .expect("the header defines its version")
        .1;
    rest.split(')').next().unwrap().parse().unwrap()
}

/// Bump the copy's `FB_RECIPES_ABI_VERSION` by one.
fn bump(root: &Path) {
    let path = root.join("rust").join(RECIPES_H);
    let v = version(root);
    let text = std::fs::read_to_string(&path).unwrap();
    let from = format!("{VERSION}{v})");
    std::fs::write(
        &path,
        text.replacen(&from, &format!("{VERSION}{})", v + 1), 1),
    )
    .unwrap();
}

/// A declaration the header reaches, added with the version left alone, is
/// refused in both modes and nothing is written (`docs/gpu_layer.md`, "What
/// the port changes"): one in the module's own header, one in a header it
/// inlines, and one in the shared `fb_launch.h`, which reaches every module.
#[test]
fn a_declaration_changed_without_a_bump_is_refused() {
    let changes = [
        ("own", RECIPES_H),
        ("inlined", "recipes/kernels/diis/diis_solve.h"),
        ("shared", "device_headers/fb_launch.h"),
    ];
    for (case, file) in changes {
        let root = recipes_copy(case);
        assert_eq!(
            bindings_script(&root, true).0,
            0,
            "{case}: the copy is current"
        );
        let v = version(&root);
        append(&root, file, "#define FB_ABI_BUMP_TEST ((int32_t)1)");
        let before = std::fs::read_to_string(root.join("rust").join(RECIPES_BINDINGS)).unwrap();
        for check in [true, false] {
            let (code, text) = bindings_script(&root, check);
            assert_eq!(code, 2, "{case}, check {check}: {text}");
            assert!(
                text.contains(&format!(
                    "FB_RECIPES_ABI_VERSION {v}, but it is still {v}: bump it"
                )),
                "{case}, check {check}: {text}"
            );
        }
        let after = std::fs::read_to_string(root.join("rust").join(RECIPES_BINDINGS)).unwrap();
        assert_eq!(before, after, "{case}: the refused write wrote");
        let _ = std::fs::remove_dir_all(root);
    }
}

/// What the bump rule lets through: the same declaration added with the
/// version bumped regenerates, and then checks clean; a comment-only change
/// regenerates with the version left alone.
#[test]
fn a_bump_or_a_comment_only_change_regenerates() {
    let root = recipes_copy("bumped");
    append(
        &root,
        "recipes/kernels/diis/diis_solve.h",
        "#define FB_ABI_BUMP_TEST ((int32_t)1)",
    );
    bump(&root);
    let (code, text) = bindings_script(&root, true);
    assert_eq!(code, 1, "stale, not refused: {text}");
    assert!(!text.contains("bump it"), "{text}");
    assert_eq!(bindings_script(&root, false).0, 0);
    let (code, text) = bindings_script(&root, true);
    assert_eq!(code, 0, "{text}");
    assert!(
        text.contains("rust/recipes/sys/src/bindings.rs is current"),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(root);

    let root = recipes_copy("comment");
    // A doc comment, which the bindings carry and the digest does not.
    let path = root.join("rust").join(RECIPES_H);
    let text = std::fs::read_to_string(&path).unwrap();
    let text = text.replacen(
        VERSION,
        &format!(
            "/// One more line of doc.
{VERSION}"
        ),
        1,
    );
    std::fs::write(&path, text).unwrap();
    let before = std::fs::read_to_string(root.join("rust").join(RECIPES_BINDINGS)).unwrap();
    let (code, text) = bindings_script(&root, false);
    assert_eq!(code, 0, "{text}");
    let after = std::fs::read_to_string(root.join("rust").join(RECIPES_BINDINGS)).unwrap();
    assert!(
        after.contains("/// One more line of doc."),
        "the doc did not reach the bindings"
    );
    assert_ne!(before, after);
    let (code, text) = bindings_script(&root, true);
    assert_eq!(code, 0, "{text}");
    let _ = std::fs::remove_dir_all(root);
}
