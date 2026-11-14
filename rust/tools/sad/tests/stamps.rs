//! `free-breakfast-sad-stamps` run as a process (tier 3: it links no
//! library): its STO-3G stamps against the committed `data/sad/STO-3G/`,
//! byte for byte, and its refusals, each its kind's pinned exit code and a
//! message naming the cause, with no panic and nothing written.
//!
//! The stamps' bar is exact: `fb-sad` runs the C++ generator's operations
//! in the C++ order, so every element's SCF converges to the same bits
//! (`fb-io-sad`'s `baked_matches_computed_for_hco` is the same bar), and
//! `fb-io-npy` writes the C++ writer's bytes. STO-3G is the set the gate
//! can afford: every one of its 36 elements in a fraction of a second. One
//! run each.
//!
//! Each case's data directory is its own, under cargo's
//! `CARGO_TARGET_TMPDIR`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fb_core_error_handling::ErrorKind;

/// The repository's `data/`.
fn repo_data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data")
}

/// A fresh, empty directory named `name`.
fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `free-breakfast-sad-stamps <args>`, its output printed.
fn stamps(args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_free-breakfast-sad-stamps"))
        .args(args)
        .output()
        .expect("free-breakfast-sad-stamps starts");
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    out
}

/// The `.npy` file names in `dir`, sorted.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// Every file of the committed `data/sad/STO-3G/` is the one in
/// `written`, byte for byte, and `written` holds nothing else.
fn expect_committed_sto3g(written: &Path) {
    let committed = repo_data().join("sad/STO-3G");
    let names = listing(&committed);
    assert_eq!(names.len(), 36, "STO-3G carries H to Kr");
    assert_eq!(listing(written), names, "the same files");
    let differ: Vec<&String> = names
        .iter()
        .filter(|name| {
            std::fs::read(committed.join(name)).unwrap()
                != std::fs::read(written.join(name)).unwrap()
        })
        .collect();
    println!("{} stamps, differing: {differ:?}", names.len());
    assert!(differ.is_empty(), "differing stamps: {differ:?}");
}

/// A refusal: exit code `kind`'s, no panic, stderr naming every one of
/// `named`.
fn expect_refused(out: &Output, kind: ErrorKind, named: &[&str]) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "a panic: {stderr}");
    assert_eq!(out.status.code(), Some(kind.exit_code()), "{kind:?}'s code");
    for n in named {
        assert!(stderr.contains(n), "names '{n}': {stderr}");
    }
}

/// The acceptance run: STO-3G, every element, is the committed directory.
/// Two spellings of the set are one set; the set's stale file goes, and
/// another set's directory is left alone.
#[test]
fn sto3g_is_the_committed_stamps() {
    let data = workdir("sto3g");
    let sad = data.join("sad");
    std::fs::create_dir_all(sad.join("STO-3G")).unwrap();
    std::fs::write(sad.join("STO-3G/119.npy"), b"stale").unwrap();
    std::fs::create_dir_all(sad.join("cc-pVDZ")).unwrap();
    std::fs::write(sad.join("cc-pVDZ/001.npy"), b"kept").unwrap();

    let out = stamps(&[data.to_str().unwrap(), "sto-3g", "STO-3G"]);
    assert_eq!(out.status.code(), Some(0), "exits 0");
    expect_committed_sto3g(&sad.join("STO-3G"));
    assert_eq!(
        std::fs::read(sad.join("cc-pVDZ/001.npy")).unwrap(),
        b"kept",
        "another set is not touched"
    );
}

/// `--all` regenerates every document of `<data>/basis/`, by its BSE name,
/// and removes every other directory of `<data>/sad/`.
#[test]
fn all_regenerates_every_document_and_nothing_else() {
    let data = workdir("all");
    std::fs::create_dir_all(data.join("basis")).unwrap();
    std::fs::copy(
        repo_data().join("basis/sto-3g.json"),
        data.join("basis/sto-3g.json"),
    )
    .unwrap();
    std::fs::create_dir_all(data.join("sad/gone")).unwrap();
    std::fs::write(data.join("sad/gone/001.npy"), b"gone").unwrap();

    let out = stamps(&[data.to_str().unwrap(), "--all"]);
    assert_eq!(out.status.code(), Some(0), "exits 0");
    assert_eq!(listing(&data.join("sad")), ["STO-3G"], "only the set");
    expect_committed_sto3g(&data.join("sad/STO-3G"));
}

/// A name that is neither compiled in nor a file: `resolve_basis`'s `Io`
/// error, before anything is written.
#[test]
fn unknown_basis_is_refused_as_io() {
    let data = workdir("unknown");
    let out = stamps(&[data.to_str().unwrap(), "no-such-basis"]);
    expect_refused(
        &out,
        ErrorKind::Io,
        &["no-such-basis", "not a compiled-in basis"],
    );
    assert!(!data.join("sad").exists(), "nothing written");
}

/// A data directory that cannot be made, under a regular file: `Io`,
/// naming it.
#[test]
fn unwritable_output_is_refused_as_io() {
    let dir = workdir("unwritable");
    std::fs::write(dir.join("file"), b"").unwrap();
    let data = dir.join("file/data");
    let out = stamps(&[data.to_str().unwrap(), "STO-3G"]);
    expect_refused(&out, ErrorKind::Io, &["cannot create", "file/data/sad"]);
}

/// A basis read from a path is not the compiled-in set, even when its
/// document is: `InvalidConfig`.
#[test]
fn path_basis_is_refused_as_not_compiled_in() {
    let data = workdir("path");
    let path = repo_data().join("basis/sto-3g.json");
    let out = stamps(&[data.to_str().unwrap(), path.to_str().unwrap()]);
    expect_refused(
        &out,
        ErrorKind::InvalidConfig,
        &["not compiled into this binary"],
    );
    assert!(!data.join("sad").exists(), "nothing written");
}

/// No set named, or `--all` beside a name: usage errors, `InvalidConfig`,
/// never a quiet `--all`.
#[test]
fn usage_is_refused_as_invalid_config() {
    let data = workdir("usage");
    let data = data.to_str().unwrap();
    expect_refused(&stamps(&[data]), ErrorKind::InvalidConfig, &["usage:"]);
    expect_refused(
        &stamps(&[data, "STO-3G", "--all"]),
        ErrorKind::InvalidConfig,
        &["--all takes no basis names"],
    );
}
