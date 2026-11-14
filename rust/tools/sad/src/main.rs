//! `tools/sad` (`free-breakfast-sad-stamps`): compute the SAD atomic
//! density of every element of the compiled-in basis sets it is asked for,
//! and write each one as `<data>/sad/<basis>/<Z>.npy`.
//!
//! ```text
//! free-breakfast-sad-stamps <repo>/data (<basis>... | --all)
//! ```
//!
//! The generator behind the baked stamps `fb-io-sad` compiles in, and the
//! C++ tree's `src/io/sad` too: while both trees exist the committed stamps
//! are one set, and either generator reproduces them byte for byte. It runs
//! exactly the computation the guess would run at runtime: the compiled-in
//! basis through `resolve_basis`, its primitives normalized as the BasisSet
//! step normalizes them, the atom's shells alone at the origin, and
//! `fb_sad::sad_density` over them with no lookup. It links `fb-sad` and not
//! `fb-io-sad`, so the stamps it overwrites never feed it.
//!
//! The sets to regenerate are named on the command line, each resolved as
//! `resolve_basis` resolves a name (so `cc-pvdz` and `cc-pVDZ` are one set,
//! computed once), or `--all` for every `<data>/basis/*.json`. A run names
//! its sets because each costs minutes: the one set a change touched is what
//! wants regenerating. Resolving goes through the binary, not the file, so a
//! set this build does not carry is an error ("not compiled in"), not a
//! stamp computed over shells the runtime would never see. Rebuild first.
//!
//! Each regenerated set's `<data>/sad/<basis>/` is removed before its new
//! stamps are written, so an element that has gone leaves nothing behind;
//! the other sets' stamps are not touched. `--all` removes every
//! subdirectory of `<data>/sad/`, so a basis set that has gone leaves
//! nothing behind either. An element whose SCF is refused as too large
//! (`ErrorKind::Unsupported`) is reported and skipped: the runtime refuses
//! it the same way. Any other failure fails the run before anything is
//! removed, and the process exits with its kind's pinned code
//! ([`ErrorKind::exit_code`]).
//!
//! The SCFs run as CPU nodes of one `fb_core_thread_pool::Plan`, one per
//! element, on as many workers as the process may use
//! ([`default_n_cpu`]`(0)`, which follows the CPU affinity: `taskset`
//! bounds it).
//!
//! | C++ (`tools/sad/main.cpp`) | Here |
//! |---|---|
//! | `main`, `run` | `main`, `run`; the error printed as `error: <message>`, the exit code [`ErrorKind::exit_code`]'s |
//! | `Job` | `Job`, its density and time in a `OnceLock` its node sets |
//! | `jobs_for(spec)` | `jobs_for(spec)`, which returns the set's canonical name beside its jobs |
//! | `compute(jobs)`, a taskflow task per element | `compute(jobs)`, a CPU node per element of a `Plan` run on `HostOnly` |
//! | `every_set(basis_dir)`, each file's stem | `every_set(basis_dir)`, each document's BSE `name` field: a stem is not always a name `resolve_basis` folds to (`6-31gstar.json` is `6-31G*`) |

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;
use std::time::Instant;

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_core_thread_pool::{self as thread_pool, HostOnly, Placement, Plan, default_n_cpu};
use fb_core_types::Double3;
use fb_io_basis::resolve_basis;
use fb_io_npy::{NpyArray, write_npy};
use fb_sad::sad_density;

const ORIGIN: Double3 = Double3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

/// One element of one basis set: its shells at the origin, grouped as one
/// atom, and what its SCF made of them with the seconds it took.
struct Job {
    basis: String,
    atom: [Atom; 1],
    shells: BasisSet,
    density: OnceLock<(Result<Vec<f64>>, f64)>,
}

/// The canonical name of the compiled-in set `spec` names, and a job per
/// element of it. The `BasisSet`s are built without a name, so nothing
/// downstream could look a baked stamp up for them even if asked to.
fn jobs_for(spec: &str) -> Result<(String, Vec<Job>)> {
    let stamp = resolve_basis(spec)?;
    if stamp.name().is_empty() {
        return Err(Error::config(format!(
            "{spec} is not compiled into this binary; reconfigure and rebuild first"
        )));
    }
    let mut jobs = Vec::new();
    for (&z, element) in stamp.primitive_normalized().element_shells() {
        let element_z = u8::try_from(z).ok().and_then(Element::from_z);
        let element_z = element_z
            .ok_or_else(|| Error::config(format!("{}: no element has Z={z}", stamp.name())))?;
        let shells: Vec<Shell> = element.iter().map(|shell| shell.place(ORIGIN)).collect();
        let count = shells.len() as u32;
        jobs.push(Job {
            basis: stamp.name().to_string(),
            atom: [Atom {
                position: ORIGIN,
                element: element_z,
            }],
            shells: BasisSet::with_atoms(shells, vec![0, count], ""),
            density: OnceLock::new(),
        });
    }
    Ok((stamp.name().to_string(), jobs))
}

/// Run every job's SCF, a CPU node each: the elements are independent, and
/// the heaviest of them take minutes. Each node keeps its SCF's result in
/// its job and returns `Ok`, so one failure stops nothing else.
fn compute(jobs: &[Job]) -> Status {
    let mut plan: Plan<'_, ()> = Plan::new();
    for job in jobs {
        let name = format!("atomic scf {} Z={}", job.basis, job.atom[0].element.z());
        plan.add(name, Placement::Cpu, &[], move |_| {
            let start = Instant::now();
            let density = sad_density(&job.shells, &job.atom, None);
            let _ = job.density.set((density, start.elapsed().as_secs_f64()));
            Ok(())
        });
    }
    thread_pool::run(&plan, &HostOnly::new(0), default_n_cpu(0)).result
}

/// The BSE `name` of every `<data>/basis/*.json` document, in file-name
/// order: what `--all` regenerates.
fn every_set(basis_dir: &Path) -> Result<Vec<String>> {
    let entries = std::fs::read_dir(basis_dir)
        .map_err(|e| Error::io(format!("cannot list {}: {e}", basis_dir.display())))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| Error::io(format!("cannot list {}: {e}", basis_dir.display())))?
            .path();
        if path.extension().is_some_and(|e| e == "json") {
            files.push(path);
        }
    }
    files.sort();
    files
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
            let document: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| Error::io(format!("{}: {e}", path.display())))?;
            match document.get("name") {
                Some(serde_json::Value::String(name)) => Ok(name.clone()),
                _ => Err(Error::config(format!(
                    "{}: no BSE 'name' string",
                    path.display()
                ))),
            }
        })
        .collect()
}

fn run(args: &[String]) -> Status {
    if args.len() < 3 {
        let program = args
            .first()
            .map_or("free-breakfast-sad-stamps", String::as_str);
        return Err(Error::config(format!(
            "usage: {program} <repo>/data (<basis>... | --all)"
        )));
    }
    let data = PathBuf::from(&args[1]);
    let sad_dir = data.join("sad");

    let all = args.len() == 3 && args[2] == "--all";
    let specs = if all {
        every_set(&data.join("basis"))?
    } else {
        if args[2..].iter().any(|arg| arg == "--all") {
            return Err(Error::config("--all takes no basis names beside it"));
        }
        args[2..].to_vec()
    };

    // The canonical names of the sets being regenerated: two spellings of
    // one set are one set, computed once.
    let mut names: Vec<String> = Vec::new();
    let mut jobs: Vec<Job> = Vec::new();
    for spec in &specs {
        let (name, more) = jobs_for(spec)?;
        if !names.contains(&name) {
            names.push(name);
            jobs.extend(more);
        }
    }
    compute(&jobs)?;
    let results: Vec<&(Result<Vec<f64>>, f64)> = jobs
        .iter()
        .map(|job| job.density.get().expect("every node ran"))
        .collect();

    // Every failure but a too-large atom is checked before anything is
    // removed, so a failed run leaves the old stamps in place.
    for (density, _) in &results {
        if let Err(error) = density
            && error.kind != ErrorKind::Unsupported
        {
            return Err(error.clone());
        }
    }

    std::fs::create_dir_all(&sad_dir)
        .map_err(|e| Error::io(format!("cannot create {}: {e}", sad_dir.display())))?;
    let listing = std::fs::read_dir(&sad_dir)
        .map_err(|e| Error::io(format!("cannot list {}: {e}", sad_dir.display())))?;
    for entry in listing {
        let path = entry
            .map_err(|e| Error::io(format!("cannot list {}: {e}", sad_dir.display())))?
            .path();
        let regenerated = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| names.iter().any(|name| name == n));
        if path.is_dir() && (all || regenerated) {
            std::fs::remove_dir_all(&path)
                .map_err(|e| Error::io(format!("cannot remove {}: {e}", path.display())))?;
        }
    }

    for (job, (density, seconds)) in jobs.iter().zip(results) {
        let z = job.atom[0].element.z();
        let density = match density {
            Ok(density) => density,
            Err(error) => {
                println!("{:>8} Z={z:<3} skipped: {}", job.basis, error.message);
                continue;
            }
        };
        let dir = sad_dir.join(&job.basis);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::io(format!("cannot create {}: {e}", dir.display())))?;
        let n = job.shells.nao_sphe() as usize;
        let array = NpyArray {
            shape: vec![n, n],
            data: density.clone(),
        };
        write_npy(&dir.join(format!("{z:03}.npy")), &array)?;
        println!("{:>8} Z={z:<3} {n:>3} AOs  {seconds:8.3} s", job.basis);
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            // The codes are 1 to 8.
            ExitCode::from(error.kind.exit_code() as u8)
        }
    }
}
