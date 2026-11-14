//! The guess as tasks of a plan, device-free (tier 3): `Heap` rows and
//! `HostOnly` workers, one run each. The cases of
//! `test/gtest/pipeline_guess_gpu.cpp`, and the task form's of
//! `test/sad/sad_host.cpp` (whose serial halves are `fb-sad`'s), each named
//! after its C++ case; plus the plan's goldens.
//!
//! The bar against the serial `sad_density` is bitwise: every node runs the
//! serial form's own `stamp_density`, and placing a block is a copy. The
//! formaldehyde cases run at one CPU worker and at four, so the stamps are
//! computed concurrently in the second.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, Primitive, Shell};
use fb_core_arena::{Bufs, Device, Heap, Layout, RowId};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{HostOnly, Lane, Record};
use fb_core_types::Double3;
use fb_io_npy::{NpyArray, write_npy};
use fb_io_sad::builtin_sad_density;
use fb_molecule::Molecule;
use fb_pipeline_guess::{add_loaded_guess, add_sad_guess};
use fb_sad::sad_density;
use fb_test_support::golden;

const fn at(x: f64, y: f64, z: f64) -> Double3 {
    Double3 { x, y, z }
}

/// H2CO, C2v in the yz plane, bohr: three elements and a repeated one, the
/// C++ `kFormaldehyde`.
fn formaldehyde() -> Vec<Atom> {
    let atom = |element, position| Atom { position, element };
    vec![
        atom(Element::C, at(0.0, 0.0, 0.0)),
        atom(Element::O, at(0.0, 0.0, 2.2771)),
        atom(Element::H, at(0.0, 1.7817, -1.1111)),
        atom(Element::H, at(0.0, -1.7817, -1.1111)),
    ]
}

/// The compiled-in cc-pVDZ document read by path: the same shells as the
/// set by name, and no name.
fn ccpvdz_path() -> String {
    format!(
        "{}/../../../data/basis/cc-pVDZ.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `spec`, a name or a path, placed over `atoms` as the BasisSet step
/// places it.
fn placed(spec: &str, atoms: &[Atom]) -> BasisSet {
    fb_io_basis::resolve_basis(spec)
        .unwrap()
        .primitive_normalized()
        .place(&Molecule::new(atoms.to_vec(), 0, 1))
        .unwrap()
}

/// CPU: a row read out after the run.
struct Collect {
    row: RowId<Heap, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl Task<Heap, ()> for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Heap>) {
        host.read(self.row);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Device>, host: &Bufs<'_, Heap>, _: Lane<'_, ()>) -> Status {
        *self.out.lock().unwrap() = host.get(self.row).to_vec();
        Ok(())
    }
}

/// `dag` with a `collect` of `d` added, run once on `n_cpu` CPU workers:
/// the record, and D if the run succeeded.
fn run(mut dag: Dag<'static, Heap, ()>, d: RowId<Heap, f64>, n_cpu: usize) -> (Record, Vec<f64>) {
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add(
        "collect",
        0,
        Collect {
            row: d,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let record = instance.run(&mut HostOnly::new(0), n_cpu);
    let d = std::mem::take(&mut *out.lock().unwrap());
    (record, d)
}

/// The SAD guess of `atoms` in `basis` through a plan on `n_cpu` workers.
fn guess(basis: &BasisSet, atoms: &[Atom], n_cpu: usize) -> (Record, Vec<f64>) {
    let mut dag = Dag::new();
    let d = add_sad_guess(&mut dag, 0, basis, atoms).unwrap();
    run(dag, d, n_cpu)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The names of the plan's tasks, in plan order, from its text.
fn task_names(text: &str, n: usize) -> Vec<String> {
    text.lines()
        .skip(2)
        .take(n)
        .map(|l| l.split_whitespace().skip(3).collect::<Vec<_>>().join(" "))
        .map(|l| l.split(':').next().unwrap().to_owned())
        .collect()
}

/// `PipelineGuess.SadMatchesSadDensityBitwise` and
/// `Sad.TaskMatchesSerialBitwiseOverFormaldehyde`: the document read by
/// path, so every stamp is an SCF, against the serial form with no lookup.
#[test]
fn sad_matches_sad_density_bitwise() {
    let atoms = formaldehyde();
    let basis = placed(&ccpvdz_path(), &atoms);
    assert_eq!(basis.name(), "");
    let serial = sad_density(&basis, &atoms, None).unwrap();
    for n_cpu in [1, 4] {
        let (record, d) = guess(&basis, &atoms, n_cpu);
        record.result.unwrap();
        assert_eq!(bits(&d), bits(&serial), "{n_cpu} CPU workers");
    }
}

/// `PipelineGuess.ACompiledInBasisTakesTheBakedStamps`: the set by name
/// arrives named, and the plan's density is the serial form's with the
/// baked stamps. `fb-io-sad`'s tests hold each stamp to its SCF. Planning
/// asks only whether each stamp is carried, and the run's nodes are the
/// three baked stamps and `sad guess`: no SCF runs.
#[test]
fn a_compiled_in_basis_takes_the_baked_stamps() {
    let atoms = formaldehyde();
    let basis = placed("cc-pvdz", &atoms);
    assert_eq!(basis.name(), "cc-pVDZ");
    let baked = sad_density(&basis, &atoms, Some(builtin_sad_density)).unwrap();
    for n_cpu in [1, 4] {
        let (record, d) = guess(&basis, &atoms, n_cpu);
        record.result.unwrap();
        assert_eq!(bits(&d), bits(&baked), "{n_cpu} CPU workers");
        let mut ran: Vec<&str> = record.tasks.iter().map(|t| t.name.as_str()).collect();
        ran.sort_unstable();
        assert_eq!(
            ran,
            [
                "baked stamp C",
                "baked stamp H",
                "baked stamp O",
                "collect",
                "sad guess"
            ],
            "{n_cpu} CPU workers"
        );
    }
}

/// The plan text, its host table among it, for the set by name and the same
/// document by path: one node per distinct stamp (C, O and one for both H),
/// then `sad guess`. By name every stamp is baked and no SCF runs; by path
/// every one is an SCF. `FB_BLESS=1` rewrites the goldens.
#[test]
fn plan_goldens() {
    let atoms = formaldehyde();
    for (stem, spec, kind) in [
        ("by_name", "cc-pvdz".to_owned(), "baked stamp"),
        ("by_path", ccpvdz_path(), "atomic scf"),
    ] {
        let mut dag: Dag<'static, Heap, ()> = Dag::new();
        add_sad_guess(&mut dag, 0, &placed(&spec, &atoms), &atoms).unwrap();
        let planned = dag.plan();
        // The table's text ends an empty handoffs line in a space, which the
        // trailing-whitespace hook strips from the committed file.
        let text: String = planned
            .to_string()
            .lines()
            .map(|l| format!("{}\n", l.trim_end()))
            .collect();
        assert_eq!(
            task_names(&text, planned.len()),
            [
                format!("{kind} C"),
                format!("{kind} O"),
                format!("{kind} H"),
                "sad guess".to_owned()
            ],
            "{stem}"
        );
        golden(
            env!("CARGO_MANIFEST_DIR"),
            &format!("plan_formaldehyde_cc-pvdz_{stem}.txt"),
            &text,
        );
    }
}

/// An s shell of one unit-coefficient primitive.
fn s_shell(exponent: f64, center: Double3) -> Shell {
    Shell::new(
        0,
        vec![Primitive {
            exponent,
            coefficient: 1.0,
        }],
        center,
    )
}

/// `Sad.TaskReportsTheFirstFailingAtomsError`: H seats; O and then N need a
/// p orbital, and two s shells have none. Every stamp runs, the error is the
/// serial form's, the first failing atom's, and `sad guess` is skipped.
#[test]
fn task_reports_the_first_failing_atoms_error() {
    let atoms = [
        Atom {
            position: at(0.0, 0.0, 0.0),
            element: Element::H,
        },
        Atom {
            position: at(0.0, 0.0, 2.0),
            element: Element::O,
        },
        Atom {
            position: at(0.0, 0.0, 4.0),
            element: Element::N,
        },
    ];
    let h = placed("cc-pvdz", &atoms[..1]);
    let mut shells = h.shells().to_vec();
    let mut offsets = vec![0, shells.len() as u32];
    for atom in &atoms[1..] {
        shells.push(s_shell(10.0, atom.position));
        shells.push(s_shell(1.0, atom.position));
        offsets.push(shells.len() as u32);
    }
    let basis = BasisSet::with_atoms(shells, offsets, "");

    let serial = sad_density(&basis, &atoms, None).unwrap_err();
    assert!(serial.message.contains("Z=8"), "{}", serial.message);
    let (record, _) = guess(&basis, &atoms, 4);
    let error = record.result.unwrap_err();
    assert_eq!(error.kind, serial.kind);
    assert_eq!(error.message, serial.message);
    let ran: Vec<&str> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    for stamp in ["atomic scf H", "atomic scf O", "atomic scf N"] {
        assert!(ran.contains(&stamp), "{stamp} did not run: {ran:?}");
    }
    assert!(!ran.contains(&"sad guess"), "{ran:?}");
}

/// `PipelineGuess.ABasisPlacedOverOtherAtomsIsInvalidConfig`, and the task
/// form's refusal before any SCF: at planning, with nothing added.
#[test]
fn a_basis_placed_over_other_atoms_is_invalid_config() {
    let atoms = formaldehyde();
    let basis = placed("cc-pvdz", &atoms[..3]);
    let mut dag: Dag<'static, Heap, ()> = Dag::new();
    let error = add_sad_guess(&mut dag, 0, &basis, &atoms).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(dag.plan().is_empty());
}

/// A directory of the system's temporary one, this process's own.
fn scratch(case: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fb-pipeline-guess-{}-{case}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The loaded guess of `from` over `basis`, through a plan.
fn load(basis: &BasisSet, from: &Path) -> (Record, Vec<f64>) {
    let mut dag = Dag::new();
    let d = add_loaded_guess(&mut dag, 0, basis, from.to_owned()).unwrap();
    run(dag, d, 1)
}

/// `PipelineGuess.ALoadedGuessIsTheSavedDensityBitwise`, and
/// `test/pipeline/guess_host.cpp`'s refusals: the wrong shape is
/// `InvalidConfig` and a missing file `Io`, each naming the file, at
/// planning here with nothing added (the C++ refuses when the step runs);
/// and the node's own check, of a file replaced after planning.
#[test]
fn a_loaded_guess_is_the_saved_density_bitwise() {
    let atoms = formaldehyde();
    let basis = placed("cc-pvdz", &atoms);
    let (record, d) = guess(&basis, &atoms, 1);
    record.result.unwrap();
    let n = basis.nao_sphe() as usize;
    let dir = scratch("load");
    let saved = dir.join("d.npy");
    let square = NpyArray {
        shape: vec![n, n],
        data: d.clone(),
    };
    write_npy(&saved, &square).unwrap();
    let (record, loaded) = load(&basis, &saved);
    record.result.unwrap();
    assert_eq!(bits(&loaded), bits(&d));

    let wrong = dir.join("wrong.npy");
    let narrow = NpyArray {
        shape: vec![n, n - 1],
        data: d[..n * (n - 1)].to_vec(),
    };
    write_npy(&wrong, &narrow).unwrap();
    let missing = dir.join("missing.npy");
    for (path, kind) in [
        (&wrong, ErrorKind::InvalidConfig),
        (&missing, ErrorKind::Io),
    ] {
        let mut dag: Dag<'static, Heap, ()> = Dag::new();
        let error = add_loaded_guess(&mut dag, 0, &basis, path.clone()).unwrap_err();
        assert_eq!(error.kind, kind, "{}", error.message);
        assert!(
            error.message.contains(&path.display().to_string()),
            "{}",
            error.message
        );
        assert!(dag.plan().is_empty());
    }

    // Planned over the square file, which is then replaced: the node checks
    // the file it reads.
    let replaced = dir.join("replaced.npy");
    write_npy(&replaced, &square).unwrap();
    let mut dag = Dag::new();
    let d = add_loaded_guess(&mut dag, 0, &basis, replaced.clone()).unwrap();
    write_npy(&replaced, &narrow).unwrap();
    let error = run(dag, d, 1).0.result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{}", error.message);
    std::fs::remove_dir_all(&dir).unwrap();
}
