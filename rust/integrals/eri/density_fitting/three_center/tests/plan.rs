//! Water's packed build over two batches (tier 3): the plan text, each
//! worker's device and pinned table among it with the arena's size, is a
//! committed golden, `FB_BLESS=1` rewriting it (`docs/testing.md`,
//! "Performance"); admission of a one-batch plan's device table; U's
//! slots; and the refusals planning makes of a basis, and of records
//! handed in with other tables.
//! Planning touches no device, but the load check reads the kernel library,
//! so it must be built.

mod common;

use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::add_packed_three_center;
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::{AuxTable, add_packed_three_center_tasks};
use fb_test_support::golden;

/// Water's packed build over two batches of its kept pairs, the second
/// starting mid-class, planned after the records: per batch, `packed
/// three-center tables`, `packed three-center` and `spherical transform`,
/// with one `harmonics` and `upload U` per basis for both; each batch's
/// output `naux` × its spherical columns. The plan text is the committed
/// golden `plan_h2o_packed_two_batches.txt`.
#[test]
fn packed_plan_golden() {
    let (basis, aux) = common::water();
    let mut dag = Dag::new();
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let map = PairMap::new(&record_tables(&basis), None, 0.0);
    let (n, naux) = (map.pairs().len(), aux.nao_sphe() as usize);
    let split = n / 2;
    let (mut u, mut u_aux) = (None, None);
    let mut outs = Vec::new();
    for pairs in [0..split, split..n] {
        let blocks = map.sphe().blocks();
        let columns = blocks[pairs.end - 1].end() - blocks[pairs.start].column;
        let out = add_packed_three_center(
            &mut dag,
            0,
            &basis,
            &classes,
            &aux,
            &map,
            pairs.clone(),
            pairs.len(),
            &mut u,
            &mut u_aux,
        )
        .unwrap();
        outs.push((out, columns));
    }
    let planned = dag.plan();
    for (out, columns) in outs {
        assert_eq!(planned.device_table(0).len(out), naux * columns);
    }
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_h2o_packed_two_batches.txt",
        &text,
    );
}

/// Both of a batch's tensors are rows of the plan's device table at
/// planning: the Cartesian batch (`n_col_cart · naux_cart` doubles) and
/// its spherical transform (`naux · n_col`), each named `(mn|P) packed`,
/// so a budget one byte short of the two is refused by admission as
/// `Allocation`, naming each with its bytes, and the table's own total is
/// admitted.
#[test]
fn admission_refuses_a_budget_short_of_the_two_tensors() {
    let (basis, aux) = common::water();
    let fixture = common::plan(&basis, &aux);
    let table = fixture.planned.device_table(0);
    let cartesian = 8 * fixture.map.cart().n_columns() * aux.nao_cart() as usize;
    let spherical = 8 * fixture.map.sphe().n_columns() * fixture.n_aux;
    let err = fb_core_arena::admit(table, cartesian + spherical - 1, "gpu0").unwrap_err();
    assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
    for named in [
        format!("(mn|P) packed ({cartesian} B)"),
        format!("(mn|P) packed ({spherical} B)"),
    ] {
        assert!(err.message.contains(&named), "{named} not in: {err}");
    }
    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
}

/// An orbital shell past f and an aux shell past i are refused at planning
/// as `Unsupported`, naming the shell, before anything is added: the
/// launcher would refuse their classes only at run time.
#[test]
fn a_shell_past_the_kernels_is_unsupported_at_planning() {
    let shell = |l| Shell::zipped(l, &[1.0], &[1.0], Default::default());
    let fine = BasisSet::new(vec![shell(0), shell(1)]);
    for (basis, aux, named) in [
        (
            BasisSet::new(vec![shell(0), shell(4)]),
            fine.clone(),
            "orbital shell 1 has momentum 4",
        ),
        (
            fine.clone(),
            BasisSet::new(vec![shell(7)]),
            "aux shell 0 has momentum 7",
        ),
    ] {
        let mut dag = Dag::new();
        let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
        let map = PairMap::new(&record_tables(&basis), None, 0.0);
        let all = 0..map.pairs().len();
        let err = add_packed_three_center(
            &mut dag,
            0,
            &basis,
            &classes,
            &aux,
            &map,
            all.clone(),
            all.len(),
            &mut None,
            &mut None,
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
        assert!(err.message.contains(named), "{err}");
        let mut records = Dag::new();
        fb_integrals_mmd::add_records(&mut records, 0, &basis).unwrap();
        assert_eq!(
            dag.plan().to_string(),
            records.plan().to_string(),
            "nothing added"
        );
    }
}

/// U and U_aux come from the caller's slots (`fb_recipes_matrix_ao::add_harmonics`):
/// empty ones get one `harmonics` task and one `upload U` copy node each and
/// are left holding the upload's device row, and a second build over the
/// same slots adds none and runs after both uploads, reading their rows, so
/// neither transform is staged or copied up twice.
#[test]
fn u_and_u_aux_are_read_from_the_slots() {
    let (basis, aux) = common::water();
    let mut dag = Dag::new();
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let map = PairMap::new(&record_tables(&basis), None, 0.0);
    let (mut u, mut u_aux) = (None, None);
    let mut build = |u: &mut Option<_>, u_aux: &mut Option<_>| {
        let all = 0..map.pairs().len();
        add_packed_three_center(
            &mut dag,
            0,
            &basis,
            &classes,
            &aux,
            &map,
            all.clone(),
            all.len(),
            u,
            u_aux,
        )
        .unwrap();
    };
    build(&mut u, &mut u_aux);
    assert!(u.is_some() && u_aux.is_some(), "both slots are filled");
    build(&mut u, &mut u_aux);
    let planned = dag.plan();
    let names: Vec<String> = planned
        .to_string()
        .lines()
        .skip(2)
        .take(planned.len())
        .map(|l| l.split_whitespace().skip(3).collect::<Vec<_>>().join(" "))
        .map(|l| l.split(':').next().unwrap().to_owned())
        .collect();
    let count = |name: &str| names.iter().filter(|n| *n == name).count();
    assert_eq!(count("harmonics"), 2, "one harmonics per basis: {names:?}");
    let uploads: Vec<usize> = (0..names.len())
        .filter(|&t| names[t] == "upload U")
        .collect();
    assert_eq!(uploads.len(), 2, "one upload per basis: {names:?}");
    let last = names.len() - 1;
    // The second build adds `packed three-center tables`, `packed
    // three-center` and `spherical transform` alone.
    assert_eq!(names[last], "spherical transform", "{names:?}");
    assert_eq!(names[last - 1], "packed three-center", "{names:?}");
    assert_eq!(names[last - 2], "packed three-center tables", "{names:?}");
    assert!(
        uploads.iter().all(|&t| t < last - 2),
        "the second build adds no harmonics: {names:?}"
    );
    let reads = planned.device_table(0).reads();
    for t in uploads {
        assert!(
            reads.contains(&(t, last)),
            "the second build's transform reads upload {t}'s row"
        );
    }
}

/// Records handed in with another basis's tables are refused at planning,
/// in every build: the cells' record ranges and AO offsets come from the
/// tables, and index the records' rows and the Cartesian tensor.
#[test]
#[should_panic(
    expected = "Pre-condition of add_packed_three_center_tasks violated: classes are the records \
                of tables"
)]
fn records_of_other_tables_are_refused() {
    let (basis, aux) = common::water();
    let mut dag = Dag::new();
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let mut table = AuxTable::new(1);
    table.push_shell(0, [0.0; 3], 0, [(1.0, 1.0)]);
    // The aux basis's tables, of other classes than the records'.
    let other = fb_integrals_mmd::record_tables(&aux);
    add_packed_three_center_tasks(&mut dag, 0, &classes, &other, &table, &[0], 1);
}
