//! Water's plan (tier 3): the plan text of the records and the tensors'
//! tasks, the device and pinned tables among it with the arena's size, is a
//! committed golden (one build's `conventional jk` is in the SCF's,
//! `fb-pipeline-scf`'s `plan_scf_h2o_cc-pvdz_conventional.txt`), `FB_BLESS=1` rewriting
//! it (`docs/testing.md`, "Performance"); admission of its device table;
//! and the refusals at planning of a shell above d and of a basis past
//! 150 Cartesian AOs. Planning touches no device, but the load checks read
//! both kernel libraries, so they must be built.

mod common;

use std::path::PathBuf;

use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_integrals_sys::packed_pairs;
use fb_test_support::golden;

/// Water in cc-pVDZ, the d case.
fn water() -> fb_basis::BasisSet {
    common::basis_on(&common::data("h2o.xyz"), "cc-pvdz")
}

#[test]
fn plan_golden() {
    let basis = water();
    let mut dag = Dag::new();
    common::add_step(&mut dag, &basis).expect("water plans");
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = dag
        .plan()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_h2o_cc-pvdz.txt", &text);
}

/// `ERI_pp` and `X_pp` are rows of the device table at planning, `n_pp²`
/// doubles each over the Cartesian AOs (25 for water in cc-pVDZ, so
/// `n_pp = 325`), so the block admission checks counts them: a budget one
/// byte short of the block is refused as `Allocation` before anything
/// runs, naming the bytes, and the block itself is admitted. Catches a
/// tensor built outside the table, which admission would never see, and
/// one sized by the spherical count.
#[test]
fn admission_counts_both_tensors() {
    let basis = water();
    let mut dag = Dag::new();
    let eri = common::add_step(&mut dag, &basis).unwrap();
    let n_pp = packed_pairs(basis.nao_cart() as usize).unwrap();
    assert_eq!((eri.n_cart, eri.n_sph, n_pp), (25, 24, 325));
    let planned = dag.plan();
    let table = planned.device_table(0);
    assert_eq!(table.len(eri.eri), n_pp * n_pp, "ERI_pp's length");
    assert_eq!(table.len(eri.x), n_pp * n_pp, "X_pp's length");
    assert!(
        table.total() >= 2 * 8 * n_pp * n_pp,
        "both tensors live at the peak"
    );
    let cap = table.total() - 1;
    let err = fb_core_arena::admit(table, cap, "gpu0").unwrap_err();
    assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
    let named = format!("needs {} bytes, 1 over its cap", table.total());
    assert!(err.message.contains(&named), "{named} not in: {err}");
    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
}

/// Water in cc-pVTZ has f shells, above the kernels' d: `Unsupported` at
/// planning, naming the shell's momentum and the limit, with nothing added.
#[test]
fn a_shell_above_d_is_unsupported() {
    let basis = common::basis_on(&common::data("h2o.xyz"), "cc-pvtz");
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let before = dag.len();
    let err =
        fb_integrals_eri_conventional::add_conventional(&mut dag, 0, &basis, &classes, &mut None)
            .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
    assert!(
        err.message.contains("has momentum 3") && err.message.contains("through l = 2"),
        "{err}"
    );
    assert_eq!(dag.len(), before, "a refusal adds nothing");
}

/// Seven waters in cc-pVDZ, 175 Cartesian AOs, past the cap of 150:
/// `Unsupported` at planning, naming the count, with nothing added.
#[test]
fn a_basis_past_150_cartesian_aos_is_unsupported() {
    let mut xyz = String::from("21\nseven waters, Angstrom\n");
    for w in 0..7 {
        let z = 3.0 * f64::from(w);
        xyz += &format!(
            "O 0.0 0.0 {:.4}\nH 0.0 0.7572 {:.4}\nH 0.0 -0.7572 {:.4}\n",
            z + 0.1173,
            z - 0.4692,
            z - 0.4692
        );
    }
    let path: PathBuf = std::env::temp_dir().join(format!(
        "fb-integrals-eri-conventional-{}-waters.xyz",
        std::process::id()
    ));
    std::fs::write(&path, xyz).unwrap();
    let basis = common::basis_on(&path, "cc-pvdz");
    let _ = std::fs::remove_file(&path);
    assert_eq!(basis.nao_cart(), 175);
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let before = dag.len();
    let err =
        fb_integrals_eri_conventional::add_conventional(&mut dag, 0, &basis, &classes, &mut None)
            .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
    assert!(
        err.message.contains("175 Cartesian AOs") && err.message.contains("at most 150"),
        "{err}"
    );
    assert_eq!(dag.len(), before, "a refusal adds nothing");
}
