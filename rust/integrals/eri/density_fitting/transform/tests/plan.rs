//! The transform's plan (tier 3): an s, p and d basis's every pair as one
//! batch, the plan text with its device table (the two scratch rows and
//! the output, their bytes) the committed golden
//! `tests/golden/plan_spd_naux25.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"); and the refusals planning makes.
//! Planning touches no device; the crate links the kernel library through
//! the packed map's, so it must be built.

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_transform::add_spherical_transform;
use fb_integrals_mmd::record_tables;
use fb_test_support::golden;

/// The aux basis's Cartesian and spherical counts: s, p, d, f and g
/// shells.
const N_AUX: [usize; 2] = [35, 25];

/// GPU: an input, declared and never computed.
struct Input {
    name: &'static str,
    len: usize,
}

impl GpuTask for Input {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

/// The map of an s, a p and a d shell on two centres, every pair kept.
fn map() -> PairMap {
    let at = |z| Double3 { x: 0.0, y: 0.0, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(0, &[1.0], &[1.0], at(0.0)),
        Shell::zipped(1, &[0.7], &[1.0], at(0.0)),
        Shell::zipped(2, &[0.5], &[1.0], at(1.5)),
    ]);
    PairMap::new(&record_tables(&basis), None, 0.0)
}

/// The batch's producer and the two U rows' on worker 0 of a new plan,
/// for `n_aux`.
fn inputs(dag: &mut Dag<'_>, map: &PairMap, n_aux: [usize; 2]) -> [RowId<Device, f64>; 3] {
    let (cart, sphe) = (map.cart(), map.sphe());
    [
        ("(mn|P) cartesian", n_aux[0] * cart.n_columns()),
        ("U", cart.nao() * sphe.nao()),
        ("U aux", n_aux[0] * n_aux[1]),
    ]
    .map(|(name, len)| dag.add_gpu(name, 0, Input { name, len }))
}

#[test]
fn plan_golden() {
    let map = map();
    let mut dag = Dag::new();
    let [x, u, u_aux] = inputs(&mut dag, &map, N_AUX);
    let out = add_spherical_transform(
        &mut dag,
        0,
        &map,
        0..map.pairs().len(),
        N_AUX,
        x,
        [u, u_aux],
    )
    .unwrap();
    let planned = dag.plan();
    assert_eq!(
        planned.device_table(0).len(out),
        N_AUX[1] * map.sphe().n_columns()
    );
    assert_eq!(planned.predecessors(3), [0, 1, 2]);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_spd_naux25.txt", &text);
}

/// An aux basis whose `n_aux · n_b_sph`, the bra GEMM's rows and leading
/// dimension, passes `i32` is refused at planning as `Unsupported`, naming
/// the count, and nothing is added: the casts in the task would wrap.
#[test]
fn a_count_past_i32_is_unsupported_at_planning() {
    let map = map();
    let mut dag = Dag::new();
    let [x, u, u_aux] = inputs(&mut dag, &map, [1, 1]);
    // d's 5 spherical functions times this pass i32; it alone does not.
    let n_aux = (i32::MAX / 4) as usize;
    let err = add_spherical_transform(
        &mut dag,
        0,
        &map,
        0..map.pairs().len(),
        [n_aux, n_aux],
        x,
        [u, u_aux],
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
    assert!(
        err.message
            .contains(&format!("a GEMM count of {}", n_aux * 5)),
        "{err}"
    );
    assert_eq!(
        dag.plan()
            .to_string()
            .matches("spherical transform")
            .count(),
        0
    );
}

/// An empty batch is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a batch of at least one kept pair")]
fn an_empty_batch_is_refused() {
    let map = map();
    let mut dag = Dag::new();
    let [x, u, u_aux] = inputs(&mut dag, &map, N_AUX);
    let _ = add_spherical_transform(&mut dag, 0, &map, 1..1, N_AUX, x, [u, u_aux]);
}
