//! Two matrices transformed in one plan (tier 3): U is staged once, by one
//! `harmonics` task, and both transforms read its row. Planning touches no
//! device; the tasks producing the Cartesian matrices declare a row and are
//! never run.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// GPU: a Cartesian matrix, declared and never computed.
struct Cartesian {
    name: &'static str,
    len: usize,
}

impl Task for Cartesian {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

#[test]
fn two_matrices_share_one_harmonics_task() {
    // cc-pVDZ on water has d shells, so U is not the identity.
    let molecule = fb_io_xyz::parse_xyz(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let n_cart = basis.nao_cart() as usize;
    assert!(basis.nao_sphe() < basis.nao_cart());

    let mut dag = Dag::new();
    let mut u = None;
    let t_cart = dag.add(
        "T cartesian",
        0,
        Cartesian {
            name: "T cartesian",
            len: n_cart * n_cart,
        },
    );
    add_cartesian_to_spherical(&mut dag, 0, &basis, &mut u, "T", t_cart);
    assert!(
        u.is_some(),
        "the first transform leaves U's row in the slot"
    );
    let v_cart = dag.add(
        "V cartesian",
        0,
        Cartesian {
            name: "V cartesian",
            len: n_cart * n_cart,
        },
    );
    add_cartesian_to_spherical(&mut dag, 0, &basis, &mut u, "V", v_cart);
    let planned = dag.plan();

    // T cartesian, harmonics, its transform, V cartesian, its transform.
    let names: Vec<String> = planned
        .to_string()
        .lines()
        .skip(2)
        .take(planned.len())
        .map(|l| l.split_whitespace().skip(3).collect::<Vec<_>>().join(" "))
        .map(|l| l.split(':').next().unwrap().to_owned())
        .collect();
    assert_eq!(
        names,
        [
            "T cartesian",
            "harmonics",
            "spherical transform",
            "V cartesian",
            "spherical transform"
        ]
    );
    // Both transforms run after the one harmonics task, reading its row.
    assert!(planned.predecessors(2).contains(&1));
    assert!(planned.predecessors(4).contains(&1));
    assert!(planned.predecessors(4).contains(&3));
}
