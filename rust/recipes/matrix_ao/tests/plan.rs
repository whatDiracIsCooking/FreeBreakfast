//! Two matrices transformed in one plan (tier 3): U is staged once, by one
//! `harmonics` task, copied up once, by one `upload U` copy node, and
//! both transforms read that one device row. Planning touches no device;
//! the tasks producing the Cartesian matrices declare a row and are never
//! run.

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// GPU: a Cartesian matrix, declared and never computed.
struct Cartesian {
    name: &'static str,
    len: usize,
}

impl GpuTask for Cartesian {
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

#[test]
fn two_matrices_share_one_device_u() {
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
    let (n_cart, n_sph) = (basis.nao_cart() as usize, basis.nao_sphe() as usize);
    assert!(basis.nao_sphe() < basis.nao_cart());

    let mut dag = Dag::new();
    let mut u = None;
    let t_cart = dag.add_gpu(
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
    let v_cart = dag.add_gpu(
        "V cartesian",
        0,
        Cartesian {
            name: "V cartesian",
            len: n_cart * n_cart,
        },
    );
    add_cartesian_to_spherical(&mut dag, 0, &basis, &mut u, "V", v_cart);
    let u = u.expect("U's row in the slot");
    let planned = dag.plan();
    let text = planned.to_string();

    // T cartesian, harmonics, its upload, T's transform, V cartesian, V's
    // transform: one upload for both.
    let names: Vec<String> = text
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
            "upload U",
            "spherical transform",
            "V cartesian",
            "spherical transform"
        ]
    );
    // The upload reads the harmonics task's pinned row; both transforms
    // read the upload's device row, U, nao_cart × nao_sphe doubles.
    assert!(planned.predecessors(2).contains(&1));
    let device = planned.device_table(0);
    assert_eq!(device.len(u), n_cart * n_sph);
    let reads = device.reads();
    assert!(reads.contains(&(2, 3)) && reads.contains(&(2, 5)));
    assert!(planned.predecessors(5).contains(&4));
    // The device table holds one row named U, written once and read by
    // both transforms: no transform copies U into scratch of its own.
    let device_text = text
        .split("worker 0 pinned table")
        .next()
        .and_then(|t| t.split("worker 0 device table").nth(1))
        .expect("the device table's text");
    // Each row: offset, bytes, owner, life, `name:`, users.
    let u_rows: Vec<String> = device_text
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .filter(|w| w.get(4) == Some(&"U:"))
        .map(|w| w[1..].join(" "))
        .collect();
    assert_eq!(
        u_rows,
        [format!("{} edge 2-5 U: 2w 3r 5r", 8 * n_cart * n_sph)]
    );
}
