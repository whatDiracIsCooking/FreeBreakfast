//! A transform's bits within one process do not depend on the plan around
//! it (tier 4, device 0): the guarantee `docs/testing.md`, "Determinism and
//! placements", rests a bitwise comparison of two plans on.
//!
//! Water in cc-pVTZ (`nao_cart` 65, `nao_sphe` 58), three fixed Cartesian
//! matrices, each transformed by [`add_cartesian_to_spherical`] in two
//! plans: the bare one, and the same with one more stream task, unused, U's
//! row copied into a row nobody reads, between the upload and the
//! transforms. The second GEMM's shape, `58 × 58 × 65` with A transposed,
//! is one hipBLASLt has no tuned kernel for on the gfx1200, so it takes one
//! of two matching kernels, which differ in their bits, and which one
//! varies between processes (#1575). Within one process the choice is
//! made once: the four runs, bare, extended, bare, extended, are the same
//! bits. A library that chose per call, or per plan, fails this.

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_recipes_matrix_ao::{add_cartesian_to_spherical, add_harmonics};

/// CPU: the `k`-th fixed, non-symmetric Cartesian matrix, `len` doubles,
/// computed into a pinned row: every element its own value.
struct Fill {
    k: usize,
    len: usize,
}

impl CpuTask for Fill {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("cartesian host", self.len)
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>) -> Status {
        for (i, x) in host.take_mut(*row).iter_mut().enumerate() {
            *x = (((i * 7919 + self.k * 104_729) % 1000) as f64 / 997.0).sin();
        }
        Ok(())
    }
}

/// GPU: a device row copied into a device output row, the unused task.
struct Unused {
    device: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Unused {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.device);
        device.output("copy", self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2d(&device.take_mut(*row), device.get(self.device), stream)
    }
}

/// CPU: the pinned copies read into `out`.
struct Collect(Vec<RowId<Pinned, f64>>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>) {
        for &r in &self.0 {
            host.read(r);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>) -> Status {
        let mut out = self.1.lock().unwrap();
        for &r in &self.0 {
            out.extend_from_slice(host.get(r));
        }
        Ok(())
    }
}

/// The three matrices transformed in one plan, with the unused task when
/// `extended`, run once on device 0: every spherical element, in order.
fn transformed(basis: &fb_basis::BasisSet, extended: bool) -> Vec<f64> {
    let (n_cart, n_sph) = (basis.nao_cart() as usize, basis.nao_sphe() as usize);
    let mut dag = Dag::new();
    let mut u = None;
    let u_row = add_harmonics(&mut dag, 0, basis, &mut u);
    if extended {
        dag.add_gpu(
            "unused",
            0,
            Unused {
                device: u_row,
                len: n_cart * n_sph,
            },
        );
    }
    let mut spherical = Vec::new();
    for k in 0..3_usize {
        let host = dag.add_cpu(
            "fill",
            0,
            Fill {
                k,
                len: n_cart * n_cart,
            },
        );
        let cartesian = dag.upload(0, host, "copy", n_cart * n_cart);
        spherical.push(add_cartesian_to_spherical(
            &mut dag,
            0,
            basis,
            &mut u,
            &format!("M{k}"),
            cartesian,
        ));
    }
    let down = spherical
        .into_iter()
        .map(|m| dag.download(0, m, "spherical host", n_sph * n_sph))
        .collect();
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu("collect", 0, Collect(down, Arc::clone(&out)));
    dag.blas_workspace(0);
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    std::mem::take(&mut *out.lock().unwrap())
}

/// The four runs, bare, extended, bare, extended, in one process: the same
/// bits, element for element. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn two_plans_give_the_same_bits_in_one_process() {
    let molecule = fb_io_xyz::parse_xyz(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n",
        0,
        1,
    )
    .unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvtz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let runs: Vec<Vec<f64>> = [false, true, false, true]
        .into_iter()
        .map(|extended| transformed(&basis, extended))
        .collect();
    let n = runs[0].len();
    assert_eq!(n, 3 * 58 * 58);
    for (r, run) in runs.iter().enumerate().skip(1) {
        let differ = run
            .iter()
            .zip(&runs[0])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("run {r} against run 0: {differ} of {n} elements differ");
        assert_eq!(differ, 0, "run {r} against run 0");
    }
}
