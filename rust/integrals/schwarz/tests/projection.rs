//! The projection inequality the density-fitting screen rests on, on a
//! card (tier 4, device 0): with the Coulomb metric,
//! `Σ_Q B_Q,mn² = (mn|mn)_DF <= (mn|mn)`, so a shell pair whose every
//! Cartesian `sqrt((mn|mn))` is below the threshold has columns of B below
//! it too.
//!
//! B is the device's own: the three-center integrals
//! (`fb_integrals_eri_density_fitting_three_center`), the metric and the
//! fit, as the DF step builds them, over the spherical functions. Those are
//! fixed combinations of the Cartesian ones, `|μν) = Σ_mn U_mμ U_nν |mn)`
//! (U, [`fb_basis::spherical_transform_matrix`]), so the statement
//! through U is, by the triangle inequality in the Coulomb norm,
//!
//! ```text
//! Σ_Q B_Q,μν² <= (μν|μν) <= (Σ_mn |U_mμ| |U_nν| sqrt((mn|mn)))²
//! ```
//!
//! and for s and p shells, whose blocks of U are the identity, it is the
//! Cartesian statement itself, element by element. The pairs are
//! `fb_integrals_mmd::record_tables`' (its `ClassPairs`), in its order, so
//! the walk over the diagonal's blocks is the one the screen's caller
//! makes.
//!
//! The bar is the rounding of both sides, `1e-10` relative, the libcint
//! comparison's: the fit's and the integrals' rounding is near 1e-14 of
//! max, far inside it. One run.

use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_integrals_eri_density_fitting_fit::add_density_fit;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_three_center::add_three_center;

/// The bar on `Σ_Q B² / bound² - 1`.
const BAR: f64 = 1e-10;

/// Water in cc-pVDZ and cc-pVDZ-JKFIT, placed as the DF step places them.
fn water() -> (BasisSet, BasisSet) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h2o.xyz");
    let molecule = fb_io_xyz::read_xyz(&path, 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers water");
    let aux = fb_io_basis::resolve_aux_basis("cc-pvdz-jkfit")
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers water");
    (basis, aux)
}

/// B of `basis` fitted over `aux` on device 0, one run, on the host:
/// `naux · nao²` doubles, element `(Q, n + nao m)` at `Q + naux (n + nao m)`.
fn fitted(basis: &BasisSet, aux: &BasisSet) -> Vec<f64> {
    let (nao, naux) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, basis).expect("the library loads");
    let mut u_aux = None;
    let tensor = add_three_center(&mut dag, 0, basis, &classes, aux, &mut None, &mut u_aux)
        .expect("the kernels cover both sets");
    let metric = add_metric(&mut dag, 0, aux, &mut u_aux).expect("the kernels cover the aux set");
    let (b, _) =
        add_density_fit(&mut dag, 0, nao, naux, metric, tensor).expect("the library loads");
    let len = naux * nao * nao;
    let pinned = dag.add_gpu("download", 0, Download { b, len });
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            pinned,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let got = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(got.len(), len, "B downloaded");
    got
}

/// Water in cc-pVDZ / cc-pVDZ-JKFIT: every spherical pair's column of the
/// device's B within its bound through U from the host diagonal, the
/// Cartesian statement exactly on the s and p shells. Catches a diagonal
/// below the true `(mn|mn)` (a normalisation smaller than the kernels', a
/// quartet dropped), and a diagonal walked in another order than the
/// tables' pairs. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_b_is_within_the_host_diagonal() {
    let (basis, aux) = water();
    let b = fitted(&basis, &aux);
    let (nc, ns, naux) = (
        basis.nao_cart() as usize,
        basis.nao_sphe() as usize,
        aux.nao_sphe() as usize,
    );
    let mut u = vec![0.0; nc * ns];
    fb_basis::spherical_transform_matrix(&basis, &mut u);

    let tables = fb_integrals_mmd::record_tables(&basis);
    let pairs: Vec<(u32, u32)> = tables
        .classes()
        .iter()
        .flat_map(|c| c.bra.iter().copied().zip(c.ket.iter().copied()))
        .collect();
    let q = fb_integrals_schwarz::schwarz_diagonal(&basis, &pairs);

    let shells = basis.shells();
    let (mut at, mut worst, mut tightest, mut misses) = (0, f64::MIN, f64::MAX, Vec::new());
    for &(a, c) in &pairs {
        let (sa, sc) = (&shells[a as usize], &shells[c as usize]);
        let (na, nb) = (
            sa.function_count_cart() as usize,
            sc.function_count_cart() as usize,
        );
        let block = &q[at..at + na * nb];
        at += na * nb;
        let (ca, cc) = (
            basis.ao_offset_cart(a) as usize,
            basis.ao_offset_cart(c) as usize,
        );
        let (pa, pc) = (
            basis.ao_offset_sphe(a) as usize,
            basis.ao_offset_sphe(c) as usize,
        );
        for mu in pa..pa + sa.function_count_sphe() as usize {
            for nu in pc..pc + sc.function_count_sphe() as usize {
                let mut bound = 0.0;
                for m in 0..na {
                    for n in 0..nb {
                        bound +=
                            (u[ca + m + nc * mu] * u[cc + n + nc * nu]).abs() * block[m * nb + n];
                    }
                }
                let col = &b[naux * (nu + ns * mu)..][..naux];
                let norm2: f64 = col.iter().map(|v| v * v).sum();
                let ratio = norm2 / (bound * bound) - 1.0;
                worst = worst.max(ratio);
                tightest = tightest.min(ratio.abs());
                if ratio.is_nan() || ratio > BAR {
                    misses.push(format!(
                        "shells ({a}, {c}), functions ({mu}, {nu}): Σ B² {norm2:e} > bound² {:e}",
                        bound * bound
                    ));
                }
            }
        }
    }
    assert_eq!(at, q.len(), "every block walked");
    println!(
        "water cc-pVDZ / cc-pVDZ-JKFIT: {} pairs, max Σ B² / bound² - 1 = {worst:e}, closest \
         |Σ B² / bound² - 1| = {tightest:e}, bar {BAR:e}",
        pairs.len()
    );
    assert!(
        misses.is_empty(),
        "{} columns past the bound:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

/// GPU: B, `len` doubles, copied into a pinned row on the stream.
struct Download {
    b: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.b);
        host.output("B host", self.len)
    }

    fn run(
        &self,
        pinned: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(host, *pinned, device.get(self.b), stream)
    }
}

/// CPU: the pinned copy read out, after the executor waited for the
/// download.
struct Collect {
    pinned: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.pinned);
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = host.get(self.pinned).to_vec();
        Ok(())
    }
}
