//! The AO core Hamiltonian `H = T + V`, the kinetic energy and the
//! nuclear attraction of point nuclei, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"). The C++
//! `src/integrals/one_electron/core_hamiltonian` and the pipeline's
//! spherical transform (`MatrixAOPair::cartesian_to_spherical`) are the
//! reference design.
//!
//! [`add_core_hamiltonian`] adds, after the records
//! ([`fb_integrals_mmd::add_records`]) and the nuclei's upload
//! ([`fb_molecule_nuclei::add_nuclei`]), both passed in so one build and
//! one upload serve every reader of a plan, one build task,
//! **`core hamiltonian`**, on the worker's stream: the Cartesian matrix
//! zeroed, every class's kinetic launch
//! ([`fb_integrals_one_electron_kinetic::kinetic_classes`]) and then every
//! class's nuclear-attraction launch on top
//! ([`fb_integrals_one_electron_potential_en::potential_en_classes`]). A
//! chain of launches on one stream is one task (`docs/programming_model.md`,
//! "Open questions"): both kernels only add, so H is formed in place, with
//! no T or V matrix of its own and no separate addition. Then the shared
//! spherical transform, `H = Uᵀ H_cart U`
//! ([`fb_recipes_matrix_ao::add_cartesian_to_spherical`]), U's slot passed
//! in so a plan stages U once.
//!
//! # Forking the classes: an option, off
//!
//! By default every class's launch goes in order on the worker's lane 0,
//! and the task declares no fork ([`GpuTask::forks`] is 0): bitwise the
//! launch order the build had before forks existed, and the plan goldens
//! are that order's. A builder that sets [`ClassRows::fork`] on the
//! records it passes in opts in, for this build and for the overlap,
//! kinetic and nuclear-attraction builds over the same records alike:
//! each family then launches a class per fork of the task's stream and
//! joins them back ([`DeviceStream::fork_join`]), as the C++
//! `launch_each`, and the task declares a fork per class. **The T forks
//! are joined before the V forks are made**: two fork-joins, as the C++
//! `kinetic` and then `add_nuclear_attraction`, rather than one fork per
//! (class, kind). Both are pure adds, so either is correct; this one
//! reuses the two families' own launchers, and one fork per (class, kind)
//! measured within noise of it.
//!
//! It is off because it benches slower. The `core hamiltonian` task on a
//! gfx1200 (HIP, release), serial against forked, each the median of 300
//! runs after 10 warm-up runs, in two alternating rounds (a range is the
//! two rounds); CUDA not measured:
//!
//! | Fixture | Task device time, serial → forked | Run wall time, serial → forked |
//! |---|---|---|
//! | H2O / STO-3G (3 classes) | 78 µs → 146 µs | 1.12–1.19 ms → 1.38–1.47 ms (+16–31%) |
//! | H2O / cc-pVDZ (6 classes) | 327 µs → 511–695 µs | 1.91–2.27 ms → 2.35–2.80 ms (+23%) |
//! | CO / cc-pVTZ (10 classes) | 1.56–1.58 ms → 1.42–1.55 ms | 4.73–5.15 ms → 4.92–5.12 ms (±5%) |
//!
//! The forks' events cost more than the classes' overlap wins at these
//! sizes. The likelier payoff is the density-fitted three-center build,
//! whose classes are larger, the follow-up #1632 names.
//!
//! H is not renormalized, as T and V are not: an AO's norm is its published
//! contraction's ([`fb_basis::ShellStamp::primitive_normalized`]).
//!
//! | C++ | Here |
//! |---|---|
//! | `core_hamiltonian(res, classes, molecule, h.cartesian())`: `kinetic` (zeroed, then one launch per class) and `add_nuclear_attraction` on top | the `core hamiltonian` task of [`add_core_hamiltonian`]: zeroed, then [`kinetic_classes`] and [`potential_en_classes`] on the worker's lane 0, each class's records, formed on a lane of their own, an event wait on the device; the records' [`ClassRows`] passed in by the caller |
//! | `launch_each` in each: one stream per class, its `device_stream()`'s fork stream, forked from `res.stream()` and joined back | with [`ClassRows::fork`] set (off by default: every class in order on lane 0), [`DeviceStream::fork_join`] in each of [`kinetic_classes`] and [`potential_en_classes`]: a fork per class from the worker's fork streams, joined back into lane 0 before the next family forks, so every V launch follows every T launch; never a lane's stream |
//! | `check_one_electron_build`'s `InvalidConfig` for a class above f, `la > lb` or `n_ao²` past `INT32_MAX` | `fb-integrals-sys`'s launchers: a contract for the class, an always-on `assert!` of the matrix's `n_cart²` and the `i32` bound; [`add_core_hamiltonian`] asserts the `i32` bound at planning, and the records hold only `la <= lb` |
//! | `check_nuclei`'s `InvalidConfig`, before anything launches | [`add_core_hamiltonian`]'s `InvalidConfig` at planning for [`NucleiRows`] with no nucleus, adding nothing; nuclei of another worker are another table's rows, which [`Dag::add_gpu`] refuses with a panic |
//! | `molecule.view()`, the `NucleiView` | [`NucleiRows`] from [`fb_molecule_nuclei::add_nuclei`], passed in, so one upload serves every reader of the plan |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `res.dev_idx()` made current for the call | nothing: a stream task runs with its worker's device current |
//!
//! [`kinetic_classes`]: fb_integrals_one_electron_kinetic::kinetic_classes
//! [`potential_en_classes`]: fb_integrals_one_electron_potential_en::potential_en_classes

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::memset;
use fb_integrals_one_electron_kinetic::kinetic_classes;
use fb_integrals_one_electron_potential_en::potential_en_classes;
use fb_integrals_sys::{ClassRows, Fields};
use fb_molecule_nuclei::NucleiRows;
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add the spherical AO core Hamiltonian of `basis` over `nuclei` to `dag`
/// on GPU worker `worker`, and return H: an output row of `nao_sphe²`
/// doubles, column-major, symmetric up to rounding.
///
/// `core hamiltonian` (stream: the Cartesian matrix zeroed, [`kinetic_classes`] and then
/// [`potential_en_classes`] into it, each a fork per class joined back
/// before the next when the classes opt in ([`ClassRows::fork`]), reading the records and the nuclei as edges), then the spherical transform of [`add_cartesian_to_spherical`]:
/// `harmonics` and `upload U` (U into a pinned row, then its
/// device row), only when `u` is `None`, and
/// `spherical transform` (stream: `H = Uᵀ H_cart U`). `nuclei` are the
/// molecule's, uploaded on `worker` by [`fb_molecule_nuclei::add_nuclei`]
/// earlier in `dag`, the molecule the basis is placed on; `classes` are the
/// records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`, so one build serves
/// every reader of the plan; `u` is the plan's slot for U over `basis` on
/// `worker`, shared with every other matrix the plan transforms.
///
/// # Errors
/// `InvalidConfig` for `nuclei` with no nucleus (`n_nuc` not positive: the
/// nuclear-attraction wrapper's refusal, at planning), before anything is
/// added.
///
/// # Panics
/// If `nao_cart²` does not fit an `i32` (the kernels index the matrix in
/// 32 bits), or `classes` or `nuclei` are rows of another worker's or
/// another plan's table ([`Dag::add_gpu`]).
///
/// [`kinetic_classes`]: fb_integrals_one_electron_kinetic::kinetic_classes
/// [`potential_en_classes`]: fb_integrals_one_electron_potential_en::potential_en_classes
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_core_hamiltonian(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    nuclei: NucleiRows,
    u: &mut Option<RowId<Device, f64>>,
) -> Result<RowId<Device, f64>> {
    if nuclei.n_nuc < 1 {
        return Err(Error::config(format!(
            "add_core_hamiltonian: {} nuclei; the nuclear-attraction kernels need at least one",
            nuclei.n_nuc
        )));
    }
    let n_cart = basis.nao_cart() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_core_hamiltonian: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let h_cart = dag.add_gpu(
        "core hamiltonian",
        worker,
        CoreHamiltonian {
            classes: classes.to_vec(),
            nuclei,
            n_cart,
        },
    );
    Ok(add_cartesian_to_spherical(
        dag, worker, basis, u, "H", h_cart,
    ))
}

/// GPU: the Cartesian core Hamiltonian, `nao_cart²`, zeroed, then T summed
/// class by class ([`kinetic_classes`]) and V on top
/// ([`potential_en_classes`]).
struct CoreHamiltonian {
    classes: Vec<ClassRows>,
    nuclei: NucleiRows,
    n_cart: usize,
}

impl GpuTask for CoreHamiltonian {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device, Fields::KINETIC.union(Fields::POTENTIAL_EN));
        }
        self.nuclei.read(device);
        device.output("H cartesian", self.n_cart * self.n_cart)
    }

    fn forks(&self) -> usize {
        if self.classes.iter().any(|c| c.fork) {
            self.classes.len()
        } else {
            0
        }
    }

    fn run(
        &self,
        h: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let mut h = device.take_mut(*h);
        memset::zero(&mut h, stream)?;
        // Each family launches in order on this stream, or, when the classes
        // opt in, forks a stream per class and joins them back into this one
        // before it returns; either way every V launch follows every T
        // launch, and each adds into `h` in turn.
        kinetic_classes(stream, device, &self.classes, &mut h)?;
        potential_en_classes(stream, device, &self.classes, &self.nuclei, &mut h)
    }
}
