//! The MainBus: the config, what setup derives from it, the ledger, each
//! GPU's `GpuResources`, the manager's lock, and one slot per step. A
//! deliberate god object; a step's `run` is handed a view of it and the
//! guards the wrapper took, never the bus.
//!
//! Setup is the unbudgeted host phase: it parses the config, the geometry
//! and the bases, and **counts** every length a step's arrays need into
//! [`Dims`] by a walk over the shells that builds no array
//! ([`for_each_pair`]). Each `mem_budget` is then a function of the
//! dimensions, the step's options and the caps.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use fb_basis::BasisSet;
use fb_core_error_handling::{Error, Result, Status};
use fb_molecule::Molecule;
use serde::Deserialize;

use crate::budget::{
    Backing, Cap, Caps, Kind, Lease, Ledger, LockId, ManagerId, device_margin, host_margin,
};
use crate::device::{GpuResources, PartMut, ScopedDevice, Stream};
use crate::flow::{Bus, Placement, Slot};
use crate::host::host_available;
use crate::steps::core_hamiltonian::CoreHamiltonianOutput;
use crate::steps::eri::EriManager;
use crate::steps::guess::GuessOutput;
use crate::steps::metric::MetricOutput;
use crate::steps::orthogonalizer::OrthogonalizerOutput;
use crate::steps::overlap::OverlapOutput;
use crate::steps::scf::ScfOutput;

/// The config file: one table per step, plus the run's resources. A step
/// whose table is absent is not in the DAG.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The geometry.
    pub molecule: MoleculeTable,
    /// The orbital basis.
    pub basis: BasisTable,
    /// The density-fitted integrals; absent, no `metric` or `eri` step.
    pub eri: Option<EriTable>,
    /// The RHF loop; absent, no `scf` step.
    pub scf: Option<ScfTable>,
    /// GPUs, threads and caps.
    #[serde(default)]
    pub resources: ResourcesTable,
}

/// `[molecule]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeTable {
    /// The XYZ file, relative to the config's directory.
    pub xyz: PathBuf,
}

/// `[basis]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisTable {
    /// A compiled-in name or a BSE JSON path.
    pub name: String,
}

/// `[eri]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EriTable {
    /// The fitting set, a compiled-in name.
    pub aux_basis: String,
    /// The step limit: each GPU's ERI buffer, MiB. Absent, 2 GiB, a fixed
    /// constant and a placeholder.
    pub mem_per_gpu_mib: Option<f64>,
    /// The build's ring depth: 2, or 1 for the test that holds one
    /// against the other.
    #[serde(default = "two")]
    pub ring: usize,
}

fn two() -> usize {
    2
}

/// `[scf]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScfTable {
    /// The most Fock builds.
    pub max_iterations: usize,
    /// `|ΔE|` under which the energy has converged, hartree.
    pub energy_tolerance: f64,
    /// `‖Xᵀ (F D S - S D F) X‖` under which the density has.
    pub commutator_tolerance: f64,
    /// The DIIS subspace's size.
    pub diis_history: usize,
}

impl Default for ScfTable {
    fn default() -> Self {
        ScfTable {
            max_iterations: 100,
            energy_tolerance: 1e-10,
            commutator_tolerance: 1e-7,
            diis_history: 8,
        }
    }
}

/// `[resources]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResourcesTable {
    /// The GPUs of the run, each by the device it is on; the first is
    /// gpu0, and `num_gpus` is the list's length. A device named twice is
    /// two logical GPUs on one physical device, a test facility.
    pub devices: Vec<i32>,
    /// The executor's width.
    pub threads: usize,
    /// The device cap per GPU, MiB; absent, the GPU's free memory after
    /// its handles exist, less its margin.
    pub device_cap_mib: Option<u64>,
    /// The host cap, heap and pinned together, MiB; absent, the host's
    /// available memory at setup less its margin.
    pub host_cap_mib: Option<u64>,
    /// Test mode: the wrapper fills each scratch lease with NaN before
    /// `run`.
    pub nan_fill: bool,
}

impl Default for ResourcesTable {
    fn default() -> Self {
        ResourcesTable {
            devices: vec![0],
            threads: 4,
            device_cap_mib: None,
            host_cap_mib: None,
            nan_fill: false,
        }
    }
}

impl Config {
    /// Parse `text`, resolving `[molecule] xyz` against `dir`.
    ///
    /// # Errors
    /// `InvalidConfig`, with the parser's message.
    pub fn parse(text: &str, dir: &Path) -> Result<Config> {
        let mut config: Config =
            toml::from_str(text).map_err(|e| Error::config(format!("config: {e}")))?;
        if config.molecule.xyz.is_relative() {
            config.molecule.xyz = dir.join(&config.molecule.xyz);
        }
        Ok(config)
    }
}

/// One class of orbital shell pairs, counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassDim {
    /// The bra shells' momentum.
    pub la: i32,
    /// The ket shells', at least `la`.
    pub lb: i32,
    /// Its shell pairs.
    pub n_pair: usize,
    /// Its primitive pair records.
    pub n_rec: usize,
}

impl ClassDim {
    /// A pair's spherical columns.
    pub fn sph(&self) -> usize {
        n_sph_of(self.la) * n_sph_of(self.lb)
    }
}

/// The spherical functions of a shell of momentum `l`.
pub fn n_sph_of(l: i32) -> usize {
    (2 * l + 1) as usize
}

/// The dimensions: every length a step's arrays need, counted by setup and
/// built by none of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    /// Spherical AOs.
    pub n_ao: usize,
    /// Doubly occupied orbitals.
    pub n_occ: usize,
    /// Orbital shells.
    pub n_shell: usize,
    /// Orbital primitives.
    pub n_prim: usize,
    /// Orbital shell pairs `i <= j`, unscreened.
    pub n_shell_pair: usize,
    /// The spherical pair columns of the tree's packed pair layout,
    /// unscreened: the bound B is budgeted at.
    pub n_col: usize,
    /// The shell-pair classes, `(la, lb)` ascending.
    pub classes: Vec<ClassDim>,
    /// Spherical fitting functions; 0 with no `[eri]`.
    pub n_aux: usize,
    /// Fitting shells.
    pub n_aux_shell: usize,
    /// Fitting primitives: the aux records.
    pub n_aux_prim: usize,
    /// The fitting shells of each momentum.
    pub aux_shells_of_l: Vec<usize>,
}

impl Dims {
    /// The index of class `(la, lb)`.
    ///
    /// # Panics
    /// If the basis has no such class.
    pub fn class_of(&self, la: i32, lb: i32) -> usize {
        self.classes
            .iter()
            .position(|c| c.la == la && c.lb == lb)
            .expect("a class of the basis")
    }
}

/// Every shell pair `i <= j` of `basis` in the tree's sweep order, swapped
/// when `i` has the higher momentum: `f(la, lb, bra, ket)`. The pairs of
/// one class come in the order its tables list them. The walk builds
/// nothing: setup counts with it, and a step writes its tables with it
/// straight into pinned scratch.
pub fn for_each_pair(basis: &BasisSet, mut f: impl FnMut(i32, i32, u32, u32)) {
    let shells = basis.shells();
    let n = basis.nshells();
    for i in 0..n {
        for j in i..n {
            let (li, lj) = (
                shells[i as usize].angular_momentum(),
                shells[j as usize].angular_momentum(),
            );
            if li > lj {
                f(lj, li, j, i);
            } else {
                f(li, lj, i, j);
            }
        }
    }
}

/// The classes of `basis`, counted: `(la, lb)` ascending.
fn count_classes(basis: &BasisSet) -> Vec<ClassDim> {
    let shells = basis.shells();
    let mut classes: Vec<ClassDim> = Vec::new();
    for_each_pair(basis, |la, lb, bra, ket| {
        let n_rec =
            (shells[bra as usize].nprimitives() * shells[ket as usize].nprimitives()) as usize;
        match classes.iter_mut().find(|c| c.la == la && c.lb == lb) {
            Some(c) => {
                c.n_pair += 1;
                c.n_rec += n_rec;
            }
            None => classes.push(ClassDim {
                la,
                lb,
                n_pair: 1,
                n_rec,
            }),
        }
    });
    classes.sort_by_key(|c| (c.la, c.lb));
    classes
}

/// One acquisition of a lock by a step, for the lock-order test and the
/// timeline.
#[derive(Clone, Debug)]
pub struct Acquisition {
    /// The step.
    pub step: &'static str,
    /// The lock.
    pub lock: LockId,
    /// When the wrapper asked for it, from setup's end.
    pub asked: Duration,
    /// When it had it.
    pub locked: Duration,
    /// When it let go, after the step-end sync.
    pub released: Duration,
    /// Whether the lock was held by another step when asked for.
    pub waited: bool,
}

/// What setup fills, before budget mode. Every step may read it without
/// declaring anything: it is not another step's product.
pub struct Shared {
    /// The parsed config.
    pub config: Config,
    /// The geometry.
    pub molecule: Molecule,
    /// The orbital basis, primitive-normalized and placed.
    pub basis: BasisSet,
    /// The fitting set, likewise; `None` with no `[eri]`.
    pub aux: Option<BasisSet>,
    /// The dimensions.
    pub dims: Dims,
    /// The nuclear repulsion energy, hartree.
    pub e_nuc: f64,
    /// The device each GPU is on.
    pub physical: Vec<i32>,
    gpus: Vec<Mutex<GpuResources>>,
    eri_lock: Mutex<()>,
    epoch: Instant,
    acquisitions: Mutex<Vec<Acquisition>>,
}

impl Shared {
    /// The GPUs of the run.
    pub fn num_gpus(&self) -> usize {
        self.physical.len()
    }

    /// Every acquisition so far, in the order the locks were released.
    pub fn acquisitions(&self) -> Vec<Acquisition> {
        self.acquisitions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// syevd's work buffer at order `n` on gpu0, in doubles: a size query,
    /// which needs the live handle. A device-free budget reports 0.
    ///
    /// # Errors
    /// The query's failure.
    pub fn syevd_lwork(&self, n: usize) -> Result<usize> {
        match self.gpus.first() {
            Some(gpu0) => gpu0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .syevd_lwork(n),
            None => Ok(0),
        }
    }
}

/// The run's one shared object.
pub struct MainBus {
    shared: Shared,
    ledger: Ledger,
    pub(crate) overlap: Slot<OverlapOutput, Lane>,
    pub(crate) core_hamiltonian: Slot<CoreHamiltonianOutput, Lane>,
    pub(crate) metric: Slot<MetricOutput, Lane>,
    pub(crate) guess: Slot<GuessOutput, Lane>,
    pub(crate) orthogonalizer: Slot<OrthogonalizerOutput, Lane>,
    pub(crate) eri: Slot<EriManager, Lane>,
    pub(crate) scf: Slot<ScfOutput, Lane>,
}

impl MainBus {
    /// Setup: parse the geometry and the bases on the host, count the
    /// dimensions, make each GPU's `GpuResources`, and read and clamp the
    /// caps (a GPU's after its handles exist). Not budgeted. With
    /// [`Backing::Dry`] no device is touched: the plan can be budgeted and
    /// refused or accepted, not run.
    ///
    /// # Errors
    /// The readers' refusals; an open shell (`Unsupported`); a config with
    /// no GPU or no thread; a device the runtime does not see.
    pub fn setup(config: Config, backing: Backing) -> Result<Arc<MainBus>> {
        let molecule = fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1)?;
        let electrons = molecule.electron_count();
        if electrons % 2 != 0 {
            return Err(Error::unsupported(
                "the prototype runs closed-shell RHF of a neutral molecule only",
            ));
        }
        let basis = fb_io_basis::resolve_basis(&config.basis.name)?
            .primitive_normalized()
            .place(&molecule)?;
        let aux = match &config.eri {
            Some(eri) => Some(
                fb_io_basis::resolve_aux_basis(&eri.aux_basis)?
                    .primitive_normalized()
                    .place(&molecule)?,
            ),
            None => None,
        };
        let classes = count_classes(&basis);
        let mut aux_shells_of_l = Vec::new();
        for shell in aux.iter().flat_map(|a| a.shells()) {
            let l = shell.angular_momentum() as usize;
            if aux_shells_of_l.len() <= l {
                aux_shells_of_l.resize(l + 1, 0);
            }
            aux_shells_of_l[l] += 1;
        }
        let n_shell = basis.nshells() as usize;
        let dims = Dims {
            n_ao: basis.nao_sphe() as usize,
            n_occ: (electrons / 2) as usize,
            n_shell,
            n_prim: basis
                .shells()
                .iter()
                .map(|s| s.nprimitives() as usize)
                .sum(),
            n_shell_pair: n_shell * (n_shell + 1) / 2,
            n_col: classes.iter().map(|c| c.n_pair * c.sph()).sum(),
            classes,
            n_aux: aux.as_ref().map_or(0, |a| a.nao_sphe() as usize),
            n_aux_shell: aux.as_ref().map_or(0, |a| a.nshells() as usize),
            n_aux_prim: aux.as_ref().map_or(0, |a| {
                a.shells().iter().map(|s| s.nprimitives() as usize).sum()
            }),
            aux_shells_of_l,
        };
        let e_nuc = fb_molecule::nuclear_repulsion_energy(&molecule);

        let res = &config.resources;
        if res.devices.is_empty() || res.threads == 0 {
            return Err(Error::config(
                "[resources] devices names a GPU and threads is at least 1",
            ));
        }
        let real = backing == Backing::Real;
        let mut gpus = Vec::new();
        if real {
            fb_recipes_sys::check_library().map_err(Error::gpu)?;
            fb_integrals_sys::check_library().map_err(Error::gpu)?;
            let seen = crate::device::device_count()?;
            if let Some(d) = res.devices.iter().find(|&&d| d < 0 || d >= seen) {
                return Err(Error::config(format!(
                    "[resources] devices names device {d}, but {seen} are visible"
                )));
            }
            // The handles first: a defaulted cap then already sees them.
            for (gpu, &physical) in res.devices.iter().enumerate() {
                gpus.push(Mutex::new(GpuResources::new(gpu, physical)?));
            }
        }
        // Each GPU's cap, clamped to what its device has now. Logical GPUs
        // on one device share its reading and its margin evenly.
        let mut device = Vec::new();
        for &physical in &res.devices {
            let sharing = res.devices.iter().filter(|&&d| d == physical).count() as u64;
            let written = res.device_cap_mib.map(|mib| mib << 20);
            device.push(if real {
                let (free, total) = crate::device::memory(physical)?;
                Cap::resolve(
                    written,
                    Some(free / sharing),
                    device_margin(total) / sharing,
                )
            } else {
                Cap::resolve(written, None, 0)
            });
        }
        let available = host_available()?;
        let host = Cap::resolve(
            res.host_cap_mib.map(|mib| mib << 20),
            Some(available),
            host_margin(available),
        );
        let ledger = Ledger::new(
            Caps { device, host },
            backing,
            res.devices.clone(),
            res.nan_fill,
        );
        Ok(Arc::new(MainBus {
            shared: Shared {
                physical: res.devices.clone(),
                config,
                molecule,
                basis,
                aux,
                dims,
                e_nuc,
                gpus,
                eri_lock: Mutex::new(()),
                epoch: Instant::now(),
                acquisitions: Mutex::new(Vec::new()),
            },
            ledger,
            overlap: Slot::new(),
            core_hamiltonian: Slot::new(),
            metric: Slot::new(),
            guess: Slot::new(),
            orthogonalizer: Slot::new(),
            eri: Slot::new(),
            scf: Slot::new(),
        }))
    }
}

/// A step's own streams on one GPU it spans: its kernels and copies go
/// there. Made by the wrapper in budget mode, synchronized by the wrapper
/// after every `run`.
pub struct GpuLane {
    gpu: usize,
    physical: i32,
    streams: Vec<Stream>,
}

impl GpuLane {
    /// The GPU.
    pub fn gpu(&self) -> usize {
        self.gpu
    }

    /// Make the GPU's device current on this thread.
    pub fn current(&self) -> ScopedDevice {
        ScopedDevice::new(self.physical)
    }

    /// The step's `i`-th stream on this GPU.
    pub fn stream(&self, i: usize) -> &Stream {
        &self.streams[i]
    }

    /// The device the GPU is on.
    pub fn physical(&self) -> i32 {
        self.physical
    }
}

/// A step's streams, per GPU it spans; none for a host step.
pub struct Lane {
    /// The step's GPUs, ascending: gpu0 first.
    pub gpus: Vec<GpuLane>,
}

impl Lane {
    /// The step's streams on gpu0.
    ///
    /// # Panics
    /// For a host step, which has no stream.
    pub fn gpu0(&self) -> &GpuLane {
        self.gpus.first().expect("a host step has no stream")
    }
}

/// What `run` receives of the locks: the guards the wrapper took, in rank
/// order, before `run`. `run` never sees a `Mutex`, so nothing takes a lock
/// inside it.
pub struct Held<'a> {
    _current: Option<ScopedDevice>,
    manager: Option<MutexGuard<'a, ()>>,
    gpus: Vec<Option<MutexGuard<'a, GpuResources>>>,
    taken: Vec<(LockId, Duration, Duration, bool)>,
}

impl Held<'_> {
    /// GPU `gpu`'s resources.
    ///
    /// # Panics
    /// If the step did not declare that GPU's lock.
    #[track_caller]
    pub fn gpu(&mut self, gpu: usize) -> &mut GpuResources {
        self.gpus
            .get_mut(gpu)
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("the step did not declare the lock of gpu {gpu}"))
    }

    /// Every GPU's resources, for a fan-out across them, one each.
    ///
    /// # Panics
    /// If the step did not declare every GPU's lock.
    #[track_caller]
    pub fn all_gpus(&mut self) -> Vec<&mut GpuResources> {
        self.gpus
            .iter_mut()
            .map(|g| {
                &mut **g
                    .as_mut()
                    .expect("the step did not declare every GPU's lock")
            })
            .collect()
    }

    /// Whether the step holds `manager`'s lock.
    pub fn holds_manager(&self, manager: ManagerId) -> bool {
        let ManagerId::Eri = manager;
        self.manager.is_some()
    }
}

impl Bus for MainBus {
    type Shared = Shared;
    type Lane = Lane;
    type Hold<'a> = Held<'a>;

    fn shared(&self) -> &Shared {
        &self.shared
    }

    fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn make_lane(&self, step: &'static str, placement: Placement, streams: usize) -> Result<Lane> {
        let n = self.shared.num_gpus();
        let spans = match placement {
            Placement::Host => 0..0,
            Placement::Gpu0 => 0..1,
            Placement::AllGpus => 0..n,
        };
        for lock in self.ledger.locks_of(step) {
            if let LockId::Gpu(g) = lock
                && !spans.contains(&g)
            {
                return Err(Error::config(format!(
                    "step '{step}' declares the lock of gpu {g}, which it does not run on"
                )));
            }
        }
        if self.ledger.backing() != Backing::Real {
            return Ok(Lane { gpus: Vec::new() });
        }
        let mut gpus = Vec::new();
        for gpu in spans {
            let physical = self.shared.physical[gpu];
            gpus.push(GpuLane {
                gpu,
                physical,
                streams: (0..streams)
                    .map(|_| Stream::new(physical))
                    .collect::<Result<_>>()?,
            });
        }
        Ok(Lane { gpus })
    }

    fn acquire<'a>(
        &'a self,
        step: &'static str,
        placement: Placement,
        lane: &mut Lane,
        lease: &Lease<'_>,
    ) -> Result<Held<'a>> {
        let shared = &self.shared;
        // The node's thread gets gpu0; an all-GPU step's inner tasks make
        // their own device current.
        let current = (placement != Placement::Host).then(|| ScopedDevice::new(shared.physical[0]));
        let mut held = Held {
            _current: current,
            manager: None,
            gpus: (0..shared.num_gpus()).map(|_| None).collect(),
            taken: Vec::new(),
        };
        // The declared set, in rank order: managers, then GPUs by index.
        for lock in self.ledger.locks_of(step) {
            let asked = shared.epoch.elapsed();
            match lock {
                LockId::Manager(ManagerId::Eri) => {
                    let (guard, waited) = lock_noting(&shared.eri_lock);
                    held.manager = Some(guard);
                    held.taken
                        .push((lock, asked, shared.epoch.elapsed(), waited));
                }
                LockId::Gpu(g) => {
                    let (mut guard, waited) = lock_noting(&shared.gpus[g]);
                    held.taken
                        .push((lock, asked, shared.epoch.elapsed(), waited));
                    let (workspace, bytes) = self
                        .ledger
                        .workspace(g)
                        .expect("the final node allocated each GPU's handle workspace");
                    guard.bind_workspace(workspace, bytes)?;
                    held.gpus[g] = Some(guard);
                }
            }
        }
        if self.ledger.nan_fill() {
            lease.fill_host();
            for gpu in &lane.gpus {
                if let Some((ptr, bytes, false)) = lease.whole(gpu.gpu(), Kind::Device) {
                    // SAFETY: the step's own region on this GPU, its for
                    // the run that has not started.
                    let mut region = unsafe { PartMut::<u8>::from_raw(ptr, bytes) };
                    gpu.stream(0).fill(&mut region, 0xFF)?;
                    // The step's other streams must find the fill done.
                    gpu.stream(0).sync()?;
                }
            }
        }
        Ok(held)
    }

    fn release(&self, step: &'static str, lane: &Lane, hold: Held<'_>) -> Status {
        // Every stream the step has is synchronized, whatever the first
        // one says; then each held GPU's stream.
        let mut status = Ok(());
        let mut note = |synced: Status| {
            if status.is_ok() {
                status = synced;
            }
        };
        for gpu in &lane.gpus {
            for stream in &gpu.streams {
                note(stream.sync());
            }
        }
        for guard in hold.gpus.iter().flatten() {
            note(guard.stream().sync());
        }
        // Only now the locks: the next holder finds no call in flight.
        let released = self.shared.epoch.elapsed();
        let Held {
            gpus,
            manager,
            taken,
            ..
        } = hold;
        drop(gpus);
        drop(manager);
        let mut log = self
            .shared
            .acquisitions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (lock, asked, locked, waited) in taken {
            log.push(Acquisition {
                step,
                lock,
                asked,
                locked,
                released,
                waited,
            });
        }
        status
    }
}

/// Take `mutex`, noting whether another holder had it.
fn lock_noting<T>(mutex: &Mutex<T>) -> (MutexGuard<'_, T>, bool) {
    match mutex.try_lock() {
        Ok(guard) => (guard, false),
        Err(std::sync::TryLockError::Poisoned(e)) => (e.into_inner(), false),
        Err(std::sync::TryLockError::WouldBlock) => {
            (mutex.lock().unwrap_or_else(|e| e.into_inner()), true)
        }
    }
}
