//! The MainBus: the config, what setup derives from it, the ledger, the
//! locks (each GPU's `GpuResources` and each manager's lock), and one slot
//! per step. A deliberate god object; a step's `run` is handed a View of
//! it and the guards the wrapper took, never the bus.
//!
//! Setup is the unbudgeted host phase: it parses the config, the geometry
//! and the bases, and **counts** every length a step's arrays need into
//! [`Dims`] by a walk over the shells that builds no array
//! ([`for_each_pair`]). Each `mem_budget` is then a function of the
//! dimensions, the step's options and the caps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fb_basis::BasisSet;
use fb_core_budget::{Backing, Budget, Caps, Ledger};
use fb_core_error_handling::{Error, Result};
use fb_core_flow::{Bus, Locks, Slot};
use fb_molecule::Molecule;
use serde::Deserialize;

use crate::steps::core_hamiltonian::CoreHamiltonianOutput;
use crate::steps::guess::GuessOutput;
use crate::steps::metric::MetricOutput;
use crate::steps::orthogonalizer::OrthogonalizerOutput;
use crate::steps::overlap::OverlapOutput;

/// The config file: one table per step family, plus the run's resources.
/// A step whose table is absent is not in the DAG.
///
/// ```
/// use std::path::Path;
/// use fb_steps::Config;
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n";
/// let config = Config::parse(text, Path::new("/data")).unwrap();
/// assert_eq!(config.molecule.xyz, Path::new("/data/h2o.xyz"));
/// assert!(config.eri.is_none());
/// assert_eq!(config.resources.devices, [0]);
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The geometry.
    pub molecule: MoleculeTable,
    /// The orbital basis.
    pub basis: BasisTable,
    /// The density-fitted integrals; absent, no `metric` step.
    pub eri: Option<EriTable>,
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
    /// The fitting set, a compiled-in name or a BSE JSON path.
    pub aux_basis: String,
}

/// `[resources]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResourcesTable {
    /// The GPUs of the run, each by the device it is on; the first is
    /// gpu0, and the run's GPU count is the list's length. A device named
    /// twice is two logical GPUs on one physical device, a test facility.
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
    #[allow(
        clippy::disallowed_macros,
        reason = "the config's default device list, one entry, made before any step"
    )]
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
    /// Parse `text`, resolving a relative `[molecule] xyz` against `dir`.
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

    /// Read and parse the config file at `path`.
    ///
    /// # Errors
    /// `Io` if it cannot be read; [`Config::parse`]'s.
    pub fn read(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::io(format!("cannot read '{}': {e}", path.display())))?;
        Config::parse(&text, path.parent().unwrap_or(Path::new(".")))
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
    #[must_use]
    pub fn sph(&self) -> usize {
        n_sph_of(self.la) * n_sph_of(self.lb)
    }
}

/// The spherical functions of a shell of momentum `l`.
fn n_sph_of(l: i32) -> usize {
    (2 * l + 1).unsigned_abs() as usize
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
    /// The spherical pair columns of the packed pair layout, unscreened:
    /// the bound B is budgeted at.
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
    /// Count the dimensions of `basis`, `aux` and a closed shell of
    /// `electrons` electrons, building no array.
    #[must_use]
    pub fn count(basis: &BasisSet, aux: Option<&BasisSet>, electrons: usize) -> Dims {
        let classes = count_classes(basis);
        let mut aux_shells_of_l = Vec::new();
        for shell in aux.iter().flat_map(|a| a.shells()) {
            let l = shell.angular_momentum().unsigned_abs() as usize;
            if aux_shells_of_l.len() <= l {
                #[allow(
                    clippy::disallowed_methods,
                    reason = "setup, the unbudgeted host phase: one count per angular momentum"
                )]
                aux_shells_of_l.resize(l + 1, 0);
            }
            aux_shells_of_l[l] += 1;
        }
        let prims =
            |b: &BasisSet| -> usize { b.shells().iter().map(|s| s.nprimitives() as usize).sum() };
        let n_shell = basis.nshells() as usize;
        Dims {
            n_ao: basis.nao_sphe() as usize,
            n_occ: electrons / 2,
            n_shell,
            n_prim: prims(basis),
            n_shell_pair: n_shell * (n_shell + 1) / 2,
            n_col: classes.iter().map(|c| c.n_pair * c.sph()).sum(),
            classes,
            n_aux: aux.map_or(0, |a| a.nao_sphe() as usize),
            n_aux_shell: aux.map_or(0, |a| a.nshells() as usize),
            n_aux_prim: aux.map_or(0, prims),
            aux_shells_of_l,
        }
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
    locks: Locks,
}

impl Shared {
    /// The GPUs of the run, whether or not a device is behind them.
    #[must_use]
    pub fn num_gpus(&self) -> usize {
        self.config.resources.devices.len()
    }

    /// syevd's work buffer at order `n` on gpu0, in doubles: a size query
    /// under gpu0's lock, which needs the live handle, so a plan with no
    /// device behind it reports 0 (`--budget-only`).
    ///
    /// # Errors
    /// The query's failure.
    pub fn syevd_lwork(&self, budget: &Budget<'_>, n: usize) -> Result<usize> {
        if self.locks.physical().is_empty() {
            return Ok(0);
        }
        self.locks.query(budget, 0, |gpu0| gpu0.syevd_lwork(n))
    }
}

/// The run's one shared object.
pub struct MainBus {
    shared: Shared,
    ledger: Ledger,
    pub(crate) overlap: Slot<OverlapOutput>,
    pub(crate) core_hamiltonian: Slot<CoreHamiltonianOutput>,
    pub(crate) metric: Slot<MetricOutput>,
    pub(crate) guess: Slot<GuessOutput>,
    pub(crate) orthogonalizer: Slot<OrthogonalizerOutput>,
}

impl MainBus {
    /// Setup: parse the geometry and the bases on the host, count the
    /// dimensions, make each GPU's `GpuResources`, and read and clamp the
    /// caps (a GPU's after its handles exist). Not budgeted.
    ///
    /// [`Backing::Real`] runs on the devices `[resources] devices` names.
    /// [`Backing::Host`] and [`Backing::Dry`] touch no device: a
    /// host-backed plan runs its host steps into heap "device" buffers, and
    /// a dry one can be budgeted and refused or accepted, not run, each
    /// solver work buffer counted as zero.
    ///
    /// # Errors
    /// The readers' refusals; an odd electron count (`Unsupported`); a
    /// config with no GPU or no thread, or a device the runtime does not
    /// see (`InvalidConfig`); the recipes library's mismatch; a cap's
    /// reading.
    pub fn setup(config: Config, backing: Backing) -> Result<Arc<MainBus>> {
        let molecule = fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1)?;
        let electrons = molecule.electron_count().unsigned_abs() as usize;
        if !electrons.is_multiple_of(2) {
            return Err(Error::unsupported(
                "the steps run a closed-shell neutral molecule only",
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
        let dims = Dims::count(&basis, aux.as_ref(), electrons);
        let e_nuc = fb_molecule::nuclear_repulsion_energy(&molecule);

        let res = &config.resources;
        if res.devices.is_empty() || res.threads == 0 {
            return Err(Error::config(
                "[resources] devices names a GPU and threads is at least 1",
            ));
        }
        let real = backing == Backing::Real;
        if real {
            fb_recipes_sys::check_library().map_err(Error::gpu)?;
            let seen = fb_core_device::device_count()?;
            if let Some(d) = res.devices.iter().find(|&&d| d < 0 || d >= seen) {
                return Err(Error::config(format!(
                    "[resources] devices names device {d}, but {seen} are visible"
                )));
            }
        }
        // The handles first: a defaulted cap then already sees them.
        let locks = Locks::new(if real { &res.devices } else { &[] });
        let written = res.device_cap_mib.map(|mib| mib << 20);
        #[allow(
            clippy::disallowed_macros,
            reason = "setup, the unbudgeted host phase: one written cap per GPU"
        )]
        let caps = Caps::read(
            &res.devices,
            &vec![written; res.devices.len()],
            res.host_cap_mib.map(|mib| mib << 20),
            backing,
        )?;
        let ledger = Ledger::new(caps, backing, res.devices.clone(), res.nan_fill);
        Ok(Arc::new(MainBus {
            shared: Shared {
                config,
                molecule,
                basis,
                aux,
                dims,
                e_nuc,
                locks,
            },
            ledger,
            overlap: Slot::new(),
            core_hamiltonian: Slot::new(),
            metric: Slot::new(),
            guess: Slot::new(),
            orthogonalizer: Slot::new(),
        }))
    }
}

impl Bus for MainBus {
    type Shared = Shared;

    fn shared(&self) -> &Shared {
        &self.shared
    }

    fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn locks(&self) -> &Locks {
        &self.shared.locks
    }
}
