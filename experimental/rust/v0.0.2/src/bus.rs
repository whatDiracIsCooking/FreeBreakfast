//! The MainBus: the config, what setup derives from it, the ledger, the
//! device resources, and one slot per step. A deliberate god object; a
//! step's `run` is handed a view of it, never the bus.
//!
//! Setup is the unbudgeted host phase: it parses the config, the geometry
//! and the bases, and fills the dimensions every `mem_budget` is a function
//! of.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_integrals_sys::RecordTables;
use fb_molecule::Molecule;
use serde::Deserialize;

use crate::budget::{Backing, Caps, Ledger};
use crate::device::{DeviceResources, ForkStream};
use crate::flow::{Bus, Placement, Slot};
use crate::steps::core_hamiltonian::CoreHamiltonianOutput;
use crate::steps::eri::EriManager;
use crate::steps::guess::GuessOutput;
use crate::steps::harmonics::HarmonicsOutput;
use crate::steps::orthogonalizer::OrthogonalizerOutput;
use crate::steps::overlap::OverlapOutput;
use crate::steps::records::RecordsOutput;
use crate::steps::scf::ScfOutput;

/// What a defaulted device cap leaves free: for the memory taken after
/// budget mode, the kernels' modules and stack.
pub const DEVICE_MARGIN: u64 = 768 << 20;

/// The config file: one table per step, plus the run's resources. A step
/// whose table is absent is not in the DAG.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The geometry.
    pub molecule: MoleculeTable,
    /// The orbital basis.
    pub basis: BasisTable,
    /// The density-fitted integrals; absent, no `eri` step.
    pub eri: Option<EriTable>,
    /// The initial guess; absent, no `guess` step.
    pub guess: Option<GuessTable>,
    /// The SCF; absent, no `scf` step.
    pub scf: Option<ScfTable>,
    /// Devices, threads and caps.
    #[serde(default)]
    pub resources: ResourcesTable,
}

/// `[molecule]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeTable {
    /// The XYZ file, relative to the config's directory.
    pub xyz: PathBuf,
    /// The charge.
    #[serde(default)]
    pub charge: i32,
    /// The spin multiplicity.
    #[serde(default = "one")]
    pub multiplicity: i32,
}

fn one() -> i32 {
    1
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
}

/// `[guess]`: SAD, the only kind.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuessTable {}

/// `[scf]`: the old step's options and defaults.
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
    /// The cards the run uses; gpu0 is the privileged one.
    pub num_gpus: usize,
    /// The executor's width.
    pub threads: usize,
    /// The device cap per card, MiB; absent, the card's free memory at
    /// setup less [`DEVICE_MARGIN`].
    pub device_cap_mib: Option<u64>,
    /// The pinned cap per card, MiB.
    pub pinned_cap_mib: u64,
    /// The budgeted heap's cap, MiB; absent, half of what the host has.
    pub heap_cap_mib: Option<u64>,
}

impl Default for ResourcesTable {
    fn default() -> Self {
        ResourcesTable {
            num_gpus: 1,
            threads: 4,
            device_cap_mib: None,
            pinned_cap_mib: 1024,
            heap_cap_mib: None,
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

/// The dimensions: what every `mem_budget` is a function of, with the
/// step's own options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// Atoms.
    pub n_atom: usize,
    /// Spherical AOs.
    pub n_ao: usize,
    /// Cartesian AOs.
    pub n_cart: usize,
    /// Doubly occupied orbitals.
    pub n_occ: usize,
    /// Spherical fitting functions; 0 with no `[eri]`.
    pub n_aux: usize,
    /// Cartesian fitting functions.
    pub n_aux_cart: usize,
    /// The Cartesian pair columns of the tree's own pair layout, unscreened:
    /// the bound (mn|P) is budgeted at.
    pub n_col: usize,
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
    /// The orbital basis's shell and class tables, on the host.
    pub tables: RecordTables,
    /// The dimensions.
    pub dims: Dims,
    /// The nuclear repulsion energy, hartree.
    pub e_nuc: f64,
    /// One per device: the library stream, handles and workspace.
    pub resources: Vec<Mutex<DeviceResources>>,
}

impl Shared {
    /// The device's library resources.
    pub fn library(&self, device: usize) -> &Mutex<DeviceResources> {
        &self.resources[device]
    }
}

/// What the wrapper holds for a step: its streams, one per device it spans;
/// none for a host step.
pub struct Lane(Vec<ForkStream>);

impl Lane {
    /// The step's stream on gpu0, or on its one device.
    ///
    /// # Panics
    /// For a host step, which has no stream.
    pub fn gpu(&self) -> &ForkStream {
        self.0.first().expect("a host step has no stream")
    }

    /// Every stream of the step, one per device it spans.
    pub fn streams(&self) -> &[ForkStream] {
        &self.0
    }
}

/// The run's one shared object.
pub struct MainBus {
    shared: Shared,
    ledger: Ledger,
    pub(crate) harmonics: Slot<HarmonicsOutput, Lane>,
    pub(crate) records: Slot<RecordsOutput, Lane>,
    pub(crate) overlap: Slot<OverlapOutput, Lane>,
    pub(crate) core_hamiltonian: Slot<CoreHamiltonianOutput, Lane>,
    pub(crate) orthogonalizer: Slot<OrthogonalizerOutput, Lane>,
    pub(crate) eri: Slot<EriManager, Lane>,
    pub(crate) guess: Slot<GuessOutput, Lane>,
    pub(crate) scf: Slot<ScfOutput, Lane>,
}

impl MainBus {
    /// Setup: parse the geometry and the bases on the host, fill the
    /// dimensions, make each device's resources, and read the caps. Not
    /// budgeted. With [`Backing::Dry`] no device is touched: the plan can be
    /// budgeted and refused or accepted, not run.
    ///
    /// # Errors
    /// The readers' refusals; an open shell or a charged molecule
    /// (`Unsupported`: the prototype is closed-shell RHF of a neutral);
    /// more cards asked for than wwr sees.
    pub fn setup(config: Config, backing: Backing) -> Result<Arc<MainBus>> {
        let table = &config.molecule;
        let molecule = fb_io_xyz::read_xyz(&table.xyz, table.charge, table.multiplicity)?;
        let electrons = molecule.electron_count();
        if table.multiplicity != 1 || electrons % 2 != 0 || table.charge != 0 {
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
        let tables = fb_integrals_mmd::record_tables(&basis);
        let n_cart_of = |l: i32| ((l + 1) * (l + 2) / 2) as usize;
        let n_col = tables
            .classes()
            .iter()
            .map(|c| c.bra.len() * n_cart_of(c.la) * n_cart_of(c.lb))
            .sum();
        let dims = Dims {
            n_atom: molecule.natoms() as usize,
            n_ao: basis.nao_sphe() as usize,
            n_cart: basis.nao_cart() as usize,
            n_occ: (electrons / 2) as usize,
            n_aux: aux.as_ref().map_or(0, |a| a.nao_sphe() as usize),
            n_aux_cart: aux.as_ref().map_or(0, |a| a.nao_cart() as usize),
            n_col,
        };
        let e_nuc = fb_molecule::nuclear_repulsion_energy(&molecule);

        let res = &config.resources;
        if res.num_gpus == 0 || res.threads == 0 {
            return Err(Error::config(
                "[resources] num_gpus and threads are at least 1",
            ));
        }
        let real = backing == Backing::Real;
        let mut resources = Vec::new();
        let mut device_caps = Vec::new();
        if real {
            fb_recipes_sys::check_library().map_err(Error::gpu)?;
            fb_integrals_sys::check_library().map_err(Error::gpu)?;
            let seen = fb_core_wwr::runtime::device_count()?;
            if res.num_gpus > seen {
                return Err(Error::config(format!(
                    "[resources] num_gpus = {}, but {seen} are visible",
                    res.num_gpus
                )));
            }
        }
        for device in 0..res.num_gpus {
            let cap = match res.device_cap_mib {
                // An explicit cap is used as written: the user owns the margin.
                Some(mib) => mib << 20,
                None if real => crate::device::free_memory(device)?.saturating_sub(DEVICE_MARGIN),
                None => u64::MAX,
            };
            device_caps.push(cap);
            if real {
                resources.push(Mutex::new(DeviceResources::new(device)));
            }
        }
        let host_available = fb_core_arena::host_available()? as u64;
        let caps = Caps {
            device: device_caps,
            pinned: vec![res.pinned_cap_mib << 20; res.num_gpus],
            heap: res.heap_cap_mib.map_or(host_available / 2, |mib| mib << 20),
            host_available,
        };
        Ok(Arc::new(MainBus {
            shared: Shared {
                config,
                molecule,
                basis,
                aux,
                tables,
                dims,
                e_nuc,
                resources,
            },
            ledger: Ledger::new(caps, backing),
            harmonics: Slot::new(),
            records: Slot::new(),
            overlap: Slot::new(),
            core_hamiltonian: Slot::new(),
            orthogonalizer: Slot::new(),
            eri: Slot::new(),
            guess: Slot::new(),
            scf: Slot::new(),
        }))
    }
}

impl Bus for MainBus {
    type Shared = Shared;
    type Lane = Lane;
    type Guard = Option<ScopedDeviceIndex>;

    fn shared(&self) -> &Shared {
        &self.shared
    }

    fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn enter(&self, placement: Placement) -> Option<ScopedDeviceIndex> {
        if self.ledger.backing() == Backing::Dry {
            return None;
        }
        match placement {
            Placement::Host => None,
            Placement::Gpu(device) => Some(ScopedDeviceIndex::new(device as i32)),
            // The privileged card: a multi-GPU step's inner tasks make their
            // own card current.
            Placement::AllGpus => Some(ScopedDeviceIndex::new(0)),
        }
    }

    fn make_lane(&self, placement: Placement) -> Result<Lane> {
        if self.ledger.backing() == Backing::Dry {
            return Ok(Lane(Vec::new()));
        }
        Ok(Lane(match placement {
            Placement::Host => Vec::new(),
            Placement::Gpu(device) => vec![ForkStream::new(device)],
            Placement::AllGpus => (0..self.shared.resources.len())
                .map(|device| {
                    let _current = ScopedDeviceIndex::new(device as i32);
                    ForkStream::new(device)
                })
                .collect(),
        }))
    }

    fn leave(&self, lane: &Lane) -> Status {
        lane.streams().iter().try_for_each(ForkStream::sync)
    }

    fn bind_workspace(&self, device: usize, bytes: u64) -> Status {
        self.shared.resources[device]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bind_workspace(bytes as usize);
        Ok(())
    }
}
