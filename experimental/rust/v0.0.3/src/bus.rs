//! The MainBus: the config, what setup derives from it, the ledger, the
//! cards' resources, and one slot per step. A deliberate god object; a
//! step's `run` is handed a view of it, never the bus.
//!
//! Setup is the unbudgeted host phase: it parses the config, the geometry
//! and the bases, and counts every length a step's arrays need into
//! [`Dims`], so that each `mem_budget` is a function of the dimensions, the
//! step's options and the caps.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use fb_basis::BasisSet;
use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_integrals_sys::RecordTables;
use fb_molecule::Molecule;
use serde::Deserialize;

use crate::budget::{Backing, Cap, Caps, Kind, Lease, Ledger, device_margin, host_margin};
use crate::device::{CardLane, CardResources, Lane};
use crate::flow::{Bus, Placement, Slot};
use crate::steps::eri::EriOutput;
use crate::steps::guess::GuessOutput;
use crate::steps::jk::JkOutput;
use crate::steps::records::RecordsOutput;

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
    /// The J and K build over the guess density; absent, no `jk` step.
    pub jk: Option<JkTable>,
    /// Cards, threads and caps.
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
    /// The step's own limit on each card's device scratch, MiB: it picks
    /// the batch width. Absent, a fraction of the card's cap.
    pub limit_mib: Option<f64>,
}

/// `[guess]`: SAD, the only kind.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuessTable {}

/// `[jk]`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JkTable {
    /// The step's own limit on each card's device scratch, MiB.
    pub limit_mib: Option<f64>,
}

/// `[resources]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResourcesTable {
    /// The cards of the run, each by the device it is on; the first is
    /// gpu0, the privileged one, and `num_gpus` is the list's length. A
    /// device named twice is two logical cards on one physical card, each
    /// with its own lock, workspace, streams and scratch slabs: how the
    /// multi-card path is run on a one-card box.
    pub devices: Vec<i32>,
    /// The executor's width.
    pub threads: usize,
    /// The device cap per card, MiB; absent, the card's free memory at
    /// setup less its margin.
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
    /// A pair's Cartesian columns.
    pub fn cart(&self) -> usize {
        n_cart_of(self.la) * n_cart_of(self.lb)
    }

    /// A pair's spherical columns.
    pub fn sph(&self) -> usize {
        n_sph_of(self.la) * n_sph_of(self.lb)
    }
}

/// The Cartesian functions of a shell of momentum `l`.
pub fn n_cart_of(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
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
    /// Cartesian fitting functions.
    pub n_aux_cart: usize,
    /// Fitting shells.
    pub n_aux_shell: usize,
    /// Fitting primitives: the aux records.
    pub n_aux_prim: usize,
    /// The fitting shells of each momentum.
    pub aux_shells_of_l: Vec<usize>,
}

/// One acquisition of a card by a step, for the lock-order test and the
/// timeline: when the wrapper had the card's lock, and when it let go.
#[derive(Clone, Debug)]
pub struct Held {
    /// The step.
    pub step: &'static str,
    /// The card.
    pub card: usize,
    /// When the lock was taken, from setup's end.
    pub locked: Duration,
    /// When it was released, after the step-end sync.
    pub released: Duration,
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
    /// The orbital basis's shell and class tables, on the host. Setup
    /// builds them to count from; README.md says why that is a finding.
    pub tables: RecordTables,
    /// The dimensions.
    pub dims: Dims,
    /// One per card: its device and its lock.
    pub cards: Vec<CardResources>,
    epoch: Instant,
    held: Mutex<Vec<Held>>,
}

impl Shared {
    /// Every acquisition so far, in the order the locks were released.
    pub fn held(&self) -> Vec<Held> {
        self.held.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// The run's one shared object.
pub struct MainBus {
    shared: Shared,
    ledger: Ledger,
    pub(crate) records: Slot<RecordsOutput, Lane>,
    pub(crate) guess: Slot<GuessOutput, Lane>,
    pub(crate) eri: Slot<EriOutput, Lane>,
    pub(crate) jk: Slot<JkOutput, Lane>,
}

impl MainBus {
    /// Setup: parse the geometry and the bases on the host, count the
    /// dimensions, make each card's lock, and read and clamp the caps. Not
    /// budgeted. With [`Backing::Dry`] no device is touched: the plan can
    /// be budgeted and refused or accepted, not run.
    ///
    /// # Errors
    /// The readers' refusals; a config with no card or no thread; a device
    /// wwr does not see.
    pub fn setup(config: Config, backing: Backing) -> Result<Arc<MainBus>> {
        let molecule = fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1)?;
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
        let classes: Vec<ClassDim> = tables
            .classes()
            .iter()
            .map(|c| ClassDim {
                la: c.la,
                lb: c.lb,
                n_pair: c.bra.len(),
                n_rec: c.n_rec() as usize,
            })
            .collect();
        let mut aux_shells_of_l = Vec::new();
        for shell in aux.iter().flat_map(|a| a.shells()) {
            let l = shell.angular_momentum() as usize;
            if aux_shells_of_l.len() <= l {
                aux_shells_of_l.resize(l + 1, 0);
            }
            aux_shells_of_l[l] += 1;
        }
        let shells = tables.shells();
        let dims = Dims {
            n_ao: basis.nao_sphe() as usize,
            n_shell: shells.ao_offset.len(),
            n_prim: shells.exponent.len(),
            n_shell_pair: tables.n_pairs(),
            n_col: classes.iter().map(|c| c.n_pair * c.sph()).sum(),
            classes,
            n_aux: aux.as_ref().map_or(0, |a| a.nao_sphe() as usize),
            n_aux_cart: aux.as_ref().map_or(0, |a| a.nao_cart() as usize),
            n_aux_shell: aux.as_ref().map_or(0, |a| a.nshells() as usize),
            n_aux_prim: aux.as_ref().map_or(0, |a| {
                a.shells().iter().map(|s| s.nprimitives() as usize).sum()
            }),
            aux_shells_of_l,
        };

        let res = &config.resources;
        if res.devices.is_empty() || res.threads == 0 {
            return Err(Error::config(
                "[resources] devices names a card and threads is at least 1",
            ));
        }
        let real = backing == Backing::Real;
        if real {
            fb_recipes_sys::check_library().map_err(Error::gpu)?;
            fb_integrals_sys::check_library().map_err(Error::gpu)?;
            let seen = fb_core_wwr::runtime::device_count()? as i32;
            if let Some(d) = res.devices.iter().find(|&&d| d < 0 || d >= seen) {
                return Err(Error::config(format!(
                    "[resources] devices names device {d}, but {seen} are visible"
                )));
            }
        }
        // Each card's cap, clamped to what its device has now. Logical
        // cards on one device share its reading and its margin evenly.
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
        let available = fb_core_arena::host_available()? as u64;
        let host = Cap::resolve(
            res.host_cap_mib.map(|mib| mib << 20),
            Some(available),
            host_margin(available),
        );
        let cards = res
            .devices
            .iter()
            .map(|&physical| CardResources {
                physical,
                lock: Mutex::new(()),
            })
            .collect();
        let ledger = Ledger::new(
            Caps { device, host },
            backing,
            res.devices.clone(),
            res.nan_fill,
        );
        Ok(Arc::new(MainBus {
            shared: Shared {
                config,
                molecule,
                basis,
                aux,
                tables,
                dims,
                cards,
                epoch: Instant::now(),
                held: Mutex::new(Vec::new()),
            },
            ledger,
            records: Slot::new(),
            guess: Slot::new(),
            eri: Slot::new(),
            jk: Slot::new(),
        }))
    }
}

/// What the wrapper holds around a `run`: gpu0 current on the node's
/// thread, and the locks of the cards the step calls the library on.
pub struct Hold<'a> {
    _current: Option<ScopedDeviceIndex>,
    locks: Vec<(usize, Duration, MutexGuard<'a, ()>)>,
}

impl Bus for MainBus {
    type Shared = Shared;
    type Lane = Lane;
    type Hold<'a> = Hold<'a>;

    fn shared(&self) -> &Shared {
        &self.shared
    }

    fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn make_lane(&self, step: &'static str, placement: Placement) -> Result<Lane> {
        let library = self.ledger.library_cards(step);
        let cards = match placement {
            Placement::Host => 0..0,
            Placement::Gpu0 => 0..1,
            Placement::AllGpus => 0..self.shared.cards.len(),
        };
        if let Some(c) = library.iter().find(|c| !cards.contains(c)) {
            return Err(Error::config(format!(
                "step '{step}' declares the library on card {c}, which it does not run on"
            )));
        }
        if self.ledger.backing() != Backing::Real {
            return Ok(Lane { cards: Vec::new() });
        }
        Ok(Lane {
            cards: cards
                .map(|c| CardLane::new(c, self.shared.cards[c].physical, library.contains(&c)))
                .collect(),
        })
    }

    fn acquire<'a>(
        &'a self,
        step: &'static str,
        placement: Placement,
        lane: &mut Lane,
        lease: &Lease<'_>,
    ) -> Result<Hold<'a>> {
        // The node's thread gets gpu0; an all-GPU step's inner tasks make
        // their own card current.
        let current = (placement != Placement::Host)
            .then(|| ScopedDeviceIndex::new(self.shared.cards[0].physical));
        // The locks, in card order: two multi-card steps cannot deadlock.
        let mut locks = Vec::new();
        for card in lane.cards.iter().filter(|c| c.calls_library()) {
            let guard = self.shared.cards[card.card()]
                .lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            locks.push((card.card(), self.shared.epoch.elapsed(), guard));
            let (workspace, bytes) = self
                .ledger
                .workspace(card.card())
                .expect("the final node allocated each card's workspace");
            card.bind(workspace, bytes)?;
        }
        if self.ledger.nan_fill() {
            lease.fill_host();
            for card in &lane.cards {
                if let Some((ptr, bytes, false)) = lease.whole(card.card(), Kind::Device) {
                    card.fill(ptr, bytes)?;
                }
            }
        }
        let _ = step;
        Ok(Hold {
            _current: current,
            locks,
        })
    }

    fn release(&self, step: &'static str, lane: &Lane, hold: Hold<'_>) -> Status {
        // Every card is synchronized, whatever the first one says.
        let mut status = Ok(());
        for card in &lane.cards {
            let synced = card.sync();
            if status.is_ok() {
                status = synced;
            }
        }
        // Only now the locks: the next holder finds no call in flight.
        let released = self.shared.epoch.elapsed();
        let mut held = self.shared.held.lock().unwrap_or_else(|e| e.into_inner());
        for (card, locked, guard) in hold.locks {
            drop(guard);
            held.push(Held {
                step,
                card,
                locked,
                released,
            });
        }
        status
    }
}
