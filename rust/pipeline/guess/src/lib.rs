//! `src/pipeline/guess` (module `free_breakfast.pipeline.guess`): the
//! initial-guess density D, spin-summed, `nao_sphe` square, column-major,
//! in the spherical layout, as tasks of a [`Dag`]. SAD is the only kind
//! built, beside a guess read back from a `.npy` file.
//!
//! In the model the atomic SCFs that the C++ step spreads over a Taskflow
//! subflow are CPU nodes the planner sees, never a pool inside a task
//! (`docs/design.md`, "What it holds"). The distinct (Z, shells) stamps are
//! known from the basis at planning, so [`add_sad_guess`] adds:
//!
//! - **one CPU node per distinct stamp**, in order of each stamp's first
//!   atom ([`fb_sad::distinct_stamps`]), its block written into a host row
//!   of its own (`stamp <symbol>`). The node is `baked stamp <symbol>` when
//!   the basis carries a name ([`BasisSet::name`], set only for a compiled-in
//!   set) and `fb-io-sad` has the element ([`builtin_sad_density`]), which
//!   it parses; `atomic scf <symbol>` otherwise, which runs `fb-sad`'s
//!   atomic SCF. Either way through [`fb_sad::stamp_density`], so the block
//!   is bitwise the serial [`fb_sad::sad_density`]'s. A basis read from a
//!   path carries no name, so it runs every SCF even if its document is a
//!   carried set, as in the C++.
//! - **`sad guess`**, on the CPU after every stamp: D zeroed, and each
//!   stamp's block placed on the diagonal block of every atom that wears
//!   it.
//!
//! [`add_loaded_guess`] is the step's `load`: one CPU node, `load guess`,
//! that reads D from a `.npy` file in place of the stamps. Either row is a
//! host row; [`add_guess_upload`] adds `upload guess`, the copy on the
//! worker's stream that makes it the device output row the SCF reads.
//!
//! **Errors.** A basis not placed over these atoms is refused at planning.
//! Every other failure is a node's, at run time. The stamps are independent,
//! so every one runs; the run's error is the failure lowest in rank
//! ([`fb_core_thread_pool::run`]), which is the first failing stamp in stamp
//! order, the serial form's, and `sad guess`, their descendant, is skipped.
//!
//! The two guess functions are generic over the host kind and the stream, so
//! the device-free tests run them on `Heap` rows and `HostOnly` workers;
//! [`add_guess_upload`] is the production kinds' alone.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `GuessType`, `GuessStep::type` | nothing yet: SAD is the only kind, and the config that would carry it is the binary's |
//! | `GuessStep::run(in, subflow)`, over `sad_density(flow, …, builtin_sad_density)` | [`add_sad_guess`]: a node per stamp and `sad guess` |
//! | `sad_density`'s task form: one task, a subtask per distinct stamp, its explicit join | the stamp nodes and `sad guess`, ordered by the rows they write and read |
//! | `GuessStep::load(from, in)` | [`add_loaded_guess`], whose `load guess` node reads the file when it runs |
//! | `GuessOutput`, a host `std::vector<double>` the first device reader uploads | the host row `D`, and [`add_guess_upload`]'s device row `D` |
//! | `GuessStep::footprint`, nothing | the rows' sizes in the plan's tables |

use std::path::PathBuf;

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, Device, HostKind, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_io_sad::builtin_sad_density;
use fb_sad::{AtomicDensityStamp, SadStampLookup, distinct_stamps, stamp_density};
use wwr_sys::wwr_memcpy_kind;

/// Add the SAD guess of `atoms` in `basis` to `dag` on GPU worker `worker`,
/// and return D: a host output row of `nao_sphe²` doubles, column-major.
///
/// `basis` is placed over `atoms`, in the same order
/// ([`fb_basis::BasisStamp::place`]). One CPU node per distinct (Z, shells),
/// `baked stamp <symbol>` or `atomic scf <symbol>`, then `sad guess`; the
/// crate's doc has which is which. Each node owns what it reads, so the plan
/// outlives `basis` and `atoms`.
///
/// # Errors
/// `InvalidConfig`, before anything is added, if `basis` is not grouped
/// over as many atoms as `atoms` holds ([`distinct_stamps`]). At run time,
/// the first failing stamp's error ([`fb_sad::sad_density`]'s, for that
/// atom).
pub fn add_sad_guess<'a, H: HostKind, S: 'a>(
    dag: &mut Dag<'a, H, S>,
    worker: usize,
    basis: &BasisSet,
    atoms: &[Atom],
) -> Result<RowId<H, f64>> {
    let sites = distinct_stamps(basis, atoms)?;
    let name = basis.name();
    let mut stamps = Vec::with_capacity(sites.len());
    for site in sites {
        let z = site.stamp.z;
        let symbol = Element::from_z(z as u8).map_or("?", Element::symbol);
        let baked = !name.is_empty() && builtin_sad_density(name, z).is_some();
        let block: usize = site
            .stamp
            .shells
            .iter()
            .map(|shell| shell.function_count_sphe() as usize)
            .sum();
        let row = dag.add(
            if baked {
                format!("baked stamp {symbol}")
            } else {
                format!("atomic scf {symbol}")
            },
            worker,
            Stamp {
                z,
                shells: site.stamp.shells.to_vec(),
                basis: name.to_owned(),
                baked,
                row: format!("stamp {symbol}"),
                len: block * block,
            },
        );
        stamps.push(Site {
            row,
            block,
            offsets: site.offsets,
        });
    }
    Ok(dag.add(
        "sad guess",
        worker,
        Assemble {
            stamps,
            n: basis.nao_sphe() as usize,
        },
    ))
}

/// Add a guess read back from the `.npy` file `from` to `dag` on GPU worker
/// `worker`, in place of [`add_sad_guess`], and return D: a host output row
/// of `nao_sphe²` doubles, column-major, as [`add_sad_guess`] returns it.
///
/// One CPU node, `load guess`, which reads the file when it runs: a float64
/// array of `basis`'s spherical AO order, in either layout. The file carries
/// only the numbers, so only the shape is checked: a basis of the same size
/// but another order is not caught.
///
/// # Errors
/// At run time: `Io` for a file that cannot be read or parsed
/// ([`fb_io_npy::read_npy`]), `InvalidConfig` for a shape other than
/// `nao_sphe × nao_sphe`.
pub fn add_loaded_guess<'a, H: HostKind, S: 'a>(
    dag: &mut Dag<'a, H, S>,
    worker: usize,
    basis: &BasisSet,
    from: PathBuf,
) -> RowId<H, f64> {
    dag.add(
        "load guess",
        worker,
        Load {
            from,
            n: basis.nao_sphe() as usize,
        },
    )
}

/// Add `upload guess` to `dag` on GPU worker `worker`: the host guess `d`
/// of [`add_sad_guess`] or [`add_loaded_guess`] over `basis` copied on the
/// worker's stream into a device output row `D`, which it returns, for the
/// SCF to read.
///
/// # Panics
/// If `d` is a row of another worker's table ([`Dag::add`]).
pub fn add_guess_upload(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    d: RowId<Pinned, f64>,
) -> RowId<Device, f64> {
    let n = basis.nao_sphe() as usize;
    dag.add("upload guess", worker, Upload { d, len: n * n })
}

/// CPU: one distinct (Z, shells)'s block, its SCF's or its baked one, into
/// its host row.
struct Stamp {
    z: i32,
    shells: Vec<Shell>,
    /// The basis's name, empty for one read from a path.
    basis: String,
    /// Whether `fb-io-sad` has the block, decided at planning.
    baked: bool,
    /// The row's name and length.
    row: String,
    len: usize,
}

impl<H: HostKind, S> Task<H, S> for Stamp {
    type Rows = RowId<H, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) -> Self::Rows {
        host.output(&self.row, self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, H>,
        _: Lane<'_, S>,
    ) -> Status {
        let stamp = AtomicDensityStamp {
            z: self.z,
            shells: &self.shells,
            density: Vec::new(),
        };
        let lookup: Option<SadStampLookup> = self.baked.then_some(builtin_sad_density);
        // stamp_density checks a baked block's size against the shells.
        let density = stamp_density(&stamp, &self.basis, lookup)?;
        host.take_mut(*row).copy_from_slice(&density);
        Ok(())
    }
}

/// One stamp's row, its order and the first AO of each atom that wears it.
struct Site<H: HostKind> {
    row: RowId<H, f64>,
    block: usize,
    offsets: Vec<usize>,
}

/// CPU: D zeroed, then every stamp's block on its atoms' diagonal blocks,
/// in stamp order, as the serial form places them.
struct Assemble<H: HostKind> {
    stamps: Vec<Site<H>>,
    n: usize,
}

impl<H: HostKind, S> Task<H, S> for Assemble<H> {
    type Rows = RowId<H, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) -> Self::Rows {
        for site in &self.stamps {
            host.read(site.row);
        }
        host.output("D", self.n * self.n)
    }

    fn run(
        &self,
        d: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, H>,
        _: Lane<'_, S>,
    ) -> Status {
        let (n, d) = (self.n, host.take_mut(*d));
        d.fill(0.0);
        for site in &self.stamps {
            let (block, stamp) = (site.block, host.get(site.row));
            for &offset in &site.offsets {
                for j in 0..block {
                    let column = offset + n * (offset + j);
                    d[column..column + block].copy_from_slice(&stamp[block * j..block * (j + 1)]);
                }
            }
        }
        Ok(())
    }
}

/// CPU: D read from a `.npy` file, its shape checked.
struct Load {
    from: PathBuf,
    n: usize,
}

impl<H: HostKind, S> Task<H, S> for Load {
    type Rows = RowId<H, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) -> Self::Rows {
        host.output("D", self.n * self.n)
    }

    fn run(
        &self,
        d: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, H>,
        _: Lane<'_, S>,
    ) -> Status {
        let array = fb_io_npy::read_npy(&self.from)?;
        let n = self.n;
        if array.shape != [n, n] {
            return Err(Error::config(format!(
                "[guess] {} is not {n} x {n}, the basis's spherical AO count",
                self.from.display()
            )));
        }
        host.take_mut(*d).copy_from_slice(&array.data);
        Ok(())
    }
}

/// GPU: the host guess copied into a device row on the worker's stream.
struct Upload {
    d: RowId<Pinned, f64>,
    len: usize,
}

impl Task for Upload {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.d);
        device.output("D", self.len)
    }

    fn run(
        &self,
        d: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let (src, dst) = (host.get(self.d), device.take_mut(*d));
        // SAFETY: two rows of `len` doubles, on the task's stream. The
        // pinned row is not rewritten before the copy completes: its bytes'
        // next owner is ordered after this task's event by a handoff.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size_of_val(src),
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
                stream.stream(),
            )
        })
    }
}
