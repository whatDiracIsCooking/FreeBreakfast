//! `src/sad` (module `free_breakfast.sad`): the Superposition-of-Atomic-
//! Densities initial guess, each atom's spherically averaged atomic SCF
//! density on its block of the molecular density matrix, in the spherical AO
//! layout. Host code that links nothing.
//!
//! **The model.** One restricted Hartree–Fock per atom, with fractional
//! occupations, over the atom's own shells. Its Z electrons (the neutral
//! atom's count) fill subshells in the Madelung order (`1s 2s 2p 3s 3p 4s 3d
//! 4p …` through `7p`, enough for Z ≤ 118); the d/f anomalies (Cr, Cu, …) are
//! not encoded, because a spherically averaged guess does not need them. The
//! k-th subshell of momentum l is seated on the k-th lowest orbital of the
//! atom's l block, its electrons spread evenly over the `2l+1` m. That
//! density is spherical, so its Fock matrix is block-diagonal by (l, m) and
//! the same for every m: the SCF diagonalizes one small radial block per l
//! (the m-average of F over that momentum's shells), which is PySCF's
//! `AtomSphAverageRHF` exactly. The energy is the restricted closed-shell
//! expression over the spin-summed density, `E = ½ tr D (H + F)`,
//! `F = H + J − K/2`, so a fractionally occupied orbital keeps part of its
//! own self-interaction. Core-Hamiltonian start, Pulay DIIS on the stacked
//! per-l commutators `F P S − S P F`, canonical orthogonalization within each
//! l block; the tolerance (1e-9) bounds both the largest commutator element
//! and the energy change.
//!
//! **One center, so no Boys function.** Every integral of the atomic SCF has
//! all four centers at the nucleus: the Hermite expansion reduces to its
//! same-center recurrence, and the Boys function is only ever needed at 0,
//! `F_n(0) = 1/(2n+1)`. S, T, V and `(ij|kl)` are closed form, computed over
//! the Cartesian components and carried to the spherical layout with
//! [`fb_basis::spherical_transform_matrix`]. `(ij|kl)` evaluates only the
//! canonical shell quartets and writes each to its eight permutations. The
//! whole n⁴ tensor is held, so an atom past 96 Cartesian AOs (≈680 MB) is
//! refused as [`ErrorKind::Unsupported`](fb_core_error_handling::ErrorKind).
//!
//! **One SCF per distinct (Z, shells).** A molecule pays for its elements,
//! not its atoms. The density is a one-center, so geometry-free, block, an
//! `AtomicDensityStamp`, written onto each atom's diagonal block that has
//! that Z and those shells; between atoms the guess is zero.
//!
//! **Baked stamps.** [`sad_density`] takes an optional [`SadStampLookup`].
//! For a basis that carries a name ([`BasisSet::name`], set only for a
//! compiled-in set), each distinct stamp asks it first, and an element it has
//! skips its SCF. The baked block is checked against the shells by size only
//! ([`ErrorKind::InvalidConfig`](fb_core_error_handling::ErrorKind) if it
//! differs): the name is what says the shells are the ones it was computed
//! over. This crate never knows where the blocks come from.
//!
//! **What the trace means.** Each atomic block holds Z electrons against the
//! atom's overlap, so `trace(D S)` is `Σ_A Z_A` exactly; `trace(D)` is not,
//! since shells of one atom overlap. The guess places `Σ_A Z_A` electrons
//! whatever the molecule's net charge; rescaling is the SCF driver's call.
//!
//! **Cartesian consumers.** A Cartesian-layout density is `U D Uᵀ` with
//! `U = spherical_transform_matrix(basis)`: density transforms the opposite
//! way to the `Uᵀ M U` of Fock and overlap matrices.
//!
//! **The C++ operation order is kept, everywhere.** Every loop here runs in
//! the C++ order, every expression groups as the C++ one does, and `pow`,
//! `sqrt` and `hypot` are the C library's, so the stamps this crate computes
//! can be compared bitwise with the C++ generator's (`data/sad/`), by the
//! stamp issue. Each module says where that is relied on. Two places cannot
//! promise it by construction and say so: the eigenvalue sort's ties
//! (`atomic_scf::eigen`), and floating-point contraction, which neither tree
//! asks for and which the C++ build gets only on a target with FMA when the
//! compiler contracts (Rust never contracts).
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `SadStampLookup`, a function pointer | [`SadStampLookup`], a `fn` pointer; `Option` for the C++ `nullptr` |
//! | `sad_density(basis, atoms, baked = nullptr)` | [`sad_density`], `baked` always given |
//! | `sad_density(flow, basis, atoms, out, baked)`, the task form | not here: in the model each SCF is a CPU node of the plan, `fb-pipeline-guess`'s `add_sad_guess`, which plans over [`distinct_stamps`] and runs [`stamp_density`] in each node |
//! | `sad.cpp`: `AtomicDensityStamp`, `StampSites`, `distinct_stamps`, `stamp_density` | [`AtomicDensityStamp`], [`StampSites`], [`distinct_stamps`], [`stamp_density`], public for the task form; `stamp_density` takes the basis's name, the one thing of the basis it reads |
//! | `sad.cpp`: `place`, `same_shells`, `stamp_and_place` | the private items of those names here |
//! | `:aufbau`: `SubshellFill`, `kAufbauOrder`, `subshell_occupation` | `aufbau`'s `SubshellFill`, `AUFBAU_ORDER`, `subshell_occupation` |
//! | `:hermite`: `one_center_hermite`, `one_center_hermite_coulomb` | `atomic_scf::hermite`'s functions of those names |
//! | `:one_electron`: `one_center_pair_loop`, `one_center_overlap_1d`, `one_center_overlap`, `one_center_kinetic`, `one_center_nuclear` | `atomic_scf::one_electron`'s functions of those names; the loop takes a closure for the template's functor |
//! | `:eri`: `one_center_eri` | `atomic_scf::eri::one_center_eri` |
//! | `:integrals`: `kMaxAtomicAoCart`, `rotate_transform`, `to_spherical`, `AtomicIntegrals`, `atomic_integrals` | `atomic_scf::integrals`'s `MAX_ATOMIC_AO_CART` and the items of those names |
//! | `:gemm`: `gemm_tn`, `gemm_nn` | `atomic_scf::gemm`'s functions of those names |
//! | `:eigen`: `SymmetricEigen`, `jacobi_eigen` | `atomic_scf::eigen`'s items of those names |
//! | `:diis`: `bordered_solve`, `AtomicDiis` | `atomic_scf::diis`'s items of those names |
//! | `:angular_block`: `kLinearDependence`, `AngularBlock`, `angular_blocks`, `angular_density` | `atomic_scf::angular_block`'s `LINEAR_DEPENDENCE` and the items of those names |
//! | `:scf`: `kDiisDepth`, `rhf_fock`, `AtomicScf`, `atomic_scf` | `atomic_scf`'s `DIIS_DEPTH` and the items of those names |
//! | `AtomicScf::orbital_energies`, and the diagonalization after convergence that fills it | nothing: no caller reads them, and the density is taken before it |
//! | `AtomicScf::energy`, `iterations` | the same fields, logged at debug when an SCF converges |
//! | the re-export of `free_breakfast.third_party.taskflow` | nothing: there is no task form here |
//!
//! Everything but [`sad_density`], [`SadStampLookup`] and the four items the
//! task form plans with is private to the crate, as the C++ partitions are to
//! the module.

#![forbid(unsafe_code)]

mod atomic_scf;
mod aufbau;

use fb_atom::Atom;
use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::{Error, Result, Status};
use fb_molecule::nuclear_charge;

use crate::atomic_scf::atomic_scf;

/// Where [`sad_density`] finds an element's density instead of running its
/// SCF: `SadStampLookup`. Given a basis set's name ([`BasisSet::name`],
/// never empty) and an atomic number, that atom's converged block over its
/// own spherical AOs, flat column-major, or `None` for an element it does
/// not have.
pub type SadStampLookup = fn(basis: &str, z: i32) -> Option<Vec<f64>>;

/// The SAD initial-guess density: each atom's spherically averaged atomic
/// SCF density on its block of `basis`'s spherical layout, zeros between
/// atoms: `sad_density`, the serial form.
///
/// One SCF per distinct (Z, shells) pair, in order of each pair's first
/// atom: [`fb_basis::BasisStamp::place`] gives every atom of an element the
/// same shells.
///
/// `basis` is made by [`fb_basis::BasisStamp::place`] over `atoms`, in the
/// same order, so it carries the per-atom shell grouping
/// ([`BasisSet::natoms`]). `baked` is asked first for each element's density
/// when `basis` is named; `None`, or an unnamed basis, computes every one.
///
/// Returns `n × n`, `n = basis.nao_sphe()`, flat column-major, element
/// `(i, j)` at `i + j n`. The first failing atom's error otherwise:
///
/// - `InvalidConfig` if `basis.natoms()` differs from the atom count,
///   including a basis built from a bare shell list, which carries no
///   grouping at all; if an atom's shells are not on one center; if some
///   atom's basis has fewer linearly independent orbitals of some momentum
///   than it has occupied subshells of it; or if `baked` returns a block of
///   the wrong size for an atom's shells.
/// - `Unsupported` for an atom past 96 Cartesian AOs, where its n⁴
///   two-electron tensor no longer fits well under a gigabyte.
/// - `MaxIterationsExceeded` if an atomic SCF does not converge.
pub fn sad_density(
    basis: &BasisSet,
    atoms: &[Atom],
    baked: Option<SadStampLookup>,
) -> Result<Vec<f64>> {
    let mut sites = distinct_stamps(basis, atoms)?;
    let n = basis.nao_sphe() as usize;
    let mut density = vec![0.0; n * n];
    for site in &mut sites {
        stamp_and_place(site, basis, baked, n, &mut density)?;
    }
    Ok(density)
}

/// Whether two atoms share an SCF, `same_shells`: their shells match exactly
/// (momentum, exponents and coefficients), whatever their centers.
fn same_shells(a: &[Shell], b: &[Shell]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.angular_momentum() == y.angular_momentum() && x.primitives() == y.primitives()
        })
}

/// The spherical AO count of `shells`.
fn spherical_count(shells: &[Shell]) -> usize {
    shells
        .iter()
        .map(|shell| shell.function_count_sphe() as usize)
        .sum()
}

/// One atom's converged SCF density over its own spherical AOs,
/// column-major: `AtomicDensityStamp`. Every integral of the atomic SCF is
/// one-center, so the block is geometry-free: every atom with this Z and
/// these shells wears the same one.
pub struct AtomicDensityStamp<'a> {
    /// The atomic number.
    pub z: i32,
    /// The atom's shells, on its center.
    pub shells: &'a [Shell],
    /// The block, `n × n` over the shells' spherical AOs; empty until its
    /// SCF runs.
    pub density: Vec<f64>,
}

/// Write `stamp`'s block onto the diagonal of the `n × n` column-major
/// `density`, starting at AO `offset`: `place`.
fn place(stamp: &AtomicDensityStamp<'_>, offset: usize, n: usize, density: &mut [f64]) {
    let block = spherical_count(stamp.shells);
    for j in 0..block {
        for i in 0..block {
            density[(offset + i) + (n * (offset + j))] = stamp.density[i + (block * j)];
        }
    }
}

/// One distinct (Z, shells) of a molecule and where it goes, `StampSites`:
/// its stamp, whose density stays empty until its SCF runs, and the first AO
/// of each atom that wears it.
pub struct StampSites<'a> {
    /// The stamp.
    pub stamp: AtomicDensityStamp<'a>,
    /// The first spherical AO of each atom that wears it, in atom order.
    pub offsets: Vec<usize>,
}

/// The distinct (Z, shells) of `atoms` over `basis`, in order of each one's
/// first atom, `distinct_stamps`: the first of them to fail is the one the
/// first failing atom wears. An atom with no shells wears none.
///
/// # Errors
/// `InvalidConfig` if `basis.natoms()` differs from the atom count.
pub fn distinct_stamps<'a>(basis: &'a BasisSet, atoms: &[Atom]) -> Result<Vec<StampSites<'a>>> {
    if basis.natoms() as usize != atoms.len() {
        return Err(Error::config(format!(
            "sad_density: the basis groups its shells over {} atoms but {} were given; the \
             basis was not placed over these atoms",
            basis.natoms(),
            atoms.len()
        )));
    }
    let mut sites: Vec<StampSites<'a>> = Vec::new();
    for (a, atom) in (0..basis.natoms()).zip(atoms) {
        let count = basis.atom_shell_count(a) as usize;
        if count == 0 {
            continue;
        }
        let first = basis.atom_shell_offset(a);
        let shells = &basis.shells()[first as usize..first as usize + count];
        let z = nuclear_charge(atom);
        let k = match sites
            .iter()
            .position(|s| s.stamp.z == z && same_shells(s.stamp.shells, shells))
        {
            Some(k) => k,
            None => {
                sites.push(StampSites {
                    stamp: AtomicDensityStamp {
                        z,
                        shells,
                        density: Vec::new(),
                    },
                    offsets: Vec::new(),
                });
                sites.len() - 1
            }
        };
        sites[k].offsets.push(basis.ao_offset_sphe(first) as usize);
    }
    Ok(sites)
}

/// `stamp`'s density, `stamp_density`: `baked`'s for `basis_name`, the
/// basis's [`BasisSet::name`], when it is not empty and `baked` has this
/// element, else its own SCF's. A baked block is checked against the shells
/// only by size, the one thing a lookup by name can get wrong that a guess
/// cannot survive.
///
/// # Errors
/// [`sad_density`]'s, for this one stamp.
pub fn stamp_density(
    stamp: &AtomicDensityStamp<'_>,
    basis_name: &str,
    baked: Option<SadStampLookup>,
) -> Result<Vec<f64>> {
    if let Some(lookup) = baked
        && !basis_name.is_empty()
        && let Some(density) = lookup(basis_name, stamp.z)
    {
        let block = spherical_count(stamp.shells);
        if density.len() != block * block {
            return Err(Error::config(format!(
                "sad_density: the baked density for Z={} in {} holds {} values, but the atom's \
                 shells need {block}x{block}",
                stamp.z,
                basis_name,
                density.len()
            )));
        }
        return Ok(density);
    }
    let scf = atomic_scf(stamp.z, stamp.shells)?;
    log::debug!(
        "sad_density: Z={} converged in {} iterations, E = {:.12}",
        stamp.z,
        scf.iterations,
        scf.energy
    );
    Ok(scf.density)
}

/// Fill `site`'s stamp, then place it on each of its atoms' blocks of the
/// `n × n` `density`: `stamp_and_place`.
fn stamp_and_place(
    site: &mut StampSites<'_>,
    basis: &BasisSet,
    baked: Option<SadStampLookup>,
    n: usize,
    density: &mut [f64],
) -> Status {
    site.stamp.density = stamp_density(&site.stamp, basis.name(), baked)?;
    for &offset in &site.offsets {
        place(&site.stamp, offset, n, density);
    }
    Ok(())
}
