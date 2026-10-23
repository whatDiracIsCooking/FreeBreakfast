/// @file
/// @brief Aggregated interface for the Molden orbital format module.
///
/// `import free_breakfast.io.molden;` gives the writer of the Molden format
/// (https://www.theochem.ru.nl/molden/molden_format.html): a molecule's atoms,
/// its spherical basis and a set of orbitals, as the text Molden, Jmol,
/// IboView, Multiwfn and PySCF's `pyscf.tools.molden.load` all read.

export module free_breakfast.io.molden;

export import :writer;
