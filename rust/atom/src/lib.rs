//! `src/atom` (module `free_breakfast.atom`): the chemical element and one
//! atom positioned in space.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `element.h`: `Element`, `enum class : uint8_t`, `Sentinel` = 0, `H` = 1 ... `Og` = 118 | [`Element`], `#[repr(u8)]`, the same values |
//! | `static_cast<uint8_t>(element)` | [`Element::z`] |
//! | `static_cast<Element>(z)` | [`Element::from_z`], `None` above 118 |
//! | `element_symbol` | [`Element::symbol`] |
//! | `element_from_symbol` | [`Element::from_symbol`] |
//! | `atom.h`: `Atom` | [`Atom`] |
//! | the `free_breakfast::atom::headers` include path, and the re-export from the global module fragment | nothing: one crate, one definition |
//!
//! **The header-and-module pair has no counterpart.** The C++ defines
//! `Element` and `Atom` in headers so device code can `#include` the very
//! entity a module importer names. Here nothing on the device reads an
//! `Atom`: a reader's nuclei reach a kernel as arrays a task uploads (the
//! `fb_molecule` crate's doc says how), so the crate is the one definition.
//!
//! **`from_z` refuses what `static_cast` does not.** The C++ cast yields an
//! `Element` for any byte; a Rust enum may hold only its variants, so a Z
//! past oganesson is `None` rather than a value with no symbol.
//!
//! **`from_symbol` keeps the C++ answer for no match**, the
//! [`Element::Sentinel`] the default already means "no element" by, rather
//! than an `Option` beside it: the XYZ reader turns a sentinel into its own
//! error naming the line.
//!
//! The variants are the symbols, so one list in this file defines the enum,
//! its discriminants (Z, counted from `Sentinel` = 0), and the symbol table
//! both lookups read.

#![forbid(unsafe_code)]

use fb_core_types::Double3;

/// Defines [`Element`] and the two tables indexed by Z from one list of
/// symbols in order of Z, each with its element's name as the variant's doc.
macro_rules! elements {
    ($($symbol:ident $name:literal),+ $(,)?) => {
        /// A chemical element, named by its symbol and valued by its atomic
        /// number Z: `Element`.
        ///
        /// The discriminant is the nuclear charge, so [`Element::z`] is Z,
        /// and every Z in 1 ..= 118 round-trips through its symbol.
        /// [`Element::Sentinel`] (Z = 0) is the default, "no element yet":
        /// it names no nucleus.
        #[repr(u8)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub enum Element {
            /// No element: the default, unset value.
            #[default]
            Sentinel = 0,
            $(#[doc = $name] $symbol,)+
        }

        /// Every element by Z, `Sentinel` at index 0.
        const BY_Z: [Element; 119] = [Element::Sentinel, $(Element::$symbol,)+];

        /// `kElementSymbols`: every symbol by Z, the sentinel's empty one at
        /// index 0.
        const SYMBOLS: [&str; 119] = ["", $(stringify!($symbol),)+];
    };
}

elements! {
    H "Hydrogen", He "Helium", Li "Lithium", Be "Beryllium", B "Boron",
    C "Carbon", N "Nitrogen", O "Oxygen", F "Fluorine", Ne "Neon",
    Na "Sodium", Mg "Magnesium", Al "Aluminium", Si "Silicon",
    P "Phosphorus", S "Sulfur", Cl "Chlorine", Ar "Argon", K "Potassium",
    Ca "Calcium", Sc "Scandium", Ti "Titanium", V "Vanadium",
    Cr "Chromium", Mn "Manganese", Fe "Iron", Co "Cobalt", Ni "Nickel",
    Cu "Copper", Zn "Zinc", Ga "Gallium", Ge "Germanium", As "Arsenic",
    Se "Selenium", Br "Bromine", Kr "Krypton", Rb "Rubidium",
    Sr "Strontium", Y "Yttrium", Zr "Zirconium", Nb "Niobium",
    Mo "Molybdenum", Tc "Technetium", Ru "Ruthenium", Rh "Rhodium",
    Pd "Palladium", Ag "Silver", Cd "Cadmium", In "Indium", Sn "Tin",
    Sb "Antimony", Te "Tellurium", I "Iodine", Xe "Xenon", Cs "Caesium",
    Ba "Barium", La "Lanthanum", Ce "Cerium", Pr "Praseodymium",
    Nd "Neodymium", Pm "Promethium", Sm "Samarium", Eu "Europium",
    Gd "Gadolinium", Tb "Terbium", Dy "Dysprosium", Ho "Holmium",
    Er "Erbium", Tm "Thulium", Yb "Ytterbium", Lu "Lutetium",
    Hf "Hafnium", Ta "Tantalum", W "Tungsten", Re "Rhenium", Os "Osmium",
    Ir "Iridium", Pt "Platinum", Au "Gold", Hg "Mercury", Tl "Thallium",
    Pb "Lead", Bi "Bismuth", Po "Polonium", At "Astatine", Rn "Radon",
    Fr "Francium", Ra "Radium", Ac "Actinium", Th "Thorium",
    Pa "Protactinium", U "Uranium", Np "Neptunium", Pu "Plutonium",
    Am "Americium", Cm "Curium", Bk "Berkelium", Cf "Californium",
    Es "Einsteinium", Fm "Fermium", Md "Mendelevium", No "Nobelium",
    Lr "Lawrencium", Rf "Rutherfordium", Db "Dubnium", Sg "Seaborgium",
    Bh "Bohrium", Hs "Hassium", Mt "Meitnerium", Ds "Darmstadtium",
    Rg "Roentgenium", Cn "Copernicium", Nh "Nihonium", Fl "Flerovium",
    Mc "Moscovium", Lv "Livermorium", Ts "Tennessine", Og "Oganesson",
}

impl Element {
    /// The atomic number Z, 0 for [`Element::Sentinel`].
    pub const fn z(self) -> u8 {
        self as u8
    }

    /// The element of atomic number `z`, [`Element::Sentinel`] for 0, and
    /// `None` past oganesson (118).
    pub const fn from_z(z: u8) -> Option<Element> {
        if (z as usize) < BY_Z.len() {
            Some(BY_Z[z as usize])
        } else {
            None
        }
    }

    /// The symbol (`"H"`, `"Og"`); empty for [`Element::Sentinel`]:
    /// `element_symbol`.
    pub const fn symbol(self) -> &'static str {
        SYMBOLS[self as usize]
    }

    /// The element `symbol` names, or [`Element::Sentinel`] if none does:
    /// `element_from_symbol`.
    ///
    /// Case-insensitive, since files vary between `"CL"` and `"Cl"`: the
    /// first byte is compared upper-cased and the second lower-cased. The
    /// whole symbol must match (`"CO"` is cobalt, never carbon), and nothing
    /// is trimmed: the readers split on whitespace before they look one up.
    pub const fn from_symbol(symbol: &str) -> Element {
        let key = symbol.as_bytes();
        if key.is_empty() || key.len() > 2 {
            return Element::Sentinel;
        }
        let first = key[0].to_ascii_uppercase();
        let mut z = 1;
        while z < SYMBOLS.len() {
            let candidate = SYMBOLS[z].as_bytes();
            if candidate.len() == key.len()
                && candidate[0] == first
                && (key.len() == 1 || candidate[1] == key[1].to_ascii_lowercase())
            {
                return BY_Z[z];
            }
            z += 1;
        }
        Element::Sentinel
    }
}

/// One atom: a nucleus of a given element at a Cartesian position, in bohr:
/// `Atom`.
///
/// Plain data, with nothing to protect: both fields are public, and a
/// struct literal builds one (`Atom { position, element: Element::C }`).
/// The default sits at the origin carrying [`Element::Sentinel`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Atom {
    /// The position, in bohr.
    pub position: Double3,
    /// The element, whose Z is the nucleus's charge.
    pub element: Element,
}

// `atom_host.cpp`'s static_asserts that hold here: one byte, and the
// lookups usable in a constant expression.
const _: () = assert!(size_of::<Element>() == 1);
const _: () = assert!(matches!(Element::Og.symbol().as_bytes(), b"Og"));
const _: () = assert!(Element::from_symbol("Og") as u8 == Element::Og as u8);
const _: () = assert!(Element::Sentinel.symbol().is_empty());
const _: () = assert!(Element::from_symbol("") as u8 == Element::Sentinel as u8);

/// `test/atom/atom_host.cpp`, one test per case, each named after it.
#[cfg(test)]
mod tests {
    use super::{Atom, Element};
    use fb_core_types::Double3;

    /// Every element, Z = 1 ..= 118.
    fn every_element() -> impl Iterator<Item = Element> {
        (1..=118).map(|z| Element::from_z(z).unwrap())
    }

    #[test]
    fn element_value_is_atomic_number() {
        assert_eq!(Element::Sentinel.z(), 0);
        assert_eq!(Element::H.z(), 1);
        assert_eq!(Element::C.z(), 6);
        assert_eq!(Element::O.z(), 8);
        assert_eq!(Element::Fe.z(), 26);
        assert_eq!(Element::U.z(), 92);
        assert_eq!(Element::Og.z(), 118);
    }

    /// No C++ counterpart: the cast it replaces cannot fail.
    #[test]
    fn from_z_inverts_z_and_stops_at_oganesson() {
        for z in 0..=118 {
            assert_eq!(Element::from_z(z).map(Element::z), Some(z));
        }
        assert_eq!(Element::from_z(119), None);
        assert_eq!(Element::from_z(u8::MAX), None);
    }

    #[test]
    fn default_element_is_sentinel() {
        assert_eq!(Element::default(), Element::Sentinel);
    }

    #[test]
    fn default_atom_sits_at_origin_with_no_element() {
        let atom = Atom::default();
        assert_eq!(atom.position.x, 0.0);
        assert_eq!(atom.position.y, 0.0);
        assert_eq!(atom.position.z, 0.0);
        assert_eq!(atom.element, Element::Sentinel);
    }

    #[test]
    fn brace_initialises_position_then_element() {
        let atom = Atom {
            position: Double3 {
                x: 1.0,
                y: -2.5,
                z: 3.25,
            },
            element: Element::N,
        };
        assert_eq!(atom.position.x, 1.0);
        assert_eq!(atom.position.y, -2.5);
        assert_eq!(atom.position.z, 3.25);
        assert_eq!(atom.element, Element::N);
    }

    #[test]
    fn copy_is_independent_of_the_original() {
        let original = Atom {
            position: Double3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            element: Element::C,
        };
        let mut copy = original;
        copy.position.x = -1.0;
        copy.element = Element::Si;
        assert_eq!(original.position.x, 1.0);
        assert_eq!(original.element, Element::C);
        assert_eq!(copy.position.x, -1.0);
        assert_eq!(copy.position.y, 2.0);
        assert_eq!(copy.element, Element::Si);
    }

    #[test]
    fn every_symbol_round_trips() {
        for element in every_element() {
            assert_eq!(Element::from_symbol(element.symbol()), element);
        }
        assert_eq!(Element::Sentinel.symbol(), "");
    }

    #[test]
    fn symbols_are_a_capital_then_at_most_one_lowercase() {
        for element in every_element() {
            let symbol = element.symbol().as_bytes();
            assert!(symbol.len() == 1 || symbol.len() == 2, "{element:?}");
            assert!(symbol[0].is_ascii_uppercase(), "{element:?}");
            if symbol.len() == 2 {
                assert!(symbol[1].is_ascii_lowercase(), "{element:?}");
            }
        }
    }

    /// No C++ counterpart: here the variant's name is the symbol, so this
    /// pins that the table and the variants are one list.
    #[test]
    fn symbol_is_the_variant_name() {
        for element in every_element() {
            assert_eq!(format!("{element:?}"), element.symbol());
        }
    }

    #[test]
    fn symbols_of_sampled_elements() {
        assert_eq!(Element::H.symbol(), "H");
        assert_eq!(Element::C.symbol(), "C");
        assert_eq!(Element::Cl.symbol(), "Cl");
        assert_eq!(Element::Fe.symbol(), "Fe");
        assert_eq!(Element::U.symbol(), "U");
    }

    #[test]
    fn symbol_lookup_ignores_case() {
        assert_eq!(Element::from_symbol("Cl"), Element::Cl);
        assert_eq!(Element::from_symbol("CL"), Element::Cl);
        assert_eq!(Element::from_symbol("cl"), Element::Cl);
        assert_eq!(Element::from_symbol("cL"), Element::Cl);
        assert_eq!(Element::from_symbol("h"), Element::H);
    }

    /// A one-letter symbol is not a prefix match for a two-letter one, and
    /// folding a two-letter symbol's case never splits it into two elements
    /// ("CO" is cobalt, not carbon).
    #[test]
    fn symbol_lookup_matches_the_whole_symbol() {
        assert_eq!(Element::from_symbol("C"), Element::C);
        assert_eq!(Element::from_symbol("Co"), Element::Co);
        assert_eq!(Element::from_symbol("CO"), Element::Co);
        assert_eq!(Element::from_symbol("Cs"), Element::Cs);
        assert_eq!(Element::from_symbol("N"), Element::N);
        assert_eq!(Element::from_symbol("NA"), Element::Na);
    }

    #[test]
    fn unknown_symbol_is_sentinel() {
        assert_eq!(Element::from_symbol(""), Element::Sentinel);
        assert_eq!(Element::from_symbol("Xx"), Element::Sentinel);
        assert_eq!(Element::from_symbol("Abc"), Element::Sentinel);
        assert_eq!(Element::from_symbol("J"), Element::Sentinel);
        assert_eq!(Element::from_symbol("1"), Element::Sentinel);
    }

    /// No trimming: the readers split on whitespace before they look a
    /// symbol up.
    #[test]
    fn padded_symbol_is_sentinel() {
        assert_eq!(Element::from_symbol(" H"), Element::Sentinel);
        assert_eq!(Element::from_symbol("H "), Element::Sentinel);
        assert_eq!(Element::from_symbol(" Fe"), Element::Sentinel);
    }
}
