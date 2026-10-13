/// @file
/// @brief The chemical element as a strong enum, one enumerator per element.
///
/// @c Element is a strongly typed stand-in for the nuclear charge Z: an
/// element is named by its symbol (`Element::H`, `Element::Og`) and its
/// underlying value is exactly Z. The underlying type is @c uint8_t because
/// the largest confirmed element is Z=118, which fits comfortably in a byte.
///
/// This header is the single definition of @c Element. CUDA / other headers
/// reach it by `#include <element.h>`; C++ modules reach the very same entity by
/// `import free_breakfast.atom;`, which includes this header in its
/// global module fragment and re-exports the name from within the namespace.
///
/// It includes @c <cstdint> rather than relying on `import std`, so it works
/// unchanged in a module's global module fragment and in CUDA / HIP device code.

#pragma once

#include <cstdint>

namespace free_breakfast {

/// @brief A chemical element, named by its symbol and valued by its atomic
///        number Z.
///
/// The enumerator value is the nuclear charge, so `static_cast<std::uint8_t>(Z)`
/// yields Z directly and any Z in [1, 118] round-trips through a symbol. The
/// list covers hydrogen (Z=1) through oganesson (Z=118). @c Sentinel (Z=0) is
/// the default-constructed "no element" value: it names no real nucleus and is
/// what a default-constructed @c Element{} holds.
enum class Element : std::uint8_t {
  Sentinel = 0, // No element; the default-constructed / "unset" value
  H = 1,        // Hydrogen
  He = 2,       // Helium
  Li = 3,       // Lithium
  Be = 4,       // Beryllium
  B = 5,        // Boron
  C = 6,        // Carbon
  N = 7,        // Nitrogen
  O = 8,        // Oxygen
  F = 9,        // Fluorine
  Ne = 10,      // Neon
  Na = 11,      // Sodium
  Mg = 12,      // Magnesium
  Al = 13,      // Aluminium
  Si = 14,      // Silicon
  P = 15,       // Phosphorus
  S = 16,       // Sulfur
  Cl = 17,      // Chlorine
  Ar = 18,      // Argon
  K = 19,       // Potassium
  Ca = 20,      // Calcium
  Sc = 21,      // Scandium
  Ti = 22,      // Titanium
  V = 23,       // Vanadium
  Cr = 24,      // Chromium
  Mn = 25,      // Manganese
  Fe = 26,      // Iron
  Co = 27,      // Cobalt
  Ni = 28,      // Nickel
  Cu = 29,      // Copper
  Zn = 30,      // Zinc
  Ga = 31,      // Gallium
  Ge = 32,      // Germanium
  As = 33,      // Arsenic
  Se = 34,      // Selenium
  Br = 35,      // Bromine
  Kr = 36,      // Krypton
  Rb = 37,      // Rubidium
  Sr = 38,      // Strontium
  Y = 39,       // Yttrium
  Zr = 40,      // Zirconium
  Nb = 41,      // Niobium
  Mo = 42,      // Molybdenum
  Tc = 43,      // Technetium
  Ru = 44,      // Ruthenium
  Rh = 45,      // Rhodium
  Pd = 46,      // Palladium
  Ag = 47,      // Silver
  Cd = 48,      // Cadmium
  In = 49,      // Indium
  Sn = 50,      // Tin
  Sb = 51,      // Antimony
  Te = 52,      // Tellurium
  I = 53,       // Iodine
  Xe = 54,      // Xenon
  Cs = 55,      // Caesium
  Ba = 56,      // Barium
  La = 57,      // Lanthanum
  Ce = 58,      // Cerium
  Pr = 59,      // Praseodymium
  Nd = 60,      // Neodymium
  Pm = 61,      // Promethium
  Sm = 62,      // Samarium
  Eu = 63,      // Europium
  Gd = 64,      // Gadolinium
  Tb = 65,      // Terbium
  Dy = 66,      // Dysprosium
  Ho = 67,      // Holmium
  Er = 68,      // Erbium
  Tm = 69,      // Thulium
  Yb = 70,      // Ytterbium
  Lu = 71,      // Lutetium
  Hf = 72,      // Hafnium
  Ta = 73,      // Tantalum
  W = 74,       // Tungsten
  Re = 75,      // Rhenium
  Os = 76,      // Osmium
  Ir = 77,      // Iridium
  Pt = 78,      // Platinum
  Au = 79,      // Gold
  Hg = 80,      // Mercury
  Tl = 81,      // Thallium
  Pb = 82,      // Lead
  Bi = 83,      // Bismuth
  Po = 84,      // Polonium
  At = 85,      // Astatine
  Rn = 86,      // Radon
  Fr = 87,      // Francium
  Ra = 88,      // Radium
  Ac = 89,      // Actinium
  Th = 90,      // Thorium
  Pa = 91,      // Protactinium
  U = 92,       // Uranium
  Np = 93,      // Neptunium
  Pu = 94,      // Plutonium
  Am = 95,      // Americium
  Cm = 96,      // Curium
  Bk = 97,      // Berkelium
  Cf = 98,      // Californium
  Es = 99,      // Einsteinium
  Fm = 100,     // Fermium
  Md = 101,     // Mendelevium
  No = 102,     // Nobelium
  Lr = 103,     // Lawrencium
  Rf = 104,     // Rutherfordium
  Db = 105,     // Dubnium
  Sg = 106,     // Seaborgium
  Bh = 107,     // Bohrium
  Hs = 108,     // Hassium
  Mt = 109,     // Meitnerium
  Ds = 110,     // Darmstadtium
  Rg = 111,     // Roentgenium
  Cn = 112,     // Copernicium
  Nh = 113,     // Nihonium
  Fl = 114,     // Flerovium
  Mc = 115,     // Moscovium
  Lv = 116,     // Livermorium
  Ts = 117,     // Tennessine
  Og = 118,     // Oganesson
};

} // namespace free_breakfast
