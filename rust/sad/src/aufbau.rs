//! `:aufbau`: the neutral atom's aufbau fill, [`AUFBAU_ORDER`] and
//! [`subshell_occupation`], which the atomic SCF seats on its l blocks'
//! orbitals.

use fb_core_error_handling::{Error, check_or_abort};

/// One subshell in the aufbau filling order: `SubshellFill`.
pub(crate) struct SubshellFill {
    /// Angular momentum (0 = s, 1 = p, 2 = d, 3 = f).
    pub(crate) l: i32,
    /// Electrons that fill it, 2(2l+1).
    pub(crate) capacity: i32,
}

/// The Madelung (n+l, then n) filling order, 1s 2s 2p 3s 3p 4s 3d 4p … 7p:
/// long enough to seat any Z in [1, 118]: `kAufbauOrder`.
///
/// Written out rather than generated so the order is auditable at a glance.
pub(crate) const AUFBAU_ORDER: [SubshellFill; 19] = [
    SubshellFill { l: 0, capacity: 2 },  // 1s
    SubshellFill { l: 0, capacity: 2 },  // 2s
    SubshellFill { l: 1, capacity: 6 },  // 2p
    SubshellFill { l: 0, capacity: 2 },  // 3s
    SubshellFill { l: 1, capacity: 6 },  // 3p
    SubshellFill { l: 0, capacity: 2 },  // 4s
    SubshellFill { l: 2, capacity: 10 }, // 3d
    SubshellFill { l: 1, capacity: 6 },  // 4p
    SubshellFill { l: 0, capacity: 2 },  // 5s
    SubshellFill { l: 2, capacity: 10 }, // 4d
    SubshellFill { l: 1, capacity: 6 },  // 5p
    SubshellFill { l: 0, capacity: 2 },  // 6s
    SubshellFill { l: 3, capacity: 14 }, // 4f
    SubshellFill { l: 2, capacity: 10 }, // 5d
    SubshellFill { l: 1, capacity: 6 },  // 6p
    SubshellFill { l: 0, capacity: 2 },  // 7s
    SubshellFill { l: 3, capacity: 14 }, // 5f
    SubshellFill { l: 2, capacity: 10 }, // 6d
    SubshellFill { l: 1, capacity: 6 },  // 7p
];

/// Ground-state electron count per subshell for a neutral atom of atomic
/// number `z`, aligned to [`AUFBAU_ORDER`]: `subshell_occupation`.
///
/// A plain aufbau fill: full subshells hold their capacity, the last holds
/// the remainder, and trailing empty subshells are omitted, so the entries
/// sum to exactly `z`. `z` is in [0, 118]; 0 (`Element::Sentinel`) gives an
/// empty result, and anything outside aborts: no `Element` spells it, so it
/// is a caller bug.
pub(crate) fn subshell_occupation(z: i32) -> Vec<i32> {
    if !(0..=118).contains(&z) {
        check_or_abort::<()>(Err(Error::config(format!(
            "subshell_occupation: Z={z} is outside [0, 118]"
        ))));
    }
    let mut occupation = Vec::new();
    let mut remaining = z;
    for subshell in &AUFBAU_ORDER {
        if remaining == 0 {
            break;
        }
        let here = remaining.min(subshell.capacity);
        occupation.push(here);
        remaining -= here;
    }
    occupation
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Z seats exactly its electrons, and each subshell's capacity is
    /// `2(2l+1)`.
    #[test]
    fn fills_every_element_with_its_electrons() {
        for fill in &AUFBAU_ORDER {
            assert_eq!(fill.capacity, 2 * (2 * fill.l + 1));
        }
        assert!(subshell_occupation(0).is_empty());
        for z in 1..=118 {
            assert_eq!(subshell_occupation(z).iter().sum::<i32>(), z, "Z={z}");
        }
        // Zn: 1s2 2s2 2p6 3s2 3p6 4s2 3d10.
        assert_eq!(subshell_occupation(30), [2, 2, 6, 2, 6, 2, 10]);
    }
}
