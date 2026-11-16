//! What the localization tests share: each localized orbital's Fock
//! expectation, by which the step sorts them.

/// Each localized orbital's `<i|F|i> = Σ_k (C_occᵀ S C_local)_ki² ε_k`,
/// over the SCF's `n_occ` occupied orbitals `k`: `c` (`n × n`), `s`
/// (`n × n`) and `c_local` (`n × n_occ`), all column-major, and `eps` the
/// SCF's orbital energies. `C_occᵀ S C_local` is the localization's U, so
/// this is the diagonal of the C++ `Uᵀ diag(ε_occ) U`.
pub fn fock_expectations(c: &[f64], s: &[f64], c_local: &[f64], eps: &[f64]) -> Vec<f64> {
    let n = eps.len();
    let n_occ = c_local.len() / n;
    (0..n_occ)
        .map(|i| {
            let sc: Vec<f64> = (0..n)
                .map(|mu| (0..n).map(|nu| s[nu * n + mu] * c_local[i * n + nu]).sum())
                .collect();
            (0..n_occ)
                .map(|k| {
                    let u: f64 = (0..n).map(|mu| c[k * n + mu] * sc[mu]).sum();
                    u * u * eps[k]
                })
                .sum()
        })
        .collect()
}

/// Whether `energies` never fall by more than 1e-12 hartree, the
/// round-off of their sums of at most a few hundred terms of at most a few
/// hartree: the step sorts by its own sums of the same terms, from its U
/// rather than `C_occᵀ S C_local`, so two energies equal to round-off may
/// come out in either order here.
pub fn ascending(energies: &[f64]) -> bool {
    energies.windows(2).all(|e| e[0] <= e[1] + 1e-12)
}
