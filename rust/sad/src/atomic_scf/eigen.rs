//! `:eigen`: a symmetric eigensolver for the SCF's l blocks, cyclic Jacobi.

/// The eigendecomposition of a symmetric n × n matrix: `SymmetricEigen`.
pub(super) struct SymmetricEigen {
    /// Ascending.
    pub(super) values: Vec<f64>,
    /// n × n, column-major, column j for `values[j]`.
    pub(super) vectors: Vec<f64>,
}

/// The eigendecomposition of the symmetric `n × n` column-major `a`, by
/// cyclic Jacobi: `jacobi_eigen`.
///
/// The l blocks are a handful of shells, where Jacobi is exact to round-off
/// and needs no LAPACK. The sweeps, the rotations and their formulas
/// (`hypot` and `copysign` from the C library, as the C++ `std::` ones) are
/// the C++ ones. The final ordering is a stable sort by the diagonal, where
/// the C++ `std::ranges::sort` is not stable: the two agree unless two
/// eigenvalues are exactly equal, which the radial blocks of distinct shells
/// do not produce.
pub(super) fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> SymmetricEigen {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i + (n * i)] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for j in 0..n {
            for i in 0..j {
                off += a[i + (n * j)] * a[i + (n * j)];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p + (n * q)];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q + (n * q)] - a[p + (n * p)]) / (2.0 * apq);
                let t = 1.0_f64.copysign(theta) / (theta.abs() + theta.hypot(1.0));
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    let akp = a[k + (n * p)];
                    let akq = a[k + (n * q)];
                    a[k + (n * p)] = (c * akp) - (s * akq);
                    a[k + (n * q)] = (s * akp) + (c * akq);
                }
                for k in 0..n {
                    let apk = a[p + (n * k)];
                    let aqk = a[q + (n * k)];
                    a[p + (n * k)] = (c * apk) - (s * aqk);
                    a[q + (n * k)] = (s * apk) + (c * aqk);
                }
                for k in 0..n {
                    let vkp = v[k + (n * p)];
                    let vkq = v[k + (n * q)];
                    v[k + (n * p)] = (c * vkp) - (s * vkq);
                    v[k + (n * q)] = (s * vkp) + (c * vkq);
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x + (n * x)].total_cmp(&a[y + (n * y)]));
    let mut out = SymmetricEigen {
        values: vec![0.0; n],
        vectors: vec![0.0; n * n],
    };
    for (j, &o) in order.iter().enumerate() {
        out.values[j] = a[o + (n * o)];
        out.vectors[n * j..n * (j + 1)].copy_from_slice(&v[n * o..n * (o + 1)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[2 1; 1 2]` has eigenvalues 1 and 3, ascending, and `V Λ Vᵀ` is the
    /// matrix back.
    #[test]
    fn decomposes_a_two_by_two() {
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let eig = jacobi_eigen(a.clone(), 2);
        assert!((eig.values[0] - 1.0).abs() < 1e-15);
        assert!((eig.values[1] - 3.0).abs() < 1e-15);
        for j in 0..2 {
            for i in 0..2 {
                let back: f64 = (0..2)
                    .map(|k| eig.vectors[i + 2 * k] * eig.values[k] * eig.vectors[j + 2 * k])
                    .sum();
                assert!((back - a[i + 2 * j]).abs() < 1e-14);
            }
        }
    }
}
