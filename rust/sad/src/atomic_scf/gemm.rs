//! `:gemm`: the two host matrix products the SCF's l blocks need. The
//! blocks are a handful of shells across, so plain loops, not BLAS, in the
//! C++ loop order: each element is the same sum in the same order.

/// `C = Aᵀ B` for A (k × m) and B (k × n), all column-major: `gemm_tn`.
pub(super) fn gemm_tn(a: &[f64], b: &[f64], k: usize, m: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for r in 0..k {
                sum += a[r + (k * i)] * b[r + (k * j)];
            }
            c[i + (m * j)] = sum;
        }
    }
    c
}

/// `C = A B` for A (m × k) and B (k × n), all column-major: `gemm_nn`.
pub(super) fn gemm_nn(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for r in 0..k {
            let w = b[r + (k * j)];
            for i in 0..m {
                c[i + (m * j)] += a[i + (m * r)] * w;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_of_a_two_by_three() {
        // A = [1 2 3; 4 5 6] column-major, B = Aᵀ.
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // A Aᵀ = [14 32; 32 77].
        assert_eq!(gemm_nn(&a, &b, 2, 3, 2), [14.0, 32.0, 32.0, 77.0]);
        // Aᵀ A with A as k × m = 2 × 3.
        assert_eq!(
            gemm_tn(&a, &a, 2, 3, 3),
            [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
        );
    }
}
