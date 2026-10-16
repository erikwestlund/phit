//! Intercept-only fits repeat a few likelihood terms many times, so plain summation
//! drifts by far more than the line-search floor; the exact Newton step must still be
//! accepted.
use stats_core::{fit_logistic, fit_poisson, fit_probit};

fn shuffled_outcomes(n: usize, ones: usize, seed: u64) -> Vec<f64> {
    let mut y: Vec<f64> = (0..n).map(|i| if i < ones { 1.0 } else { 0.0 }).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        y.swap(i, (state >> 33) as usize % (i + 1));
    }
    y
}

#[test]
fn intercept_only_glms_converge_at_large_n() {
    for (case, n) in [1_000usize, 5_000, 20_000, 50_000].into_iter().enumerate() {
        for k in 1..=12u64 {
            let ones = n * (5 + 7 * k as usize) / 100;
            let y = shuffled_outcomes(n, ones, k + 31 * case as u64);
            let x = vec![vec![1.0]; n];
            let mean = ones as f64 / n as f64;
            let logit = fit_logistic(x.clone(), y.clone())
                .unwrap_or_else(|e| panic!("logistic n={n} ones={ones}: {e:?}"));
            assert!((logit.coefficients[0] - (mean / (1.0 - mean)).ln()).abs() < 1e-9);
            fit_probit(x.clone(), y.clone())
                .unwrap_or_else(|e| panic!("probit n={n} ones={ones}: {e:?}"));
            let counts: Vec<f64> = y.iter().map(|v| v * 3.0).collect();
            let poisson = fit_poisson(x, counts)
                .unwrap_or_else(|e| panic!("poisson n={n} ones={ones}: {e:?}"));
            assert!((poisson.coefficients[0] - (3.0 * mean).ln()).abs() < 1e-9);
        }
    }
}
