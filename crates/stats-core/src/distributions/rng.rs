//! Explicit local ChaCha20 state and inverse-CDF batch sampling.
//! No global state, entropy access or changes to the estimator RNG dependencies.
use crate::{ErrorKind, StatsError};
use rand::{Rng, SeedableRng, distr::Open01};
use rand_chacha::ChaCha20Rng;

/// Bound a single returned numeric array's allocation/work.
pub const MAX_RNG_DRAWS: usize = 1_000_000;

/// Sample a validated distribution by mapping open-interval uniforms through
/// its quantile. Every call restarts the explicitly supplied seed; the same
/// seed/parameters reproduce prefixes within the pinned algorithm/version.
/// Nonfinite generated values reject the entire call (no partial output).
pub fn sample_quantiles(
    n: usize,
    seed: u64,
    quantile: impl Fn(f64) -> f64,
) -> Result<Vec<f64>, StatsError> {
    if n > MAX_RNG_DRAWS {
        return Err(StatsError::new(
            ErrorKind::Unsupported,
            "n must not exceed 1,000,000 samples per call",
        ));
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Open01 excludes 0/1, so ordinary unbounded-support quantile endpoints
        // cannot create artificial infinities. It uses 52 random bits for f64.
        let p: f64 = rng.sample(Open01);
        let value = quantile(p);
        if !value.is_finite() {
            return Err(StatsError::new(
                ErrorKind::Nonfinite,
                "sample exceeded floating-point range for these parameters",
            ));
        }
        out.push(value);
    }
    Ok(out)
}
