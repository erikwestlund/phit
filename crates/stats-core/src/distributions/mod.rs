//! Probability distributions: densities, log densities, CDFs, survival
//! functions S(x) = P(X > x), their logarithms and quantiles.
//!
//! Tail probabilities come from cdflib's ports of TOMS 708 (incomplete beta),
//! TOMS 654 (incomplete gamma) and Cody's normal CDF, which compute the lower
//! and upper tails independently. This module adds log-scale tails beyond the
//! double range, Loader's saddle-point densities, accurate quantiles and the
//! input/result policy shared by the scalar and batch PHP functions:
//!
//! - parameters must be finite and in their documented domain;
//! - evaluation points may be ±∞ but not NaN;
//! - quantile probabilities must lie in [0, 1];
//! - out-of-support points give density 0 (log density -∞), never an error;
//! - a NaN result is a numerical failure and is reported as an error.
mod continuous;
mod discrete;
mod quantile;
mod rng;
mod special;

pub use continuous::{
    Beta, ChiSquare, Exponential, FisherF, Gamma, LogNormal, Logistic, Normal, StudentT,
};
pub use discrete::{Binomial, MAX_BINOMIAL_SIZE, NegativeBinomial, Poisson};
pub use rng::{MAX_RNG_DRAWS, sample_quantiles};

use crate::{ErrorKind, StatsError};

pub(crate) fn finite(name: &str, value: f64) -> Result<f64, StatsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StatsError::new(
            ErrorKind::Nonfinite,
            format!("{name} must be finite"),
        ))
    }
}

pub(crate) fn positive(name: &str, value: f64) -> Result<f64, StatsError> {
    if finite(name, value)? > 0.0 {
        Ok(value)
    } else {
        Err(StatsError::new(
            ErrorKind::InvalidInput,
            format!("{name} must be positive"),
        ))
    }
}

fn result(value: f64) -> Result<f64, StatsError> {
    if value.is_nan() {
        Err(StatsError::new(
            ErrorKind::Nonconvergence,
            "numerical evaluation failed for these arguments",
        ))
    } else {
        Ok(value)
    }
}

/// Evaluate a density, CDF or survival function at x (±∞ allowed, NaN rejected).
pub fn evaluate(x: f64, f: impl Fn(f64) -> f64) -> Result<f64, StatsError> {
    if x.is_nan() {
        return Err(StatsError::new(
            ErrorKind::Nonfinite,
            "evaluation points must not be NaN",
        ));
    }
    result(f(x))
}

/// Evaluate a quantile function at a probability in [0, 1].
pub fn evaluate_quantile(p: f64, f: impl Fn(f64) -> f64) -> Result<f64, StatsError> {
    if p.is_nan() {
        return Err(StatsError::new(
            ErrorKind::Nonfinite,
            "probabilities must not be NaN",
        ));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::new(
            ErrorKind::InvalidInput,
            "probabilities must be in [0, 1]",
        ));
    }
    result(f(p))
}

/// Batch form of [`evaluate`]: one output per input, in order.
pub fn evaluate_all(values: &[f64], f: impl Fn(f64) -> f64) -> Result<Vec<f64>, StatsError> {
    values.iter().map(|&x| evaluate(x, &f)).collect()
}

/// Batch form of [`evaluate_quantile`]: one output per input, in order.
pub fn evaluate_quantiles(
    probabilities: &[f64],
    f: impl Fn(f64) -> f64,
) -> Result<Vec<f64>, StatsError> {
    probabilities
        .iter()
        .map(|&p| evaluate_quantile(p, &f))
        .collect()
}
