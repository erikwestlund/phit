//! Poisson count regression; no generic optimizer or GLM framework.
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{ErrorKind, StatsError};

#[derive(Debug, Clone)]
pub struct PoissonResult {
    pub coefficients: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_int: Vec<[f64; 2]>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub aic: f64,
    pub df_residual: usize,
    pub iterations: usize,
}

fn error(kind: ErrorKind, message: &str) -> StatsError {
    StatsError::new(kind, message)
}

#[cfg(test)]
fn deviance(eta: &DVector<f64>, y: &DVector<f64>) -> f64 {
    // exp_m1(d)-d preserves precision when a fitted mean is close to y.
    2.0 * eta
        .iter()
        .zip(y.iter())
        .map(|(&e, &v)| {
            if v == 0.0 {
                e.exp()
            } else {
                let d = e - v.ln();
                v * (d.exp_m1() - d)
            }
        })
        .sum::<f64>()
}

fn weighted_likelihood(eta: &DVector<f64>, y: &DVector<f64>, weights: &[f64]) -> f64 {
    eta.iter()
        .zip(y.iter())
        .zip(weights)
        .map(|((e, v), w)| {
            w * if *v >= 16.0 {
                // Log mass via deviance + Stirling error avoids subtracting O(y log y)
                // terms when a large fitted mean is close to the observed count.
                let d = e - v.ln();
                let inv = 1.0 / v;
                let inv2 = inv * inv;
                let correction = inv
                    * (1.0 / 12.0
                        + inv2
                            * (-1.0 / 360.0
                                + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0 + inv2 / 1188.0))));
                -v * (d.exp_m1() - d)
                    - 0.5 * ((2.0 * std::f64::consts::PI).ln() + v.ln())
                    - correction
            } else {
                v * e - e.exp() - statrs::function::gamma::ln_gamma(v + 1.0)
            }
        })
        .sum()
}

fn weighted_deviance(eta: &DVector<f64>, y: &DVector<f64>, weights: &[f64]) -> f64 {
    2.0 * eta
        .iter()
        .zip(y.iter())
        .zip(weights)
        .map(|((e, v), w)| {
            w * if *v == 0.0 {
                e.exp()
            } else {
                let d = e - v.ln();
                v * (d.exp_m1() - d)
            }
        })
        .sum::<f64>()
}

// Log-sum-exp keeps exposures finite even for offsets beyond exp()'s range.
fn log_sum_exp(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}
fn null_intercept(y: &DVector<f64>, weights: &[f64], offset: &[f64]) -> f64 {
    log_sum_exp(
        y.iter()
            .zip(weights)
            .filter(|(v, _)| **v > 0.0)
            .map(|(v, w)| v.ln() + w.ln()),
    ) - log_sum_exp(offset.iter().zip(weights).map(|(o, w)| o + w.ln()))
}

pub fn fit_poisson(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<PoissonResult, StatsError> {
    fit_with_limit(x, y, 100)
}

fn fit_with_limit(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    max_iterations: usize,
) -> Result<PoissonResult, StatsError> {
    fit_data_limit(x, y, vec![], vec![], max_iterations, true)
}

pub fn fit_poisson_with_data(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
) -> Result<PoissonResult, StatsError> {
    fit_data_limit(x, y, weights, offset, 100, true)
}

fn fit_data_limit(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    max_iterations: usize,
    compute_null: bool,
) -> Result<PoissonResult, StatsError> {
    let n = x.len();
    if n == 0 || x[0].is_empty() {
        return Err(error(
            ErrorKind::InvalidInput,
            "x must have at least one row and column",
        ));
    }
    let p = x[0].len();
    if y.len() != n || x.iter().any(|row| row.len() != p) {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "x must be rectangular with one y value per row",
        ));
    }
    if n <= p {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "poisson regression requires n > p",
        ));
    }
    if x.iter().flatten().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(error(ErrorKind::Nonfinite, "x and y must be finite"));
    }
    if y.iter().any(|v| *v < 0.0 || v.fract() != 0.0) {
        return Err(error(
            ErrorKind::InvalidInput,
            "x must be finite and y must contain nonnegative integer counts",
        ));
    }
    if y.iter().all(|v| *v == 0.0) {
        return Err(error(
            ErrorKind::InvalidInput,
            "poisson regression requires at least one positive count",
        ));
    }
    let (weights, offset) = crate::observations::prepare(n, weights, offset, false)?;
    let intercept = (0..p).any(|j| x.iter().all(|row| row[j] == 1.0));
    let design = DMatrix::from_row_iterator(n, p, x.into_iter().flatten());
    let outcomes = DVector::from_vec(y);
    let scales: Vec<f64> = (0..p).map(|j| design.column(j).norm()).collect();
    if scales.iter().any(|s| !s.is_finite() || *s == 0.0) {
        return Err(error(
            ErrorKind::RankDeficiency,
            "design is rank deficient or exceeds floating-point range",
        ));
    }
    let scaled = DMatrix::from_fn(n, p, |i, j| design[(i, j)] / scales[j]);
    let mut theta = DVector::zeros(p);
    let offset_vector = DVector::from_vec(offset.clone());
    // Preserve the ordinary start and its iteration path for established datasets.
    // Extreme exposures/counts need a finite intercept before exp() and IRLS.
    let mut eta = offset_vector.clone();
    if (offset.iter().any(|o| o.abs() > 20.0) || outcomes.iter().any(|y| *y > 1e8))
        && let Some(j) = (0..p).find(|j| design.column(*j).iter().all(|v| *v == 1.0))
    {
        theta[j] = null_intercept(&outcomes, &weights, &offset) * scales[j];
        eta = &scaled * &theta + &offset_vector;
    }
    let mut likelihood = weighted_likelihood(&eta, &outcomes, &weights);
    let mut converged_at = None;
    for iteration in 1..=max_iterations {
        let root_weights = eta.map(|e: f64| (0.5 * e).exp());
        if root_weights.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorKind::Nonfinite,
                "Poisson information exceeds floating-point range",
            ));
        }
        let weighted = DMatrix::from_fn(n, p, |i, j| {
            scaled[(i, j)] * root_weights[i] * weights[i].sqrt()
        });
        let svd = weighted.svd(true, true);
        let tolerance = f64::EPSILON * n.max(p) as f64 * svd.singular_values[0];
        if svd.singular_values.iter().any(|s| *s <= tolerance) {
            return Err(error(
                ErrorKind::Nonconvergence,
                "design/information is rank deficient; a boundary MLE may prevent a finite fit",
            ));
        }
        // Solve the Newton increment as a weighted least-squares problem:
        // (A' W A) delta = A'(y-mu). SVD avoids forming/inverting A' W A.
        let rhs = DVector::from_iterator(
            n,
            (0..n).map(|i| {
                let score = outcomes[i] - eta[i].exp();
                if root_weights[i] == 0.0 && score == 0.0 {
                    0.0
                } else {
                    weights[i].sqrt() * score / root_weights[i]
                }
            }),
        );
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorKind::Nonconvergence,
                "poisson calculation overflowed; a boundary MLE or extreme scaling may be present",
            ));
        }
        let step = svd.solve(&rhs, tolerance).map_err(|_| {
            error(
                ErrorKind::FactorizationFailure,
                "least-squares factorization failed",
            )
        })?;
        if step.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorKind::Nonfinite,
                "poisson step exceeds floating-point range",
            ));
        }
        // Concave likelihood: halve an overlarge Newton step until it improves.
        let mut fraction = 1.0;
        let mut accepted = None;
        for _ in 0..31 {
            let candidate = &theta + fraction * &step;
            let candidate_eta = &scaled * &candidate + &offset_vector;
            let candidate_likelihood = weighted_likelihood(&candidate_eta, &outcomes, &weights);
            if candidate_likelihood.is_finite()
                && candidate_likelihood
                    >= likelihood - 16.0 * f64::EPSILON * (1.0 + likelihood.abs())
            {
                accepted = Some((candidate, candidate_eta, candidate_likelihood));
                break;
            }
            fraction *= 0.5;
        }
        let (next_theta, next_eta, next_likelihood) = accepted.ok_or_else(|| {
            error(
                ErrorKind::Nonconvergence,
                "poisson line search failed to improve the likelihood",
            )
        })?;
        // Require a small *full* Newton step, not merely a tiny damped step or
        // a flat likelihood at a boundary.
        let small_step = (0..p).all(|j| step[j].abs() <= 1e-8 * (1.0 + next_theta[j].abs()));
        let small_likelihood_change =
            (next_likelihood - likelihood).abs() <= 1e-10 * (1.0 + likelihood.abs());
        theta = next_theta;
        eta = next_eta;
        likelihood = next_likelihood;
        if small_step && small_likelihood_change {
            converged_at = Some(iteration);
            break;
        }
    }
    let iterations = converged_at.ok_or_else(|| {
        error(ErrorKind::Nonconvergence, &format!("poisson regression did not converge in {max_iterations} iterations; a boundary MLE or extreme scaling may be present"))
    })?;
    let coefficients = DVector::from_iterator(p, (0..p).map(|j| theta[j] / scales[j]));
    // Information is evaluated at the final estimate, not the preceding iterate.
    let weighted = DMatrix::from_fn(n, p, |i, j| {
        scaled[(i, j)] * (0.5 * eta[i]).exp() * weights[i].sqrt()
    });
    let svd = weighted.svd(false, true);
    let tolerance = f64::EPSILON * n.max(p) as f64 * svd.singular_values[0];
    if svd.singular_values.iter().any(|s| *s <= tolerance) {
        return Err(error(
            ErrorKind::RankDeficiency,
            "final poisson information is numerically rank deficient",
        ));
    }
    let vt = svd.v_t.as_ref().ok_or_else(|| {
        error(
            ErrorKind::Internal,
            "SVD did not return right singular vectors",
        )
    })?;
    // Var(beta)=(X' W X)^-1. Poisson dispersion is fixed at 1; no RSS/(n-p).
    let covariance = DMatrix::from_fn(p, p, |i, j| {
        (0..p)
            .map(|k| vt[(k, i)] * vt[(k, j)] / svd.singular_values[k].powi(2))
            .sum::<f64>()
            / scales[i]
            / scales[j]
    });
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| error(ErrorKind::Internal, "invalid normal distribution"))?;
    let critical = normal.inverse_cdf(0.975);
    let std_errors: Vec<f64> = (0..p).map(|j| covariance[(j, j)].sqrt()).collect();
    let z_statistics: Vec<f64> = (0..p).map(|j| coefficients[j] / std_errors[j]).collect();
    let p_values: Vec<f64> = z_statistics
        .iter()
        .map(|z| 2.0 * normal.sf(z.abs()))
        .collect();
    let conf_int: Vec<[f64; 2]> = (0..p)
        .map(|j| {
            [
                coefficients[j] - critical * std_errors[j],
                coefficients[j] + critical * std_errors[j],
            ]
        })
        .collect();
    if coefficients
        .iter()
        .chain(covariance.iter())
        .chain(std_errors.iter())
        .chain(z_statistics.iter())
        .chain(p_values.iter())
        .chain(conf_int.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(error(
            ErrorKind::Nonfinite,
            "poisson inference exceeds floating-point range",
        ));
    }
    let fitted: Vec<f64> = eta.iter().map(|e| e.exp()).collect();
    let residuals = outcomes
        .iter()
        .zip(fitted.iter())
        .map(|(v, mean)| v - mean)
        .collect();
    let null_eta = if intercept && compute_null {
        let log_mean = null_intercept(&outcomes, &weights, &offset);
        offset_vector.map(|o| o + log_mean)
    } else {
        offset_vector
    };
    let model_deviance = weighted_deviance(&eta, &outcomes, &weights);
    let null_deviance = weighted_deviance(&null_eta, &outcomes, &weights);
    let aic = 2.0 * p as f64 - 2.0 * likelihood;
    if [model_deviance, null_deviance, aic]
        .iter()
        .any(|v| !v.is_finite())
        || fitted.iter().any(|v| !v.is_finite())
    {
        return Err(error(
            ErrorKind::Nonfinite,
            "Poisson statistics exceed floating-point range",
        ));
    }
    Ok(PoissonResult {
        coefficients: coefficients.iter().copied().collect(),
        covariance: (0..p)
            .map(|i| (0..p).map(|j| covariance[(i, j)]).collect())
            .collect(),
        std_errors,
        z_statistics,
        p_values,
        conf_int,
        fitted,
        residuals,
        log_likelihood: likelihood,
        deviance: model_deviance,
        null_deviance,
        aic,
        df_residual: n - p,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deviance_preserves_small_positive_counts_and_zero_rows() {
        assert!(
            (deviance(
                &DVector::from_vec(vec![0., 0.]),
                &DVector::from_vec(vec![1., 0.])
            ) - 2.)
                .abs()
                < 1e-14
        );
        assert!(deviance(&DVector::from_vec(vec![1e-8]), &DVector::from_vec(vec![1.])) > 0.0);
    }
    #[test]
    fn refuses_a_fit_when_the_iteration_budget_is_exhausted() {
        assert!(
            fit_with_limit(
                vec![vec![1., 0.], vec![1., 1.], vec![1., 2.], vec![1., 3.]],
                vec![0., 1., 2., 3.],
                1
            )
            .unwrap_err()
            .0
            .contains("did not converge")
        );
    }
}
