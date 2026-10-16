//! Binary Bernoulli probit regression with Fisher scoring; no generic optimizer or GLM framework.
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{ErrorKind, StatsError};

#[derive(Debug, Clone)]
pub struct ProbitResult {
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

// A continued fraction for the normal Mills ratio keeps log tails finite
// beyond the range where Phi(x) underflows. Ordinary values use statrs erfc.
fn log_probability(value: f64) -> f64 {
    if value < -8.0 {
        let x = -value;
        let mut denominator = x;
        for k in (1..=100).rev() {
            denominator = x + k as f64 / denominator;
        }
        -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln() - denominator.ln()
    } else {
        (0.5 * statrs::function::erf::erfc(-value / std::f64::consts::SQRT_2)).ln()
    }
}

fn probability(value: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-value / std::f64::consts::SQRT_2)
}

fn log_density(value: f64) -> f64 {
    -0.5 * value * value - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

fn root_weight(value: f64) -> f64 {
    (log_density(value) - 0.5 * (log_probability(value) + log_probability(-value))).exp()
}

fn separated(eta: &DVector<f64>, y: &DVector<f64>) -> bool {
    // This vector itself certifies complete/quasi-complete separation if every
    // signed margin is nonnegative and at least one is strictly positive.
    // It is a certificate check, not an exhaustive linear-programming detector.
    if eta
        .iter()
        .zip(y)
        .any(|(e, v)| *v > 0.0 && *v < 1.0 && *e != 0.0)
    {
        return false;
    }
    let margins: Vec<f64> = eta
        .iter()
        .zip(y.iter())
        .map(|(e, v)| (2.0 * v - 1.0) * e)
        .collect();
    margins.iter().all(|m| *m >= 0.0) && margins.iter().any(|m| *m > 0.0)
}

fn weighted_likelihood(eta: &DVector<f64>, y: &DVector<f64>, weights: &[f64]) -> f64 {
    eta.iter()
        .zip(y.iter())
        .zip(weights)
        .map(|((e, v), w)| w * (v * log_probability(*e) + (1.0 - v) * log_probability(-*e)))
        .sum()
}
// Independent of beta: compute the grouped-binomial likelihood constant once.
fn likelihood_constant(y: &DVector<f64>, weights: &[f64]) -> f64 {
    y.iter()
        .zip(weights)
        .map(|(v, w)| {
            if *v == 0.0 || *v == 1.0 {
                return 0.0;
            }
            let successes = w * v;
            statrs::function::gamma::ln_gamma(w + 1.0)
                - statrs::function::gamma::ln_gamma(successes + 1.0)
                - statrs::function::gamma::ln_gamma(w - successes + 1.0)
        })
        .sum()
}
fn saturated_likelihood(y: &DVector<f64>, weights: &[f64]) -> f64 {
    y.iter()
        .zip(weights)
        .map(|(v, w)| {
            let logp = if *v == 0.0 { 0.0 } else { v * v.ln() };
            let logq = if *v == 1.0 {
                0.0
            } else {
                (1.0 - v) * (-v).ln_1p()
            };
            w * (logp + logq)
        })
        .sum()
}

pub fn fit_probit(x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<ProbitResult, StatsError> {
    fit_with_limit(x, y, 100)
}

fn fit_with_limit(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    max_iterations: usize,
) -> Result<ProbitResult, StatsError> {
    fit_data_limit(x, y, vec![], vec![], max_iterations, true)
}

pub fn fit_probit_with_data(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
) -> Result<ProbitResult, StatsError> {
    fit_data_limit(x, y, weights, offset, 100, true)
}

fn fit_data_limit(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    max_iterations: usize,
    compute_null: bool,
) -> Result<ProbitResult, StatsError> {
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
            "probit regression requires n > p",
        ));
    }
    if x.iter().flatten().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(error(ErrorKind::Nonfinite, "x and y must be finite"));
    }
    if y.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(error(
            ErrorKind::InvalidInput,
            "x must be finite and y must be finite proportions in [0,1], with integer successes under the supplied trial weights",
        ));
    }
    let (weights, offset) = crate::observations::prepare(n, weights, offset, true)?;
    if y.iter()
        .zip(&weights)
        .any(|(v, w)| (w * v - (w * v).round()).abs() > 1e-8)
    {
        return Err(error(
            ErrorKind::InvalidInput,
            "binary/grouped-binomial outcomes require integer trial counts and successes",
        ));
    }
    // Accepted count-rounding tolerance is representation tolerance, not fractional events.
    let y: Vec<f64> = y
        .iter()
        .zip(&weights)
        .map(|(v, w)| (v * w).round() / w)
        .collect();
    if y.iter().all(|v| *v == 0.0) || y.iter().all(|v| *v == 1.0) {
        return Err(error(
            ErrorKind::InvalidInput,
            "probit regression requires both outcome classes",
        ));
    }
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
    // Preserve the ordinary zero start; use a finite intercept only for extreme offsets.
    // Without an explicit ones column the existing zero-coefficient start remains.
    let mut eta = offset_vector.clone();
    if (offset.iter().any(|o| o.abs() > 8.0))
        && let Some(j) = (0..p).find(|j| design.column(*j).iter().all(|v| *v == 1.0))
    {
        let mean = outcomes
            .iter()
            .zip(&weights)
            .map(|(y, w)| y * w)
            .sum::<f64>()
            / weights.iter().sum::<f64>();
        let center = offset.iter().map(|o| o / n as f64).sum::<f64>();
        let link = Normal::new(0.0, 1.0)
            .map_err(|_| error(ErrorKind::Internal, "invalid normal distribution"))?
            .inverse_cdf(mean);
        theta[j] = (link - center) * scales[j];
        eta = &scaled * &theta + &offset_vector;
    }
    let mut likelihood = weighted_likelihood(&eta, &outcomes, &weights);
    let mut converged_at = None;
    for iteration in 1..=max_iterations {
        // Fisher scoring uses expected information, matching R glm(probit).
        let root_weights = eta.map(root_weight);
        if root_weights.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorKind::Nonfinite,
                "probit information exceeds floating-point range",
            ));
        }
        let weighted = DMatrix::from_fn(n, p, |i, j| {
            scaled[(i, j)] * root_weights[i] * weights[i].sqrt()
        });
        // QR first; the SVD of the small R has the same singular values and V.
        let qr = weighted.qr();
        let svd = qr.r().svd(true, true);
        let tolerance = f64::EPSILON * n.max(p) as f64 * svd.singular_values[0];
        if svd.singular_values.iter().any(|s| *s <= tolerance) {
            return Err(error(
                ErrorKind::Nonconvergence,
                "design/information is rank deficient; separation may prevent a finite MLE",
            ));
        }
        // Solve the Fisher-scoring increment as a weighted least-squares problem:
        // (A' W A) delta = A' score; W = phi(eta)^2/[Phi(eta)Phi(-eta)]. SVD avoids forming/inverting A' W A.
        let rhs = DVector::from_iterator(
            n,
            (0..n).map(|i| {
                let score = outcomes[i] * (log_density(eta[i]) - log_probability(eta[i])).exp()
                    - (1.0 - outcomes[i]) * (log_density(eta[i]) - log_probability(-eta[i])).exp();
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
                "probit calculation overflowed; separation or extreme scaling may be present",
            ));
        }
        let mut rhs = rhs;
        qr.q_tr_mul(&mut rhs);
        let step = svd.solve(&rhs.rows(0, p), tolerance).map_err(|_| {
            error(
                ErrorKind::FactorizationFailure,
                "least-squares factorization failed",
            )
        })?;
        if step.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorKind::Nonfinite,
                "probit step exceeds floating-point range",
            ));
        }
        // Concave likelihood: halve an overlarge scoring step until it improves.
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
                "probit line search failed to improve the likelihood",
            )
        })?;
        if separated(&(&next_eta - &offset_vector), &outcomes) {
            return Err(error(
                ErrorKind::Nonconvergence,
                "complete or quasi-complete separation: no finite unpenalized probit MLE",
            ));
        }
        // Require a small *full* Newton step, not merely a tiny damped step or
        // a flat likelihood that can falsely converge under separation.
        let small_step = (0..p).all(|j| step[j].abs() <= 1e-11 * (1.0 + next_theta[j].abs()));
        let small_likelihood_change =
            (next_likelihood - likelihood).abs() <= 1e-13 * (1.0 + likelihood.abs());
        theta = next_theta;
        eta = next_eta;
        likelihood = next_likelihood;
        if small_step && small_likelihood_change {
            converged_at = Some(iteration);
            break;
        }
    }
    let iterations = converged_at.ok_or_else(|| {
        error(ErrorKind::Nonconvergence, &format!("probit regression did not converge in {max_iterations} iterations; separation may be present"))
    })?;
    let coefficients = DVector::from_iterator(p, (0..p).map(|j| theta[j] / scales[j]));
    // Information is evaluated at the final estimate, not the preceding iterate.
    let weighted = DMatrix::from_fn(n, p, |i, j| {
        scaled[(i, j)] * root_weight(eta[i]) * weights[i].sqrt()
    });
    if weighted.iter().any(|v| !v.is_finite()) {
        return Err(error(
            ErrorKind::Nonfinite,
            "final probit information exceeds floating-point range",
        ));
    }
    let svd = weighted.qr().r().svd(false, true);
    let tolerance = f64::EPSILON * n.max(p) as f64 * svd.singular_values[0];
    if svd.singular_values.iter().any(|s| *s <= tolerance) {
        return Err(error(
            ErrorKind::RankDeficiency,
            "final probit information is numerically rank deficient",
        ));
    }
    let vt = svd.v_t.as_ref().ok_or_else(|| {
        error(
            ErrorKind::Internal,
            "SVD did not return right singular vectors",
        )
    })?;
    // Var(beta)=(X' W X)^-1. Bernoulli dispersion is fixed at 1; no RSS/(n-p).
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
            "probit inference exceeds floating-point range",
        ));
    }
    let fitted: Vec<f64> = eta.iter().map(|e| probability(*e)).collect();
    let residuals = outcomes
        .iter()
        .zip(fitted.iter())
        .map(|(v, probability)| v - probability)
        .collect();
    let constant = likelihood_constant(&outcomes, &weights);
    let null_likelihood = if !compute_null {
        likelihood
    } else if intercept {
        // With offsets, the null intercept needs its own one-coefficient fit.
        fit_data_limit(
            vec![vec![1.0]; n],
            outcomes.iter().copied().collect(),
            weights.clone(),
            offset.clone(),
            max_iterations,
            false,
        )?
        .log_likelihood
            - constant
    } else {
        weighted_likelihood(&offset_vector, &outcomes, &weights)
    };
    let saturated = saturated_likelihood(&outcomes, &weights);
    if [constant, likelihood + constant, saturated, null_likelihood]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(error(
            ErrorKind::Nonfinite,
            "binomial likelihood statistics exceed floating-point range",
        ));
    }
    let gap_tolerance = 64.0 * f64::EPSILON * (1.0 + saturated.abs() + likelihood.abs());
    if saturated - likelihood < -gap_tolerance || saturated - null_likelihood < -gap_tolerance {
        return Err(error(
            ErrorKind::Internal,
            "binomial likelihood exceeds saturated likelihood beyond rounding precision",
        ));
    }
    if [
        2.0 * (saturated - likelihood),
        2.0 * (saturated - null_likelihood),
        2.0 * p as f64 - 2.0 * (likelihood + constant),
    ]
    .iter()
    .any(|v| !v.is_finite())
    {
        return Err(error(
            ErrorKind::Nonfinite,
            "binomial deviance or AIC exceeds floating-point range",
        ));
    }
    Ok(ProbitResult {
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
        log_likelihood: likelihood + constant,
        deviance: (2.0 * (saturated - likelihood)).max(0.0),
        null_deviance: (2.0 * (saturated - null_likelihood)).max(0.0),
        aic: 2.0 * p as f64 - 2.0 * (likelihood + constant),
        df_residual: n - p,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_functions_handle_extreme_logits() {
        assert_eq!(probability(-1000.0), 0.0);
        assert_eq!(probability(1000.0), 1.0);
        // scipy.special.log_ndtr references, including underflowing Phi.
        for (x, expected) in [
            (-10.0, -53.23128515051248),
            (-40.0, -804.6084420137539),
            (-1000.0, -500007.8266948122),
        ] {
            assert!((log_probability(x) - expected).abs() < 1e-9);
        }
        assert!((log_probability(-8.000001) - log_probability(-7.999999)).abs() < 0.00002);
    }

    #[test]
    fn refuses_a_fit_when_the_iteration_budget_is_exhausted() {
        assert!(
            fit_with_limit(
                vec![vec![1., 0.], vec![1., 1.], vec![1., 2.], vec![1., 3.]],
                vec![0., 1., 0., 0.],
                1
            )
            .unwrap_err()
            .0
            .contains("did not converge")
        );
    }
}
