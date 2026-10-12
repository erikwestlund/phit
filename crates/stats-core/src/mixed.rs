//! Gaussian random-intercept maximum likelihood, with one grouping factor.
use crate::{ErrorKind, StatsError};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MixedResult {
    pub coefficients: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub std_errors: Vec<f64>,
    pub z_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub conf_int: Vec<[f64; 2]>,
    pub fixed_fitted: Vec<f64>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub marginal_residuals: Vec<f64>,
    pub random_effects: Vec<f64>,
    pub group_ids: Vec<usize>,
    pub group_sizes: Vec<usize>,
    pub residual_variance: f64,
    pub random_intercept_variance: f64,
    pub sigma: f64,
    pub icc: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub df_residual: usize,
    pub evaluations: usize,
    pub boundary: bool,
}
fn error(kind: ErrorKind, message: &str) -> StatsError {
    StatsError::new(kind, message)
}

// Like lme4's penalized least-squares elimination, separate within-group
// deviations from group means. Only the mean directions depend on rho.
// Mathematical reference: https://lme4.github.io/lme4/articles/PLSvGLS.pdf
// Cache a square-root summary: QR of [centered X, centered y] preserves every
// residual norm, without subtracting nearly equal cross-products X'X - B'B.
struct ProfileData {
    within: DMatrix<f64>,
    means: DMatrix<f64>, // sqrt(m_g) times each group's [X, y] mean
    sizes: Vec<usize>,
    n: usize,
    p: usize,
}
impl ProfileData {
    fn new(design: &DMatrix<f64>, scales: &[f64], y: &[f64], members: &[Vec<usize>]) -> Self {
        let n = design.nrows();
        let p = design.ncols();
        let mut centered = DMatrix::from_fn(n, p + 1, |i, j| {
            if j == p {
                y[i]
            } else {
                design[(i, j)] / scales[j]
            }
        });
        let mut means = DMatrix::zeros(members.len(), p + 1);
        for (g, rows) in members.iter().enumerate() {
            for j in 0..=p {
                // Sum deviations from one observation to retain precision when
                // measurements have a large common offset.
                let origin = centered[(rows[0], j)];
                let mean = origin
                    + rows.iter().map(|i| centered[(*i, j)] - origin).sum::<f64>()
                        / rows.len() as f64;
                means[(g, j)] = mean * (rows.len() as f64).sqrt();
                for i in rows {
                    centered[(*i, j)] -= mean;
                }
            }
        }
        Self {
            within: centered.qr().unpack_r(),
            means,
            sizes: members.iter().map(Vec::len).collect(),
            n,
            p,
        }
    }

    // ||C^-1/2 (y-X beta)||^2 equals the residual norm in this compact
    // (p+1+G)-row system. No observation-sized allocation in the profile loop.
    fn weighted(&self, rho: f64) -> DMatrix<f64> {
        let rows = self.within.nrows();
        DMatrix::from_fn(rows + self.sizes.len(), self.p + 1, |i, j| {
            if i < rows {
                self.within[(i, j)]
            } else {
                let g = i - rows;
                self.means[(g, j)] / (1.0 + self.sizes[g] as f64 * rho).sqrt()
            }
        })
    }
}
struct Profile {
    beta: DVector<f64>,
    information_inverse: Option<DMatrix<f64>>,
    variance: f64,
    objective: f64,
    derivative: f64,
}
fn profile(data: &ProfileData, t: f64, inference: bool) -> Result<Profile, StatsError> {
    let rho = t.exp_m1();
    let n = data.n;
    let p = data.p;
    let compact = data.weighted(rho);
    let a = compact.columns(0, p).into_owned();
    let b = compact.column(p).into_owned();
    let svd = a.svd(true, true);
    // Keep the original sample-size-based rank threshold after compression.
    let tolerance = f64::EPSILON * n.max(p) as f64 * svd.singular_values[0];
    if svd
        .singular_values
        .iter()
        .any(|s| !s.is_finite() || *s <= tolerance)
    {
        return Err(error(
            ErrorKind::RankDeficiency,
            "mixed-model fixed design/information is numerically rank deficient",
        ));
    }
    let beta = svd.solve(&b, tolerance).map_err(|_| {
        error(
            ErrorKind::FactorizationFailure,
            "least-squares factorization failed",
        )
    })?;
    let residual = b - compact.columns(0, p) * &beta;
    let q = residual.norm_squared();
    if !q.is_finite() || q <= 0.0 {
        return Err(error(
            ErrorKind::Nonfinite,
            "mixed-model residual variance is zero or exceeds floating-point range",
        ));
    }
    let variance = q / n as f64; // ML scale, not RSS/(n-p), and not REML.
    let logdet = data
        .sizes
        .iter()
        .map(|m| (*m as f64 * rho).ln_1p())
        .sum::<f64>();
    let objective =
        0.5 * (n as f64 * ((2.0 * std::f64::consts::PI).ln() + variance.ln() + 1.0) + logdet);
    // Envelope theorem: q' uses the group residual sums, reconstructed from
    // cached means; no d beta/d rho and no second pass over observations.
    let q_derivative = -(0..data.sizes.len())
        .map(|g| {
            let mean_residual =
                data.means[(g, p)] - (0..p).map(|j| data.means[(g, j)] * beta[j]).sum::<f64>();
            data.sizes[g] as f64 * (mean_residual / (1.0 + data.sizes[g] as f64 * rho)).powi(2)
        })
        .sum::<f64>();
    let logdet_derivative = data
        .sizes
        .iter()
        .map(|m| *m as f64 / (1.0 + *m as f64 * rho))
        .sum::<f64>();
    let derivative = 0.5 * (1.0 + rho) * (n as f64 * q_derivative / q + logdet_derivative);
    if !objective.is_finite() || !derivative.is_finite() {
        return Err(error(
            ErrorKind::Nonfinite,
            "mixed-model likelihood exceeds floating-point range",
        ));
    }
    // Covariance is needed once, at the selected optimum, not at every trial.
    let information_inverse = if inference {
        let vt = svd.v_t.as_ref().ok_or_else(|| {
            error(
                ErrorKind::Internal,
                "SVD did not return right singular vectors",
            )
        })?;
        Some(DMatrix::from_fn(p, p, |i, j| {
            (0..p)
                .map(|k| vt[(k, i)] * vt[(k, j)] / svd.singular_values[k].powi(2))
                .sum()
        }))
    } else {
        None
    };
    Ok(Profile {
        beta,
        information_inverse,
        variance,
        objective,
        derivative,
    })
}

/// Gaussian y_i = x_i' beta + b_group + epsilon_i, b~N(0,tau^2), epsilon~N(0,sigma^2).
/// Group identifiers are arbitrary usize values; results follow first appearance.
pub fn fit_random_intercept(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    groups: Vec<usize>,
) -> Result<MixedResult, StatsError> {
    let n = x.len();
    if n == 0 || x[0].is_empty() {
        return Err(error(
            ErrorKind::InvalidInput,
            "x must have at least one row and column",
        ));
    }
    let p = x[0].len();
    if y.len() != n || groups.len() != n || x.iter().any(|r| r.len() != p) {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "x must be rectangular with one y and group per row",
        ));
    }
    if n <= p {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "mixed model requires n > p",
        ));
    }
    if x.iter().flatten().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(error(ErrorKind::Nonfinite, "x and y must be finite"));
    }
    let mut group_ids = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut map = HashMap::new();
    let mut indices = Vec::with_capacity(n);
    for (i, id) in groups.into_iter().enumerate() {
        let index = *map.entry(id).or_insert_with(|| {
            group_ids.push(id);
            members.push(Vec::new());
            members.len() - 1
        });
        members[index].push(i);
        indices.push(index);
    }
    if members.len() < 2 || members.len() == n {
        return Err(error(
            ErrorKind::InvalidInput,
            "need at least two groups and at least one repeated group to identify variances",
        ));
    }
    let design = DMatrix::from_row_iterator(n, p, x.into_iter().flatten());
    let outcomes = DMatrix::from_column_slice(n, 1, &y);
    let scales: Vec<f64> = (0..p).map(|j| design.column(j).norm()).collect();
    if scales.iter().any(|s| !s.is_finite() || *s == 0.0) {
        return Err(error(
            ErrorKind::RankDeficiency,
            "fixed design is rank deficient or exceeds floating-point range",
        ));
    }
    let data = ProfileData::new(&design, &scales, &y, &members);
    let at_zero = profile(&data, 0.0, false)?;
    // Catch numerically perfect fixed fits: the Gaussian ML scale has no positive optimum.
    if at_zero.variance
        <= f64::EPSILON.powi(2) * (n as f64).powi(2) * outcomes.norm_squared() / n as f64
    {
        return Err(error(
            ErrorKind::InvalidInput,
            "numerically perfect fit: no positive residual-variance MLE",
        ));
    }
    let mut best_t = 0.0;
    let mut best = at_zero.objective;
    let mut previous_t = 0.0;
    let mut previous_derivative = at_zero.derivative;
    let mut evaluations = 1;
    // Method-specific one-dimensional profile search, including the exact tau=0
    // boundary. Scan a fixed log-ratio grid, refine every negative-to-positive
    // score crossing, then select the lowest likelihood objective encountered.
    for k in 1..=64 {
        let t = k as f64 * 0.5;
        let value = profile(&data, t, false)?;
        evaluations += 1;
        if previous_derivative < 0.0 && value.derivative >= 0.0 {
            let mut lower = previous_t;
            let mut upper = t;
            for _ in 0..40 {
                let middle = (lower + upper) / 2.0;
                let candidate = profile(&data, middle, false)?;
                evaluations += 1;
                if candidate.derivative < 0.0 {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            let candidate_t = (lower + upper) / 2.0;
            let candidate = profile(&data, candidate_t, false)?;
            evaluations += 1;
            if candidate.objective < best {
                best = candidate.objective;
                best_t = candidate_t;
            }
        }
        if k == 64 && (value.derivative <= 0.0 || value.objective < best) {
            return Err(error(
                ErrorKind::Nonconvergence,
                "mixed-model optimum exceeds variance-ratio search range; a zero residual-variance boundary may be present",
            ));
        }
        previous_t = t;
        previous_derivative = value.derivative;
    }
    let final_fit = profile(&data, best_t, true)?;
    evaluations += 1;
    let rho = best_t.exp_m1();
    let variance = final_fit.variance;
    let tau = variance * rho;
    let beta = DVector::from_iterator(p, (0..p).map(|j| final_fit.beta[j] / scales[j]));
    let information_inverse = final_fit.information_inverse.as_ref().ok_or_else(|| {
        error(
            ErrorKind::InvalidInput,
            "final mixed-model covariance was not computed",
        )
    })?;
    let covariance = DMatrix::from_fn(p, p, |i, j| {
        variance * information_inverse[(i, j)] / scales[i] / scales[j]
    });
    let std_errors: Vec<f64> = (0..p).map(|j| covariance[(j, j)].sqrt()).collect();
    let z_statistics: Vec<f64> = (0..p).map(|j| beta[j] / std_errors[j]).collect();
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| error(ErrorKind::Internal, "invalid normal distribution"))?;
    let critical = normal.inverse_cdf(0.975);
    let p_values: Vec<f64> = z_statistics
        .iter()
        .map(|z| 2.0 * normal.sf(z.abs()))
        .collect();
    let conf_int: Vec<[f64; 2]> = (0..p)
        .map(|j| {
            [
                beta[j] - critical * std_errors[j],
                beta[j] + critical * std_errors[j],
            ]
        })
        .collect();
    let fixed_fitted = &design * &beta;
    let marginal = outcomes.column(0) - &fixed_fitted;
    let random_effects: Vec<f64> = members
        .iter()
        .map(|g| {
            let mean = g.iter().map(|i| marginal[*i]).sum::<f64>() / g.len() as f64;
            let shrinkage = (g.len() as f64 * rho) / (1.0 + g.len() as f64 * rho);
            shrinkage * mean
        })
        .collect();
    let fitted: Vec<f64> = (0..n)
        .map(|i| fixed_fitted[i] + random_effects[indices[i]])
        .collect();
    let residuals: Vec<f64> = (0..n).map(|i| y[i] - fitted[i]).collect();
    if beta
        .iter()
        .chain(covariance.iter())
        .chain(std_errors.iter())
        .chain(z_statistics.iter())
        .chain(p_values.iter())
        .chain(conf_int.iter().flatten())
        .chain(random_effects.iter())
        .chain(fitted.iter())
        .chain(residuals.iter())
        .chain(marginal.iter())
        .any(|v| !v.is_finite())
        || !tau.is_finite()
    {
        return Err(error(
            ErrorKind::Nonfinite,
            "mixed-model inference exceeds floating-point range",
        ));
    }
    Ok(MixedResult {
        coefficients: beta.iter().copied().collect(),
        covariance: (0..p)
            .map(|i| (0..p).map(|j| covariance[(i, j)]).collect())
            .collect(),
        std_errors,
        z_statistics,
        p_values,
        conf_int,
        fixed_fitted: fixed_fitted.iter().copied().collect(),
        fitted,
        residuals,
        marginal_residuals: marginal.iter().copied().collect(),
        random_effects,
        group_ids,
        group_sizes: members.iter().map(Vec::len).collect(),
        residual_variance: variance,
        random_intercept_variance: tau,
        sigma: variance.sqrt(),
        icc: rho / (1.0 + rho),
        log_likelihood: -final_fit.objective,
        aic: 2.0 * (p + 2) as f64 + 2.0 * final_fit.objective,
        df_residual: n - p,
        evaluations,
        boundary: best_t == 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_square_root_matches_observation_level_gls() {
        let n = 207;
        let design = DMatrix::from_fn(n, 4, |i, j| {
            let x = (i as f64 * 0.37).sin();
            match j {
                0 => 1.0,
                1 => x,
                // Highly correlated, but full rank: compression must not form
                // a cancellation-prone difference of Gram matrices.
                2 => 2.0 * x + 0.001 * (i as f64 * 0.81).cos(),
                _ => (i as f64 * 0.13).cos(),
            }
        });
        let members: Vec<Vec<usize>> = (0..17)
            .map(|g| (0..n).filter(|i| i % 17 == g).collect())
            .collect();
        for offset in [0.0, 1e6] {
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    offset
                        + 2.0 * design[(i, 1)]
                        + 0.7 * design[(i, 3)]
                        + (i % 17) as f64 * 0.23
                        + (i as f64 * 1.9).sin()
                })
                .collect();
            let data = ProfileData::new(&design, &[1.0; 4], &y, &members);
            for t in [0.0_f64, 0.3, 2.0, 12.0, 32.0] {
                let rho = t.exp_m1();
                let mut full =
                    DMatrix::from_fn(n, 5, |i, j| if j == 4 { y[i] } else { design[(i, j)] });
                for rows in &members {
                    for j in 0..5 {
                        let mean =
                            rows.iter().map(|i| full[(*i, j)]).sum::<f64>() / rows.len() as f64;
                        for i in rows {
                            full[(*i, j)] = (full[(*i, j)] - mean)
                                + mean / (1.0 + rows.len() as f64 * rho).sqrt();
                        }
                    }
                }
                let svd = full.columns(0, 4).into_owned().svd(true, true);
                let beta = svd.solve(&full.column(4).into_owned(), 1e-12).unwrap();
                let residual = full.column(4) - full.columns(0, 4) * &beta;
                let reference_q = residual.norm_squared();
                let cached = profile(&data, t, true).unwrap();
                assert!(
                    (cached.variance * n as f64 - reference_q).abs() <= 1e-6 + reference_q * 1e-7
                );
                // At t=32 the full whitening attenuates the intercept by
                // roughly 1e-8; coefficient agreement in that weak direction
                // is limited by subtraction rounding. Check residual norms
                // there, and coefficients at the well-resolved trial ratios.
                for j in 0..if t < 32.0 { 4 } else { 0 } {
                    assert!(
                        (cached.beta[j] - beta[j]).abs() <= 1e-5 + beta[j].abs() * 1e-7,
                        "t={t}, offset={offset}, coefficient {j}: {} != {}",
                        cached.beta[j],
                        beta[j]
                    );
                }
                // Direct residual norms at nonoptimal coefficients independently
                // exercise the augmented QR summary, not only its minimizer.
                let trial = DVector::from_vec(vec![offset + 1.0, 0.3, -0.1, 0.2]);
                let compact = data.weighted(rho);
                let q1 = (compact.column(4) - compact.columns(0, 4) * &trial).norm_squared();
                let q2 = (full.column(4) - full.columns(0, 4) * trial).norm_squared();
                assert!((q1 - q2).abs() <= 1e-6 + q2 * 1e-7);
            }
        }
    }
}
