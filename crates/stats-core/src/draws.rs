//! Literal, row-oriented posterior-draw kernels. No fitting or row filtering.
use crate::{ErrorKind, StatsError};
use nalgebra::{DMatrix, DVector};

/// Reliability-weight covariance normalization (never frequency counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    Population,
    Unbiased,
}

#[derive(Debug, Clone)]
pub struct DrawMoments {
    pub mean: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct CovarianceMap {
    pub matrix: Vec<Vec<f64>>,
    pub log_abs_det: f64,
}

#[derive(Debug, Clone)]
pub struct AffineDraws {
    pub draws: Vec<Vec<f64>>,
    pub log_abs_det: f64,
}

fn error(kind: ErrorKind, message: &str) -> StatsError {
    StatsError::new(kind, message)
}

fn shape(rows: &[Vec<f64>]) -> Result<(usize, usize), StatsError> {
    if rows.is_empty() || rows[0].is_empty() {
        return Err(error(
            ErrorKind::InvalidInput,
            "matrices must have at least one row and column",
        ));
    }
    let p = rows[0].len();
    if rows.iter().any(|row| row.len() != p) {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "matrices must be rectangular",
        ));
    }
    if rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err(error(
            ErrorKind::Nonfinite,
            "matrix elements must be finite",
        ));
    }
    Ok((rows.len(), p))
}

fn finite(value: f64) -> Result<f64, StatsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(error(
            ErrorKind::Nonfinite,
            "draw operation exceeded floating-point range",
        ))
    }
}

// Neumaier sum, matching the descriptive-statistics convention.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut correction) = (0.0, 0.0);
    for v in values {
        let t = total + v;
        correction += if total.abs() >= v.abs() {
            (total - t) + v
        } else {
            (v - t) + total
        };
        total = t;
    }
    total + correction
}

fn rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| matrix.row(i).iter().copied().collect())
        .collect()
}

fn matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    DMatrix::from_row_iterator(rows.len(), rows[0].len(), rows.iter().flatten().copied())
}

/// Weighted moments. Zero-weight rows are validated but contribute nothing.
/// Singular covariance is valid. Unbiased weights express reliability, not counts.
pub fn draw_moments(
    draws: &[Vec<f64>],
    weights: Option<&[f64]>,
    normalization: Normalization,
) -> Result<DrawMoments, StatsError> {
    let (n, p) = shape(draws)?;
    let weights = match weights {
        None => vec![1.0 / n as f64; n],
        Some(w) => {
            if w.len() != n {
                return Err(error(
                    ErrorKind::DimensionMismatch,
                    "weights must have one element per draw",
                ));
            }
            if w.iter().any(|v| !v.is_finite()) {
                return Err(error(ErrorKind::Nonfinite, "weights must be finite"));
            }
            if w.iter().any(|&v| v < 0.0) {
                return Err(error(
                    ErrorKind::InvalidInput,
                    "weights must be nonnegative",
                ));
            }
            let largest = w.iter().copied().fold(0.0, f64::max);
            if largest == 0.0 {
                return Err(error(
                    ErrorKind::InvalidInput,
                    "weights must have positive total weight",
                ));
            }
            // Scaling before summation avoids overflow even for weights near MAX.
            let total = sum(w.iter().map(|v| v / largest));
            w.iter().map(|v| (v / largest) / total).collect()
        }
    };
    let denominator = match normalization {
        Normalization::Population => 1.0,
        Normalization::Unbiased => 1.0 - sum(weights.iter().map(|w| w * w)),
    };
    if denominator <= 0.0 {
        return Err(error(
            ErrorKind::InvalidInput,
            "unbiased covariance normalization is undefined; need at least two effective draws",
        ));
    }
    let anchor = weights.iter().position(|&w| w > 0.0).unwrap();
    let mut mean = Vec::with_capacity(p);
    let mut centered = DMatrix::zeros(n, p);
    for j in 0..p {
        // Anchor near the data so large common offsets do not erase variation.
        let origin = draws[anchor][j];
        let m = finite(
            origin
                + sum(draws
                    .iter()
                    .zip(&weights)
                    .filter(|(_, w)| **w > 0.0)
                    .map(|(x, w)| w * (x[j] - origin))),
        )?;
        let m = finite(
            m + sum(draws
                .iter()
                .zip(&weights)
                .filter(|(_, w)| **w > 0.0)
                .map(|(x, w)| w * (x[j] - m))),
        )?;
        mean.push(m);
        for i in 0..n {
            if weights[i] > 0.0 {
                centered[(i, j)] = finite(draws[i][j] - m)?;
            }
        }
    }
    let mut covariance = DMatrix::zeros(p, p);
    for j in 0..p {
        for k in 0..=j {
            let c = finite(
                sum((0..n).map(|i| (weights[i] * centered[(i, j)]) * centered[(i, k)]))
                    / denominator,
            )?;
            covariance[(j, k)] = c;
            covariance[(k, j)] = c;
        }
    }
    Ok(DrawMoments {
        mean,
        covariance: rows(&covariance),
    })
}

/// Symmetry tolerance on diagonal-standardized entries: 64*EPS*max(1, |a|, |b|).
/// Squared Cholesky pivots on that scale must exceed 64*EPS*dimension.
/// Accepted rounding-level asymmetry is averaged; no diagonal regularization.
fn covariance_factor(covariance: &[Vec<f64>]) -> Result<DMatrix<f64>, StatsError> {
    let (n, p) = shape(covariance)?;
    if n != p {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "covariance matrices must be square",
        ));
    }
    if (0..p).any(|j| covariance[j][j] <= 0.0) {
        return Err(error(
            ErrorKind::FactorizationFailure,
            "covariance must be positive definite",
        ));
    }
    let scales: Vec<f64> = (0..p).map(|j| covariance[j][j].sqrt()).collect();
    let mut standardized = DMatrix::identity(p, p);
    for i in 0..p {
        for j in 0..i {
            let a = finite(covariance[i][j] / scales[i] / scales[j])?;
            let b = finite(covariance[j][i] / scales[i] / scales[j])?;
            if (a - b).abs() > 64.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0) {
                return Err(error(
                    ErrorKind::InvalidInput,
                    "covariance matrices must be symmetric within tolerance",
                ));
            }
            standardized[(i, j)] = 0.5 * a + 0.5 * b;
            standardized[(j, i)] = standardized[(i, j)];
        }
    }
    let mut lower = standardized
        .cholesky()
        .ok_or_else(|| {
            error(
                ErrorKind::FactorizationFailure,
                "covariance must be positive definite",
            )
        })?
        .l();
    for i in 0..p {
        if lower[(i, i)].powi(2) <= 64.0 * f64::EPSILON * p as f64 {
            return Err(error(
                ErrorKind::FactorizationFailure,
                "covariance is numerically singular within tolerance",
            ));
        }
        for j in 0..=i {
            lower[(i, j)] = finite(lower[(i, j)] * scales[i])?;
        }
    }
    Ok(lower)
}

/// A = L_target L_source^-1, computed by a triangular solve, never an inverse.
/// The map is deterministic for the supplied parameter order, but not unique.
pub fn draw_covariance_map(
    source_covariance: &[Vec<f64>],
    target_covariance: &[Vec<f64>],
) -> Result<CovarianceMap, StatsError> {
    let source = covariance_factor(source_covariance)?;
    let target = covariance_factor(target_covariance)?;
    if source.nrows() != target.nrows() {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "covariances must have equal dimensions",
        ));
    }
    // L_source^T A^T = L_target^T.
    let a = source
        .transpose()
        .solve_upper_triangular(&target.transpose())
        .ok_or_else(|| {
            error(
                ErrorKind::FactorizationFailure,
                "covariance triangular solve failed",
            )
        })?
        .transpose();
    for &v in a.iter() {
        finite(v)?;
    }
    let log_abs_det = finite(sum(
        (0..source.nrows()).map(|j| target[(j, j)].ln() - source[(j, j)].ln())
    ))?;
    Ok(CovarianceMap {
        matrix: rows(&a),
        log_abs_det,
    })
}

// Row/column equilibration avoids rejecting a diagonal map just because of units.
// Numerical rank requires sigma_min > 64*EPS*p*sigma_max on this scaled matrix.
fn affine_log_det(a: &DMatrix<f64>) -> Result<f64, StatsError> {
    let p = a.nrows();
    let mut scaled = a.clone();
    let mut log_scale = 0.0;
    for i in 0..p {
        let scale = scaled.row(i).iter().map(|v| v.abs()).fold(0.0, f64::max);
        if scale == 0.0 {
            return Err(error(
                ErrorKind::RankDeficiency,
                "affine matrix must be invertible",
            ));
        }
        // Divide individually: the reciprocal can overflow for subnormal scales.
        for j in 0..p {
            scaled[(i, j)] /= scale;
        }
        log_scale += scale.ln();
    }
    for j in 0..p {
        let scale = scaled.column(j).iter().map(|v| v.abs()).fold(0.0, f64::max);
        if scale == 0.0 {
            return Err(error(
                ErrorKind::RankDeficiency,
                "affine matrix must be invertible",
            ));
        }
        for i in 0..p {
            scaled[(i, j)] /= scale;
        }
        log_scale += scale.ln();
    }
    let singular = scaled.clone().svd(false, false).singular_values;
    if singular.min() <= 64.0 * f64::EPSILON * p as f64 * singular.max() {
        return Err(error(
            ErrorKind::RankDeficiency,
            "affine matrix is numerically singular within tolerance",
        ));
    }
    // Sum log pivots instead of forming a determinant that can under/overflow.
    let upper = scaled.lu().u();
    finite(log_scale + sum((0..p).map(|j| upper[(j, j)].abs().ln())))
}

/// Apply y = A*x + b to each row in order. No intercept is inserted.
pub fn draw_affine(
    draws: &[Vec<f64>],
    transform: &[Vec<f64>],
    shift: &[f64],
) -> Result<AffineDraws, StatsError> {
    let (_, p) = shape(draws)?;
    let (r, c) = shape(transform)?;
    if r != c || c != p || shift.len() != p {
        return Err(error(
            ErrorKind::DimensionMismatch,
            "affine matrix must be square and match draws and shift dimensions",
        ));
    }
    if shift.iter().any(|v| !v.is_finite()) {
        return Err(error(ErrorKind::Nonfinite, "shift elements must be finite"));
    }
    let a = matrix(transform);
    let log_abs_det = affine_log_det(&a)?;
    let mut output = matrix(draws) * a.transpose();
    let b = DVector::from_column_slice(shift);
    for mut row in output.row_iter_mut() {
        row += b.transpose();
    }
    for &v in output.iter() {
        finite(v)?;
    }
    Ok(AffineDraws {
        draws: rows(&output),
        log_abs_det,
    })
}
