use crate::{ErrorKind, StatsError};
/// Validate prior weights and known offsets before fitting. An empty vector means default.
pub(crate) fn prepare(
    n: usize,
    weights: Vec<f64>,
    offset: Vec<f64>,
    binary: bool,
) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
    let weights = if weights.is_empty() {
        vec![1.0; n]
    } else {
        weights
    };
    let offset = if offset.is_empty() {
        vec![0.0; n]
    } else {
        offset
    };
    if weights.len() != n || offset.len() != n {
        return Err(StatsError::new(
            ErrorKind::DimensionMismatch,
            "weights and offset must have one value per row",
        ));
    }
    if weights.iter().chain(offset.iter()).any(|v| !v.is_finite()) {
        return Err(StatsError::new(
            ErrorKind::Nonfinite,
            "weights and offset must be finite",
        ));
    }
    if weights
        .iter()
        .any(|v| *v <= 0.0 || (binary && v.fract() != 0.0))
    {
        return Err(StatsError::new(
            ErrorKind::InvalidInput,
            "weights must be positive (integer trials for binary outcomes)",
        ));
    }
    if binary
        && weights
            .iter()
            .any(|w| *w > crate::MAX_LOGISTIC_MIXED_TRIALS)
    {
        return Err(StatsError::new(
            ErrorKind::Unsupported,
            "binomial trial weights must not exceed 1000000; larger-count likelihood arithmetic is not verified",
        ));
    }
    Ok((weights, offset))
}
