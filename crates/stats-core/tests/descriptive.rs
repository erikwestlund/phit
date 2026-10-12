use serde_json::Value;
use stats_core::{ErrorKind, descriptive::*};

fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-12 * b.abs(),
        "{a} != {b}"
    );
}
fn values(v: &Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn all_saved_r_descriptive_references() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/descriptive-r.json")).unwrap();
    for c in fixture["descriptive"].as_array().unwrap() {
        let x = values(&c["values"]);
        for (key, actual) in [
            ("mean", mean(&x).unwrap()),
            ("variance", variance(&x, false).unwrap()),
            ("sd", sd(&x, false).unwrap()),
            ("median", median(&x).unwrap()),
            ("population_variance", variance(&x, true).unwrap()),
        ] {
            close(actual, c[key].as_f64().unwrap());
        }
        let p = values(&c["quantile_probabilities"]);
        let q = values(&c["quantiles"]);
        for ((a, b), p) in quantiles(&x, &p).unwrap().iter().zip(&q).zip(p) {
            close(*a, *b);
            assert_eq!(*a, quantile(&x, p).unwrap());
        }
    }
    for c in fixture["pairs"].as_array().unwrap() {
        let (x, y) = (values(&c["x"]), values(&c["y"]));
        close(
            covariance(&x, &y, false).unwrap(),
            c["covariance"].as_f64().unwrap(),
        );
        close(
            covariance(&x, &y, true).unwrap(),
            c["population_covariance"].as_f64().unwrap(),
        );
        close(
            correlation(&x, &y).unwrap(),
            c["correlation"].as_f64().unwrap(),
        );
    }
    for c in fixture["weighted"].as_array().unwrap() {
        let (x, w) = (values(&c["values"]), values(&c["weights"]));
        close(weighted_mean(&x, &w).unwrap(), c["mean"].as_f64().unwrap());
        for (key, kind) in [
            ("frequency", WeightKind::Frequency),
            ("reliability", WeightKind::Reliability),
            ("population", WeightKind::Population),
        ] {
            close(
                weighted_variance(&x, &w, kind).unwrap(),
                c[key].as_f64().unwrap(),
            );
        }
        let scaled: Vec<f64> = w.iter().map(|v| v * 2.).collect();
        close(
            weighted_variance(&x, &w, WeightKind::Reliability).unwrap(),
            weighted_variance(&x, &scaled, WeightKind::Reliability).unwrap(),
        );
    }
}

#[test]
fn invalid_and_degenerate_descriptive_inputs() {
    for x in [vec![], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(mean(&x).is_err());
    }
    assert_eq!(
        covariance(&[1., 2.], &[1.], false).unwrap_err().kind(),
        ErrorKind::DimensionMismatch
    );
    assert_eq!(
        weighted_mean(&[1., 2.], &[1.]).unwrap_err().kind(),
        ErrorKind::DimensionMismatch
    );
    assert_eq!(variance(&[1.], true).unwrap(), 0.);
    assert!(variance(&[1.], false).is_err());
    assert!(correlation(&[1., 1.], &[2., 3.]).unwrap().is_nan());
    assert!(quantiles(&[1., 2.], &[]).unwrap().is_empty());
    for p in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(quantile(&[1., 2.], p).is_err());
    }
    for w in [[0., 0.], [-1., 2.], [f64::NAN, 1.], [f64::INFINITY, 1.]] {
        assert!(weighted_mean(&[1., 2.], &w).is_err());
    }
    assert!(weighted_variance(&[1., 2.], &[0., 2.], WeightKind::Reliability).is_err());
    assert_eq!(
        weighted_variance(&[1., 2.], &[0., 2.], WeightKind::Population).unwrap(),
        0.
    );
}

#[test]
fn overflow_is_not_reported_as_zero_variance() {
    assert_eq!(
        variance(&[-1e200, 1e200], false).unwrap_err().kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        weighted_variance(&[-1e200, 1e200], &[1., 1.], WeightKind::Frequency)
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        covariance(&[-1e200, 1e200], &[1., 2.], false)
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        mean(&[f64::MAX, f64::MAX]).unwrap_err().kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        weighted_mean(&[1., 2.], &[f64::MAX, f64::MAX])
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
}
