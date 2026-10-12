use nalgebra::DMatrix;
use serde_json::Value;
use stats_core::{ErrorKind, draws::*};

fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-11 + 1e-11 * b.abs(),
        "{a} != {b}"
    );
}
fn compare(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            close(x, y);
        }
    }
}
fn mat(x: &[Vec<f64>]) -> DMatrix<f64> {
    DMatrix::from_row_iterator(x.len(), x[0].len(), x.iter().flatten().copied())
}
fn decode(v: &Value) -> Vec<Vec<f64>> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn independent_decimal_references_and_identities() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/draws-python.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let expected = &case["expected"];
        match case["operation"].as_str().unwrap() {
            "moments" => {
                let x = decode(&case["draws"]);
                let w: Option<Vec<f64>> = serde_json::from_value(case["weights"].clone()).unwrap();
                let norm = if case["normalization"] == "unbiased" {
                    Normalization::Unbiased
                } else {
                    Normalization::Population
                };
                let actual = draw_moments(&x, w.as_deref(), norm).unwrap();
                let mean: Vec<f64> = serde_json::from_value(expected["mean"].clone()).unwrap();
                for (&a, &b) in actual.mean.iter().zip(&mean) {
                    close(a, b);
                }
                compare(&actual.covariance, &decode(&expected["covariance"]));
                if let Some(w) = w {
                    let scaled: Vec<f64> = w.iter().map(|w| w * 0.25).collect();
                    let again = draw_moments(&x, Some(&scaled), norm).unwrap();
                    compare(&actual.covariance, &again.covariance);
                }
            }
            "covariance_map" => {
                let source = decode(&case["source_covariance"]);
                let target = decode(&case["target_covariance"]);
                let actual = draw_covariance_map(&source, &target).unwrap();
                compare(&actual.matrix, &decode(&expected["matrix"]));
                close(
                    actual.log_abs_det,
                    expected["log_abs_det"].as_f64().unwrap(),
                );
                let a = mat(&actual.matrix);
                let recovered = &a * mat(&source) * a.transpose();
                for i in 0..target.len() {
                    for j in 0..target.len() {
                        close(recovered[(i, j)], target[i][j]);
                        if target[i][j] != 0.0 {
                            assert!((recovered[(i, j)] / target[i][j] - 1.0).abs() < 1e-12);
                        }
                    }
                }
                let reverse = draw_covariance_map(&target, &source).unwrap();
                close(actual.log_abs_det + reverse.log_abs_det, 0.0);
                let identity = mat(&reverse.matrix) * a;
                for i in 0..source.len() {
                    for j in 0..source.len() {
                        close(identity[(i, j)], f64::from(i == j));
                    }
                }
            }
            "affine" => {
                let x = decode(&case["draws"]);
                let a = decode(&case["matrix"]);
                let b: Vec<f64> = serde_json::from_value(case["shift"].clone()).unwrap();
                let actual = draw_affine(&x, &a, &b).unwrap();
                let expected_draws = decode(&expected["draws"]);
                compare(&actual.draws, &expected_draws);
                // Relative checks also constrain subnormal-sized fixture outputs.
                for (row, reference) in actual.draws.iter().zip(&expected_draws) {
                    for (&v, &r) in row.iter().zip(reference) {
                        if r != 0.0 {
                            assert!((v / r - 1.0).abs() < 1e-12);
                        }
                    }
                }
                close(
                    actual.log_abs_det,
                    expected["log_abs_det"].as_f64().unwrap(),
                );
                let inv = decode(&case["inverse"]);
                let shift: Vec<f64> =
                    serde_json::from_value(case["inverse_shift"].clone()).unwrap();
                let back = draw_affine(&actual.draws, &inv, &shift).unwrap();
                compare(&back.draws, &x);
                close(actual.log_abs_det + back.log_abs_det, 0.0);
            }
            _ => panic!("unknown fixture operation"),
        }
    }
}

#[test]
fn moments_validation_and_zero_weight_rows() {
    use Normalization::{Population, Unbiased};
    let x = vec![vec![1., 2.], vec![3., 4.]];
    for (draws, kind) in [
        (vec![], ErrorKind::InvalidInput),
        (vec![vec![]], ErrorKind::InvalidInput),
        (vec![vec![1.], vec![2., 3.]], ErrorKind::DimensionMismatch),
        (vec![vec![f64::NAN]], ErrorKind::Nonfinite),
        (vec![vec![f64::INFINITY]], ErrorKind::Nonfinite),
    ] {
        assert_eq!(
            draw_moments(&draws, None, Population).unwrap_err().kind(),
            kind
        );
    }
    for (w, norm, kind) in [
        (vec![1.], Population, ErrorKind::DimensionMismatch),
        (vec![0., 0.], Population, ErrorKind::InvalidInput),
        (vec![-1., 2.], Population, ErrorKind::InvalidInput),
        (vec![1., f64::NAN], Population, ErrorKind::Nonfinite),
        (vec![1., f64::INFINITY], Population, ErrorKind::Nonfinite),
        (vec![1., 0.], Unbiased, ErrorKind::InvalidInput),
        (vec![1., 1e-30], Unbiased, ErrorKind::InvalidInput),
    ] {
        assert_eq!(draw_moments(&x, Some(&w), norm).unwrap_err().kind(), kind);
    }
    assert_eq!(
        draw_moments(&x[..1], None, Unbiased).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    // Extreme finite zero-weight rows cannot affect moments, even through overflow.
    let extreme = vec![vec![1., 2.], vec![-f64::MAX, f64::MAX]];
    let m = draw_moments(&extreme, Some(&[1., 0.]), Population).unwrap();
    assert_eq!(m.mean, vec![1., 2.]);
    assert_eq!(m.covariance, vec![vec![0.; 2]; 2]);
    assert!(draw_moments(&[vec![1.], vec![f64::NAN]], Some(&[1., 0.]), Population).is_err());
    assert_eq!(
        draw_moments(&[vec![-1e200], vec![1e200]], None, Population)
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
}

#[test]
fn covariance_and_affine_failures_are_explicit() {
    let identity = vec![vec![1., 0.], vec![0., 1.]];
    for (cov, kind) in [
        (vec![], ErrorKind::InvalidInput),
        (vec![vec![1., 0.]], ErrorKind::DimensionMismatch),
        (vec![vec![1., 0.], vec![0.]], ErrorKind::DimensionMismatch),
        (vec![vec![1., 0.2], vec![0.3, 1.]], ErrorKind::InvalidInput),
        (
            vec![vec![1., 1.], vec![1., 1.]],
            ErrorKind::FactorizationFailure,
        ),
        (
            vec![vec![1., 2.], vec![2., 1.]],
            ErrorKind::FactorizationFailure,
        ),
        (
            vec![vec![0., 0.], vec![0., 1.]],
            ErrorKind::FactorizationFailure,
        ),
        (
            vec![vec![-1., 0.], vec![0., 1.]],
            ErrorKind::FactorizationFailure,
        ),
        (
            vec![vec![1., 0.], vec![0., f64::INFINITY]],
            ErrorKind::Nonfinite,
        ),
        (
            vec![vec![1., 1. - 1e-15], vec![1. - 1e-15, 1.]],
            ErrorKind::FactorizationFailure,
        ),
    ] {
        assert_eq!(
            draw_covariance_map(&cov, &identity).unwrap_err().kind(),
            kind
        );
        assert_eq!(
            draw_covariance_map(&identity, &cov).unwrap_err().kind(),
            kind
        );
    }
    assert_eq!(
        draw_covariance_map(&identity, &[vec![1.]])
            .unwrap_err()
            .kind(),
        ErrorKind::DimensionMismatch
    );
    let slightly_asymmetric = vec![vec![1., 0.2 + 1e-15], vec![0.2, 1.]];
    assert!(draw_covariance_map(&slightly_asymmetric, &identity).is_ok());
    let clearly_asymmetric = vec![vec![1., 0.2 + 1e-12], vec![0.2, 1.]];
    assert_eq!(
        draw_covariance_map(&clearly_asymmetric, &identity)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    // Near-singularity threshold is explicit; this distinguishable pivot passes.
    let near = vec![vec![1., 1. - 1e-10], vec![1. - 1e-10, 1.]];
    assert!(draw_covariance_map(&near, &identity).is_ok());
    for (a, kind) in [
        (vec![vec![1., 2.]], ErrorKind::DimensionMismatch),
        (vec![vec![1., 2.], vec![2., 4.]], ErrorKind::RankDeficiency),
        (vec![vec![0., 0.], vec![0., 1.]], ErrorKind::RankDeficiency),
        (
            vec![vec![1., 1.], vec![1., 1. + 1e-15]],
            ErrorKind::RankDeficiency,
        ),
        (vec![vec![1., f64::NAN], vec![0., 1.]], ErrorKind::Nonfinite),
    ] {
        assert_eq!(
            draw_affine(&identity, &a, &[0., 0.]).unwrap_err().kind(),
            kind
        );
    }
    assert_eq!(
        draw_affine(&identity, &identity, &[0.]).unwrap_err().kind(),
        ErrorKind::DimensionMismatch
    );
    assert_eq!(
        draw_affine(&identity, &identity, &[0., f64::NAN])
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        draw_affine(&[vec![f64::MAX]], &[vec![2.]], &[0.])
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
}

#[test]
fn weighted_covariance_transforms_with_the_draws() {
    let x = vec![vec![1., -2.], vec![4., 0.], vec![-1., 3.], vec![2., 1.]];
    let a = vec![vec![2., 1.], vec![-3., 0.5]];
    let weights = [1., 0., 3., 2.];
    for norm in [Normalization::Population, Normalization::Unbiased] {
        let before = draw_moments(&x, Some(&weights), norm).unwrap();
        let after = draw_moments(
            &draw_affine(&x, &a, &[7., -4.]).unwrap().draws,
            Some(&weights),
            norm,
        )
        .unwrap();
        let expected = mat(&a) * mat(&before.covariance) * mat(&a).transpose();
        for i in 0..2 {
            for j in 0..2 {
                close(after.covariance[i][j], expected[(i, j)]);
            }
        }
    }
}

#[test]
fn subnormal_affine_scale_does_not_form_an_overflowing_reciprocal() {
    let output = draw_affine(&[vec![1.], vec![2.]], &[vec![1e-310]], &[0.]).unwrap();
    assert_eq!(output.draws, vec![vec![1e-310], vec![2e-310]]);
    close(output.log_abs_det, 1e-310_f64.ln());
}
