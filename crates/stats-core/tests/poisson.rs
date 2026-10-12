use serde_json::Value;
use stats_core::{PoissonResult, fit_poisson};

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 + 1e-7 * expected.abs(),
        "{label}: {actual} != {expected}"
    );
}

fn compare(result: &PoissonResult, expected: &Value) {
    for (key, values) in [
        ("coefficients", &result.coefficients),
        ("std_errors", &result.std_errors),
        ("z_statistics", &result.z_statistics),
        ("p_values", &result.p_values),
        ("fitted", &result.fitted),
        ("residuals", &result.residuals),
    ] {
        for (i, value) in values.iter().enumerate() {
            close(*value, expected[key][i].as_f64().unwrap(), key);
        }
    }
    for (i, row) in result.covariance.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            close(
                *value,
                expected["covariance"][i][j].as_f64().unwrap(),
                "covariance",
            );
        }
    }
    for (i, row) in result.conf_int.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            close(
                *value,
                expected["conf_int"][i][j].as_f64().unwrap(),
                "conf_int",
            );
        }
    }
    for (key, value) in [
        ("log_likelihood", result.log_likelihood),
        ("deviance", result.deviance),
        ("null_deviance", result.null_deviance),
        ("aic", result.aic),
    ] {
        close(value, expected[key].as_f64().unwrap(), key);
    }
    assert_eq!(
        result.df_residual,
        expected["df_residual"].as_u64().unwrap() as usize
    );
    assert!((1..=100).contains(&result.iterations));
}

#[test]
fn agrees_with_r_for_poisson_models() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/poisson-r.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let result = fit_poisson(
            serde_json::from_value(case["x"].clone()).unwrap(),
            serde_json::from_value(case["y"].clone()).unwrap(),
        )
        .unwrap();
        compare(&result, &case["expected"]);
    }
}

#[test]
fn rejects_invalid_samples() {
    for (x, y) in [
        (vec![], vec![]),
        (vec![vec![]], vec![0.]),
        (vec![vec![1.], vec![1., 2.], vec![1.]], vec![0., 1., 0.]),
        (vec![vec![1.]; 3], vec![0., 1.]),
        (vec![vec![1., 0.], vec![1., 1.]], vec![0., 1.]),
        (vec![vec![1.]; 3], vec![0., 0.5, 1.]),
        (vec![vec![1.]; 3], vec![0., f64::NAN, 1.]),
        (vec![vec![f64::INFINITY]; 3], vec![0., 1., 0.]),
        (vec![vec![1.]; 3], vec![0., 0., 0.]),
        (vec![vec![1., 1.]; 4], vec![0., 1., 0., 1.]),
    ] {
        assert!(fit_poisson(x, y).is_err());
    }
}

#[test]
fn predictor_units_preserve_probabilities_and_inference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/poisson-r.json")).unwrap();
    let case = &fixture["cases"][0];
    let mut x: Vec<Vec<f64>> = serde_json::from_value(case["x"].clone()).unwrap();
    for row in &mut x {
        row[1] *= 1e-12;
    }
    let result = fit_poisson(x, serde_json::from_value(case["y"].clone()).unwrap()).unwrap();
    close(
        result.coefficients[1] * 1e-12,
        case["expected"]["coefficients"][1].as_f64().unwrap(),
        "rescaled coefficient",
    );
    close(
        result.std_errors[1] * 1e-12,
        case["expected"]["std_errors"][1].as_f64().unwrap(),
        "rescaled SE",
    );
    for (i, value) in result.fitted.iter().enumerate() {
        close(
            *value,
            case["expected"]["fitted"][i].as_f64().unwrap(),
            "rescaled fitted",
        );
    }
}

#[test]
fn accepts_constant_positive_counts_and_rejects_a_boundary_mle() {
    let fit = fit_poisson(vec![vec![1.]; 4], vec![1.; 4]).unwrap();
    close(fit.coefficients[0], 0., "constant-count intercept");
    close(fit.std_errors[0], 0.5, "constant-count SE");
    close(fit.deviance, 0., "saturated constant counts");
    let boundary = fit_poisson(
        vec![
            vec![1., 0.],
            vec![1., 0.],
            vec![1., 0.],
            vec![1., 1.],
            vec![1., 1.],
            vec![1., 1.],
        ],
        vec![1., 2., 3., 0., 0., 0.],
    )
    .unwrap_err();
    assert!(boundary.0.contains("converge") || boundary.0.contains("rank deficient"));
}
