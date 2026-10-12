use serde_json::Value;
use stats_core::{ProbitResult, fit_probit};

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 + 1e-7 * expected.abs(),
        "{label}: {actual} != {expected}"
    );
}

fn compare(result: &ProbitResult, expected: &Value) {
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
fn agrees_with_r_for_binary_probit_models() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/probit-r.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let result = fit_probit(
            serde_json::from_value(case["x"].clone()).unwrap(),
            serde_json::from_value(case["y"].clone()).unwrap(),
        )
        .unwrap();
        compare(&result, &case["expected"]);
    }
}

#[test]
fn rejects_invalid_and_separated_samples() {
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
        assert!(fit_probit(x, y).is_err());
    }
    let complete = fit_probit(
        vec![vec![1., -2.], vec![1., -1.], vec![1., 1.], vec![1., 2.]],
        vec![0., 0., 1., 1.],
    )
    .unwrap_err();
    assert!(complete.0.contains("separation"));
    let quasi = fit_probit(
        vec![
            vec![1., -2.],
            vec![1., -1.],
            vec![1., 0.],
            vec![1., 0.],
            vec![1., 1.],
            vec![1., 2.],
        ],
        vec![0., 0., 0., 1., 1., 1.],
    )
    .unwrap_err();
    assert!(quasi.0.contains("separation"));
}

#[test]
fn predictor_units_preserve_probabilities_and_inference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/probit-r.json")).unwrap();
    let case = &fixture["cases"][0];
    let mut x: Vec<Vec<f64>> = serde_json::from_value(case["x"].clone()).unwrap();
    for row in &mut x {
        row[1] *= 1e-12;
    }
    let result = fit_probit(x, serde_json::from_value(case["y"].clone()).unwrap()).unwrap();
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
