use serde_json::{Value, json};
use stats_core::{LogisticMixedOptions, fit_logistic_random_intercept};
fn compare(a: &Value, b: &Value, label: &str) {
    if let Some(values) = b.as_array() {
        assert_eq!(a.as_array().unwrap().len(), values.len(), "{label} shape");
        for (i, v) in values.iter().enumerate() {
            compare(&a[i], v, &format!("{label}/{i}"));
        }
    } else if b.is_boolean() || label.ends_with("/df_residual") || label.ends_with("/n_parameters")
    {
        assert_eq!(a, b, "{label}");
    } else {
        let a = a.as_f64().unwrap();
        let b = b.as_f64().unwrap();
        let (abs, rel) = if label.contains("covariance")
            || label.contains("std_errors")
            || label.contains("z_statistics")
            || label.contains("conf_int")
            || label.contains("p_values")
        {
            (2e-5, 2e-3)
        } else if label.ends_with("log_likelihood")
            || label.ends_with("aic")
            || label.ends_with("bic")
        {
            (1e-6, 1e-8)
        } else {
            (2e-5, 2e-4)
        };
        assert!(
            (a - b).abs() <= abs + rel * b.abs(),
            "{label}: {a} != {b} (difference {})",
            (a - b).abs()
        );
    }
}
#[test]
fn matches_lme4_laplace_and_adaptive_quadrature() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/logistic-mixed-r.json"
    ))
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let r = fit_logistic_random_intercept(
            serde_json::from_value(c["x"].clone()).unwrap(),
            serde_json::from_value(c["y"].clone()).unwrap(),
            serde_json::from_value(c["groups"].clone()).unwrap(),
            LogisticMixedOptions {
                weights: serde_json::from_value(c["weights"].clone()).unwrap(),
                offset: serde_json::from_value(c["offset"].clone()).unwrap(),
                quadrature_points: c["quadrature_points"].as_u64().unwrap() as usize,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let actual = json!({"coefficients":r.coefficients,"covariance":r.covariance,"std_errors":r.std_errors,"z_statistics":r.z_statistics,
            "p_values":r.p_values,"conf_int":r.conf_int,"fixed_fitted":r.fixed_fitted,"fitted":r.fitted,"linear_predictor":r.linear_predictor,
            "residuals":r.residuals,"random_effects":r.random_effects,"random_intercept_sd":r.random_intercept_sd,"random_intercept_variance":r.random_intercept_variance,
            "log_likelihood":r.log_likelihood,"neg2_log_likelihood":r.neg2_log_likelihood,"aic":r.aic,"bic":r.bic,
            "n_parameters":r.n_parameters,"df_residual":r.df_residual,"singular":r.singular});
        for (k, v) in c["expected"].as_object().unwrap() {
            compare(&actual[k], v, &format!("{}/{k}", c["name"]));
        }
    }
}
#[test]
fn zero_random_variance_reduces_to_logistic_glm() {
    let x = vec![vec![1.]; 48];
    let y: Vec<f64> = (0..48).map(|i| (i % 2) as f64).collect();
    let groups: Vec<usize> = (0..48).map(|i| i / 4).collect();
    let glm = stats_core::fit_logistic(x.clone(), y.clone()).unwrap();
    for points in [1, 9, 25] {
        let r = fit_logistic_random_intercept(
            x.clone(),
            y.clone(),
            groups.clone(),
            LogisticMixedOptions {
                quadrature_points: points,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(r.boundary);
        assert_eq!(r.random_intercept_variance, 0.0);
        assert_eq!(r.covariance, glm.covariance);
        assert_eq!(r.coefficients, glm.coefficients);
        assert_eq!(r.log_likelihood, glm.log_likelihood);
        assert!(r.random_effects.iter().all(|v| *v == 0.0));
    }
}
#[test]
fn rejects_invalid_data_controls_and_nonconvergence() {
    let x = vec![vec![1.]; 8];
    let y = vec![0., 1., 0., 1., 0., 1., 0., 1.];
    let g = vec![0, 0, 1, 1, 2, 2, 3, 3];
    for options in [
        LogisticMixedOptions {
            quadrature_points: 0,
            ..Default::default()
        },
        LogisticMixedOptions {
            quadrature_points: 62,
            ..Default::default()
        },
        LogisticMixedOptions {
            weights: vec![0.5; 8],
            ..Default::default()
        },
        LogisticMixedOptions {
            offset: vec![f64::NAN; 8],
            ..Default::default()
        },
        LogisticMixedOptions {
            tolerance: f64::NAN,
            ..Default::default()
        },
        LogisticMixedOptions {
            max_evaluations: 4,
            ..Default::default()
        },
    ] {
        assert!(fit_logistic_random_intercept(x.clone(), y.clone(), g.clone(), options).is_err());
    }
    assert!(
        fit_logistic_random_intercept(x.clone(), y.clone(), vec![0; 8], Default::default())
            .is_err()
    );
    assert!(
        fit_logistic_random_intercept(x.clone(), y.clone(), (0..8).collect(), Default::default())
            .is_err()
    );
    assert!(
        fit_logistic_random_intercept(
            vec![vec![1., 1.]; 8],
            y.clone(),
            g.clone(),
            Default::default()
        )
        .is_err()
    );
    assert!(
        fit_logistic_random_intercept(x.clone(), vec![1.; 8], g.clone(), Default::default())
            .is_err()
    );
    assert!(
        fit_logistic_random_intercept(
            vec![vec![-2.], vec![-1.], vec![1.], vec![2.]],
            vec![0., 0., 1., 1.],
            vec![0, 0, 1, 1],
            Default::default()
        )
        .is_err()
    );
}
#[test]
fn grouped_boundary_and_predictor_units_preserve_contract() {
    let x = vec![vec![1.]; 48];
    let y = vec![0.5; 48];
    let groups: Vec<usize> = (0..48).map(|i| i / 4).collect();
    let glm =
        stats_core::fit_logistic_with_data(x.clone(), y.clone(), vec![10.; 48], vec![]).unwrap();
    let r = fit_logistic_random_intercept(
        x,
        y,
        groups,
        LogisticMixedOptions {
            weights: vec![10.; 48],
            quadrature_points: 9,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(r.boundary);
    assert_eq!(r.covariance, glm.covariance);
    assert_eq!(r.log_likelihood, glm.log_likelihood);
    let f: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/logistic-mixed-r.json"
    ))
    .unwrap();
    let c = &f["cases"][0];
    let x: Vec<Vec<f64>> = serde_json::from_value(c["x"].clone()).unwrap();
    let y: Vec<f64> = serde_json::from_value(c["y"].clone()).unwrap();
    let groups: Vec<usize> = serde_json::from_value(c["groups"].clone()).unwrap();
    let a = fit_logistic_random_intercept(x.clone(), y.clone(), groups.clone(), Default::default())
        .unwrap();
    let mut xx = x;
    for row in &mut xx {
        row[1] *= 1e-9;
    }
    let b = fit_logistic_random_intercept(xx, y, groups, Default::default()).unwrap();
    assert!((a.coefficients[1] - b.coefficients[1] * 1e-9).abs() < 1e-6);
    for (a, b) in a.fitted.iter().zip(&b.fitted) {
        assert!((a - b).abs() < 1e-6);
    }
}
#[test]
fn trial_count_limit_preserves_verified_binomial_log_mass() {
    let options = LogisticMixedOptions {
        weights: vec![1_000_000.; 8],
        ..Default::default()
    };
    let r = fit_logistic_random_intercept(
        vec![vec![1.]; 8],
        vec![0.5; 8],
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        options,
    )
    .unwrap();
    // R dbinom(500000, 1000000, 0.5, log=TRUE) = -7.133546881626864.
    assert!((r.log_likelihood - 8.0 * -7.133546881626864).abs() < 1e-6);
    assert!(
        fit_logistic_random_intercept(
            vec![vec![1.]; 8],
            vec![0.5; 8],
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            LogisticMixedOptions {
                weights: vec![1_000_001.; 8],
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn rounding_tolerance_does_not_create_fictitious_events() {
    assert!(
        fit_logistic_random_intercept(
            vec![vec![1.]; 8],
            vec![1e-10; 8],
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            Default::default()
        )
        .is_err()
    );
    let clean = vec![0., 1., 0., 1., 0., 1., 0., 1.];
    let near: Vec<f64> = clean
        .iter()
        .map(|v| if *v == 0. { 1e-10 } else { 1. - 1e-10 })
        .collect();
    let a = fit_logistic_random_intercept(
        vec![vec![1.]; 8],
        clean,
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        Default::default(),
    )
    .unwrap();
    let b = fit_logistic_random_intercept(
        vec![vec![1.]; 8],
        near,
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        Default::default(),
    )
    .unwrap();
    assert_eq!(a.coefficients, b.coefficients);
    assert_eq!(a.log_likelihood, b.log_likelihood);
}
