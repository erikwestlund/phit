use serde_json::{Value, json};
use stats_core::{LmmOptions, LmmResult, RandomTerm, fit_lmm};
fn compare(a: &Value, b: &Value, label: &str) {
    if let Some(values) = b.as_array() {
        assert_eq!(a.as_array().unwrap().len(), values.len(), "{label} shape");
        for (i, v) in values.iter().enumerate() {
            compare(&a[i], v, &format!("{label}/{i}"));
        }
    } else if let Some(values) = b.as_object() {
        for (k, v) in values {
            compare(&a[k], v, &format!("{label}/{k}"));
        }
    } else if b.is_boolean() || label.ends_with("/df_residual") || label.ends_with("/n_parameters")
    {
        assert_eq!(a, b, "{label}");
    } else {
        let a = a.as_f64().unwrap();
        let b = b.as_f64().unwrap();
        let (abs, rel) =
            if label.contains("log_likelihood") || label.ends_with("aic") || label.ends_with("bic")
            {
                (5e-6, 1e-9)
            } else if label.contains("/bobyqa/") {
                (1e-5, 1e-5)
            } else {
                (1e-3, 3e-4)
            };
        assert!(
            (a - b).abs() <= abs + rel * b.abs(),
            "{label}: {a} != {b} (difference {})",
            (a - b).abs()
        );
    }
}
fn result_json(r: &LmmResult) -> Value {
    json!({"coefficients":r.coefficients,"covariance":r.covariance,"std_errors":r.std_errors,
    "z_statistics":r.z_statistics,"p_values":r.p_values,"conf_int":r.conf_int,
    "fixed_fitted":r.fixed_fitted,"fitted":r.fitted,"residuals":r.residuals,
    "marginal_residuals":r.marginal_residuals,"residual_variance":r.residual_variance,
    "sigma":r.sigma,"log_likelihood":r.log_likelihood,"aic":r.aic,"bic":r.bic,
    "df_residual":r.df_residual,"n_parameters":r.n_parameters,"singular":r.singular,
    "terms":r.terms.iter().map(|t|json!({"covariance":t.covariance,"effects":t.effects})).collect::<Vec<_>>()})
}
#[test]
fn matches_four_lme4_optimizers_on_expanded_models() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/lmm-expanded-r.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let terms = c["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| RandomTerm {
                z: serde_json::from_value(t["z"].clone()).unwrap(),
                groups: serde_json::from_value(t["groups"].clone()).unwrap(),
                correlated: t["correlated"].as_bool().unwrap(),
            })
            .collect();
        let options = LmmOptions {
            reml: c["reml"].as_bool().unwrap(),
            weights: serde_json::from_value(c["weights"].clone()).unwrap(),
            offset: serde_json::from_value(c["offset"].clone()).unwrap(),
            ..Default::default()
        };
        let r = fit_lmm(
            serde_json::from_value(c["x"].clone()).unwrap(),
            serde_json::from_value(c["y"].clone()).unwrap(),
            terms,
            options,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let actual = result_json(&r);
        for (optimizer, reference) in c["references"].as_object().unwrap() {
            compare(
                &actual,
                &reference["result"],
                &format!("{}/{optimizer}", c["name"]),
            );
        }
    }
}
#[test]
fn special_case_matches_fast_random_intercept_and_balanced_reml() {
    let x = vec![vec![1.0]; 12];
    let y = vec![-3., -2., -1., -2., -1., 0., 0., 1., 2., 1., 2., 3.];
    let groups = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let fast = stats_core::fit_random_intercept(x.clone(), y.clone(), groups.clone()).unwrap();
    let term = RandomTerm {
        z: x.clone(),
        groups,
        correlated: true,
    };
    let ml = fit_lmm(x.clone(), y.clone(), vec![term.clone()], Default::default()).unwrap();
    assert!((ml.log_likelihood - fast.log_likelihood).abs() < 1e-8);
    let reml = fit_lmm(
        x,
        y,
        vec![term],
        LmmOptions {
            reml: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((reml.residual_variance - 1.0).abs() < 1e-6);
    assert!((reml.terms[0].covariance[0][0] - 3.0).abs() < 1e-6);
}
#[test]
fn rejects_invalid_controls_designs_and_nonconvergence() {
    let x = vec![vec![1.]; 8];
    let y = vec![1., 2., 4., 5., 0., 1., 8., 9.];
    let term = RandomTerm {
        z: x.clone(),
        groups: vec![0, 0, 1, 1, 2, 2, 3, 3],
        correlated: true,
    };
    for options in [
        LmmOptions {
            weights: vec![0.; 8],
            ..Default::default()
        },
        LmmOptions {
            offset: vec![f64::NAN; 8],
            ..Default::default()
        },
        LmmOptions {
            start: Some(vec![-1.]),
            ..Default::default()
        },
        LmmOptions {
            max_evaluations: 4,
            ..Default::default()
        },
        LmmOptions {
            tolerance: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(fit_lmm(x.clone(), y.clone(), vec![term.clone()], options).is_err());
    }
    assert!(fit_lmm(x.clone(), y.clone(), vec![], Default::default()).is_err());
    assert!(fit_lmm(vec![vec![1., 1.]; 8], y, vec![term], Default::default()).is_err());
}
#[test]
fn nelder_mead_matches_reference_and_obeys_budget() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/lmm-expanded-r.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let terms = c["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| RandomTerm {
                z: serde_json::from_value(t["z"].clone()).unwrap(),
                groups: serde_json::from_value(t["groups"].clone()).unwrap(),
                correlated: t["correlated"].as_bool().unwrap(),
            })
            .collect();
        let r = fit_lmm(
            serde_json::from_value(c["x"].clone()).unwrap(),
            serde_json::from_value(c["y"].clone()).unwrap(),
            terms,
            LmmOptions {
                optimizer: stats_core::LmmOptimizer::NelderMead,
                reml: c["reml"].as_bool().unwrap(),
                weights: serde_json::from_value(c["weights"].clone()).unwrap(),
                offset: serde_json::from_value(c["offset"].clone()).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        compare(
            &result_json(&r),
            &c["references"]["Nelder_Mead"]["result"],
            &format!("{}/Nelder_Mead", c["name"]),
        );
    }
    assert!(
        fit_lmm(
            vec![vec![1.]; 8],
            vec![1., 2., 4., 5., 0., 1., 8., 9.],
            vec![RandomTerm {
                z: vec![vec![1.]; 8],
                groups: vec![0, 0, 1, 1, 2, 2, 3, 3],
                correlated: true
            }],
            LmmOptions {
                optimizer: stats_core::LmmOptimizer::NelderMead,
                max_evaluations: 4,
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[test]
fn reml_restores_fixed_column_units_and_prior_weight_determinants() {
    let f: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/lmm-expanded-r.json")).unwrap();
    let c = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "sleepstudy_reml")
        .unwrap();
    let x: Vec<Vec<f64>> = serde_json::from_value(c["x"].clone()).unwrap();
    let y: Vec<f64> = serde_json::from_value(c["y"].clone()).unwrap();
    let t = &c["terms"][0];
    let term = RandomTerm {
        z: serde_json::from_value(t["z"].clone()).unwrap(),
        groups: serde_json::from_value(t["groups"].clone()).unwrap(),
        correlated: true,
    };
    let base = fit_lmm(
        x.clone(),
        y.clone(),
        vec![term.clone()],
        LmmOptions {
            reml: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut xx = x.clone();
    for row in &mut xx {
        row[1] *= 10.;
    }
    let changed = fit_lmm(
        xx,
        y.clone(),
        vec![term.clone()],
        LmmOptions {
            reml: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((changed.log_likelihood - base.log_likelihood + 10f64.ln()).abs() < 1e-7);
    assert!((changed.coefficients[1] * 10. - base.coefficients[1]).abs() < 1e-7);
    let weighted = fit_lmm(
        x,
        y,
        vec![term],
        LmmOptions {
            reml: true,
            weights: vec![5.; 180],
            ..Default::default()
        },
    )
    .unwrap();
    assert!((weighted.log_likelihood - base.log_likelihood).abs() < 1e-7);
    assert!((weighted.residual_variance / 5. - base.residual_variance).abs() < 1e-3);
    for (a, b) in weighted.fitted.iter().zip(&base.fitted) {
        assert!((a - b).abs() < 1e-4);
    }
}
