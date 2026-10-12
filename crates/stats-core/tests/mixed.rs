use serde_json::Value;
use stats_core::fit_random_intercept;
fn close(a: f64, b: f64, label: &str) {
    assert!(
        (a - b).abs() <= 1e-7 + 1e-6 * b.abs(),
        "{label}: {a} != {b}"
    );
}
fn check_reference_cases(input: &str, reference: &str) {
    let fixture: Value = serde_json::from_str(input).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let close = |a: f64, b: f64, key: &str| {
            let policy = &case["tolerance"];
            let atol = policy["quantity_absolute"][key]
                .as_f64()
                .unwrap_or_else(|| policy["absolute"].as_f64().unwrap_or(1e-7));
            let rtol = policy["quantity_relative"][key]
                .as_f64()
                .unwrap_or_else(|| policy["relative"].as_f64().unwrap_or(1e-6));
            assert!(
                (a - b).abs() <= atol + rtol * b.abs(),
                "{}/{key}: {a} != {b}",
                case["name"]
            );
        };
        let r = fit_random_intercept(
            serde_json::from_value(case["x"].clone()).unwrap(),
            serde_json::from_value(case["y"].clone()).unwrap(),
            serde_json::from_value(case["groups"].clone()).unwrap(),
        )
        .unwrap();
        let e = &case[reference]["result"];
        for (key, values) in [
            ("coefficients", &r.coefficients),
            ("std_errors", &r.std_errors),
            ("z_statistics", &r.z_statistics),
            ("p_values", &r.p_values),
            ("fixed_fitted", &r.fixed_fitted),
            ("fitted", &r.fitted),
            ("residuals", &r.residuals),
            ("marginal_residuals", &r.marginal_residuals),
            ("random_effects", &r.random_effects),
        ] {
            assert_eq!(values.len(), e[key].as_array().unwrap().len());
            for (i, a) in values.iter().enumerate() {
                close(*a, e[key][i].as_f64().unwrap(), key);
            }
        }
        for (key, values) in [
            ("covariance", r.covariance),
            (
                "conf_int",
                r.conf_int.into_iter().map(|v| v.to_vec()).collect(),
            ),
        ] {
            for (i, row) in values.iter().enumerate() {
                for (j, a) in row.iter().enumerate() {
                    close(*a, e[key][i][j].as_f64().unwrap(), key);
                }
            }
        }
        for (key, a) in [
            ("residual_variance", r.residual_variance),
            ("random_intercept_variance", r.random_intercept_variance),
            ("sigma", r.sigma),
            ("icc", r.icc),
            ("log_likelihood", r.log_likelihood),
            ("aic", r.aic),
        ] {
            close(a, e[key].as_f64().unwrap(), key);
        }
        assert_eq!(r.df_residual, e["df_residual"].as_u64().unwrap() as usize);
        assert_eq!(r.boundary, case["name"] == "boundary");
    }
}
#[test]
fn matches_lme4_marginal_ml() {
    check_reference_cases(
        include_str!("../../../tests/fixtures/lmm-r.json"),
        "reference",
    );
}
#[test]
fn matches_lme4_on_published_datasets() {
    check_reference_cases(
        include_str!("../../../tests/fixtures/lmm-public-r.json"),
        "reference",
    );
}
#[test]
fn matches_nlme_on_published_datasets() {
    check_reference_cases(
        include_str!("../../../tests/fixtures/lmm-public-r.json"),
        "nlme_reference",
    );
}
#[test]
fn balanced_intercept_has_known_ml_solution() {
    let mut y = Vec::new();
    let mut groups = Vec::new();
    for (g, mean) in [-2., -1., 1., 2.].iter().enumerate() {
        for delta in [-1., 0., 1.] {
            y.push(mean + delta);
            groups.push(g);
        }
    }
    let r = fit_random_intercept(vec![vec![1.]; 12], y, groups).unwrap();
    close(r.coefficients[0], 0., "intercept");
    close(r.residual_variance, 1., "within variance");
    close(r.random_intercept_variance, 13. / 6., "between variance");
    for (i, mean) in [-2., -1., 1., 2.].iter().enumerate() {
        close(r.random_effects[i], mean * 13. / 15., "shrinkage");
    }
}
#[test]
fn rejects_invalid_and_unidentifiable_samples() {
    for (x, y, g) in [
        (vec![], vec![], vec![]),
        (vec![vec![]], vec![1.], vec![0]),
        (vec![vec![1.]; 4], vec![1., 2., 3., 4.], vec![0, 0]),
        (vec![vec![1.]; 4], vec![1., 2., 3., 4.], vec![0, 0, 0, 0]),
        (vec![vec![1.]; 4], vec![1., 2., 3., 4.], vec![0, 1, 2, 3]),
        (
            vec![vec![1., 1.]; 4],
            vec![1., 2., 3., 4.],
            vec![0, 0, 1, 1],
        ),
        (
            vec![vec![1.]; 4],
            vec![1., f64::NAN, 3., 4.],
            vec![0, 0, 1, 1],
        ),
        (vec![vec![1.]; 4], vec![2.; 4], vec![0, 0, 1, 1]),
        (vec![vec![1.]; 4], vec![1., 1., 2., 2.], vec![0, 0, 1, 1]),
    ] {
        assert!(fit_random_intercept(x, y, g).is_err());
    }
}
#[test]
fn grouping_order_and_predictor_units_preserve_fit() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/lmm-r.json")).unwrap();
    let c = &fixture["cases"][0];
    let mut x: Vec<Vec<f64>> = serde_json::from_value(c["x"].clone()).unwrap();
    let mut y: Vec<f64> = serde_json::from_value(c["y"].clone()).unwrap();
    let mut g: Vec<usize> = serde_json::from_value(c["groups"].clone()).unwrap();
    let a = fit_random_intercept(x.clone(), y.clone(), g.clone()).unwrap();
    for row in &mut x {
        row[1] *= 1e-12;
    }
    x.reverse();
    y.reverse();
    g.reverse();
    let b = fit_random_intercept(x, y, g).unwrap();
    close(a.coefficients[1], b.coefficients[1] * 1e-12, "units");
    close(a.log_likelihood, b.log_likelihood, "row order");
    for (i, v) in a.fitted.iter().enumerate() {
        close(*v, b.fitted[b.fitted.len() - i - 1], "fitted row order");
    }
}
