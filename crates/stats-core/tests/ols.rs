use serde_json::Value;
use stats_core::fit_ols;

fn close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 + 1e-7 * expected.abs(),
        "{label}: {actual} != {expected}"
    );
}
#[test]
fn agrees_with_r() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/ols-r.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let x = serde_json::from_value(case["x"].clone()).unwrap();
        let y = serde_json::from_value(case["y"].clone()).unwrap();
        let r = fit_ols(x, y).unwrap();
        let e = &case["expected"];
        for (i, row) in r.covariance.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                close(
                    *value,
                    e["covariance"][i][j].as_f64().unwrap(),
                    "covariance",
                );
            }
        }
        let name = case["name"].as_str().unwrap();
        let perfect = name == "perfect" || name == "zero";
        for (key, values) in [
            ("coefficients", &r.coefficients),
            ("fitted", &r.fitted),
            ("residuals", &r.residuals),
            ("std_errors", &r.std_errors),
        ] {
            for (i, v) in values.iter().enumerate() {
                close(*v, e[key][i].as_f64().unwrap(), key);
            }
        }
        if !perfect {
            for (key, values) in [("t_statistics", &r.t_statistics), ("p_values", &r.p_values)] {
                for (i, v) in values.iter().enumerate() {
                    close(*v, e[key][i].as_f64().unwrap(), key);
                }
            }
            for (key, v) in [
                ("r_squared", r.r_squared),
                ("adj_r_squared", r.adj_r_squared),
                ("log_likelihood", r.log_likelihood),
                ("rss", r.rss),
                ("sigma", r.sigma),
                ("residual_variance", r.residual_variance),
            ] {
                close(v, e[key].as_f64().unwrap(), key);
            }
        }
        for (i, ci) in r.conf_int.iter().enumerate() {
            for (j, v) in ci.iter().enumerate() {
                if let Some(expected) = e["conf_int"][i][j].as_f64() {
                    close(*v, expected, "conf_int");
                } else {
                    assert_eq!(*v, 0.0);
                }
            }
        }
        assert_eq!(r.df_residual, e["df_residual"].as_u64().unwrap() as usize);
        if name == "zero" {
            assert!(r.t_statistics.iter().all(|v| v.is_nan()));
            assert!(r.r_squared.is_nan());
            assert_eq!(r.log_likelihood, f64::INFINITY);
        }
    }
}
#[test]
fn rejects_invalid_inputs() {
    for (x, y) in [
        (vec![], vec![]),
        (vec![vec![]], vec![1.]),
        (vec![vec![1.], vec![1., 2.]], vec![1., 2.]),
        (vec![vec![1.], vec![2.]], vec![1.]),
        (vec![vec![1., 2.], vec![1., 3.]], vec![1., 2.]),
        (vec![vec![1., 1.]; 3], vec![1., 2., 3.]),
        (vec![vec![0.]; 3], vec![1., 2., 3.]),
        (vec![vec![f64::NAN]; 3], vec![1., 2., 3.]),
        (vec![vec![1.]; 3], vec![1., f64::INFINITY, 3.]),
    ] {
        assert!(fit_ols(x, y).is_err());
    }
}
#[test]
fn predictor_units_do_not_change_fit() {
    let a = fit_ols(
        vec![vec![1., 1.], vec![1., 2.], vec![1., 3.], vec![1., 4.]],
        vec![2., 3., 5., 6.],
    )
    .unwrap();
    let b = fit_ols(
        vec![
            vec![1., 1e-12],
            vec![1., 2e-12],
            vec![1., 3e-12],
            vec![1., 4e-12],
        ],
        vec![2., 3., 5., 6.],
    )
    .unwrap();
    close(
        a.coefficients[1],
        b.coefficients[1] * 1e-12,
        "scaled coefficient",
    );
    for (x, y) in a.fitted.iter().zip(b.fitted) {
        close(*x, y, "scaled fit");
    }
}
