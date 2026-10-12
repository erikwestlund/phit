use serde_json::{Value, json};
use stats_core::*;
fn compare(a: &Value, b: &Value, label: &str) {
    if let Some(values) = b.as_array() {
        assert_eq!(a.as_array().unwrap().len(), values.len());
        for (i, v) in values.iter().enumerate() {
            compare(&a[i], v, &format!("{label}/{i}"));
        }
    } else {
        let a = a.as_f64().unwrap();
        let b = b.as_f64().unwrap();
        assert!(
            (a - b).abs() <= 1e-7 + 1e-6 * b.abs(),
            "{label}: {a} != {b}"
        );
    }
}
#[test]
fn weighted_offset_fits_match_base_r_for_every_fixed_estimator() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/r-compatibility.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let x = serde_json::from_value(c["x"].clone()).unwrap();
        let y = serde_json::from_value(c["y"].clone()).unwrap();
        let w = serde_json::from_value(c["weights"].clone()).unwrap();
        let o = serde_json::from_value(c["offset"].clone()).unwrap();
        macro_rules! glm_json {($r:expr)=>{{let r=$r.unwrap();json!({"coefficients":r.coefficients,"covariance":r.covariance,"std_errors":r.std_errors,
            "z_statistics":r.z_statistics,"p_values":r.p_values,"conf_int":r.conf_int,"fitted":r.fitted,"residuals":r.residuals,
            "log_likelihood":r.log_likelihood,"deviance":r.deviance,"null_deviance":r.null_deviance,"aic":r.aic,"df_residual":r.df_residual})}};}
        let actual = match c["model"].as_str().unwrap() {
            "ols" => {
                let r = fit_ols_with_data(x, y, w, o).unwrap();
                json!({"coefficients":r.coefficients,"covariance":r.covariance,"std_errors":r.std_errors,
                "t_statistics":r.t_statistics,"p_values":r.p_values,"conf_int":r.conf_int,"fitted":r.fitted,"residuals":r.residuals,
                "rss":r.rss,"residual_variance":r.residual_variance,"sigma":r.sigma,"r_squared":r.r_squared,"adj_r_squared":r.adj_r_squared,
                "df_residual":r.df_residual,"log_likelihood":r.log_likelihood})
            }
            "logistic" => glm_json!(fit_logistic_with_data(x, y, w, o)),
            "probit" => glm_json!(fit_probit_with_data(x, y, w, o)),
            "poisson" => glm_json!(fit_poisson_with_data(x, y, w, o)),
            _ => unreachable!(),
        };
        for (k, v) in c["expected"].as_object().unwrap() {
            compare(&actual[k], v, &format!("{}/{k}", c["name"]));
        }
    }
}
#[test]
fn invalid_weights_offsets_and_binary_trial_counts_are_errors() {
    for (w, o) in [
        (vec![0.; 4], vec![]),
        (vec![-1.; 4], vec![]),
        (vec![f64::INFINITY; 4], vec![]),
        (vec![1.; 3], vec![]),
        (vec![], vec![f64::NAN; 4]),
        (vec![], vec![0.; 3]),
    ] {
        assert!(
            fit_ols_with_data(
                vec![vec![1.]; 4],
                vec![1., 2., 3., 4.],
                w.clone(),
                o.clone()
            )
            .is_err()
        );
        assert!(
            fit_logistic_with_data(
                vec![vec![1.]; 4],
                vec![0., 1., 0., 1.],
                w.clone(),
                o.clone()
            )
            .is_err()
        );
        assert!(
            fit_probit_with_data(
                vec![vec![1.]; 4],
                vec![0., 1., 0., 1.],
                w.clone(),
                o.clone()
            )
            .is_err()
        );
        assert!(fit_poisson_with_data(vec![vec![1.]; 4], vec![0., 1., 2., 1.], w, o).is_err());
    }
    assert!(
        fit_logistic_with_data(
            vec![vec![1.]; 4],
            vec![0., 1., 0., 1.],
            vec![0.5; 4],
            vec![]
        )
        .is_err()
    );
}
