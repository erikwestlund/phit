use serde_json::Value;
use stats_core::{ErrorKind, distributions::*};

fn quantile(name: &str, p: &Value) -> Box<dyn Fn(f64) -> f64> {
    let v = |key: &str| p[key].as_f64().unwrap();
    macro_rules! q {
        ($d:expr) => {{
            let d = $d;
            Box::new(move |u| d.quantile(u))
        }};
    }
    match name {
        "uniform" => Box::new(|u| u),
        "normal" => q!(Normal::new(v("mean"), v("sd")).unwrap()),
        "t" => q!(StudentT::new(v("df")).unwrap()),
        "chisquare" => q!(ChiSquare::new(v("df")).unwrap()),
        "f" => q!(FisherF::new(v("df1"), v("df2")).unwrap()),
        "beta" => q!(Beta::new(v("alpha"), v("beta")).unwrap()),
        "gamma" => q!(Gamma::new(v("shape"), v("scale")).unwrap()),
        "exponential" => q!(Exponential::new(v("rate")).unwrap()),
        "lognormal" => q!(LogNormal::new(v("meanlog"), v("sdlog")).unwrap()),
        "logistic" => q!(Logistic::new(v("location"), v("scale")).unwrap()),
        "binomial" => q!(Binomial::new(p["size"].as_u64().unwrap(), v("prob")).unwrap()),
        "poisson" => q!(Poisson::new(v("lambda")).unwrap()),
        "nbinom" => q!(NegativeBinomial::new(v("size"), v("prob")).unwrap()),
        "nbinom_mu" => q!(NegativeBinomial::with_mean(v("size"), v("mu")).unwrap()),
        _ => panic!("unknown family"),
    }
}

#[test]
fn independent_chacha_and_r_quantile_references() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/rng-python-r.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let seed = case["seed"].as_u64().unwrap();
        let family = case["distribution"].as_str().unwrap();
        let q = quantile(family, &case["params"]);
        let result = sample_quantiles(n, seed, &q).unwrap();
        let repeated = sample_quantiles(n, seed, &q).unwrap();
        assert_eq!(result, repeated);
        assert_eq!(sample_quantiles(7, seed, &q).unwrap(), result[..7]);
        for (&actual, expected) in result.iter().zip(case["expected"].as_array().unwrap()) {
            let expected = expected.as_f64().unwrap();
            if family == "uniform"
                || matches!(family, "binomial" | "poisson" | "nbinom" | "nbinom_mu")
            {
                assert_eq!(actual, expected, "{}", case["name"]);
            } else {
                assert!(
                    (actual - expected).abs() <= 1e-15 + 1e-12 * expected.abs(),
                    "{family}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn sampling_limits_and_no_shared_state() {
    assert!(sample_quantiles(0, 42, |p| p).unwrap().is_empty());
    assert_eq!(
        sample_quantiles(MAX_RNG_DRAWS + 1, 42, |p| p)
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        sample_quantiles(1, 42, |_| f64::INFINITY)
            .unwrap_err()
            .kind(),
        ErrorKind::Nonfinite
    );
    assert_eq!(
        sample_quantiles(1, 42, |_| f64::NAN).unwrap_err().kind(),
        ErrorKind::Nonfinite
    );
    let a = sample_quantiles(128, 42, |p| p).unwrap();
    let b = sample_quantiles(128, 43, |p| p).unwrap();
    assert_ne!(a, b);
    assert_eq!(a, sample_quantiles(128, 42, |p| p).unwrap());
    assert!(a.iter().all(|&p| p > 0. && p < 1.));
}

#[test]
fn generated_normal_and_count_samples_have_expected_moments() {
    let normal = Normal::new(3., 2.).unwrap();
    let poisson = Poisson::new(12.).unwrap();
    let x = sample_quantiles(50000, 173, |p| normal.quantile(p)).unwrap();
    let k = sample_quantiles(50000, 174, |p| poisson.quantile(p)).unwrap();
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    assert!((mean(&x) - 3.).abs() < 0.06);
    assert!((stats_core::descriptive::variance(&x, true).unwrap() - 4.).abs() < 0.15);
    assert!((mean(&k) - 12.).abs() < 0.1);
    assert!((stats_core::descriptive::variance(&k, true).unwrap() - 12.).abs() < 0.4);
    assert!(k.iter().all(|&v| v >= 0. && v.fract() == 0.));
}
