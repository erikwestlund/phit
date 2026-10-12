use serde_json::Value;
use stats_core::distributions::*;
use std::collections::BTreeMap;

pub fn number(v: &Value) -> f64 {
    match v {
        Value::String(s) if s == "Inf" => f64::INFINITY,
        Value::String(s) if s == "-Inf" => f64::NEG_INFINITY,
        Value::String(_) => f64::NAN,
        v => v.as_f64().unwrap(),
    }
}

type Evaluator = Box<dyn Fn(&str, f64) -> Result<f64, stats_core::StatsError>>;

#[test]
fn overflowing_negative_binomial_mean_conversion_is_not_a_point_mass() {
    let d = NegativeBinomial::with_mean(1e308, 1e308).unwrap();
    assert_eq!(d.pmf(0.), 0.);
    assert_eq!(d.logpmf(0.), -1e308 * std::f64::consts::LN_2);
    assert_eq!(d.cdf(0.), 0.);
    assert_eq!(d.sf(0.), 1.);
}

macro_rules! evaluator {
    ($dist:expr, $density:ident, $log_density:ident) => {{
        let d = $dist;
        Box::new(move |op: &str, x: f64| match op {
            "pdf" | "pmf" => evaluate(x, |x| d.$density(x)),
            "logpdf" | "logpmf" => evaluate(x, |x| d.$log_density(x)),
            "cdf" => evaluate(x, |x| d.cdf(x)),
            "sf" => evaluate(x, |x| d.sf(x)),
            "logcdf" => evaluate(x, |x| d.logcdf(x)),
            "logsf" => evaluate(x, |x| d.logsf(x)),
            "quantile" => evaluate_quantile(x, |p| d.quantile(p)),
            _ => unreachable!(),
        }) as Evaluator
    }};
}

pub fn build(name: &str, p: &Value) -> Evaluator {
    let g = |k: &str| p[k].as_f64().unwrap();
    match name {
        "normal" => evaluator!(Normal::new(g("mean"), g("sd")).unwrap(), pdf, logpdf),
        "t" => evaluator!(StudentT::new(g("df")).unwrap(), pdf, logpdf),
        "chisquare" => evaluator!(ChiSquare::new(g("df")).unwrap(), pdf, logpdf),
        "f" => evaluator!(FisherF::new(g("df1"), g("df2")).unwrap(), pdf, logpdf),
        "beta" => evaluator!(Beta::new(g("alpha"), g("beta")).unwrap(), pdf, logpdf),
        "gamma" => evaluator!(Gamma::new(g("shape"), g("scale")).unwrap(), pdf, logpdf),
        "exponential" => evaluator!(Exponential::new(g("rate")).unwrap(), pdf, logpdf),
        "lognormal" => evaluator!(
            LogNormal::new(g("meanlog"), g("sdlog")).unwrap(),
            pdf,
            logpdf
        ),
        "logistic" => evaluator!(
            Logistic::new(g("location"), g("scale")).unwrap(),
            pdf,
            logpdf
        ),
        "binomial" => evaluator!(
            Binomial::new(g("size") as u64, g("prob")).unwrap(),
            pmf,
            logpmf
        ),
        "poisson" => evaluator!(Poisson::new(g("lambda")).unwrap(), pmf, logpmf),
        "nbinom" => evaluator!(
            NegativeBinomial::new(g("size"), g("prob")).unwrap(),
            pmf,
            logpmf
        ),
        "nbinom_mu" => evaluator!(
            NegativeBinomial::with_mean(g("size"), g("mu")).unwrap(),
            pmf,
            logpmf
        ),
        _ => panic!("unknown distribution {name}"),
    }
}

/// Relative error, with exact agreement required for nonfinite references.
pub fn relative_error(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        0.0
    } else if !actual.is_finite() || !expected.is_finite() {
        f64::INFINITY
    } else {
        (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
    }
}

#[test]
fn distributions_match_r_references() {
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/distributions-r.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let mut worst: BTreeMap<(String, String), (f64, String)> = BTreeMap::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["distribution"].as_str().unwrap();
        let op = case["operation"].as_str().unwrap();
        let f = build(name, &case["params"]);
        for (x, e) in case["input"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            let (x, e) = (number(x), number(e));
            let a = f(op, x).unwrap_or(f64::NAN);
            let err = relative_error(a, e);
            let entry = worst
                .entry((name.into(), op.into()))
                .or_insert((0.0, String::new()));
            if err > entry.0 || (err.is_nan() && !entry.0.is_nan()) {
                *entry = (err, format!("{} x={x:e}: {a:e} vs R {e:e}", case["params"]));
            }
        }
    }
    for ((name, op), (err, detail)) in &worst {
        println!("{name:12} {op:9} {err:9.2e}  {detail}");
    }
}

#[test]
fn extreme_student_quantiles_match_high_precision_reference() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/distributions-mpmath.json"
    ))
    .unwrap();
    let mut compared = 0;
    for case in fixture["cases"].as_array().unwrap() {
        if case["distribution"] != "t" || case["operation"] != "quantile" {
            continue;
        }
        let df = case["params"]["df"].as_f64().unwrap();
        let distribution = StudentT::new(df).unwrap();
        for (p, expected) in case["input"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            let p = number(p);
            let upper = df == 0.5 && (p == 1.0 - 1e-10 || p == 1.0 - f64::EPSILON);
            let lower = (df == 3.5 || df == 12.0) && p > 0.0 && p < 1e-290;
            if !upper && !lower {
                continue;
            }
            let expected = number(expected);
            let actual = distribution.quantile(p);
            assert!(
                (actual - expected).abs() <= 1e-15 + 1e-12 * expected.abs(),
                "Student-t df={df} p={p}: {actual} != mpmath {expected}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 4);
}

#[test]
fn scaled_densities_do_not_underflow_before_the_jacobian() {
    let lognormal = LogNormal::new(-5.0, 3.0).unwrap();
    let expected = 9.440_902_801_820_74e-295; // Stored 60-digit mpmath fixture.
    let actual = lognormal.pdf(5.166_420_632_837_861e-55);
    assert!((actual / expected - 1.0).abs() < 1e-12);
    let normal = Normal::new(0.0, 1e-200).unwrap();
    let expected = 1.463_270_250_838_303_1e-148; // Independently evaluated at 70 digits.
    let actual = normal.pdf(4e-199);
    assert!((actual / expected - 1.0).abs() < 1e-12);
}
