//! Assess public distribution kernels against the existing estimator formulas.
//! No estimator is changed by this experiment. R critical values are the same
//! saved references exercised by the OLS core tests.
use serde_json::{Value, json};
use statrs::distribution::{ContinuousCDF, StudentsT};
use stats_core::distributions::{Normal, StudentT};

fn legacy_normal_logcdf(value: f64) -> f64 {
    if value < -8. {
        let x = -value;
        let mut denominator = x;
        for k in (1..=100).rev() {
            denominator = x + k as f64 / denominator;
        }
        -0.5 * x * x - 0.5 * (2. * std::f64::consts::PI).ln() - denominator.ln()
    } else {
        (0.5 * statrs::function::erf::erfc(-value / std::f64::consts::SQRT_2)).ln()
    }
}

fn legacy_t_critical(df: f64) -> f64 {
    let distribution = StudentsT::new(0., 1., df).unwrap();
    let (mut lower, mut upper) = (0., 1.);
    while distribution.sf(upper) > 0.025 {
        upper *= 2.;
    }
    for _ in 0..64 {
        let midpoint = (lower + upper) / 2.;
        if distribution.sf(midpoint) > 0.025 {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    (lower + upper) / 2.
}

fn number(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else if v == f64::NEG_INFINITY {
        json!("-Inf")
    } else if v == f64::INFINITY {
        json!("Inf")
    } else {
        json!("NaN")
    }
}

fn main() {
    let mut t = Vec::new();
    for (df, reference) in [
        (1., 12.706204736174692),
        (3., 3.1824463052837078),
        (30., 2.0422724563012378),
        (997., 1.9623462360894495),
        (99997., 1.9599877082463284),
        (999997., 1.9599663568212229),
    ] {
        let legacy = legacy_t_critical(df);
        let candidate = StudentT::new(df).unwrap().quantile(0.975);
        t.push(json!({"df":df,"reference_r":reference,"legacy":legacy,"candidate":candidate,
            "legacy_absolute_error":(legacy-reference).abs(),"candidate_absolute_error":(candidate-reference).abs(),
            "bitwise_equal":legacy.to_bits()==candidate.to_bits()}));
    }
    let reference: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/distributions-mpmath.json"
    ))
    .unwrap();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            c["distribution"] == "normal"
                && c["operation"] == "logcdf"
                && c["params"]["mean"] == 0
                && c["params"]["sd"] == 1
        })
        .unwrap();
    let normal = Normal::new(0., 1.).unwrap();
    let mut tails = Vec::new();
    for (x, r) in case["input"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["expected"].as_array().unwrap())
    {
        if let Some(x) = x.as_f64() {
            let legacy = legacy_normal_logcdf(x);
            let candidate = normal.logcdf(x);
            tails.push(json!({"x":x,"reference_mpmath":r,"legacy":number(legacy),"candidate":number(candidate),
                "bitwise_equal":legacy.to_bits()==candidate.to_bits()}));
        }
    }
    println!(
        "{}",
        json!({"generator":"cargo run -p stats-core --release --locked --example distribution_reuse_audit",
        "decision":"Retain the existing OLS/probit numerical paths in this release: the public candidates differ in rounding. Candidate t quantiles improve large-df reference error, but changing model outputs needs a separate compatibility decision; the distribution APIs do not require it.",
        "t_critical":t,"normal_logcdf":tails})
    );
}
