//! Audit raw statrs 0.19 against the saved R reference values.
//! Run: cargo run -p stats-core --release --example statrs_audit
//! Reports the worst relative error per distribution/operation, using
//! ln(cdf)/ln(sf) for log probabilities because statrs has no log-tail API.
use serde_json::Value;
use statrs::distribution::{
    Beta, Binomial, ChiSquared, Continuous, ContinuousCDF, Discrete, DiscreteCDF, Exp,
    FisherSnedecor, Gamma, LogNormal, NegativeBinomial, Normal, Poisson, StudentsT,
};
use std::collections::BTreeMap;

fn number(v: &Value) -> f64 {
    match v {
        Value::String(s) if s == "Inf" => f64::INFINITY,
        Value::String(s) if s == "-Inf" => f64::NEG_INFINITY,
        Value::String(_) => f64::NAN,
        v => v.as_f64().unwrap(),
    }
}

fn error(actual: f64, expected: f64) -> f64 {
    if actual == expected || (actual.is_nan() && expected.is_nan()) {
        0.0
    } else if !actual.is_finite() || !expected.is_finite() {
        f64::INFINITY
    } else {
        (actual - expected).abs() / expected.abs().max(1e-300)
    }
}

enum Dist {
    C(Box<dyn ContinuousLike>),
    D(Box<dyn DiscreteLike>),
}
trait ContinuousLike {
    fn eval(&self, op: &str, x: f64) -> f64;
}
trait DiscreteLike {
    fn eval(&self, op: &str, x: f64) -> f64;
}
impl<T: Continuous<f64, f64> + ContinuousCDF<f64, f64>> ContinuousLike for T {
    fn eval(&self, op: &str, x: f64) -> f64 {
        match op {
            "pdf" => self.pdf(x),
            "logpdf" => self.ln_pdf(x),
            "cdf" => self.cdf(x),
            "sf" => self.sf(x),
            "logcdf" => self.cdf(x).ln(),
            "logsf" => self.sf(x).ln(),
            _ => self.inverse_cdf(x),
        }
    }
}
impl<T: Discrete<u64, f64> + DiscreteCDF<u64, f64>> DiscreteLike for T {
    fn eval(&self, op: &str, x: f64) -> f64 {
        if op == "quantile" {
            return self.inverse_cdf(x) as f64;
        }
        if x < 0.0 {
            return match op {
                "pmf" | "cdf" => 0.0,
                "logpmf" | "logcdf" => f64::NEG_INFINITY,
                "sf" => 1.0,
                _ => 0.0,
            };
        }
        let k = x.floor() as u64;
        match op {
            "pmf" if x.fract() != 0.0 => 0.0,
            "logpmf" if x.fract() != 0.0 => f64::NEG_INFINITY,
            "pmf" => self.pmf(k),
            "logpmf" => self.ln_pmf(k),
            "cdf" => self.cdf(k),
            "sf" => self.sf(k),
            "logcdf" => self.cdf(k).ln(),
            _ => self.sf(k).ln(),
        }
    }
}

fn build(name: &str, p: &Value) -> Option<Dist> {
    let g = |k: &str| p[k].as_f64().unwrap();
    Some(match name {
        "normal" => Dist::C(Box::new(Normal::new(g("mean"), g("sd")).ok()?)),
        "t" => Dist::C(Box::new(StudentsT::new(0.0, 1.0, g("df")).ok()?)),
        "chisquare" => Dist::C(Box::new(ChiSquared::new(g("df")).ok()?)),
        "f" => Dist::C(Box::new(FisherSnedecor::new(g("df1"), g("df2")).ok()?)),
        "beta" => Dist::C(Box::new(Beta::new(g("alpha"), g("beta")).ok()?)),
        "gamma" => Dist::C(Box::new(Gamma::new(g("shape"), 1.0 / g("scale")).ok()?)),
        "exponential" => Dist::C(Box::new(Exp::new(g("rate")).ok()?)),
        "lognormal" => Dist::C(Box::new(LogNormal::new(g("meanlog"), g("sdlog")).ok()?)),
        "binomial" => Dist::D(Box::new(Binomial::new(g("prob"), g("size") as u64).ok()?)),
        "poisson" => Dist::D(Box::new(Poisson::new(g("lambda")).ok()?)),
        "nbinom" => Dist::D(Box::new(NegativeBinomial::new(g("size"), g("prob")).ok()?)),
        "nbinom_mu" => {
            let (mu, size) = (g("mu"), g("size"));
            Dist::D(Box::new(
                NegativeBinomial::new(size, size / (size + mu)).ok()?,
            ))
        }
        _ => return None, // statrs 0.19 has no logistic distribution.
    })
}

fn main() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/distributions-r.json"
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut worst: BTreeMap<(String, String), (f64, String)> = BTreeMap::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["distribution"].as_str().unwrap();
        let op = case["operation"].as_str().unwrap();
        let key = (name.to_owned(), op.to_owned());
        let Some(dist) = build(name, &case["params"]) else {
            worst.insert(
                key,
                (f64::NAN, "unsupported parameters/distribution".into()),
            );
            continue;
        };
        for (x, expected) in case["input"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            let (x, expected) = (number(x), number(expected));
            if op == "quantile" && !(0.0..=1.0).contains(&x) {
                continue;
            }
            let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &dist {
                Dist::C(d) => d.eval(op, x),
                Dist::D(d) => d.eval(op, x),
            }))
            .unwrap_or(f64::NAN);
            let e = error(actual, expected);
            let entry = worst.entry(key.clone()).or_insert((0.0, String::new()));
            if (e.is_nan() || e > entry.0) && !entry.0.is_nan() {
                *entry = (
                    e,
                    format!("{} x={x:e}: {actual:e} vs R {expected:e}", case["params"]),
                );
            }
        }
    }
    for ((name, op), (e, detail)) in worst {
        println!("{name:12} {op:9} {e:9.2e}  {detail}");
    }
}
