//! Internal Rust loop on the same JSON inputs as the PHP/R/SciPy workers.
use serde_json::{Value, json};
use stats_core::{descriptive, distributions::*};
use std::{env, fs::File, hint::black_box, time::Instant};

fn compact(values: &[f64]) -> Value {
    let n = values.len();
    let indices: Vec<usize> = (0..17).map(|i| i * (n - 1) / 16).collect();
    json!({"length": n, "sum": values.iter().sum::<f64>(),
        "sum_squares": values.iter().map(|v| v*v).sum::<f64>(),
        "min": values.iter().copied().fold(f64::INFINITY, f64::min),
        "max": values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "sample": indices.iter().map(|&i| values[i]).collect::<Vec<_>>()})
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let start = Instant::now();
    let input: Value = serde_json::from_reader(File::open(&args[1]).unwrap()).unwrap();
    let x: Vec<f64> = serde_json::from_value(input["values"].clone()).unwrap();
    let op = input["operation"].as_str().unwrap();
    let normal = Normal::new(0., 1.).unwrap();
    let t = StudentT::new(12.).unwrap();
    let binomial = Binomial::new(100, 0.3).unwrap();
    let poisson = Poisson::new(12.).unwrap();
    let beta = Beta::new(2., 5.).unwrap();
    let chi = ChiSquare::new(1e6).unwrap();
    let f: Box<dyn Fn(f64) -> f64> = match op {
        "normal_cdf" => Box::new(move |v| normal.cdf(v)),
        "normal_sf" => Box::new(move |v| normal.sf(v)),
        "t_quantile" => Box::new(move |v| t.quantile(v)),
        "binomial_logpmf" => Box::new(move |v| binomial.logpmf(v)),
        "poisson_logpmf" => Box::new(move |v| poisson.logpmf(v)),
        "beta_quantile" => Box::new(move |v| beta.quantile(v)),
        "chisquare_cdf" => Box::new(move |v| chi.cdf(v)),
        "mean" | "variance" => Box::new(|v| v),
        _ => panic!("unknown operation"),
    };
    let calculate = || -> Vec<f64> {
        match op {
            "mean" => vec![descriptive::mean(black_box(&x)).unwrap()],
            "variance" => vec![descriptive::variance(black_box(&x), false).unwrap()],
            "t_quantile" | "beta_quantile" => evaluate_quantiles(black_box(&x), &f).unwrap(),
            _ => evaluate_all(black_box(&x), &f).unwrap(),
        }
    };
    let load = start.elapsed().as_secs_f64();
    let batches = input["batch"].as_u64().unwrap();
    let repeats: usize = args[3].parse().unwrap();
    let mut result = calculate();
    let mut times = Vec::new();
    if args[2] == "time" {
        for _ in 0..repeats {
            let start = Instant::now();
            for _ in 0..batches {
                result = black_box(calculate());
            }
            times.push(start.elapsed().as_secs_f64() / batches as f64);
        }
    }
    println!(
        "{}",
        json!({"load_seconds": load, "seconds": times, "result": compact(&result)})
    );
}
