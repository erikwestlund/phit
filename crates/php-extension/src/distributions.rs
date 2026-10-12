//! PHP bindings for probability distributions and descriptive statistics.
//!
//! Every operation has a scalar function and an `_array` batch function. Both
//! construct the distribution once, apply the same validation through
//! stats_core::distributions::{evaluate, evaluate_quantile}, and differ only in
//! iterating a PHP array natively instead of crossing the PHP boundary per value.
use crate::stats_exception;
use ext_php_rs::{
    boxed::ZBox, convert::FromZval, convert::StubLiteral, flags::DataType, prelude::*,
    types::ZendHashTable, types::Zval,
};
use stats_core::distributions::{self as dist, evaluate, evaluate_quantile};
use stats_core::{ErrorKind, StatsError};

/// A PHP int or float argument. ext-php-rs' f64 accepts floats only; PHP
/// callers routinely pass integers such as degrees of freedom.
#[derive(Clone, Copy)]
pub struct Real(pub f64);
impl FromZval<'_> for Real {
    const TYPE: DataType = DataType::Double;
    fn from_zval(zval: &Zval) -> Option<Self> {
        zval.double()
            .or_else(|| zval.long().map(|v| v as f64))
            .map(Real)
    }
}
impl From<f64> for Real {
    fn from(value: f64) -> Self {
        Real(value)
    }
}
impl StubLiteral for Real {
    fn stub_literal(&self) -> String {
        self.0.stub_literal()
    }
}

fn error(kind: ErrorKind, message: &str) -> PhpException {
    stats_exception(StatsError::new(kind, message))
}

// Array elements must be PHP ints or floats; there is no missing-value policy.
fn element(value: &Zval) -> PhpResult<f64> {
    value
        .double()
        .or_else(|| value.long().map(|v| v as f64))
        .ok_or_else(|| {
            error(
                ErrorKind::InvalidInput,
                "array elements must be integers or floats",
            )
        })
}

/// Apply f to every element, preserving iteration order; keys are not kept.
fn map_array(
    values: &ZendHashTable,
    f: impl Fn(f64) -> Result<f64, StatsError>,
) -> PhpResult<ZBox<ZendHashTable>> {
    let capacity = u32::try_from(values.len())
        .map_err(|_| error(ErrorKind::Unsupported, "array is too large"))?;
    let mut out = ZendHashTable::with_capacity(capacity);
    for value in values.values() {
        out.push(f(element(value)?).map_err(stats_exception)?)?;
    }
    Ok(out)
}

fn trials(value: i64) -> Result<u64, StatsError> {
    u64::try_from(value)
        .map_err(|_| StatsError::new(ErrorKind::InvalidInput, "size must be nonnegative"))
}

// One scalar/batch pair for one operation.
macro_rules! pair {
    (
        $(#[$attr:meta])*
        $scalar:ident, $batch:ident, $point:ident, $points:ident, $check:ident, $method:ident,
        ($($param:ident: $ty:ty),*) => $build:expr
    ) => {
        #[php_function]
        $(#[$attr])*
        pub fn $scalar($point: Real, $($param: $ty),*) -> PhpResult<f64> {
            let distribution = $build.map_err(stats_exception)?;
            $check($point.0, |v| distribution.$method(v)).map_err(stats_exception)
        }
        #[php_function]
        $(#[$attr])*
        pub fn $batch($points: &ZendHashTable, $($param: $ty),*) -> PhpResult<ZBox<ZendHashTable>> {
            let distribution = $build.map_err(stats_exception)?;
            map_array($points, |v| $check(v, |v| distribution.$method(v)))
        }
    };
}

// Seven evaluation operations and explicitly seeded batch sampling.
macro_rules! distribution {
    (
        $(#[$attr:meta])*
        register: $register:ident,
        rng: $f_rng:ident,
        point: $point:ident, $points:ident, density: $density:ident, $log_density:ident,
        params: ($($param:ident: $ty:ty),*) => $build:expr,
        functions: [
            $f_density:ident, $f_density_array:ident, $f_log_density:ident, $f_log_density_array:ident,
            $f_cdf:ident, $f_cdf_array:ident, $f_logcdf:ident, $f_logcdf_array:ident,
            $f_sf:ident, $f_sf_array:ident, $f_logsf:ident, $f_logsf_array:ident,
            $f_quantile:ident, $f_quantile_array:ident
        ]
    ) => {
        pair!($(#[$attr])* $f_density, $f_density_array, $point, $points, evaluate, $density, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_log_density, $f_log_density_array, $point, $points, evaluate, $log_density, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_cdf, $f_cdf_array, $point, $points, evaluate, cdf, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_logcdf, $f_logcdf_array, $point, $points, evaluate, logcdf, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_sf, $f_sf_array, $point, $points, evaluate, sf, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_logsf, $f_logsf_array, $point, $points, evaluate, logsf, ($($param: $ty),*) => $build);
        pair!($(#[$attr])* $f_quantile, $f_quantile_array, p, probabilities, evaluate_quantile, quantile, ($($param: $ty),*) => $build);
        #[php_function]
        $(#[$attr])*
        pub fn $f_rng(n: i64, seed: i64, $($param: $ty),*) -> PhpResult<Vec<f64>> {
            let distribution = $build.map_err(stats_exception)?;
            let n = usize::try_from(n).map_err(|_| error(ErrorKind::InvalidInput, "n must be nonnegative"))?;
            let seed = u64::try_from(seed).map_err(|_| error(ErrorKind::InvalidInput, "seed must be nonnegative"))?;
            dist::sample_quantiles(n, seed, |p| distribution.quantile(p)).map_err(stats_exception)
        }
        fn $register(module: ModuleBuilder) -> ModuleBuilder {
            module
                .function(wrap_function!($f_density))
                .function(wrap_function!($f_density_array))
                .function(wrap_function!($f_log_density))
                .function(wrap_function!($f_log_density_array))
                .function(wrap_function!($f_cdf))
                .function(wrap_function!($f_cdf_array))
                .function(wrap_function!($f_logcdf))
                .function(wrap_function!($f_logcdf_array))
                .function(wrap_function!($f_sf))
                .function(wrap_function!($f_sf_array))
                .function(wrap_function!($f_logsf))
                .function(wrap_function!($f_logsf_array))
                .function(wrap_function!($f_quantile))
                .function(wrap_function!($f_quantile_array))
                .function(wrap_function!($f_rng))
        }
    };
}

distribution!(
    #[php(defaults(mean = 0.0, sd = 1.0))]
    register: register_normal,
    rng: stats_normal_rng,
    point: x, values, density: pdf, logpdf,
    params: (mean: Real, sd: Real) => dist::Normal::new(mean.0, sd.0),
    functions: [
        stats_normal_pdf, stats_normal_pdf_array, stats_normal_logpdf, stats_normal_logpdf_array,
        stats_normal_cdf, stats_normal_cdf_array, stats_normal_logcdf, stats_normal_logcdf_array,
        stats_normal_sf, stats_normal_sf_array, stats_normal_logsf, stats_normal_logsf_array,
        stats_normal_quantile, stats_normal_quantile_array
    ]
);
distribution!(
    register: register_t,
    rng: stats_t_rng,
    point: x, values, density: pdf, logpdf,
    params: (df: Real) => dist::StudentT::new(df.0),
    functions: [
        stats_t_pdf, stats_t_pdf_array, stats_t_logpdf, stats_t_logpdf_array,
        stats_t_cdf, stats_t_cdf_array, stats_t_logcdf, stats_t_logcdf_array,
        stats_t_sf, stats_t_sf_array, stats_t_logsf, stats_t_logsf_array,
        stats_t_quantile, stats_t_quantile_array
    ]
);
distribution!(
    register: register_chisquare,
    rng: stats_chisquare_rng,
    point: x, values, density: pdf, logpdf,
    params: (df: Real) => dist::ChiSquare::new(df.0),
    functions: [
        stats_chisquare_pdf, stats_chisquare_pdf_array, stats_chisquare_logpdf, stats_chisquare_logpdf_array,
        stats_chisquare_cdf, stats_chisquare_cdf_array, stats_chisquare_logcdf, stats_chisquare_logcdf_array,
        stats_chisquare_sf, stats_chisquare_sf_array, stats_chisquare_logsf, stats_chisquare_logsf_array,
        stats_chisquare_quantile, stats_chisquare_quantile_array
    ]
);
distribution!(
    register: register_f,
    rng: stats_f_rng,
    point: x, values, density: pdf, logpdf,
    params: (df1: Real, df2: Real) => dist::FisherF::new(df1.0, df2.0),
    functions: [
        stats_f_pdf, stats_f_pdf_array, stats_f_logpdf, stats_f_logpdf_array,
        stats_f_cdf, stats_f_cdf_array, stats_f_logcdf, stats_f_logcdf_array,
        stats_f_sf, stats_f_sf_array, stats_f_logsf, stats_f_logsf_array,
        stats_f_quantile, stats_f_quantile_array
    ]
);
distribution!(
    register: register_beta,
    rng: stats_beta_rng,
    point: x, values, density: pdf, logpdf,
    params: (alpha: Real, beta: Real) => dist::Beta::new(alpha.0, beta.0),
    functions: [
        stats_beta_pdf, stats_beta_pdf_array, stats_beta_logpdf, stats_beta_logpdf_array,
        stats_beta_cdf, stats_beta_cdf_array, stats_beta_logcdf, stats_beta_logcdf_array,
        stats_beta_sf, stats_beta_sf_array, stats_beta_logsf, stats_beta_logsf_array,
        stats_beta_quantile, stats_beta_quantile_array
    ]
);
distribution!(
    #[php(defaults(scale = 1.0))]
    register: register_gamma,
    rng: stats_gamma_rng,
    point: x, values, density: pdf, logpdf,
    params: (shape: Real, scale: Real) => dist::Gamma::new(shape.0, scale.0),
    functions: [
        stats_gamma_pdf, stats_gamma_pdf_array, stats_gamma_logpdf, stats_gamma_logpdf_array,
        stats_gamma_cdf, stats_gamma_cdf_array, stats_gamma_logcdf, stats_gamma_logcdf_array,
        stats_gamma_sf, stats_gamma_sf_array, stats_gamma_logsf, stats_gamma_logsf_array,
        stats_gamma_quantile, stats_gamma_quantile_array
    ]
);
distribution!(
    #[php(defaults(rate = 1.0))]
    register: register_exponential,
    rng: stats_exponential_rng,
    point: x, values, density: pdf, logpdf,
    params: (rate: Real) => dist::Exponential::new(rate.0),
    functions: [
        stats_exponential_pdf, stats_exponential_pdf_array, stats_exponential_logpdf, stats_exponential_logpdf_array,
        stats_exponential_cdf, stats_exponential_cdf_array, stats_exponential_logcdf, stats_exponential_logcdf_array,
        stats_exponential_sf, stats_exponential_sf_array, stats_exponential_logsf, stats_exponential_logsf_array,
        stats_exponential_quantile, stats_exponential_quantile_array
    ]
);
distribution!(
    #[php(defaults(meanlog = 0.0, sdlog = 1.0))]
    register: register_lognormal,
    rng: stats_lognormal_rng,
    point: x, values, density: pdf, logpdf,
    params: (meanlog: Real, sdlog: Real) => dist::LogNormal::new(meanlog.0, sdlog.0),
    functions: [
        stats_lognormal_pdf, stats_lognormal_pdf_array, stats_lognormal_logpdf, stats_lognormal_logpdf_array,
        stats_lognormal_cdf, stats_lognormal_cdf_array, stats_lognormal_logcdf, stats_lognormal_logcdf_array,
        stats_lognormal_sf, stats_lognormal_sf_array, stats_lognormal_logsf, stats_lognormal_logsf_array,
        stats_lognormal_quantile, stats_lognormal_quantile_array
    ]
);
distribution!(
    #[php(defaults(location = 0.0, scale = 1.0))]
    register: register_logistic,
    rng: stats_logistic_rng,
    point: x, values, density: pdf, logpdf,
    params: (location: Real, scale: Real) => dist::Logistic::new(location.0, scale.0),
    functions: [
        stats_logistic_pdf, stats_logistic_pdf_array, stats_logistic_logpdf, stats_logistic_logpdf_array,
        stats_logistic_cdf, stats_logistic_cdf_array, stats_logistic_logcdf, stats_logistic_logcdf_array,
        stats_logistic_sf, stats_logistic_sf_array, stats_logistic_logsf, stats_logistic_logsf_array,
        stats_logistic_quantile, stats_logistic_quantile_array
    ]
);
distribution!(
    register: register_binomial,
    rng: stats_binomial_rng,
    point: k, counts, density: pmf, logpmf,
    params: (size: i64, prob: Real) => trials(size).and_then(|n| dist::Binomial::new(n, prob.0)),
    functions: [
        stats_binomial_pmf, stats_binomial_pmf_array, stats_binomial_logpmf, stats_binomial_logpmf_array,
        stats_binomial_cdf, stats_binomial_cdf_array, stats_binomial_logcdf, stats_binomial_logcdf_array,
        stats_binomial_sf, stats_binomial_sf_array, stats_binomial_logsf, stats_binomial_logsf_array,
        stats_binomial_quantile, stats_binomial_quantile_array
    ]
);
distribution!(
    register: register_poisson,
    rng: stats_poisson_rng,
    point: k, counts, density: pmf, logpmf,
    params: (lambda: Real) => dist::Poisson::new(lambda.0),
    functions: [
        stats_poisson_pmf, stats_poisson_pmf_array, stats_poisson_logpmf, stats_poisson_logpmf_array,
        stats_poisson_cdf, stats_poisson_cdf_array, stats_poisson_logcdf, stats_poisson_logcdf_array,
        stats_poisson_sf, stats_poisson_sf_array, stats_poisson_logsf, stats_poisson_logsf_array,
        stats_poisson_quantile, stats_poisson_quantile_array
    ]
);
distribution!(
    register: register_nbinom,
    rng: stats_nbinom_rng,
    point: k, counts, density: pmf, logpmf,
    params: (size: Real, prob: Real) => dist::NegativeBinomial::new(size.0, prob.0),
    functions: [
        stats_nbinom_pmf, stats_nbinom_pmf_array, stats_nbinom_logpmf, stats_nbinom_logpmf_array,
        stats_nbinom_cdf, stats_nbinom_cdf_array, stats_nbinom_logcdf, stats_nbinom_logcdf_array,
        stats_nbinom_sf, stats_nbinom_sf_array, stats_nbinom_logsf, stats_nbinom_logsf_array,
        stats_nbinom_quantile, stats_nbinom_quantile_array
    ]
);
distribution!(
    register: register_nbinom_mu,
    rng: stats_nbinom_mu_rng,
    point: k, counts, density: pmf, logpmf,
    params: (size: Real, mu: Real) => dist::NegativeBinomial::with_mean(size.0, mu.0),
    functions: [
        stats_nbinom_mu_pmf, stats_nbinom_mu_pmf_array, stats_nbinom_mu_logpmf, stats_nbinom_mu_logpmf_array,
        stats_nbinom_mu_cdf, stats_nbinom_mu_cdf_array, stats_nbinom_mu_logcdf, stats_nbinom_mu_logcdf_array,
        stats_nbinom_mu_sf, stats_nbinom_mu_sf_array, stats_nbinom_mu_logsf, stats_nbinom_mu_logsf_array,
        stats_nbinom_mu_quantile, stats_nbinom_mu_quantile_array
    ]
);

pub fn register(module: ModuleBuilder) -> ModuleBuilder {
    let module = register_normal(module);
    let module = register_t(module);
    let module = register_chisquare(module);
    let module = register_f(module);
    let module = register_beta(module);
    let module = register_gamma(module);
    let module = register_exponential(module);
    let module = register_lognormal(module);
    let module = register_logistic(module);
    let module = register_binomial(module);
    let module = register_poisson(module);
    let module = register_nbinom(module);
    register_nbinom_mu(module)
}
