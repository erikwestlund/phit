//! Discrete distributions on the nonnegative integers. Probability-mass
//! functions are zero at non-integers; CDFs are step functions, so
//! cdf(k) = P(X ≤ ⌊k⌋) and sf(k) = P(X > ⌊k⌋).
use super::continuous::std_quantile;
use super::quantile;
use super::special::{
    TINY, ln_beta_lower_fraction, ln_binomial_raw, ln_binomial_raw_split, ln_gamma_lower_series,
    ln_gamma_upper_fraction, ln_poisson_raw, log_tail,
};
use super::{finite, positive};
use crate::{ErrorKind, StatsError};
use cdflib::special::{beta_inc, gamma_inc};

/// Largest supported binomial size: integers are exact in a double up to 2^53.
pub const MAX_BINOMIAL_SIZE: u64 = 1 << 53;

fn refine(value: f64, ln_value: impl FnOnce() -> f64) -> f64 {
    if value >= TINY {
        return value;
    }
    let ln_value = ln_value();
    if ln_value.is_nan() {
        value
    } else {
        ln_value.exp()
    }
}

fn is_count(k: f64) -> bool {
    k >= 0.0 && k.is_finite() && k.fract() == 0.0
}

// Cornish–Fisher starting value for the discrete quantile search.
fn start(p: f64, mean: f64, sd: f64, skewness: f64) -> f64 {
    let z = std_quantile(p).clamp(-40.0, 40.0);
    mean + sd * (z + skewness * (z * z - 1.0) / 6.0)
}

/// Binomial distribution: successes in `size` trials with success probability `prob`.
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    n: f64,
    p: f64,
    q: f64,
    ln_p: f64,
    ln_q: f64,
}
impl Binomial {
    pub fn new(size: u64, prob: f64) -> Result<Self, StatsError> {
        let prob = finite("prob", prob)?;
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::new(
                ErrorKind::InvalidInput,
                "prob must be in [0, 1]",
            ));
        }
        if size > MAX_BINOMIAL_SIZE {
            return Err(StatsError::new(
                ErrorKind::Unsupported,
                "size must not exceed 2^53 (exact integer range of a double)",
            ));
        }
        Ok(Self {
            n: size as f64,
            p: prob,
            q: 1.0 - prob,
            ln_p: prob.ln(),
            ln_q: (-prob).ln_1p(),
        })
    }
    fn exact(&self, k: f64) -> Option<(f64, f64)> {
        if k < 0.0 {
            Some((0.0, 1.0))
        } else if k >= self.n || self.p == 0.0 {
            Some((1.0, 0.0))
        } else if self.q == 0.0 {
            Some((0.0, 1.0))
        } else {
            None
        }
    }
    fn tails(&self, k: f64) -> (f64, f64) {
        if let Some(exact) = self.exact(k) {
            return exact;
        }
        let k = k.floor();
        let (lower, upper) = beta_inc(self.n - k, k + 1.0, self.q, self.p);
        (
            refine(lower, || self.ln_lower(k)),
            refine(upper, || self.ln_upper(k)),
        )
    }
    fn ln_lower(&self, k: f64) -> f64 {
        ln_beta_lower_fraction(self.n - k, k + 1.0, self.q, self.ln_q, self.ln_p)
    }
    fn ln_upper(&self, k: f64) -> f64 {
        ln_beta_lower_fraction(k + 1.0, self.n - k, self.p, self.ln_p, self.ln_q)
    }
    pub fn pmf(&self, k: f64) -> f64 {
        self.logpmf(k).exp()
    }
    pub fn logpmf(&self, k: f64) -> f64 {
        if !is_count(k) || k > self.n {
            return f64::NEG_INFINITY;
        }
        ln_binomial_raw(k, self.n, self.p, self.q)
    }
    pub fn cdf(&self, k: f64) -> f64 {
        self.tails(k).0
    }
    pub fn sf(&self, k: f64) -> f64 {
        self.tails(k).1
    }
    pub fn logcdf(&self, k: f64) -> f64 {
        if let Some((lower, _)) = self.exact(k) {
            return lower.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(lower, upper, || self.ln_lower(k.floor()))
    }
    pub fn logsf(&self, k: f64) -> f64 {
        if let Some((_, upper)) = self.exact(k) {
            return upper.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(upper, lower, || self.ln_upper(k.floor()))
    }
    pub fn quantile(&self, p: f64) -> f64 {
        if self.p == 0.0 || self.n == 0.0 {
            return 0.0;
        }
        if self.q == 0.0 {
            return if p == 0.0 { 0.0 } else { self.n };
        }
        let sd = (self.n * self.p * self.q).sqrt();
        let first = start(p, self.n * self.p, sd, (self.q - self.p) / sd);
        quantile::discrete(p, first, self.n, &|k| self.cdf(k), &|k| self.sf(k), &|k| {
            self.logcdf(k)
        })
    }
}

/// Poisson distribution with mean `lambda`.
#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    lambda: f64,
}
impl Poisson {
    pub fn new(lambda: f64) -> Result<Self, StatsError> {
        let lambda = finite("lambda", lambda)?;
        if lambda < 0.0 {
            return Err(StatsError::new(
                ErrorKind::InvalidInput,
                "lambda must be nonnegative",
            ));
        }
        Ok(Self { lambda })
    }
    fn exact(&self, k: f64) -> Option<(f64, f64)> {
        if k < 0.0 {
            Some((0.0, 1.0))
        } else if self.lambda == 0.0 || k.is_infinite() {
            Some((1.0, 0.0))
        } else {
            None
        }
    }
    fn tails(&self, k: f64) -> (f64, f64) {
        if let Some(exact) = self.exact(k) {
            return exact;
        }
        let a = k.floor() + 1.0;
        let (lower_gamma, upper_gamma) = gamma_inc(a, self.lambda);
        (
            refine(upper_gamma, || ln_gamma_upper_fraction(a, self.lambda)),
            refine(lower_gamma, || ln_gamma_lower_series(a, self.lambda)),
        )
    }
    pub fn pmf(&self, k: f64) -> f64 {
        self.logpmf(k).exp()
    }
    pub fn logpmf(&self, k: f64) -> f64 {
        if !is_count(k) {
            return f64::NEG_INFINITY;
        }
        ln_poisson_raw(k, self.lambda)
    }
    pub fn cdf(&self, k: f64) -> f64 {
        self.tails(k).0
    }
    pub fn sf(&self, k: f64) -> f64 {
        self.tails(k).1
    }
    pub fn logcdf(&self, k: f64) -> f64 {
        if let Some((lower, _)) = self.exact(k) {
            return lower.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(lower, upper, || {
            ln_gamma_upper_fraction(k.floor() + 1.0, self.lambda)
        })
    }
    pub fn logsf(&self, k: f64) -> f64 {
        if let Some((_, upper)) = self.exact(k) {
            return upper.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(upper, lower, || {
            ln_gamma_lower_series(k.floor() + 1.0, self.lambda)
        })
    }
    pub fn quantile(&self, p: f64) -> f64 {
        if self.lambda == 0.0 {
            return 0.0;
        }
        let sd = self.lambda.sqrt();
        let first = start(p, self.lambda, sd, 1.0 / sd);
        quantile::discrete(
            p,
            first,
            f64::INFINITY,
            &|k| self.cdf(k),
            &|k| self.sf(k),
            &|k| self.logcdf(k),
        )
    }
}

/// Negative binomial distribution: failures before the `size`-th success,
/// P(X = k) = Γ(k + size)/(Γ(size) k!) prob^size (1 - prob)^k.
/// `with_mean` gives the same distribution with mean mu and Var = mu + mu²/size.
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial {
    size: f64,
    p: f64,
    q: f64,
    ln_p: f64,
    ln_q: f64,
}
impl NegativeBinomial {
    pub fn new(size: f64, prob: f64) -> Result<Self, StatsError> {
        let size = positive("size", size)?;
        let prob = finite("prob", prob)?;
        if !(prob > 0.0 && prob <= 1.0) {
            return Err(StatsError::new(
                ErrorKind::InvalidInput,
                "prob must be in (0, 1]",
            ));
        }
        Ok(Self {
            size,
            p: prob,
            q: 1.0 - prob,
            ln_p: prob.ln(),
            ln_q: (-prob).ln_1p(),
        })
    }
    pub fn with_mean(size: f64, mu: f64) -> Result<Self, StatsError> {
        let size = positive("size", size)?;
        let mu = finite("mu", mu)?;
        if mu < 0.0 {
            return Err(StatsError::new(
                ErrorKind::InvalidInput,
                "mu must be nonnegative",
            ));
        }
        // p = size/(size + mu) and q = mu/(size + mu), each without cancellation.
        Ok(Self {
            size,
            p: size / (size + mu),
            q: mu / (size + mu),
            ln_p: -(mu / size).ln_1p(),
            ln_q: if mu == 0.0 {
                f64::NEG_INFINITY
            } else {
                -(size / mu).ln_1p()
            },
        })
    }
    fn exact(&self, k: f64) -> Option<(f64, f64)> {
        if k < 0.0 {
            Some((0.0, 1.0))
        } else if self.q == 0.0 || k.is_infinite() {
            Some((1.0, 0.0))
        } else {
            None
        }
    }
    fn tails(&self, k: f64) -> (f64, f64) {
        if let Some(exact) = self.exact(k) {
            return exact;
        }
        let k = k.floor();
        let (lower, upper) = beta_inc(self.size, k + 1.0, self.p, self.q);
        (
            refine(lower, || self.ln_lower(k)),
            refine(upper, || self.ln_upper(k)),
        )
    }
    fn ln_lower(&self, k: f64) -> f64 {
        ln_beta_lower_fraction(self.size, k + 1.0, self.p, self.ln_p, self.ln_q)
    }
    fn ln_upper(&self, k: f64) -> f64 {
        ln_beta_lower_fraction(k + 1.0, self.size, self.q, self.ln_q, self.ln_p)
    }
    pub fn pmf(&self, k: f64) -> f64 {
        self.logpmf(k).exp()
    }
    pub fn logpmf(&self, k: f64) -> f64 {
        if !is_count(k) {
            return f64::NEG_INFINITY;
        }
        if self.q == 0.0 {
            return if k == 0.0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if k == 0.0 {
            return self.size * self.ln_p;
        }
        // size/(size + k) times the binomial density of size successes in size + k trials.
        (self.size / (self.size + k)).ln() + ln_binomial_raw_split(self.size, k, self.p, self.q)
    }
    pub fn cdf(&self, k: f64) -> f64 {
        self.tails(k).0
    }
    pub fn sf(&self, k: f64) -> f64 {
        self.tails(k).1
    }
    pub fn logcdf(&self, k: f64) -> f64 {
        if let Some((lower, _)) = self.exact(k) {
            return lower.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(lower, upper, || self.ln_lower(k.floor()))
    }
    pub fn logsf(&self, k: f64) -> f64 {
        if let Some((_, upper)) = self.exact(k) {
            return upper.ln();
        }
        let (lower, upper) = self.tails(k);
        log_tail(upper, lower, || self.ln_upper(k.floor()))
    }
    pub fn quantile(&self, p: f64) -> f64 {
        if self.q == 0.0 {
            return 0.0;
        }
        let mean = self.size * self.q / self.p;
        let sd = (mean / self.p).sqrt();
        let first = start(p, mean, sd, (1.0 + self.q) / (self.size * self.q).sqrt());
        quantile::discrete(
            p,
            first,
            f64::INFINITY,
            &|k| self.cdf(k),
            &|k| self.sf(k),
            &|k| self.logcdf(k),
        )
    }
}
