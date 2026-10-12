//! Numerical pieces cdflib does not provide: log-scale tails beyond the double
//! range, Loader's saddle-point densities and logistic-scale helpers.
use cdflib::special::{beta_log, gamma_log};
use std::f64::consts::PI;

/// Probabilities below this are recomputed on the log scale instead of taking
/// ln(p), so log tails stay accurate after p itself underflows.
pub(super) const TINY: f64 = 1e-280;
const MAX_TERMS: usize = 100_000;
const FPMIN: f64 = 1e-300;
pub(super) const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// ln(lower) from independently computed complementary probabilities.
/// `extreme` supplies ln(lower) once `lower` is too small for ln(lower);
/// it returns NaN outside its verified regime, which callers report as an error.
pub(super) fn log_tail(lower: f64, upper: f64, extreme: impl FnOnce() -> f64) -> f64 {
    if lower > 0.5 {
        (-upper).ln_1p()
    } else if lower >= TINY {
        lower.ln()
    } else {
        extreme()
    }
}

/// ln Φ(z) for z < -8 from the continued fraction for the Mills ratio.
/// Also used by the probit estimator for the same extreme lower tail.
pub(crate) fn normal_log_lower_tail(z: f64) -> f64 {
    let x = -z;
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let mut denominator = x;
    for k in (1..=100).rev() {
        denominator = x + k as f64 / denominator;
    }
    -0.5 * x * x - LN_SQRT_2PI - denominator.ln()
}

/// ln P(a, x) by the power series; valid for x < a + 1.
pub(super) fn ln_gamma_lower_series(a: f64, x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= a + 1.0 {
        return f64::NAN;
    }
    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..MAX_TERMS {
        term *= x / (a + n as f64);
        sum += term;
        if term < sum * f64::EPSILON {
            return a * x.ln() - x - gamma_log(a) + sum.ln();
        }
    }
    f64::NAN
}

/// ln Q(a, x) by Lentz's continued fraction; valid for x > a + 1.
pub(super) fn ln_gamma_upper_fraction(a: f64, x: f64) -> f64 {
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if x <= a + 1.0 {
        return f64::NAN;
    }
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..MAX_TERMS {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            return -x + a * x.ln() - gamma_log(a) + h.ln();
        }
    }
    f64::NAN
}

/// ln I_x(a, b) by the continued fraction; valid for x < (a + 1)/(a + b + 2).
/// Callers pass ln x and ln(1 - x) computed without cancellation.
pub(super) fn ln_beta_lower_fraction(a: f64, b: f64, x: f64, ln_x: f64, ln_y: f64) -> f64 {
    if ln_x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x >= (a + 1.0) / (a + b + 2.0) {
        return f64::NAN;
    }
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_TERMS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            return a * ln_x + b * ln_y - beta_log(a, b) - a.ln() + h.ln();
        }
    }
    f64::NAN
}

// Stirling-series error ln Γ(n + 1) - (n + 1/2) ln n + n - ln √(2π) at n = k/2,
// k = 1..30, computed with mpmath at 50 digits (docs/distributions.qmd).
#[allow(clippy::excessive_precision)] // Preserve the generated reference decimals; f64 rounds them.
const STIRLING_HALVES: [f64; 30] = [
    1.534_264_097_200_273_5e-1,
    8.106_146_679_532_726e-2,
    5.481_412_105_191_765_4e-2,
    4.134_069_595_540_929_4e-2,
    3.316_287_351_993_628_7e-2,
    2.767_792_568_499_834e-2,
    2.374_616_365_629_749_6e-2,
    2.079_067_210_376_509_3e-2,
    1.848_845_053_267_318_5e-2,
    1.664_469_118_982_119_2e-2,
    1.513_497_322_191_737_9e-2,
    1.387_612_882_307_074_8e-2,
    1.281_046_524_292_022_7e-2,
    1.189_670_994_589_177e-2,
    1.110_455_975_820_691_7e-2,
    1.041_126_526_197_209_6e-2,
    9.799_416_126_158_803e-3,
    9.255_462_182_712_733e-3,
    8.768_700_134_139_385e-3,
    8.330_563_433_362_871e-3,
    7.934_114_564_314_02e-3,
    7.573_675_487_951_841e-3,
    7.244_554_301_320_383e-3,
    6.942_840_107_209_53e-3,
    6.665_247_032_707_682e-3,
    6.408_994_188_004_207e-3,
    6.171_712_263_039_458e-3,
    5.951_370_112_758_848e-3,
    5.746_216_513_010_116e-3,
    5.554_733_551_962_801e-3,
];

/// Stirling-series error δ(n) = ln Γ(n + 1) - (n + 1/2) ln n + n - ln √(2π).
pub(super) fn stirling_error(n: f64) -> f64 {
    const S0: f64 = 1.0 / 12.0;
    const S1: f64 = 1.0 / 360.0;
    const S2: f64 = 1.0 / 1260.0;
    const S3: f64 = 1.0 / 1680.0;
    const S4: f64 = 1.0 / 1188.0;
    if n <= 15.0 {
        let twice = n + n;
        if twice == twice.floor() && twice >= 1.0 {
            return STIRLING_HALVES[twice as usize - 1];
        }
        return gamma_log(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI;
    }
    let nn = n * n;
    if n > 500.0 {
        (S0 - S1 / nn) / n
    } else if n > 80.0 {
        (S0 - (S1 - S2 / nn) / nn) / n
    } else if n > 35.0 {
        (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    } else {
        (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
    }
}

/// Deviance term x ln(x / np) + np - x without cancellation when x ≈ np.
pub(super) fn deviance_term(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let mut v = (x - np) / (x + np);
        let s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        v *= v;
        let mut s = s;
        for j in 1..1000 {
            ej *= v;
            let next = s + ej / (2 * j + 1) as f64;
            if next == s {
                return next;
            }
            s = next;
        }
    }
    x * (x / np).ln() + np - x
}

/// ln of λ^x e^-λ / Γ(x + 1) for real x ≥ 0 (Loader's saddle-point form).
pub(super) fn ln_poisson_raw(x: f64, lambda: f64) -> f64 {
    if lambda == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if !lambda.is_finite() || x < 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 {
        return -lambda;
    }
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }
    -stirling_error(x) - deviance_term(x, lambda) - 0.5 * (2.0 * PI * x).ln()
}

/// ln of Γ(n + 1)/(Γ(x + 1)Γ(n - x + 1)) p^x q^(n-x) for real 0 ≤ x ≤ n,
/// with q = 1 - p supplied separately (Loader's saddle-point form).
pub(super) fn ln_binomial_raw(x: f64, n: f64, p: f64, q: f64) -> f64 {
    if x < 0.0 || x > n {
        return f64::NEG_INFINITY;
    }
    ln_binomial_raw_split(x, n - x, p, q)
}

/// As `ln_binomial_raw` with n = x + m, for callers that know m = n - x exactly.
pub(super) fn ln_binomial_raw_split(x: f64, m: f64, p: f64, q: f64) -> f64 {
    let n = x + m;
    if p == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if q == 0.0 {
        return if m == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x == 0.0 {
        if n == 0.0 {
            return 0.0;
        }
        return if p < 0.1 {
            -deviance_term(n, n * q) - n * p
        } else {
            n * q.ln()
        };
    }
    if m == 0.0 {
        return if q < 0.1 {
            -deviance_term(n, n * p) - n * q
        } else {
            n * p.ln()
        };
    }
    let lc = stirling_error(n)
        - stirling_error(x)
        - stirling_error(m)
        - deviance_term(x, n * p)
        - deviance_term(m, n * q);
    let lf = (2.0 * PI).ln() + x.ln() + (m / n).ln();
    lc - 0.5 * lf
}

/// ln(1 + e^t) without overflow.
pub(super) fn softplus(t: f64) -> f64 {
    t.max(0.0) + (-t.abs()).exp().ln_1p()
}

/// ln(1 - e^-t) for t ≥ 0 (Mächler's log1mexp).
pub(super) fn ln_one_minus_exp_neg(t: f64) -> f64 {
    if t <= std::f64::consts::LN_2 {
        (-(-t).exp_m1()).ln()
    } else {
        (-(-t).exp()).ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stirling_error_matches_log_gamma_definition() {
        for n in [0.25, 3.7, 12.0, 15.5, 20.0, 40.0, 100.0, 1000.0] {
            let direct = statrs::function::gamma::ln_gamma(n + 1.0) - (n + 0.5) * f64::ln(n) + n
                - LN_SQRT_2PI;
            assert!((stirling_error(n) - direct).abs() < 1e-12, "{n}");
        }
    }

    #[test]
    fn deviance_term_is_nonnegative_and_stable() {
        assert_eq!(deviance_term(5.0, 5.0), 0.0);
        assert!(deviance_term(1e6, 1e6 + 1.0) > 0.0);
        let direct = 7.0 * (7.0f64 / 3.0).ln() + 3.0 - 7.0;
        assert!((deviance_term(7.0, 3.0) - direct).abs() < 1e-14);
    }
}
