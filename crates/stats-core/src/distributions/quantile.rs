//! Quantile searches shared by the distributions.
//!
//! Continuous quantiles solve ln F(x) = ln p (or ln S(x) = ln(1 - p) in the
//! upper half, where 1 - p is exact) by Newton steps safeguarded by a bracket.
//! Working on the log scale keeps p down to the smallest doubles accurate.
use super::special::TINY;

/// Monotone log-tail functions of one distribution.
pub(super) struct LogTails<'a> {
    pub log_cdf: &'a dyn Fn(f64) -> f64,
    pub log_sf: &'a dyn Fn(f64) -> f64,
    pub log_pdf: &'a dyn Fn(f64) -> f64,
}

/// Solve F(x) = p on the support (lower, upper), starting near `start`.
/// p must be in [0, 1]; the endpoints map to the support bounds.
pub(super) fn continuous(p: f64, start: f64, lower: f64, upper: f64, f: &LogTails) -> f64 {
    if p == 0.0 {
        return lower;
    }
    if p == 1.0 {
        return upper;
    }
    let use_lower = p <= 0.5;
    let target = if use_lower { p.ln() } else { (1.0 - p).ln() };
    // g increases with x and is zero at the quantile.
    let g = |x: f64| {
        if use_lower {
            (f.log_cdf)(x) - target
        } else {
            target - (f.log_sf)(x)
        }
    };
    let slope = |x: f64| {
        let log_tail = if use_lower {
            (f.log_cdf)(x)
        } else {
            (f.log_sf)(x)
        };
        ((f.log_pdf)(x) - log_tail).exp()
    };

    let mut x = if start.is_finite() && start > lower && start < upper {
        start
    } else if lower.is_finite() && upper.is_finite() {
        0.5 * (lower + upper)
    } else if lower.is_finite() {
        lower + 1.0
    } else if upper.is_finite() {
        upper - 1.0
    } else {
        0.0
    };
    let mut gx = g(x);
    if gx == 0.0 {
        return x;
    }
    // A failed tail evaluation (NaN) must surface as a failure, not steer the search.
    if gx.is_nan() {
        return f64::NAN;
    }
    // Bracket [a, b] with g(a) < 0 < g(b).
    let (mut a, mut b) = (lower, upper);
    if gx < 0.0 {
        a = x;
        if upper.is_infinite() {
            let mut next = x;
            loop {
                next = expand(next, 1.0);
                if next == f64::INFINITY {
                    return upper;
                }
                let gn = g(next);
                if gn.is_nan() {
                    return f64::NAN;
                }
                if gn >= 0.0 {
                    b = next;
                    break;
                }
                a = next;
            }
        }
    } else {
        b = x;
        if lower.is_infinite() {
            let mut next = x;
            loop {
                next = expand(next, -1.0);
                if next == f64::NEG_INFINITY {
                    return lower;
                }
                let gn = g(next);
                if gn.is_nan() {
                    return f64::NAN;
                }
                if gn <= 0.0 {
                    a = next;
                    break;
                }
                b = next;
            }
        } else if lower == 0.0 && g(f64::MIN_POSITIVE) > 0.0 {
            // The quantile underflows below the smallest normal double.
            return 0.0;
        }
    }

    for _ in 0..400 {
        let step = gx / slope(x);
        let mut next = x - step;
        if !(next.is_finite() && next > a && next < b) {
            next = midpoint(a, b);
        }
        if next == x || (next - x).abs() <= 4.0 * f64::EPSILON * next.abs() {
            return next;
        }
        x = next;
        gx = g(x);
        if gx == 0.0 {
            return x;
        }
        if gx.is_nan() {
            return f64::NAN;
        }
        if gx < 0.0 {
            a = x;
        } else {
            b = x;
        }
        if b - a <= 4.0 * f64::EPSILON * a.abs().max(b.abs()) {
            return 0.5 * (a + b);
        }
    }
    f64::NAN
}

// Grow outward quickly: additive near zero, then squaring the magnitude, and
// finally the largest double before concluding the quantile is infinite.
fn expand(x: f64, direction: f64) -> f64 {
    if x * direction <= 2.0 {
        x + 2.0 * direction
    } else if x.abs() == f64::MAX {
        direction * f64::INFINITY
    } else {
        let next = x * x.abs();
        if next.is_finite() {
            next
        } else {
            direction * f64::MAX
        }
    }
}

// Geometric bisection across many orders of magnitude, arithmetic otherwise.
fn midpoint(a: f64, b: f64) -> f64 {
    if a >= 0.0 && b > 4.0 * a.max(f64::MIN_POSITIVE) {
        a.max(f64::MIN_POSITIVE).sqrt() * b.sqrt()
    } else if b <= 0.0 && a < 4.0 * b.min(-f64::MIN_POSITIVE) {
        -(a.abs().sqrt() * b.abs().max(f64::MIN_POSITIVE).sqrt())
    } else {
        a + 0.5 * (b - a)
    }
}

/// Smallest integer k in [0, max] with F(k) ≥ p, starting the search at `start`.
/// As in R, p is reduced by 64 machine epsilons (relative) so that rounding in
/// F(k) does not move the result past a probability that F reaches exactly.
pub(super) fn discrete(
    p: f64,
    start: f64,
    max: f64,
    cdf: &dyn Fn(f64) -> f64,
    sf: &dyn Fn(f64) -> f64,
    log_cdf: &dyn Fn(f64) -> f64,
) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return max;
    }
    const FUZZ: f64 = 64.0 * f64::EPSILON;
    // A NaN tail compares as "not reached"; remember it and fail at the end.
    let failed = std::cell::Cell::new(false);
    let reached = |k: f64| {
        let (value, hit) = if p > 0.5 {
            let v = sf(k);
            (v, v <= (1.0 - p) * (1.0 + FUZZ))
        } else if p < TINY {
            let v = log_cdf(k);
            (v, v >= p.ln() + (-FUZZ).ln_1p())
        } else {
            let v = cdf(k);
            (v, v >= p * (1.0 - FUZZ))
        };
        if value.is_nan() {
            failed.set(true);
        }
        hit
    };
    let limit = if max.is_finite() {
        max
    } else {
        9.007_199_254_740_992e15
    };
    let mut k = if start.is_finite() {
        start.floor().clamp(0.0, limit)
    } else {
        0.0
    };
    // Find lo (not reached, or -1) and hi (reached) around the start.
    let (mut lo, mut hi);
    if reached(k) {
        hi = k;
        let mut step = 1.0;
        loop {
            let candidate = hi - step;
            if candidate < 0.0 {
                lo = -1.0;
                break;
            }
            if reached(candidate) {
                hi = candidate;
                step *= 2.0;
            } else {
                lo = candidate;
                break;
            }
        }
    } else {
        lo = k;
        let mut step = 1.0;
        loop {
            let candidate = (lo + step).min(limit);
            if reached(candidate) {
                hi = candidate;
                break;
            }
            if candidate >= limit {
                if failed.get() {
                    return f64::NAN;
                }
                return if max.is_finite() { max } else { f64::INFINITY };
            }
            lo = candidate;
            step *= 2.0;
        }
    }
    while hi - lo > 1.0 {
        k = (lo + 0.5 * (hi - lo)).floor();
        if reached(k) {
            hi = k;
        } else {
            lo = k;
        }
    }
    if failed.get() { f64::NAN } else { hi }
}
