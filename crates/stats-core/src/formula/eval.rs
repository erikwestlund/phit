//! Evaluate formula variables over all input rows, before missing-value deletion,
//! as R's model.frame does. NaN results are missing (R's na.omit removes them).
use super::parse::{Arg, Expr, Op, deparse, format_number};
use super::{Column, Data};
use crate::{ErrorKind, StatsError};

fn error(message: impl Into<String>) -> StatsError {
    StatsError::new(ErrorKind::InvalidInput, message)
}

#[derive(Clone, Debug)]
pub(crate) enum Value {
    Num(Vec<Option<f64>>),
    /// Numeric matrix as columns, with R column-name suffixes (poly: "1", "2", ...).
    Mat(Vec<Vec<Option<f64>>>, Vec<String>),
    Logical(Vec<Option<bool>>),
    Text(Vec<Option<String>>),
    Factor {
        levels: Vec<String>,
        codes: Vec<Option<usize>>,
        ordered: bool,
    },
    Str(String),
}

impl Value {
    pub(crate) fn missing(&self, i: usize) -> bool {
        match self {
            Value::Num(v) => v[i].is_none(),
            Value::Mat(cols, _) => cols.iter().any(|c| c[i].is_none()),
            Value::Logical(v) => v[i].is_none(),
            Value::Text(v) => v[i].is_none(),
            Value::Factor { codes, .. } => codes[i].is_none(),
            Value::Str(_) => false,
        }
    }
}

fn clean(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

fn numeric(value: Value, what: &str) -> Result<Vec<Option<f64>>, StatsError> {
    match value {
        Value::Num(v) => Ok(v),
        Value::Logical(v) => Ok(v
            .into_iter()
            .map(|b| b.map(|b| if b { 1.0 } else { 0.0 }))
            .collect()),
        _ => Err(error(format!(
            "'{what}' requires numeric or logical values, not a factor or string"
        ))),
    }
}

fn map1(v: Vec<Option<f64>>, f: impl Fn(f64) -> f64) -> Value {
    Value::Num(v.into_iter().map(|x| x.and_then(|x| clean(f(x)))).collect())
}

fn map2(a: &[Option<f64>], b: &[Option<f64>], f: impl Fn(f64, f64) -> f64) -> Vec<Option<f64>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.zip(*y).and_then(|(x, y)| clean(f(x, y))))
        .collect()
}

/// Sorted unique labels in C-locale (byte) order, as R's sort() under LC_COLLATE=C.
pub(crate) fn sorted_levels<'a>(values: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut levels: Vec<String> = values.cloned().collect();
    levels.sort();
    levels.dedup();
    levels
}

fn to_factor(value: Value, ordered: bool) -> Result<Value, StatsError> {
    Ok(match value {
        Value::Num(v) => {
            let mut distinct: Vec<f64> = v.iter().flatten().copied().collect();
            distinct.sort_by(|a, b| a.total_cmp(b));
            distinct.dedup();
            let levels = distinct.iter().map(|x| format_number(*x)).collect();
            let codes = v
                .iter()
                .map(|x| x.map(|x| distinct.partition_point(|d| *d < x)))
                .collect();
            Value::Factor {
                levels,
                codes,
                ordered,
            }
        }
        Value::Logical(v) => {
            let has = |b: bool| v.contains(&Some(b));
            let mut levels = Vec::new();
            if has(false) {
                levels.push("FALSE".to_string());
            }
            if has(true) {
                levels.push("TRUE".to_string());
            }
            let codes = v
                .iter()
                .map(|x| {
                    x.map(|b| {
                        levels
                            .iter()
                            .position(|l| l == if b { "TRUE" } else { "FALSE" })
                            .unwrap_or(0)
                    })
                })
                .collect();
            Value::Factor {
                levels,
                codes,
                ordered,
            }
        }
        Value::Text(v) => {
            let levels = sorted_levels(v.iter().flatten());
            let codes = v
                .iter()
                .map(|x| x.as_ref().and_then(|x| levels.binary_search(x).ok()))
                .collect();
            Value::Factor {
                levels,
                codes,
                ordered,
            }
        }
        Value::Factor { levels, codes, .. } => {
            // factor(f) drops unused levels and keeps the remaining order.
            let (levels, codes) = drop_unused(levels, codes);
            Value::Factor {
                levels,
                codes,
                ordered,
            }
        }
        _ => return Err(error("factor() requires a single column")),
    })
}

pub(crate) fn drop_unused(
    levels: Vec<String>,
    codes: Vec<Option<usize>>,
) -> (Vec<String>, Vec<Option<usize>>) {
    let mut used = vec![false; levels.len()];
    for c in codes.iter().flatten() {
        used[*c] = true;
    }
    let mut map = vec![0; levels.len()];
    let mut kept = Vec::new();
    for (i, level) in levels.into_iter().enumerate() {
        if used[i] {
            map[i] = kept.len();
            kept.push(level);
        }
    }
    let codes = codes.into_iter().map(|c| c.map(|c| map[c])).collect();
    (kept, codes)
}

/// Orthonormal polynomial basis (degree columns, constant removed) of centered x,
/// equal to R's poly()/contr.poly QR construction: each column is the normalized
/// residual of x^k after projection on lower powers.
pub(crate) fn orthonormal_poly(x: &[f64], degree: usize) -> Result<Vec<Vec<f64>>, StatsError> {
    let mut distinct: Vec<f64> = x.to_vec();
    distinct.sort_by(|a, b| a.total_cmp(b));
    distinct.dedup();
    if degree >= distinct.len() {
        return Err(error("'degree' must be less than number of unique points"));
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let centered: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
    for k in 0..=degree {
        let mut v: Vec<f64> = centered.iter().map(|c| c.powi(k as i32)).collect();
        // Two Gram-Schmidt passes keep the basis orthogonal to rounding precision.
        for _ in 0..2 {
            for q in &basis {
                let dot: f64 = q.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(StatsError::new(
                ErrorKind::RankDeficiency,
                "polynomial basis is numerically rank deficient",
            ));
        }
        basis.push(v.into_iter().map(|a| a / norm).collect());
    }
    basis.remove(0);
    Ok(basis)
}

fn arg<'a>(args: &'a [Arg], position: usize, name: &str) -> Option<&'a Expr> {
    if let Some(a) = args.iter().find(|a| a.name.as_deref() == Some(name)) {
        return Some(&a.value);
    }
    args.iter()
        .filter(|a| a.name.is_none())
        .nth(position)
        .map(|a| &a.value)
}

fn constant(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Num(v) => Some(*v),
        Expr::Unary('-', e) => constant(e).map(|v| -v),
        Expr::Paren(e) => constant(e),
        _ => None,
    }
}

fn flag(expr: Option<&Expr>, default: bool, what: &str) -> Result<bool, StatsError> {
    match expr {
        None => Ok(default),
        Some(Expr::Bool(b)) => Ok(*b),
        Some(_) => Err(error(format!("{what} must be TRUE or FALSE"))),
    }
}

pub(crate) fn evaluate(expr: &Expr, data: &Data) -> Result<Value, StatsError> {
    let n = data.n;
    match expr {
        Expr::Num(v) => Ok(Value::Num(vec![Some(*v); n])),
        Expr::Bool(b) => Ok(Value::Logical(vec![Some(*b); n])),
        Expr::Str(s) => Ok(Value::Str(s.clone())),
        Expr::Dot => Err(error("'.' is only valid as a formula term")),
        Expr::Paren(e) => evaluate(e, data),
        Expr::Name(name) => {
            let i = data
                .names
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| error(format!("unknown column '{name}' in formula")))?;
            Ok(match &data.columns[i] {
                Column::Numeric(v) => Value::Num(v.clone()),
                Column::Logical(v) => Value::Logical(v.clone()),
                Column::Text(v) => Value::Text(v.clone()),
                Column::Factor {
                    levels,
                    codes,
                    ordered,
                } => Value::Factor {
                    levels: levels.clone(),
                    codes: codes.clone(),
                    ordered: *ordered,
                },
                Column::Invalid { message, .. } => {
                    return Err(error(format!("column '{name}': {message}")));
                }
            })
        }
        Expr::Unary(op, e) => {
            let v = evaluate(e, data)?;
            match op {
                '-' => Ok(map1(numeric(v, &deparse(expr))?, |x| -x)),
                '+' => Ok(Value::Num(numeric(v, &deparse(expr))?)),
                '!' => match v {
                    Value::Logical(v) => Ok(Value::Logical(
                        v.into_iter().map(|b| b.map(|b| !b)).collect(),
                    )),
                    Value::Num(v) => Ok(Value::Logical(
                        v.into_iter().map(|x| x.map(|x| x == 0.0)).collect(),
                    )),
                    _ => Err(error("'!' requires logical values")),
                },
                _ => Err(error(format!("cannot evaluate '{}'", deparse(expr)))),
            }
        }
        Expr::Binary(op, l, r) => binary(*op, evaluate(l, data)?, evaluate(r, data)?, expr),
        Expr::Call(name, args) => call(name, args, data, expr),
    }
}

fn labels_of(value: &Value) -> Option<Vec<Option<String>>> {
    match value {
        Value::Text(v) => Some(v.clone()),
        Value::Factor { levels, codes, .. } => {
            Some(codes.iter().map(|c| c.map(|c| levels[c].clone())).collect())
        }
        _ => None,
    }
}

fn binary(op: Op, left: Value, right: Value, expr: &Expr) -> Result<Value, StatsError> {
    let text = deparse(expr);
    if matches!(op, Op::Eq | Op::Ne) {
        let pair = match (&left, &right) {
            (Value::Str(s), other) | (other, Value::Str(s)) => Some(
                labels_of(other)
                    .map(|labels| (labels, s.clone()))
                    .ok_or_else(|| {
                        error(format!("'{text}' compares a string with a non-string"))
                    })?,
            ),
            _ => None,
        };
        if let Some((labels, s)) = pair {
            let eq = op == Op::Eq;
            return Ok(Value::Logical(
                labels
                    .into_iter()
                    .map(|v| v.map(|v| (v == s) == eq))
                    .collect(),
            ));
        }
    }
    if matches!(op, Op::And | Op::Or) {
        let a = numeric(left, &text)?;
        let b = numeric(right, &text)?;
        return Ok(Value::Logical(
            a.iter()
                .zip(&b)
                .map(|(x, y)| {
                    let x = x.map(|x| x != 0.0);
                    let y = y.map(|y| y != 0.0);
                    match op {
                        Op::And => match (x, y) {
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            (Some(true), Some(true)) => Some(true),
                            _ => None,
                        },
                        _ => match (x, y) {
                            (Some(true), _) | (_, Some(true)) => Some(true),
                            (Some(false), Some(false)) => Some(false),
                            _ => None,
                        },
                    }
                })
                .collect(),
        ));
    }
    let a = numeric(left, &text)?;
    let b = numeric(right, &text)?;
    let compare = |f: fn(f64, f64) -> bool| {
        Value::Logical(
            a.iter()
                .zip(&b)
                .map(|(x, y)| x.zip(*y).map(|(x, y)| f(x, y)))
                .collect(),
        )
    };
    Ok(match op {
        Op::Add => Value::Num(map2(&a, &b, |x, y| x + y)),
        Op::Sub => Value::Num(map2(&a, &b, |x, y| x - y)),
        Op::Mul => Value::Num(map2(&a, &b, |x, y| x * y)),
        Op::Div => Value::Num(map2(&a, &b, |x, y| x / y)),
        Op::Pow => Value::Num(map2(&a, &b, |x, y| {
            // R: 1^y and x^0 are 1 even for NaN/Inf operands.
            if x == 1.0 || y == 0.0 { 1.0 } else { x.powf(y) }
        })),
        Op::Lt => compare(|x, y| x < y),
        Op::Gt => compare(|x, y| x > y),
        Op::Le => compare(|x, y| x <= y),
        Op::Ge => compare(|x, y| x >= y),
        Op::Eq => compare(|x, y| x == y),
        Op::Ne => compare(|x, y| x != y),
        _ => {
            return Err(error(format!(
                "operator '{text}' is not supported inside I() or a transform"
            )));
        }
    })
}

fn call(name: &str, args: &[Arg], data: &Data, expr: &Expr) -> Result<Value, StatsError> {
    let text = deparse(expr);
    let first = || arg(args, 0, "x").ok_or_else(|| error(format!("{name}() needs an argument")));
    let unary = |f: fn(f64) -> f64| -> Result<Value, StatsError> {
        if args.len() != 1 {
            return Err(error(format!("{name}() takes one argument")));
        }
        Ok(map1(numeric(evaluate(first()?, data)?, &text)?, f))
    };
    match name {
        "I" | "offset" => {
            if args.len() != 1 {
                return Err(error(format!("{name}() takes one argument")));
            }
            let v = evaluate(first()?, data)?;
            if name == "offset" {
                return Ok(Value::Num(numeric(v, &text)?));
            }
            Ok(v)
        }
        "log" => {
            let x = numeric(evaluate(first()?, data)?, &text)?;
            match arg(args, 1, "base") {
                None if args.len() == 1 => Ok(map1(x, f64::ln)),
                Some(base) if args.len() == 2 => {
                    let base =
                        constant(base).ok_or_else(|| error("log() base must be a number"))?;
                    Ok(map1(x, move |v| v.ln() / base.ln()))
                }
                _ => Err(error("log() takes x and an optional base")),
            }
        }
        "log2" => unary(f64::log2),
        "log10" => unary(f64::log10),
        "log1p" => unary(f64::ln_1p),
        "exp" => unary(f64::exp),
        "expm1" => unary(f64::exp_m1),
        "sqrt" => unary(f64::sqrt),
        "abs" => unary(f64::abs),
        "sin" => unary(f64::sin),
        "cos" => unary(f64::cos),
        "tan" => unary(f64::tan),
        "as.numeric" | "as.double" => {
            if args.len() != 1 {
                return Err(error(format!("{name}() takes one argument")));
            }
            match evaluate(first()?, data)? {
                Value::Factor { codes, .. } => Ok(Value::Num(
                    codes
                        .into_iter()
                        .map(|c| c.map(|c| c as f64 + 1.0))
                        .collect(),
                )),
                v => Ok(Value::Num(numeric(v, &text)?)),
            }
        }
        "factor" | "as.factor" | "ordered" => {
            if args.len() != 1 {
                return Err(error(format!(
                    "{name}() supports one argument; set levels in a factor descriptor"
                )));
            }
            to_factor(evaluate(first()?, data)?, name == "ordered")
        }
        "poly" => {
            let x = numeric(evaluate(first()?, data)?, &text)?;
            let degree_expr = args
                .iter()
                .find(|a| a.name.as_deref() == Some("degree"))
                .or_else(|| args.iter().filter(|a| a.name.is_none()).nth(1))
                .map(|a| &a.value);
            let degree = match degree_expr {
                None => 1.0,
                Some(e) => constant(e).ok_or_else(|| error("poly() degree must be a number"))?,
            };
            if degree < 1.0 || degree.fract() != 0.0 || degree > 25.0 {
                return Err(error("poly() degree must be an integer from 1 to 25"));
            }
            let degree = degree as usize;
            for a in args {
                if let Some(n) = &a.name
                    && !["x", "degree", "raw"].contains(&n.as_str())
                {
                    return Err(error(format!("unsupported poly() argument '{n}'")));
                }
            }
            if args.iter().filter(|a| a.name.is_none()).count() > 2 {
                return Err(error("poly() of several variables is not supported"));
            }
            let raw = flag(
                args.iter()
                    .find(|a| a.name.as_deref() == Some("raw"))
                    .map(|a| &a.value),
                false,
                "poly() raw",
            )?;
            let names = (1..=degree).map(|k| k.to_string()).collect();
            if raw {
                let cols = (1..=degree)
                    .map(|k| x.iter().map(|v| v.map(|v| v.powi(k as i32))).collect())
                    .collect();
                return Ok(Value::Mat(cols, names));
            }
            if x.iter().any(Option::is_none) {
                return Err(error(
                    "missing values are not allowed in orthogonal poly(); use raw = TRUE or remove them",
                ));
            }
            let x: Vec<f64> = x.into_iter().flatten().collect();
            let cols = orthonormal_poly(&x, degree)?
                .into_iter()
                .map(|c| c.into_iter().map(Some).collect())
                .collect();
            Ok(Value::Mat(cols, names))
        }
        "cbind" => {
            let mut cols = Vec::new();
            let mut names = Vec::new();
            for a in args {
                cols.push(numeric(evaluate(&a.value, data)?, &text)?);
                names.push(a.name.clone().unwrap_or_else(|| deparse(&a.value)));
            }
            if cols.is_empty() {
                return Err(error("cbind() needs arguments"));
            }
            Ok(Value::Mat(cols, names))
        }
        _ => Err(StatsError::new(
            ErrorKind::Unsupported,
            format!("unsupported function '{name}()' in formula"),
        )),
    }
}
