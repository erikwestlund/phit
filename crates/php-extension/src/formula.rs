//! Formula interfaces: stats_model_matrix, stats_lm, stats_glm and stats_lmer.
//!
//! Data arrive as named columns (or an object with toModelData()/toColumns()).
//! The R-compatible formula engine lives in stats_core::formula; this module only
//! converts PHP values, calls the same estimation cores as the matrix functions
//! and adds coefficient names to their unchanged result contracts.
use crate::{finish_result, input_exception, stats_exception};
use ext_php_rs::{
    boxed::ZBox,
    prelude::*,
    types::{ZendHashTable, Zval},
    zend::ExecutorGlobals,
};
use stats_core::ErrorKind;
use stats_core::formula::{Column, Data, DesignOptions, GroupLabel, ModelDesign, Response};

fn invalid(message: &str) -> PhpException {
    input_exception(ErrorKind::InvalidInput, message)
}

fn factor_descriptor(name: &str, table: &ZendHashTable) -> PhpResult<Column> {
    let malformed = |detail: &str| {
        invalid(&format!(
            "column '{name}' is not a valid factor descriptor (exactly levels: list<string>, codes: list<?int>, ordered: bool): {detail}"
        ))
    };
    if table.len() != 3 {
        return Err(malformed("unexpected keys"));
    }
    let levels = table
        .get("levels")
        .and_then(|v| v.array())
        .ok_or_else(|| malformed("levels must be an array"))?
        .values()
        .map(|v| v.str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| malformed("levels must be strings"))?;
    let codes = table
        .get("codes")
        .and_then(|v| v.array())
        .ok_or_else(|| malformed("codes must be an array"))?
        .values()
        .map(|v| {
            if v.is_null() {
                Ok(None)
            } else {
                match v.long() {
                    Some(c) if c >= 0 && (c as usize) < levels.len() => Ok(Some(c as usize)),
                    Some(_) => Err(malformed("codes must be 0-based indexes into levels")),
                    None => Err(malformed("codes must be integers or null")),
                }
            }
        })
        .collect::<PhpResult<Vec<_>>>()?;
    let ordered = table
        .get("ordered")
        .and_then(|v| v.bool())
        .ok_or_else(|| malformed("ordered must be a boolean"))?;
    Ok(Column::Factor {
        levels,
        codes,
        ordered,
    })
}

fn column(name: &str, value: &Zval) -> PhpResult<Column> {
    let table = value.array().ok_or_else(|| {
        invalid(&format!(
            "column '{name}' must be an array of values or a factor descriptor"
        ))
    })?;
    if !table.has_numerical_keys() {
        return factor_descriptor(name, table);
    }
    let (mut numbers, mut bools, mut strings, mut nested) = (false, false, false, false);
    for v in table.values() {
        if v.is_null() {
        } else if v.is_long() || v.is_double() {
            if v.double().is_some_and(f64::is_infinite) {
                return Err(input_exception(
                    ErrorKind::Nonfinite,
                    &format!("column '{name}' contains an infinite value"),
                ));
            }
            numbers = true;
        } else if v.is_bool() {
            bools = true;
        } else if v.is_string() {
            strings = true;
        } else {
            nested = true;
        }
    }
    let len = table.len();
    if nested {
        return Ok(Column::Invalid {
            len,
            message: "list-columns (arrays or objects as values) cannot be used in a model".into(),
        });
    }
    if u8::from(numbers) + u8::from(bools) + u8::from(strings) > 1 {
        return Ok(Column::Invalid {
            len,
            message: "values mix numbers, booleans and strings".into(),
        });
    }
    Ok(if strings {
        Column::Text(table.values().map(|v| v.str().map(str::to_owned)).collect())
    } else if bools {
        Column::Logical(
            table
                .values()
                .map(|v| v.bool().filter(|_| !v.is_null()))
                .collect(),
        )
    } else {
        Column::Numeric(
            table
                .values()
                .map(|v| {
                    v.double()
                        .or_else(|| v.long().map(|l| l as f64))
                        .filter(|x| !x.is_nan())
                })
                .collect(),
        )
    })
}

fn columns_from(table: &ZendHashTable) -> PhpResult<Data> {
    let mut columns = Vec::with_capacity(table.len());
    for (key, value) in table.iter() {
        let name = key.to_string();
        if !matches!(
            key,
            ext_php_rs::types::ArrayKey::String(_) | ext_php_rs::types::ArrayKey::Str(_)
        ) {
            return Err(invalid(
                "data must be an associative array of columns (name => list of values), not a list of rows",
            ));
        }
        columns.push((name.clone(), column(&name, value)?));
    }
    Data::new(columns).map_err(stats_exception)
}

fn model_data(data: &Zval) -> PhpResult<Data> {
    if let Some(table) = data.array() {
        return columns_from(table);
    }
    if let Some(object) = data.object() {
        for method in ["toModelData", "toColumns"] {
            match object.try_call_method(method, vec![]) {
                Ok(result) => {
                    ExecutorGlobals::throw_if_exception()?;
                    let table = result.array().ok_or_else(|| {
                        invalid(&format!("{method}() must return an array of columns"))
                    })?;
                    return columns_from(table);
                }
                Err(ext_php_rs::error::Error::Callable) => continue,
                Err(e) => return Err(PhpException::from(e.to_string())),
            }
        }
    }
    Err(invalid(
        "data must be an array of columns or an object with toModelData() or toColumns()",
    ))
}

fn missing_policy(missing: &str) -> PhpResult<bool> {
    match missing {
        "drop" => Ok(true),
        "raise" => Ok(false),
        _ => Err(invalid("missing must be 'drop' or 'raise'")),
    }
}

fn weights_vector(weights: Option<&Zval>, data: &Data) -> PhpResult<Option<Vec<Option<f64>>>> {
    let Some(weights) = weights.filter(|w| !w.is_null()) else {
        return Ok(None);
    };
    let values: Vec<Option<f64>> = if let Some(name) = weights.str() {
        match data.column(name) {
            Some(Column::Numeric(v)) => v.clone(),
            Some(_) => return Err(invalid(&format!("weights column '{name}' must be numeric"))),
            None => return Err(invalid(&format!("unknown weights column '{name}'"))),
        }
    } else if let Some(table) = weights.array() {
        table
            .values()
            .map(crate::number)
            .collect::<PhpResult<Vec<_>>>()?
    } else {
        return Err(invalid("weights must be null, an array or a column name"));
    };
    if values.len() != data.rows() {
        return Err(input_exception(
            ErrorKind::DimensionMismatch,
            "optional vectors must have one value per original observation",
        ));
    }
    if values.iter().flatten().any(|w| *w <= 0.0) {
        return Err(invalid("weights must be positive"));
    }
    Ok(Some(values))
}

struct Prepared {
    design: ModelDesign,
    weights: Vec<f64>,
}

fn prepare(
    formula: &str,
    data: &Zval,
    missing: &str,
    weights: Option<&Zval>,
    drop_unused_levels: bool,
) -> PhpResult<Prepared> {
    let drop_missing = missing_policy(missing)?;
    let frame = model_data(data)?;
    let weights = weights_vector(weights, &frame)?;
    let options = DesignOptions {
        drop_missing,
        drop_unused_levels,
        extra: weights.into_iter().collect(),
    };
    let design =
        stats_core::formula::model_design(formula, &frame, &options).map_err(stats_exception)?;
    let weights = design
        .extra
        .first()
        .cloned()
        .unwrap_or_else(|| vec![1.0; design.x.len()]);
    Ok(Prepared { design, weights })
}

fn group_values(groups: &[GroupLabel]) -> PhpResult<ZBox<ZendHashTable>> {
    let mut out = ZendHashTable::new();
    for g in groups {
        match g {
            GroupLabel::Integer(v) => out.push(*v)?,
            GroupLabel::Text(v) => out.push(v.clone())?,
        }
    }
    Ok(out)
}

fn rows(design: &ModelDesign) -> (Vec<i64>, Vec<i64>) {
    (
        design.retained_rows.iter().map(|v| *v as i64).collect(),
        design.dropped_rows.iter().map(|v| *v as i64).collect(),
    )
}

fn random_table(design: &ModelDesign) -> PhpResult<ZBox<ZendHashTable>> {
    let mut random = ZendHashTable::new();
    for term in &design.random {
        let mut t = ZendHashTable::new();
        t.insert("label", term.label.clone())?;
        t.insert("group", term.group.clone())?;
        t.insert("columns", term.columns.clone())?;
        t.insert("correlated", term.correlated)?;
        t.insert("z", term.z.clone())?;
        t.insert("groups", group_values(&term.groups)?)?;
        random.push(t)?;
    }
    Ok(random)
}

/// Shared naming keys added to every formula fit.
fn add_names(output: &mut ZendHashTable, design: &ModelDesign) -> PhpResult<()> {
    output.insert("formula", design.formula.clone())?;
    output.insert("response", design.response_name.clone())?;
    output.insert("columns", design.columns.clone())?;
    output.insert("terms", design.terms.clone())?;
    output.insert(
        "assign",
        design.assign.iter().map(|v| *v as i64).collect::<Vec<_>>(),
    )?;
    output.insert("intercept", design.intercept)?;
    let mut named = ZendHashTable::new();
    if let Some(coefficients) = output.get("coefficients").and_then(|v| v.array()) {
        for (name, value) in design.columns.iter().zip(coefficients.values()) {
            named.insert(name.as_str(), value.shallow_clone())?;
        }
    }
    output.insert("named", named)?;
    Ok(())
}

fn reject_random(design: &ModelDesign, function: &str) -> PhpResult<()> {
    if design.random.is_empty() {
        Ok(())
    } else {
        Err(input_exception(
            ErrorKind::Unsupported,
            &format!(
                "{function} does not fit random-effect terms such as ({}); use stats_lmer, or stats_lmm_fit with stats_model_matrix()['random']",
                design.random[0].label
            ),
        ))
    }
}

fn numeric_response(design: &ModelDesign, function: &str) -> PhpResult<Vec<f64>> {
    match &design.response {
        Some(Response::Numeric(v)) => Ok(v.clone()),
        None => Err(invalid(&format!("{function} needs a response: 'y ~ ...'"))),
        Some(_) => Err(invalid(&format!(
            "{function} needs a numeric or logical response"
        ))),
    }
}

/// Build R's model.matrix() for a formula over named columns.
#[php_function]
#[php(defaults(missing = "\"drop\""))]
pub fn stats_model_matrix(
    formula: &str,
    data: &Zval,
    missing: &str,
) -> PhpResult<ZBox<ZendHashTable>> {
    let prepared = prepare(formula, data, missing, None, false)?;
    let design = prepared.design;
    let mut out = ZendHashTable::new();
    match &design.response {
        None => out.insert("y", Option::<Vec<f64>>::None)?,
        Some(Response::Numeric(v)) => out.insert("y", v.clone())?,
        Some(Response::Factor { levels, codes }) => {
            out.insert("y", codes.iter().map(|c| *c as f64).collect::<Vec<_>>())?;
            out.insert("response_levels", levels.clone())?;
        }
        Some(Response::Binomial {
            successes,
            failures,
        }) => out.insert(
            "y",
            successes
                .iter()
                .zip(failures)
                .map(|(s, f)| vec![*s, *f])
                .collect::<Vec<_>>(),
        )?,
    }
    out.insert("x", design.x.clone())?;
    out.insert("columns", design.columns.clone())?;
    out.insert("intercept", design.intercept)?;
    out.insert("terms", design.terms.clone())?;
    out.insert(
        "assign",
        design.assign.iter().map(|v| *v as i64).collect::<Vec<_>>(),
    )?;
    out.insert("formula", design.formula.clone())?;
    out.insert("response", design.response_name.clone())?;
    out.insert("offset", design.offset.clone())?;
    out.insert("random", random_table(&design)?)?;
    let (retained, dropped) = rows(&design);
    out.insert("n_input", design.n_input)?;
    out.insert("n_observations", retained.len())?;
    out.insert("n_dropped", dropped.len())?;
    out.insert("retained_rows", retained)?;
    out.insert("dropped_rows", dropped.clone())?;
    if !dropped.is_empty() {
        ext_php_rs::error::php_error(
            &ext_php_rs::flags::ErrorType::Warning,
            &format!(
                "stats_model_matrix dropped {} of {} observations with missing values (listwise deletion)",
                dropped.len(),
                design.n_input
            ),
        );
    }
    Ok(out)
}

/// Linear model from a formula: the stats_ols core plus coefficient names.
#[php_function]
#[php(defaults(missing = "\"drop\""))]
pub fn stats_lm(
    formula: &str,
    data: &Zval,
    missing: &str,
    weights: Option<&Zval>,
) -> PhpResult<ZBox<ZendHashTable>> {
    let Prepared { design, weights } = prepare(formula, data, missing, weights, true)?;
    reject_random(&design, "stats_lm")?;
    let y = numeric_response(&design, "stats_lm")?;
    let offset = design.offset.clone().unwrap_or_default();
    let result = stats_core::fit_ols_with_data(design.x.clone(), y, weights, offset)
        .map_err(stats_exception)?;
    let mut output = crate::ols_output(result)?;
    add_names(&mut output, &design)?;
    let (retained, dropped) = rows(&design);
    finish_result(output, design.n_input, retained, dropped, "stats_lm")
}

fn binomial_response(design: &ModelDesign, weights: &mut [f64]) -> PhpResult<Vec<f64>> {
    let y = match &design.response {
        Some(Response::Numeric(v)) => v.clone(),
        Some(Response::Factor { codes, .. }) => {
            // R: the first level is failure and every other level success.
            codes.iter().map(|c| f64::from(u8::from(*c != 0))).collect()
        }
        Some(Response::Binomial {
            successes,
            failures,
        }) => {
            let mut y = Vec::with_capacity(successes.len());
            for ((s, f), w) in successes.iter().zip(failures).zip(weights.iter_mut()) {
                if *s < 0.0 || *f < 0.0 || s.fract() != 0.0 || f.fract() != 0.0 || s + f <= 0.0 {
                    return Err(invalid(
                        "cbind(successes, failures) needs nonnegative integer counts with positive totals",
                    ));
                }
                y.push(s / (s + f));
                *w *= s + f;
            }
            y
        }
        None => return Err(invalid("stats_glm needs a response: 'y ~ ...'")),
    };
    if y.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(invalid("binary/grouped-binomial y must be in [0,1]"));
    }
    if weights.iter().any(|w| w.fract() != 0.0) {
        return Err(invalid("binomial weights must be integer trial counts"));
    }
    if y.iter()
        .zip(weights.iter())
        .any(|(v, w)| (v * w - (v * w).round()).abs() > 1e-8)
    {
        return Err(invalid(
            "binary/grouped-binomial outcomes require integer trial counts and successes",
        ));
    }
    Ok(y)
}

/// Generalized linear model from a formula over the existing GLM cores.
#[php_function]
#[php(defaults(family = "\"gaussian\"", link = "\"\"", missing = "\"drop\""))]
pub fn stats_glm(
    formula: &str,
    data: &Zval,
    family: &str,
    link: &str,
    missing: &str,
    weights: Option<&Zval>,
) -> PhpResult<ZBox<ZendHashTable>> {
    let link = match (family, link) {
        ("gaussian", "" | "identity") => "identity",
        ("binomial", "" | "logit") => "logit",
        ("binomial", "probit") => "probit",
        ("poisson", "" | "log") => "log",
        ("gaussian" | "binomial" | "poisson", other) => {
            return Err(input_exception(
                ErrorKind::Unsupported,
                &format!(
                    "link '{other}' is not supported for family '{family}'; supported: gaussian/identity, binomial/logit, binomial/probit, poisson/log"
                ),
            ));
        }
        _ => {
            return Err(input_exception(
                ErrorKind::Unsupported,
                &format!("family '{family}' is not supported; use gaussian, binomial or poisson"),
            ));
        }
    };
    let Prepared {
        design,
        mut weights,
    } = prepare(formula, data, missing, weights, true)?;
    reject_random(&design, "stats_glm")?;
    let offset = design.offset.clone().unwrap_or_default();
    let x = design.x.clone();
    let mut output = match link {
        "identity" => {
            let y = numeric_response(&design, "stats_glm")?;
            let n = y.len();
            let off = if offset.is_empty() {
                vec![0.0; n]
            } else {
                offset.clone()
            };
            let result = stats_core::fit_ols_with_data(x, y.clone(), weights.clone(), offset)
                .map_err(stats_exception)?;
            let p = result.coefficients.len();
            let deviance = result.rss;
            // R's glm null deviance: intercept-plus-offset fit, else the offset alone.
            let centered: Vec<f64> = y.iter().zip(&off).map(|(y, o)| y - o).collect();
            let center = if design.intercept {
                centered
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| v * w)
                    .sum::<f64>()
                    / weights.iter().sum::<f64>()
            } else {
                0.0
            };
            let null_deviance = centered
                .iter()
                .zip(&weights)
                .map(|(v, w)| w * (v - center).powi(2))
                .sum::<f64>();
            let aic = -2.0 * result.log_likelihood + 2.0 * (p as f64 + 1.0);
            let mut output = crate::ols_output(result)?;
            output.insert("deviance", deviance)?;
            output.insert("null_deviance", null_deviance)?;
            output.insert("aic", aic)?;
            output
        }
        "logit" | "probit" => {
            let y = binomial_response(&design, &mut weights)?;
            if link == "logit" {
                glm_output!(
                    stats_core::fit_logistic_with_data(x, y, weights, offset)
                        .map_err(stats_exception)?
                )
            } else {
                glm_output!(
                    stats_core::fit_probit_with_data(x, y, weights, offset)
                        .map_err(stats_exception)?
                )
            }
        }
        _ => {
            let y = numeric_response(&design, "stats_glm")?;
            if y.iter().any(|v| *v < 0.0 || v.fract() != 0.0) {
                return Err(invalid("Poisson y must contain nonnegative integer counts"));
            }
            glm_output!(
                stats_core::fit_poisson_with_data(x, y, weights, offset)
                    .map_err(stats_exception)?
            )
        }
    };
    output.insert("family", family)?;
    output.insert("link", link)?;
    add_names(&mut output, &design)?;
    let (retained, dropped) = rows(&design);
    finish_result(output, design.n_input, retained, dropped, "stats_glm")
}

fn group_indices(groups: &[GroupLabel]) -> (Vec<usize>, Vec<crate::GroupLabel>) {
    // First-appearance order after deletion, as stats_lmm_fit/stats_logistic_mixed use.
    let mut labels = Vec::new();
    let mut map = std::collections::HashMap::new();
    let indices = groups
        .iter()
        .map(|g| {
            *map.entry(g.clone()).or_insert_with(|| {
                labels.push(match g {
                    GroupLabel::Integer(v) => crate::GroupLabel::Integer(*v),
                    GroupLabel::Text(v) => crate::GroupLabel::Text(v.clone()),
                });
                labels.len() - 1
            })
        })
        .collect();
    (indices, labels)
}

fn require_random(design: &ModelDesign, function: &str) -> PhpResult<()> {
    if design.random.is_empty() {
        return Err(invalid(&format!(
            "{function} needs at least one random-effect term such as (1 | g); use stats_lm or stats_glm for fixed effects only"
        )));
    }
    Ok(())
}

fn positive_count(value: i64, what: &str, kind: ErrorKind) -> PhpResult<usize> {
    usize::try_from(value).map_err(|_| input_exception(kind, &format!("{what} must be positive")))
}

/// Gaussian mixed model from an lme4-style formula over the stats_lmm_fit core.
#[php_function]
#[php(defaults(
    reml = true,
    missing = "\"drop\"",
    max_evaluations = 20000,
    tolerance = 1e-7,
    optimizer = "\"bobyqa\"",
    objective_tolerance = 1e-12
))]
#[allow(clippy::too_many_arguments)] // Named PHP arguments mirror stats_lmm_fit.
pub fn stats_lmer(
    formula: &str,
    data: &Zval,
    reml: bool,
    missing: &str,
    weights: Option<&Zval>,
    start: Option<&ZendHashTable>,
    max_evaluations: i64,
    tolerance: f64,
    optimizer: &str,
    objective_tolerance: f64,
) -> PhpResult<ZBox<ZendHashTable>> {
    let optimizer = match optimizer {
        "bobyqa" => stats_core::LmmOptimizer::Bobyqa,
        "nelder_mead" => stats_core::LmmOptimizer::NelderMead,
        _ => {
            return Err(input_exception(
                ErrorKind::Unsupported,
                "optimizer must be 'bobyqa' or 'nelder_mead'",
            ));
        }
    };
    let start = start
        .map(|v| {
            v.values()
                .map(|v| {
                    crate::number(v)?.ok_or_else(|| invalid("start cannot contain missing values"))
                })
                .collect::<PhpResult<Vec<_>>>()
        })
        .transpose()?;
    let Prepared { design, weights } = prepare(formula, data, missing, weights, true)?;
    require_random(&design, "stats_lmer")?;
    let y = numeric_response(&design, "stats_lmer")?;
    let mut term_labels = Vec::new();
    let terms = design
        .random
        .iter()
        .map(|term| {
            let (groups, labels) = group_indices(&term.groups);
            term_labels.push(labels);
            stats_core::RandomTerm {
                z: term.z.clone(),
                groups,
                correlated: term.correlated,
            }
        })
        .collect();
    let options = stats_core::LmmOptions {
        reml,
        optimizer,
        objective_tolerance,
        offset: design.offset.clone().unwrap_or_else(|| vec![0.0; y.len()]),
        weights,
        start,
        max_evaluations: positive_count(
            max_evaluations,
            "max_evaluations",
            ErrorKind::InvalidInput,
        )?,
        tolerance,
    };
    let result =
        stats_core::fit_lmm(design.x.clone(), y, terms, options).map_err(stats_exception)?;
    let mut output = crate::lmm_output(result, reml, term_labels)?;
    if let Some(random) = output.get_mut("random").and_then(|v| v.array_mut()) {
        for (i, term) in design.random.iter().enumerate() {
            if let Some(t) = random.get_index_mut(i as i64).and_then(|v| v.array_mut()) {
                t.insert("label", term.label.clone())?;
                t.insert("group", term.group.clone())?;
                t.insert("columns", term.columns.clone())?;
                t.insert("correlated", term.correlated)?;
            }
        }
    }
    add_names(&mut output, &design)?;
    let (retained, dropped) = rows(&design);
    finish_result(output, design.n_input, retained, dropped, "stats_lmer")
}

/// Binomial random-intercept model from a formula over the stats_logistic_mixed core.
#[php_function]
#[php(defaults(
    family = "\"binomial\"",
    link = "\"\"",
    missing = "\"drop\"",
    quadrature_points = 1,
    max_evaluations = 20000,
    tolerance = 1e-7
))]
#[allow(clippy::too_many_arguments)] // Named PHP arguments mirror stats_logistic_mixed.
pub fn stats_glmer(
    formula: &str,
    data: &Zval,
    family: &str,
    link: &str,
    missing: &str,
    weights: Option<&Zval>,
    quadrature_points: i64,
    max_evaluations: i64,
    tolerance: f64,
) -> PhpResult<ZBox<ZendHashTable>> {
    if family != "binomial" || !matches!(link, "" | "logit") {
        return Err(input_exception(
            ErrorKind::Unsupported,
            &format!(
                "stats_glmer supports family 'binomial' with the logit link only (got '{family}'/'{link}')"
            ),
        ));
    }
    let Prepared {
        design,
        mut weights,
    } = prepare(formula, data, missing, weights, true)?;
    require_random(&design, "stats_glmer")?;
    if design.random.len() != 1 || design.random[0].columns != ["(Intercept)"] {
        return Err(input_exception(
            ErrorKind::Unsupported,
            "stats_glmer supports exactly one random intercept, (1 | g); random slopes and several terms are not implemented",
        ));
    }
    let y = binomial_response(&design, &mut weights)?;
    let (indices, labels) = group_indices(&design.random[0].groups);
    let options = stats_core::LogisticMixedOptions {
        offset: design.offset.clone().unwrap_or_else(|| vec![0.0; y.len()]),
        weights,
        quadrature_points: positive_count(
            quadrature_points,
            "quadrature_points",
            ErrorKind::Unsupported,
        )?,
        max_evaluations: positive_count(
            max_evaluations,
            "max_evaluations",
            ErrorKind::InvalidInput,
        )?,
        tolerance,
    };
    let r = stats_core::fit_logistic_random_intercept(design.x.clone(), y, indices, options)
        .map_err(stats_exception)?;
    let mut output = crate::logistic_mixed_output(r, &labels)?;
    let mut random = ZendHashTable::new();
    let mut term = ZendHashTable::new();
    term.insert("label", design.random[0].label.clone())?;
    term.insert("group", design.random[0].group.clone())?;
    term.insert("columns", design.random[0].columns.clone())?;
    random.push(term)?;
    output.insert("random", random)?;
    add_names(&mut output, &design)?;
    let (retained, dropped) = rows(&design);
    finish_result(output, design.n_input, retained, dropped, "stats_glmer")
}

pub fn register(module: ModuleBuilder) -> ModuleBuilder {
    module
        .function(wrap_function!(stats_model_matrix))
        .function(wrap_function!(stats_lm))
        .function(wrap_function!(stats_glm))
        .function(wrap_function!(stats_lmer))
        .function(wrap_function!(stats_glmer))
}
