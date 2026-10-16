//! R `terms.formula` semantics: variables in order of first appearance, terms as
//! variable sets, sorted by interaction order, and the contrast/dummy coding
//! heuristic from Chambers & Hastie (1992, p. 38) used by R's `TermCode`.
use super::parse::{Expr, Op, deparse};
use crate::{ErrorKind, StatsError};

fn error(message: impl Into<String>) -> StatsError {
    StatsError::new(ErrorKind::InvalidInput, message)
}

/// A term is a sorted set of variable indices; sorting by index is R's variable order.
pub(crate) type Term = Vec<usize>;

#[derive(Clone, Debug)]
pub(crate) struct Terms {
    /// Variable expressions and their deparsed names (R's identity for variables).
    pub variables: Vec<(String, Expr)>,
    pub response: Option<usize>,
    pub terms: Vec<Term>,
    pub intercept: bool,
    pub offsets: Vec<usize>,
}

impl Terms {
    pub(crate) fn label(&self, term: &Term) -> String {
        term.iter()
            .map(|v| self.variables[*v].0.clone())
            .collect::<Vec<_>>()
            .join(":")
    }

    /// 1 = code by contrasts, 2 = code by indicator (dummy) variables (R's TermCode).
    pub(crate) fn coding(&self, index: usize, variable: usize) -> u8 {
        let reduced: Vec<usize> = self.terms[index]
            .iter()
            .copied()
            .filter(|v| *v != variable)
            .collect();
        if reduced.is_empty()
            || self.terms[..index]
                .iter()
                .any(|t| reduced.iter().all(|v| t.contains(v)))
        {
            1
        } else {
            2
        }
    }
}

struct Encoder<'a> {
    variables: Vec<(String, Expr)>,
    intercept: bool,
    offsets: Vec<usize>,
    frame_names: &'a [String],
    excluded: Vec<String>,
}

fn union(a: &Term, b: &Term) -> Term {
    let mut t: Term = a.iter().chain(b.iter()).copied().collect();
    t.sort_unstable();
    t.dedup();
    t
}

fn trim(terms: Vec<Term>) -> Vec<Term> {
    let mut out: Vec<Term> = Vec::new();
    for t in terms {
        if !t.is_empty() && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

fn interact(left: &[Term], right: &[Term]) -> Vec<Term> {
    let mut out = Vec::new();
    for l in left {
        for r in right {
            out.push(union(l, r));
        }
    }
    trim(out)
}

impl Encoder<'_> {
    fn install(&mut self, expr: Expr) -> usize {
        let key = deparse(&expr);
        if let Some(i) = self.variables.iter().position(|(k, _)| *k == key) {
            return i;
        }
        self.variables.push((key, expr));
        self.variables.len() - 1
    }

    fn encode(&mut self, expr: &Expr, negate: bool) -> Result<Vec<Term>, StatsError> {
        Ok(match expr {
            Expr::Num(v) if *v == 1.0 => {
                self.intercept = !negate;
                vec![]
            }
            Expr::Num(v) if *v == 0.0 => {
                self.intercept = negate;
                vec![]
            }
            Expr::Num(_) => {
                return Err(error(
                    "invalid model formula: numeric terms other than 0 or 1",
                ));
            }
            Expr::Paren(inner) => self.encode(inner, negate)?,
            Expr::Unary('+', inner) => self.encode(inner, negate)?,
            Expr::Unary('-', inner) => {
                // A leading -term only removes; nothing precedes it.
                self.encode(inner, !negate)?;
                vec![]
            }
            Expr::Dot => {
                if self.frame_names.is_empty() {
                    return Err(error("'.' in formula requires data columns"));
                }
                let names: Vec<String> = self
                    .frame_names
                    .iter()
                    .filter(|n| !self.excluded.contains(n))
                    .cloned()
                    .collect();
                names
                    .into_iter()
                    .map(|n| vec![self.install(Expr::Name(n))])
                    .collect()
            }
            Expr::Call(name, _) if name == "offset" => {
                let v = self.install(expr.clone());
                if !self.offsets.contains(&v) {
                    self.offsets.push(v);
                }
                vec![]
            }
            Expr::Name(_) | Expr::Call(..) | Expr::Bool(_) | Expr::Str(_) => {
                vec![vec![self.install(expr.clone())]]
            }
            Expr::Binary(op, l, r) => match op {
                Op::Add => {
                    let mut left = self.encode(l, negate)?;
                    left.extend(self.encode(r, negate)?);
                    trim(left)
                }
                Op::Sub => {
                    let left = self.encode(l, negate)?;
                    let right = self.encode(r, !negate)?;
                    left.into_iter().filter(|t| !right.contains(t)).collect()
                }
                Op::Colon => {
                    let left = self.encode(l, negate)?;
                    let right = self.encode(r, negate)?;
                    interact(&left, &right)
                }
                Op::Mul => {
                    let left = self.encode(l, negate)?;
                    let right = self.encode(r, negate)?;
                    let both = interact(&left, &right);
                    trim(left.into_iter().chain(right).chain(both).collect())
                }
                Op::Div => {
                    let left = self.encode(l, negate)?;
                    let right = self.encode(r, negate)?;
                    let all: Term = left.iter().fold(Vec::new(), |a, t| union(&a, t));
                    let nested: Vec<Term> = right.iter().map(|t| union(&all, t)).collect();
                    trim(left.into_iter().chain(nested).collect())
                }
                Op::In => {
                    let left = self.encode(l, negate)?;
                    let right = self.encode(r, negate)?;
                    let all: Term = right.iter().fold(Vec::new(), |a, t| union(&a, t));
                    trim(left.iter().map(|t| union(t, &all)).collect())
                }
                Op::Pow => {
                    let power = match r.as_ref() {
                        Expr::Num(p) if *p >= 1.0 && p.fract() == 0.0 => *p as usize,
                        _ => {
                            return Err(error(
                                "invalid power in formula: use a positive integer, e.g. (a + b)^2",
                            ));
                        }
                    };
                    let base = self.encode(l, negate)?;
                    let mut result = base.clone();
                    let mut acc = base.clone();
                    for _ in 1..power {
                        acc = interact(&acc, &base);
                        result = trim(result.into_iter().chain(acc.clone()).collect());
                    }
                    result
                }
                Op::Or | Op::DoubleOr => {
                    return Err(error(
                        "random-effect bars must be parenthesized, additive terms such as (1 | g)",
                    ));
                }
                _ => {
                    return Err(error(format!(
                        "invalid model formula: wrap '{}' in I(...)",
                        deparse(expr)
                    )));
                }
            },
            Expr::Unary(..) => {
                return Err(error(format!(
                    "invalid model formula: wrap '{}' in I(...)",
                    deparse(expr)
                )));
            }
        })
    }
}

/// Random-effect bar after expansion of nested grouping (g1/g2 -> g1, g1:g2).
#[derive(Clone, Debug)]
pub(crate) struct Bar {
    pub lhs: Expr,
    pub groups: Vec<Expr>,
    pub correlated: bool,
}

impl Bar {
    pub(crate) fn group_label(&self) -> String {
        self.groups
            .iter()
            .map(deparse)
            .collect::<Vec<_>>()
            .join(":")
    }
    pub(crate) fn label(&self) -> String {
        format!(
            "{} {} {}",
            deparse(&self.lhs),
            if self.correlated { "|" } else { "||" },
            self.group_label()
        )
    }
}

fn group_factors(expr: &Expr) -> Result<Vec<Vec<Expr>>, StatsError> {
    match expr {
        Expr::Paren(inner) => group_factors(inner),
        Expr::Binary(Op::Colon, l, r) => {
            let left = group_factors(l)?;
            let right = group_factors(r)?;
            let mut out = Vec::new();
            for a in &left {
                for b in &right {
                    out.push(a.iter().chain(b.iter()).cloned().collect());
                }
            }
            Ok(out)
        }
        Expr::Binary(Op::Div, l, r) => {
            // lme4: (x | a/b) = (x | a) + (x | a:b).
            let left = group_factors(l)?;
            let right = group_factors(r)?;
            let all: Vec<Expr> = left.iter().flatten().cloned().collect();
            let mut out = left;
            for b in right {
                out.push(all.iter().chain(b.iter()).cloned().collect());
            }
            Ok(out)
        }
        Expr::Name(_) | Expr::Call(..) => Ok(vec![vec![expr.clone()]]),
        _ => Err(error(format!(
            "unsupported grouping expression '{}' in random-effect term",
            deparse(expr)
        ))),
    }
}

/// Split the right-hand side into fixed additive parts and random-effect bars.
pub(crate) fn split_bars(
    expr: &Expr,
    fixed: &mut Vec<(bool, Expr)>,
    bars: &mut Vec<Bar>,
    negate: bool,
) -> Result<(), StatsError> {
    let bar = |inner: &Expr, bars: &mut Vec<Bar>| -> Result<bool, StatsError> {
        if let Expr::Binary(op @ (Op::Or | Op::DoubleOr), lhs, grp) = inner {
            for groups in group_factors(grp)? {
                bars.push(Bar {
                    lhs: (**lhs).clone(),
                    groups,
                    correlated: *op == Op::Or,
                });
            }
            return Ok(true);
        }
        Ok(false)
    };
    match expr {
        Expr::Binary(Op::Add, l, r) => {
            split_bars(l, fixed, bars, negate)?;
            split_bars(r, fixed, bars, negate)
        }
        Expr::Binary(Op::Sub, l, r) => {
            split_bars(l, fixed, bars, negate)?;
            split_bars(r, fixed, bars, !negate)
        }
        Expr::Paren(inner) if bar(inner, bars)? => Ok(()),
        Expr::Binary(Op::Or | Op::DoubleOr, ..) if bar(expr, bars)? => Ok(()),
        _ => {
            fixed.push((negate, expr.clone()));
            Ok(())
        }
    }
}

/// Encode a right-hand side (bars already removed) plus an optional response.
pub(crate) fn build_terms(
    response: Option<&Expr>,
    parts: &[(bool, Expr)],
    frame_names: &[String],
) -> Result<Terms, StatsError> {
    let mut excluded = Vec::new();
    if let Some(r) = response {
        collect_names(r, &mut excluded);
    }
    let mut encoder = Encoder {
        variables: Vec::new(),
        intercept: true,
        offsets: Vec::new(),
        frame_names,
        excluded,
    };
    let response_index = response.map(|r| encoder.install(r.clone()));
    let mut terms: Vec<Term> = Vec::new();
    for (negate, part) in parts {
        let encoded = encoder.encode(part, *negate)?;
        if *negate {
            terms.retain(|t| !encoded.contains(t));
        } else {
            terms = trim(terms.into_iter().chain(encoded).collect());
        }
    }
    if let Some(r) = response_index
        && terms.iter().any(|t| t.contains(&r))
    {
        return Err(error(
            "the response cannot also appear on the right-hand side of the formula",
        ));
    }
    // R's default keep.order = FALSE: stable sort by interaction order.
    terms.sort_by_key(|t| t.len());
    Ok(Terms {
        variables: encoder.variables,
        response: response_index,
        terms,
        intercept: encoder.intercept,
        offsets: encoder.offsets,
    })
}

fn collect_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Name(n) => out.push(n.clone()),
        Expr::Call(_, args) => args.iter().for_each(|a| collect_names(&a.value, out)),
        Expr::Unary(_, e) | Expr::Paren(e) => collect_names(e, out),
        Expr::Binary(_, l, r) => {
            collect_names(l, out);
            collect_names(r, out);
        }
        _ => {}
    }
}
