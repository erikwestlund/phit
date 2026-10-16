//! Tokenizer, Pratt parser and R-compatible deparser for model formulas.
//!
//! Operator precedence follows R's `?Syntax`: `^` > unary `-`/`+` > `:` >
//! `%in%` > `*`/`/` > binary `+`/`-` > comparisons > `!` > `&` > `|` > `~`.
use crate::{ErrorKind, StatsError};

fn error(message: impl Into<String>) -> StatsError {
    StatsError::new(ErrorKind::InvalidInput, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Tilde,
    Or,
    DoubleOr,
    And,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    Add,
    Sub,
    Mul,
    Div,
    In,
    Colon,
    Pow,
}

impl Op {
    fn text(self) -> &'static str {
        match self {
            Op::Tilde => "~",
            Op::Or => "|",
            Op::DoubleOr => "||",
            Op::And => "&",
            Op::Lt => "<",
            Op::Gt => ">",
            Op::Le => "<=",
            Op::Ge => ">=",
            Op::Eq => "==",
            Op::Ne => "!=",
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::In => "%in%",
            Op::Colon => ":",
            Op::Pow => "^",
        }
    }
    // (left binding power, right binding power)
    fn binding(self) -> (u8, u8) {
        match self {
            Op::Tilde => (2, 3),
            Op::Or | Op::DoubleOr => (4, 5),
            Op::And => (6, 7),
            Op::Lt | Op::Gt | Op::Le | Op::Ge | Op::Eq | Op::Ne => (10, 11),
            Op::Add | Op::Sub => (12, 13),
            Op::Mul | Op::Div => (14, 15),
            Op::In => (16, 17),
            Op::Colon => (18, 19),
            Op::Pow => (23, 22),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Expr {
    Num(f64),
    Bool(bool),
    Name(String),
    Str(String),
    Dot,
    Call(String, Vec<Arg>),
    Unary(char, Box<Expr>),
    Binary(Op, Box<Expr>, Box<Expr>),
    Paren(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Num(f64),
    Str(String),
    Name(String),
    Op(Op),
    Not,
    LParen,
    RParen,
    Comma,
    Assign,
}

fn tokenize(text: &str) -> Result<Vec<Token>, StatsError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let next = chars.get(i + 1).copied();
        if c.is_ascii_digit() || (c == '.' && next.is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    i = j;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let literal: String = chars[start..i].iter().collect();
            let value = literal
                .parse::<f64>()
                .map_err(|_| error(format!("invalid number '{literal}' in formula")))?;
            // R integer suffix: 2L.
            if i < chars.len() && chars[i] == 'L' {
                i += 1;
            }
            out.push(Token::Num(value));
            continue;
        }
        if c.is_alphabetic() || c == '.' || c == '_' {
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
            out.push(Token::Name(chars[start..i].iter().collect()));
            continue;
        }
        if c == '"' || c == '\'' {
            let mut value = String::new();
            i += 1;
            loop {
                match chars.get(i) {
                    None => return Err(error("unterminated string in formula")),
                    Some(q) if *q == c => break,
                    Some('\\') => {
                        value.push(
                            *chars
                                .get(i + 1)
                                .ok_or_else(|| error("unterminated string in formula"))?,
                        );
                        i += 2;
                    }
                    Some(ch) => {
                        value.push(*ch);
                        i += 1;
                    }
                }
            }
            i += 1;
            out.push(Token::Str(value));
            continue;
        }
        if c == '`' {
            let start = i + 1;
            i += 1;
            while i < chars.len() && chars[i] != '`' {
                i += 1;
            }
            if i >= chars.len() {
                return Err(error("unterminated backtick name in formula"));
            }
            let name: String = chars[start..i].iter().collect();
            if name.is_empty() {
                return Err(error("empty backtick name in formula"));
            }
            out.push(Token::Name(name));
            i += 1;
            // Distinguish a quoted name from the bare `.` operator.
            continue;
        }
        let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
        let (token, width) = match two.as_str() {
            "||" => (Token::Op(Op::DoubleOr), 2),
            "&&" => (Token::Op(Op::And), 2),
            "<=" => (Token::Op(Op::Le), 2),
            ">=" => (Token::Op(Op::Ge), 2),
            "==" => (Token::Op(Op::Eq), 2),
            "!=" => (Token::Op(Op::Ne), 2),
            "**" => (Token::Op(Op::Pow), 2),
            _ => match c {
                '~' => (Token::Op(Op::Tilde), 1),
                '|' => (Token::Op(Op::Or), 1),
                '&' => (Token::Op(Op::And), 1),
                '<' => (Token::Op(Op::Lt), 1),
                '>' => (Token::Op(Op::Gt), 1),
                '+' => (Token::Op(Op::Add), 1),
                '-' => (Token::Op(Op::Sub), 1),
                '*' => (Token::Op(Op::Mul), 1),
                '/' => (Token::Op(Op::Div), 1),
                ':' => (Token::Op(Op::Colon), 1),
                '^' => (Token::Op(Op::Pow), 1),
                '!' => (Token::Not, 1),
                '(' => (Token::LParen, 1),
                ')' => (Token::RParen, 1),
                ',' => (Token::Comma, 1),
                '=' => (Token::Assign, 1),
                '%' => {
                    let rest: String = chars[i..].iter().collect();
                    if rest.starts_with("%in%") {
                        (Token::Op(Op::In), 4)
                    } else {
                        return Err(error("only the %in% operator is supported in formulas"));
                    }
                }
                _ => return Err(error(format!("unexpected character '{c}' in formula"))),
            },
        };
        out.push(token);
        i += width;
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn expect(&mut self, token: Token, what: &str) -> Result<(), StatsError> {
        if self.next().as_ref() == Some(&token) {
            Ok(())
        } else {
            Err(error(format!("expected {what} in formula")))
        }
    }
    fn prefix(&mut self) -> Result<Expr, StatsError> {
        match self.next() {
            Some(Token::Num(v)) => Ok(Expr::Num(v)),
            Some(Token::Str(v)) => Ok(Expr::Str(v)),
            Some(Token::Name(name)) => {
                if self.peek() == Some(&Token::LParen) {
                    self.pos += 1;
                    let mut args = Vec::new();
                    if self.peek() == Some(&Token::RParen) {
                        self.pos += 1;
                        return Ok(Expr::Call(name, args));
                    }
                    loop {
                        let named = matches!(
                            (self.peek(), self.tokens.get(self.pos + 1)),
                            (Some(Token::Name(_)), Some(Token::Assign))
                        );
                        let arg_name = if named {
                            let Some(Token::Name(n)) = self.next() else {
                                unreachable!()
                            };
                            self.pos += 1;
                            Some(n)
                        } else {
                            None
                        };
                        let value = self.expr(0)?;
                        args.push(Arg {
                            name: arg_name,
                            value,
                        });
                        match self.next() {
                            Some(Token::Comma) => continue,
                            Some(Token::RParen) => break,
                            _ => return Err(error("expected ',' or ')' in function call")),
                        }
                    }
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(match name.as_str() {
                        "." => Expr::Dot,
                        "TRUE" | "T" => Expr::Bool(true),
                        "FALSE" | "F" => Expr::Bool(false),
                        _ => Expr::Name(name),
                    })
                }
            }
            Some(Token::LParen) => {
                let inner = self.expr(0)?;
                self.expect(Token::RParen, "')'")?;
                Ok(Expr::Paren(Box::new(inner)))
            }
            Some(Token::Op(Op::Sub)) => Ok(Expr::Unary('-', Box::new(self.expr(20)?))),
            Some(Token::Op(Op::Add)) => Ok(Expr::Unary('+', Box::new(self.expr(20)?))),
            Some(Token::Op(Op::Tilde)) => Ok(Expr::Unary('~', Box::new(self.expr(3)?))),
            Some(Token::Not) => Ok(Expr::Unary('!', Box::new(self.expr(8)?))),
            Some(_) => Err(error("unexpected token in formula")),
            None => Err(error("formula ended unexpectedly")),
        }
    }
    fn expr(&mut self, min: u8) -> Result<Expr, StatsError> {
        let mut left = self.prefix()?;
        while let Some(Token::Op(op)) = self.peek() {
            let op = *op;
            let (lbp, rbp) = op.binding();
            if lbp < min {
                break;
            }
            self.pos += 1;
            let right = self.expr(rbp)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }
}

pub(crate) fn parse(text: &str) -> Result<Expr, StatsError> {
    let tokens = tokenize(text)?;
    if tokens.is_empty() {
        return Err(error("formula is empty"));
    }
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.expr(0)?;
    if parser.pos < parser.tokens.len() {
        return Err(error("unexpected trailing input in formula"));
    }
    Ok(expr)
}

const RESERVED: [&str; 19] = [
    "if",
    "else",
    "repeat",
    "while",
    "function",
    "for",
    "next",
    "break",
    "TRUE",
    "FALSE",
    "NULL",
    "Inf",
    "NaN",
    "NA",
    "NA_integer_",
    "NA_real_",
    "NA_character_",
    "in",
    "T",
];

/// R's syntactic-name rule; other names are deparsed in backticks.
pub(crate) fn syntactic(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let second = name.chars().nth(1);
    let starts =
        first.is_alphabetic() || (first == '.' && !second.is_some_and(|c| c.is_ascii_digit()));
    starts
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '.' || c == '_')
        && !(RESERVED.contains(&name) && name != "T")
        && name != "..."
}

pub(crate) fn deparse_name(name: &str) -> String {
    if syntactic(name) {
        name.to_owned()
    } else {
        format!("`{name}`")
    }
}

/// R's 15-significant-digit number formatting (deparse, as.character, factor labels).
pub(crate) fn format_number(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "Inf".into()
        } else {
            "-Inf".into()
        };
    }
    if value == 0.0 {
        return "0".into();
    }
    let rounded: f64 = format!("{value:.14e}").parse().unwrap_or(value);
    let mut digits = 15;
    for d in 1..=15 {
        let candidate: f64 = format!("{:.*e}", d - 1, value).parse().unwrap_or(f64::NAN);
        if candidate == rounded {
            digits = d;
            break;
        }
    }
    let sci_raw = format!("{:.*e}", digits - 1, value);
    let (mantissa, exponent) = sci_raw.split_once('e').unwrap_or((&sci_raw, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sci = format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { '-' } else { '+' },
        exponent.abs()
    );
    let decimals = (digits as i32 - 1 - exponent).max(0) as usize;
    let fixed = format!("{value:.decimals$}");
    if fixed.len() <= sci.len() { fixed } else { sci }
}

pub(crate) fn deparse(expr: &Expr) -> String {
    match expr {
        Expr::Num(v) => format_number(*v),
        Expr::Bool(v) => if *v { "TRUE" } else { "FALSE" }.into(),
        Expr::Name(n) => deparse_name(n),
        Expr::Str(v) => format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")),
        Expr::Dot => ".".into(),
        Expr::Call(name, args) => {
            let args: Vec<String> = args
                .iter()
                .map(|a| match &a.name {
                    Some(n) => format!("{} = {}", deparse_name(n), deparse(&a.value)),
                    None => deparse(&a.value),
                })
                .collect();
            format!("{}({})", deparse_name(name), args.join(", "))
        }
        Expr::Unary(op, e) => format!("{op}{}", deparse(e)),
        Expr::Paren(e) => format!("({})", deparse(e)),
        Expr::Binary(op, l, r) => match op {
            Op::Pow | Op::Colon | Op::Div => format!("{}{}{}", deparse(l), op.text(), deparse(r)),
            _ => format!("{} {} {}", deparse(l), op.text(), deparse(r)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_follow_r_deparse() {
        assert_eq!(format_number(100000.0), "1e+05");
        assert_eq!(format_number(10000.0), "10000");
        assert_eq!(format_number(0.0001), "1e-04");
        assert_eq!(format_number(123456.7), "123456.7");
        assert_eq!(format_number(1e15), "1e+15");
        assert_eq!(format_number(0.123_456_789_012_345_68), "0.123456789012346");
        assert_eq!(format_number(2.0), "2");
        assert_eq!(format_number(-0.5), "-0.5");
    }

    #[test]
    fn precedence_and_spacing() {
        let e = parse("y ~ log(x, base=2) + I(x^2*3) + a:b*c - 1").unwrap();
        assert_eq!(
            deparse(&e),
            "y ~ log(x, base = 2) + I(x^2 * 3) + a:b * c - 1"
        );
        let e = parse("y ~ x + (1 + z | g)").unwrap();
        assert_eq!(deparse(&e), "y ~ x + (1 + z | g)");
        assert_eq!(deparse(&parse("`a b` ~ -x").unwrap()), "`a b` ~ -x");
    }
}
