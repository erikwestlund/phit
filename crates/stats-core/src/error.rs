use std::{error::Error, fmt};

/// Stable statistical error categories. PHP exposes these as Exception codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorKind {
    InvalidInput = 1001,
    DimensionMismatch = 1002,
    Nonfinite = 1003,
    RankDeficiency = 1004,
    FactorizationFailure = 1005,
    Nonconvergence = 1006,
    Unsupported = 1007,
    Internal = 1008,
}

/// The message remains accessible as .0; categories are independent of wording.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsError(pub String, ErrorKind);
impl StatsError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self(message.into(), kind)
    }
    pub fn kind(&self) -> ErrorKind {
        self.1
    }
}
impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for StatsError {}
