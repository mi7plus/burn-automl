//! Error and result types shared across the core engine.

use thiserror::Error;

/// The result type used throughout `automl-core`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors surfaced by the optimization engine.
///
/// These are intended to be *actionable* per the release gates in the PRD
/// (§26): each variant should tell the user what went wrong and, where
/// possible, which parameter or trial was involved.
#[derive(Debug, Error)]
pub enum Error {
    /// A parameter was requested with a distribution incompatible with the
    /// one already registered under the same name in this study.
    #[error("parameter `{name}` was already defined with a different distribution ({existing} vs {requested})")]
    DistributionMismatch {
        /// The parameter name.
        name: String,
        /// The distribution kind previously registered.
        existing: String,
        /// The distribution kind requested now.
        requested: String,
    },

    /// A parameter value was requested but is absent from a `ParamSet`.
    #[error("parameter `{0}` is missing from the parameter set")]
    MissingParam(String),

    /// A parameter value had a different type than the caller expected.
    #[error("parameter `{name}` is a {actual} but was requested as {expected}")]
    ParamTypeMismatch {
        /// The parameter name.
        name: String,
        /// The type actually stored.
        actual: String,
        /// The type the caller requested.
        expected: String,
    },

    /// A distribution was constructed with invalid bounds.
    #[error("invalid distribution for `{name}`: {reason}")]
    InvalidDistribution {
        /// The parameter name.
        name: String,
        /// Why the distribution is invalid.
        reason: String,
    },

    /// A referenced study, trial, or entity does not exist in storage.
    #[error("{kind} `{id}` not found in storage")]
    NotFound {
        /// The kind of entity (e.g. "study", "trial").
        kind: &'static str,
        /// Its identifier, rendered as a string.
        id: String,
    },

    /// A categorical distribution was declared with no choices.
    #[error("categorical parameter `{0}` has no choices")]
    EmptyCategorical(String),

    /// An objective closure or task adapter reported a failure.
    #[error("objective evaluation failed: {0}")]
    Objective(String),

    /// A storage backend failed.
    #[error("storage error: {0}")]
    Storage(String),

    /// Serialization/deserialization of a persisted record failed.
    #[error("serialization error: {0}")]
    Serde(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serde(e.to_string())
    }
}
