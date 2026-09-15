//! # automl-core
//!
//! The framework-agnostic optimization engine at the heart of `burn-automl`.
//!
//! Per the implementation plan, this crate understands *experiments*, not
//! tasks: parameter spaces, trials, observations, objectives, budgets, pruning
//! decisions, persistence and reproducibility — and nothing about regression, a
//! Transformer, PPO, or a U-Net. Domain-specific task layers (shipped in later
//! crates) translate an ML problem into these generic primitives via
//! [`objective::TaskAdapter`].
//!
//! ## The universal loop
//!
//! ```text
//! Study -> Sampler -> ParamSet -> Storage(enqueue/start)
//!       -> Objective -> intermediate reports -> Pruner
//!       -> Storage(complete) -> Sampler observes -> next trial
//! ```
//!
//! ## What v0.1 provides
//!
//! - [`distribution`]/[`space`]: flat and conditional/hierarchical search spaces.
//! - [`sampler`]: the `Sampler` trait with Random and Grid baselines that
//!   handle every conditional space (the correctness floor from the plan §5.1).
//! - [`pruner`]: the `Pruner` trait with median and no-op pruners.
//! - [`storage`]: the `Storage` trait with a thread-safe in-memory backend and
//!   idempotent-by-`(trial, step)` reporting.
//! - [`budget`]: `Budget` as a trait with trial/wall-time/epoch built-ins.
//! - [`study`]: the `Study` handle and its deterministic, replayable loop.
//!
//! ```
//! use automl_core::prelude::*;
//!
//! let space = SearchSpace::new().add("x", Distribution::float(-5.0, 5.0));
//! let mut study = Study::builder(space)
//!     .minimize("loss")
//!     .seed(42)
//!     .build()
//!     .unwrap();
//!
//! let n = study
//!     .optimize_n(&|p: &ParamSet, _r: &mut dyn ReportSink| {
//!         let x = p.float("x")?;
//!         Ok(NamedMetrics::single("loss", (x - 2.0).powi(2)))
//!     }, 100)
//!     .unwrap();
//! assert_eq!(n, 100);
//! let best = study.best_trial().unwrap().unwrap();
//! assert!((best.params.float("x").unwrap() - 2.0).abs() < 1.0);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod budget;
pub mod distribution;
pub mod error;
pub mod evolution;
pub mod executor;
pub mod importance;
pub mod metrics;
pub mod objective;
pub mod param;
pub mod pareto;
pub mod pruner;
pub mod sampler;
pub mod space;
pub mod storage;
pub mod study;
pub mod tpe;
pub mod trial;

/// Persistent SQLite storage backend (enable the `sqlite` feature).
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use error::{Error, Result};

/// The common imports for using the engine.
pub mod prelude {
    pub use crate::budget::{
        Budget, Consumption, EpochBudget, StepBudget, TrialBudget, Unbounded, WallTimeBudget,
    };
    pub use crate::distribution::Distribution;
    pub use crate::error::{Error, Result};
    pub use crate::evolution::EvolutionarySampler;
    pub use crate::executor::{Executor, ResourceSpec, SequentialExecutor, ThreadExecutor};
    pub use crate::importance::{importance, ParamImportance};
    pub use crate::metrics::{Direction, NamedMetrics, Objective as ObjectiveSpec};
    pub use crate::objective::{Objective, ReportSink, TaskAdapter};
    pub use crate::param::{ParamSet, ParamValue};
    pub use crate::pareto::{Member, ParetoFront};
    pub use crate::pruner::{AshaPruner, MedianPruner, MultiObjectivePruner, NoPruner, Pruner};
    pub use crate::sampler::{GridSampler, RandomSampler, Sampler};
    pub use crate::space::{Condition, SearchSpace};
    #[cfg(feature = "sqlite")]
    pub use crate::sqlite::SqliteStorage;
    pub use crate::storage::{InMemoryStorage, Storage, StudyMeta};
    pub use crate::study::{Study, StudyBuilder};
    pub use crate::tpe::TpeSampler;
    pub use crate::trial::{StudyId, TrialHistory, TrialId, TrialRecord, TrialState};
}
