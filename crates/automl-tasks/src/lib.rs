//! # automl-tasks
//!
//! Framework-agnostic task adapters for `burn-automl` — classical methods that
//! need no deep-learning backend, so a plain HPO user gets clustering and
//! anomaly detection without compiling Burn (PRD §30: `automl-tasks` depends
//! only on `automl-core`). Each adapter is a thin builder over an
//! `automl_core::Study` plus a pre-built search space (§4.2).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod anomaly;
pub mod clustering;
pub mod pipeline;
pub mod rl;

pub use anomaly::AutoAnomaly;
pub use clustering::{kmeans, silhouette, AutoCluster, Clustering};
pub use pipeline::AutoPipeline;
pub use rl::{AutoRl, GridWorld, QLearner};
