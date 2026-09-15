//! Environment and timing provenance for reproducibility (PRD §19).
//!
//! The persistence table in §19 requires each trial to record its environment
//! (crate versions, backend, OS/arch) and timing (queued/start/end, wall time)
//! alongside its parameters, metrics and seed. These types capture the
//! framework-agnostic parts of that: the core crate version and OS/arch, plus
//! millisecond timestamps for each lifecycle transition. Backend/device details
//! are an adapter concern (e.g. `automl-burn`) and can be layered on top.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since the Unix epoch, or 0 if the clock is before it.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A snapshot of the environment a trial ran in (the framework-agnostic subset
/// of §19's "Environment" row).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvSnapshot {
    /// Operating system (`std::env::consts::OS`, e.g. "windows", "linux").
    pub os: String,
    /// CPU architecture (`std::env::consts::ARCH`, e.g. "x86_64").
    pub arch: String,
    /// The `automl-core` crate version that produced the trial.
    pub core_version: String,
}

impl EnvSnapshot {
    /// Capture the current environment.
    pub fn capture() -> Self {
        EnvSnapshot {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            core_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

impl Default for EnvSnapshot {
    fn default() -> Self {
        EnvSnapshot::capture()
    }
}

/// Lifecycle timestamps for a trial (§19's "Timing" row), in Unix milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TrialTiming {
    /// When the trial was enqueued.
    pub queued_at_ms: u64,
    /// When execution started, if it has.
    pub started_at_ms: Option<u64>,
    /// When the trial reached a terminal state, if it has.
    pub completed_at_ms: Option<u64>,
}

impl TrialTiming {
    /// A timing record marking "queued now".
    pub fn queued_now() -> Self {
        TrialTiming {
            queued_at_ms: now_ms(),
            started_at_ms: None,
            completed_at_ms: None,
        }
    }

    /// Wall-clock duration from start to completion, if both are recorded.
    pub fn wall_time_ms(&self) -> Option<u64> {
        match (self.started_at_ms, self.completed_at_ms) {
            (Some(s), Some(e)) => Some(e.saturating_sub(s)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_capture_is_populated() {
        let env = EnvSnapshot::capture();
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert_eq!(env.core_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn wall_time_needs_both_ends() {
        let mut t = TrialTiming::queued_now();
        assert_eq!(t.wall_time_ms(), None);
        t.started_at_ms = Some(1000);
        t.completed_at_ms = Some(1500);
        assert_eq!(t.wall_time_ms(), Some(500));
    }
}
