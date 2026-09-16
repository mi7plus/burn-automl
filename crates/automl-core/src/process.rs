//! Process executor tier: crash-isolated workers as OS subprocesses
//! (roadmap; PRD §18.1 executor tiers, §18 "process isolation").
//!
//! The thread executor shares an address space, so a segfault or abort in one
//! trial takes down the whole run. The **process tier** spawns each worker as a
//! separate OS process against a shared, lease-backed [`Storage`]: if a worker
//! crashes, only that process dies, its trial's lease expires, and the trial is
//! recovered and re-run. This reuses the distributed protocol
//! ([`crate::distributed`]) — a process pool is just a set of workers coordinated
//! through storage — so the core carries no objective code across the process
//! boundary. The user supplies a *worker command* (a binary that runs
//! [`crate::distributed::Worker`] against the study); [`ProcessPool`] spawns,
//! supervises and restarts it.
//!
//! Because the pool waits for every worker in a round to exit before recovering
//! orphans, recovery is timing-independent: once no worker is alive, every
//! dangling lease provably belongs to a dead process and is safe to requeue.

use crate::error::{Error, Result};
use crate::provenance::now_ms;
use crate::storage::Storage;
use crate::trial::{StudyId, TrialState};
use std::ffi::OsString;
use std::process::Command;
use std::sync::Arc;

/// Outcome of a [`ProcessPool::run`]: how the run drained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessReport {
    /// Number of supervision rounds executed (initial spawn + restarts).
    pub rounds: usize,
    /// Total worker processes spawned across all rounds.
    pub spawned: usize,
    /// Worker processes that exited with a non-zero status (crashes).
    pub crashed: usize,
    /// Whether the study's queue drained (every trial reached a terminal state).
    pub drained: bool,
}

/// A supervisor that runs trials in crash-isolated worker subprocesses.
///
/// Each spawned process is `program` followed by the configured base arguments
/// and, last, a unique worker id (`w0`, `w1`, …). The worker binary is expected
/// to open the same study and run a [`crate::distributed::Worker`] loop.
pub struct ProcessPool {
    program: OsString,
    base_args: Vec<OsString>,
    envs: Vec<(OsString, OsString)>,
    workers: usize,
    lease_ttl_ms: u64,
    max_restarts: usize,
}

impl ProcessPool {
    /// A pool that launches `program` as its worker binary.
    pub fn new(program: impl Into<OsString>) -> Self {
        ProcessPool {
            program: program.into(),
            base_args: Vec::new(),
            envs: Vec::new(),
            workers: 1,
            lease_ttl_ms: 30_000,
            max_restarts: 8,
        }
    }

    /// Append a base argument passed to every spawned worker (before its id).
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.base_args.push(arg.into());
        self
    }

    /// Set an environment variable on every spawned worker process.
    pub fn env(mut self, key: impl Into<OsString>, val: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), val.into()));
        self
    }

    /// Number of worker processes per round.
    pub fn workers(mut self, n: usize) -> Self {
        self.workers = n.max(1);
        self
    }

    /// The lease TTL the workers use; the pool waits past it before reclaiming a
    /// crashed worker's trials.
    pub fn lease_ttl_ms(mut self, ttl: u64) -> Self {
        self.lease_ttl_ms = ttl.max(1);
        self
    }

    /// Maximum restart rounds after the first, before giving up.
    pub fn max_restarts(mut self, n: usize) -> Self {
        self.max_restarts = n;
        self
    }

    /// Run supervised workers until the study drains or the restart budget is
    /// exhausted. Between rounds, orphaned leases (from crashed or exited
    /// workers) are recovered so their trials are retried.
    pub fn run(&self, storage: &Arc<dyn Storage>, study: StudyId) -> Result<ProcessReport> {
        let mut report = ProcessReport {
            rounds: 0,
            spawned: 0,
            crashed: 0,
            drained: false,
        };

        for round in 0..=self.max_restarts {
            if self.is_drained(storage, study)? {
                report.drained = true;
                break;
            }
            report.rounds = round + 1;

            // Spawn all workers for this round.
            let mut children = Vec::with_capacity(self.workers);
            for w in 0..self.workers {
                let mut cmd = Command::new(&self.program);
                cmd.args(&self.base_args).arg(format!("w{round}_{w}"));
                for (k, v) in &self.envs {
                    cmd.env(k, v);
                }
                match cmd.spawn() {
                    Ok(child) => {
                        report.spawned += 1;
                        children.push(child);
                    }
                    Err(e) => {
                        return Err(Error::Storage(format!(
                            "failed to spawn worker process: {e}"
                        )));
                    }
                }
            }

            // Wait for every worker to exit; a non-zero status is a crash.
            for mut child in children {
                match child.wait() {
                    Ok(status) if status.success() => {}
                    Ok(_) => report.crashed += 1,
                    Err(_) => report.crashed += 1,
                }
            }

            // No worker is alive now, so every outstanding lease belongs to a dead
            // process: reclaim them all by recovering as of a time past any lease.
            let horizon = now_ms().saturating_add(self.lease_ttl_ms).saturating_add(1);
            storage.recover_orphans(study, horizon)?;
        }

        report.drained = report.drained || self.is_drained(storage, study)?;
        Ok(report)
    }

    /// Whether every trial in the study has reached a terminal state.
    fn is_drained(&self, storage: &Arc<dyn Storage>, study: StudyId) -> Result<bool> {
        let history = storage.load_history(study)?;
        Ok(history.records().iter().all(|r| {
            matches!(
                r.state,
                TrialState::Complete
                    | TrialState::Pruned
                    | TrialState::Failed
                    | TrialState::Cancelled
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_defaults_are_sane() {
        // A pool over a program that will never be spawned in this unit test; we
        // only check the builder wiring here (integration test drives real spawns).
        let pool = ProcessPool::new("noop")
            .arg("a")
            .arg("b")
            .workers(3)
            .lease_ttl_ms(500)
            .max_restarts(4);
        assert_eq!(pool.workers, 3);
        assert_eq!(pool.lease_ttl_ms, 500);
        assert_eq!(pool.max_restarts, 4);
        assert_eq!(pool.base_args.len(), 2);
    }
}
