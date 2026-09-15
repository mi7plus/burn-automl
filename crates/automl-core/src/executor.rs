//! Executors: where trials actually run (PRD §4.1, §18.1, §29 item 8).
//!
//! The plan defines four executor tiers (§18.1). v0.1 ships the two in-process
//! tiers:
//!
//! - [`SequentialExecutor`] — single-threaded, deterministic; the default and
//!   the tier used for debugging and CI.
//! - [`ThreadExecutor`] — a fixed-size thread pool for cheap CPU-bound
//!   objectives.
//!
//! The process and distributed tiers (crash isolation, multi-node worker pools
//! with the lease protocol of §18.2) arrive in later releases.
//!
//! ## Design note — batch API vs. the handle API
//!
//! The plan sketches a handle-based `Executor::spawn(spec, objective) ->
//! TrialHandle` signature aimed at the distributed tier. For the in-process
//! tiers a simpler *batch* API is sufficient and keeps the ask-tell loop
//! honest: the [`Study`](crate::study::Study) asks for a batch of parameter
//! sets, enqueues each as a running trial (so successive `suggest` calls see a
//! growing history and stay diverse), then hands the executor a batch of
//! self-contained jobs to run. The handle/lease API is introduced alongside the
//! distributed tier where cancellation and orphan recovery actually need it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A unit of work the executor runs: a self-contained closure that evaluates
/// one trial and writes its result to storage. Jobs never touch the sampler, so
/// they are independent and safe to run concurrently.
pub type Job<'a> = Box<dyn FnOnce() + Send + 'a>;

/// CPU, GPU, memory and custom resource declarations (PRD §4.2 `ResourceSpec`).
///
/// Used today to describe executor capacity; admission control against these
/// declarations (§18.3) arrives with the distributed tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    /// Number of CPU cores (fractional allowed for shared cores).
    pub cpus: f64,
    /// Number of GPUs (fractional for MPS-style sharing).
    pub gpus: f64,
    /// Memory in megabytes.
    pub memory_mb: u64,
    /// Arbitrary named resources (e.g. `"tpu"`, `"disk_gb"`).
    pub custom: BTreeMap<String, f64>,
}

impl Default for ResourceSpec {
    fn default() -> Self {
        ResourceSpec {
            cpus: 1.0,
            gpus: 0.0,
            memory_mb: 0,
            custom: BTreeMap::new(),
        }
    }
}

impl ResourceSpec {
    /// A spec requesting `cpus` CPU cores and nothing else.
    pub fn cpus(cpus: f64) -> Self {
        ResourceSpec {
            cpus,
            ..Default::default()
        }
    }

    /// Add a GPU request.
    pub fn with_gpus(mut self, gpus: f64) -> Self {
        self.gpus = gpus;
        self
    }

    /// Add a memory request in megabytes.
    pub fn with_memory_mb(mut self, memory_mb: u64) -> Self {
        self.memory_mb = memory_mb;
        self
    }
}

/// Runs batches of trial jobs, possibly concurrently (PRD §4.1).
pub trait Executor: Send + Sync {
    /// Run every job in the batch, blocking until all have finished. The order
    /// in which jobs run is unspecified, but every job is guaranteed to have
    /// completed when this returns.
    fn run_all(&self, jobs: Vec<Job<'_>>);

    /// The executor's resource capacity, used for scheduling decisions.
    fn capacity(&self) -> ResourceSpec;

    /// How many jobs this executor prefers to receive per batch. The study uses
    /// it as the ask-batch size; `1` yields the deterministic sequential path.
    fn preferred_batch_size(&self) -> usize;

    /// A short, stable name for provenance and logging.
    fn name(&self) -> &'static str;
}

/// In-process, single-threaded executor. Deterministic; the default tier.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialExecutor;

impl Executor for SequentialExecutor {
    fn run_all(&self, jobs: Vec<Job<'_>>) {
        for job in jobs {
            job();
        }
    }

    fn capacity(&self) -> ResourceSpec {
        ResourceSpec::cpus(1.0)
    }

    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn name(&self) -> &'static str {
        "sequential"
    }
}

/// In-process, fixed-size thread pool for cheap CPU-bound objectives.
///
/// Jobs are distributed across at most `max_threads` OS threads via a scoped
/// spawn, so no `'static` bound is imposed on the captured objective or storage
/// — they only need to outlive the `run_all` call.
#[derive(Debug, Clone, Copy)]
pub struct ThreadExecutor {
    max_threads: usize,
}

impl ThreadExecutor {
    /// A thread executor using up to `max_threads` worker threads (clamped to
    /// at least 1).
    pub fn new(max_threads: usize) -> Self {
        ThreadExecutor {
            max_threads: max_threads.max(1),
        }
    }

    /// A thread executor sized to the machine's available parallelism.
    pub fn with_available_parallelism() -> Self {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        ThreadExecutor::new(n)
    }
}

impl Executor for ThreadExecutor {
    fn run_all(&self, jobs: Vec<Job<'_>>) {
        if jobs.is_empty() {
            return;
        }
        if self.max_threads == 1 || jobs.len() == 1 {
            for job in jobs {
                job();
            }
            return;
        }

        // Round-robin the jobs into up to max_threads buckets, then run each
        // bucket on its own scoped thread. Scoped threads let jobs borrow
        // non-'static state (the objective, storage) for the duration.
        let n_threads = self.max_threads.min(jobs.len());
        let mut buckets: Vec<Vec<Job<'_>>> = (0..n_threads).map(|_| Vec::new()).collect();
        for (i, job) in jobs.into_iter().enumerate() {
            buckets[i % n_threads].push(job);
        }

        std::thread::scope(|scope| {
            for bucket in buckets {
                scope.spawn(move || {
                    for job in bucket {
                        job();
                    }
                });
            }
        });
    }

    fn capacity(&self) -> ResourceSpec {
        ResourceSpec::cpus(self.max_threads as f64)
    }

    fn preferred_batch_size(&self) -> usize {
        self.max_threads
    }

    fn name(&self) -> &'static str {
        "thread"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[test]
    fn sequential_runs_all_in_order() {
        let order = Mutex::new(Vec::new());
        let jobs: Vec<Job> = (0..5)
            .map(|i| {
                let order = &order;
                Box::new(move || order.lock().unwrap().push(i)) as Job
            })
            .collect();
        SequentialExecutor.run_all(jobs);
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn thread_executor_runs_every_job() {
        let counter = AtomicUsize::new(0);
        let jobs: Vec<Job> = (0..100)
            .map(|_| {
                let counter = &counter;
                Box::new(move || {
                    counter.fetch_add(1, Ordering::SeqCst);
                }) as Job
            })
            .collect();
        ThreadExecutor::new(4).run_all(jobs);
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn thread_executor_borrows_non_static_state() {
        // A stack-local the jobs borrow; scoped threads must keep it valid.
        let data = [1u64, 2, 3, 4];
        let sum = AtomicUsize::new(0);
        let jobs: Vec<Job> = data
            .iter()
            .map(|&v| {
                let sum = &sum;
                Box::new(move || {
                    sum.fetch_add(v as usize, Ordering::SeqCst);
                }) as Job
            })
            .collect();
        ThreadExecutor::new(3).run_all(jobs);
        assert_eq!(sum.load(Ordering::SeqCst), 10);
    }
}
