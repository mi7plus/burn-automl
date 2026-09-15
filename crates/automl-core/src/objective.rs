//! Objectives, report sinks, and the `TaskAdapter` trait.
//!
//! The load-bearing invariant from PRD §4.2: *any workload that can produce
//! named metrics can be optimized*, and every downstream `Auto*` API is just a
//! [`TaskAdapter`] plus a pre-built [`SearchSpace`]. This module defines the
//! thin seam between the generic engine and a concrete workload:
//!
//! - [`ReportSink`] — how a running trial streams intermediate metrics back and
//!   asks whether it has been pruned/cancelled (the poll point from §18.2).
//! - [`Objective`] — the simplest workload: a closure over parameters.
//! - [`TaskAdapter`] — a workload that also owns its search space, so an
//!   `Auto*` builder can hand the engine both at once.

use crate::error::Result;
use crate::metrics::NamedMetrics;
use crate::param::ParamSet;
use crate::space::SearchSpace;
use crate::trial::TrialId;

/// A callback a running objective uses to stream intermediate metrics and to
/// check whether the study has asked it to stop.
///
/// Objectives should call [`ReportSink::report`] at each checkpointing step and
/// then consult [`ReportSink::should_stop`] — this is the cancellation poll
/// point the distributed protocol requires workers to honor (§18.2), surfaced
/// here so single-process objectives use the exact same contract.
pub trait ReportSink {
    /// The trial currently being evaluated.
    fn trial_id(&self) -> TrialId;

    /// Report intermediate metrics at `step`. Idempotent per `(trial, step)`.
    fn report(&mut self, step: u64, metrics: NamedMetrics) -> Result<()>;

    /// Whether the trial should stop now (pruned or cancelled). Objectives are
    /// expected to poll this at least once per checkpointing interval.
    fn should_stop(&self) -> bool;
}

/// The simplest optimizable workload: a function from parameters to metrics.
///
/// The closure receives a [`ReportSink`] so it can report intermediate values
/// and honor pruning even in a single call. Objectives that don't produce a
/// learning curve can ignore the sink and just return final metrics.
pub trait Objective: Send + Sync {
    /// Evaluate the objective for one parameter assignment.
    fn evaluate(&self, params: &ParamSet, report: &mut dyn ReportSink) -> Result<NamedMetrics>;
}

/// Blanket impl so a plain closure is an [`Objective`].
impl<F> Objective for F
where
    F: Fn(&ParamSet, &mut dyn ReportSink) -> Result<NamedMetrics> + Send + Sync,
{
    fn evaluate(&self, params: &ParamSet, report: &mut dyn ReportSink) -> Result<NamedMetrics> {
        (self)(params, report)
    }
}

/// A workload that owns its search space (PRD §4.1 `TaskAdapter<Ctx>`).
///
/// `Ctx` carries whatever shared, read-only context a run needs (a dataset
/// handle, a device, an environment factory) without threading it through the
/// engine. High-level `Auto*` APIs are thin builders over an adapter plus its
/// pre-populated space.
pub trait TaskAdapter<Ctx>: Send + Sync {
    /// Build the search space this adapter optimizes over.
    fn build_space(&self) -> SearchSpace;

    /// Run one trial for a parameter assignment against the shared context.
    fn run(
        &self,
        params: &ParamSet,
        ctx: &Ctx,
        report: &mut dyn ReportSink,
    ) -> Result<NamedMetrics>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NamedMetrics;
    use crate::param::ParamValue;
    use crate::trial::TrialId;

    struct NullSink(TrialId);
    impl ReportSink for NullSink {
        fn trial_id(&self) -> TrialId {
            self.0
        }
        fn report(&mut self, _step: u64, _metrics: NamedMetrics) -> Result<()> {
            Ok(())
        }
        fn should_stop(&self) -> bool {
            false
        }
    }

    #[test]
    fn closure_is_an_objective() {
        let obj = |p: &ParamSet, _r: &mut dyn ReportSink| {
            let x = p.float("x").unwrap();
            Ok(NamedMetrics::single("y", x * x))
        };
        let mut params = ParamSet::new();
        params.insert("x", ParamValue::Float(3.0));
        let mut sink = NullSink(TrialId(0));
        let m = Objective::evaluate(&obj, &params, &mut sink).unwrap();
        assert_eq!(m.get("y"), Some(9.0));
    }
}
