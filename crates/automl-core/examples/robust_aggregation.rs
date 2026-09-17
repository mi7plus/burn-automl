//! Robust aggregation for noisy objectives: `replicate` evaluates a noisy
//! objective several times under derived seeds and reduces the replicates to an
//! outlier-resistant estimate plus a spread (confidence proxy).
//!
//! ```bash
//! cargo run -p automl-core --release --example robust_aggregation
//! ```

use automl_core::objective::ReportSink;
use automl_core::prelude::*;
use automl_core::trial::TrialId;

/// A sink that ignores reports (we call `replicate` directly here).
struct NullSink;
impl ReportSink for NullSink {
    fn trial_id(&self) -> TrialId {
        TrialId(0)
    }
    fn report(&mut self, _s: u64, _m: NamedMetrics) -> Result<()> {
        Ok(())
    }
    fn should_stop(&self) -> bool {
        false
    }
}

fn main() {
    // A "noisy" objective: reward is 1.0 or 3.0 depending on the seed parity, plus
    // a rare huge outlier — median shrugs it off, mean does not.
    let eval = |seed: u64, _s: &mut dyn ReportSink| {
        let reward = if seed == 3 {
            100.0
        } else if seed % 2 == 1 {
            1.0
        } else {
            3.0
        };
        Ok(NamedMetrics::single("reward", reward))
    };

    for agg in [
        Aggregator::Mean,
        Aggregator::Median,
        Aggregator::TrimmedMean(0.2),
    ] {
        let mut sink = NullSink;
        let out = replicate(agg, 5, 1, "reward", &mut sink, eval).unwrap();
        println!(
            "{:<20} reward = {:.3}  (spread {:.3})",
            format!("{agg:?}"),
            out.get("reward").unwrap_or(f64::NAN),
            out.get("reward.spread").unwrap_or(f64::NAN),
        );
    }
}
