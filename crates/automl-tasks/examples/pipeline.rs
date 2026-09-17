//! Pipeline AutoML with `AutoPipeline`: jointly searches a preprocessing
//! transform (standardize / normalize / none) and a classifier (nearest-centroid
//! or k-NN). The data hides the signal in a small-scale feature next to a
//! large-scale noise feature, so only a scaling pipeline recovers it.
//!
//! ```bash
//! cargo run -p automl-tasks --release --example pipeline
//! ```

use automl_tasks::AutoPipeline;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..200 {
        let label = rng.gen_range(0..2);
        let signal = if label == 0 {
            rng.gen_range(-1.0..0.0)
        } else {
            rng.gen_range(0.0..1.0)
        };
        x.push(vec![rng.gen_range(-500.0f32..500.0), signal]);
        y.push(label);
    }
    let study = AutoPipeline::new(x, y).trials(16).seed(1).fit().unwrap();
    let best = study.best_trial().unwrap().unwrap();
    let plan = AutoPipeline::space().decode(&best.params).unwrap();
    println!(
        "best pipeline: {} — accuracy {:.1}%",
        plan.describe(),
        best.final_value("accuracy").unwrap_or(0.0),
    );
}
