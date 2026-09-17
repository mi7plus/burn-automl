//! Multi-fidelity optimization with Hyperband / BOHB.
//!
//! A fidelity-aware objective evaluates a configuration at a given resource
//! (here, "epochs"): cheap evaluations are noisy, expensive ones accurate.
//! Hyperband spends most of its budget on cheap evaluations and only trains the
//! survivors at full resource. Pairing it with a TPE sampler makes it BOHB.
//!
//! ```bash
//! cargo run -p automl-core --release --example hyperband
//! ```

use automl_core::prelude::*;
use rand::{Rng, SeedableRng};

fn main() {
    // True objective: minimize (x - 2)^2 + (y + 1)^2. A low-resource evaluation
    // adds noise that shrinks as resource grows.
    let objective = |p: &ParamSet, resource: u64| {
        let x = p.float("x").unwrap();
        let y = p.float("y").unwrap();
        let truth = (x - 2.0).powi(2) + (y + 1.0).powi(2);
        let key = (x * 1e3) as i64 as u64 ^ (y * 1e3) as i64 as u64 ^ resource;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(key);
        let noise = rng.gen_range(-1.0..1.0) / resource as f64;
        NamedMetrics::single("loss", truth + noise)
    };

    let space = SearchSpace::new()
        .add("x", Distribution::float(-5.0, 5.0))
        .add("y", Distribution::float(-5.0, 5.0));

    let hb = Hyperband::new("loss", Direction::Minimize, 27, 3);

    // Random sampling = Hyperband; TPE sampling = BOHB.
    for (label, mut sampler) in [
        (
            "Hyperband (random)",
            Box::new(RandomSampler::new(1)) as Box<dyn Sampler>,
        ),
        (
            "BOHB (TPE)",
            Box::new(TpeSampler::new("loss", Direction::Minimize, 1)),
        ),
    ] {
        let out = hb.optimize(&space, sampler.as_mut(), objective);
        println!(
            "{label:22} best loss {:.4} at ({:.3}, {:.3}) — {} evals, {} total resource",
            out.best_score,
            out.best_params.float("x").unwrap_or(f64::NAN),
            out.best_params.float("y").unwrap_or(f64::NAN),
            out.evaluations,
            out.total_resource,
        );
    }
    println!("(optimum is loss 0 at x=2, y=-1)");
}
