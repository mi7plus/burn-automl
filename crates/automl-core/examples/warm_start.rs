//! Warm-starting a study by transferring the best configurations of a prior one.
//!
//! A first study optimizes a function from scratch. A second, *related* study is
//! seeded with the first study's best configurations via `WarmStartSampler`, so
//! it starts near known-good regions instead of exploring blindly.
//!
//! ```bash
//! cargo run -p automl-core --release --example warm_start
//! ```

use automl_core::prelude::*;

fn objective(p: &ParamSet, _s: &mut dyn ReportSink) -> Result<NamedMetrics> {
    let x = p.float("x")?;
    let y = p.float("y")?;
    Ok(NamedMetrics::single(
        "loss",
        (x - 2.0).powi(2) + (y + 1.0).powi(2),
    ))
}

fn run(sampler: impl Sampler + 'static, n: u64) -> Study {
    let space = SearchSpace::new()
        .add("x", Distribution::float(-5.0, 5.0))
        .add("y", Distribution::float(-5.0, 5.0));
    let mut study = Study::builder(space)
        .minimize("loss")
        .sampler(sampler)
        .seed(1)
        .build()
        .unwrap();
    study.optimize_n(&objective, n).unwrap();
    study
}

fn best_loss(study: &Study) -> f64 {
    study
        .best_trial()
        .unwrap()
        .and_then(|t| t.final_value("loss"))
        .unwrap_or(f64::NAN)
}

fn main() {
    // Prior study: optimize from scratch.
    let prior = run(TpeSampler::new("loss", Direction::Minimize, 1), 40);
    println!("prior study best loss: {:.4}", best_loss(&prior));

    // Transfer its 5 best configurations into a fresh study.
    let seeds = best_configs(&prior.history().unwrap(), "loss", Direction::Minimize, 5);
    let warm = run(
        WarmStartSampler::new(TpeSampler::new("loss", Direction::Minimize, 2), seeds),
        10,
    );

    // Compare against a cold study with the same tiny budget.
    let cold = run(TpeSampler::new("loss", Direction::Minimize, 2), 10);

    println!(
        "warm-started (10 trials) best loss: {:.4}",
        best_loss(&warm)
    );
    println!(
        "cold start   (10 trials) best loss: {:.4}",
        best_loss(&cold)
    );
    println!("(warm-started should be at least as good, having reused good regions)");
}
