//! End-to-end AutoML over a real Burn model: search MLP hyperparameters to
//! maximize MNIST validation accuracy, with TPE sampling and median pruning of
//! weak trials by their learning curve.
//!
//! Running this **downloads the MNIST dataset** on first use and trains on the
//! CPU, so it takes a few minutes.
//!
//! ```bash
//! cargo run -p automl-burn --release --example mnist_search
//! ```

use automl_burn::MnistMlpObjective;
use automl_core::prelude::*;

fn main() -> Result<()> {
    let objective = MnistMlpObjective {
        epochs: 3,
        batch_size: 64,
        train_size: 4000,
        test_size: 1000,
    };

    let mut study = Study::builder(MnistMlpObjective::search_space())
        .name("mnist-mlp")
        .maximize("accuracy")
        .sampler(TpeSampler::new("accuracy", Direction::Maximize, 7))
        .pruner(MedianPruner::new("accuracy", Direction::Maximize).with_warmup_steps(1))
        .seed(7)
        .build()?;

    let n_trials = 15;
    println!("searching {n_trials} MLP configurations on MNIST...");
    study.optimize_n(&objective, n_trials)?;

    let best = study.best_trial()?.expect("a completed trial");
    println!("\nbest configuration:");
    println!("  lr          = {:.5}", best.params.float("lr")?);
    println!("  hidden_size = {}", best.params.int("hidden_size")?);
    println!("  num_layers  = {}", best.params.int("num_layers")?);
    println!("  dropout     = {:.3}", best.params.float("dropout")?);
    println!(
        "  accuracy    = {:.2}%",
        best.final_value("accuracy").unwrap_or(f64::NAN)
    );

    println!("\nhyperparameter importance:");
    for imp in study.importance()? {
        println!("  {:<12} {:.3}", imp.param, imp.importance);
    }
    Ok(())
}
