//! Reinforcement learning with `AutoRl`: tabular Q-learning on a slippery
//! corridor, searching learning rate / discount / exploration to maximize the
//! robust (median) episode return over noisy rollouts.
//!
//! ```bash
//! cargo run -p automl-tasks --release --example reinforcement_learning
//! ```

use automl_tasks::{AutoRl, GridWorld};

fn main() {
    let env = GridWorld::new(5, 0.1); // 5-cell corridor, 10% slip
    let study = AutoRl::new(env)
        .train_episodes(200)
        .eval_episodes(15)
        .replicates(3)
        .trials(12)
        .seed(7)
        .fit()
        .unwrap();
    let best = study.best_trial().unwrap().unwrap();
    println!(
        "best robust return {:.2}  (alpha={:.3}, gamma={:.3}, epsilon={:.3})",
        best.final_value("return").unwrap_or(f64::NAN),
        best.params.float("alpha").unwrap_or(0.0),
        best.params.float("gamma").unwrap_or(0.0),
        best.params.float("epsilon").unwrap_or(0.0),
    );
}
