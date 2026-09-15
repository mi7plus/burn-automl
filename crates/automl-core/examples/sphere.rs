//! End-to-end example: minimize the 2-D Sphere function, one of the standard
//! continuous-optimization benchmarks from the plan's benchmark suite (§24.1).
//!
//! Run with:
//! ```bash
//! cargo run -p automl-core --example sphere
//! ```

use automl_core::prelude::*;

fn main() -> Result<()> {
    // Sphere: f(x, y) = x^2 + y^2, minimized at the origin.
    let space = SearchSpace::new()
        .add("x", Distribution::float(-5.0, 5.0))
        .add("y", Distribution::float(-5.0, 5.0));

    let mut study = Study::builder(space)
        .name("sphere-2d")
        .minimize("loss")
        .sampler(RandomSampler::new(2024))
        .seed(2024)
        .build()?;

    let objective = |p: &ParamSet, _sink: &mut dyn ReportSink| -> Result<NamedMetrics> {
        let x = p.float("x")?;
        let y = p.float("y")?;
        Ok(NamedMetrics::single("loss", x * x + y * y))
    };

    let trials = study.optimize_n(&objective, 500)?;

    let best = study.best_trial()?.expect("a completed trial");
    println!("ran {trials} trials");
    println!(
        "best: x={:.4} y={:.4} loss={:.6}",
        best.params.float("x")?,
        best.params.float("y")?,
        best.final_value("loss").unwrap_or(f64::NAN),
    );

    println!("parameter importance:");
    for imp in study.importance()? {
        println!("  {:<4} {:.3}", imp.param, imp.importance);
    }
    Ok(())
}
