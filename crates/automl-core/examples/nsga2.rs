//! Multi-objective optimization with the NSGA-II sampler.
//!
//! Optimizes the two-objective ZDT1 problem, whose Pareto front is a curve. The
//! NSGA-II sampler evolves a population toward the whole front and spreads
//! solutions along it. We print the resulting front and its hypervolume, and
//! compare against random search over the same budget.
//!
//! ```bash
//! cargo run -p automl-core --release --example nsga2
//! ```

use automl_core::prelude::*;

fn zdt1(p: &ParamSet, _s: &mut dyn ReportSink) -> Result<NamedMetrics> {
    let x = p.float("x")?;
    let y = p.float("y")?;
    let g = 1.0 + 9.0 * y;
    let f2 = g * (1.0 - (x / g).sqrt());
    Ok(NamedMetrics::new().with("f1", x).with("f2", f2))
}

fn run(sampler: impl Sampler + 'static) -> Study {
    let space = SearchSpace::new()
        .add("x", Distribution::float(0.0, 1.0))
        .add("y", Distribution::float(0.0, 1.0));
    let mut study = Study::builder(space)
        .minimize("f1")
        .minimize("f2")
        .sampler(sampler)
        .seed(7)
        .build()
        .unwrap();
    study.optimize_n(&zdt1, 150).unwrap();
    study
}

fn main() {
    let reference = NamedMetrics::new().with("f1", 1.1).with("f2", 11.0);
    let objectives = vec![
        ("f1".to_string(), Direction::Minimize),
        ("f2".to_string(), Direction::Minimize),
    ];

    let nsga = run(Nsga2Sampler::new(objectives, 7));
    let random = run(RandomSampler::new(7));

    let front = nsga.pareto_front().unwrap();
    println!("NSGA-II Pareto front ({} points):", front.len());
    // Member.values are aligned to the front's objective order: [f1, f2].
    let mut pts: Vec<(f64, f64)> = front
        .members()
        .iter()
        .filter_map(|m| Some((*m.values.first()?, *m.values.get(1)?)))
        .collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (f1, f2) in pts.iter().take(10) {
        println!("  f1={f1:.3}  f2={f2:.3}");
    }
    if pts.len() > 10 {
        println!("  … {} more", pts.len() - 10);
    }

    println!(
        "\nhypervolume  —  NSGA-II: {:.3}   random: {:.3}",
        nsga.hypervolume(&reference).unwrap(),
        random.hypervolume(&reference).unwrap(),
    );
}
