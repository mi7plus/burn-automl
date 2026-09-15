//! Standard continuous-optimization benchmark functions (PRD §24.1) and a
//! Random vs TPE vs Evolutionary comparison over them.
//!
//! Reports mean best objective and solve-rate per sampler over repeated seeds,
//! the metrics the plan's benchmark suite calls for. Run with:
//! ```bash
//! cargo run -p automl-core --release --example benchmarks
//! ```

use automl_core::prelude::*;

/// A benchmark: a search space plus a function to minimize, with a known
/// optimum value and a "solved" threshold.
struct Benchmark {
    name: &'static str,
    dims: usize,
    low: f64,
    high: f64,
    threshold: f64,
    f: fn(&[f64]) -> f64,
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

fn rosenbrock(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|v| v * v).sum();
    let sum_cos: f64 = x
        .iter()
        .map(|v| (2.0 * std::f64::consts::PI * v).cos())
        .sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

fn rastrigin(x: &[f64]) -> f64 {
    let a = 10.0;
    a * x.len() as f64
        + x.iter()
            .map(|v| v * v - a * (2.0 * std::f64::consts::PI * v).cos())
            .sum::<f64>()
}

fn himmelblau(x: &[f64]) -> f64 {
    // 2-D only.
    (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2)
}

/// Read the trial's `x0..xN` parameters into a vector.
fn read_point(p: &ParamSet, dims: usize) -> Result<Vec<f64>> {
    (0..dims).map(|i| p.float(&format!("x{i}"))).collect()
}

/// Run one benchmark with one sampler; return (best value, trials-to-threshold).
fn run(
    bench: &Benchmark,
    sampler: impl Sampler + 'static,
    seed: u64,
    n_trials: u64,
) -> (f64, Option<u64>) {
    let mut space = SearchSpace::new();
    for i in 0..bench.dims {
        space = space.add(format!("x{i}"), Distribution::float(bench.low, bench.high));
    }
    let f = bench.f;
    let dims = bench.dims;
    let mut study = Study::builder(space)
        .name(bench.name)
        .minimize("loss")
        .sampler(sampler)
        .seed(seed)
        .build()
        .expect("valid study");

    study
        .optimize_n(
            &move |p: &ParamSet, _s: &mut dyn ReportSink| {
                Ok(NamedMetrics::single("loss", f(&read_point(p, dims)?)))
            },
            n_trials,
        )
        .expect("optimize");

    let history = study.history().expect("history");
    let mut best = f64::INFINITY;
    let mut solved_at = None;
    for (i, rec) in history.records().iter().enumerate() {
        if let Some(v) = rec.final_value("loss") {
            if v < best {
                best = v;
            }
            if solved_at.is_none() && best <= bench.threshold {
                solved_at = Some(i as u64 + 1);
            }
        }
    }
    (best, solved_at)
}

fn main() {
    let benches = [
        Benchmark {
            name: "sphere-4d",
            dims: 4,
            low: -5.0,
            high: 5.0,
            threshold: 1e-2,
            f: sphere,
        },
        Benchmark {
            name: "rosenbrock-2d",
            dims: 2,
            low: -2.0,
            high: 2.0,
            threshold: 1.0,
            f: rosenbrock,
        },
        Benchmark {
            name: "ackley-3d",
            dims: 3,
            low: -5.0,
            high: 5.0,
            threshold: 1.0,
            f: ackley,
        },
        Benchmark {
            name: "rastrigin-2d",
            dims: 2,
            low: -5.12,
            high: 5.12,
            threshold: 2.0,
            f: rastrigin,
        },
        Benchmark {
            name: "himmelblau",
            dims: 2,
            low: -5.0,
            high: 5.0,
            threshold: 1e-2,
            f: himmelblau,
        },
    ];

    let n_trials = 300;
    let seeds: Vec<u64> = (0..8).collect();

    // Mean best objective and solve-rate over repeated seeds (§24.1).
    #[derive(Clone, Copy)]
    enum Which {
        Random,
        Tpe,
        Evo,
    }
    let bench_sampler = |b: &Benchmark, which: Which| -> (f64, f64) {
        let mut best_sum = 0.0;
        let mut solved = 0usize;
        for &s in &seeds {
            let (best, solve) = match which {
                Which::Random => run(b, RandomSampler::new(s), s, n_trials),
                Which::Tpe => run(
                    b,
                    TpeSampler::new("loss", Direction::Minimize, s),
                    s,
                    n_trials,
                ),
                Which::Evo => run(
                    b,
                    EvolutionarySampler::new("loss", Direction::Minimize, s),
                    s,
                    n_trials,
                ),
            };
            best_sum += best;
            if solve.is_some() {
                solved += 1;
            }
        }
        (
            best_sum / seeds.len() as f64,
            solved as f64 / seeds.len() as f64,
        )
    };

    println!("mean over {} seeds, {n_trials} trials each\n", seeds.len());
    println!(
        "{:<16} {:>11} {:>7} {:>11} {:>7} {:>11} {:>7}",
        "benchmark", "random", "solve%", "tpe", "solve%", "evolution", "solve%"
    );
    println!("{}", "-".repeat(76));
    for b in &benches {
        let (rb, rs) = bench_sampler(b, Which::Random);
        let (tb, ts) = bench_sampler(b, Which::Tpe);
        let (eb, es) = bench_sampler(b, Which::Evo);
        println!(
            "{:<16} {:>11.4} {:>6.0}% {:>11.4} {:>6.0}% {:>11.4} {:>6.0}%",
            b.name,
            rb,
            rs * 100.0,
            tb,
            ts * 100.0,
            eb,
            es * 100.0
        );
    }
}
