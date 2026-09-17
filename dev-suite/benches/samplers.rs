//! Criterion benchmark of end-to-end optimization throughput per sampler.
//!
//! Times a full `optimize_n` over the Ackley function for each sampler, so a
//! performance regression in the sampling/study loop shows up as a slower
//! benchmark. Run with:
//!
//! ```bash
//! cargo bench -p automl-core
//! ```

use automl_core::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Ackley in `dims` dimensions over `x0..xN`.
fn ackley(p: &ParamSet, dims: usize) -> f64 {
    let xs: Vec<f64> = (0..dims)
        .map(|i| p.float(&format!("x{i}")).unwrap())
        .collect();
    let n = dims as f64;
    let sum_sq: f64 = xs.iter().map(|v| v * v).sum();
    let sum_cos: f64 = xs
        .iter()
        .map(|v| (2.0 * std::f64::consts::PI * v).cos())
        .sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

fn space(dims: usize) -> SearchSpace {
    let mut s = SearchSpace::new();
    for i in 0..dims {
        s = s.add(format!("x{i}"), Distribution::float(-5.0, 5.0));
    }
    s
}

fn optimize(sampler: impl Sampler + 'static, dims: usize, n: u64) -> f64 {
    let mut study = Study::builder(space(dims))
        .minimize("loss")
        .sampler(sampler)
        .seed(1)
        .build()
        .unwrap();
    study
        .optimize_n(
            &|p: &ParamSet, _s: &mut dyn ReportSink| {
                Ok(NamedMetrics::single("loss", ackley(p, dims)))
            },
            n,
        )
        .unwrap();
    study
        .best_trial()
        .unwrap()
        .and_then(|t| t.final_value("loss"))
        .unwrap_or(f64::NAN)
}

fn bench_samplers(c: &mut Criterion) {
    let dims = 3;
    let n = 60;
    let mut group = c.benchmark_group("optimize_ackley_3d_60trials");
    group.bench_function("random", |b| {
        b.iter(|| optimize(RandomSampler::new(1), black_box(dims), black_box(n)))
    });
    group.bench_function("tpe", |b| {
        b.iter(|| {
            optimize(
                TpeSampler::new("loss", Direction::Minimize, 1),
                black_box(dims),
                black_box(n),
            )
        })
    });
    group.bench_function("tpe_multivariate", |b| {
        b.iter(|| {
            optimize(
                TpeSampler::new("loss", Direction::Minimize, 1).multivariate(true),
                black_box(dims),
                black_box(n),
            )
        })
    });
    group.bench_function("evolution", |b| {
        b.iter(|| {
            optimize(
                EvolutionarySampler::new("loss", Direction::Minimize, 1),
                black_box(dims),
                black_box(n),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_samplers);
criterion_main!(benches);
