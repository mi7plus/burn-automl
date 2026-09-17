# Sampler benchmark report

The optimization benchmark the PRD requires (§24.1): the standard continuous test
functions, each optimized by every sampler over repeated seeds, reporting **mean
best objective** (lower is better) and **solve-rate** (fraction of seeds that
reached the function's threshold). Reproduce with:

```bash
cargo run -p automl-core --release --example benchmarks
```

## Setup

- **300 trials** per run, **8 seeds** per (sampler, benchmark) cell, means reported.
- Samplers: `Random`, `TPE` (independent per-dimension), `TPE-mv` (multivariate /
  joint, `TpeSampler::multivariate(true)`), `Evolutionary` (real-coded GA).
- Benchmarks and solve thresholds: sphere-4d (1e-2), rosenbrock-2d (1.0),
  ackley-3d (1.0), rastrigin-2d (2.0), himmelblau (1e-2).

## Results (mean best objective, and solve-rate)

| Benchmark | Random | Random slv% | TPE | TPE slv% | **TPE-mv** | **TPE-mv slv%** | Evolution | Evo slv% |
|-----------|-------:|-----:|----:|-----:|-------:|-----:|----------:|-----:|
| sphere-4d | 3.2998 | 0% | 2.4786 | 0% | **0.2508** | 0% | 0.2504 | 0% |
| rosenbrock-2d | 0.1540 | 100% | 0.4823 | 88% | **0.1404** | **100%** | 0.6482 | 88% |
| ackley-3d | 3.2461 | 12% | 2.2521 | 38% | **1.4313** | 25% | 1.5715 | 25% |
| rastrigin-2d | 3.3954 | 25% | 3.0751 | 62% | **0.9351** | **88%** | 1.4326 | 75% |
| himmelblau | 0.5337 | 0% | 1.3675 | 38% | **0.0812** | 12% | 0.4064 | 0% |

(Bold = best mean best-objective in the row.)

## Reading it

- **Multivariate TPE is the strongest sampler overall** — best or tied-best mean
  on 4 of 5 functions, and the best solve-rate on rastrigin (88%). Anchoring each
  candidate on a good trial and perturbing all dimensions together lets it follow
  correlations that independent-per-dimension TPE cannot.
- **The multivariate fix is decisive on coupled and high-dimension problems.** On
  sphere-4d it improves plain TPE ~10× (0.25 vs 2.48) and matches the evolutionary
  sampler; on rosenbrock's curved valley it beats plain TPE (0.14 vs 0.48) and
  reaches 100% solve; on rastrigin it more than triples the improvement over TPE.
- **Independent TPE still helps ranking-dominated multimodal problems** (ackley,
  rastrigin solve-rate) over random, but its per-dimension independence hurts it on
  coupled (rosenbrock) and separable-but-additive (sphere) landscapes — exactly the
  gap multivariate mode closes.
- **Random is a genuine baseline, not a strawman**: on low-dimension smooth basins
  (rosenbrock-2d, himmelblau) 300 random samples already do well, which is why a
  sampler has to *beat* it, not just run.

## Recommendation

For continuous or mixed spaces where parameters interact, prefer
`TpeSampler::new(...).multivariate(true)`. Keep the evolutionary sampler for very
rugged landscapes, and independent TPE where dimensions are genuinely separable and
cheap. All samplers are correct over conditional/hierarchical spaces; multivariate
mode preserves the per-branch §5.1 fallback.
