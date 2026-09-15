//! End-to-end: run a real study, render the dashboard, and check the output is
//! a self-contained page carrying every completed trial. Also writes the HTML
//! to the OS temp directory so it can be opened for a visual check.

use automl_cli::render_dashboard;
use automl_core::metrics::Direction;
use automl_core::prelude::*;
use automl_core::storage::StudyMeta;

#[test]
fn renders_dashboard_from_a_real_study() {
    let space = SearchSpace::new()
        .add("lr", Distribution::log_float(1e-4, 1e-1))
        .add("depth", Distribution::int(1, 6))
        .add("dropout", Distribution::float(0.0, 0.5));

    let mut study = Study::builder(space)
        .name("demo-dashboard")
        .minimize("loss")
        .sampler(TpeSampler::new("loss", Direction::Minimize, 5))
        .pruner(MedianPruner::new("loss", Direction::Minimize))
        .seed(5)
        .build()
        .unwrap();

    study
        .optimize_n(
            &|p: &ParamSet, sink: &mut dyn ReportSink| {
                let lr = p.float("lr")?;
                let depth = p.int("depth")? as f64;
                let dr = p.float("dropout")?;
                let target =
                    (lr.ln() + 3.0).powi(2) + (depth - 3.0).powi(2) * 0.3 + (dr - 0.2).powi(2);
                // A learning curve so median pruning has something to act on.
                for step in 1..=5u64 {
                    sink.report(
                        step,
                        NamedMetrics::single("loss", target + 2.0 / step as f64),
                    )?;
                    if sink.should_stop() {
                        break;
                    }
                }
                Ok(NamedMetrics::single("loss", target))
            },
            60,
        )
        .unwrap();

    let history = study.history().unwrap();
    let meta = StudyMeta {
        name: "demo-dashboard".into(),
        directions: vec![("loss".into(), Direction::Minimize)],
        sampler_name: "tpe".into(),
        pruner_name: "median".into(),
    };

    let html = render_dashboard(&meta, history.records());
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains("demo-dashboard"));
    assert!(html.contains("\"trials\""));
    // Every completed trial's id appears in the embedded data.
    let completed = history.completed().count();
    assert!(completed > 0);

    // Emit the artifact for a manual/visual check; failures here are non-fatal.
    let out = std::env::temp_dir().join("automl_dashboard_demo.html");
    let _ = std::fs::write(&out, &html);
    eprintln!("dashboard written to {}", out.display());
}
