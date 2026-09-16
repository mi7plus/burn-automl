//! `automl` — command-line tools for burn-automl.
//!
//! ```text
//! automl list <study.db>
//! automl dashboard <study.db> [--study <id>] [-o <out.html>]
//! ```
//!
//! Both commands are read-only against a SQLite study store (§25): they never
//! write to the database, only read it.

use automl_cli::{render_dashboard, summarize};
use automl_core::distributed::{Poll, Worker};
use automl_core::metrics::NamedMetrics;
use automl_core::objective::ReportSink;
use automl_core::param::ParamSet;
use automl_core::sqlite::SqliteStorage;
use automl_core::storage::Storage;
use automl_core::trial::StudyId;
use std::process::ExitCode;
use std::sync::Arc;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            eprintln!("\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage:
  automl list <study.db>
  automl dashboard <study.db> [--study <id>] [-o <out.html>]
  automl worker <study.db> <study-id> <lease-ttl-ms> <worker-id>";

fn run(args: &[String]) -> Result<(), String> {
    let command = args.first().map(String::as_str).ok_or("no command given")?;
    match command {
        "list" => cmd_list(&args[1..]),
        "dashboard" => cmd_dashboard(&args[1..]),
        "worker" => cmd_worker(&args[1..]),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    }
}

/// Run a crash-isolated worker against a persisted study (the process-executor
/// tier, PRD §18). This worker uses a built-in 2-D benchmark objective over
/// params `x`/`y`; real deployments ship their own worker binary with their own
/// objective, following the same `distributed::Worker` loop. Set
/// `AUTOML_WORKER_CRASH_AFTER=<n>` to abort after completing `n` trials, which
/// exercises orphan recovery.
fn cmd_worker(args: &[String]) -> Result<(), String> {
    let db = args.first().ok_or("missing <study.db>")?;
    let study_id: u64 = args
        .get(1)
        .ok_or("missing <study-id>")?
        .parse()
        .map_err(|_| "bad study id")?;
    let ttl: u64 = args
        .get(2)
        .ok_or("missing <lease-ttl-ms>")?
        .parse()
        .map_err(|_| "bad lease ttl")?;
    let worker_id = args.get(3).cloned().unwrap_or_else(|| "w0".into());
    let crash_after: Option<usize> = std::env::var("AUTOML_WORKER_CRASH_AFTER")
        .ok()
        .and_then(|v| v.parse().ok());

    let storage: Arc<dyn Storage> = Arc::new(open(db)?);
    let study = StudyId(study_id);
    let worker = Worker::new(worker_id, ttl);

    // A deterministic 2-D benchmark: minimize (x-2)^2 + (y+1)^2.
    let objective = |p: &ParamSet, _s: &mut dyn ReportSink| {
        let x = p.float("x")?;
        let y = p.float("y")?;
        Ok(NamedMetrics::single(
            "loss",
            (x - 2.0).powi(2) + (y + 1.0).powi(2),
        ))
    };

    let mut ran = 0usize;
    loop {
        match worker
            .poll(&storage, study, &objective)
            .map_err(|e| e.to_string())?
        {
            Poll::Idle => break,
            Poll::Ran(_) => {
                ran += 1;
                if crash_after == Some(ran) {
                    // Simulate a hard crash mid-run: exit non-zero without draining.
                    std::process::exit(101);
                }
            }
            Poll::LostLease(_) => {}
        }
    }
    println!("worker completed {ran} trials");
    Ok(())
}

fn open(db: &str) -> Result<SqliteStorage, String> {
    SqliteStorage::open(db).map_err(|e| format!("opening `{db}`: {e}"))
}

fn cmd_list(args: &[String]) -> Result<(), String> {
    let db = args.first().ok_or("missing <study.db>")?;
    let storage = open(db)?;
    let ids = storage.study_ids().map_err(|e| e.to_string())?;
    if ids.is_empty() {
        println!("no studies in `{db}`");
        return Ok(());
    }
    for id in ids {
        let meta = storage.load_meta(id).map_err(|e| e.to_string())?;
        let history = storage.load_history(id).map_err(|e| e.to_string())?;
        println!("[{}] {}", id.0, summarize(&meta, history.records()));
    }
    Ok(())
}

fn cmd_dashboard(args: &[String]) -> Result<(), String> {
    let db = args.first().ok_or("missing <study.db>")?;
    let mut study_id: Option<u64> = None;
    let mut out = String::from("automl-dashboard.html");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--study" => {
                let v = args.get(i + 1).ok_or("--study needs a value")?;
                study_id = Some(v.parse().map_err(|_| format!("bad study id `{v}`"))?);
                i += 2;
            }
            "-o" | "--out" => {
                out = args.get(i + 1).ok_or("-o needs a value")?.clone();
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    let storage = open(db)?;
    let ids = storage.study_ids().map_err(|e| e.to_string())?;
    let id = match study_id {
        Some(v) => StudyId(v),
        // Default to the most recent study.
        None => *ids.last().ok_or_else(|| format!("no studies in `{db}`"))?,
    };

    let meta = storage.load_meta(id).map_err(|e| e.to_string())?;
    let history = storage.load_history(id).map_err(|e| e.to_string())?;
    let html = render_dashboard(&meta, history.records());
    std::fs::write(&out, html).map_err(|e| format!("writing `{out}`: {e}"))?;
    println!(
        "wrote dashboard for study [{}] \"{}\" to {out}",
        id.0, meta.name
    );
    Ok(())
}
