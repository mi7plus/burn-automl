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
use automl_core::sqlite::SqliteStorage;
use automl_core::storage::Storage;
use automl_core::trial::StudyId;
use std::process::ExitCode;

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
  automl dashboard <study.db> [--study <id>] [-o <out.html>]";

fn run(args: &[String]) -> Result<(), String> {
    let command = args.first().map(String::as_str).ok_or("no command given")?;
    match command {
        "list" => cmd_list(&args[1..]),
        "dashboard" => cmd_dashboard(&args[1..]),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    }
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
