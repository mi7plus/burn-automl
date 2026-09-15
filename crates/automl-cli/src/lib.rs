//! `automl-cli` library: study inspection and the read-only dashboard v0
//! (PRD §25).
//!
//! The plan pulls a minimal dashboard forward from the v1.0 checklist so the
//! tool is inspectable while it is dogfooded. This v0 renders a **self-contained
//! HTML page** — trial list, objective-over-time, and per-parameter scatter —
//! read-only against a study's stored history, with no new persisted state and
//! no external assets. It inherits whatever consistency guarantees `Storage`
//! already provides (§25); a live local server can layer on later.

use automl_core::metrics::Direction;
use automl_core::storage::StudyMeta;
use automl_core::trial::{TrialRecord, TrialState};
use serde_json::json;

/// Render a study's history as a standalone HTML dashboard string.
pub fn render_dashboard(meta: &StudyMeta, records: &[TrialRecord]) -> String {
    let trials: Vec<serde_json::Value> = records
        .iter()
        .map(|r| {
            let params: serde_json::Map<String, serde_json::Value> = r
                .params
                .iter()
                .map(|(k, v)| (k.clone(), param_json(v)))
                .collect();
            let metrics: serde_json::Map<String, serde_json::Value> = r
                .final_metrics
                .as_ref()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
                .unwrap_or_default();
            json!({
                "id": r.id.0,
                "state": state_str(r.state),
                "wall_ms": r.wall_time_ms(),
                "params": params,
                "metrics": metrics,
            })
        })
        .collect();

    let objectives: Vec<serde_json::Value> = meta
        .directions
        .iter()
        .map(|(name, dir)| json!({ "name": name, "direction": dir_str(*dir) }))
        .collect();

    let config = json!({
        "study": meta.name,
        "sampler": meta.sampler_name,
        "pruner": meta.pruner_name,
        "objectives": objectives,
        "trials": trials,
    });

    let data = serde_json::to_string(&config).unwrap_or_else(|_| "{}".into());
    TEMPLATE
        .replace("/*__DATA__*/null", &data)
        .replace("__TITLE__", &html_escape(&meta.name))
}

/// A short one-line text summary of a study (used by `automl list`).
pub fn summarize(meta: &StudyMeta, records: &[TrialRecord]) -> String {
    let completed = records
        .iter()
        .filter(|r| r.state == TrialState::Complete)
        .count();
    let pruned = records
        .iter()
        .filter(|r| r.state == TrialState::Pruned)
        .count();
    let failed = records
        .iter()
        .filter(|r| r.state == TrialState::Failed)
        .count();
    format!(
        "{} — {} trials ({} complete, {} pruned, {} failed) · sampler={} · pruner={}",
        meta.name,
        records.len(),
        completed,
        pruned,
        failed,
        meta.sampler_name,
        meta.pruner_name,
    )
}

fn param_json(v: &automl_core::param::ParamValue) -> serde_json::Value {
    use automl_core::param::ParamValue::*;
    match v {
        Float(x) => json!(x),
        Int(x) => json!(x),
        Categorical(s) => json!(s),
        Bool(b) => json!(b),
    }
}

fn state_str(s: TrialState) -> &'static str {
    match s {
        TrialState::Waiting => "waiting",
        TrialState::Running => "running",
        TrialState::Complete => "complete",
        TrialState::Pruned => "pruned",
        TrialState::Failed => "failed",
        TrialState::Cancelled => "cancelled",
    }
}

fn dir_str(d: Direction) -> &'static str {
    match d {
        Direction::Minimize => "minimize",
        Direction::Maximize => "maximize",
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The dashboard HTML/JS template. `/*__DATA__*/null` is replaced with the study
/// config JSON and `__TITLE__` with the study name.
const TEMPLATE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>burn-automl · __TITLE__</title>
<style>
  :root { color-scheme: light dark; --fg:#1b1b1f; --bg:#fafafa; --muted:#6b6b73;
          --line:#d9d9e0; --accent:#4f46e5; --pruned:#b45309; --failed:#b91c1c; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e7e7ea; --bg:#161619; --muted:#9a9aa3; --line:#33333b; --accent:#8b85f0; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         color:var(--fg); background:var(--bg); padding:24px; }
  h1 { font-size:20px; margin:0 0 2px; }
  .sub { color:var(--muted); margin-bottom:20px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  @media (max-width:800px){ .grid{ grid-template-columns:1fr; } }
  .card { border:1px solid var(--line); border-radius:10px; padding:16px; background:var(--bg); }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em;
             color:var(--muted); margin:0 0 12px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  .state-pruned{ color:var(--pruned); } .state-failed{ color:var(--failed); }
  .tablewrap { max-height:420px; overflow:auto; }
  select { font:inherit; color:var(--fg); background:var(--bg); border:1px solid var(--line);
           border-radius:6px; padding:3px 6px; }
  svg { width:100%; height:260px; display:block; }
  .axis { stroke:var(--line); } .tick { fill:var(--muted); font-size:10px; }
  .dot { fill:var(--accent); } .best { stroke:var(--accent); fill:none; stroke-width:2; }
</style></head>
<body>
<h1>__TITLE__</h1>
<div class="sub" id="sub"></div>
<div class="grid">
  <div class="card"><h2>Objective over time</h2><div id="curve"></div></div>
  <div class="card"><h2>Parameter scatter <span id="scatterctl"></span></h2><div id="scatter"></div></div>
</div>
<div class="card" style="margin-top:24px"><h2>Trials</h2><div class="tablewrap"><div id="table"></div></div></div>

<script>
const CONFIG = /*__DATA__*/null;
(function(){
  if(!CONFIG){ document.body.innerHTML += "<p>No data.</p>"; return; }
  const objs = CONFIG.objectives||[];
  const trials = CONFIG.trials||[];
  const obj0 = objs[0];
  document.getElementById("sub").textContent =
    `sampler: ${CONFIG.sampler} · pruner: ${CONFIG.pruner} · objectives: ` +
    objs.map(o=>`${o.name} (${o.direction})`).join(", ") + ` · ${trials.length} trials`;

  const done = trials.filter(t=>t.state==="complete" && obj0 && (obj0.name in t.metrics));
  const better = (a,b)=> obj0.direction==="minimize" ? a<b : a>b;

  // ---- objective over time: value per trial + best-so-far line ----
  function chart(el, pts, line){
    const W=460,H=260,P=34;
    if(!pts.length){ el.innerHTML="<p style='color:var(--muted)'>No completed trials yet.</p>"; return; }
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const sx=v=> P+(W-2*P)*((v-x0)/((x1-x0)||1));
    const sy=v=> H-P-(H-2*P)*((v-y0)/((y1-y0)||1));
    let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
    s+=`<line class="axis" x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}"/>`;
    s+=`<line class="axis" x1="${P}" y1="${P}" x2="${P}" y2="${H-P}"/>`;
    s+=`<text class="tick" x="${P}" y="${H-P+14}">${x0}</text>`;
    s+=`<text class="tick" x="${W-P}" y="${H-P+14}" text-anchor="end">${x1}</text>`;
    s+=`<text class="tick" x="${P-6}" y="${sy(y1)}" text-anchor="end">${fmt(y1)}</text>`;
    s+=`<text class="tick" x="${P-6}" y="${sy(y0)}" text-anchor="end">${fmt(y0)}</text>`;
    for(const p of pts) s+=`<circle class="dot" cx="${sx(p[0])}" cy="${sy(p[1])}" r="2.5"/>`;
    if(line&&line.length){ s+=`<polyline class="best" points="`+
      line.map(p=>`${sx(p[0])},${sy(p[1])}`).join(" ")+`"/>`; }
    el.innerHTML=s+`</svg>`;
  }
  function fmt(v){ return Math.abs(v)>=1000||(v!==0&&Math.abs(v)<0.01)? v.toExponential(2): (Math.round(v*1000)/1000); }

  if(obj0){
    const pts=done.map(t=>[t.id, t.metrics[obj0.name]]);
    let best=null; const line=pts.map(p=>{ if(best===null||better(p[1],best)) best=p[1]; return [p[0],best]; });
    chart(document.getElementById("curve"), pts, line);
  }

  // ---- parameter scatter (choose a numeric param) ----
  const paramNames=[...new Set(trials.flatMap(t=>Object.keys(t.params||{})))];
  const numeric=paramNames.filter(n=> done.some(t=> typeof t.params[n]==="number"));
  const sel=document.createElement("select");
  numeric.forEach(n=>{ const o=document.createElement("option"); o.value=n; o.textContent=n; sel.appendChild(o); });
  document.getElementById("scatterctl").appendChild(sel);
  function scatter(){
    const name=sel.value; const el=document.getElementById("scatter");
    if(!name||!obj0){ el.innerHTML=""; return; }
    const pts=done.filter(t=>typeof t.params[name]==="number").map(t=>[t.params[name], t.metrics[obj0.name]]);
    chart(el, pts, null);
  }
  sel.addEventListener("change", scatter); scatter();

  // ---- trials table ----
  const cols=["id","state","wall (ms)",...objs.map(o=>o.name),...paramNames];
  let html="<table><thead><tr>"+cols.map(c=>`<th>${c}</th>`).join("")+"</tr></thead><tbody>";
  for(const t of trials){
    html+="<tr>";
    html+=`<td>${t.id}</td><td class="state-${t.state}">${t.state}</td>`;
    html+=`<td>${t.wall_ms==null?"":t.wall_ms}</td>`;
    for(const o of objs){ const v=t.metrics[o.name]; html+=`<td>${v==null?"":fmt(v)}</td>`; }
    for(const p of paramNames){ const v=t.params[p]; html+=`<td>${v==null?"":(typeof v==="number"?fmt(v):v)}</td>`; }
    html+="</tr>";
  }
  document.getElementById("table").innerHTML=html+"</tbody></table>";
})();
</script>
</body></html>"##;

#[cfg(test)]
mod tests {
    use super::*;
    use automl_core::metrics::NamedMetrics;
    use automl_core::param::{ParamSet, ParamValue};
    use automl_core::trial::{StudyId, TrialId};

    fn meta() -> StudyMeta {
        StudyMeta {
            name: "demo".into(),
            directions: vec![("loss".into(), Direction::Minimize)],
            sampler_name: "tpe".into(),
            pruner_name: "median".into(),
        }
    }

    fn rec(id: u64, x: f64, loss: f64, state: TrialState) -> TrialRecord {
        let mut r = TrialRecord::new(
            TrialId(id),
            StudyId(0),
            ParamSet::new().with("x", ParamValue::Float(x)),
            id,
        );
        r.state = state;
        r.final_metrics = Some(NamedMetrics::single("loss", loss));
        r
    }

    #[test]
    fn dashboard_embeds_data_and_is_self_contained() {
        let records = vec![
            rec(0, 1.0, 0.5, TrialState::Complete),
            rec(1, 2.0, 0.2, TrialState::Complete),
        ];
        let html = render_dashboard(&meta(), &records);
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("demo"));
        // Data is embedded (no external fetch needed).
        assert!(html.contains("\"trials\""));
        assert!(html.contains("\"loss\""));
        // No external asset references.
        assert!(!html.contains("http://"));
        assert!(!html.to_lowercase().contains("https://"));
        // The placeholder was substituted.
        assert!(!html.contains("/*__DATA__*/null"));
        assert!(!html.contains("__TITLE__"));
    }

    #[test]
    fn summary_counts_states() {
        let records = vec![
            rec(0, 1.0, 0.5, TrialState::Complete),
            rec(1, 2.0, 0.2, TrialState::Pruned),
            rec(2, 3.0, 0.9, TrialState::Failed),
        ];
        let s = summarize(&meta(), &records);
        assert!(s.contains("3 trials"));
        assert!(s.contains("1 complete"));
        assert!(s.contains("1 pruned"));
        assert!(s.contains("1 failed"));
    }
}
