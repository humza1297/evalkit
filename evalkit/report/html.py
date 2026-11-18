"""Self-contained HTML report. Every value from data is escaped (it may be client content)."""
from __future__ import annotations

from html import escape

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--text:#1d1d1b;--muted:#6b6b66;--border:#e2e2dc;--accent:#2f6fdb;
--good:#1f8a4c;--mid:#b7791f;--bad:#c0392b;--warnbg:#fff4d6;--warnfg:#7a5300;--code:#f0f0ec}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--text:#ececea;--muted:#a3a39c;--border:#33332f;
--accent:#7aa7ff;--good:#4cc283;--mid:#e0a84a;--bad:#ef6f61;--warnbg:#3a2f12;--warnfg:#f3d38a;--code:#262623}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.muted{color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}td.num,th.num{text-align:right}
tr.top td{font-weight:600}.banner{background:var(--warnbg);color:var(--warnfg);border-radius:8px;padding:10px 14px;margin:14px 0;font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--border);font-size:12px;margin-right:6px}
.good{color:var(--good)}.mid{color:var(--mid)}.bad{color:var(--bad)}details{border-top:1px solid var(--border);padding:8px 0}
summary{cursor:pointer}pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:8px 10px;border-radius:6px;margin:6px 0;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}.label{font-size:12px;color:var(--muted);margin-top:6px}
"""


def _f(v, fmt="{:.2f}", none="–"):
    return none if v is None else fmt.format(v)


def _cls(v):
    if v is None:
        return ""
    return "good" if v >= 0.75 else "mid" if v >= 0.5 else "bad"


def _q(v, se=None):
    if v is None:
        return '<span class="muted">–</span>'
    extra = f' <span class="muted">±{se:.2f}</span>' if se is not None else ""
    return f'<span class="{_cls(v)}">{v:.2f}</span>{extra}'


def render_html(summary: dict) -> str:
    meta, per, ranking, scorers = summary["meta"], summary["targets"], summary["ranking"], summary["scorer_names"]
    e = escape
    w = meta["weights"]
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Eval report: {e(meta['project'])}</title><style>{CSS}</style></head><body><main>"]
    out.append(f"<h1>Eval report: {e(meta['project'])}</h1>")
    out.append(f"<div class='muted'>{e(meta.get('started', ''))} · {meta['n_cases']} cases · dataset {e(meta['dataset_hash'])} · "
               f"judge {e(str(meta.get('judge') or 'none'))}</div><div style='margin-top:8px'>")
    out.append(f"<span class='badge'>{'confidential: local only' if meta.get('confidential') else 'not confidential'}</span>")
    out.append(f"<span class='badge'>weights: quality {w.get('quality', 0)} · latency {w.get('latency', 0)} · cost {w.get('cost', 0)}</span></div>")
    if meta.get("is_mock"):
        out.append("<div class='banner'>MOCK RUN: outputs come from the offline mock provider. This is a pipeline demo, not a real model ranking.</div>")
    if not meta.get("finished"):
        out.append("<div class='banner'>Run incomplete: results are partial. Resume with <code>evalkit run CONFIG --resume RUN_DIR</code>.</div>")

    # leaderboard
    out.append("<h2>Leaderboard</h2><div class='panel'><table><tr><th>#</th><th>Target</th><th class='num'>Score</th><th class='num'>Quality</th>"
               "<th class='num'>Errors</th><th class='num'>p50 s</th><th class='num'>p95 s</th><th class='num'>$/1k queries</th>"
               "<th class='num'>Max prompt tok</th><th class='num'>VRAM GB</th><th>Models</th></tr>")
    for name in ranking:
        t = per[name]
        err_cls = "bad" if (t["error_rate"] or 0) > 0.05 else ""
        gate = " <span class='bad' title='below min_quality: no latency/cost credit'>gated</span>" if t.get("gated") else ""
        out.append(f"<tr class='{'top' if t['rank'] == 1 else ''}'><td>{t['rank']}</td><td>{e(name)}{gate}</td><td class='num'>{_f(t['composite'])}</td>"
                   f"<td class='num'>{_q(t['quality'], t['quality_se'])}</td><td class='num {err_cls}'>{t['errors']}/{t['n']}</td>"
                   f"<td class='num'>{_f(t['latency_p50'])}</td><td class='num'>{_f(t['latency_p95'])}</td>"
                   f"<td class='num'>{_f(t['cost_per_1k'], '${:.3f}')}</td><td class='num'>{_f(t.get('max_prompt_tokens'), '{:,}')}</td>"
                   f"<td class='num {'bad' if t.get('fits') is False else ''}' title='{e(str(t.get('hardware', {}).get('source', '')))}'>"
                   f"{_f(t.get('hardware', {}).get('vram_gb'), '{:.1f}')}{'' if t.get('fits') is not False else ' (too big)'}"
                   f"{'' if 'measured' in str(t.get('hardware', {}).get('source', '')) or t.get('hardware', {}).get('vram_gb') is None else '*'}</td>"
                   f"<td class='muted'>{e(', '.join(t['models']))}</td></tr>")
    b = meta.get("budgets") or {}
    out.append(f"</table><div class='label'>Score = weighted quality, latency and cost. Latency/cost get full marks within budget "
               f"(p50 ≤ {b.get('latency_s', '?')}s, ≤ ${b.get('cost_per_1k', '?')}/1k queries) and decay beyond it; "
               f"targets below quality {b.get('min_quality', '?')} are gated (no latency/cost credit). "
               "If max prompt tokens sits exactly at a round number (e.g. 4,096), the server is probably truncating context. "
               "VRAM is measured from Ollama when the model was loaded; * = estimated from size and quantization. "
               "Quality = weighted mean of scorers per case; a target error counts as 0; ± is the standard error. "
               "Differences within ± are likely noise.</div></div>")

    d = summary.get("diff")
    if d:
        warn = []
        if d["dataset_changed"]:
            warn.append("dataset changed")
        if d["scorers_changed"]:
            warn.append("scorers/judge changed")
        out.append(f"<h2>Changes vs {e(d['baseline'])}</h2><div class='panel'>")
        if warn:
            out.append(f"<div class='banner'>Not like-for-like: {e(', '.join(warn))}. Deltas mix system changes with measurement changes.</div>")
        out.append("<table><tr><th>Target</th><th class='num'>Quality Δ</th><th class='num'>Score Δ</th><th class='num'>Rank</th><th>Config</th></tr>")
        for name in ranking:
            t = d["targets"].get(name, {})
            if t.get("new"):
                out.append(f"<tr><td>{e(name)}</td><td colspan='4' class='muted'>new target</td></tr>")
                continue
            qd, cd = t.get("quality_delta"), t.get("composite_delta")
            out.append(f"<tr><td>{e(name)}</td><td class='num {'good' if (qd or 0) > 0.02 else 'bad' if (qd or 0) < -0.02 else ''}'>{_f(qd, '{:+.2f}')}</td>"
                       f"<td class='num'>{_f(cd, '{:+.2f}')}</td><td class='num'>{t.get('rank_before')} → {t.get('rank_now')}</td>"
                       f"<td>{'changed' if t.get('config_changed') else 'same'}</td></tr>")
        out.append("</table>")
        if d["removed_targets"]:
            out.append(f"<div class='label'>Removed: {e(', '.join(d['removed_targets']))}</div>")
        out.append(f"<div class='label'>{d['n_regressions']} case regressions, {d['n_improvements']} improvements (|Δ| ≥ 0.25)</div>")
        for r in d["regressions"][:8]:
            out.append(f"<details><summary><span class='bad'>{r['before']:.2f} → {r['now']:.2f}</span> · {e(r['target'])} · "
                       f"<code>{e(r['case_id'])}</code> {e(r['input'])}</summary><div class='grid'>"
                       f"<div><div class='label'>Before</div><pre>{e(r['old_output'])}</pre></div>"
                       f"<div><div class='label'>Now</div><pre>{e(r['new_output'])}</pre></div></div></details>")
        out.append("</div>")

    cal = summary.get("calibration")
    if cal:
        out.append("<h2>Judge calibration</h2><div class='panel'><table><tr><th>Scorer</th><th class='num'>Labels</th>"
                   "<th class='num'>Within 1 pt</th><th class='num'>Pass/fail agree</th><th class='num'>Correlation</th><th>Verdict</th></tr>")
        for name, c in cal.get("scorers", {}).items():
            verdict_cls = "good" if c["verdict"] == "trust" else "bad"
            out.append(f"<tr><td>{e(name)}</td><td class='num'>{c['n']}</td><td class='num'>{_f(c.get('within_1'))}</td>"
                       f"<td class='num'>{_f(c.get('pass_agreement'))}</td><td class='num'>{_f(c.get('correlation'))}</td>"
                       f"<td class='{verdict_cls}'>{e(c['verdict'])}</td></tr>")
        out.append("</table><div class='label'>Human labels vs judge on the same outputs. 'trust' needs ≥ 80% within one point "
                   "and correlation ≥ 0.6 on at least 20 labels. Otherwise fix the rubric or judge model before trusting rankings.</div></div>")

    errs = [(n, per[n]) for n in ranking if per[n]["errors"]]
    if errs:
        out.append("<h2>Target errors</h2><div class='panel'>")
        for n, t in errs:
            out.append(f"<div><b>{e(n)}</b>: {t['errors']} errors<pre>{e(chr(10).join(t['error_samples']))}</pre></div>")
        out.append("</div>")

    # scorer breakdown
    sw = {s["name"]: s["weight"] for s in meta["scorers"]}
    out.append("<h2>By scorer</h2><div class='panel'><table><tr><th>Target</th>" + "".join(
        f"<th class='num'>{e(s)}<br><span style='text-transform:none'>w={sw.get(s, 1):g}</span></th>" for s in scorers) + "</tr>")
    for name in ranking:
        cells = []
        for s in scorers:
            st = per[name]["scorers"][s]
            note = f" <span class='bad'>({st['errors']} err)</span>" if st["errors"] else ""
            cells.append(f"<td class='num'>{_q(st['mean'])} <span class='muted'>n={st['n']}</span>{note}</td>")
        out.append(f"<tr><td>{e(name)}</td>{''.join(cells)}</tr>")
    out.append("</table><div class='label'>n = cases where the scorer applied; w = weight in quality (w=0 is diagnostic only). "
               "Scorer errors (e.g. unparseable judge output) are excluded from quality.</div></div>")

    # tag breakdown
    all_tags = sorted({tag for t in per.values() for tag in t["tags"]})
    if all_tags:
        out.append("<h2>By case type</h2><div class='panel'><table><tr><th>Target</th>" + "".join(f"<th class='num'>{e(t)}</th>" for t in all_tags) + "</tr>")
        for name in ranking:
            tg = per[name]["tags"]
            out.append(f"<tr><td>{e(name)}</td>" + "".join(
                f"<td class='num'>{_q(tg[t]['mean']) if t in tg else '–'} <span class='muted'>{('n=' + str(tg[t]['n'])) if t in tg else ''}</span></td>"
                for t in all_tags) + "</tr>")
        out.append("</table></div>")

    # worst cases
    out.append("<h2>Worst cases per target</h2>")
    for name in ranking:
        out.append(f"<div class='panel' style='margin-bottom:10px'><b>{e(name)}</b>")
        for r in per[name]["worst"]:
            c, resp = r["case"], r["response"]
            out.append(f"<details><summary>{_q(r['quality'])} · <code>{e(c['id'])}</code> {e(c['input'][:140])}</summary><div class='grid'>")
            out.append(f"<div><div class='label'>Expected</div><pre>{e(str(c.get('expected') or '(none)'))}</pre></div>")
            out.append(f"<div><div class='label'>Output</div><pre>{e(resp.get('error') or resp.get('output') or '')}</pre></div></div>")
            sc = "\n".join(f"{s['name']}: {'–' if s['value'] is None else round(s['value'], 2)}  {s.get('error') or s.get('reason') or ''}" for s in r["scores"])
            out.append(f"<div class='label'>Scores</div><pre>{e(sc)}</pre>")
            if resp.get("retrieved"):
                docs = ", ".join(f"{ch['doc_id']}" for ch in resp["retrieved"])
                out.append(f"<div class='label'>Retrieved from: {e(docs)}</div>")
            out.append("</details>")
        out.append("</div>")
    out.append("</main></body></html>")
    return "".join(out)
