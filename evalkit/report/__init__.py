from __future__ import annotations

import json
import math
from pathlib import Path

from ..core.types import CaseResult
from ..core.util import percentile


def load_results(run_dir: Path) -> tuple[dict, list[CaseResult]]:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    valid = set(meta.get("keys") or [])
    by_key: dict[str, CaseResult] = {}
    path = run_dir / "results.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = CaseResult.from_dict(json.loads(line))
                    if not valid or r.key in valid:
                        by_key[r.key] = r
    return meta, list(by_key.values())


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _stderr(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


DEFAULT_BUDGETS = {"latency_s": 3.0, "cost_per_1k": 1.0, "min_quality": 0.5}


def budget_score(value: float | None, budget: float, missing: float = 0.0) -> float:
    """1.0 within budget, then decays as budget/value. Absolute, so small gaps between targets don't dominate.
    No measurement (e.g. every call failed) earns `missing`, so failing systems get no speed/cost credit."""
    if value is None:
        return missing
    if value <= budget:
        return 1.0
    return budget / value


def aggregate(meta: dict, results: list[CaseResult], worst_n: int = 5) -> dict:
    scorer_names = [s["name"] for s in meta["scorers"]]
    targets_meta = {t["name"]: t for t in meta["targets"]}
    per: dict[str, dict] = {}
    for name in targets_meta:
        rs = [r for r in results if r.target == name]
        ok = [r for r in rs if not r.response.error]
        qs = [r.quality for r in rs if r.quality is not None]
        lat = [r.response.latency_s for r in ok]
        scorers = {}
        for sn in scorer_names:
            ss = [s for r in ok for s in r.scores if s.name == sn]
            vals = [s.value for s in ss if s.value is not None]
            scorers[sn] = {"mean": _mean(vals), "n": len(vals), "skipped": sum(1 for s in ss if s.value is None and not s.error),
                           "errors": sum(1 for s in ss if s.error)}
        tags: dict[str, list[float]] = {}
        for r in rs:
            for tag in r.case.tags or ["untagged"]:
                if r.quality is not None:
                    tags.setdefault(tag, []).append(r.quality)
        worst = sorted((r for r in rs if r.quality is not None), key=lambda r: r.quality)[:worst_n]
        per[name] = {
            "name": name,
            "type": targets_meta[name]["type"],
            "models": targets_meta[name]["models"],
            "n": len(rs),
            "errors": len(rs) - len(ok),
            "error_rate": (len(rs) - len(ok)) / len(rs) if rs else None,
            "error_samples": sorted({r.response.error for r in rs if r.response.error})[:3],
            "quality": _mean(qs),
            "quality_se": _stderr(qs),
            "latency_p50": percentile(lat, 50),
            "latency_p95": percentile(lat, 95),
            "cost_per_1k": (sum(r.response.cost_usd for r in rs) / len(rs) * 1000) if rs else None,
            "avg_tokens": _mean([r.response.prompt_tokens + r.response.completion_tokens for r in ok]),
            "max_prompt_tokens": max((r.response.prompt_tokens for r in ok), default=None),
            "hardware": (meta.get("hardware") or {}).get(name) or {},
            "cached_frac": _mean([1.0 if r.response.cached else 0.0 for r in ok]),
            "scorers": scorers,
            "tags": {t: {"mean": _mean(v), "n": len(v)} for t, v in sorted(tags.items())},
            "worst": [r.to_dict() for r in worst],
        }

    w = meta["weights"]
    budgets = {**DEFAULT_BUDGETS, **(meta.get("budgets") or {})}
    wsum = sum(w.values()) or 1.0
    for v in per.values():
        q = v["quality"] if v["quality"] is not None else 0.0
        has_ok = v["errors"] < v["n"]
        v["latency_score"] = budget_score(v["latency_p50"], budgets["latency_s"])
        v["cost_score"] = budget_score(v["cost_per_1k"] if has_ok else None, budgets["cost_per_1k"])
        # quality gate: below min_quality, speed and cost earn nothing (a fast wrong answer is worthless)
        v["gated"] = q < budgets["min_quality"]
        vram = v["hardware"].get("vram_gb")
        v["fits"] = None if vram is None or "vram_gb" not in budgets else vram <= budgets["vram_gb"]
        lat_s, cost_s = (0.0, 0.0) if v["gated"] else (v["latency_score"], v["cost_score"])
        v["composite"] = (w.get("quality", 0) * q + w.get("latency", 0) * lat_s + w.get("cost", 0) * cost_s) / wsum
    ranking = sorted(per, key=lambda k: (-per[k]["composite"], -(per[k]["quality"] or 0)))
    for i, k in enumerate(ranking, 1):
        per[k]["rank"] = i
    return {"meta": {**{k: v for k, v in meta.items() if k != "keys"}, "budgets": budgets}, "targets": per, "ranking": ranking,
            "scorer_names": scorer_names, "n_results": len(results)}


def write_report(run_dir: Path, baseline: Path | None | bool = None) -> dict:
    """baseline: None = auto-detect the previous run, False = no diff, Path = compare to that run."""
    from .diff import diff_runs, find_baseline
    from .html import render_html

    meta, results = load_results(run_dir)
    summary = aggregate(meta, results)
    base = find_baseline(run_dir) if baseline is None else (baseline or None)
    if base:
        summary["diff"] = diff_runs(Path(base), summary, results)
    cal = run_dir / "calibration.json"
    if cal.exists():
        summary["calibration"] = json.loads(cal.read_text(encoding="utf-8"))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "report.html").write_text(render_html(summary), encoding="utf-8")
    return summary
