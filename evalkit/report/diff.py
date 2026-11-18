"""Compare a run to a baseline run: per-target deltas and per-case regressions."""
from __future__ import annotations

import json
from pathlib import Path

from ..core.types import CaseResult

THRESHOLD = 0.25


def find_baseline(run_dir: Path) -> Path | None:
    """Most recent earlier run of the same project in the same output directory."""
    try:
        project = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["project"]
    except (OSError, ValueError, KeyError):
        return None
    candidates = []
    for d in run_dir.parent.iterdir():
        if d == run_dir or not d.is_dir() or d.name >= run_dir.name or not (d / "summary.json").exists():
            continue
        try:
            if json.loads((d / "run_meta.json").read_text(encoding="utf-8")).get("project") == project:
                candidates.append(d)
        except (OSError, ValueError):
            continue
    return max(candidates, key=lambda d: d.name) if candidates else None


def diff_runs(base_dir: Path, cur_summary: dict, cur_results: list[CaseResult]) -> dict:
    from . import aggregate, load_results

    bmeta, bres = load_results(base_dir)
    base = aggregate(bmeta, bres)
    targets = {}
    for name, cur in cur_summary["targets"].items():
        b = base["targets"].get(name)
        if not b:
            targets[name] = {"new": True}
            continue
        targets[name] = {
            "quality_delta": None if cur["quality"] is None or b["quality"] is None else cur["quality"] - b["quality"],
            "composite_delta": cur["composite"] - b["composite"],
            "rank_before": b["rank"], "rank_now": cur["rank"],
            "config_changed": next((t["fingerprint"] for t in bmeta["targets"] if t["name"] == name), None)
                              != next((t["fingerprint"] for t in cur_summary["meta"]["targets"] if t["name"] == name), None),
        }
    bq = {(r.target, r.case.id): r for r in bres if r.quality is not None}
    regressions, improvements = [], []
    for r in cur_results:
        old = bq.get((r.target, r.case.id))
        if old is None or r.quality is None:
            continue
        d = r.quality - old.quality
        row = {"target": r.target, "case_id": r.case.id, "input": r.case.input[:160], "before": old.quality, "now": r.quality,
               "old_output": (old.response.error or old.response.output)[:400], "new_output": (r.response.error or r.response.output)[:400]}
        if d <= -THRESHOLD:
            regressions.append(row)
        elif d >= THRESHOLD:
            improvements.append(row)
    regressions.sort(key=lambda x: x["now"] - x["before"])
    improvements.sort(key=lambda x: x["before"] - x["now"])
    return {
        "baseline": base_dir.name,
        "dataset_changed": bmeta.get("dataset_hash") != cur_summary["meta"].get("dataset_hash"),
        "scorers_changed": bmeta.get("scorers") != cur_summary["meta"].get("scorers")
                           or bmeta.get("judge") != cur_summary["meta"].get("judge")
                           or bmeta.get("scorer_version") != cur_summary["meta"].get("scorer_version"),
        "targets": targets,
        "removed_targets": sorted(set(base["targets"]) - set(cur_summary["targets"])),
        "regressions": regressions[:15],
        "improvements": improvements[:15],
        "n_regressions": len(regressions),
        "n_improvements": len(improvements),
    }
