"""Judge calibration: humans label a sample of a run's outputs on the judge's 1-5 scale,
and we measure how well each judge scorer agrees with them.

Labels attach to response keys (target config + case), so they label a specific output, not a case,
and they keep matching after the scorers or judge change and the run is rescored with --resume."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from .report import load_results, write_report

MIN_LABELS = 20


def judge_scorers(meta: dict) -> list[str]:
    return [s["name"] for s in meta["scorers"] if s.get("judge")]


def export_sample(run_dir: Path, out: Path, n: int = 40, seed: int = 0) -> int:
    meta, results = load_results(run_dir)
    names = judge_scorers(meta)
    if not names:
        raise ValueError("this run has no judge scorers to calibrate")
    eligible = [r for r in results if any(s.name in names and s.value is not None for s in r.scores)]
    rng = random.Random(seed)
    by_target: dict[str, list] = {}
    for r in eligible:
        by_target.setdefault(r.target, []).append(r)
    for rs in by_target.values():
        rng.shuffle(rs)
    sample = []
    while len(sample) < n and any(by_target.values()):  # round-robin across targets
        for rs in by_target.values():
            if rs and len(sample) < n:
                sample.append(rs.pop())
    rng.shuffle(sample)  # don't group by target (reduces labeler bias)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel opens it cleanly
        w = csv.writer(f)
        w.writerow(["key", "case_id", "question", "reference_answer", "output", "context"] + [f"human_{s}" for s in names])
        for r in sample:
            ctx = "\n---\n".join(c.text[:400] for c in r.response.retrieved[:3]) or (r.case.context or "")
            w.writerow([r.rkey or r.key, r.case.id, r.case.input, r.case.expected or "(unanswerable)", r.response.output, ctx] + [""] * len(names))
    return len(sample)


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_labels(run_dir: Path, labels: Path) -> dict:
    meta, results = load_results(run_dir)
    by_key = {(r.rkey or r.key): r for r in results}
    names = judge_scorers(meta)
    pairs: dict[str, list[tuple[float, float]]] = {s: [] for s in names}
    bad_rows = 0
    with open(labels, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            r = by_key.get(row.get("key", ""))
            if not r:
                bad_rows += 1
                continue
            for s in names:
                raw = (row.get(f"human_{s}") or "").strip()
                if not raw:
                    continue
                try:
                    h = float(raw)
                except ValueError:
                    bad_rows += 1
                    continue
                if not 1 <= h <= 5:
                    bad_rows += 1
                    continue
                j = next((sc.value for sc in r.scores if sc.name == s and sc.value is not None), None)
                if j is not None:
                    pairs[s].append((h, 1 + 4 * j))  # judge 0..1 back to 1..5
    out: dict = {"labels_file": str(labels), "bad_rows": bad_rows, "scorers": {}}
    for s, ps in pairs.items():
        if not ps:
            continue
        h, j = [p[0] for p in ps], [p[1] for p in ps]
        within_1 = sum(abs(a - b) <= 1 for a, b in ps) / len(ps)
        pass_agree = sum((a >= 4) == (b >= 4) for a, b in ps) / len(ps)
        corr = _corr(h, j)
        mae = sum(abs(a - b) for a, b in ps) / len(ps)
        bias = sum(b - a for a, b in ps) / len(ps)  # >0: judge is more lenient than humans
        if len(ps) < MIN_LABELS:
            verdict = f"too few labels ({len(ps)}/{MIN_LABELS})"
        elif within_1 >= 0.8 and (corr is None or corr >= 0.6):
            verdict = "trust"
        else:
            verdict = "fix rubric or judge"
        out["scorers"][s] = {"n": len(ps), "within_1": within_1, "pass_agreement": pass_agree, "correlation": corr,
                             "mae": mae, "judge_bias": bias, "verdict": verdict}
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_report(run_dir)
    return out
