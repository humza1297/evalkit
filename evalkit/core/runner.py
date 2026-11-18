"""Runs every target over every case, scores the responses, and writes results incrementally (resumable)."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..datasets.store import dataset_hash, load_cases
from .cache import Cache
from .config import ProjectConfig
from .env import ConfidentialityError, Env
from .registry import get, load_builtins
from .types import Case, CaseResult, Response, Score
from .util import stable_hash


def build_env(cfg: ProjectConfig) -> Env:
    return Env(confidential=cfg.confidential, providers_cfg=cfg.providers, local_hosts=cfg.local_hosts,
               cache=Cache(cfg.cache_path, enabled=cfg.cache))


def build_targets(cfg: ProjectConfig, env: Env) -> list:
    return [get("target", t["type"])(t, env, cfg) for t in cfg.targets]


def build_scorers(cfg: ProjectConfig, env: Env) -> list:
    return [get("scorer", s["name"])(s, env, cfg) for s in cfg.scorers]


def case_quality(resp: Response, scores: list[Score], weights: dict[str, float]) -> float | None:
    """Weighted mean of valid scores. A target error counts as 0 (the system failed the user);
    scorer/judge errors are excluded (not the target's fault) and reported separately."""
    if resp.error:
        return 0.0
    valid = [(s.value, weights.get(s.name, 1.0)) for s in scores if s.value is not None]
    total = sum(w for _, w in valid)
    if not valid or total == 0:
        return None
    return sum(v * w for v, w in valid) / total


def generate(target, case: Case) -> Response:
    try:
        return target.run(case)
    except ConfidentialityError:
        raise
    except Exception as e:  # noqa: BLE001 - any target failure is recorded, not fatal
        return Response(error=f"{type(e).__name__}: {e}")


def score_case(target_name: str, case: Case, resp: Response, scorers: list, key: str, weights: dict[str, float],
               rkey: str = "") -> CaseResult:
    scores: list[Score] = []
    if not resp.error:
        for s in scorers:
            try:
                scores.append(s.score(case, resp))
            except ConfidentialityError:
                raise
            except Exception as e:  # noqa: BLE001
                scores.append(Score(s.name, None, error=f"{type(e).__name__}: {e}"))
    return CaseResult(key=key, target=target_name, case=case, response=resp, scores=scores,
                      quality=case_quality(resp, scores, weights), rkey=rkey)


def _scorer_version() -> str:
    """Hash of built-in judge prompts and refusal patterns, so run diffs flag scoring-code changes."""
    from ..scorers import deterministic, judge

    return stable_hash([judge.JUDGE_SYSTEM, judge.CORRECTNESS, judge.FAITHFULNESS, judge.RUBRIC, judge.TONE_CRITERIA,
                        judge.POLICY_PREAMBLE, deterministic.REFUSAL_PATTERNS], 12)


def target_hardware(target, env: Env) -> dict:
    """Memory for the target's generation model (embedders are small and listed separately)."""
    from .hardware import model_hardware

    spec = target.cfg.get("model")
    if not spec:
        return {"vram_gb": None, "source": "n/a (remote endpoint)"}
    try:
        p, model = env.resolve(spec, "hardware check")
        return model_hardware(p, model)
    except Exception as e:  # noqa: BLE001
        return {"vram_gb": None, "source": f"unknown ({e})"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parallel(fn, items, workers: int, on_result: Callable) -> None:
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(fn, *it) for it in items]
        for fut in as_completed(futures):
            on_result(fut.result())
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)


def _is_mock(cfg: ProjectConfig, targets: list) -> bool:
    specs = [m for t in targets for m in t.models()] + ([cfg.judge] if cfg.judge else [])
    return bool(specs) and any(str(s).startswith("mock/") for s in specs)


def run(cfg: ProjectConfig, *, resume: Path | None = None, limit: int | None = None,
        progress: Callable[[str], None] = print) -> Path:
    load_builtins()
    if not cfg.dataset or not cfg.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {cfg.dataset}")
    if not cfg.targets:
        raise ValueError("config has no targets")
    cases = load_cases(cfg.dataset)
    limit = limit or cfg.limit
    if limit:
        cases = cases[: int(limit)]
    if not cases:
        raise ValueError(f"no usable cases in {cfg.dataset} (pending/rejected cases are excluded; run 'review')")

    env = build_env(cfg)
    targets = build_targets(cfg, env)
    scorers = build_scorers(cfg, env)
    for obj in [*targets, *scorers]:
        obj.setup()  # resolves providers -> confidentiality guard fails fast here
    weights = {s.name: s.weight for s in scorers}
    scorer_fp = stable_hash([cfg.scorers, cfg.judge])

    if resume:
        run_dir = resume
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir, n = cfg.output_dir / stamp, 1
        while run_dir.exists():
            n += 1
            run_dir = cfg.output_dir / f"{stamp}-{n:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    responses_path = run_dir / "responses.jsonl"
    # On resume, failed responses are retried (e.g. after `ollama pull`); successes are reused.
    done = {r["key"] for r in _read_jsonl(results_path) if not r["response"].get("error")} if resume else set()
    responses = {r["key"]: Response.from_dict(r["response"]) for r in _read_jsonl(responses_path)
                 if not r["response"].get("error")} if resume else {}
    old_hardware = {}
    if resume and (run_dir / "run_meta.json").exists():
        old_hardware = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8")).get("hardware") or {}

    meta = {
        "project": cfg.project,
        "config_path": str(cfg.path),
        "started": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(cfg.dataset),
        "dataset_hash": dataset_hash(cases),
        "n_cases": len(cases),
        "confidential": cfg.confidential,
        "is_mock": _is_mock(cfg, targets),
        "targets": [{"name": t.name, "type": t.type, "models": t.models(), "fingerprint": t.fingerprint(),
                     "config": t.cfg} for t in targets],
        "scorers": [{"name": s.name, "weight": s.weight, "judge": hasattr(s, "ask")} for s in scorers],
        "judge": cfg.judge,
        "scorer_version": _scorer_version(),
        "weights": cfg.weights,
        "budgets": cfg.budgets,
        "hardware": {t.name: old_hardware[t.name] for t in targets if t.name in old_hardware},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # job = (target, case, response key [target+case], result key [+scorers])
    jobs = []
    for t in targets:
        for c in cases:
            rkey = stable_hash([t.name, t.fingerprint(), c.id, c.content_hash()])
            jobs.append((t, c, rkey, stable_hash([rkey, scorer_fp])))
    meta["keys"] = [j[3] for j in jobs]  # report only uses results matching the current config + dataset
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    todo = [j for j in jobs if j[3] not in done]
    progress(f"{len(targets)} targets x {len(cases)} cases = {len(jobs)} evaluations "
             f"({len(todo)} to run, {len(jobs) - len(todo)} already done)")

    lock = threading.Lock()
    t0 = time.perf_counter()

    # Pass 1: generate, grouped by target so a local GPU loads each model once (plus an untimed warm-up).
    with open(responses_path, "a", encoding="utf-8") as out:
        for t in targets:
            gen_jobs = [(t, c, rkey) for tt, c, rkey, _ in todo if tt is t and rkey not in responses]
            if not gen_jobs:
                continue
            progress(f"generating: {t.name} ({len(gen_jobs)} cases)")
            t.warmup()
            counter = [0]

            def on_resp(item, out=out, counter=counter, total=len(gen_jobs)):
                rkey, resp = item
                with lock:
                    responses[rkey] = resp
                    out.write(json.dumps({"key": rkey, "response": resp.to_dict()}, ensure_ascii=False) + "\n")
                    out.flush()
                    counter[0] += 1
                    if counter[0] % 10 == 0 or counter[0] == total:
                        progress(f"  {counter[0]}/{total} ({time.perf_counter() - t0:.1f}s)")

            _parallel(lambda t, c, rkey: (rkey, generate(t, c)), gen_jobs, cfg.concurrency, on_resp)
            meta.setdefault("hardware", {})[t.name] = target_hardware(t, env)  # model is loaded right now
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pass 2: score (judge model stays loaded for the whole pass).
    if todo:
        progress(f"scoring {len(todo)} responses")
    with open(results_path, "a", encoding="utf-8") as out:
        def on_result(res: CaseResult):
            with lock:
                out.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                out.flush()

        _parallel(lambda t, c, rkey, key: score_case(t.name, c, responses[rkey], scorers, key, weights, rkey),
                  todo, cfg.concurrency, on_result)
    progress(f"done in {time.perf_counter() - t0:.1f}s")

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    env.cache.close()

    from ..report import write_report

    write_report(run_dir)
    return run_dir
