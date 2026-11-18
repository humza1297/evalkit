import csv
import json

from evalkit.calibrate import export_sample, score_labels
from evalkit.core.config import load_config
from evalkit.core.hardware import estimate_vram_gb, params_from_name
from evalkit.core.runner import run
from evalkit.report import write_report
from evalkit.report.diff import find_baseline

from .conftest import write_config
from .test_rag_http import corpus  # noqa: F401


def test_params_and_vram_estimates():
    assert params_from_name("llama3.2:3b") == 3e9
    assert params_from_name("qwen3:0.6b") == 0.6e9
    assert params_from_name("gemma3:270m") == 270e6
    assert params_from_name("nomic-embed-text") is None
    assert 4.5 < estimate_vram_gb(8e9, "Q4_K_M") < 6.5
    assert estimate_vram_gb(8e9, "F16") > 18


def test_run_diff_detects_regression(project):
    cfg = load_config(project)
    first = run(cfg, progress=lambda *_: None)
    # make "extractive" worse: swap its model for the always-refusing one
    cfg.targets[0]["model"] = "mock/lazy"
    second = run(cfg, progress=lambda *_: None)
    assert second != first and find_baseline(second) == first
    s = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    d = s["diff"]
    assert d["baseline"] == first.name
    assert d["targets"]["extractive"]["config_changed"] is True
    assert d["targets"]["extractive"]["quality_delta"] < 0
    assert d["n_regressions"] >= 1
    assert "Changes vs" in (second / "report.html").read_text(encoding="utf-8")
    assert "diff" not in write_report(second, baseline=False)


def _rag_run(tmp_path, corpus):  # noqa: F811
    from .conftest import write_cases

    write_cases(tmp_path / "eval.jsonl", [
        {"id": f"q{i}", "input": q, "expected": a, "tags": ["fact"]}
        for i, (q, a) in enumerate([
            ("What is the standard listing commission?", "The standard listing commission is 5 percent"),
            ("How is commission split between agents?", "Commission is split 50/50"),
            ("What time does the office open on weekdays?", "The office opens at 9am"),
            ("What are Saturday hours?", "Saturday hours are 10am to 2pm"),
            ("How long are earnest money deposits held?", "held in escrow for 30 days"),
        ] * 5)
    ])
    cfg = write_config(tmp_path / "evalkit.yaml", {
        "project": "cal", "corpus": "docs", "dataset": "eval.jsonl", "judge": "mock/judge",
        "scorers": ["answer_correctness", "refusal"],
        "targets": [{"name": "a", "type": "rag", "model": "mock/extractive"}, {"name": "b", "type": "rag", "model": "mock/lazy"}],
    })
    return run(load_config(cfg), progress=lambda *_: None)


def test_calibration_roundtrip(tmp_path, corpus):  # noqa: F811
    run_dir = _rag_run(tmp_path, corpus)
    labels = tmp_path / "labels.csv"
    n = export_sample(run_dir, labels, n=30)
    assert n == 30
    with open(labels, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert "human_answer_correctness" in rows[0] and "human_refusal" not in rows[0]
    assert not any(k.startswith("judge_") for k in rows[0])  # judge scores hidden from labelers
    # simulate a human who agrees with the judge: copy judge score from results
    results = {json.loads(line)["rkey"]: json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()}
    for r in rows:
        v = next(s["value"] for s in results[r["key"]]["scores"] if s["name"] == "answer_correctness")
        r["human_answer_correctness"] = str(round(1 + 4 * v))
    with open(labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = score_labels(run_dir, labels)
    c = res["scorers"]["answer_correctness"]
    assert c["n"] == 30 and c["within_1"] == 1.0 and c["verdict"] == "trust"
    assert "Judge calibration" in (run_dir / "report.html").read_text(encoding="utf-8")


def test_labels_survive_rescoring(tmp_path, corpus):  # noqa: F811
    run_dir = _rag_run(tmp_path, corpus)
    labels = tmp_path / "labels.csv"
    export_sample(run_dir, labels, n=20)
    with open(labels, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["human_answer_correctness"] = "3"
    with open(labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # change the scorer config (e.g. a judge setting), rescore on resume
    cfg = load_config(tmp_path / "evalkit.yaml")
    cfg.scorers = [{"name": "answer_correctness", "weight": 3.0, "max_tokens": 200}, {"name": "refusal"}]
    msgs = []
    run(cfg, resume=run_dir, progress=msgs.append)
    assert not any(m.startswith("generating") for m in msgs)
    res = score_labels(run_dir, labels)
    assert res["bad_rows"] == 0 and res["scorers"]["answer_correctness"]["n"] == 20


def test_resume_retries_failed_responses(project):
    cfg = load_config(project)
    cfg.targets.append({"name": "flaky", "type": "model", "model": "mock/notpulled"})
    run_dir = run(cfg, progress=lambda *_: None)
    s1 = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s1["targets"]["flaky"]["errors"] == 3
    # resuming (e.g. after `ollama pull`) must retry the failed responses, not reuse them
    msgs = []
    run(cfg, resume=run_dir, progress=msgs.append)
    assert any("generating: flaky (3 cases)" in m for m in msgs)  # errors retried, successes reused
    assert not any(m.startswith("generating: extractive") for m in msgs)
    assert json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))["hardware"]["extractive"]["source"] == "mock"


def test_calibration_flags_disagreement(tmp_path, corpus):  # noqa: F811
    run_dir = _rag_run(tmp_path, corpus)
    labels = tmp_path / "labels.csv"
    export_sample(run_dir, labels, n=25)
    with open(labels, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    results = {json.loads(line)["rkey"]: json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()}
    for r in rows:  # human says the opposite of the judge
        v = next(s["value"] for s in results[r["key"]]["scores"] if s["name"] == "answer_correctness")
        r["human_answer_correctness"] = str(round(5 - 4 * v))
    with open(labels, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert score_labels(run_dir, labels)["scorers"]["answer_correctness"]["verdict"] == "fix rubric or judge"
