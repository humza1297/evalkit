import json

import pytest

from evalkit.core.config import load_config
from evalkit.core.runner import run
from evalkit.core.util import strip_reasoning
from evalkit.report import aggregate, budget_score, load_results
from evalkit.targets import EmptyOutputError, clean_output

from .conftest import write_config


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm</think>\nThe fee is 5%.") == "The fee is 5%."
    assert strip_reasoning("<think>still thinking, cut off") == ""
    assert strip_reasoning("plain") == "plain"
    with pytest.raises(EmptyOutputError):
        clean_output("<think>never finished")


def test_budget_score():
    assert budget_score(0.5, 1.0) == 1.0
    assert budget_score(2.0, 1.0) == 0.5
    assert budget_score(None, 1.0) == 0.0


def test_all_error_target_gets_no_speed_credit(project):
    cfg = load_config(project)
    cfg.targets.append({"name": "broken", "type": "model", "model": "mock/nonexistent"})
    run_dir = run(cfg, progress=lambda *_: None)
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    b = s["targets"]["broken"]
    assert b["quality"] == 0.0 and b["composite"] == 0.0 and b["gated"]
    assert s["ranking"][-1] == "broken"


def test_rescoring_reuses_responses(project):
    """Changing scorers must not regenerate responses (matters for live HTTP targets)."""
    cfg = load_config(project)
    run_dir = run(cfg, progress=lambda *_: None)
    n_resp = len((run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines())
    cfg.scorers = [{"name": "refusal", "weight": 2.0}]
    msgs = []
    run(cfg, resume=run_dir, progress=msgs.append)
    assert not any(m.startswith("generating") for m in msgs)
    assert len((run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == n_resp
    meta, results = load_results(run_dir)
    assert len(results) == 6 and all(len(r.scores) == 1 for r in results)


def test_min_quality_gate(tmp_path, project):
    cfg = load_config(project)
    cfg.budgets = {"min_quality": 0.9}
    run_dir = run(cfg, progress=lambda *_: None)
    meta, results = load_results(run_dir)
    s = aggregate(meta, results)
    lazy = s["targets"]["lazy"]
    assert lazy["gated"] and lazy["composite"] == pytest.approx(0.6 * lazy["quality"])


def test_generator_flags_corpus_hits(tmp_path):
    from evalkit.core.env import Env
    from evalkit.datasets.generator import generate_cases

    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("Pets are welcome in the office on Fridays. " * 5, encoding="utf-8")
    (d / "b.md").write_text("Parking is free behind the building for all clients. " * 5, encoding="utf-8")
    cases, _ = generate_cases(Env(), d, "mock/generator", n=5, mix={"unanswerable": 1.0}, progress=lambda *_: None)
    assert cases and all("check_hits" in c.metadata for c in cases)
