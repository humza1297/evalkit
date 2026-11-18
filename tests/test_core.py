import json

import pytest

from evalkit.core.cache import Cache
from evalkit.core.config import ConfigError, load_config
from evalkit.core.env import ConfidentialityError, Env, is_local_host
from evalkit.core.runner import case_quality, run
from evalkit.core.types import Case, Response, Score

from .conftest import write_config


def test_is_local_host():
    assert is_local_host("localhost")
    assert is_local_host("127.0.0.1")
    assert is_local_host("::1")
    assert is_local_host("192.168.1.20")
    assert is_local_host("10.0.0.5")
    assert not is_local_host("api.groq.com")
    assert not is_local_host("8.8.8.8")
    assert not is_local_host(None)
    assert is_local_host("client-bot", ["client-bot"])


def test_guard_blocks_remote_for_confidential():
    env = Env(confidential=True)
    with pytest.raises(ConfidentialityError):
        env.resolve("groq/llama-3.1-8b-instant", "target")
    with pytest.raises(ConfidentialityError):
        env.check_url("https://example.com/chat", "http target")
    p, model = env.resolve("ollama/llama3.2:3b", "target")
    assert p.local and model == "llama3.2:3b"
    env.check_url("http://localhost:8080/chat", "http target")


def test_guard_allows_remote_when_not_confidential():
    env = Env(confidential=False)
    p, _ = env.resolve("groq/llama-3.1-8b-instant", "target")
    assert not p.local


def test_bad_model_spec():
    with pytest.raises(ValueError):
        Env().resolve("llama3", "target")
    with pytest.raises(ValueError):
        Env().resolve("nosuchprovider/x", "target")


def test_chat_cache_preserves_latency():
    env = Env(cache=Cache(None))
    msgs = [{"role": "user", "content": "Context:\nThe sky is blue today.\n\nQuestion: What color is the sky today?"}]
    a = env.chat("mock/extractive", msgs)
    b = env.chat("mock/extractive", msgs)
    assert not a.cached and b.cached
    assert a.text == b.text and a.latency_s == b.latency_s > 0


def test_case_quality_rules():
    ok = Response(output="x")
    assert case_quality(ok, [Score("a", 1.0), Score("b", 0.0)], {"a": 3, "b": 1}) == 0.75
    assert case_quality(ok, [Score("a", None, error="judge broke"), Score("b", 0.5)], {}) == 0.5
    assert case_quality(ok, [Score("a", None, reason="n/a")], {}) is None
    assert case_quality(Response(error="boom"), [], {}) == 0.0


def test_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path / "a.yaml", {"targetz": []}))
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path / "b.yaml", {"targets": [{"model": "mock/x"}, {"model": "mock/x"}]}))
    cfg = load_config(write_config(tmp_path / "c.yaml", {"pack": "rag", "targets": [{"model": "mock/x"}]}))
    assert cfg.confidential is True
    assert any(s["name"] == "faithfulness" for s in cfg.scorers)
    assert cfg.targets[0]["system"]


def test_case_roundtrip():
    c = Case.from_dict({"id": 1, "input": "q", "expected": "a", "source_doc": "d1"})
    assert c.id == "1" and c.metadata["source_doc"] == "d1"
    assert Case.from_dict(c.to_dict()).content_hash() == c.content_hash()


def test_end_to_end_mock(project):
    cfg = load_config(project)
    run_dir = run(cfg, progress=lambda *_: None)
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["is_mock"] is True
    assert summary["ranking"][0] == "extractive"
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "MOCK RUN" in html and "extractive" in html


def test_resume_skips_done(project):
    cfg = load_config(project)
    run_dir = run(cfg, progress=lambda *_: None)
    msgs = []
    run(cfg, resume=run_dir, progress=msgs.append)
    assert "0 to run" in msgs[0]
    assert len((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_confidential_run_fails_fast(project, tmp_path):
    cfg = load_config(project)
    cfg.targets.append({"name": "remote", "type": "model", "model": "groq/llama-3.1-8b-instant"})
    with pytest.raises(ConfidentialityError):
        run(cfg, progress=lambda *_: None)


def test_keys_distinguish_duplicate_cases_and_twin_targets(tmp_path):
    from .conftest import write_cases

    write_cases(tmp_path / "eval.jsonl", [{"id": "a", "input": "same?"}, {"id": "b", "input": "same?"}])
    cfg = load_config(write_config(tmp_path / "e.yaml", {
        "dataset": "eval.jsonl", "scorers": ["refusal"],
        "targets": [{"name": "t1", "model": "mock/lazy"}, {"name": "t2", "model": "mock/lazy"}],
    }))
    run_dir = run(cfg, progress=lambda *_: None)
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["targets"]["t1"]["n"] == 2 and s["targets"]["t2"]["n"] == 2
    assert s["meta"]["hardware"]["t1"]["source"] == "mock"


def test_target_error_recorded_not_fatal(project):
    cfg = load_config(project)
    cfg.targets.append({"name": "badmock", "type": "model", "model": "mock/nonexistent"})
    run_dir = run(cfg, progress=lambda *_: None)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bad = summary["targets"]["badmock"]
    assert bad["errors"] == 3 and bad["quality"] == 0.0
