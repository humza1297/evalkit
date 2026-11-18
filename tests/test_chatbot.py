import json

import pytest

from evalkit.core.config import ProjectConfig, load_config
from evalkit.core.env import Env
from evalkit.core.registry import get
from evalkit.core.runner import run
from evalkit.core.types import Case, Response
from evalkit.datasets.generator import CHATBOT_MIX, generate_cases
from evalkit.targets.http import _render
from evalkit.targets.model import build_messages

from .conftest import write_cases, write_config
from .test_rag_http import corpus  # noqa: F401

HISTORY = [{"role": "user", "content": "What is the listing commission?"},
           {"role": "assistant", "content": "It is 5 percent of the sale price."}]


def test_case_history_validation():
    c = Case.from_dict({"id": "1", "input": "And for luxury homes?", "history": HISTORY})
    assert c.history == HISTORY and "ASSISTANT: It is 5 percent" in c.transcript()
    assert c.content_hash() != Case.from_dict({"id": "1", "input": "And for luxury homes?"}).content_hash()
    with pytest.raises(ValueError):
        Case.from_dict({"id": "2", "input": "x", "history": [{"role": "system", "content": "hi"}]})


def test_messages_include_history():
    msgs = build_messages("sys", "And for luxury homes?", HISTORY)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_http_history_placeholder():
    body = _render({"messages": "{history}", "text": "{input}"}, {"history": HISTORY, "input": "hi {x}"})
    assert body == {"messages": HISTORY, "text": "hi {x}"}


def _scorer(name, tmp_path, **params):
    proj = ProjectConfig(path=tmp_path / "x.yaml", base_dir=tmp_path, project="t", judge="mock/judge")
    s = get("scorer", name)({"name": name, **params}, Env(), proj)
    s.setup()
    return s


def test_tone_policy_and_forbidden(tmp_path):
    c = Case(id="1", input="Can you lower my commission?", history=HISTORY)
    assert _scorer("tone", tmp_path).score(c, Response(output="Happy to help!")).value is not None
    with pytest.raises(ValueError):
        _scorer("policy", tmp_path)
    assert _scorer("policy", tmp_path, rules=["Never promise a discount"]).score(c, Response(output="Sure.")).value is not None
    mnc = _scorer("must_not_contain", tmp_path, phrases=["Acme Realty", r"re:\b\d{3}-\d{2}-\d{4}\b"])
    assert mnc.score(c, Response(output="Try Acme Realty instead")).value == 0.0
    assert mnc.score(c, Response(output="SSN 123-45-6789")).value == 0.0
    assert mnc.score(c, Response(output="We can help.")).value == 1.0


def test_followup_generation_and_chatbot_run(tmp_path, corpus):  # noqa: F811
    cases, errors = generate_cases(Env(), corpus, "mock/generator", n=10, mix=CHATBOT_MIX,
                                   personas=["an anxious first-time buyer", "a busy investor"], progress=lambda *_: None)
    fu = [c for c in cases if "followup" in c.tags]
    assert fu and all(len(c.history) == 2 for c in fu), errors
    assert {c.metadata["persona"] for c in cases} == {"an anxious first-time buyer", "a busy investor"}

    write_cases(tmp_path / "eval.jsonl", [{**c.to_dict(), "status": "accepted"} for c in cases])
    cfg = write_config(tmp_path / "evalkit.yaml", {
        "project": "bot", "corpus": "docs", "dataset": "eval.jsonl", "pack": "chatbot", "judge": "mock/judge",
        "targets": [{"name": "bot", "type": "rag", "model": "mock/extractive"}],
    })
    run_dir = run(load_config(cfg), progress=lambda *_: None)
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert s["targets"]["bot"]["n"] == len(cases)
    assert "tone" in s["scorer_names"] and "followup" in s["targets"]["bot"]["tags"]
