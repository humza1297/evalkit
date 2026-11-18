import json
import os
import stat

import pytest

from evalkit.cli import main
from evalkit.core.env import ConfidentialityError, Env
from evalkit.core.types import Case
from evalkit.datasets.generator import _allocate, generate_cases, parse_generated
from evalkit.datasets.review import review
from evalkit.datasets.store import freeze, load_cases, save_cases

from .conftest import write_config
from .test_rag_http import corpus  # noqa: F401  (fixture)


def test_parse_generated():
    assert parse_generated('{"question": "What is the fee?", "answer": "5%"}') == ("What is the fee?", "5%", [])
    assert parse_generated('<think>hm</think> ```json\n{"question": "Is there parking?", "answer": null}```') == ("Is there parking?", None, [])
    assert parse_generated("no json") is None
    nested = ('{"history": [{"role": "user", "content": "Fee?"}, {"role": "assistant", "content": "5%."}], '
              '"question": "And for luxury homes?", "answer": "4%"}')
    q, a, h = parse_generated(nested)
    assert q == "And for luxury homes?" and a == "4%" and len(h) == 2


def test_allocate():
    c = _allocate(10, {"fact": 0.4, "numeric": 0.2, "procedural": 0.1, "multi_doc": 0.1, "unanswerable": 0.2})
    assert sum(c.values()) == 10 and c["fact"] == 4


def test_generate_cases_mock(corpus):  # noqa: F811
    cases, errors = generate_cases(Env(), corpus, "mock/generator", n=8, progress=lambda *_: None)
    assert len(cases) >= 5, errors
    assert all(c.status == "pending" and "generated" in c.tags for c in cases)
    un = [c for c in cases if c.unanswerable]
    assert un and all(c.expected is None for c in un)
    assert all(c.metadata["source_docs"] for c in cases if not c.unanswerable)


def test_generator_respects_guard(corpus):  # noqa: F811
    with pytest.raises(ConfidentialityError):
        generate_cases(Env(confidential=True), corpus, "groq/some-model", n=2, progress=lambda *_: None)


def test_review_scripted(tmp_path):
    path = tmp_path / "eval.jsonl"
    save_cases(path, [Case(id=str(i), input=f"q{i}", expected=f"a{i}", status="pending") for i in range(5)])
    answers = iter(["a", "r", "e", "better question?", "-", "s", "q"])
    counts = review(path, input_fn=lambda _: next(answers), print_fn=lambda *_: None)
    assert counts == {"accepted": 1, "edited": 1, "rejected": 1, "skipped": 1, "remaining": 1}
    all_cases = {c.id: c for c in load_cases(path, include_all=True)}
    assert all_cases["2"].input == "better question?" and all_cases["2"].unanswerable and all_cases["2"].expected is None
    assert [c.id for c in load_cases(path)] == ["0", "2"]


def test_freeze(tmp_path):
    path = tmp_path / "eval.jsonl"
    save_cases(path, [Case(id="1", input="q", status="accepted"), Case(id="2", input="q2", status="pending")])
    out = freeze(path)
    assert out.name == "eval.v1.jsonl"
    assert [c.id for c in load_cases(out)] == ["1"]
    assert not os.stat(out).st_mode & stat.S_IWRITE
    assert freeze(path).name == "eval.v2.jsonl"
    for p in tmp_path.glob("eval.v*.jsonl"):
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


def test_cli_build_review_run(tmp_path, corpus, capsys):  # noqa: F811
    cfg = write_config(tmp_path / "evalkit.yaml", {
        "project": "cli", "corpus": "docs", "dataset": "eval.jsonl", "pack": "rag", "judge": "mock/judge",
        "generator": {"model": "mock/generator", "n": 6},
        "targets": [{"name": "rag", "type": "rag", "model": "mock/extractive"}],
    })
    assert main(["build-dataset", str(cfg)]) == 0
    assert main(["run", str(cfg)]) == 2  # nothing reviewed yet -> no usable cases
    assert main(["review", str(tmp_path / "eval.jsonl"), "--accept-all"]) == 0
    assert main(["check", str(cfg)]) == 0
    assert main(["run", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "Ranking:" in out and "mock run" in out
    run_dirs = list((tmp_path / "runs").iterdir())
    assert (run_dirs[0] / "report.html").exists()
    assert main(["report", str(run_dirs[0])]) == 0
    assert json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))["targets"]["rag"]["n"] >= 4
