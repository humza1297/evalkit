import pytest

from evalkit.core.config import ProjectConfig
from evalkit.core.env import Env
from evalkit.core.registry import get
from evalkit.core.types import Case, Chunk, Response
from evalkit.scorers.judge import parse_judge


@pytest.mark.parametrize("text,expected", [
    ('{"score": 4, "reason": "good"}', 4.0),
    ('Sure! ```json\n{"score": 5, "reason": "ok"}\n```', 5.0),
    ('<think>{"score": 1}</think>{"score": 3, "reason": "partial"}', 3.0),
    ('{"score": "2", "reason": "x"}', 2.0),
    ("Score: 4/5 because it was mostly right", 4.0),
    ('{"score": 9}', None),
    ("I think it's pretty good", None),
    ("", None),
])
def test_parse_judge(text, expected):
    r = parse_judge(text)
    assert (r[0] if r else None) == expected


def _scorer(name, judge, tmp_path, **params):
    proj = ProjectConfig(path=tmp_path / "x.yaml", base_dir=tmp_path, project="t", judge=judge)
    s = get("scorer", name)({"name": name, **params}, Env(), proj)
    s.setup()
    return s


def test_broken_judge_is_error_not_zero(tmp_path):
    s = _scorer("answer_correctness", "mock/broken", tmp_path)
    sc = s.score(Case(id="1", input="q", expected="5 percent"), Response(output="5 percent"))
    assert sc.value is None and "unparseable" in sc.error


def test_correctness_and_faithfulness(tmp_path):
    ac = _scorer("answer_correctness", "mock/judge", tmp_path)
    good = ac.score(Case(id="1", input="commission?", expected="The commission is 5 percent"), Response(output="Commission is 5 percent."))
    bad = ac.score(Case(id="1", input="commission?", expected="The commission is 5 percent"), Response(output="Parking is free."))
    assert good.value > bad.value
    assert ac.score(Case(id="2", input="q", tags=["unanswerable"]), Response(output="x")).value is None
    f = _scorer("faithfulness", "mock/judge", tmp_path)
    resp = Response(output="The office opens at 9am.", retrieved=[Chunk("c", "office.txt", "The office opens at 9am on weekdays.")])
    assert f.score(Case(id="3", input="q"), resp).value == 1.0
    assert f.score(Case(id="3", input="q"), Response(output="I don't know.")).value is None


def test_judge_required(tmp_path):
    with pytest.raises(ValueError):
        _scorer("faithfulness", None, tmp_path)
    with pytest.raises(ValueError):
        _scorer("rubric", "mock/judge", tmp_path)
