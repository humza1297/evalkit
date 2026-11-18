import json
from pathlib import Path

import pytest
import yaml

from evalkit.core.registry import load_builtins

load_builtins()


def write_cases(path: Path, cases: list[dict]) -> Path:
    path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    return path


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


CASES = [
    {"id": "c1", "input": "What is the office phone number?", "expected": "555-0100",
     "context": "The office phone number is 555-0100. We open at 9am.", "tags": ["fact"], "metadata": {"must_contain": ["555-0100"]}},
    {"id": "c2", "input": "What time does the office open?", "expected": "9am",
     "context": "The office phone number is 555-0100. The office opens at 9am on weekdays.", "tags": ["fact"],
     "metadata": {"must_contain": ["9am"]}},
    {"id": "c3", "input": "Who is the CEO of Mars?", "context": "The office opens at 9am.", "tags": ["unanswerable"]},
]


@pytest.fixture
def project(tmp_path):
    write_cases(tmp_path / "eval.jsonl", CASES)
    cfg = {
        "project": "test",
        "dataset": "eval.jsonl",
        "concurrency": 2,
        "targets": [
            {"name": "extractive", "type": "model", "model": "mock/extractive"},
            {"name": "lazy", "type": "model", "model": "mock/lazy"},
        ],
        "scorers": ["contains", "refusal"],
    }
    return write_config(tmp_path / "evalkit.yaml", cfg)
