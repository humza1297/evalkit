"""Dataset storage: JSONL of cases. Frozen datasets are versioned and read-only."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ..core.types import Case
from ..core.util import stable_hash

USABLE_STATUSES = (None, "accepted")


def load_cases(path: Path, include_all: bool = False) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                case = Case.from_dict(json.loads(line))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id '{case.id}'")
            seen.add(case.id)
            if include_all or case.status in USABLE_STATUSES:
                cases.append(case)
    return cases


def save_cases(path: Path, cases: list[Case]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def dataset_hash(cases: list[Case]) -> str:
    return stable_hash([c.to_dict() for c in cases])


def freeze(path: Path) -> Path:
    """Write accepted cases to <stem>.v<N>.jsonl (read-only). Returns the new path."""
    cases = load_cases(path)  # accepted / unreviewed-by-design only
    for c in cases:
        c.status = None
    n = 1
    while (out := path.with_name(f"{path.stem}.v{n}.jsonl")).exists():
        n += 1
    save_cases(out, cases)
    os.chmod(out, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return out
