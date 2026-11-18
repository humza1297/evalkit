from __future__ import annotations

import hashlib
import json
import re
import string
from typing import Any, Iterable

_PUNCT = str.maketrans("", "", string.punctuation)

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    this that these those it its what which who whom whose when where why how do does did
    can could should would will shall may might must i you he she we they me my your our their
    not no yes there here than then so such about into over under any all some each per""".split()
)


def stable_hash(obj: Any, length: int = 16) -> str:
    data = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:length]


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.lower().translate(_PUNCT).split())


def tokens(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", (text or "").lower())


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def content_words(text: str | None) -> list[str]:
    return [_stem(t) for t in tokens(text) if t not in STOPWORDS and (len(t) > 2 or t.isdigit())]


def f1_overlap(pred: str, ref: str) -> float:
    p, r = content_words(pred), content_words(ref)
    if not p or not r:
        return 0.0
    common: dict[str, int] = {}
    rc: dict[str, int] = {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    for t in p:
        if rc.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            rc[t] -= 1
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(r)
    return 2 * precision * recall / (precision + recall)


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str | None) -> str:
    """Remove <think>...</think> blocks (qwen3, deepseek-r1...). An unclosed <think> means output was truncated mid-reasoning."""
    text = _THINK_BLOCK.sub("", text or "")
    if re.search(r"<think>", text, re.I):
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip()


def fill(template: str, **values: str) -> str:
    """Replace {name} placeholders without str.format (templates and data may contain braces)."""
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", val if val is not None else "")
    return out
