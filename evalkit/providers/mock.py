"""Deterministic offline provider for tests and demos. Its outputs are NOT a real model ranking.

Models:
  mock/extractive        answers with the context sentence that best overlaps the question
  mock/extractive-large  same, but simulated as slower (for latency/cost demos)
  mock/lazy              always says it doesn't know
  mock/echo              returns the user message
  mock/broken            returns garbage (for judge-parsing tests)
  mock/judge             scores '### ANSWER' against '### REFERENCE' or '### CONTEXT', replies JSON
  mock/generator         fake test-case generator (JSON question/answer from a passage)
  any model for embed()  hashed bag-of-words vectors
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from ..core.registry import register
from ..core.util import content_words, f1_overlap
from .base import ChatResult, Provider

REFUSAL = "I don't know based on the provided documents."
_LATENCY_FACTOR = {"extractive": 1.0, "extractive-large": 3.5, "lazy": 0.3, "echo": 0.2, "broken": 0.5, "judge": 1.0}


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^###\s*([A-Z ]+)\s*$", text, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].strip()] = parts[i + 1].strip()
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]


def _extract(user: str) -> str:
    m = re.search(r"Context:\s*(.*?)\n\s*Question:\s*(.*)", user, flags=re.S | re.I)
    context, question = (m.group(1), m.group(2)) if m else (user, user)
    qwords = set(content_words(question))
    if not qwords:
        return REFUSAL
    best, best_overlap = "", 0
    for sent in _sentences(context):
        overlap = len(qwords & set(content_words(sent)))
        if overlap > best_overlap:
            best, best_overlap = sent, overlap
    if best_overlap < max(2, math.ceil(0.4 * len(qwords))):
        return REFUSAL
    return best


def _generate(user: str, call_no: int) -> str:
    """Fake test-case generator: question from a passage's key words, answer = that sentence."""
    qtype = (re.search(r"^### TYPE:\s*(\w+)", user, re.M) or [None, "fact"])[1]
    passage = re.split(r"^### PASSAGE.*$", user, flags=re.M)
    sents = _sentences(passage[1] if len(passage) > 1 else user)
    sents = [s for s in sents if len(content_words(s)) >= 3] or sents or ["(empty)"]
    sent = sents[(int(hashlib.md5(user.encode("utf-8")).hexdigest(), 16) + call_no) % len(sents)]
    words = content_words(sent)[:4]
    topic = " ".join(words)
    if qtype == "unanswerable":
        places = ["Antarctica", "the Moon", "Atlantis", "Narnia", "Mars", "Pluto"]
        return json.dumps({"question": f"What is the policy on {' '.join(words[:2])} for clients in {places[call_no % len(places)]}?", "answer": None})
    if qtype == "followup":
        return json.dumps({"history": [{"role": "user", "content": f"Tell me about {topic}."},
                                       {"role": "assistant", "content": sent}],
                           "question": f"And how does that work for {['new', 'returning', 'out-of-state'][call_no % 3]} clients?",
                           "answer": sent})
    templates = {"fact": "What does the office say about {t}?", "numeric": "What are the numbers for {t}?",
                 "procedural": "How do I handle {t}?", "multi_doc": "How do the policies on {t} fit together?"}
    return json.dumps({"question": templates.get(qtype, templates["fact"]).format(t=topic), "answer": sent})


def _judge(user: str) -> str:
    s = _sections(user)
    answer = s.get("ANSWER", "")
    if "REFERENCE" in s and REFUSAL.lower()[:12] in answer.lower():
        return json.dumps({"score": 2, "reason": "mock: declined to answer"})
    if "REFERENCE" in s:
        x = f1_overlap(answer, s["REFERENCE"])
    elif "CONTEXT" in s:
        aw = content_words(answer)
        cw = set(content_words(s["CONTEXT"]))
        x = sum(w in cw for w in aw) / len(aw) if aw else 1.0
    else:
        x = 0.6
    score = 1 + round(4 * x)
    return json.dumps({"score": score, "reason": f"mock overlap={x:.2f}"})


@register("provider", "mock")
class MockProvider(Provider):
    base_url = "mock://local"
    local = True

    def chat(self, model, messages, temperature=0.0, max_tokens=512):
        user = messages[-1]["content"] if messages else ""
        kind = model.split(":")[0]
        if kind.startswith("extractive"):
            text = _extract(user)
        elif kind == "lazy":
            text = REFUSAL
        elif kind == "echo":
            text = user
        elif kind == "broken":
            text = "<<garbled output, no json here>>"
        elif kind == "judge":
            text = _judge(user)
        elif kind == "generator":
            self._calls = getattr(self, "_calls", 0) + 1
            text = _generate(user, self._calls)
        else:
            raise ValueError(f"unknown mock model '{model}'")
        pt = sum(len(m.get("content", "").split()) for m in messages)
        ct = len(text.split())
        latency = (0.15 + 0.004 * pt + 0.02 * ct) * _LATENCY_FACTOR.get(kind, 1.0)
        return ChatResult(text=text, prompt_tokens=pt, completion_tokens=ct, latency_s=round(latency, 4))

    def embed(self, model, texts, dim: int = 256):
        out = []
        for t in texts:
            v = [0.0] * dim
            for w in content_words(t):
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                v[h % dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out
