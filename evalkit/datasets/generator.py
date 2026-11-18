"""Generate draft test cases from a client's documents. Every generated case is status=pending until
someone reviews it: generated cases are NOT ground truth until reviewed."""
from __future__ import annotations

import json
import random
import re
from typing import Callable

from ..core.env import ConfidentialityError, Env
from ..core.types import Case, Chunk
from ..core.util import normalize, stable_hash
from ..loaders import ingest
from ..rag.chunking import chunk_documents

TYPES = {
    "fact": "a question answered by a single specific fact stated in the passage",
    "numeric": "a question whose answer is a number, price, percentage, date, duration or deadline stated in the passage",
    "procedural": "a 'how do I' or 'what are the steps' question that the passage answers",
    "multi_doc": "a question that can only be answered by combining information from BOTH passages",
    "unanswerable": "a realistic question on the same topic that the passage does NOT answer (the answer must be absent)",
    "followup": (
        "short conversation: the customer asks a first question that the passage answers, the assistant answers it, "
        "then the customer asks a FOLLOW-UP question that only makes sense given the earlier turns "
        "(it uses words like 'that', 'it' or 'what about...') and that the passage also answers"
    ),
}
DEFAULT_MIX = {"fact": 0.4, "numeric": 0.2, "procedural": 0.1, "multi_doc": 0.1, "unanswerable": 0.2}
CHATBOT_MIX = {"fact": 0.25, "numeric": 0.15, "procedural": 0.1, "followup": 0.3, "unanswerable": 0.2}

FOLLOWUP_FORMAT = ('Reply with ONLY a JSON object: {"history": [{"role": "user", "content": "..."}, '
                   '{"role": "assistant", "content": "..."}], "question": "<the follow-up>", "answer": "<answer to the follow-up>"}')

PROMPT = """You are writing test questions for an AI assistant used by {business}.
Write {desc}.
Phrase the question the way {persona} would naturally ask it, without quoting the passage verbatim.
{answer_rule}
{format}
### TYPE: {type}
{passages}"""

SIMPLE_FORMAT = 'Reply with ONLY a JSON object: {"question": "...", "answer": ...}'
ANSWER_RULE = "The answer must be short, specific and fully supported by the passage(s)."
UNANSWERABLE_RULE = 'Set "answer" to null.'


def _json_objects(text: str):
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text, i)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def parse_generated(text: str) -> tuple[str, str | None, list[dict]] | None:
    """-> (question, answer or None, history turns)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    for obj in _json_objects(text):
        q = obj.get("question")
        if isinstance(q, str) and len(q.strip()) > 5:
            a = obj.get("answer")
            hist = [{"role": t["role"], "content": str(t["content"])} for t in obj.get("history") or []
                    if isinstance(t, dict) and t.get("role") in ("user", "assistant") and t.get("content")]
            return q.strip(), (str(a).strip() if a not in (None, "", "null") else None), hist
    return None


def _allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    mix = {k: v for k, v in mix.items() if k in TYPES and v > 0}
    total = sum(mix.values()) or 1
    counts = {k: int(n * v / total) for k, v in mix.items()}
    rest = sorted(mix, key=lambda k: -(n * mix[k] / total - counts[k]))
    for k in rest[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def generate_cases(env: Env, corpus, model: str, n: int = 20, mix: dict[str, float] | None = None,
                   business: str = "a small business", persona: str = "a client of the business",
                   seed: int = 0, existing: list[Case] | None = None, personas: list[str] | None = None,
                   progress: Callable[[str], None] = print) -> tuple[list[Case], list[str]]:
    """personas: optional list of customer personas, cycled across cases (e.g. "an anxious first-time buyer")."""
    personas = list(personas or []) or [persona]
    env.resolve(model, "dataset generator")  # confidentiality guard before any document is read
    docs, _ = ingest(corpus)
    chunks = [c for c in chunk_documents(docs, 1200, 0) if len(c.text) >= 120] or chunk_documents(docs, 1200, 0)
    if not chunks:
        raise ValueError(f"no text found in corpus {corpus}")
    rng = random.Random(seed)
    order = chunks[:]
    rng.shuffle(order)
    seen = {normalize(c.input) for c in existing or []}
    counts = _allocate(n, mix or DEFAULT_MIX)
    multi_ok = len({c.doc_id for c in chunks}) > 1
    if not multi_ok and counts.get("multi_doc"):
        counts["fact"] = counts.get("fact", 0) + counts.pop("multi_doc")
    from ..rag.retrieval import BM25

    corpus_index = BM25(env, {})  # checks "unanswerable" questions against the WHOLE corpus, not one passage
    corpus_index.index(chunks)
    cases: list[Case] = []
    errors: list[str] = []
    i = 0
    for qtype, k in counts.items():
        made, attempts = 0, 0
        while made < k and attempts < k * 3:
            attempts += 1
            first = order[i % len(order)]
            i += 1
            picked: list[Chunk] = [first]
            if qtype == "multi_doc":
                others = [c for c in chunks if c.doc_id != first.doc_id]
                picked.append(rng.choice(others))
            passages = "\n".join(f"### PASSAGE {j + 1} (source: {c.doc_id})\n{c.text}\n" for j, c in enumerate(picked))
            who = personas[(len(cases) + attempts) % len(personas)]
            desc = TYPES[qtype] if qtype == "followup" else "ONE " + TYPES[qtype]
            prompt = PROMPT.format(business=business, persona=who, desc=desc, type=qtype, passages=passages,
                                   answer_rule=UNANSWERABLE_RULE if qtype == "unanswerable" else ANSWER_RULE,
                                   format=FOLLOWUP_FORMAT if qtype == "followup" else SIMPLE_FORMAT)
            try:
                r = env.chat(model, [{"role": "user", "content": prompt}], "dataset generator",
                             temperature=0.7, max_tokens=400, use_cache=False)
            except ConfidentialityError:
                raise
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
                if len(errors) >= 5 and not cases:
                    raise RuntimeError(f"generator keeps failing: {errors[-1]}") from e
                continue
            parsed = parse_generated(r.text)
            if not parsed:
                errors.append(f"unparseable generator output: {r.text[:100]!r}")
                continue
            q, a, history = parsed
            if qtype != "unanswerable" and not a:
                errors.append(f"no answer for {qtype} question: {q[:80]}")
                continue
            if qtype == "followup" and not history:
                errors.append(f"follow-up without conversation history: {q[:80]}")
                continue
            if normalize(q) in seen:
                continue
            seen.add(normalize(q))
            sources = sorted({c.doc_id for c in picked})
            meta = {"source_docs": [] if qtype == "unanswerable" else sources, "near_docs": sources, "generator": model,
                    "persona": who}
            if qtype == "unanswerable":
                meta["check_hits"] = [{"doc": h.doc_id, "score": h.score, "snippet": h.text[:200]}
                                      for h in corpus_index.search(q, 3)]
            cases.append(Case(
                id=f"gen-{stable_hash([q, sources, history], 10)}", input=q, expected=None if qtype == "unanswerable" else a,
                tags=[qtype, "generated"], status="pending", metadata=meta, history=history,
            ))
            made += 1
        progress(f"  {qtype}: {made}/{k}")
    return cases, errors
