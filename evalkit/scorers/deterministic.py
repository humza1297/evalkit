from __future__ import annotations

import json
import re

from ..core.registry import register
from ..core.types import Case, Response, Score
from ..core.util import normalize
from . import Scorer

# Kept narrow on purpose: phrases like "not included in the price" are legitimate answers.
# Hand-check the refusal column on real outputs before trusting it.
_SRC = r"(?:the )?(?:provided |given |available )?(?:documents?|context|information|sources?|materials?|text|records?)"
REFUSAL_PATTERNS = [
    r"\bi (?:do not|don't|cannot|can't|am unable to) (?:know|find|answer|determine|say|see|tell)\b",
    rf"\b(?:isn't|is not|not|aren't|are not) (?:mentioned|specified|stated|covered|provided|included|available|addressed) in {_SRC}",
    r"\b(?:i have|there is|there's|i've got) no (?:information|details?|data|record)\b",
    r"\bi (?:do not|don't) have (?:any |enough )?(?:information|details?|data)\b",
    rf"\b{_SRC} (?:do(?:es)? not|don't|doesn't) (?:mention|say|specify|state|contain|include|cover|provide|address)\b",
    r"\b(?:unable|not able) to (?:find|answer|determine|locate)\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


@register("scorer", "exact_match")
class ExactMatch(Scorer):
    def score(self, case: Case, resp: Response) -> Score:
        if case.expected is None or case.unanswerable:
            return self.skip("no expected answer")
        return self.ok(normalize(resp.output) == normalize(case.expected))


@register("scorer", "contains")
class Contains(Scorer):
    """Every metadata.must_contain item appears in the output (normalized). An item may be a list of
    alternatives, e.g. [["5 percent", "5%"]]. Only applies when must_contain is set: full reference answers
    get paraphrased, so use answer_correctness for those."""

    def score(self, case: Case, resp: Response) -> Score:
        needles = case.metadata.get("must_contain") or []
        if not needles:
            return self.skip("no must_contain")
        out = normalize(resp.output)
        raw = (resp.output or "").lower()
        found = []
        for n in needles:
            alts = n if isinstance(n, list) else [n]
            if any(normalize(a) in out or a.lower() in raw for a in alts):
                found.append(n)
        return self.ok(len(found) / len(needles), f"found {len(found)}/{len(needles)}")


@register("scorer", "must_not_contain")
class MustNotContain(Scorer):
    """1.0 if none of the forbidden phrases appear (params.phrases + metadata.must_not_contain), e.g. competitor
    names, internal codes, phone numbers of staff who left. Phrases are case-insensitive; prefix 're:' for regex."""

    def score(self, case: Case, resp: Response) -> Score:
        phrases = list(self.params.get("phrases") or []) + list(case.metadata.get("must_not_contain") or [])
        if not phrases:
            return self.skip("no forbidden phrases")
        text = resp.output or ""
        hits = []
        for p in phrases:
            if p.startswith("re:"):
                if re.search(p[3:], text, re.I):
                    hits.append(p)
            elif p.lower() in text.lower():
                hits.append(p)
        return self.ok(0.0 if hits else 1.0, f"forbidden: {', '.join(hits)}" if hits else "clean")


@register("scorer", "regex")
class Regex(Scorer):
    def score(self, case: Case, resp: Response) -> Score:
        pattern = case.metadata.get("regex") or self.params.get("pattern")
        if not pattern:
            return self.skip("no pattern")
        try:
            return self.ok(bool(re.search(pattern, resp.output or "", re.I | re.S)))
        except re.error as e:
            return self.fail(f"bad regex: {e}")


@register("scorer", "json_valid")
class JsonValid(Scorer):
    def score(self, case: Case, resp: Response) -> Score:
        text = (resp.output or "").strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            json.loads(text)
            return self.ok(1.0)
        except ValueError as e:
            return self.ok(0.0, f"invalid JSON: {e}")


@register("scorer", "retrieval_hit")
class RetrievalHit(Scorer):
    """Fraction of the case's gold source docs (metadata.source_docs / source_doc) that were retrieved."""

    def score(self, case: Case, resp: Response) -> Score:
        gold = case.metadata.get("source_docs") or ([case.metadata["source_doc"]] if case.metadata.get("source_doc") else [])
        if not gold or case.unanswerable:
            return self.skip("no gold source docs")
        if not resp.retrieved:
            return self.skip("target does not expose retrieval")
        got = {c.doc_id for c in resp.retrieved}
        hits = [g for g in gold if g in got]
        return self.ok(len(hits) / len(gold), f"retrieved {len(hits)}/{len(gold)} gold docs")


@register("scorer", "refusal")
class RefusalWhenUnknown(Scorer):
    """Unanswerable cases only: 1 if the system declines instead of inventing an answer.
    Answerable cases are skipped: rewarding "didn't refuse" would rank a confident wrong answer above an
    honest "I don't know". False refusals are scored by answer_correctness (refusing ranks just above wrong)."""

    def score(self, case: Case, resp: Response) -> Score:
        if not case.unanswerable:
            return self.skip("answerable case")
        refused = is_refusal(resp.output)
        return self.ok(refused, "correctly refused" if refused else "answered an unanswerable question")
