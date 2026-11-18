"""LLM-as-judge scorers. Judge replies are parsed defensively. An unparseable reply is a scorer
error (excluded from quality and counted in the report), never a silent 0."""
from __future__ import annotations

import json
import re

from ..core.registry import register
from ..core.types import Case, Response, Score
from .deterministic import is_refusal
from . import Scorer

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of answers produced by an AI assistant for a small business. "
    'Reply with ONLY a JSON object: {"score": <integer 1-5>, "reason": "<one short sentence>"}.'
)

CORRECTNESS = """Rate how well the ANSWER matches the REFERENCE answer for the QUESTION.
5 = fully correct and complete; 4 = correct with minor omissions; 3 = partially correct, nothing false;
2 = declines to answer or says it doesn't know (honest but unhelpful);
1 = states something wrong, contradicts the reference, or invents facts.
A wrong answer is worse than declining. Ignore wording and style differences; judge facts only.
{conversation_block}
### QUESTION
{question}

### REFERENCE
{reference}

### ANSWER
{answer}
"""

FAITHFULNESS = """Rate whether every factual claim in the ANSWER is supported by the CONTEXT.
5 = every claim is supported; 4 = one minor unsupported detail; 3 = some unsupported claims;
2 = mostly unsupported; 1 = contradicts the context or invents facts.
Do not judge whether the answer is helpful, only whether it is grounded in the context.

### CONTEXT
{context}

### ANSWER
{answer}
"""

RUBRIC = """Rate the ANSWER against these CRITERIA (5 = fully meets them, 1 = fails them).

### CRITERIA
{criteria}
{conversation_block}
### QUESTION
{question}
{reference_block}
### ANSWER
{answer}
"""

TONE_CRITERIA = (
    "The reply sounds like a friendly, professional small-business representative: polite, clear, concise, "
    "no jargon, no condescension, and it directly addresses what the customer asked."
)
POLICY_PREAMBLE = "The reply must follow ALL of these business rules. Any violation means a score of 1 or 2.\n"

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def parse_judge(text: str) -> tuple[float, str] | None:
    """Return (score 1..5, reason), or None if no valid score can be found."""
    text = _THINK.sub("", text or "")
    text = re.sub(r"```(?:json)?", "", text)
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            try:
                s = float(obj["score"])
            except (TypeError, ValueError):
                continue
            if 1 <= s <= 5:
                return s, str(obj.get("reason", ""))[:300]
    m = re.search(r"score\W{0,5}([1-5](?:\.\d+)?)\b", text, re.I)
    if m:
        return float(m.group(1)), text.strip()[:300]
    return None


class JudgeScorer(Scorer):
    def setup(self) -> None:
        self.model = self.params.get("model") or getattr(self.project, "judge", None)
        if not self.model:
            raise ValueError(f"scorer '{self.name}' needs a judge model: set 'judge:' in the config or 'model' on the scorer")
        self.env.resolve(self.model, f"judge for scorer '{self.name}'")

    def ask(self, prompt: str) -> Score:
        r = self.env.chat(self.model, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                          f"judge for scorer '{self.name}'", temperature=0.0, max_tokens=int(self.params.get("max_tokens", 300)))
        parsed = parse_judge(r.text)
        if parsed is None:
            return self.fail(f"unparseable judge output: {r.text[:150]!r}")
        s, reason = parsed
        return self.ok((s - 1) / 4, reason)


@register("scorer", "answer_correctness")
class AnswerCorrectness(JudgeScorer):
    def score(self, case: Case, resp: Response) -> Score:
        if not case.expected or case.unanswerable:
            return self.skip("no reference answer")
        conv = f"\n### CONVERSATION SO FAR\n{case.transcript()}\n" if case.history else ""
        return self.ask(CORRECTNESS.format(question=case.input, reference=case.expected, answer=resp.output, conversation_block=conv))


@register("scorer", "faithfulness")
class Faithfulness(JudgeScorer):
    def score(self, case: Case, resp: Response) -> Score:
        context = "\n\n".join(c.text for c in resp.retrieved) or case.context
        if not context:
            return self.skip("no context to check against")
        if is_refusal(resp.output):
            return self.skip("answer is a refusal")
        return self.ask(FAITHFULNESS.format(context=context, answer=resp.output))


@register("scorer", "rubric")
class Rubric(JudgeScorer):
    """Custom criteria. Use `as:` to run several rubrics side by side."""

    use_reference = True

    def criteria(self) -> str:
        return self.params.get("criteria", "")

    def setup(self) -> None:
        super().setup()
        if not self.criteria():
            raise ValueError(f"scorer '{self.name}': 'criteria' is required")

    def score(self, case: Case, resp: Response) -> Score:
        ref = f"\n### REFERENCE\n{case.expected}\n" if case.expected and self.use_reference else ""
        conv = f"\n### CONVERSATION SO FAR\n{case.transcript()}\n" if case.history else ""
        return self.ask(RUBRIC.format(criteria=self.criteria(), conversation_block=conv, question=case.input,
                                      reference_block=ref, answer=resp.output))


@register("scorer", "tone")
class Tone(Rubric):
    """Customer-facing tone. Override with `criteria:` for a different brand voice."""

    use_reference = False

    def criteria(self) -> str:
        return self.params.get("criteria") or TONE_CRITERIA


@register("scorer", "policy")
class Policy(Rubric):
    """Business rules, e.g. rules: ["Never give legal or tax advice", "Never promise a sale price"]."""

    use_reference = False

    def criteria(self) -> str:
        rules = self.params.get("rules") or []
        return (POLICY_PREAMBLE + "\n".join(f"- {r}" for r in rules)) if rules else ""

    def setup(self) -> None:
        if not self.params.get("rules"):
            raise ValueError(f"scorer '{self.name}': 'rules' (a list of business rules) is required")
        super().setup()
