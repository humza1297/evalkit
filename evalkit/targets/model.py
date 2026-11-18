from __future__ import annotations

from ..core.registry import register
from ..core.types import Case, Response
from ..core.util import fill
from . import Target, clean_output

DEFAULT_PROMPT = "{input}"
DEFAULT_CONTEXT_PROMPT = "Context:\n{context}\n\nQuestion: {input}"


def build_messages(system: str | None, user: str, history: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs += [{"role": t["role"], "content": t["content"]} for t in history or []]
    return msgs + [{"role": "user", "content": user}]


@register("target", "model")
class ModelTarget(Target):
    """Bare model + prompt template. Placeholders: {input}, {context} (from the case)."""

    type = "model"

    def setup(self) -> None:
        if "model" not in self.cfg:
            raise ValueError(f"target '{self.name}': 'model' is required")
        self.env.resolve(self.cfg["model"], f"target '{self.name}'")

    def run(self, case: Case) -> Response:
        template = self.cfg.get("prompt") or (DEFAULT_CONTEXT_PROMPT if case.context else DEFAULT_PROMPT)
        user = fill(template, input=case.input, context=case.context or "")
        r = self.env.chat(
            self.cfg["model"], build_messages(self.cfg.get("system"), user, case.history), f"target '{self.name}'",
            temperature=float(self.cfg.get("temperature", 0.0)), max_tokens=int(self.cfg.get("max_tokens", 512)),
        )
        return Response(output=clean_output(r.text), latency_s=r.latency_s, prompt_tokens=r.prompt_tokens,
                        completion_tokens=r.completion_tokens, cost_usd=r.cost_usd, cached=r.cached)
