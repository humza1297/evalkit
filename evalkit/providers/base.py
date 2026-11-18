from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float | None = None  # provider may report simulated/real latency; else Env measures it


class Provider:
    """A model host. `local` is derived from where it runs, never from a bare flag."""

    name: str = ""
    base_url: str = ""
    local: bool = False

    def __init__(self, name: str, cfg: dict | None = None):
        self.name = name
        self.cfg = dict(cfg or {})
        # pricing: {model: [usd_per_1M_input, usd_per_1M_output]}, "*" = default
        self.pricing: dict[str, list[float]] = dict(self.cfg.get("pricing") or {})

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512) -> ChatResult:
        raise NotImplementedError

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def ping(self) -> str:
        return "ok"

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price = self.pricing.get(model) or self.pricing.get("*")
        if not price:
            return 0.0
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
