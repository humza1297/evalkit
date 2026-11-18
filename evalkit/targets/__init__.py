from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import Case, Response
from ..core.util import stable_hash, strip_reasoning

if TYPE_CHECKING:
    from ..core.config import ProjectConfig
    from ..core.env import Env


class EmptyOutputError(RuntimeError):
    pass


def clean_output(text: str) -> str:
    """Strip reasoning blocks. An answer that is empty after stripping is a target error, not a wrong answer."""
    out = strip_reasoning(text)
    if not out:
        raise EmptyOutputError("empty answer after removing <think> reasoning (output truncated? raise max_tokens)")
    return out


class Target:
    """A system under test: a bare model, a RAG pipeline, or a client endpoint."""

    type = ""

    def __init__(self, cfg: dict, env: "Env", project: "ProjectConfig"):
        self.cfg = dict(cfg)
        self.env = env
        self.project = project
        self.name: str = cfg["name"]

    def fingerprint(self) -> str:
        return stable_hash({k: v for k, v in self.cfg.items() if k != "name"})

    def models(self) -> list[str]:
        """Model specs this target uses (for reports and hardware fit)."""
        return [self.cfg["model"]] if "model" in self.cfg else []

    def setup(self) -> None:
        """Resolve providers (runs the confidentiality guard early) and build indexes."""

    def warmup(self) -> None:
        """One untimed call so model load time isn't measured as latency."""
        if "model" in self.cfg and not str(self.cfg["model"]).startswith("mock/"):
            try:
                self.env.chat(self.cfg["model"], [{"role": "user", "content": "Reply with OK."}], f"warmup '{self.name}'",
                              max_tokens=8, use_cache=False)
            except Exception:  # noqa: BLE001 - real errors surface on the first case
                pass

    def run(self, case: Case) -> Response:
        raise NotImplementedError
