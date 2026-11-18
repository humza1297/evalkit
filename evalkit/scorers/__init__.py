from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.types import Case, Response, Score

if TYPE_CHECKING:
    from ..core.config import ProjectConfig
    from ..core.env import Env


class Scorer:
    """score() returns a value in [0, 1], or value=None when not applicable (reason) or failed (error)."""

    kind = ""

    def __init__(self, params: dict, env: "Env", project: "ProjectConfig"):
        self.params = dict(params)
        self.env = env
        self.project = project
        self.name: str = params.get("as") or params["name"]
        self.weight: float = float(params.get("weight", 1.0))

    def setup(self) -> None:
        pass

    def score(self, case: Case, resp: Response) -> Score:
        raise NotImplementedError

    def skip(self, reason: str) -> Score:
        return Score(self.name, None, reason=reason)

    def ok(self, value: float, reason: str = "") -> Score:
        return Score(self.name, max(0.0, min(1.0, float(value))), reason=reason)

    def fail(self, error: str) -> Score:
        return Score(self.name, None, error=error)
