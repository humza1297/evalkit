from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .util import stable_hash


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class Case:
    id: str
    input: str
    expected: str | None = None
    context: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str | None = None  # None/"accepted" = usable; "pending"/"rejected" = excluded
    history: list[dict[str, str]] = field(default_factory=list)  # prior turns [{role, content}]; `input` is the next user turn

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        if "id" not in data or "input" not in data:
            raise ValueError(f"case is missing 'id' or 'input': {d!r:.200}")
        data["id"] = str(data["id"])
        case = cls(**data)
        case.metadata = {**extra, **(case.metadata or {})}
        case.tags = list(case.tags or [])
        case.history = list(case.history or [])
        for turn in case.history:
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
                raise ValueError(f"case '{case.id}': history turns must be {{role: user|assistant, content: str}}")
        return case

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {})} | {"id": self.id, "input": self.input}

    def content_hash(self) -> str:
        return stable_hash([self.input, self.expected, self.context, sorted(self.tags), self.metadata, self.history])

    def transcript(self) -> str:
        """Prior turns as text (for judges and retrieval)."""
        return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in self.history)

    @property
    def unanswerable(self) -> bool:
        return "unanswerable" in self.tags


@dataclass
class Response:
    output: str = ""
    retrieved: list[Chunk] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Response":
        d = dict(d)
        d["retrieved"] = [Chunk.from_dict(c) for c in d.get("retrieved", [])]
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class Score:
    name: str
    value: float | None  # 0..1; None = not applicable or failed (see error)
    reason: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CaseResult:
    key: str
    target: str
    case: Case
    response: Response
    scores: list[Score]
    quality: float | None
    rkey: str = ""  # response key (target + case, independent of scorers): stable across rescoring

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "rkey": self.rkey,
            "target": self.target,
            "case": self.case.to_dict(),
            "response": self.response.to_dict(),
            "scores": [s.to_dict() for s in self.scores],
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CaseResult":
        return cls(
            key=d["key"],
            target=d["target"],
            case=Case.from_dict(d["case"]),
            response=Response.from_dict(d["response"]),
            scores=[Score(**s) for s in d["scores"]],
            quality=d.get("quality"),
            rkey=d.get("rkey", ""),
        )
