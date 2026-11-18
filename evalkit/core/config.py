from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..packs import PACKS

TOP_LEVEL_KEYS = {
    "project", "confidential", "local_hosts", "corpus", "dataset", "output_dir", "concurrency",
    "providers", "targets", "scorers", "judge", "weights", "budgets", "pack", "cache", "generator", "limit",
}
DEFAULT_WEIGHTS = {"quality": 0.6, "latency": 0.2, "cost": 0.2}


class ConfigError(ValueError):
    pass


@dataclass
class ProjectConfig:
    path: Path
    base_dir: Path
    project: str
    confidential: bool = True
    local_hosts: list[str] = field(default_factory=list)
    corpus: Path | None = None
    dataset: Path | None = None
    output_dir: Path = Path("runs")
    concurrency: int = 4
    providers: dict[str, dict] = field(default_factory=dict)
    targets: list[dict] = field(default_factory=list)
    scorers: list[dict] = field(default_factory=list)
    judge: str | None = None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    budgets: dict[str, float] = field(default_factory=dict)  # latency_s, cost_per_1k
    pack: str | None = None
    cache: bool = True
    generator: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_path(self) -> Path:
        return self.base_dir / ".evalkit" / "cache.sqlite"

    def resolve_path(self, p: str | Path | None) -> Path | None:
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def _norm_scorer(s: Any) -> dict:
    if isinstance(s, str):
        return {"name": s}
    if isinstance(s, dict) and "name" in s:
        return dict(s)
    if isinstance(s, dict) and len(s) == 1:  # {contains: {weight: 2}}
        (name, params), = s.items()
        return {"name": name, **(params or {})}
    raise ConfigError(f"bad scorer entry: {s!r}")


def _default_target_name(t: dict) -> str:
    return f"{t.get('type', 'model')}:{t.get('model') or t.get('url') or '?'}"


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(TOP_LEVEL_KEYS))})")

    cfg = ProjectConfig(path=path, base_dir=path.parent, project=str(raw.get("project") or path.stem), raw=raw)
    cfg.confidential = bool(raw.get("confidential", True))  # safe default
    cfg.local_hosts = list(raw.get("local_hosts") or [])
    cfg.corpus = cfg.resolve_path(raw.get("corpus"))
    cfg.dataset = cfg.resolve_path(raw.get("dataset"))
    cfg.output_dir = cfg.resolve_path(raw.get("output_dir") or "runs")  # type: ignore[assignment]
    cfg.concurrency = max(1, int(raw.get("concurrency", 4)))
    cfg.providers = dict(raw.get("providers") or {})
    cfg.judge = raw.get("judge")
    cfg.cache = bool(raw.get("cache", True))
    cfg.generator = dict(raw.get("generator") or {})
    cfg.limit = raw.get("limit")
    cfg.weights = {**DEFAULT_WEIGHTS, **(raw.get("weights") or {})}
    for k, v in cfg.weights.items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"weight '{k}' must be a non-negative number")
    cfg.budgets = dict(raw.get("budgets") or {})
    for k, v in cfg.budgets.items():
        if k not in ("latency_s", "cost_per_1k", "min_quality", "vram_gb") or not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"budget '{k}' must be latency_s, cost_per_1k, min_quality or vram_gb with a non-negative number")

    cfg.pack = raw.get("pack")
    pack = {}
    if cfg.pack:
        if cfg.pack not in PACKS:
            raise ConfigError(f"unknown pack '{cfg.pack}'. available: {', '.join(PACKS)}")
        pack = PACKS[cfg.pack]

    scorers = raw.get("scorers")
    cfg.scorers = [_norm_scorer(s) for s in (scorers if scorers is not None else pack.get("scorers", []))]

    targets = []
    for i, t in enumerate(raw.get("targets") or []):
        if not isinstance(t, dict):
            raise ConfigError(f"target #{i + 1} must be a mapping")
        t = {**pack.get("target_defaults", {}), "type": "model", **t}
        t.setdefault("name", _default_target_name(t))
        targets.append(t)
    names = [t["name"] for t in targets]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ConfigError(f"duplicate target names: {', '.join(sorted(dupes))}. Give each target a unique 'name'.")
    cfg.targets = targets
    return cfg
