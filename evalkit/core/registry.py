"""Tiny plugin registry. Kinds: target, scorer, loader, retriever, provider."""
from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = {}


def register(kind: str, name: str) -> Callable[[Any], Any]:
    def deco(obj: Any) -> Any:
        _REGISTRY.setdefault(kind, {})[name] = obj
        return obj

    return deco


def get(kind: str, name: str) -> Any:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"unknown {kind} '{name}'. available: {', '.join(available(kind)) or '(none)'}") from None


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


BUILTIN_MODULES = [
    "evalkit.providers.mock",
    "evalkit.providers.openai_compat",
    "evalkit.loaders",
    "evalkit.rag.retrieval",
    "evalkit.targets.model",
    "evalkit.targets.rag",
    "evalkit.targets.http",
    "evalkit.scorers.deterministic",
    "evalkit.scorers.judge",
]


def load_builtins() -> None:
    """Import built-in plugin modules so they register themselves."""
    import importlib

    for mod in BUILTIN_MODULES:
        importlib.import_module(mod)
