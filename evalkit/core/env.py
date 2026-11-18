"""Runtime environment: provider resolution (the single confidentiality choke point) + cached model calls."""
from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from .cache import Cache
from .registry import get
from .util import stable_hash


class ConfidentialityError(RuntimeError):
    """Raised when a confidential project would send data to a non-local host."""


def is_local_host(host: str | None, extra: tuple[str, ...] | list[str] = ()) -> bool:
    if not host:
        return False
    h = host.strip("[]").lower()
    if h in {e.lower() for e in extra} or h == "localhost" or h.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


@dataclass
class ChatOutcome:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float
    cached: bool


class Env:
    def __init__(self, *, confidential: bool = True, providers_cfg: dict | None = None,
                 local_hosts: list[str] | tuple[str, ...] = (), cache: Cache | None = None):
        self.confidential = confidential
        self.providers_cfg = providers_cfg or {}
        self.local_hosts = tuple(local_hosts)
        self.cache = cache or Cache(None)
        self._providers: dict = {}
        self._lock = threading.Lock()

    # ---- provider resolution / guard -------------------------------------------------
    def provider(self, name: str):
        from ..providers import PRESETS

        with self._lock:
            if name in self._providers:
                return self._providers[name]
            cfg = {**PRESETS.get(name, {}), **(self.providers_cfg.get(name) or {})}
            if name == "mock" or cfg.get("type") == "mock":
                p = get("provider", "mock")(name, cfg)
            else:
                if "base_url" not in cfg:
                    raise ValueError(f"unknown provider '{name}': add it under 'providers:' with a base_url")
                local = is_local_host(urlparse(cfg["base_url"]).hostname, self.local_hosts)
                p = get("provider", "openai_compat")(name, cfg, local=local)
            self._providers[name] = p
            return p

    def resolve(self, spec: str, purpose: str = "model"):
        name, _, model = spec.partition("/")
        if not name or not model:
            raise ValueError(f"{purpose}: model spec '{spec}' must look like 'provider/model' (e.g. ollama/llama3.2:3b)")
        p = self.provider(name)
        if self.confidential and not p.local:
            raise ConfidentialityError(
                f"{purpose} '{spec}' would send confidential data to remote host {p.base_url}. "
                "Use a local provider, add the host to 'local_hosts' if it is on the client's network, "
                "or set 'confidential: false'."
            )
        return p, model

    def check_url(self, url: str, purpose: str) -> None:
        host = urlparse(url).hostname
        if self.confidential and not is_local_host(host, self.local_hosts):
            raise ConfidentialityError(
                f"{purpose} URL {url} is not a local/private host and the project is confidential. "
                "Add the host to 'local_hosts' if it runs inside the client's network."
            )

    # ---- cached calls ---------------------------------------------------------------
    def chat(self, spec: str, messages: list[dict], purpose: str = "model",
             temperature: float = 0.0, max_tokens: int = 512, use_cache: bool = True) -> ChatOutcome:
        provider, model = self.resolve(spec, purpose)
        key = "chat:" + stable_hash([spec, messages, temperature, max_tokens], 32)
        hit = self.cache.get(key) if use_cache else None
        if hit is not None:
            text, pt, ct, latency = hit["text"], hit["pt"], hit["ct"], hit["latency"]
            cached = True
        else:
            t0 = time.perf_counter()
            r = provider.chat(model, messages, temperature=temperature, max_tokens=max_tokens)
            latency = r.latency_s if r.latency_s is not None else time.perf_counter() - t0
            text, pt, ct = r.text, r.prompt_tokens, r.completion_tokens
            if use_cache:
                self.cache.set(key, {"text": text, "pt": pt, "ct": ct, "latency": latency})
            cached = False
        return ChatOutcome(text, pt, ct, latency, provider.cost(model, pt, ct), cached)

    def embed(self, spec: str, texts: list[str], purpose: str = "embedder", batch: int = 32) -> list[list[float]]:
        provider, model = self.resolve(spec, purpose)
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        keys = ["emb:" + stable_hash([spec, t], 32) for t in texts]
        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is None:
                missing.append(i)
            else:
                out[i] = v
        for s in range(0, len(missing), batch):
            idx = missing[s:s + batch]
            vecs = provider.embed(model, [texts[i] for i in idx])
            for i, v in zip(idx, vecs):
                out[i] = v
                self.cache.set(keys[i], v)
        return out  # type: ignore[return-value]
