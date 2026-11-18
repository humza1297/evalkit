"""OpenAI-compatible chat + embeddings client (Ollama, vLLM, LM Studio, OpenRouter, Together, Groq...)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..core.registry import register
from .base import ChatResult, Provider


class ProviderError(RuntimeError):
    pass


@register("provider", "openai_compat")
class OpenAICompatProvider(Provider):
    def __init__(self, name: str, cfg: dict | None = None, local: bool = False):
        super().__init__(name, cfg)
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.local = local
        self.timeout = float(self.cfg.get("timeout", 120))
        self.retries = int(self.cfg.get("retries", 2))
        key_env = self.cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else self.cfg.get("api_key", "")

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                last = ProviderError(f"{self.name} HTTP {e.code}: {body}")
                if e.code < 500 and e.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = ProviderError(f"{self.name} unreachable at {self.base_url}: {e}")
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise last  # type: ignore[misc]

    def chat(self, model, messages, temperature=0.0, max_tokens=512):
        data = self._post("/chat/completions", {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"{self.name}: unexpected chat response: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        return ChatResult(text=text, prompt_tokens=int(usage.get("prompt_tokens", 0)), completion_tokens=int(usage.get("completion_tokens", 0)))

    def embed(self, model, texts):
        data = self._post("/embeddings", {"model": model, "input": texts})
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            raise ProviderError(f"{self.name}: expected {len(texts)} embeddings, got {len(items)}")
        return [d["embedding"] for d in items]

    def list_models(self) -> list[str] | None:
        req = urllib.request.Request(self.base_url + "/models", headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return [str(m.get("id")) for m in json.loads(resp.read().decode("utf-8")).get("data", [])]
        except Exception:  # noqa: BLE001
            return None

    def ping(self) -> str:
        req = urllib.request.Request(self.base_url + "/models", headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id") for m in data.get("data", [])]
            return f"ok ({len(ids)} models: {', '.join(map(str, ids[:8]))}{'...' if len(ids) > 8 else ''})"
        except Exception as e:  # noqa: BLE001
            return f"unreachable: {e}"
