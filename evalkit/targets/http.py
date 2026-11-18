from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ..core.registry import register
from ..core.types import Case, Chunk, Response
from . import Target, clean_output


def extract(obj: Any, path: str | None) -> Any:
    """Tiny JSONPath subset: '$.a.b[0].c' or 'a.b.0.c'. Empty/None path returns obj."""
    if not path:
        return obj
    parts = [p for p in re.split(r"\.|\[|\]", path.lstrip("$")) if p]
    cur = obj
    for p in parts:
        if isinstance(cur, list):
            cur = cur[int(p)]
        elif isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"path '{path}': key '{p}' not found (have: {', '.join(map(str, cur))})")
            cur = cur[p]
        else:
            raise KeyError(f"path '{path}': cannot index {type(cur).__name__} with '{p}'")
    return cur


def _render(template: Any, values: dict[str, Any]) -> Any:
    """Fill {input}/{case_id} etc. inside every string of a JSON-like template (JSON-safe).
    A string that is exactly "{history}" becomes the list of prior turns (not a string)."""
    if isinstance(template, str):
        m = re.fullmatch(r"\{(\w+)\}", template)
        if m and m.group(1) in values and not isinstance(values[m.group(1)], str):
            return values[m.group(1)]
        out = template
        for k, v in values.items():
            if isinstance(v, str):
                out = out.replace("{" + k + "}", v)
        return out
    if isinstance(template, list):
        return [_render(t, values) for t in template]
    if isinstance(template, dict):
        return {k: _render(v, values) for k, v in template.items()}
    return template


def _expand_env(s: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), s)


@register("target", "http")
class HttpTarget(Target):
    """Wraps an existing pipeline/chatbot endpoint.

    cfg: url, method=POST, headers={}, body={"input": "{input}"}, output_path="output",
         context_path (optional; list of strings or {text, source} objects), source_key="source", timeout=60
    """

    type = "http"

    def models(self):
        return [f"http:{self.cfg.get('url')}"]

    def setup(self) -> None:
        if "url" not in self.cfg:
            raise ValueError(f"target '{self.name}': 'url' is required")
        self.env.check_url(self.cfg["url"], f"target '{self.name}'")

    def run(self, case: Case) -> Response:
        values = {"input": case.input, "case_id": case.id, "context": case.context or "", "history": case.history}
        body = _render(self.cfg.get("body", {"input": "{input}"}), values)
        headers = {"Content-Type": "application/json", **{k: _expand_env(str(v)) for k, v in (self.cfg.get("headers") or {}).items()}}
        req = urllib.request.Request(self.cfg["url"], data=json.dumps(body).encode("utf-8"), headers=headers,
                                     method=self.cfg.get("method", "POST"))
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout", 60))) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
        latency = time.perf_counter() - t0
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise RuntimeError(f"endpoint did not return JSON: {raw[:200]}") from e
        output = extract(data, self.cfg.get("output_path", "output"))
        retrieved: list[Chunk] = []
        if self.cfg.get("context_path"):
            items = extract(data, self.cfg["context_path"]) or []
            sk = self.cfg.get("source_key", "source")
            for i, it in enumerate(items):
                if isinstance(it, dict):
                    retrieved.append(Chunk(id=f"r{i}", doc_id=str(it.get(sk, f"r{i}")), text=str(it.get("text", ""))))
                else:
                    retrieved.append(Chunk(id=f"r{i}", doc_id=f"r{i}", text=str(it)))
        return Response(output=clean_output(output if isinstance(output, str) else json.dumps(output)),
                        retrieved=retrieved, latency_s=latency)

    def warmup(self) -> None:
        pass  # a live client endpoint: don't send extra traffic
