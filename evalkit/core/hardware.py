"""Hardware fit: how much GPU memory a target's model needs.

Measured from Ollama (/api/ps while loaded) when possible, else estimated from /api/show details
or from the model name (e.g. '8b' -> 8e9 params) at the given quantization."""
from __future__ import annotations

import json
import re
import urllib.request

BYTES_PER_PARAM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8": 1.07, "q6": 0.82, "q5": 0.69, "q4": 0.57, "q3": 0.45, "q2": 0.33}
OVERHEAD = 1.2  # KV cache + runtime buffers at modest context


def _bytes_per_param(quant: str | None) -> float:
    q = (quant or "q4").lower()
    for k, v in BYTES_PER_PARAM.items():
        if q.startswith(k):
            return v
    return BYTES_PER_PARAM["q4"]


def params_from_name(model: str) -> float | None:
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bm])\b", model.lower())
    if not m:
        return None
    n = float(m.group(1))
    return n * (1e9 if m.group(2) == "b" else 1e6)


def estimate_vram_gb(params: float | None, quant: str | None = None) -> float | None:
    if not params:
        return None
    return round(params * _bytes_per_param(quant) * OVERHEAD / 1e9, 1)


def _ollama_root(base_url: str) -> str:
    return re.sub(r"/v1/?$", "", base_url.rstrip("/"))


def _get(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict | None:
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8") if payload else None,
                                     headers={"Content-Type": "application/json"}, method="POST" if payload else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def model_hardware(provider, model: str) -> dict:
    """Best available memory figure for one model spec on one provider."""
    info: dict = {"model": f"{provider.name}/{model}"}
    if provider.name == "mock":
        return {**info, "vram_gb": None, "source": "mock"}
    if provider.name == "ollama" or provider.cfg.get("type") == "ollama":
        root = _ollama_root(provider.base_url)
        ps = _get(root + "/api/ps") or {}
        for m in ps.get("models", []):
            if m.get("name") in (model, f"{model}:latest") or m.get("model") in (model, f"{model}:latest"):
                details = m.get("details") or {}
                return {**info, "params": details.get("parameter_size"), "quant": details.get("quantization_level"),
                        "vram_gb": round(m.get("size_vram", m.get("size", 0)) / 1e9, 1), "source": "measured (ollama ps)"}
        show = _get(root + "/api/show", {"model": model}) or {}
        details = show.get("details") or {}
        if details:
            params = params_from_name(str(details.get("parameter_size", "")))
            return {**info, "params": details.get("parameter_size"), "quant": details.get("quantization_level"),
                    "vram_gb": estimate_vram_gb(params, details.get("quantization_level")), "source": "estimated (ollama show)"}
    params = params_from_name(model)
    return {**info, "params": f"{params / 1e9:g}B" if params else None, "quant": "assumed q4",
            "vram_gb": estimate_vram_gb(params), "source": "estimated (from name)" if params else "unknown"}
