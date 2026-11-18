from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np

from ..core.registry import get, register
from ..core.types import Chunk
from ..core.util import content_words


class Retriever:
    def __init__(self, env, cfg: dict):
        self.env = env
        self.cfg = cfg
        self.chunks: list[Chunk] = []

    def setup(self) -> None:
        """Resolve any providers (confidentiality guard)."""

    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[Chunk]:
        raise NotImplementedError


@register("retriever", "bm25")
class BM25(Retriever):
    k1, b = 1.5, 0.75

    def index(self, chunks):
        self.chunks = chunks
        self.docs = [Counter(content_words(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = content_words(query)
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out

    def search(self, query, k):
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [replace(self.chunks[i], score=round(sc[i], 4)) for i in order if sc[i] > 0]


@register("retriever", "embedding")
class EmbeddingRetriever(Retriever):
    def setup(self):
        self.spec = self.cfg.get("embedder")
        if not self.spec:
            raise ValueError("embedding retriever needs 'embedder' (e.g. ollama/nomic-embed-text)")
        self.env.resolve(self.spec, "embedder")

    def _embed(self, texts, prefix: str = ""):
        m = np.asarray(self.env.embed(self.spec, [prefix + t for t in texts]), dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def index(self, chunks):
        self.chunks = chunks
        self.matrix = (self._embed([c.text for c in chunks], self.cfg.get("doc_prefix", ""))
                       if chunks else np.zeros((0, 1), dtype=np.float32))

    def search(self, query, k):
        if not self.chunks:
            return []
        sims = self.matrix @ self._embed([query], self.cfg.get("query_prefix", ""))[0]
        order = np.argsort(-sims)[:k]
        return [replace(self.chunks[i], score=round(float(sims[i]), 4)) for i in order]


@register("retriever", "hybrid")
class Hybrid(Retriever):
    """Reciprocal-rank fusion of BM25 and embeddings."""

    def setup(self):
        self.parts = [get("retriever", "bm25")(self.env, self.cfg), get("retriever", "embedding")(self.env, self.cfg)]
        for p in self.parts:
            p.setup()

    def index(self, chunks):
        self.chunks = chunks
        for p in self.parts:
            p.index(chunks)

    def search(self, query, k, rrf_k: int = 60):
        fused: dict[str, float] = {}
        by_id = {c.id: c for c in self.chunks}
        for p in self.parts:
            for rank, c in enumerate(p.search(query, k * 3)):
                fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [replace(by_id[i], score=round(fused[i], 5)) for i in top]
