from __future__ import annotations

import threading
import time

from ..core.registry import get, register
from ..core.types import Case, Response
from ..core.util import fill, stable_hash
from ..loaders import ingest
from ..rag.chunking import chunk_documents
from . import Target, clean_output
from .model import build_messages

DEFAULT_RAG_PROMPT = "Context:\n{context}\n\nQuestion: {input}"

_CORPUS_CACHE: dict[str, tuple] = {}
_CORPUS_LOCK = threading.Lock()


def load_corpus(path):
    key = str(path)
    with _CORPUS_LOCK:
        if key not in _CORPUS_CACHE:
            docs, skipped = ingest(path)
            _CORPUS_CACHE[key] = (docs, skipped, stable_hash([[d.id, d.text] for d in docs]))
        return _CORPUS_CACHE[key]


@register("target", "rag")
class RagTarget(Target):
    """Reference RAG pipeline: chunk -> retrieve -> generate. Every stage is configurable, so chunking,
    retrievers and embedders are compared alongside the LLM.

    cfg: model, corpus, chunk_size=800, chunk_overlap=100, retriever=bm25|embedding|hybrid, embedder, top_k=4,
         prompt ({input}, {context}), system, temperature, max_tokens
    """

    type = "rag"
    corpus_hash = ""

    def models(self):
        return [m for m in (self.cfg.get("model"), self.cfg.get("embedder")) if m]

    def fingerprint(self) -> str:
        return stable_hash([super().fingerprint(), self.corpus_hash])

    def setup(self) -> None:
        if "model" not in self.cfg:
            raise ValueError(f"target '{self.name}': 'model' is required")
        self.env.resolve(self.cfg["model"], f"target '{self.name}'")
        corpus = self.project.resolve_path(self.cfg.get("corpus")) or self.project.corpus
        if not corpus:
            raise ValueError(f"target '{self.name}': no corpus (set 'corpus' on the target or project)")
        docs, _skipped, self.corpus_hash = load_corpus(corpus)
        if not docs:
            raise ValueError(f"target '{self.name}': no documents loaded from {corpus}")
        chunks = chunk_documents(docs, int(self.cfg.get("chunk_size", 800)), int(self.cfg.get("chunk_overlap", 100)))
        self.retriever = get("retriever", self.cfg.get("retriever", "bm25"))(self.env, self.cfg)
        self.retriever.setup()
        self.retriever.index(chunks)
        self.n_chunks = len(chunks)

    def run(self, case: Case) -> Response:
        t0 = time.perf_counter()
        query = case.input
        if case.history and self.cfg.get("retrieval_query", "history") == "history":
            # naive follow-up handling: earlier user turns give the retriever the topic ("what about luxury homes?")
            query = " ".join([t["content"] for t in case.history if t["role"] == "user"][-2:] + [case.input])
        hits = self.retriever.search(query, int(self.cfg.get("top_k", 4)))
        retrieval_s = time.perf_counter() - t0
        context = "\n\n".join(f"[{i}] (source: {c.doc_id})\n{c.text}" for i, c in enumerate(hits, 1)) or "(no relevant documents found)"
        user = fill(self.cfg.get("prompt") or DEFAULT_RAG_PROMPT, input=case.input, context=context)
        r = self.env.chat(
            self.cfg["model"], build_messages(self.cfg.get("system"), user, case.history), f"target '{self.name}'",
            temperature=float(self.cfg.get("temperature", 0.0)), max_tokens=int(self.cfg.get("max_tokens", 512)),
        )
        return Response(output=clean_output(r.text), retrieved=hits, latency_s=retrieval_s + r.latency_s, prompt_tokens=r.prompt_tokens,
                        completion_tokens=r.completion_tokens, cost_usd=r.cost_usd, cached=r.cached)
