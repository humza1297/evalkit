from __future__ import annotations

import re

from ..core.types import Chunk, Document


def _split_long(text: str, size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        while len(s) > size:  # hard split very long sentences at a space
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunks of ~size chars; each chunk starts with ~overlap chars from the previous one."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))
    paras: list[str] = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if p:
            paras.extend(_split_long(p, size) if len(p) > size else [p])
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.find(" ") + 1:]
            cur = f"{tail}\n\n{p}".strip() if tail else p
        else:
            cur = f"{cur}\n\n{p}".strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 100) -> list[Chunk]:
    out = []
    for d in docs:
        for i, text in enumerate(chunk_text(d.text, size, overlap)):
            out.append(Chunk(id=f"{d.id}::{i}", doc_id=d.id, text=text, metadata={"chunk": i}))
    return out
