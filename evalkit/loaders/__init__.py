"""File loaders -> Documents. Register new types with @register("loader", ".ext") (images/audio later)."""
from __future__ import annotations

import csv
from pathlib import Path

from ..core.registry import _REGISTRY, get, register
from ..core.types import Document


@register("loader", ".txt")
@register("loader", ".md")
def load_text(path: Path, doc_id: str) -> list[Document]:
    return [Document(doc_id, path.read_text(encoding="utf-8", errors="replace"), {"path": str(path), "type": "text"})]


@register("loader", ".csv")
def load_csv(path: Path, doc_id: str) -> list[Document]:
    """One document per row ('col: value' lines) so each record is retrievable on its own."""
    docs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if k and v not in (None, ""))
            if text:
                docs.append(Document(f"{doc_id}#row{i}", text, {"path": str(path), "type": "csv_row", "row": i}))
    return docs


@register("loader", ".pdf")
def load_pdf(path: Path, doc_id: str) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF support needs pypdf: pip install 'evalkit[pdf]'") from e
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "").strip() for p in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    return [Document(doc_id, text, {"path": str(path), "type": "pdf", "pages": len(pages)})] if text else []


@register("loader", ".docx")
def load_docx(path: Path, doc_id: str) -> list[Document]:
    try:
        import docx
    except ImportError as e:
        raise ImportError("DOCX support needs python-docx: pip install 'evalkit[docx]'") from e
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = "\n\n".join(parts)
    return [Document(doc_id, text, {"path": str(path), "type": "docx"})] if text else []


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY.get("loader", {}))


def ingest(root: Path) -> tuple[list[Document], list[str]]:
    """Load every supported file under root. Returns (documents, skipped files with reasons)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus not found: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    docs: list[Document] = []
    skipped: list[str] = []
    exts = set(supported_extensions())
    for p in files:
        rel = p.name if root.is_file() else p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        ext = p.suffix.lower()
        if ext not in exts:
            skipped.append(f"{rel} (unsupported type)")
            continue
        try:
            loaded = get("loader", ext)(p, rel)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{rel} ({type(e).__name__}: {e})")
            continue
        if not loaded:
            skipped.append(f"{rel} (no text extracted; scanned PDF? OCR comes in a later phase)")
        docs.extend(loaded)
    return docs, skipped
