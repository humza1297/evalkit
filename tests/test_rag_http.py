import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evalkit.core.config import load_config
from evalkit.core.env import ConfidentialityError, Env
from evalkit.core.registry import get
from evalkit.core.runner import run
from evalkit.core.types import Case, Chunk, Response
from evalkit.loaders import ingest
from evalkit.rag.chunking import chunk_text
from evalkit.targets.http import extract

from .conftest import write_cases, write_config

DOCS = {
    "commission.md": "Commission Policy\n\nThe standard listing commission is 5 percent of the sale price.\n\n"
                     "Commission is split 50/50 between the listing agent and the buyer agent.",
    "office.txt": "Office Hours\n\nThe office opens at 9am and closes at 6pm on weekdays. Saturday hours are 10am to 2pm.",
}


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    (d / "agents.csv").write_text("name,phone\nDana Ruiz,555-0101\nSam Okafor,555-0102\n", encoding="utf-8")
    (d / "image.png").write_bytes(b"\x89PNG")
    import docx

    doc = docx.Document()
    doc.add_paragraph("Earnest money deposits are held in escrow for 30 days.")
    doc.save(str(d / "escrow.docx"))
    return d


def test_ingest(corpus):
    docs, skipped = ingest(corpus)
    ids = {d.id for d in docs}
    assert {"commission.md", "office.txt", "escrow.docx", "agents.csv#row1", "agents.csv#row2"} <= ids
    assert any("image.png" in s for s in skipped)
    assert "Dana Ruiz" in next(d.text for d in docs if d.id == "agents.csv#row1")


def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 360 for c in chunks)
    long = chunk_text("x" * 1000, size=200, overlap=0)
    assert all(len(c) <= 200 for c in long)


def test_bm25_and_embedding_retrieval(corpus):
    from evalkit.rag.chunking import chunk_documents

    docs, _ = ingest(corpus)
    chunks = chunk_documents(docs, 300, 50)
    env = Env()
    for name, cfg in [("bm25", {}), ("embedding", {"embedder": "mock/hash"}), ("hybrid", {"embedder": "mock/hash"})]:
        r = get("retriever", name)(env, cfg)
        r.setup()
        r.index(chunks)
        hits = r.search("What is the listing commission?", 2)
        assert hits[0].doc_id == "commission.md", name


def test_embedding_retriever_respects_guard():
    r = get("retriever", "embedding")(Env(confidential=True), {"embedder": "together/some-embedder"})
    with pytest.raises(ConfidentialityError):
        r.setup()


def test_rag_target_end_to_end(tmp_path, corpus):
    write_cases(tmp_path / "eval.jsonl", [
        {"id": "q1", "input": "What is the standard listing commission?", "expected": "5 percent",
         "tags": ["fact"], "metadata": {"source_doc": "commission.md", "must_contain": ["5 percent"]}},
        {"id": "q2", "input": "What time does the office open on weekdays?", "expected": "9am",
         "tags": ["fact"], "metadata": {"source_doc": "office.txt", "must_contain": ["9am"]}},
        {"id": "q3", "input": "What is the pet policy for the Denver branch?", "tags": ["unanswerable"]},
    ])
    cfg_path = write_config(tmp_path / "evalkit.yaml", {
        "project": "rag-test", "corpus": "docs", "dataset": "eval.jsonl", "pack": "rag", "judge": "mock/judge",
        "targets": [
            {"name": "bm25-extractive", "type": "rag", "model": "mock/extractive", "retriever": "bm25", "top_k": 2},
            {"name": "lazy", "type": "rag", "model": "mock/lazy"},
        ],
    })
    run_dir = run(load_config(cfg_path), progress=lambda *_: None)
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    best = s["targets"]["bm25-extractive"]
    assert s["ranking"][0] == "bm25-extractive"
    assert best["scorers"]["retrieval_hit"]["mean"] == 1.0
    assert best["scorers"]["contains"]["mean"] == 1.0
    assert best["scorers"]["refusal"]["mean"] == 1.0
    assert best["scorers"]["answer_correctness"]["n"] == 2


# ---- HTTP target ---------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["message"]
        if "fail" in q:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        payload = {"data": {"reply": f"echo: {q}", "sources": [{"source": "office.txt", "text": "hours"}]}}
        out = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}/chat"
    srv.shutdown()


def test_extract_paths():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert extract(data, "$.a.b[1].c") == 2
    assert extract(data, "a.b.0.c") == 1
    with pytest.raises(KeyError):
        extract(data, "a.x")


def test_http_target(server, tmp_path):
    from evalkit.core.config import ProjectConfig

    proj = ProjectConfig(path=tmp_path / "x.yaml", base_dir=tmp_path, project="t")
    t = get("target", "http")({"name": "bot", "url": server, "body": {"message": "{input}", "meta": {"id": "{case_id}"}},
                               "output_path": "$.data.reply", "context_path": "$.data.sources"}, Env(), proj)
    t.setup()
    r = t.run(Case(id="1", input='He said "hi" {x}'))
    assert r.output == 'echo: He said "hi" {x}'
    assert r.retrieved[0].doc_id == "office.txt"
    with pytest.raises(RuntimeError):
        t.run(Case(id="2", input="please fail"))


def test_http_target_guard(tmp_path):
    from evalkit.core.config import ProjectConfig

    proj = ProjectConfig(path=tmp_path / "x.yaml", base_dir=tmp_path, project="t")
    t = get("target", "http")({"name": "bot", "url": "https://bot.example.com/chat"}, Env(confidential=True), proj)
    with pytest.raises(ConfidentialityError):
        t.setup()
    t2 = get("target", "http")({"name": "bot", "url": "http://client-bot:8080/chat"}, Env(local_hosts=["client-bot"]), proj)
    t2.setup()


# ---- deterministic scorers ------------------------------------------------------------

def _score(name, case, output, retrieved=None, **params):
    s = get("scorer", name)({"name": name, **params}, Env(), None)
    return s.score(case, Response(output=output, retrieved=retrieved or []))


def test_deterministic_scorers():
    c = Case(id="1", input="q", expected="5 percent", metadata={"must_contain": [["5 percent", "5%"]]})
    assert _score("contains", c, "It is 5 Percent.").value == 1.0
    assert _score("contains", c, "It is 5% of the price.").value == 1.0
    assert _score("contains", c, "Four percent.").value == 0.0
    assert _score("contains", Case(id="9", input="q", expected="x"), "x").value is None  # no must_contain: n/a
    assert _score("exact_match", c, "5 percent!").value == 1.0
    assert _score("exact_match", Case(id="2", input="q"), "x").value is None
    assert _score("regex", c, "Call 555-0100", pattern=r"\d{3}-\d{4}").value == 1.0
    assert _score("json_valid", c, '```json\n{"a": 1}\n```').value == 1.0
    assert _score("json_valid", c, "{bad").value == 0.0
    un = Case(id="3", input="q", tags=["unanswerable"])
    assert _score("refusal", un, "I don't know based on the documents.").value == 1.0
    assert _score("refusal", un, "The CEO is Bob.").value == 0.0
    assert _score("refusal", c, "That is not mentioned in the documents.").value is None  # answerable: n/a
    assert _score("refusal", un, "The provided context does not mention a CEO.").value == 1.0
    from evalkit.scorers.deterministic import is_refusal

    assert is_refusal("That is not mentioned in the documents.")
    assert not is_refusal("Closing costs are not included in the commission.")
    gold = Case(id="4", input="q", metadata={"source_docs": ["a", "b"]})
    assert _score("retrieval_hit", gold, "x", [Chunk("1", "a", "t")]).value == 0.5
    assert _score("retrieval_hit", gold, "x").value is None
