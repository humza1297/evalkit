# Build Checklist

This file tracks progress so work survives session limits. Tick `[x]` only when an item is **on disk and its tests pass**.
**Resuming in a new session:** read `PLAN.md` and this file, then continue from the first unchecked item.
Conventions: package `evalkit/`, Python 3.11+ (dev box has 3.14), venv at `.venv/`, tests via `.venv\Scripts\python -m pytest -q`. Open every file with `encoding="utf-8"`.

**Last verified green at:** Phase 3 + final review fixes (58 tests passing). The final fixes: calibration joins on the response key so labels survive rescoring; resume retries failed responses and keeps hardware info; `refusal` scores only unanswerable cases, and the correctness rubric ranks "don't know" (2) above a wrong answer (1); a scorer-version hash flags scoring-code changes in diffs. Next: real Ollama run (waits for user approval of about 13 GB of model pulls). Phases 4–5 are on hold (hosted keys / multimodal deferred by user)

## Phase 0: Scaffolding
- [x] `pyproject.toml` (core deps: pyyaml, numpy; extras `pdf`=pypdf, `docx`=python-docx; dev: pytest), `.gitignore` (.venv, runs, clients, cache)
- [x] `README.md` (usage guide)
- [x] `.venv` created, deps installed, `pip install -e .` succeeds
- [x] `CLAUDE.md` pointing future sessions at this checklist

## Phase 1a: Thin end-to-end path (mock only)
- [x] `core/types.py` (Case, Response, Score, Document, Chunk, CaseResult) + `core/registry.py` + `core/util.py`
- [x] `providers/base.py` + `providers/mock.py` + `core/env.py` (model-spec parsing, **confidentiality guard at provider resolution**, local derived from host, cached chat/embed that keep the original latency)
- [x] `targets/model.py` (bare model + prompt template)
- [x] `scorers/deterministic.py`: exact_match, contains, regex, json_valid, retrieval_hit, refusal (n/a or error gives `None`, never 0)
- [x] `core/config.py` (YAML, relative paths, validation, packs, budgets) + `core/cache.py` (sqlite; LLM-call cache keyed on full prompt + params)
- [x] `core/runner.py`: concurrency, incremental `results.jsonl`, resume keyed on target fingerprint + case hash + scorer config
- [x] `report/__init__.py` (aggregate) + `report/html.py` (mock runs labelled as mock). Ranking uses absolute latency/cost budgets, not min-max
- [x] `cli.py` + `__main__.py` (`run`, `report`)
- [x] e2e test: mock config gives results.jsonl + report.html

## Phase 1b: Widen
- [x] `loaders/`: txt/md, csv (row per doc), pdf (lazy), docx (lazy), plus `ingest(dir)` (PDF loader has no test yet)
- [x] `rag/chunking.py` + `rag/retrieval.py` (BM25, embedding, hybrid RRF)
- [x] `providers/openai_compat.py`: chat + embeddings + retries (Ollama, vLLM, LM Studio, hosted presets), pricing. Not yet tested against a live server
- [x] `targets/rag.py` (fingerprint includes corpus hash)
- [x] `targets/http.py` (request template + JSON path; guard checks the URL host; `local_hosts` allowlist)
- [x] judge scorers: answer_correctness, faithfulness, rubric (robust parsing incl. `<think>` blocks; malformed reply gives error, not 0)
- [x] `datasets/store.py` (JSONL, freeze to read-only `<name>.vN.jsonl`)
- [x] `datasets/generator.py` (fact, numeric, procedural, multi_doc, unanswerable; guard before reading docs; no caching so reruns vary)
- [x] `datasets/review.py` (injectable input fn; saves after each decision; tested with scripted answers)
- [x] Report: per-tag breakdown, p50/p95 latency, cost/1k, error rate, weighted ranking, worst cases, ± stderr
- [x] CLI: `ingest`, `build-dataset`, `review`, `freeze`, `check` (CLI e2e test covers build, review, check, run, report)
- [x] `examples/realestate/` fictional docs + `evalkit.mock.yaml` + `evalkit.ollama.yaml` (llama3.2:3b, qwen3:4b, gemma3:4b, judge llama3.1:8b, nomic-embed-text) + 18-case seed dataset
- [x] Offline demo `python -m evalkit run examples/realestate/evalkit.mock.yaml` works; report checked in browser
- [x] `check` lists missing models; the OpenAI-compatible client is verified against the live Ollama server (errors pass through)
### Pre-real-run fixes (from review)
- [x] Two-pass runner: generate grouped by target (untimed warm-up), then score. `responses.jsonl` is separate, so rescoring never regenerates
- [x] Composite: no latency/cost credit without a measurement; `min_quality` gate (default 0.5)
- [x] `contains` only applies with `must_contain` (alternatives allowed); seed dataset updated
- [x] `<think>` stripped from target outputs; empty after stripping = target error
- [x] Generator checks unanswerable questions against the whole corpus (BM25 `check_hits`, shown in review)
- [x] Embedding `query_prefix`/`doc_prefix` (nomic-embed-text needs `search_query: `/`search_document: `)
- [x] Refusal regex narrowed; report shows max prompt tokens (context truncation hint)
- [x] PLAN.md config example matches the real schema
- [ ] Real run against Ollama (needs the user's approval to `ollama pull`, about 13 GB total; a smoke test needs only llama3.2:3b + nomic-embed-text, about 2.3 GB). On the first real run:
  - hand-check the refusal column on about 20 outputs
  - add a twin target without `query_prefix`/`doc_prefix` and compare `retrieval_hit` (sources disagree on whether Ollama adds nomic prefixes)
  - confirm the hardware column shows "measured" and that qwen3 `<think>` output is stripped
  - check whether max prompt tokens hits a context ceiling

## Phase 2: Pipelines & selection
- [x] Hardware-fit column (`core/hardware.py`: measured via Ollama `/api/ps` right after each target's pass, else `/api/show` or name-based estimate; `budgets.vram_gb` flags "too big"). The live Ollama path is untested until models are pulled
- [x] Run diff vs previous run (`report/diff.py`; auto baseline = previous run of the same project; flags dataset/scorer changes; per-case regressions). `report --baseline/--no-diff`
- [x] `evalkit calibrate export|score`: CSV sample (judge scores hidden from labelers), within-1 / pass agreement / correlation / bias, verdict, shown in report
- [x] Retries/backoff (provider `retries`, 429/5xx) + timeouts (`providers.<name>.timeout`, http target `timeout`)
- [x] Fixed key collisions: duplicate-content cases and identical-config targets (regression test added)

## Phase 3: Chatbot pack
- [x] Multi-turn case format (`history` on Case); model/rag targets send history; RAG retrieval uses recent user turns (`retrieval_query: history|last`); http body supports `"{history}"`; `chatbot` pack
- [x] Persona simulation (dataset level): `followup` generation type with conversation history, `generator.personas` cycled across cases, shown in review. A live persona-vs-bot simulator is not built
- [x] Tone/policy scorers (`tone`, `policy` with `rules`, `must_not_contain` with `re:` regex); judges see the conversation
- [x] `examples/realestate/evalkit.chatbot.mock.yaml` + `chat.jsonl` run end to end

## Phase 4: Breadth
- [ ] DeepEval/Ragas optional scorer plugins
- [ ] Hosted provider presets verified
- [ ] Optional review UI

## Phase 5: Multimodal
- [ ] Image loaders (OCR / vision)
- [ ] Audio loaders (transcription)
