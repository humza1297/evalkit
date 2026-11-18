# Modular Eval System: Plan

A system that quickly builds an evaluation suite from a client's own data, then scores either raw open-source models or the client's whole AI pipeline against it. The output is a ranked recommendation: the best model or config for that business niche.

## Assumptions (correct me)

- "Rack/drag" means **RAG** (retrieval-augmented generation).
- **What gets tested:** both raw models and full client pipelines (RAG endpoint, chatbot), behind one interface.
- **Operator:** you run it as a service for clients. v1 is a CLI with config files and an HTML report, and there's no web UI yet.
- **Inference:** pluggable, local (Ollama/vLLM) or hosted. **Local is the default**, because legal and real-estate data is confidential.
- **Data:** text and documents in v1 (PDF, DOCX, TXT, MD, CSV). Images and audio come later through the same plugin interface.
- **Language:** Python 3.11+, because that's where the eval and ML tooling lives.

## Core idea: a small core plus plugin packs

"Workflow agnostic" means the core knows nothing about RAG or chatbots. Each workflow is a **pack** (adapter + default scorers + dataset template) built on five contracts:

| Contract | Responsibility | Examples |
|---|---|---|
| **Dataset** | Test cases: `input`, optional `context`, `expected`, `metadata`, `tags` | JSONL file, generated from docs |
| **Target** (system under test) | `run(case) -> Response(output, retrieved_context, latency, tokens, cost)` | bare model, RAG pipeline, HTTP endpoint |
| **Scorer** | `score(case, response) -> Score(value, reason)` | exact match, regex, LLM judge, faithfulness, retrieval hit-rate |
| **Runner** | Runs targets × dataset × scorers with caching, retries, concurrency, resume | single implementation |
| **Report** | Aggregates scores into a leaderboard with breakdowns | HTML, JSON, CSV |

**Targets cover both modes with one interface:**
- `ModelTarget`: a bare model with a prompt template. Used for "which open-source model is best?"
- `RagTarget`: the reference RAG pipeline (chunker → embedder → vector store → model). Every stage is swappable, so chunking and embedding models get compared too, not just the LLM.
- `HttpTarget`: wraps a client's existing pipeline/chatbot endpoint through a request/response mapping in config.

**Providers:** a single OpenAI-compatible client covers Ollama, vLLM, LM Studio, and most hosted APIs (Together, Groq, OpenRouter). Add native providers only when one is needed. Every provider is tagged `local` or `remote`, and the runner **refuses remote providers on datasets marked `confidential: true`**. That check covers the judge and the test-set generator too, not only the target.

## Everything is config

One YAML file per client engagement, with no code changes between clients:

```yaml
# real schema; see examples/realestate/*.yaml for working configs
project: smith-realty
confidential: true                 # default; blocks non-local hosts for targets, judge, embedder, generator
local_hosts: [client-bot]          # client LAN hostnames treated as local
corpus: ./docs
dataset: ./eval.v1.jsonl
pack: rag                          # default system prompt + scorers
targets:
  - {name: llama3.2-3b, type: rag, model: ollama/llama3.2:3b, retriever: hybrid, embedder: ollama/nomic-embed-text, chunk_size: 800, top_k: 4}
  - {name: qwen3-4b,    type: rag, model: ollama/qwen3:4b,    retriever: bm25, top_k: 4}
  - {name: their-bot,   type: http, url: http://client-bot/api/chat, body: {message: "{input}"},
     output_path: "$.reply", context_path: "$.sources"}
scorers: [contains, refusal, retrieval_hit, answer_correctness, faithfulness]   # optional: pack provides defaults
judge: ollama/llama3.1:8b          # size to available VRAM (dev box: 8 GB)
weights: {quality: 0.7, latency: 0.2, cost: 0.1}
budgets: {latency_s: 4.0, cost_per_1k: 1.0, min_quality: 0.5}
```

## The key module: dataset builder

Small firms don't have labelled eval sets. Building one quickly is the product's real value.

1. **Ingest:** a loader per file type turns files into normalized `Document(text, metadata)`. This is the plugin point for images and audio later (OCR, transcription).
2. **Generate:** an LLM writes Q&A cases from document chunks across several types: single-fact lookup, multi-document, numeric/date, and **unanswerable questions**, which test hallucination. It also writes niche-specific personas (e.g. "client asking about closing costs").
3. **Review:** a quick review pass (CLI first, simple UI later) to accept, edit, or reject each case. Generated cases are *not* ground truth until a person has reviewed them.
4. **Version:** datasets are immutable once versioned, so results stay comparable across runs.

Build this ourselves (about 300 lines) instead of depending on Ragas's generator. Ragas's last release was v0.4.3 (Jan 2026), and its activity has reportedly stalled, so it isn't a safe foundation.

## Scoring and selection

- **Deterministic scorers first** (exact/regex/contains, JSON validity, retrieval hit-rate when gold source docs are known). They're cheap and reliable.
- **LLM-judge scorers** (correctness, faithfulness to retrieved context, tone, refusal) use rubric prompts and report a reason for each score.
- **Judge calibration:** label about 30–50 cases by hand per engagement and measure the judge's agreement with those labels. If agreement is low, fix the rubric before trusting the rankings.
- **Selection is more than quality.** The report ranks on a weighted score of quality, p50/p95 latency, cost per 1k queries, and hardware fit (VRAM needed). The recommendation for a small business is often "the 8B model is 95% as good at 1/4 the cost."
- **Reports** include the leaderboard, per-tag breakdowns (e.g. "fails on numeric questions"), the worst-case examples side by side, and a run diff against previous runs.

## Build vs reuse

Own the thin core (contracts, runner, config, reports, dataset builder). Treat existing libraries as **optional scorer plugins** so we're never locked in:
- DeepEval / Ragas metrics are wrapped as `Scorer` plugins where they add value.
- promptfoo and Inspect AI serve as design references, not dependencies.

Why own the core: none of these handles "generate a suite from a client's docs → compare models *and* pipelines → cost/latency-weighted recommendation" end to end. The core is small, and adoption risk in this space is real.

## Repo layout

```
evalkit/
  core/        # contracts, runner, cache, config schema
  providers/   # openai_compat.py (+ local/remote tagging)
  loaders/     # pdf, docx, txt, csv  (later: image, audio)
  rag/         # chunking, retrieval (bm25, embeddings)
  datasets/    # generator, reviewer, versioning
  targets/     # model, rag, http
  scorers/     # deterministic/, judge/, plugins/ (deepeval, ragas)
  packs/       # rag/, chatbot/  (adapter + scorers + templates)
  report/      # html, json, diff
  cli.py       # evalkit build-dataset | run | report | compare
clients/       # per-engagement configs + data (gitignored)
tests/
```

## Phases

1. **Vertical slice (RAG):** core contracts, OpenAI-compatible provider, PDF/TXT loaders, dataset generator plus CLI review, `RagTarget`, 3 scorers, HTML leaderboard. Done when you can run 2–3 Ollama models on a sample real-estate doc set and get a ranked report.
2. **Pipelines and selection:** `HttpTarget`, latency/cost/VRAM in the report, weighted ranking, run diffs, judge calibration workflow, confidentiality guard.
3. **Chatbot pack:** multi-turn cases, persona simulation, tone/policy scorers.
4. **Breadth:** DOCX/CSV loaders, DeepEval/Ragas scorer plugins, hosted providers, optional review UI.
5. **Multimodal:** image loaders (OCR / vision models), audio (transcription), and matching scorers.

## Open questions (can wait)

- What's your hardware? GPU VRAM decides which local models and judge sizes are realistic.
- Do you have a first pilot client or niche? A real sample doc set would sharpen phase 1.
- Should the deliverable to clients be the report only, or also a deployable config of the winning setup?
