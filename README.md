# evalkit

Build an evaluation suite from a client's own documents, then compare open-source models, RAG configurations, or the client's existing AI endpoint on it. Each run produces a ranked report that weighs quality, latency and cost.

The design is in [PLAN.md](PLAN.md) and build progress is in [CHECKLIST.md](CHECKLIST.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[pdf,docx,dev]
```

## Offline demo (no models needed)

```bash
.venv\Scripts\python -m evalkit run examples/realestate/evalkit.mock.yaml
```

The demo uses the deterministic `mock` provider, so its ranking is a pipeline demo and not a real comparison.

## Workflow for a client engagement

1. Put the client's documents in `clients/<name>/docs/`. The `clients/` directory is gitignored.
2. Copy `examples/realestate/evalkit.ollama.yaml` to `clients/<name>/evalkit.yaml` and edit it.
3. `evalkit ingest CONFIG` shows what was extracted and which files were skipped.
4. `evalkit build-dataset CONFIG --n 40` drafts test cases. They are marked `pending`.
5. `evalkit review clients/<name>/eval.jsonl` lets you accept, edit or reject each draft. Progress is saved after every decision.
6. `evalkit freeze clients/<name>/eval.jsonl` writes a read-only `eval.v1.jsonl`. Point `dataset:` at it.
7. `evalkit check CONFIG` validates the config, runs the confidentiality guard and pings the providers.
8. `evalkit run CONFIG` writes `runs/<timestamp>/report.html`. If a run is interrupted, continue it with `--resume runs/<timestamp>`.

## Concepts

| Piece | What it is | Built-ins |
|---|---|---|
| Target | System under test | `model` (bare model + prompt), `rag` (chunk → retrieve → generate), `http` (client endpoint) |
| Scorer | Value in [0, 1], or n/a, or error | `contains` (`must_contain`), `exact_match`, `regex`, `json_valid`, `must_not_contain`, `retrieval_hit`, `refusal`, `answer_correctness`, `faithfulness`, `rubric`, `tone`, `policy` |
| Provider | Model host | `ollama`, `vllm`, `lmstudio` (local); `openrouter`, `together`, `groq` (remote); `mock`; any OpenAI-compatible URL under `providers:` |
| Pack | Workflow defaults | `rag`, `chatbot`, `qa` |

**Model specs** take the form `provider/model`, for example `ollama/llama3.2:3b` or `openrouter/meta-llama/llama-3.1-8b-instruct`.

**Confidentiality.** `confidential: true` is the default. It blocks any provider or HTTP target whose host is not loopback or a private address, and that applies to the judge, the embedder and the generator as well as the targets. A client's LAN hostname can be added under `local_hosts:`.

**Scoring rules:**
- A target error counts as quality 0.
- A scorer or judge error, such as unparseable output, is excluded from quality and shown in the report.
- A scorer with weight 0 is diagnostic only.
- Latency and cost get full marks within the `budgets:` set in the config, and the score decays beyond them.

**Case format:** one JSON object per line in a `.jsonl` file.

```json
{"id": "fee-01", "input": "What is your listing commission?", "expected": "5 percent", "tags": ["numeric"],
 "metadata": {"source_docs": ["commission_policy.md"], "must_contain": ["5 percent"]}}
```

Tag a case `unanswerable` (with no `expected`) to test whether the system refuses instead of hallucinating.

## Chatbots

Use `pack: chatbot`. Cases can carry prior turns:

```json
{"id": "chat-01", "history": [{"role": "user", "content": "What is your commission?"}, {"role": "assistant", "content": "5 percent."}],
 "input": "And for homes above $1.5M?", "expected": "4 percent", "tags": ["followup"]}
```

To generate follow-up conversations, set `generator.personas: [...]`. The chatbot pack's default mix includes `followup` cases.

The extra scorers for chatbots are:
- `tone`, with an optional `criteria:` setting for brand voice.
- `policy`, with `rules: [...]`.
- `must_not_contain`, with `phrases: [...]`. A phrase starting with `re:` is treated as a regex.

For a live bot, the `http` target accepts `body: {messages: "{history}", text: "{input}"}`.

See `examples/realestate/evalkit.chatbot.mock.yaml` for a full config.

## Comparing runs and trusting the judge

- **Comparing runs:** each report includes a "Changes vs <previous run>" section, which shows per-target deltas and per-case regressions. Use `evalkit report RUN --baseline OTHER_RUN` to compare against a specific run.
- **Rescoring:** changing scorers and running with `--resume` rescores the stored responses without regenerating them.
- **Judge calibration:**
  1. `evalkit calibrate export RUN` writes a CSV of outputs. Judge scores are hidden from the labeler.
  2. Fill the `human_*` columns with a score from 1 to 5.
  3. `evalkit calibrate score RUN labels.csv` reports agreement and a verdict ("trust" or "fix rubric or judge"), and adds them to the report.
- **Hardware fit:** the VRAM column is measured from Ollama when the model is loaded, and estimated otherwise. Set `budgets.vram_gb` to flag models that are too big.

## Extending

Each extension point is a decorator:
- Scorers: `@register("scorer", "name")`
- Targets: `@register("target", "name")`
- File types: `@register("loader", ".ext")`
- Retrievers: `@register("retriever", "name")`

Add the new module to `BUILTIN_MODULES` in `evalkit/core/registry.py`.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```
