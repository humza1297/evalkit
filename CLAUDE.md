# evalkit

A modular evaluation system for comparing open-source models and client AI pipelines (RAG, chatbot) on a client's own data. The design is in `PLAN.md`.

**Always start by reading `CHECKLIST.md`.** Continue from the first unchecked item, and tick items as soon as they land on disk.

- Venv: `.venv\Scripts\python`. Tests: `.venv\Scripts\python -m pytest -q`
- Offline demo (no models needed): `.venv\Scripts\python -m evalkit run examples/realestate/evalkit.mock.yaml`
- The core (`evalkit/core`) must stay workflow-agnostic. Workflow-specific logic belongs in targets, scorers, or packs.
- Client data is confidential by default. Never route `confidential: true` projects to remote providers.
