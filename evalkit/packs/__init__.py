"""Workflow packs: defaults layered onto a project config. The core never branches on workflow."""
from __future__ import annotations

PACKS: dict[str, dict] = {
    "rag": {
        "target_defaults": {
            "system": (
                "You answer questions for a small business using ONLY the provided context. "
                "If the context does not contain the answer, say you don't know based on the provided documents. "
                "Be concise."
            ),
        },
        "scorers": [
            {"name": "contains", "weight": 1.0},
            {"name": "refusal", "weight": 1.0},
            {"name": "retrieval_hit", "weight": 0.0},  # diagnostic: shown per component, not part of quality
            {"name": "answer_correctness", "weight": 2.0},
            {"name": "faithfulness", "weight": 1.5},
        ],
    },
    "chatbot": {
        "target_defaults": {
            "system": (
                "You are the customer assistant for a small business. Be friendly, professional and brief. "
                "Use ONLY the provided context and the conversation so far. If you don't know, say you don't know "
                "and offer to connect the customer with a staff member. Never invent prices, dates or policies."
            ),
        },
        "scorers": [
            {"name": "answer_correctness", "weight": 2.0},
            {"name": "refusal", "weight": 1.0},
            {"name": "faithfulness", "weight": 1.0},
            {"name": "tone", "weight": 1.0},
            {"name": "contains", "weight": 0.5},
        ],
    },
    "qa": {
        "target_defaults": {},
        "scorers": [
            {"name": "contains", "weight": 1.0},
            {"name": "answer_correctness", "weight": 2.0},
        ],
    },
}
