"""Interactive review of pending cases. Saves after every decision, so quitting loses nothing."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .store import load_cases, save_cases

HELP = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit"


def review(path: Path, input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
           accept_all: bool = False) -> dict[str, int]:
    cases = load_cases(path, include_all=True)
    pending = [c for c in cases if c.status == "pending"]
    counts = {"accepted": 0, "edited": 0, "rejected": 0, "skipped": 0, "remaining": 0}
    if accept_all:
        for c in pending:
            c.status = "accepted"
        save_cases(path, cases)
        counts["accepted"] = len(pending)
        return counts
    print_fn(f"{len(pending)} pending cases in {path}. {HELP}")
    for n, c in enumerate(pending, 1):
        print_fn(f"\n[{n}/{len(pending)}] {c.id}  tags={','.join(c.tags)}  sources={','.join(c.metadata.get('near_docs') or c.metadata.get('source_docs') or [])}")
        for t in c.history:
            print_fn(f"  {t['role']:>9}: {t['content']}")
        if c.metadata.get("persona"):
            print_fn(f"  persona: {c.metadata['persona']}")
        print_fn(f"  Q: {c.input}")
        print_fn(f"  A: {c.expected if c.expected is not None else '(unanswerable: should refuse)'}")
        if c.metadata.get("check_hits"):
            print_fn("  Check the corpus doesn't actually answer it. Closest passages:")
            for h in c.metadata["check_hits"]:
                print_fn(f"    - {h['doc']} (bm25 {h['score']}): {h['snippet'][:140]!r}")
        choice = (input_fn("> ").strip().lower() or "s")[0]
        if choice == "q":
            counts["remaining"] = len(pending) - n + 1
            break
        if choice == "a":
            c.status = "accepted"
            counts["accepted"] += 1
        elif choice == "r":
            c.status = "rejected"
            counts["rejected"] += 1
        elif choice == "e":
            q = input_fn("  new question (enter = keep): ").strip()
            a = input_fn("  new answer (enter = keep, '-' = unanswerable): ").strip()
            if q:
                c.input = q
            if a == "-":
                c.expected = None
                c.tags = [t for t in c.tags if t != "unanswerable"] + ["unanswerable"]
                c.metadata["source_docs"] = []
            elif a:
                c.expected = a
                c.tags = [t for t in c.tags if t != "unanswerable"]
            c.status = "accepted"
            c.metadata["edited"] = True
            counts["edited"] += 1
        else:
            counts["skipped"] += 1
            continue
        save_cases(path, cases)
    return counts
