from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args) -> int:
    from .core.config import load_config
    from .core.runner import run
    from .report import aggregate, load_results

    cfg = load_config(args.config)
    run_dir = run(cfg, resume=Path(args.resume) if args.resume else None, limit=args.limit)
    meta, results = load_results(run_dir)
    s = aggregate(meta, results)
    print("\nRanking:")
    for name in s["ranking"]:
        t = s["targets"][name]
        q = "-" if t["quality"] is None else f"{t['quality']:.2f}"
        print(f"  {t['rank']}. {name:40s} score={t['composite']:.2f} quality={q} errors={t['errors']}/{t['n']}")
    if meta.get("is_mock"):
        print("  (mock run: pipeline demo, not a real model ranking)")
    print(f"\nReport: {run_dir / 'report.html'}")
    return 0


def _cmd_report(args) -> int:
    from .report import write_report

    baseline = False if args.no_diff else (Path(args.baseline) if args.baseline else None)
    s = write_report(Path(args.run_dir), baseline=baseline)
    if s.get("diff"):
        print(f"Compared with {s['diff']['baseline']}: {s['diff']['n_regressions']} regressions, {s['diff']['n_improvements']} improvements")
    print(f"Report: {Path(args.run_dir) / 'report.html'}")
    return 0


def _cmd_ingest(args) -> int:
    from .core.config import load_config
    from .loaders import ingest
    from .rag.chunking import chunk_documents

    cfg = load_config(args.config)
    if not cfg.corpus:
        raise ValueError("config has no 'corpus'")
    docs, skipped = ingest(cfg.corpus)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"{len(docs)} documents, {sum(len(d.text) for d in docs):,} chars, {len(chunks)} chunks (size {args.chunk_size})")
    for d in docs[:50]:
        print(f"  {d.id:50s} {len(d.text):>8,} chars")
    if len(docs) > 50:
        print(f"  ... {len(docs) - 50} more")
    for s in skipped:
        print(f"  skipped: {s}")
    return 0


def _cmd_build_dataset(args) -> int:
    from .core.config import load_config
    from .core.registry import load_builtins
    from .core.runner import build_env
    from .datasets.generator import generate_cases
    from .datasets.store import load_cases, save_cases

    load_builtins()
    cfg = load_config(args.config)
    gen = cfg.generator
    model = args.model or gen.get("model") or cfg.judge
    if not model:
        raise ValueError("no generator model: set generator.model (or judge) in the config, or pass --model")
    if not cfg.corpus:
        raise ValueError("config has no 'corpus'")
    out = Path(args.out) if args.out else cfg.dataset
    if not out:
        raise ValueError("no output path: set 'dataset' in the config or pass --out")
    existing = load_cases(out, include_all=True) if out.exists() else []
    env = build_env(cfg)
    n = args.n or int(gen.get("n", 20))
    print(f"Generating {n} cases from {cfg.corpus} with {model} ...")
    from .datasets.generator import CHATBOT_MIX

    mix = gen.get("mix") or (CHATBOT_MIX if cfg.pack == "chatbot" else None)
    new, errors = generate_cases(env, cfg.corpus, model, n=n, mix=mix, business=gen.get("business", "a small business"),
                                 persona=gen.get("persona", "a client of the business"), personas=gen.get("personas"),
                                 seed=args.seed, existing=existing)
    save_cases(out, existing + new)
    print(f"Added {len(new)} pending cases to {out} ({len(errors)} generation errors).")
    for e in errors[:5]:
        print(f"  error: {e}")
    print(f"Next: review them with `evalkit review {out}` (pending cases are excluded from runs).")
    return 0


def _cmd_review(args) -> int:
    from .datasets.review import review

    counts = review(Path(args.dataset), accept_all=args.accept_all)
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0


def _cmd_freeze(args) -> int:
    from .datasets.store import freeze

    out = freeze(Path(args.dataset))
    print(f"Frozen (read-only): {out}. Point 'dataset:' at it so results stay comparable.")
    return 0


def _cmd_check(args) -> int:
    from .core.config import load_config
    from .core.registry import load_builtins
    from .core.runner import build_env, build_scorers, build_targets
    from .datasets.store import load_cases

    load_builtins()
    cfg = load_config(args.config)
    env = build_env(cfg)
    ok = True
    print(f"project: {cfg.project}  confidential: {cfg.confidential}")
    if cfg.dataset and cfg.dataset.exists():
        all_cases = load_cases(cfg.dataset, include_all=True)
        usable = load_cases(cfg.dataset)
        print(f"dataset: {cfg.dataset} ({len(usable)} usable, {len(all_cases) - len(usable)} pending/rejected)")
    else:
        print(f"dataset: MISSING ({cfg.dataset})")
        ok = False
    targets = build_targets(cfg, env)
    specs = sorted({m for t in targets for m in t.models() if not m.startswith("http:")} | ({cfg.judge} if cfg.judge else set()))
    for spec in specs:
        try:
            p, model = env.resolve(spec, "check")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAIL  model {spec}: {e}")
            continue
        available = p.list_models() if hasattr(p, "list_models") else None
        if available is not None and model not in available and f"{model}:latest" not in available:
            ok = False
            print(f"  MISSING model {spec} (not on {p.name}; for Ollama run: ollama pull {model})")
        elif available is not None:
            print(f"  ok    model {spec}")
    for obj in [*targets, *build_scorers(cfg, env)]:
        kind = "target" if hasattr(obj, "run") else "scorer"
        try:
            obj.setup()
            print(f"  ok    {kind} {obj.name}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {kind} {obj.name}: {e}")
    for name, p in env._providers.items():
        print(f"  provider {name} ({'local' if p.local else 'REMOTE'} {p.base_url}): {p.ping()}")
    return 0 if ok else 1


def _cmd_calibrate(args) -> int:
    from .calibrate import export_sample, score_labels

    run_dir = Path(args.run_dir)
    if args.action == "export":
        out = Path(args.file or run_dir / "labels.csv")
        n = export_sample(run_dir, out, n=args.n)
        print(f"Wrote {n} outputs to {out}.")
        print("Fill each human_<scorer> column with 1-5 (same rubric as the judge), then run:")
        print(f"  evalkit calibrate score {run_dir} {out}")
    else:
        if not args.file:
            raise ValueError("calibrate score needs the labels CSV path")
        res = score_labels(run_dir, Path(args.file))
        for name, c in res["scorers"].items():
            corr = "-" if c["correlation"] is None else f"{c['correlation']:.2f}"
            print(f"  {name:22s} n={c['n']:3d} within1={c['within_1']:.0%} pass-agree={c['pass_agreement']:.0%} "
                  f"corr={corr} bias={c['judge_bias']:+.2f} -> {c['verdict']}")
        if res["bad_rows"]:
            print(f"  ({res['bad_rows']} rows/cells ignored: unknown key or value outside 1-5)")
        print(f"Report updated: {run_dir / 'report.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalkit", description="Modular evaluation of models and AI pipelines on your own data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run all targets over the dataset and write a report")
    r.add_argument("config")
    r.add_argument("--resume", help="existing run directory to continue")
    r.add_argument("--limit", type=int, help="only the first N cases")
    r.set_defaults(fn=_cmd_run)

    rp = sub.add_parser("report", help="re-render the report for a run directory")
    rp.add_argument("run_dir")
    rp.add_argument("--baseline", help="run directory to compare against (default: previous run)")
    rp.add_argument("--no-diff", action="store_true")
    rp.set_defaults(fn=_cmd_report)

    i = sub.add_parser("ingest", help="show what the loaders extract from the corpus")
    i.add_argument("config")
    i.add_argument("--chunk-size", type=int, default=800)
    i.add_argument("--chunk-overlap", type=int, default=100)
    i.set_defaults(fn=_cmd_ingest)

    b = sub.add_parser("build-dataset", help="generate draft test cases from the corpus (status=pending)")
    b.add_argument("config")
    b.add_argument("--n", type=int)
    b.add_argument("--model", help="generator model spec (default: generator.model, then judge)")
    b.add_argument("--out", help="dataset file to append to (default: config 'dataset')")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=_cmd_build_dataset)

    rv = sub.add_parser("review", help="accept/edit/reject pending cases interactively")
    rv.add_argument("dataset")
    rv.add_argument("--accept-all", action="store_true", help="accept every pending case without review (not recommended)")
    rv.set_defaults(fn=_cmd_review)

    f = sub.add_parser("freeze", help="write accepted cases to an immutable versioned file")
    f.add_argument("dataset")
    f.set_defaults(fn=_cmd_freeze)

    cal = sub.add_parser("calibrate", help="measure judge agreement with human labels")
    cal.add_argument("action", choices=["export", "score"])
    cal.add_argument("run_dir")
    cal.add_argument("file", nargs="?", help="labels CSV (export: output path, default RUN_DIR/labels.csv)")
    cal.add_argument("--n", type=int, default=40)
    cal.set_defaults(fn=_cmd_calibrate)

    c = sub.add_parser("check", help="validate config, run the confidentiality guard, ping providers")
    c.add_argument("config")
    c.set_defaults(fn=_cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    from .core.config import ConfigError
    from .core.env import ConfidentialityError

    try:
        return args.fn(args)
    except (ConfigError, ConfidentialityError, FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
