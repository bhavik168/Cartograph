#!/usr/bin/env python3
"""Cartographer CLI.

    python cli.py ask "your research question"
    python cli.py ask "..." --max-usd 0.50 --max-revisions 1
    python cli.py audit <run_id>
    python cli.py runs

One command produces one run directory under ``runs/<run_id>/`` containing
brief.json, trace.jsonl, tokens.jsonl, audit.json and audit.md. Nothing is
hosted and nothing is pre-run: every number you see came from your own run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent.auditor.meter import BudgetExceeded, TokenMeter
from agent.auditor.report import write_report
from agent.graph import build_graph
from agent.llm import LLMClient, LLMConfig, LLMError, available_providers
from agent.memory import DEFAULT_CHECKPOINT_DB, checkpointer
from agent.runtime import RunContext
from agent.state import MAX_REVISIONS, initial_state

RUNS_DIR = Path("runs")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _outcome(state: dict, max_revisions: int) -> str:
    critique = state.get("critique")
    revisions = state.get("revision_count", 0)
    if state.get("halted_reason"):
        return f"halted — {state['halted_reason']}"
    if critique is None:
        return "finalized without a critique"
    if critique.passed:
        return (
            "passed critic on the first pass"
            if revisions == 0
            else f"passed critic on revision {revisions} of max {max_revisions}"
        )
    return f"failed critic after {revisions} revision(s) of max {max_revisions}"


async def run_ask(args: argparse.Namespace) -> int:
    _load_env()
    providers = available_providers()
    if not providers:
        print(
            "No provider key found. Copy .env.example to .env and set "
            "ANTHROPIC_API_KEY (primary) and/or OPENAI_API_KEY (fallback).",
            file=sys.stderr,
        )
        return 2
    if providers[0] == "openai":
        print("ANTHROPIC_API_KEY not set — running on the OpenAI fallback path.")

    run_id = args.run_id or new_run_id()
    run_dir = RUNS_DIR / run_id
    started = time.time()

    print(f"run {run_id} · providers: {' -> '.join(providers)}")
    print(f"question: {args.question}\n")

    meter = TokenMeter(path=run_dir / "tokens.jsonl", max_usd=args.max_usd)
    llm = LLMClient(meter, LLMConfig())
    ctx = RunContext(
        llm=llm,
        meter=meter,
        run_dir=run_dir,
        run_id=run_id,
        max_revisions=args.max_revisions,
    )

    state: dict = {}
    exit_code = 0
    try:
        with checkpointer(DEFAULT_CHECKPOINT_DB) as saver:
            if saver is None:
                print("note: langgraph-checkpoint-sqlite not installed; "
                      "this run will not be resumable.\n")
            graph = build_graph(ctx, checkpointer=saver)
            config = {
                "configurable": {"thread_id": args.thread_id or run_id},
                "recursion_limit": args.recursion_limit,
            }
            state = await graph.ainvoke(initial_state(args.question, run_id), config)
    except BudgetExceeded as exc:
        print(f"\nbudget ceiling hit: {exc}", file=sys.stderr)
        exit_code = 1
    except LLMError as exc:
        print(f"\nLLM failure: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt:
        print("\ninterrupted — writing a partial audit from what was captured.")
        exit_code = 130
    finally:
        wall_clock = time.time() - started
        audit = write_report(
            run_dir,
            meter.events,
            run_id=run_id,
            question=args.question,
            outcome=_outcome(state, args.max_revisions) if state else "run did not complete",
            wall_clock_s=wall_clock,
        )
        meter.close()
        ctx.close()

    brief = state.get("draft")
    if brief is not None:
        print(f"\nbrief: {len(brief.claims)} claim(s), "
              f"{len(brief.open_questions)} open question(s)")
        for i, claim in enumerate(brief.claims, 1):
            print(f"  {i}. [{claim.confidence}] {claim.statement}")
        if brief.limitations:
            print("\nlimitations:")
            for limitation in brief.limitations:
                print(f"  - {limitation}")

    flags = state.get("flags") or []
    if flags:
        print(f"\ninjection flags raised: {', '.join(sorted(set(flags)))}")

    print(
        f"\n{audit.llm_calls} llm calls · {audit.total_tokens:,} tokens · "
        f"${audit.est_cost_usd:.4f} est · waste ratio {audit.waste_ratio:.1%} "
        f"· {wall_clock:.1f}s"
    )
    print(f"artifacts: {run_dir}/ (read audit.md first)")
    return exit_code


def run_audit(args: argparse.Namespace) -> int:
    run_dir = RUNS_DIR / args.run_id
    if not (run_dir / "tokens.jsonl").exists():
        print(f"no tokens.jsonl in {run_dir}", file=sys.stderr)
        return 2

    question = ""
    brief_path = run_dir / "brief.json"
    if brief_path.exists():
        try:
            question = json.loads(brief_path.read_text(encoding="utf-8")).get(
                "question", ""
            )
        except json.JSONDecodeError:
            pass

    audit = write_report(run_dir, run_id=args.run_id, question=question)
    if args.json:
        print(json.dumps(audit.model_dump(), indent=2))
    else:
        print((run_dir / "audit.md").read_text(encoding="utf-8"))
    return 0


def run_runs(_: argparse.Namespace) -> int:
    if not RUNS_DIR.exists():
        print("no runs yet")
        return 0
    rows = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "audit.json").exists()),
        reverse=True,
    )
    if not rows:
        print("no runs yet")
        return 0
    print(f"{'run_id':<22} {'tokens':>10} {'usd':>9} {'waste':>7}  outcome")
    for run_dir in rows:
        data = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
        print(
            f"{run_dir.name:<22} {data.get('total_tokens', 0):>10,} "
            f"{data.get('est_cost_usd', 0.0):>9.4f} "
            f"{data.get('waste_ratio', 0.0):>6.1%}  {data.get('outcome', '')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cartographer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="run the graph on a research question")
    ask.add_argument("question")
    ask.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="budget ceiling; the cycle halts and finalizes honestly if crossed",
    )
    ask.add_argument("--max-revisions", type=int, default=MAX_REVISIONS)
    ask.add_argument("--run-id", default=None)
    ask.add_argument(
        "--thread-id",
        default=None,
        help="checkpointer thread id; reuse one to resume a halted run",
    )
    ask.add_argument("--recursion-limit", type=int, default=50)

    audit = sub.add_parser("audit", help="re-render the audit for an existing run")
    audit.add_argument("run_id")
    audit.add_argument("--json", action="store_true")

    sub.add_parser("runs", help="list completed runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        return asyncio.run(run_ask(args))
    if args.command == "audit":
        return run_audit(args)
    return run_runs(args)


if __name__ == "__main__":
    raise SystemExit(main())
