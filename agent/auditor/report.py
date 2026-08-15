"""Renders audit.md (human) and audit.json (machine).

Runs automatically at the end of ``cli.py ask`` and standalone via
``cli.py audit <run_id>``, which re-reads ``tokens.jsonl`` from disk — so a run
that crashed halfway can still be audited from its partial stream.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.auditor.attribute import Audit, build_audit
from agent.auditor.meter import load_events
from agent.schemas import TokenEvent


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no data)_\n"
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}\n"


def render_markdown(audit: Audit) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[str] = []
    w = out.append

    w(f"# Token Audit — run {audit.run_id or ts}")
    w("")
    if audit.question:
        w(f'Question: "{audit.question}"')
    if audit.outcome:
        w(f"Outcome: {audit.outcome}")
    w("")

    w("## Totals")
    w("")
    w(
        f"- total_tokens **{audit.total_tokens:,}** "
        f"({audit.input_tokens:,} in / {audit.output_tokens:,} out"
        + (f", {audit.cached_input_tokens:,} cached" if audit.cached_input_tokens else "")
        + ")"
    )
    w(
        f"- est_cost_usd **${audit.est_cost_usd:.4f}** "
        "(per agent/auditor/pricing.py — verify rates against your provider)"
    )
    w(
        f"- wall_clock {audit.wall_clock_s:.1f}s · llm_calls {audit.llm_calls} "
        f"· failed_calls {audit.failed_calls} · schema_repairs {audit.schema_repairs}"
    )
    w("")

    w("## Where the tokens went — by node")
    w("")
    w(
        _table(
            ["node", "calls", "input", "output", "% of total", "est_usd"],
            [
                [
                    b.key,
                    str(b.calls),
                    f"{b.input_tokens:,}",
                    f"{b.output_tokens:,}",
                    f"{b.pct_of_total:.1f}%",
                    f"${b.usd:.4f}",
                ]
                for b in audit.by_node
            ],
        )
    )

    w("## Why the tokens were spent — by cause")
    w("")
    w(
        _table(
            ["cause", "tokens", "%", "class"],
            [
                [
                    b.key,
                    f"{b.total_tokens:,}",
                    f"{b.pct_of_total:.1f}%",
                    b.cause_class or "",
                ]
                for b in audit.by_cause
            ],
        )
    )
    w(
        f"- productive {audit.productive_tokens:,} · overhead "
        f"{audit.overhead_tokens:,} · waste {audit.waste_tokens:,}"
    )
    w("")
    w(
        f">>> **WASTE RATIO: {audit.waste_ratio:.1%}** "
        "(schema_repair + retry_transient + revision)"
    )
    w("")

    w("## What filled the context")
    w("")
    w(
        _table(
            ["component", "input tokens", "% of measured input"],
            [
                [key, f"{value:,}", f"{audit.composition_pct.get(key, 0.0):.1f}%"]
                for key, value in sorted(
                    audit.composition.items(), key=lambda kv: kv[1], reverse=True
                )
            ],
        )
    )

    w("## Cost of the revision loop")
    w("")
    w(
        _table(
            ["pass", "calls", "tokens", "est_usd"],
            [
                [
                    "first pass" if p.revision_index == 0 else f"revision {p.revision_index}",
                    str(p.calls),
                    f"{p.tokens:,}",
                    f"${p.usd:.4f}",
                ]
                for p in audit.pass_costs
            ],
        )
    )
    if len(audit.pass_costs) > 1:
        first = audit.pass_costs[0]
        rest = sum(p.usd for p in audit.pass_costs[1:])
        w(
            f"Quality cost ${rest:.4f} on top of a ${first.usd:.4f} first pass "
            f"({(rest / first.usd * 100) if first.usd else 0:.0f}% surcharge)."
        )
        w("")

    w("## Tier efficiency")
    w("")
    w(f"- cheap tier: {audit.cheap_call_share:.0%} of calls, "
      f"{audit.cheap_spend_share:.0%} of spend")
    w(f"- strong tier: {1 - audit.cheap_call_share:.0%} of calls, "
      f"{audit.strong_spend_share:.0%} of spend")
    w("")
    w(
        _table(
            ["provider", "calls", "tokens", "est_usd"],
            [
                [b.key, str(b.calls), f"{b.total_tokens:,}", f"${b.usd:.4f}"]
                for b in audit.by_provider
            ],
        )
    )

    w("## Recommendations")
    w("")
    w("_Rule-based, not LLM-generated: deterministic, free, and it adds no tokens "
      "to the run it is auditing._")
    w("")
    for rec in audit.recommendations:
        w(f"- {rec}")
    w("")
    return "\n".join(out)


def write_report(
    run_dir: Path | str,
    events: list[TokenEvent] | None = None,
    *,
    run_id: str = "",
    question: str = "",
    outcome: str = "",
    wall_clock_s: float = 0.0,
) -> Audit:
    """Build the audit and write both files. Reads tokens.jsonl if events are omitted."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if events is None:
        tokens_path = run_dir / "tokens.jsonl"
        events = load_events(tokens_path) if tokens_path.exists() else []

    audit = build_audit(
        events,
        run_id=run_id or run_dir.name,
        question=question,
        outcome=outcome,
        wall_clock_s=wall_clock_s,
    )
    (run_dir / "audit.json").write_text(
        json.dumps(audit.model_dump(), indent=2), encoding="utf-8"
    )
    (run_dir / "audit.md").write_text(render_markdown(audit), encoding="utf-8")
    return audit
