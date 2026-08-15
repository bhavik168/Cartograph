"""Synthesizer: findings in, draft ``Brief`` out.

Strong tier. This is where the tier split earns its keep — routing and
extraction ran on Haiku, and the one call that genuinely needs reasoning over
the whole evidence set runs on Sonnet.

Also the node that triggers scratchpad compaction: it sees the full findings
list, so it is the right place to notice that the list has outgrown its budget
and compress before the next revision carries it around again.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agent.auditor.meter import estimate_tokens
from agent.memory import compact, render_findings, should_compact
from agent.runtime import RunContext
from agent.schemas import Brief, InputComposition
from agent.state import AnalystState

SYSTEM = """You write evidenced research briefs.

Hard rules:
- Every claim carries at least one piece of evidence with a verbatim quote and
  its source filename. A claim you cannot evidence does not go in the brief; it
  goes in open_questions.
- Confidence reflects the evidence, not your prior: "high" needs multiple
  independent, directly-on-point passages.
- limitations names what this brief cannot support — gaps in the corpus,
  single-source claims, questions the evidence only partly reaches.
- Never invent a source filename. Only use sources that appear in the findings."""


def make_synthesizer(ctx: RunContext) -> Callable:
    async def synthesizer(state: AnalystState) -> dict:
        started = time.perf_counter()
        findings = state.get("findings") or []
        scratchpad = state.get("scratchpad", "")
        revision = state.get("revision_count", 0)
        directives = state.get("revision_directives") or []
        previous = state.get("draft")
        updates: dict = {}
        spans = []

        if should_compact(findings, scratchpad):
            new_pad, before, after = await compact(
                ctx.llm,
                question=state["question"],
                findings=findings,
                scratchpad=scratchpad,
                revision_index=revision,
            )
            spans.append(
                ctx.span(
                    "memory",
                    "compacted",
                    tokens_before=before,
                    tokens_after=after,
                    saved=before - after,
                )
            )
            scratchpad = new_pad
            updates["scratchpad"] = new_pad
            findings_text = scratchpad
        else:
            findings_text = render_findings(findings)

        task = f"Research question:\n{state['question']}\n\nFindings:\n{findings_text}"
        if previous is not None and directives:
            task += (
                "\n\nYour previous draft failed review. Fix exactly these points, "
                "keeping everything that already worked:\n"
                + "\n".join(f"- {d}" for d in directives)
                + "\n\nPrevious draft:\n"
                + previous.model_dump_json(indent=2)
            )

        draft: Brief = await ctx.llm.call(
            Brief,
            [("system", SYSTEM), ("user", task)],
            node="synthesizer",
            cause="revision" if revision > 0 else "synthesis",
            tier="strong",
            revision_index=revision,
            composition=InputComposition(
                system=estimate_tokens(SYSTEM),
                scratchpad=estimate_tokens(scratchpad),
                findings=estimate_tokens(findings_text),
                other=estimate_tokens(task) - estimate_tokens(findings_text),
            ),
        )
        draft.question = state["question"]

        spans.append(
            ctx.span(
                "synthesizer",
                "drafted",
                started,
                claims=len(draft.claims),
                evidence=sum(len(c.evidence) for c in draft.claims),
                revision=revision,
            )
        )
        updates["draft"] = draft
        updates["trace"] = spans
        return updates

    return synthesizer
