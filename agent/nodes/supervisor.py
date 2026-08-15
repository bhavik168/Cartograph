"""Supervisor: plans the work and routes it.

Runs on the cheap tier. Its job is a decision, not prose, so it is the clearest
case in the graph for tier routing: a Haiku call that costs a fraction of a
Sonnet call and produces a schema-constrained ``RoutingDecision``.

On the first pass it decomposes the question into sub-questions. On a revision
pass it reads the critic's directives and decides whether more research is
needed or the synthesizer can fix the draft on its own.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agent.auditor.meter import estimate_tokens
from agent.memory import render_findings
from agent.runtime import RunContext
from agent.schemas import InputComposition, RoutingDecision, SubQuestion
from agent.state import AnalystState

SYSTEM = """You are the supervisor of a research team with two specialist roles:

- researcher: searches a local document corpus and runs calculations. Give it
  one narrow, self-contained sub-question at a time.
- synthesizer: merges existing findings into a structured brief. It cannot
  gather new evidence.

Decompose only as far as the question actually needs. Fewer, sharper
sub-questions beat many overlapping ones; every extra researcher is a real cost.
Route to researcher when evidence is missing, to synthesizer when the findings
on hand are enough to draft or repair the brief."""


def make_supervisor(ctx: RunContext) -> Callable:
    async def supervisor(state: AnalystState) -> dict:
        started = time.perf_counter()
        revision = state.get("revision_count", 0)
        directives = state.get("revision_directives") or []
        findings = state.get("findings") or []
        scratchpad = state.get("scratchpad", "")

        if revision == 0:
            task = (
                f"Research question:\n{state['question']}\n\n"
                f"No research has been done yet. Decompose this into at most "
                f"{ctx.max_researchers} sub-questions and route to researcher."
            )
        else:
            task = (
                f"Research question:\n{state['question']}\n\n"
                f"A draft brief failed critic review (revision {revision} of "
                f"{ctx.max_revisions}). The critic asked for:\n"
                + "\n".join(f"- {d}" for d in directives)
                + "\n\nWhat we already know:\n"
                + (scratchpad or render_findings(findings, limit=6) or "(nothing)")
                + "\n\nIf the directives need evidence we do not have, route to "
                "researcher with sub-questions targeting exactly those gaps. If "
                "they are presentation or structure problems, route to "
                "synthesizer with no new sub-questions."
            )

        composition = InputComposition(
            system=estimate_tokens(SYSTEM),
            scratchpad=estimate_tokens(scratchpad),
            findings=estimate_tokens(render_findings(findings, limit=6)) if revision else 0,
            other=estimate_tokens(task),
        )

        decision: RoutingDecision = await ctx.llm.call(
            RoutingDecision,
            [("system", SYSTEM), ("user", task)],
            node="supervisor",
            cause="revision" if revision > 0 else "planning",
            tier="cheap",
            revision_index=revision,
            composition=composition,
        )

        # Trust the schema, not the model: clamp fan-out and guarantee the
        # graph always has somewhere to go.
        sub_questions = decision.sub_questions[: ctx.max_researchers]
        if "researcher" in decision.next_agents and not sub_questions:
            sub_questions = [SubQuestion(id="sq1", text=state["question"])]
        if not decision.next_agents:
            decision.next_agents = ["synthesizer"]

        return {
            "routing": decision,
            "pending_sub_questions": sub_questions,
            "trace": [
                ctx.span(
                    "supervisor",
                    "routed",
                    started,
                    next_agents=decision.next_agents,
                    sub_questions=len(sub_questions),
                    revision=revision,
                    rationale=decision.rationale[:300],
                )
            ],
        }

    return supervisor
