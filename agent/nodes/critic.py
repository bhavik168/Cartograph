"""Critic: scores the draft and decides whether the graph loops.

This node is the reason the graph is cyclic. A failing critique routes back to
the supervisor with targeted directives instead of returning a bad brief.

Known limitation, stated plainly here and in the README: this is an LLM judging
an LLM from the same family, so it is likely lenient about failure modes it
shares with the synthesizer. A brief that passes has "passed the critic" — it
has not been verified true. Scoring against explicit named criteria and
requiring directives on failure narrows the leniency; it does not remove it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from agent.auditor.meter import estimate_tokens
from agent.memory import render_findings
from agent.runtime import RunContext
from agent.schemas import Critique, InputComposition
from agent.state import AnalystState

CRITERIA = {
    "grounding": "Every claim's evidence actually supports it, and every quote "
    "appears in the findings. Fabricated or mismatched evidence fails outright.",
    "coverage": "The brief addresses the whole question, not a convenient slice "
    "of it.",
    "specificity": "Claims are concrete and falsifiable. Hedged, generic, or "
    "unfalsifiable statements score low.",
    "calibration": "Confidence levels match the strength of the evidence, and "
    "limitations name the real gaps rather than boilerplate.",
}

PASS_THRESHOLD = 0.7

SYSTEM = """You are a demanding reviewer of research briefs. You score against
fixed criteria and you do not grade on effort.

Score each criterion from 0.0 to 1.0:
""" + "\n".join(f"- {name}: {desc}" for name, desc in CRITERIA.items()) + f"""

Set passed=true only if every criterion is at least {PASS_THRESHOLD}.

If it fails, write revision_directives: specific, actionable instructions naming
the claim or section at fault and what would fix it. "Improve grounding" is
useless. "Claim 2 cites policy.md but the quote is about onboarding, not
retention — find a passage about retention or drop the claim" is a directive.

If it passes, revision_directives must be empty."""


def make_critic(ctx: RunContext) -> Callable:
    async def critic(state: AnalystState) -> dict:
        started = time.perf_counter()
        draft = state.get("draft")
        revision = state.get("revision_count", 0)

        if draft is None:
            # Defensive: the graph should never reach the critic without a
            # draft, but failing closed beats crashing the run.
            return {
                "critique": Critique(
                    passed=False,
                    scores={},
                    revision_directives=["No draft was produced; synthesise one."],
                ),
                "revision_directives": ["No draft was produced; synthesise one."],
                "trace": [ctx.span("critic", "no_draft", started)],
            }

        findings_text = state.get("scratchpad") or render_findings(
            state.get("findings") or []
        )
        task = (
            f"Research question:\n{state['question']}\n\n"
            f"Evidence available to the writer:\n{findings_text}\n\n"
            f"Draft brief:\n{draft.model_dump_json(indent=2)}"
        )

        critique: Critique = await ctx.llm.call(
            Critique,
            [("system", SYSTEM), ("user", task)],
            node="critic",
            cause="critique",
            tier="strong",
            revision_index=revision,
            composition=InputComposition(
                system=estimate_tokens(SYSTEM),
                findings=estimate_tokens(findings_text),
                other=estimate_tokens(task) - estimate_tokens(findings_text),
            ),
        )

        # Enforce the threshold ourselves rather than trusting the model's own
        # pass/fail: the scores are the contract, the boolean is a summary of
        # them, and models routinely produce a cheerful passed=true over
        # failing numbers.
        if critique.scores:
            below = [k for k, v in critique.scores.items() if v < PASS_THRESHOLD]
            if below and critique.passed:
                critique.passed = False
                if not critique.revision_directives:
                    critique.revision_directives = [
                        f"Raise {name}: scored {critique.scores[name]:.2f}, "
                        f"below the {PASS_THRESHOLD} threshold."
                        for name in below
                    ]
        if not critique.passed and not critique.revision_directives:
            critique.revision_directives = [
                "The critique reported failure without directives; re-examine "
                "grounding and coverage and strengthen the weakest claims."
            ]
        if critique.passed:
            critique.revision_directives = []

        return {
            "critique": critique,
            "revision_directives": critique.revision_directives,
            "trace": [
                ctx.span(
                    "critic",
                    "scored",
                    started,
                    passed=critique.passed,
                    scores=critique.scores,
                    directives=len(critique.revision_directives),
                    revision=revision,
                )
            ],
        }

    return critic
