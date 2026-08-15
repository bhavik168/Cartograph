"""Finalizer: last validation pass, then write the artifact.

Deliberately does no LLM work in the common case. It re-validates the draft,
stamps honest limitations describing how the run actually ended, and writes
``brief.json``. A run that exhausted its revisions or hit its budget ceiling
still produces a brief — it just says so, in the artifact, where a reader will
see it. Silent degradation would be the worse failure.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import ValidationError

from agent.runtime import RunContext
from agent.schemas import Brief, Claim
from agent.state import AnalystState


def _fallback_brief(state: AnalystState, reason: str) -> Brief:
    """A brief for a run that never produced one. Empty, and honest about it."""
    findings = state.get("findings") or []
    claims: list[Claim] = []
    for finding in findings:
        if finding.evidence:
            claims.append(
                Claim(
                    statement=finding.summary,
                    evidence=finding.evidence,
                    confidence="low",
                )
            )
    return Brief(
        question=state.get("question", ""),
        claims=claims,
        open_questions=[f.sub_question for f in findings if not f.evidence],
        limitations=[reason, "Assembled mechanically from raw findings, unreviewed."],
    )


def make_finalizer(ctx: RunContext) -> Callable:
    async def finalizer(state: AnalystState) -> dict:
        started = time.perf_counter()
        draft = state.get("draft")
        critique = state.get("critique")
        revision = state.get("revision_count", 0)
        halted = state.get("halted_reason")

        if draft is None:
            brief = _fallback_brief(state, halted or "No draft was ever produced.")
        else:
            brief = draft.model_copy(deep=True)

        limitations = list(brief.limitations)

        if halted:
            limitations.insert(0, halted)
        if critique is not None and not critique.passed:
            failed = ", ".join(
                f"{k} {v:.2f}" for k, v in sorted(critique.scores.items())
            )
            limitations.insert(
                0,
                f"Failed critic after {revision} revision(s) of "
                f"{ctx.max_revisions} allowed"
                + (f" (scores: {failed})" if failed else "")
                + ". Outstanding: "
                + "; ".join(critique.revision_directives[:3]),
            )
        limitations.append(
            "Reviewed by an LLM critic from the same model family as the writer; "
            "'passed' means passed that critic, not verified true."
        )

        # A deterministic grounding check the critic cannot fake: every cited
        # source must be one the researchers actually touched. A claim resting
        # entirely on an invented filename is demoted to an open question
        # rather than shipped.
        known_sources = {
            e.source for f in (state.get("findings") or []) for e in f.evidence
        }
        kept: list[Claim] = []
        open_questions = list(brief.open_questions)
        for claim in brief.claims:
            if not known_sources or any(e.source in known_sources for e in claim.evidence):
                kept.append(claim)
            else:
                cited = ", ".join(sorted({e.source for e in claim.evidence}))
                open_questions.append(claim.statement)
                limitations.append(
                    f"Dropped claim citing unknown source(s) [{cited}]: "
                    f"{claim.statement[:120]}"
                )

        brief = Brief(
            question=state.get("question", brief.question),
            claims=kept,
            open_questions=open_questions,
            limitations=limitations,
        )

        try:
            Brief.model_validate(brief.model_dump())
            valid = True
            error = None
        except ValidationError as exc:  # pragma: no cover - construction guarantees it
            valid = False
            error = str(exc)

        ctx.write_json("brief.json", brief.model_dump())

        return {
            "draft": brief,
            "trace": [
                ctx.span(
                    "finalizer",
                    "wrote_brief",
                    started,
                    claims=len(brief.claims),
                    limitations=len(brief.limitations),
                    passed_critic=bool(critique and critique.passed),
                    revisions=revision,
                    valid=valid,
                    error=error,
                )
            ],
        }

    return finalizer
