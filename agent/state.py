"""The LangGraph state object and its reducers.

The ``Annotated[list[...], operator.add]`` reducers are what let concurrently
running researcher branches write into the same state key without clobbering
each other: LangGraph merges each branch's partial update by calling the
reducer instead of overwriting. Plain keys (``draft``, ``critique``) use
last-write-wins, which is correct because only one node ever writes them.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agent.schemas import Brief, Critique, Finding, RoutingDecision, Span, SubQuestion

MAX_REVISIONS = 2


class AnalystState(TypedDict, total=False):
    # Inputs, set once at invoke time.
    question: str
    run_id: str

    # Accumulated by fan-out nodes; merged, never overwritten.
    findings: Annotated[list[Finding], operator.add]
    trace: Annotated[list[Span], operator.add]
    flags: Annotated[list[str], operator.add]

    # Single-writer keys.
    routing: RoutingDecision | None
    pending_sub_questions: list[SubQuestion]
    draft: Brief | None
    critique: Critique | None
    scratchpad: str

    # Loop control.
    revision_count: int
    revision_directives: list[str]
    halted_reason: str | None


def initial_state(question: str, run_id: str) -> AnalystState:
    return AnalystState(
        question=question,
        run_id=run_id,
        findings=[],
        trace=[],
        flags=[],
        routing=None,
        pending_sub_questions=[],
        draft=None,
        critique=None,
        scratchpad="",
        revision_count=0,
        revision_directives=[],
        halted_reason=None,
    )
