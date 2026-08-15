"""The StateGraph: wiring, conditional edges, and the bounded cycle.

    supervisor ──┬─► researcher ─► synthesizer ─► critic ─┬─► finalizer ─► END
                 └─► synthesizer ───────────────┘         │
                 ▲                                        │
                 └──────── passed=False, budget left ──────┘

Two conditional edges do all the interesting work:

``route_from_supervisor`` picks the specialist set for this pass.
``route_from_critic`` is the cycle: fail and there is revision budget left, go
back to the supervisor with directives; otherwise finalize.

The cycle is bounded by ``MAX_REVISIONS`` and by the optional USD ceiling. Both
exits finalize with an honest ``limitations`` entry rather than looping forever
or quietly reporting success.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.auditor.meter import BudgetExceeded
from agent.nodes import (
    make_critic,
    make_finalizer,
    make_researcher,
    make_supervisor,
    make_synthesizer,
)
from agent.runtime import RunContext
from agent.state import MAX_REVISIONS, AnalystState


def route_from_supervisor(state: AnalystState) -> Literal["researcher", "synthesizer"]:
    routing = state.get("routing")
    if routing is None:
        return "researcher"
    if "researcher" in routing.next_agents and state.get("pending_sub_questions"):
        return "researcher"
    return "synthesizer"


def route_from_critic(state: AnalystState, max_revisions: int = MAX_REVISIONS) -> str:
    """The cycle's exit condition. Pure, so it is directly unit-testable."""
    if state.get("halted_reason"):
        return "finalizer"
    critique = state.get("critique")
    if critique is None or critique.passed:
        return "finalizer"
    if state.get("revision_count", 0) >= max_revisions:
        return "finalizer"
    return "supervisor"


def make_revise(ctx: RunContext):
    """Increments the revision counter on the way back round the loop.

    A separate node rather than a side effect inside the critic: the counter is
    the loop bound, and burying a mutation that load-bearing inside a node that
    also makes an LLM call is how bounded loops stop being bounded.
    """

    async def revise(state: AnalystState) -> dict:
        count = state.get("revision_count", 0) + 1
        updates: dict = {
            "revision_count": count,
            "trace": [ctx.span("supervisor", "revision_started", revision=count)],
        }
        try:
            ctx.meter.check_budget()
        except BudgetExceeded as exc:
            updates["halted_reason"] = f"Halted at budget ceiling: {exc}"
            updates["trace"] = updates["trace"] + [
                ctx.span("supervisor", "budget_halt", reason=str(exc))
            ]
        return updates

    return revise


def route_after_revise(state: AnalystState) -> Literal["supervisor", "finalizer"]:
    return "finalizer" if state.get("halted_reason") else "supervisor"


def build_graph(ctx: RunContext, checkpointer=None):
    """Compile the graph. Pass a checkpointer to make runs resumable by thread id."""
    builder = StateGraph(AnalystState)

    builder.add_node("supervisor", make_supervisor(ctx))
    builder.add_node("researcher", make_researcher(ctx))
    builder.add_node("synthesizer", make_synthesizer(ctx))
    builder.add_node("critic", make_critic(ctx))
    builder.add_node("revise", make_revise(ctx))
    builder.add_node("finalizer", make_finalizer(ctx))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "synthesizer": "synthesizer"},
    )
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "critic")
    builder.add_conditional_edges(
        "critic",
        lambda s: route_from_critic(s, ctx.max_revisions),
        {"supervisor": "revise", "finalizer": "finalizer"},
    )
    builder.add_conditional_edges(
        "revise",
        route_after_revise,
        {"supervisor": "supervisor", "finalizer": "finalizer"},
    )
    builder.add_edge("finalizer", END)

    return builder.compile(checkpointer=checkpointer)
