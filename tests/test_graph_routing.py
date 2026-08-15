"""Routing, the revision cycle, and its bound — all with a stubbed LLM.

These run the real graph. No API key, no network, no cost.
"""

from __future__ import annotations

import pytest

from agent.auditor.meter import TokenMeter
from agent.graph import build_graph, route_after_revise, route_from_critic, route_from_supervisor
from agent.llm import LLMClient, LLMConfig
from agent.runtime import RunContext
from agent.schemas import (
    Brief,
    Claim,
    Critique,
    Evidence,
    Finding,
    RoutingDecision,
    SubQuestion,
)
from agent.state import initial_state
from tests.stubs import StubResponse, StubScript, stub_factory

EVIDENCE = Evidence(source="a.md", quote="retention was 84%", relevance=0.9)


def brief(n: int = 1) -> Brief:
    return Brief(
        question="q",
        claims=[
            Claim(statement=f"claim {i}", evidence=[EVIDENCE], confidence="medium")
            for i in range(n)
        ],
        open_questions=[],
        limitations=[],
    )


def finding(text: str = "sq") -> Finding:
    return Finding(sub_question=text, summary="found something", evidence=[EVIDENCE])


def make_ctx(tmp_path, script: StubScript, max_revisions: int = 2) -> RunContext:
    meter = TokenMeter()
    llm = LLMClient(
        meter,
        LLMConfig(),
        model_factory=stub_factory(script),
        providers=["anthropic"],
    )
    return RunContext(
        llm=llm,
        meter=meter,
        run_dir=tmp_path,
        run_id="test",
        max_revisions=max_revisions,
    )


# -- pure routing functions --------------------------------------------


def test_supervisor_routes_to_researcher_when_sub_questions_exist():
    state = {
        "routing": RoutingDecision(next_agents=["researcher"], rationale="r"),
        "pending_sub_questions": [SubQuestion(id="a", text="t")],
    }
    assert route_from_supervisor(state) == "researcher"


def test_supervisor_routes_to_synthesizer_without_sub_questions():
    state = {
        "routing": RoutingDecision(next_agents=["researcher"], rationale="r"),
        "pending_sub_questions": [],
    }
    assert route_from_supervisor(state) == "synthesizer"


def test_critic_failure_loops_back():
    state = {
        "critique": Critique(passed=False, revision_directives=["fix claim 2"]),
        "revision_count": 0,
    }
    assert route_from_critic(state, max_revisions=2) == "supervisor"


def test_critic_pass_finalizes():
    state = {"critique": Critique(passed=True), "revision_count": 1}
    assert route_from_critic(state, max_revisions=2) == "finalizer"


def test_exceeding_max_revisions_finalizes_even_on_failure():
    state = {
        "critique": Critique(passed=False, revision_directives=["still bad"]),
        "revision_count": 2,
    }
    assert route_from_critic(state, max_revisions=2) == "finalizer"


def test_budget_halt_short_circuits_to_finalizer():
    state = {
        "critique": Critique(passed=False, revision_directives=["bad"]),
        "revision_count": 0,
        "halted_reason": "Halted at budget ceiling: spent $0.51 of $0.50",
    }
    assert route_from_critic(state, max_revisions=2) == "finalizer"
    assert route_after_revise(state) == "finalizer"


# -- the graph, end to end ---------------------------------------------


def script_for(critiques: list[Critique], *, findings: int = 1) -> StubScript:
    return StubScript(
        responses={
            RoutingDecision: RoutingDecision(
                next_agents=["researcher"],
                rationale="need evidence",
                sub_questions=[
                    SubQuestion(id=f"sq{i}", text=f"sub-question {i}")
                    for i in range(findings)
                ],
            ),
            Finding: finding(),
            Brief: brief(2),
            Critique: critiques,
        },
        tool_responses=[StubResponse("no tools needed")],
    )


@pytest.mark.asyncio
async def test_happy_path_finalizes_without_revision(tmp_path):
    ctx = make_ctx(tmp_path, script_for([Critique(passed=True, scores={"grounding": 0.9})]))
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    assert state["revision_count"] == 0
    assert state["critique"].passed
    assert (tmp_path / "brief.json").exists()


@pytest.mark.asyncio
async def test_critic_failure_increments_revision_and_then_passes(tmp_path):
    critiques = [
        Critique(passed=False, scores={"grounding": 0.4}, revision_directives=["fix claim 2"]),
        Critique(passed=True, scores={"grounding": 0.9}),
    ]
    ctx = make_ctx(tmp_path, script_for(critiques))
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    assert state["revision_count"] == 1
    assert state["critique"].passed
    # The loop really went back round: the supervisor ran twice.
    assert sum(1 for s in state["trace"] if s.event == "routed") == 2
    # And the revision pass is attributed as such in the token stream.
    assert any(e.revision_index == 1 for e in ctx.meter.events)
    assert any(e.cause == "revision" for e in ctx.meter.events)


@pytest.mark.asyncio
async def test_cycle_is_bounded_and_degrades_honestly(tmp_path):
    always_fail = Critique(
        passed=False, scores={"grounding": 0.2}, revision_directives=["never good enough"]
    )
    ctx = make_ctx(tmp_path, script_for([always_fail]), max_revisions=2)
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    assert state["revision_count"] == 2
    assert not state["critique"].passed
    limitations = state["draft"].limitations
    assert any("Failed critic after 2 revision" in lim for lim in limitations)
    assert any("passed that critic, not verified true" in lim for lim in limitations)


@pytest.mark.asyncio
async def test_parallel_findings_accumulate_through_the_reducer(tmp_path):
    ctx = make_ctx(tmp_path, script_for([Critique(passed=True)], findings=3))
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    # Three researchers fanned out concurrently and none clobbered the others.
    assert len(state["findings"]) == 3


@pytest.mark.asyncio
async def test_critic_pass_over_failing_scores_is_overridden(tmp_path):
    # A cheerful passed=true sitting on top of failing numbers must not pass.
    lenient = Critique(passed=True, scores={"grounding": 0.3}, revision_directives=[])
    strict = Critique(passed=True, scores={"grounding": 0.95})
    ctx = make_ctx(tmp_path, script_for([lenient, strict]))
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    assert state["revision_count"] == 1


@pytest.mark.asyncio
async def test_every_llm_call_is_attributed(tmp_path):
    ctx = make_ctx(tmp_path, script_for([Critique(passed=True)]))
    graph = build_graph(ctx)

    await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    assert ctx.meter.events
    for event in ctx.meter.events:
        assert event.node, "unattributed node"
        assert event.cause, "unattributed cause"


@pytest.mark.asyncio
async def test_unevidenced_source_is_dropped_by_the_finalizer(tmp_path):
    hallucinated = Brief(
        question="q",
        claims=[
            Claim(statement="grounded", evidence=[EVIDENCE], confidence="high"),
            Claim(
                statement="invented",
                evidence=[Evidence(source="nowhere.md", quote="x", relevance=0.9)],
                confidence="high",
            ),
        ],
    )
    script = script_for([Critique(passed=True)])
    script.responses[Brief] = hallucinated
    ctx = make_ctx(tmp_path, script)
    graph = build_graph(ctx)

    state = await graph.ainvoke(initial_state("q", "test"), {"recursion_limit": 50})

    statements = [c.statement for c in state["draft"].claims]
    assert statements == ["grounded"]
    assert "invented" in state["draft"].open_questions
    assert any("nowhere.md" in lim for lim in state["draft"].limitations)
