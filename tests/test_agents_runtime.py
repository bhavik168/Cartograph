"""The Agents SDK runtime, driven by the same stub as the LangGraph runtime.

Each test builds its ``RunContext`` with the injected ``model_factory`` from
``tests.stubs``, so the SDK's model calls land in ``LLMClient`` and then in the
stub. No API key, no network, no cost.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agents")

from agents import Agent, RunContextWrapper  # noqa: E402

from agent import guards  # noqa: E402
from agent.auditor.attribute import build_audit  # noqa: E402
from agent.graph import build_graph  # noqa: E402
from agent.schemas import Brief, Claim, Critique, Evidence  # noqa: E402
from agent.sdk_runtime.guardrails import claims_without_evidence, evidence_guardrail  # noqa: E402
from agent.sdk_runtime.mcp_bridge import QuarantinedMCPServer  # noqa: E402
from agent.sdk_runtime.model import SDKRun  # noqa: E402
from agent.sdk_runtime.pipeline import TRACE_FILE, build_pipeline, run_pipeline  # noqa: E402
from agent.state import initial_state  # noqa: E402
from agent.toolsurface import InProcessTools, MCPTools  # noqa: E402
from tests.stubs import StubResponse  # noqa: E402
from tests.test_graph_routing import EVIDENCE, make_ctx, script_for  # noqa: E402

PASS = Critique(passed=True, scores={"grounding": 0.9})
CALC_6x7 = {"name": "calculator", "args": {"expression": "6*7"}, "id": "c1"}


async def run_langgraph(tmp_path, critiques, **kw):
    ctx = make_ctx(tmp_path / "langgraph", script_for(critiques, **kw))
    state = await build_graph(ctx).ainvoke(initial_state("q", "test"), {"recursion_limit": 50})
    return ctx, state


async def run_agents_sdk(tmp_path, critiques, **kw):
    ctx = make_ctx(tmp_path / "agents-sdk", script_for(critiques, **kw))
    state = await run_pipeline(ctx, "q")
    return ctx, state


def written_brief(ctx) -> Brief:
    return Brief.model_validate(json.loads((ctx.run_dir / "brief.json").read_text()))


# -- same inputs, same Brief --------------------------------------------------


@pytest.mark.asyncio
async def test_agents_sdk_brief_validates_like_langgraph(tmp_path):
    lg_ctx, _ = await run_langgraph(tmp_path, [PASS])
    sdk_ctx, sdk_state = await run_agents_sdk(tmp_path, [PASS])

    lg_brief, sdk_brief = written_brief(lg_ctx), written_brief(sdk_ctx)
    assert isinstance(sdk_state["draft"], Brief)
    assert sdk_brief == lg_brief
    assert sdk_state["revision_count"] == 0
    assert sdk_state["critique"].passed


@pytest.mark.asyncio
async def test_agents_sdk_fans_out_one_researcher_per_sub_question(tmp_path):
    _, state = await run_agents_sdk(tmp_path, [PASS], findings=3)
    assert len(state["findings"]) == 3


@pytest.mark.asyncio
async def test_agents_sdk_revision_loop_matches_langgraph(tmp_path):
    critiques = [
        Critique(passed=False, scores={"grounding": 0.4}, revision_directives=["fix claim 2"]),
        PASS,
    ]
    lg_ctx, lg_state = await run_langgraph(tmp_path, critiques)
    sdk_ctx, sdk_state = await run_agents_sdk(tmp_path, critiques)

    assert sdk_state["revision_count"] == lg_state["revision_count"] == 1
    assert any(e.cause == "revision" and e.revision_index == 1 for e in sdk_ctx.meter.events)


@pytest.mark.asyncio
async def test_agents_sdk_cycle_is_bounded(tmp_path):
    always_fail = Critique(
        passed=False, scores={"grounding": 0.2}, revision_directives=["never good enough"]
    )
    _, state = await run_agents_sdk(tmp_path, [always_fail])
    assert state["revision_count"] == 2
    assert any("Failed critic after 2 revision" in lim for lim in state["draft"].limitations)


@pytest.mark.asyncio
async def test_agents_sdk_exports_a_trace_next_to_tokens(tmp_path):
    ctx, _ = await run_agents_sdk(tmp_path, [PASS])
    lines = (ctx.run_dir / TRACE_FILE).read_text().splitlines()
    assert lines
    assert all(json.loads(line)["kind"] in {"trace", "span"} for line in lines)


# -- tools: MCP as the agent tool source, quarantined -------------------------


@pytest.mark.asyncio
async def test_agents_sdk_researcher_calls_mcp_tools_through_quarantine(tmp_path, monkeypatch):
    seen: list[str] = []
    real = guards.quarantine

    def spy(text, source, **kw):
        seen.append(source)
        return real(text, source, **kw)

    monkeypatch.setattr(guards, "quarantine", spy)
    script = script_for([PASS])
    script.tool_responses = [
        StubResponse(tool_calls=[CALC_6x7]),
        StubResponse("done"),
    ]
    ctx = make_ctx(tmp_path, script)
    async with MCPTools.in_memory() as tools:
        ctx.tools = tools
        state = await run_pipeline(ctx, "q")

    assert seen == ["calculator"]
    assert any(
        s.event == "tool_call" and s.detail.get("runtime") == "agents-sdk" for s in state["trace"]
    )
    assert state["findings"][0].tools_used == ["calculator"]


def test_researcher_agent_uses_the_mcp_bridge(tmp_path):
    ctx = make_ctx(tmp_path, script_for([PASS]))
    run = SDKRun(ctx=ctx, state=dict(initial_state("q", "test")))
    bridge = QuarantinedMCPServer(InProcessTools())
    pipeline = build_pipeline(run, bridge)

    assert pipeline.researcher.mcp_servers == [bridge]
    assert evidence_guardrail in pipeline.critic.output_guardrails
    assert [h.agent_name for h in pipeline.synthesizer.handoffs] == ["critic"]


# -- the output guardrail -----------------------------------------------------


def unevidenced_brief() -> Brief:
    # model_construct bypasses Claim's min_length=1, which is exactly the case
    # the guardrail exists for.
    bare = Claim.model_construct(statement="unsupported", evidence=[], confidence="high")
    good = Claim(statement="supported", evidence=[EVIDENCE], confidence="medium")
    return Brief.model_construct(
        question="q", claims=[good, bare], open_questions=[], limitations=[]
    )


def test_zero_evidence_claim_is_detected():
    assert claims_without_evidence(unevidenced_brief()) == ["unsupported"]


@pytest.mark.asyncio
async def test_output_guardrail_rejects_a_zero_evidence_claim():
    agent = Agent(name="synthesizer")
    wrapper = RunContextWrapper(context=None)
    result = await evidence_guardrail.run(wrapper, agent, unevidenced_brief())
    assert result.output.tripwire_triggered
    assert result.output.output_info["unevidenced_claims"] == ["unsupported"]


@pytest.mark.asyncio
async def test_output_guardrail_checks_the_handed_off_draft_on_the_critic(tmp_path):
    ctx = make_ctx(tmp_path, script_for([PASS]))
    run = SDKRun(ctx=ctx, state={"draft": unevidenced_brief()})
    wrapper = RunContextWrapper(context=run)
    result = await evidence_guardrail.run(wrapper, Agent(name="critic"), PASS)
    assert result.output.tripwire_triggered


@pytest.mark.asyncio
async def test_output_guardrail_passes_an_evidenced_brief():
    evidence = [Evidence(source="a.md", quote="x", relevance=1)]
    brief = Brief(question="q", claims=[Claim(statement="ok", evidence=evidence, confidence="low")])
    wrapper = RunContextWrapper(context=None)
    result = await evidence_guardrail.run(wrapper, Agent(name="s"), brief)
    assert not result.output.tripwire_triggered


# -- one audit trail across both runtimes -------------------------------------


@pytest.mark.asyncio
async def test_token_events_from_both_runtimes_aggregate_in_the_auditor(tmp_path):
    critiques = [
        Critique(passed=False, scores={"grounding": 0.4}, revision_directives=["fix it"]),
        PASS,
    ]
    lg_ctx, _ = await run_langgraph(tmp_path, critiques)
    sdk_ctx, _ = await run_agents_sdk(tmp_path, critiques)
    lg, sdk = lg_ctx.meter.events, sdk_ctx.meter.events

    # Every call is attributed, under both runtimes.
    for event in lg + sdk:
        assert event.node and event.cause

    # Same canned inputs, same calls: the attribution matches event for event.
    def signature(events):
        return sorted((e.node, e.cause, e.revision_index, e.tier) for e in events)

    assert signature(sdk) == signature(lg)

    lg_audit, sdk_audit, combined = build_audit(lg), build_audit(sdk), build_audit(lg + sdk)
    assert combined.llm_calls == lg_audit.llm_calls + sdk_audit.llm_calls
    assert combined.total_tokens == lg_audit.total_tokens + sdk_audit.total_tokens
    assert combined.waste_tokens == lg_audit.waste_tokens + sdk_audit.waste_tokens
    assert sdk_audit.waste_ratio == pytest.approx(lg_audit.waste_ratio)

    by_cause = {b.key: b.total_tokens for b in combined.by_cause}
    assert by_cause["revision"] == sum(e.total_tokens for e in lg + sdk if e.cause == "revision")
