"""The research pipeline on the OpenAI Agents SDK.

    supervisor ──(research_sub_question × N, concurrent)──► researcher agent-as-tool
        │                                           └─ tools: MCP server (quarantined)
        └─handoff─► synthesizer ─handoff(Brief)─► critic ─► Critique (+ guardrail)

One ``Runner.run`` is one pass. Researchers are agents-as-tools, so the SDK runs
the fan-out. The critic is a handoff whose payload is the draft ``Brief``, and
it is the terminal agent of the pass, so the output guardrail sits on it.

The revision cycle stays in plain Python around ``Runner.run`` and uses the
same predicates the LangGraph graph uses (``route_from_critic``,
``make_revise``, ``route_after_revise``). A handoff back from the critic would
put the loop bound in the model's hands. Here the bound is ordinary code, as it
is in the graph.

The run ends with the same finalizer node, so ``brief.json`` gets the same
deterministic grounding pass and the same honest ``limitations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import (
    Agent,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    RunContextWrapper,
    RunResult,
    Runner,
    handoff,
    set_trace_processors,
)
from agents.agent_output import AgentOutputSchema

from agent.graph import make_revise, route_after_revise, route_from_critic
from agent.nodes.critic import SYSTEM as CRITIC_SYSTEM
from agent.nodes.finalizer import make_finalizer
from agent.nodes.researcher import SYSTEM as RESEARCH_SYSTEM
from agent.nodes.supervisor import SYSTEM as SUPERVISOR_SYSTEM
from agent.nodes.synthesizer import SYSTEM as SYNTHESIZER_SYSTEM
from agent.runtime import RunContext
from agent.schemas import Brief, Critique, Finding
from agent.sdk_runtime.errors import GuardrailRejected
from agent.sdk_runtime.guardrails import evidence_guardrail
from agent.sdk_runtime.mcp_bridge import QuarantinedMCPServer
from agent.sdk_runtime.model import (
    RESEARCH_TOOL_NAME,
    CriticModel,
    ResearcherModel,
    SDKRun,
    SupervisorModel,
    SynthesizerModel,
)
from agent.sdk_runtime.tracing import JsonlTraceProcessor
from agent.state import initial_state
from agent.toolsurface import InProcessTools

TRACE_FILE = "agents_trace.jsonl"
MAX_PASS_TURNS = 12


@dataclass
class Pipeline:
    supervisor: Agent[SDKRun]
    researcher: Agent[SDKRun]
    synthesizer: Agent[SDKRun]
    critic: Agent[SDKRun]


def _schema(model: type) -> AgentOutputSchema:
    # Non-strict: ``Critique.scores`` is a dict, which strict mode cannot
    # express. Validation still happens twice, once in LLMClient and once here.
    return AgentOutputSchema(model, strict_json_schema=False)


def build_pipeline(run: SDKRun, tools: QuarantinedMCPServer) -> Pipeline:
    ctx = run.ctx

    critic = Agent[SDKRun](
        name="critic",
        instructions=CRITIC_SYSTEM,
        model=CriticModel(run),
        output_type=_schema(Critique),
        output_guardrails=[evidence_guardrail],
    )

    async def on_draft(wrapper: RunContextWrapper[SDKRun], draft: Brief) -> None:
        wrapper.context.merge(
            {
                "trace": [
                    ctx.span(
                        "synthesizer",
                        "handoff_to_critic",
                        claims=len(draft.claims),
                        runtime="agents-sdk",
                    )
                ]
            }
        )

    synthesizer = Agent[SDKRun](
        name="synthesizer",
        instructions=SYNTHESIZER_SYSTEM,
        model=SynthesizerModel(run),
        output_type=_schema(Brief),
        handoffs=[handoff(critic, input_type=Brief, on_handoff=on_draft)],
        output_guardrails=[evidence_guardrail],
    )

    researcher = Agent[SDKRun](
        name="researcher",
        instructions=RESEARCH_SYSTEM,
        model=ResearcherModel(run),
        mcp_servers=[tools],
        output_type=_schema(Finding),
    )

    async def summary_only(result: RunResult) -> str:
        # The full Finding is already in the run state. The supervisor only
        # needs to know the call finished.
        finding = result.final_output
        return finding.summary if isinstance(finding, Finding) else str(finding)

    def research_failed(wrapper: RunContextWrapper[Any], error: Exception) -> str:
        # Same policy as the LangGraph researcher: a failed sub-question is
        # recorded as a gap, and the run carries on.
        message = f"Research failed for this sub-question: {error}"
        run.merge(
            {
                "findings": [
                    Finding(sub_question="(failed)", summary=message, revision_index=run.revision)
                ],
                "trace": [
                    ctx.span(
                        "researcher",
                        "failed",
                        error=f"{type(error).__name__}: {error}",
                        runtime="agents-sdk",
                    )
                ],
            }
        )
        return message

    supervisor = Agent[SDKRun](
        name="supervisor",
        instructions=SUPERVISOR_SYSTEM,
        model=SupervisorModel(run),
        tools=[
            researcher.as_tool(
                tool_name=RESEARCH_TOOL_NAME,
                tool_description="Research one narrow, self-contained sub-question.",
                custom_output_extractor=summary_only,
                failure_error_function=research_failed,
                max_turns=ctx.max_tool_iterations + 2,
            )
        ],
        handoffs=[synthesizer],
    )

    return Pipeline(
        supervisor=supervisor, researcher=researcher, synthesizer=synthesizer, critic=critic
    )


async def run_pipeline(ctx: RunContext, question: str) -> dict[str, Any]:
    """Run the full bounded pipeline and return the final ``AnalystState``-shaped dict.

    Raises :class:`GuardrailRejected` if the evidence guardrail trips; the run
    is failed rather than finalized.
    """
    run = SDKRun(ctx=ctx, state=dict(initial_state(question, ctx.run_id)))
    bridge = QuarantinedMCPServer(ctx.tools or InProcessTools(), on_result=run.record_tool_result)
    pipeline = build_pipeline(run, bridge)

    processor = JsonlTraceProcessor(ctx.run_dir / TRACE_FILE)
    set_trace_processors([processor])
    try:
        while True:
            config = RunConfig(
                workflow_name="cartograph",
                group_id=ctx.run_id,
                trace_metadata={"run_id": ctx.run_id, "revision": str(run.revision)},
            )
            try:
                await Runner.run(
                    pipeline.supervisor,
                    question,
                    context=run,
                    max_turns=MAX_PASS_TURNS,
                    run_config=config,
                )
            except OutputGuardrailTripwireTriggered as exc:
                info = exc.guardrail_result.output.output_info or {}
                run.merge(
                    {"trace": [ctx.span("critic", "guardrail_tripped", **dict(info))]}
                )
                raise GuardrailRejected(
                    "evidence guardrail rejected the draft: claim(s) with no evidence: "
                    + "; ".join(info.get("unevidenced_claims", []))
                ) from exc

            if route_from_critic(run.state, ctx.max_revisions) == "finalizer":
                break
            run.merge(await make_revise(ctx)(run.state))
            if route_after_revise(run.state) == "finalizer":
                break

        run.merge(await make_finalizer(ctx)(run.state))
    finally:
        processor.shutdown()
        set_trace_processors([])

    return run.state
