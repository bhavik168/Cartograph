"""The Agents SDK's ``Model`` interface, served by ``LLMClient``.

Every model call the SDK makes lands in one of the classes below, and every
one of those calls goes through ``LLMClient.call`` or ``call_with_tools`` with
an explicit ``node`` and ``cause``. The SDK never talks to a provider directly,
so ``tokens.jsonl`` gets the same ``TokenEvent`` stream under both runtimes and
the auditor needs no changes. An unattributed call is still a bug.

The adapter bridges two ideas of what a model turn is:

* The SDK expects control flow as model output: a tool call, a handoff, or a
  final message.
* Cartograph's contract is one Pydantic-validated object per call, with a
  repair pass when validation fails.

So the planning, drafting and judging turns make one structured call, reusing
the LangGraph node functions for identical prompts, causes and composition
accounting. The adapter then turns the validated result into the SDK items
that carry it: research tool calls from a ``RoutingDecision``, a handoff that
carries a ``Brief``, a final message holding a ``Critique``.

The researcher is the exception. It runs a real tool-calling loop through
``call_with_tools``, the model decides which MCP tools to call, and the SDK
executes them.

Turns that only move control along, such as the supervisor handing off once
its researchers have returned, make no model call and emit no ``TokenEvent``.
They cost nothing, so there is nothing to attribute.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agents import Handoff, ModelResponse, Usage
from agents.models.interface import Model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agent.auditor.meter import estimate_tokens
from agent.graph import route_from_supervisor
from agent.nodes.critic import make_critic
from agent.nodes.researcher import EXTRACT_SYSTEM
from agent.nodes.researcher import SYSTEM as RESEARCH_SYSTEM
from agent.nodes.supervisor import make_supervisor
from agent.nodes.synthesizer import make_synthesizer
from agent.runtime import RunContext
from agent.schemas import Finding, InputComposition

# State keys LangGraph merges with ``operator.add``. Everything else is
# last-write-wins, exactly as in ``AnalystState``.
_ADDITIVE_KEYS = ("findings", "trace", "flags")

RESEARCH_TOOL_NAME = "research_sub_question"


@dataclass
class SDKRun:
    """Per-run state for the Agents SDK runtime, passed to the SDK as its context.

    ``state`` is an ``AnalystState`` dict. Sharing that shape lets this runtime
    reuse the LangGraph node functions, the routing predicates and the
    finalizer unchanged, which is what keeps the two runtimes' ``Brief`` and
    audit trail comparable.
    """

    ctx: RunContext
    state: dict[str, Any]

    def merge(self, update: dict[str, Any]) -> None:
        for key, value in update.items():
            if key in _ADDITIVE_KEYS:
                self.state[key] = list(self.state.get(key) or []) + list(value)
            else:
                self.state[key] = value

    @property
    def revision(self) -> int:
        return self.state.get("revision_count", 0)

    def record_tool_result(self, name: str, result: Any) -> None:
        """Called by the MCP bridge with each quarantined result."""
        update: dict[str, Any] = {
            "trace": [
                self.ctx.span(
                    "researcher",
                    "tool_call",
                    tool=name,
                    injection_flags=result.flags,
                    truncated=result.truncated,
                    tokens_in=result.original_tokens,
                    tokens_kept=result.final_tokens,
                    runtime="agents-sdk",
                )
            ]
        }
        if result.is_suspicious:
            update["flags"] = [f"{name}:{flag}" for flag in result.flags]
        self.merge(update)


# -- SDK item plumbing --------------------------------------------------------


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(_get(part, "text", "") or "") for part in content)
    return str(content)


def _items(input: str | list[Any]) -> list[Any]:
    return [{"role": "user", "content": input}] if isinstance(input, str) else list(input)


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _function_call(name: str, arguments: dict[str, Any] | str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{uuid.uuid4().hex}",
        call_id=f"call_{uuid.uuid4().hex}",
        type="function_call",
        name=name,
        arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        status="completed",
    )


def _handoff_call(
    handoffs: Sequence[Handoff], agent_name: str, arguments: str = "{}"
) -> ResponseFunctionToolCall | None:
    for h in handoffs:
        if h.agent_name == agent_name:
            return _function_call(h.tool_name, arguments)
    return None


# -- the adapter --------------------------------------------------------------


class _LLMClientModel(Model):
    """Shared plumbing. Subclasses implement one role's ``turn``."""

    def __init__(self, run: SDKRun) -> None:
        self.run = run

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Handoff],
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
        **_: Any,
    ) -> ModelResponse:
        output = await self.turn(_items(input), tools, handoffs)
        # Usage is left empty on purpose. tokens.jsonl is the only accounting
        # surface; a second tally here would drift from it under concurrency.
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Cartograph runs the Agents SDK non-streaming only")

    async def turn(
        self, items: list[Any], tools: list[Any], handoffs: list[Handoff]
    ) -> list[Any]:
        raise NotImplementedError


class SupervisorModel(_LLMClientModel):
    """Plans with a ``RoutingDecision``, fans out to researchers, then hands off."""

    async def turn(self, items, tools, handoffs):
        dispatched = any(_get(i, "type") == "function_call_output" for i in items)
        if not dispatched:
            self.run.merge(await make_supervisor(self.run.ctx)(self.run.state))
            if route_from_supervisor(self.run.state) == "researcher":
                # One tool call per sub-question in a single response. The SDK
                # runs them concurrently: this runtime's fan-out.
                return [
                    _function_call(RESEARCH_TOOL_NAME, {"input": sq.text})
                    for sq in self.run.state["pending_sub_questions"]
                ]
        self.run.state["pending_sub_questions"] = []
        call = _handoff_call(handoffs, "synthesizer")
        return [call] if call else [_message("No synthesizer handoff is configured.")]


class SynthesizerModel(_LLMClientModel):
    """Drafts the ``Brief``, then hands it to the critic as the handoff payload."""

    async def turn(self, items, tools, handoffs):
        self.run.merge(await make_synthesizer(self.run.ctx)(self.run.state))
        draft = self.run.state["draft"].model_dump_json()
        call = _handoff_call(handoffs, "critic", draft)
        return [call] if call else [_message(draft)]


class CriticModel(_LLMClientModel):
    """Scores the draft. Its ``Critique`` is the pass's final output."""

    async def turn(self, items, tools, handoffs):
        self.run.merge(await make_critic(self.run.ctx)(self.run.state))
        return [_message(self.run.state["critique"].model_dump_json())]


class ResearcherModel(_LLMClientModel):
    """A real tool loop: the model picks MCP tools, the SDK runs them.

    Mirrors ``agent.nodes.researcher._run_one``: the same prompts, the same
    iteration bound, and a separate structured extraction once the model stops
    asking for tools.
    """

    async def turn(self, items, tools, handoffs):
        ctx = self.run.ctx
        revision = self.run.revision
        cause = "revision" if revision > 0 else "research"
        started = time.perf_counter()

        sub_question = next(
            (_text(_get(i, "content")) for i in items if _get(i, "role") == "user"), ""
        )
        task = f"Sub-question: {sub_question}"
        directives = self.run.state.get("revision_directives") or []
        if directives:
            task += "\n\nThe critic specifically asked for:\n" + "\n".join(
                f"- {d}" for d in directives
            )

        messages, tools_used, tool_output_tokens, turns = self._replay(task, items)

        if turns < ctx.max_tool_iterations:
            response = await ctx.llm.call_with_tools(
                messages,
                [self._spec(t) for t in tools],
                node="researcher",
                cause=cause,
                tier="cheap",
                revision_index=revision,
                composition=InputComposition(
                    system=estimate_tokens(RESEARCH_SYSTEM),
                    tool_output=tool_output_tokens,
                    other=estimate_tokens(task),
                ),
            )
            calls = getattr(response, "tool_calls", None) or []
            if calls:
                return [_function_call(c.get("name", ""), c.get("args") or {}) for c in calls]
            messages.append(response)
        else:
            messages.append(
                HumanMessage(content="Tool budget exhausted. Report what you found so far.")
            )

        transcript = "\n\n".join(
            f"[{type(m).__name__}] {getattr(m, 'content', '')}" for m in messages[1:]
        )
        finding: Finding = await ctx.llm.call(
            Finding,
            [
                ("system", EXTRACT_SYSTEM),
                ("user", f"Sub-question: {sub_question}\n\nTranscript:\n{transcript}"),
            ],
            node="researcher",
            cause=cause,
            tier="cheap",
            revision_index=revision,
            composition=InputComposition(
                system=estimate_tokens(EXTRACT_SYSTEM),
                tool_output=tool_output_tokens,
                other=estimate_tokens(transcript) - tool_output_tokens,
            ),
        )
        finding.sub_question = sub_question
        finding.revision_index = revision
        finding.tools_used = sorted(set(tools_used))
        self.run.merge(
            {
                "findings": [finding],
                "trace": [
                    ctx.span(
                        "researcher",
                        "finding",
                        started,
                        sub_question=sub_question[:120],
                        evidence_count=len(finding.evidence),
                        tools_used=finding.tools_used,
                        runtime="agents-sdk",
                    )
                ],
            }
        )
        return [_message(finding.model_dump_json())]

    @staticmethod
    def _spec(tool: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "parameters": getattr(tool, "params_json_schema", None)
                or {"type": "object", "properties": {}},
            },
        }

    @staticmethod
    def _replay(task: str, items: list[Any]) -> tuple[list[Any], list[str], int, int]:
        """Rebuild the LangChain transcript from the SDK's input items.

        Consecutive ``function_call`` items become one ``AIMessage``, which is
        one completed tool turn. Their outputs are already quarantined: they
        came back through the MCP bridge.
        """
        messages: list[Any] = [SystemMessage(content=RESEARCH_SYSTEM), HumanMessage(content=task)]
        tools_used: list[str] = []
        tool_output_tokens = 0
        turns = 0
        pending: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal turns
            if pending:
                messages.append(AIMessage(content="", tool_calls=list(pending)))
                pending.clear()
                turns += 1

        for item in items:
            kind = _get(item, "type")
            if kind == "function_call":
                name = _get(item, "name", "")
                try:
                    args = json.loads(_get(item, "arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                pending.append({"name": name, "args": args, "id": _get(item, "call_id")})
                tools_used.append(name)
            elif kind == "function_call_output":
                flush()
                output = _text(_get(item, "output"))
                tool_output_tokens += estimate_tokens(output)
                messages.append(ToolMessage(content=output, tool_call_id=_get(item, "call_id")))
        flush()
        return messages, tools_used, tool_output_tokens, turns
