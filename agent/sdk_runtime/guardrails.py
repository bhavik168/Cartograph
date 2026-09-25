"""Output guardrail: no claim ships without evidence.

The LangGraph runtime enforces grounding deterministically. ``Claim.evidence``
has ``min_length=1``, and the finalizer drops claims that cite a source no
researcher retrieved. This guardrail is the Agents SDK counterpart. It is plain
code, not a model call, so it adds no tokens and cannot be argued with.

``Claim``'s own constraint already rejects an empty evidence list for any
payload that is validated. The guardrail exists for the ones that aren't: an
object built with ``model_construct``, or a schema someone later loosens. It
checks the invariant directly rather than trusting that the schema still
encodes it.

It is attached to the critic, because the critic is the terminal agent of a
pass and the SDK only runs output guardrails on the agent that produces the
final output. The critic's output is a ``Critique``, so the guardrail reads the
draft ``Brief`` the synthesizer handed off, which is held in the run context.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, output_guardrail

from agent.schemas import Brief


def claims_without_evidence(brief: Brief | None) -> list[str]:
    """Statements of every claim whose evidence list is empty."""
    if brief is None:
        return []
    return [claim.statement for claim in brief.claims if not claim.evidence]


def _brief_under_review(context: Any, output: Any) -> Brief | None:
    if isinstance(output, Brief):
        return output
    state = getattr(context, "state", None) or {}
    draft = state.get("draft")
    return draft if isinstance(draft, Brief) else None


@output_guardrail(name="claims_have_evidence")
async def evidence_guardrail(
    ctx: RunContextWrapper[Any], agent: Agent[Any], output: Any
) -> GuardrailFunctionOutput:
    unevidenced = claims_without_evidence(_brief_under_review(ctx.context, output))
    return GuardrailFunctionOutput(
        output_info={"unevidenced_claims": unevidenced},
        tripwire_triggered=bool(unevidenced),
    )
