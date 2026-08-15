"""Every structured value in Cartographer lives here.

Nothing in this project parses model output by hand or with a regex. Each LLM
call is bound to one of these models via ``with_structured_output``; if the
model returns something that does not validate, ``agent.llm`` runs a single
schema-repair pass and the repair is billed to the ``schema_repair`` cause.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["cheap", "strong"]
Provider = Literal["anthropic", "openai"]

Cause = Literal[
    "planning",
    "research",
    "synthesis",
    "critique",
    "finalization",
    "memory_compaction",
    "schema_repair",
    "revision",
    "retry_transient",
    "fallback",
    "injection_rescan",
]

CAUSES: tuple[str, ...] = (
    "planning",
    "research",
    "synthesis",
    "critique",
    "finalization",
    "memory_compaction",
    "schema_repair",
    "revision",
    "retry_transient",
    "fallback",
    "injection_rescan",
)

NODES: tuple[str, ...] = (
    "supervisor",
    "researcher",
    "synthesizer",
    "critic",
    "finalizer",
    "memory",
)


# --------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single quoted support for a claim, traceable to its origin."""

    source: str = Field(description="Corpus filename or tool name the quote came from.")
    quote: str = Field(max_length=300, description="Verbatim supporting excerpt.")
    relevance: float = Field(ge=0, le=1)


class Claim(BaseModel):
    statement: str
    # A claim MUST be evidenced. This constraint is the single most common
    # trigger of the schema-repair path, and that is deliberate.
    evidence: list[Evidence] = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]


class Brief(BaseModel):
    question: str
    claims: list[Claim]
    open_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Inter-node messages
# --------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """One unit of work handed to a researcher."""

    id: str
    text: str


class RoutingDecision(BaseModel):
    next_agents: list[Literal["researcher", "synthesizer"]]
    rationale: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    stop: bool = False


class Finding(BaseModel):
    """What one researcher came back with for one sub-question."""

    sub_question: str
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    revision_index: int = 0


class Critique(BaseModel):
    passed: bool
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="e.g. grounding / coverage / specificity, each 0-1.",
    )
    revision_directives: list[str] = Field(default_factory=list)

    def is_consistent(self) -> bool:
        """A pass with directives, or a fail without them, is a malformed critique."""
        return self.passed == (len(self.revision_directives) == 0)


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------


class Span(BaseModel):
    """One node execution, appended to runs/<id>/trace.jsonl."""

    ts: float = Field(default_factory=time.time)
    node: str
    event: str
    duration_ms: int = 0
    detail: dict = Field(default_factory=dict)


class InputComposition(BaseModel):
    """Rough breakdown of what filled the context on a single call.

    Measured by tokenising each block before assembly, so the numbers are an
    estimate of composition, not a second opinion on the provider's count.
    """

    system: int = 0
    scratchpad: int = 0
    tool_output: int = 0
    findings: int = 0
    schema_instructions: int = 0
    other: int = 0

    def total(self) -> int:
        return (
            self.system
            + self.scratchpad
            + self.tool_output
            + self.findings
            + self.schema_instructions
            + self.other
        )


class TokenEvent(BaseModel):
    """One LLM call. Emitted by agent.llm, never by anything else."""

    ts: float = Field(default_factory=time.time)
    node: str
    cause: Cause
    model: str
    tier: Tier
    provider: Provider
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    revision_index: int = 0
    latency_ms: int = 0
    ok: bool = True
    composition: InputComposition = Field(default_factory=InputComposition)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
