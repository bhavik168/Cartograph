"""Every shape that crosses the MCP boundary: bounded inputs, outputs, errors.

Inputs forbid unknown keys and carry real bounds; the advertised JSON schema of
each tool is generated from these models, so a caller sees the same limits the
server enforces.

Outputs compose the graph's own models — ``Brief``, ``Finding``,
``RoutingDecision``, the auditor's ``Bucket`` — unmodified. Anything that
originated outside the process (corpus text, filenames, and model output derived
from them) is nested under a single ``untrusted_content`` key, with
``injection_flags`` as its sibling, so the shape itself carries the warning. The
text content block of every result is the same payload rendered through
``guards.quarantine``.

Errors are a closed set of codes. A caller branches on ``error.code``; it never
needs to parse ``error.message``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.auditor.attribute import Bucket
from agent.schemas import Brief, Finding, RoutingDecision
from agent.state import MAX_REVISIONS
from agent.tools.calculator import MAX_EXPRESSION_CHARS

# Run and thread ids become directory names and checkpoint keys. The pattern is
# the first gate; guards.resolve_within on the resolved run directory is the
# second, and does not rely on the first.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"

MAX_QUERY_CHARS = 500
MAX_QUESTION_CHARS = 2000
MAX_TOP_K = 10
DEFAULT_TOP_K = 4


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CorpusSearchArgs(_Args):
    query: str = Field(
        min_length=1, max_length=MAX_QUERY_CHARS, description="Keywords to search for."
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Passages to return."
    )


class CalculatorArgs(_Args):
    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_CHARS,
        description='Plain arithmetic, e.g. "(1240 - 890) / 890 * 100". '
        "+ - * / // % ** and parentheses only.",
    )


class StartBriefArgs(_Args):
    question: str = Field(
        min_length=1, max_length=MAX_QUESTION_CHARS, description="The research question."
    )
    thread_id: str | None = Field(
        default=None,
        pattern=ID_PATTERN,
        description="Checkpoint thread id. Defaults to the run id. Must be unused.",
    )
    max_revisions: int | None = Field(
        default=None,
        ge=0,
        le=MAX_REVISIONS,
        description=f"Critic-driven revision passes allowed (default {MAX_REVISIONS}).",
    )
    max_usd: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Requested USD ceiling. The server applies the lower of this "
        "and its own ceiling.",
    )


class RunIdArgs(_Args):
    run_id: str = Field(pattern=ID_PATTERN, description="Id returned by start_brief.")


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


class SearchHit(BaseModel):
    doc_id: str = Field(description="Path relative to the corpus root.")
    chunk: int = Field(description="Paragraph index within the document.")
    score: float
    text: str


class SearchContent(BaseModel):
    hits: list[SearchHit]


class CorpusSearchResult(BaseModel):
    backend: Literal["bm25", "overlap"]
    hit_count: int
    untrusted_content: SearchContent
    injection_flags: list[str]


class CalculatorResult(BaseModel):
    expression: str
    value: float


RunStatusName = Literal["planning", "running", "finalized", "failed"]

FinalizedReason = Literal["passed_critic", "max_revisions", "budget_ceiling", "no_critique"]


class RunStarted(BaseModel):
    run_id: str
    thread_id: str
    status: Literal["planning"] = "planning"
    max_revisions: int
    max_usd: float = Field(description="Effective ceiling after the server's clamp.")


class StatusContent(BaseModel):
    routing: RoutingDecision | None = None
    findings: list[Finding] = Field(default_factory=list)


class RunStatus(BaseModel):
    run_id: str
    status: RunStatusName
    current_node: str | None = Field(
        description="Node executing now; the last one that ran once the run has ended."
    )
    revision_count: int
    max_revisions: int
    llm_calls: int
    tokens_spent: int
    est_usd_spent: float
    max_usd: float
    finalized_reason: FinalizedReason | None = None
    failure: str | None = None
    untrusted_content: StatusContent
    injection_flags: list[str]


class BriefContent(BaseModel):
    brief: Brief


class BriefResult(BaseModel):
    run_id: str
    finalized_reason: FinalizedReason
    reason_detail: str | None = Field(
        description="Why the run stopped where it did, when it did not pass the critic."
    )
    revision_count: int
    max_revisions: int
    untrusted_content: BriefContent
    injection_flags: list[str]


class TokenAuditResult(BaseModel):
    run_id: str
    origin: str
    llm_calls: int
    failed_calls: int
    total_tokens: int
    productive_tokens: int
    overhead_tokens: int
    waste_tokens: int
    waste_ratio: float
    est_cost_usd: float
    by_cause: list[Bucket]
    by_node: list[Bucket]


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorCode(str, Enum):
    """The complete set. Documented in the README; stable across releases."""

    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    """Arguments failed the tool's input schema. ``details.fields`` names them."""
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    UNKNOWN_RUN = "UNKNOWN_RUN"
    """No run with this id was started by this server (or survives on disk)."""
    RUN_IN_PROGRESS = "RUN_IN_PROGRESS"
    """The run has not finalized yet. Poll ``get_run_status``. Retryable."""
    RUN_FAILED = "RUN_FAILED"
    """The run ended without a brief (e.g. every provider failed)."""
    CORPUS_NOT_INDEXED = "CORPUS_NOT_INDEXED"
    """The corpus root holds no indexable .md/.txt documents."""
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    """The server's process-wide USD ceiling is spent by runs that have ended."""
    BUDGET_RESERVED = "BUDGET_RESERVED"
    """The ceiling is fully reserved by runs in flight. Retryable once one finishes."""
    RUN_LIMIT_REACHED = "RUN_LIMIT_REACHED"
    """Too many runs in flight. Retryable once one finishes."""
    THREAD_EXISTS = "THREAD_EXISTS"
    """The requested thread_id already has checkpoints."""
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    """A requested id resolved outside the directory it must stay in."""
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    """No LLM provider key is configured on the server."""
    INTERNAL = "INTERNAL"
    """Unexpected server fault. Details are logged server-side, never returned."""


RETRYABLE = frozenset(
    {ErrorCode.RUN_IN_PROGRESS, ErrorCode.RUN_LIMIT_REACHED, ErrorCode.BUDGET_RESERVED}
)


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class CartographToolError(Exception):
    """Raised inside a tool body; the server turns it into an ``ErrorEnvelope``."""

    def __init__(self, code: ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def envelope(self) -> ErrorEnvelope:
        return ErrorEnvelope(
            error=ErrorBody(
                code=self.code,
                message=self.message,
                retryable=self.code in RETRYABLE,
                details=self.details,
            )
        )
