"""Attribution: turning a token stream into an answer to "where did it go?"

A total token count is a number, not an insight. What makes the number
actionable is knowing which tokens bought new information and which were spent
re-doing work — schema repairs, transient retries, and critic-driven revisions.

Every cause belongs to exactly one class:

``productive``  tokens that advanced the run
``overhead``    tokens spent on necessary machinery (compaction, failover)
``waste``       tokens that produced no new information

The **waste ratio** is ``waste_tokens / total_tokens``. It is deterministic,
free to compute, and directly actionable: tighten the schema prompt and the
repair count drops, sharpen the critic and the revision spend drops.

One subtlety worth stating: a call is classified by its recorded ``cause``, and
``agent.llm`` stamps ``revision`` on any call issued during a revision pass. So
revision tokens are counted once, under ``revision``, not double-counted under
the node's first-pass cause.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from agent.auditor.pricing import cost_usd, is_priced
from agent.schemas import CAUSES, TokenEvent

CauseClass = Literal["productive", "overhead", "waste"]

CAUSE_CLASS: dict[str, CauseClass] = {
    "planning": "productive",
    "research": "productive",
    "synthesis": "productive",
    "critique": "productive",
    "finalization": "productive",
    "memory_compaction": "overhead",
    "fallback": "overhead",
    "injection_rescan": "overhead",
    "schema_repair": "waste",
    "revision": "waste",
    "retry_transient": "waste",
}

# Sanity: the taxonomy and the classification table must not drift apart.
assert set(CAUSE_CLASS) == set(CAUSES), "every cause must be classified"


def classify(cause: str) -> CauseClass:
    return CAUSE_CLASS.get(cause, "overhead")


def _pct(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else 0.0


class Bucket(BaseModel):
    key: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    pct_of_total: float = 0.0
    cause_class: CauseClass | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PassCost(BaseModel):
    revision_index: int
    calls: int = 0
    tokens: int = 0
    usd: float = 0.0


class Audit(BaseModel):
    """The full machine-readable audit, serialised to ``audit.json``."""

    run_id: str = ""
    question: str = ""
    outcome: str = ""
    wall_clock_s: float = 0.0

    llm_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0

    schema_repairs: int = 0
    transient_retries: int = 0
    revisions: int = 0

    by_node: list[Bucket] = Field(default_factory=list)
    by_cause: list[Bucket] = Field(default_factory=list)
    by_provider: list[Bucket] = Field(default_factory=list)

    productive_tokens: int = 0
    overhead_tokens: int = 0
    waste_tokens: int = 0
    waste_ratio: float = 0.0

    composition: dict[str, int] = Field(default_factory=dict)
    composition_pct: dict[str, float] = Field(default_factory=dict)

    pass_costs: list[PassCost] = Field(default_factory=list)

    cheap_call_share: float = 0.0
    cheap_spend_share: float = 0.0
    strong_spend_share: float = 0.0

    unpriced_models: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Rule-based recommendations
#
# Deliberately not LLM-generated: deterministic, free, unit-testable, and it
# does not add tokens to the very run it is auditing.
# --------------------------------------------------------------------------

SCHEMA_REPAIR_THRESHOLD = 2
WASTE_RATIO_THRESHOLD = 0.20
SCRATCHPAD_SHARE_THRESHOLD = 0.40
STRONG_SPEND_THRESHOLD = 0.70
TOOL_OUTPUT_SHARE_THRESHOLD = 0.50


def recommend(audit: Audit) -> list[str]:
    out: list[str] = []
    if audit.schema_repairs > SCHEMA_REPAIR_THRESHOLD:
        out.append(
            f"{audit.schema_repairs} schema repairs: tighten the schema prompt or "
            "add a few-shot example of a valid response."
        )
    if audit.waste_ratio > WASTE_RATIO_THRESHOLD:
        out.append(
            f"Waste ratio {audit.waste_ratio:.0%}: the revision loop is dominating "
            "spend. Sharpen the critic's criteria so failures are rarer and more "
            "specific."
        )
    scratchpad_share = audit.composition_pct.get("scratchpad", 0.0) / 100.0
    if scratchpad_share > SCRATCHPAD_SHARE_THRESHOLD:
        out.append(
            f"Scratchpad is {scratchpad_share:.0%} of input tokens: compact memory "
            "more aggressively or lower COMPACTION_THRESHOLD_TOKENS."
        )
    tool_share = audit.composition_pct.get("tool_output", 0.0) / 100.0
    if tool_share > TOOL_OUTPUT_SHARE_THRESHOLD:
        out.append(
            f"Tool output is {tool_share:.0%} of input tokens: lower "
            "guards.MAX_TOOL_OUTPUT_TOKENS or retrieve fewer passages per call."
        )
    if audit.strong_spend_share > STRONG_SPEND_THRESHOLD:
        out.append(
            f"Strong tier is {audit.strong_spend_share:.0%} of spend: move "
            "extraction or routing calls to the cheap tier."
        )
    if audit.transient_retries:
        out.append(
            f"{audit.transient_retries} transient retries: check rate limits or "
            "raise base_backoff_s in LLMConfig."
        )
    if audit.unpriced_models:
        out.append(
            "No price entry for "
            + ", ".join(audit.unpriced_models)
            + " — USD figures undercount. Add them to agent/auditor/pricing.py."
        )
    if not out:
        out.append("No thresholds crossed. Nothing to tune from this run.")
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def build_audit(
    events: list[TokenEvent],
    *,
    run_id: str = "",
    question: str = "",
    outcome: str = "",
    wall_clock_s: float = 0.0,
) -> Audit:
    """Aggregate a token stream. Safe on an empty stream — no division by zero."""
    audit = Audit(
        run_id=run_id,
        question=question,
        outcome=outcome,
        wall_clock_s=wall_clock_s,
        llm_calls=len(events),
    )
    if not events:
        audit.recommendations = ["No LLM calls were recorded for this run."]
        return audit

    def event_usd(e: TokenEvent) -> float:
        return cost_usd(e.model, e.input_tokens, e.output_tokens, e.cached_input_tokens)

    nodes: dict[str, Bucket] = defaultdict(lambda: Bucket(key=""))
    causes: dict[str, Bucket] = defaultdict(lambda: Bucket(key=""))
    providers: dict[str, Bucket] = defaultdict(lambda: Bucket(key=""))
    passes: dict[int, PassCost] = defaultdict(lambda: PassCost(revision_index=0))
    composition: dict[str, int] = defaultdict(int)
    unpriced: set[str] = set()

    cheap_calls = 0
    cheap_usd = 0.0
    strong_usd = 0.0

    for event in events:
        usd = event_usd(event)
        audit.input_tokens += event.input_tokens
        audit.output_tokens += event.output_tokens
        audit.cached_input_tokens += event.cached_input_tokens
        audit.est_cost_usd += usd
        if not event.ok:
            audit.failed_calls += 1
        if not is_priced(event.model):
            unpriced.add(event.model)

        for bucket, key in (
            (nodes, event.node),
            (causes, event.cause),
            (providers, event.provider),
        ):
            entry = bucket[key]
            entry.key = key
            entry.calls += 1
            entry.input_tokens += event.input_tokens
            entry.output_tokens += event.output_tokens
            entry.usd += usd

        pass_cost = passes[event.revision_index]
        pass_cost.revision_index = event.revision_index
        pass_cost.calls += 1
        pass_cost.tokens += event.total_tokens
        pass_cost.usd += usd

        for field_name, value in event.composition.model_dump().items():
            composition[field_name] += int(value)

        if event.tier == "cheap":
            cheap_calls += 1
            cheap_usd += usd
        else:
            strong_usd += usd

    audit.total_tokens = audit.input_tokens + audit.output_tokens

    for bucket in (*nodes.values(), *causes.values(), *providers.values()):
        bucket.pct_of_total = _pct(bucket.total_tokens, audit.total_tokens)
    for cause_key, bucket in causes.items():
        bucket.cause_class = classify(cause_key)

    audit.by_node = sorted(nodes.values(), key=lambda b: b.total_tokens, reverse=True)
    audit.by_cause = sorted(causes.values(), key=lambda b: b.total_tokens, reverse=True)
    audit.by_provider = sorted(
        providers.values(), key=lambda b: b.total_tokens, reverse=True
    )

    for bucket in audit.by_cause:
        if bucket.cause_class == "productive":
            audit.productive_tokens += bucket.total_tokens
        elif bucket.cause_class == "overhead":
            audit.overhead_tokens += bucket.total_tokens
        else:
            audit.waste_tokens += bucket.total_tokens
    audit.waste_ratio = (
        audit.waste_tokens / audit.total_tokens if audit.total_tokens else 0.0
    )

    audit.schema_repairs = sum(1 for e in events if e.cause == "schema_repair")
    audit.transient_retries = sum(1 for e in events if e.cause == "retry_transient")
    audit.revisions = max((e.revision_index for e in events), default=0)

    composed_total = sum(composition.values())
    audit.composition = dict(composition)
    audit.composition_pct = {
        key: _pct(value, composed_total) for key, value in composition.items()
    }

    audit.pass_costs = sorted(passes.values(), key=lambda p: p.revision_index)

    audit.cheap_call_share = cheap_calls / len(events)
    total_usd = cheap_usd + strong_usd
    audit.cheap_spend_share = cheap_usd / total_usd if total_usd else 0.0
    audit.strong_spend_share = strong_usd / total_usd if total_usd else 0.0

    audit.unpriced_models = sorted(unpriced)
    audit.recommendations = recommend(audit)
    return audit
