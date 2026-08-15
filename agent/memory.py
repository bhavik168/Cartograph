"""Persistence and the running scratchpad.

Two distinct kinds of memory:

*Checkpointing* — LangGraph's ``SqliteSaver`` writes the full state after every
super-step, keyed by thread id. A crashed or budget-halted run can be resumed
by re-invoking the graph with the same thread id.

*Scratchpad* — a short natural-language summary carried across revision cycles.
Without it, each loop through the critic re-injects the full findings list and
the context grows linearly with revisions. The compaction call is metered under
``memory_compaction`` (overhead, not waste: it is spend that buys back context),
and the pre/post token counts land in the trace so the effect is measurable
rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from agent.auditor.meter import estimate_tokens
from agent.schemas import Finding

DEFAULT_CHECKPOINT_DB = Path("runs/checkpoints.sqlite")

# Compact only once the raw findings are big enough that summarising them is
# cheaper than carrying them. Below this, compaction costs more than it saves.
COMPACTION_THRESHOLD_TOKENS = 1500


class ScratchpadSummary(BaseModel):
    """Structured output of the compaction call."""

    summary: str = Field(max_length=2000)
    unresolved: list[str] = Field(default_factory=list)


@contextmanager
def checkpointer(db_path: Path | str = DEFAULT_CHECKPOINT_DB) -> Iterator[object | None]:
    """Yield a LangGraph checkpointer, or ``None`` if the extra isn't installed.

    The graph compiles either way; without a saver you lose resumability, not
    correctness, so a missing optional dependency should not stop a run.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        yield None
        return
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver


def render_findings(findings: list[Finding], limit: int | None = None) -> str:
    """Findings as prompt text. ``limit`` keeps only the most recent N."""
    selected = findings[-limit:] if limit else findings
    blocks: list[str] = []
    for finding in selected:
        quotes = "\n".join(
            f'    - "{e.quote}" ({e.source}, relevance {e.relevance:.2f})'
            for e in finding.evidence
        )
        blocks.append(
            f"[sub-question] {finding.sub_question}\n"
            f"  summary: {finding.summary}\n"
            f"  evidence:\n{quotes or '    - (none)'}"
        )
    return "\n\n".join(blocks)


def should_compact(findings: list[Finding], scratchpad: str) -> bool:
    return (
        estimate_tokens(render_findings(findings) + scratchpad)
        > COMPACTION_THRESHOLD_TOKENS
    )


async def compact(
    llm,
    *,
    question: str,
    findings: list[Finding],
    scratchpad: str,
    revision_index: int = 0,
) -> tuple[str, int, int]:
    """Compress findings + prior scratchpad into a new scratchpad.

    Returns ``(new_scratchpad, tokens_before, tokens_after)``.
    """
    raw = render_findings(findings)
    before = estimate_tokens(raw + scratchpad)

    result: ScratchpadSummary = await llm.call(
        ScratchpadSummary,
        [
            (
                "system",
                "You compress research notes without losing anything load-bearing. "
                "Keep every source filename and every number. Drop restatements, "
                "hedging, and prose. Write for another model, not a human.",
            ),
            (
                "user",
                f"Research question: {question}\n\n"
                f"Existing summary:\n{scratchpad or '(none yet)'}\n\n"
                f"New findings:\n{raw}\n\n"
                "Produce a single consolidated summary and list what is still "
                "unresolved.",
            ),
        ],
        node="memory",
        cause="memory_compaction",
        tier="cheap",
        revision_index=revision_index,
    )

    new_scratchpad = result.summary
    if result.unresolved:
        new_scratchpad += "\n\nStill unresolved:\n" + "\n".join(
            f"- {item}" for item in result.unresolved
        )
    return new_scratchpad, before, estimate_tokens(new_scratchpad)
