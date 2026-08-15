"""Per-run context shared by every node.

LangGraph nodes are plain callables of ``(state) -> partial state``. Anything
that is not conversation state — the LLM client, the meter, the run directory —
is closed over via this object by the ``make_*`` factories in ``agent.nodes``,
which also makes every node trivially testable with a stub client.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.auditor.meter import TokenMeter
from agent.llm import LLMClient
from agent.schemas import Span

MAX_TOOL_ITERATIONS = 4
MAX_RESEARCHERS = 3


@dataclass
class RunContext:
    llm: LLMClient
    meter: TokenMeter
    run_dir: Path
    run_id: str
    max_revisions: int = 2
    max_tool_iterations: int = MAX_TOOL_ITERATIONS
    max_researchers: int = MAX_RESEARCHERS
    corpus_dir: str = "corpus"
    started_at: float = field(default_factory=time.time)
    _trace_fh: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def span(self, node: str, event: str, started: float | None = None, **detail) -> Span:
        """Build a span, stream it to trace.jsonl, and return it for the state."""
        span = Span(
            node=node,
            event=event,
            duration_ms=int((time.perf_counter() - started) * 1000) if started else 0,
            detail=detail,
        )
        self._write_trace(span)
        return span

    def _write_trace(self, span: Span) -> None:
        if self._trace_fh is None:
            self._trace_fh = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")
        self._trace_fh.write(span.model_dump_json() + "\n")
        self._trace_fh.flush()

    def write_json(self, name: str, payload: object) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def close(self) -> None:
        if self._trace_fh is not None:
            self._trace_fh.close()
            self._trace_fh = None
