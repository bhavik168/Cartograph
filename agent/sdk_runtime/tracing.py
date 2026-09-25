"""Per-run export of the Agents SDK's own traces.

The SDK's default processor uploads traces to OpenAI. Cartograph keeps
everything local, so for the duration of a run the default processors are
replaced with this one, which appends every finished trace and span to
``runs/<id>/agents_trace.jsonl`` next to ``tokens.jsonl``.

This file is the SDK's view of the run: agent turns, tool and handoff spans,
guardrail results. It is not an accounting surface. Token numbers come only
from ``tokens.jsonl``, which ``LLMClient`` writes the same way under both
runtimes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from agents.tracing import Span, Trace, TracingProcessor


class JsonlTraceProcessor(TracingProcessor):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def _write(self, kind: str, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        line = json.dumps({"kind": kind, **payload}, default=str)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        self._write("trace", trace.export())

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self._write("span", span.export())

    def force_flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def shutdown(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
