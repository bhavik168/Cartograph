"""Token capture.

The meter is passed to ``agent.llm.LLMClient`` and is the only writer of
``runs/<id>/tokens.jsonl``. Events are flushed as they happen, so a run that
crashes halfway still leaves a usable partial audit on disk.

The meter also owns the budget ceiling: ``--max-usd`` is enforced by asking the
meter between graph nodes whether spend has crossed the line.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agent.auditor.pricing import cost_usd
from agent.schemas import TokenEvent


def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (~4 chars/token).

    Used only for context-composition accounting and length caps, never for
    billing: every billed number in an audit comes from the provider's own
    usage metadata.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class BudgetExceeded(RuntimeError):
    """Raised by ``check_budget`` when the running total crosses the ceiling."""


class TokenMeter:
    def __init__(self, path: Path | None = None, max_usd: float | None = None) -> None:
        self.path = Path(path) if path else None
        self.max_usd = max_usd
        self.events: list[TokenEvent] = []
        self._lock = threading.Lock()
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    # -- capture ---------------------------------------------------------

    def record(self, event: TokenEvent) -> TokenEvent:
        with self._lock:
            self.events.append(event)
            if self._fh is not None:
                self._fh.write(event.model_dump_json() + "\n")
                self._fh.flush()
        return event

    # -- aggregates ------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.events)

    @property
    def total_usd(self) -> float:
        return sum(
            cost_usd(e.model, e.input_tokens, e.output_tokens, e.cached_input_tokens)
            for e in self.events
        )

    @property
    def call_count(self) -> int:
        return len(self.events)

    def budget_remaining(self) -> float | None:
        if self.max_usd is None:
            return None
        return self.max_usd - self.total_usd

    def over_budget(self) -> bool:
        remaining = self.budget_remaining()
        return remaining is not None and remaining <= 0

    def check_budget(self) -> None:
        if self.over_budget():
            raise BudgetExceeded(
                f"spent ${self.total_usd:.4f} of ${self.max_usd:.4f} ceiling"
            )

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> TokenMeter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_events(path: Path) -> list[TokenEvent]:
    """Read a tokens.jsonl back, tolerating a truncated final line from a crash."""
    events: list[TokenEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(TokenEvent.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return events
