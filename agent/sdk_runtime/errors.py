"""Exceptions the CLI can catch without importing the Agents SDK."""

from __future__ import annotations


class GuardrailRejected(RuntimeError):
    """An output guardrail tripped and the run was failed rather than shipped."""
