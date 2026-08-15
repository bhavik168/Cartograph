"""Tool registry. The researcher node binds whatever ``active_tools`` returns."""

from __future__ import annotations

from agent.tools.calculator import calculator
from agent.tools.corpus_search import corpus_search
from agent.tools.fetch_url import fetch_url, is_enabled

__all__ = ["calculator", "corpus_search", "fetch_url", "active_tools", "tool_map"]


def active_tools() -> list:
    tools = [corpus_search, calculator]
    if is_enabled():
        tools.append(fetch_url)
    return tools


def tool_map() -> dict:
    return {t.name: t for t in active_tools()}
