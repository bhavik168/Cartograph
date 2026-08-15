"""Retry and cross-provider failover. No API key required.

Failover is a path that is easy to write and never exercise; these tests drive
it deliberately so the "provider fallback works" claim is a tested one.
"""

from __future__ import annotations

import pytest

from agent.auditor.attribute import build_audit
from agent.auditor.meter import TokenMeter
from agent.llm import LLMClient, LLMConfig, LLMError
from agent.schemas import RoutingDecision
from tests.stubs import StubScript, stub_factory

DECISION = RoutingDecision(next_agents=["synthesizer"], rationale="ok")


class RateLimit(Exception):
    def __str__(self) -> str:
        return "429 rate limit exceeded"


def client(script: StubScript, providers=("anthropic", "openai"), **cfg) -> LLMClient:
    return LLMClient(
        TokenMeter(),
        LLMConfig(base_backoff_s=0.0, max_backoff_s=0.0, **cfg),
        model_factory=stub_factory(script),
        providers=list(providers),
    )


async def call(c: LLMClient) -> RoutingDecision:
    return await c.call(
        RoutingDecision, [("user", "go")], node="supervisor", cause="planning"
    )


@pytest.mark.asyncio
async def test_transient_error_is_retried_on_the_same_provider():
    script = StubScript(
        responses={RoutingDecision: DECISION},
        errors={RoutingDecision: [RateLimit(), None]},
    )
    c = client(script)

    assert await call(c) == DECISION

    causes = [e.cause for e in c.meter.events]
    assert causes == ["retry_transient", "planning"]
    assert all(e.provider == "anthropic" for e in c.meter.events)


@pytest.mark.asyncio
async def test_exhausting_the_primary_fails_over_to_openai():
    script = StubScript(
        responses={RoutingDecision: DECISION},
        errors={RoutingDecision: [RateLimit(), RateLimit(), RateLimit(), None]},
    )
    c = client(script, max_attempts=3)

    assert await call(c) == DECISION

    served = c.meter.events[-1]
    assert served.provider == "openai"
    # Failover tokens are attributed to `fallback`, not to the caller's cause,
    # so the audit can price the cost of resilience separately.
    assert served.cause == "fallback"
    assert served.model.startswith("gpt")

    audit = build_audit(c.meter.events)
    assert any(b.key == "fallback" for b in audit.by_cause)
    assert {b.key for b in audit.by_provider} == {"anthropic", "openai"}


@pytest.mark.asyncio
async def test_non_transient_error_skips_retries_and_fails_over_immediately():
    script = StubScript(
        responses={RoutingDecision: DECISION},
        errors={RoutingDecision: [ValueError("bad request: unsupported parameter"), None]},
    )
    c = client(script, max_attempts=3)

    assert await call(c) == DECISION
    # One failed attempt on the primary, not three: a 400 will not fix itself.
    assert sum(1 for e in c.meter.events if e.provider == "anthropic") == 1


@pytest.mark.asyncio
async def test_all_providers_failing_raises():
    script = StubScript(
        responses={RoutingDecision: DECISION},
        errors={RoutingDecision: [RateLimit()] * 12},
    )
    c = client(script, max_attempts=2)

    with pytest.raises(LLMError, match="all providers failed"):
        await call(c)
    assert all(not e.ok for e in c.meter.events)


@pytest.mark.asyncio
async def test_tier_selects_the_model():
    c = client(StubScript(responses={RoutingDecision: [DECISION, DECISION]}))

    await c.call(
        RoutingDecision, [("user", "x")], node="supervisor", cause="planning", tier="cheap"
    )
    await c.call(RoutingDecision, [("user", "x")], node="critic", cause="critique", tier="strong")

    cheap, strong = c.meter.events
    assert "haiku" in cheap.model
    assert "sonnet" in strong.model
    assert (cheap.tier, strong.tier) == ("cheap", "strong")


def test_available_providers_respects_the_environment(monkeypatch):
    from agent.llm import available_providers

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert available_providers() == []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert available_providers() == ["openai"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert available_providers() == ["anthropic", "openai"]
