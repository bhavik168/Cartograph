"""Schema constraints and the repair loop. No API key required."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.auditor.meter import TokenMeter
from agent.llm import LLMClient, LLMConfig, LLMError
from agent.schemas import Brief, Claim, Critique, Evidence, RoutingDecision
from tests.stubs import StubScript, stub_factory


def make_evidence(**kw) -> Evidence:
    return Evidence(**{"source": "a.md", "quote": "q", "relevance": 0.9, **kw})


def test_claim_requires_evidence():
    with pytest.raises(ValidationError):
        Claim(statement="unsupported", evidence=[], confidence="high")


def test_claim_with_evidence_is_valid():
    claim = Claim(statement="s", evidence=[make_evidence()], confidence="medium")
    assert claim.evidence[0].source == "a.md"


def test_quote_length_is_capped():
    with pytest.raises(ValidationError):
        make_evidence(quote="x" * 301)


@pytest.mark.parametrize("relevance", [-0.1, 1.1])
def test_relevance_bounds(relevance):
    with pytest.raises(ValidationError):
        make_evidence(relevance=relevance)


def test_confidence_is_an_enum():
    with pytest.raises(ValidationError):
        Claim(statement="s", evidence=[make_evidence()], confidence="very high")


def test_routing_rejects_unknown_agent():
    with pytest.raises(ValidationError):
        RoutingDecision(next_agents=["janitor"], rationale="r")


def test_critique_consistency_helper():
    assert Critique(passed=True).is_consistent()
    assert Critique(passed=False, revision_directives=["fix it"]).is_consistent()
    assert not Critique(passed=True, revision_directives=["fix it"]).is_consistent()
    assert not Critique(passed=False).is_consistent()


def _client(script: StubScript, **cfg) -> LLMClient:
    return LLMClient(
        TokenMeter(),
        LLMConfig(**cfg),
        model_factory=stub_factory(script),
        providers=["anthropic"],
    )


@pytest.mark.asyncio
async def test_repair_path_fires_on_invalid_output_and_is_metered():
    valid = Brief(
        question="q",
        claims=[Claim(statement="s", evidence=[make_evidence()], confidence="low")],
    )
    # First response is a dict with an unevidenced claim: invalid, so the
    # client must run exactly one repair pass and then succeed.
    script = StubScript(
        responses={
            Brief: [
                {"question": "q", "claims": [{"statement": "s", "evidence": [],
                                              "confidence": "low"}]},
                valid,
            ]
        }
    )
    client = _client(script)

    result = await client.call(
        Brief, [("user", "go")], node="synthesizer", cause="synthesis", tier="strong"
    )

    assert result == valid
    assert client.repair_count == 1
    causes = [e.cause for e in client.meter.events]
    assert causes == ["synthesis", "schema_repair"]
    assert client.meter.events[0].ok is False
    assert client.meter.events[1].ok is True


@pytest.mark.asyncio
async def test_repair_failing_twice_raises():
    bad = {"question": "q", "claims": [{"statement": "s", "evidence": [],
                                        "confidence": "low"}]}
    client = _client(StubScript(responses={Brief: [bad, bad]}))

    with pytest.raises(LLMError, match="after repair"):
        await client.call(
            Brief, [("user", "go")], node="synthesizer", cause="synthesis"
        )
    assert client.repair_count == 1


@pytest.mark.asyncio
async def test_repair_can_be_disabled():
    bad = {"question": "q", "claims": [{"statement": "s", "evidence": [],
                                        "confidence": "low"}]}
    client = _client(StubScript(responses={Brief: [bad]}), enable_repair=False)

    with pytest.raises(LLMError):
        await client.call(Brief, [("user", "go")], node="critic", cause="critique")
    assert client.repair_count == 0


@pytest.mark.asyncio
async def test_unattributed_call_is_rejected():
    client = _client(StubScript(responses={Brief: []}))
    with pytest.raises(ValueError, match="node and cause"):
        await client.call(Brief, [("user", "go")], node="", cause="synthesis")
