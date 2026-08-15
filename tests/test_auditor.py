"""Auditor arithmetic and attribution. No API key required.

The auditor is the headline feature, so its numbers are asserted exactly rather
than approximately: if the waste ratio is wrong, the project's central claim is
wrong.
"""

from __future__ import annotations

import json

import pytest

from agent.auditor.attribute import (
    CAUSE_CLASS,
    SCHEMA_REPAIR_THRESHOLD,
    build_audit,
    classify,
    recommend,
)
from agent.auditor.meter import TokenMeter, load_events
from agent.auditor.pricing import cost_usd, is_priced, rates_for
from agent.auditor.report import render_markdown, write_report
from agent.schemas import CAUSES, InputComposition, TokenEvent


def event(cause="research", *, node="researcher", inp=100, out=50, rev=0, **kw):
    return TokenEvent(
        node=node,
        cause=cause,
        model=kw.pop("model", "claude-haiku-4-5-20251001"),
        tier=kw.pop("tier", "cheap"),
        provider=kw.pop("provider", "anthropic"),
        input_tokens=inp,
        output_tokens=out,
        revision_index=rev,
        **kw,
    )


# -- taxonomy -----------------------------------------------------------


def test_every_cause_is_classified():
    assert set(CAUSE_CLASS) == set(CAUSES)
    for cause in CAUSES:
        assert CAUSE_CLASS[cause] in {"productive", "overhead", "waste"}


@pytest.mark.parametrize("cause", ["schema_repair", "revision", "retry_transient"])
def test_waste_causes(cause):
    assert classify(cause) == "waste"


@pytest.mark.parametrize("cause", ["planning", "research", "synthesis", "critique"])
def test_productive_causes(cause):
    assert classify(cause) == "productive"


@pytest.mark.parametrize("cause", ["memory_compaction", "fallback", "injection_rescan"])
def test_overhead_causes(cause):
    assert classify(cause) == "overhead"


# -- arithmetic ---------------------------------------------------------


def test_totals_are_exact():
    audit = build_audit([event(inp=100, out=50), event(inp=200, out=25)])
    assert audit.llm_calls == 2
    assert audit.input_tokens == 300
    assert audit.output_tokens == 75
    assert audit.total_tokens == 375


def test_waste_ratio_is_exact():
    events = [
        event("research", inp=600, out=0),        # 600 productive
        event("synthesis", inp=200, out=0),       # 200 productive
        event("schema_repair", inp=100, out=0),   # 100 waste
        event("revision", inp=100, out=0, rev=1),  # 100 waste
    ]
    audit = build_audit(events)

    assert audit.total_tokens == 1000
    assert audit.productive_tokens == 800
    assert audit.waste_tokens == 200
    assert audit.waste_ratio == pytest.approx(0.20)


def test_revision_index_above_zero_is_avoidable_spend():
    audit = build_audit(
        [event("research", inp=100, out=0), event("revision", inp=300, out=0, rev=1)]
    )
    assert audit.revisions == 1
    passes = {p.revision_index: p for p in audit.pass_costs}
    assert passes[0].tokens == 100
    assert passes[1].tokens == 300
    assert audit.waste_tokens == 300


def test_per_node_aggregation_is_sorted_desc():
    audit = build_audit(
        [
            event(node="supervisor", inp=10, out=5),
            event(node="synthesizer", inp=900, out=100),
            event(node="critic", inp=400, out=50),
        ]
    )
    assert [b.key for b in audit.by_node] == ["synthesizer", "critic", "supervisor"]
    assert audit.by_node[0].pct_of_total == pytest.approx(1000 / 1465 * 100)


def test_percentages_sum_to_one_hundred():
    audit = build_audit([event(inp=100), event("synthesis", inp=300)])
    assert sum(b.pct_of_total for b in audit.by_cause) == pytest.approx(100.0)


def test_failed_calls_are_counted_but_cost_nothing():
    audit = build_audit(
        [event("research", inp=100, out=50), event("retry_transient", inp=0, out=0, ok=False)]
    )
    assert audit.llm_calls == 2
    assert audit.failed_calls == 1
    assert audit.total_tokens == 150


def test_tier_shares():
    audit = build_audit(
        [
            event(tier="cheap", model="claude-haiku-4-5", inp=1000, out=1000),
            event(tier="strong", model="claude-sonnet-4-5", inp=1000, out=1000),
        ]
    )
    assert audit.cheap_call_share == pytest.approx(0.5)
    # Same tokens, ~3x the price: the strong tier dominates spend.
    assert audit.strong_spend_share > audit.cheap_spend_share
    assert audit.cheap_spend_share + audit.strong_spend_share == pytest.approx(1.0)


def test_composition_is_summed_and_percentaged():
    audit = build_audit(
        [
            event(composition=InputComposition(system=100, scratchpad=300)),
            event(composition=InputComposition(system=100, tool_output=500)),
        ]
    )
    assert audit.composition["system"] == 200
    assert audit.composition["scratchpad"] == 300
    assert audit.composition_pct["tool_output"] == pytest.approx(500 / 1000 * 100)


# -- pricing ------------------------------------------------------------


def test_pricing_prefix_match_and_cost():
    assert rates_for("claude-sonnet-4-5-20250929") == rates_for("claude-sonnet-4")
    assert cost_usd("claude-sonnet-4-5", 1000, 1000) == pytest.approx(0.003 + 0.015)


def test_cached_input_is_discounted():
    full = cost_usd("claude-sonnet-4-5", 1000, 0)
    cached = cost_usd("claude-sonnet-4-5", 1000, 0, cached_input_tokens=1000)
    assert cached == pytest.approx(full * 0.1)


def test_unknown_model_is_zero_and_surfaced():
    assert not is_priced("some-future-model")
    assert cost_usd("some-future-model", 1000, 1000) == 0.0
    audit = build_audit([event(model="some-future-model")])
    assert audit.unpriced_models == ["some-future-model"]
    assert any("pricing.py" in r for r in audit.recommendations)


# -- recommendations ----------------------------------------------------


def test_schema_repair_rule_fires_above_threshold_only():
    below = build_audit(
        [event("schema_repair") for _ in range(SCHEMA_REPAIR_THRESHOLD)]
        + [event("research", inp=100_000)]
    )
    assert not any("tighten the schema prompt" in r for r in below.recommendations)

    above = build_audit(
        [event("schema_repair") for _ in range(SCHEMA_REPAIR_THRESHOLD + 1)]
        + [event("research", inp=100_000)]
    )
    assert any("tighten the schema prompt" in r for r in above.recommendations)


def test_waste_ratio_rule_fires_above_threshold_only():
    at_threshold = build_audit(
        [event("research", inp=800, out=0), event("revision", inp=200, out=0, rev=1)]
    )
    assert at_threshold.waste_ratio == pytest.approx(0.20)
    assert not any("revision loop is dominating" in r for r in at_threshold.recommendations)

    over = build_audit(
        [event("research", inp=700, out=0), event("revision", inp=300, out=0, rev=1)]
    )
    assert any("revision loop is dominating" in r for r in over.recommendations)


def test_scratchpad_rule_fires():
    audit = build_audit(
        [event(composition=InputComposition(scratchpad=500, system=100))]
    )
    assert any("compact memory more aggressively" in r for r in audit.recommendations)


def test_strong_tier_rule_fires():
    audit = build_audit(
        [event(tier="strong", model="claude-sonnet-4-5", inp=10_000, out=10_000)]
    )
    assert any("move" in r and "cheap tier" in r for r in audit.recommendations)


def test_quiet_run_gets_a_no_op_recommendation():
    audit = build_audit([event("research", inp=100, out=10)])
    assert audit.recommendations == ["No thresholds crossed. Nothing to tune from this run."]


# -- empty / degenerate -------------------------------------------------


def test_zero_event_run_does_not_divide_by_zero():
    audit = build_audit([], run_id="empty")
    assert audit.total_tokens == 0
    assert audit.waste_ratio == 0.0
    assert audit.est_cost_usd == 0.0
    assert audit.cheap_spend_share == 0.0
    markdown = render_markdown(audit)
    assert "WASTE RATIO: 0.0%" in markdown


def test_zero_cost_run_does_not_divide_by_zero():
    audit = build_audit(
        [event(model="unpriced-model", rev=0), event(model="unpriced-model", rev=1)]
    )
    assert audit.est_cost_usd == 0.0
    assert "Quality cost" in render_markdown(audit)


# -- meter and round-trip ----------------------------------------------


def test_meter_streams_events_to_disk(tmp_path):
    path = tmp_path / "tokens.jsonl"
    with TokenMeter(path=path) as meter:
        meter.record(event("research"))
        meter.record(event("synthesis", node="synthesizer"))
        # Flushed as they happen, so a crashed run still leaves an audit.
        assert len(path.read_text().strip().splitlines()) == 2

    reloaded = load_events(path)
    assert [e.cause for e in reloaded] == ["research", "synthesis"]


def test_load_events_tolerates_a_truncated_final_line(tmp_path):
    path = tmp_path / "tokens.jsonl"
    path.write_text(event("research").model_dump_json() + '\n{"node": "cri')
    assert len(load_events(path)) == 1


def test_budget_ceiling(tmp_path):
    meter = TokenMeter(max_usd=0.01)
    assert not meter.over_budget()
    meter.record(event(model="claude-sonnet-4-5", inp=10_000, out=0))  # $0.03
    assert meter.over_budget()
    with pytest.raises(Exception, match="ceiling"):
        meter.check_budget()


def test_write_report_produces_both_files(tmp_path):
    audit = write_report(
        tmp_path,
        [event("research", inp=800, out=0), event("schema_repair", inp=200, out=0)],
        run_id="r1",
        question="does it work?",
        outcome="passed critic on the first pass",
        wall_clock_s=12.5,
    )
    assert audit.waste_ratio == pytest.approx(0.20)

    markdown = (tmp_path / "audit.md").read_text()
    assert "does it work?" in markdown
    assert "WASTE RATIO: 20.0%" in markdown
    assert "| researcher |" in markdown

    data = json.loads((tmp_path / "audit.json").read_text())
    assert data["run_id"] == "r1"
    assert data["waste_ratio"] == pytest.approx(0.20)


def test_recommend_is_deterministic():
    audit = build_audit([event("schema_repair") for _ in range(5)])
    assert recommend(audit) == recommend(audit)
