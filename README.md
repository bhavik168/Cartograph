# Cartographer

*Maps a question into an evidenced brief, and shows you exactly what it cost to
get there.*

A cyclic **LangGraph** state machine that runs a supervisor → specialist →
critic loop which can send work *back* for revision, with Pydantic-validated
structured output on every hop, real tool calling, persistent state,
cross-provider failover, and a **token auditor** that attributes every single
model call to a cause and tells you which of your tokens were wasted.

Bring your own key, run it locally, read the report it writes.

```bash
pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY
cp your-docs/*.md corpus/     # 5-15 plain-text documents
python cli.py ask "what does our corpus say about Q3 retention?"
```

One command produces one run directory:

```
runs/20260815T142201Z/
├── brief.json     the artifact: claims, evidence, confidence, limitations
├── trace.jsonl    one span per node execution
├── tokens.jsonl   one TokenEvent per LLM call, streamed as it happens
├── audit.json     machine-readable audit
└── audit.md       "where did the tokens go, and which were wasted?"
```

Read `audit.md` first. It tells you what the run actually did.

---

## The graph

```
        ┌──────────────┐
        │  supervisor  │◄──────────────────────┐
        └──────┬───────┘                       │
               │ conditional edge              │ revision directives
      ┌────────┴────────┐                      │ (bounded: MAX_REVISIONS)
      ▼                 ▼                      │
┌───────────┐    ┌───────────┐                 │
│researcher │    │researcher │  async fan-out  │
└─────┬─────┘    └─────┬─────┘                 │
      └────────┬───────┘                       │
               ▼                               │
        ┌──────────────┐                       │
        │ synthesizer  │                       │
        └──────┬───────┘                       │
               ▼                               │
        ┌──────────────┐   passed=False        │
        │    critic    │───────────────────────┘
        └──────┬───────┘
               │ passed=True OR revision_count >= MAX OR over budget
               ▼
        ┌──────────────┐
        │  finalizer   │
        └──────────────┘
```

**The cycle is the point.** The critic scores the draft against named criteria
(grounding, coverage, specificity, calibration) and, on failure, routes back to
the supervisor with targeted directives rather than returning a bad brief. Two
things bound it: `MAX_REVISIONS`, and an optional `--max-usd` ceiling checked
between nodes. Either bound finalizes anyway and stamps the reason into the
brief's `limitations` — honest degradation instead of an infinite loop or a
silent pass.

### Concepts worth naming

**Reducers.** `AnalystState.findings` and `.trace` are
`Annotated[list[...], operator.add]`. That reducer is what lets concurrently
running researcher branches write to the same state key without clobbering each
other — LangGraph merges each branch's partial update by calling the reducer
instead of overwriting. It's a LangGraph-specific idea and it's why fan-out
works at all. See `agent/state.py`.

**Every LLM call goes through one function.** `LLMClient.call()` in
`agent/llm.py` is the only place a model is invoked, and `node` and `cause` are
required keyword arguments. An unattributed call is a bug, not a gap in the
report.

**Tier routing is the cost story.** Cheap (Haiku) for routing and extraction,
strong (Sonnet) for synthesis and critique. The audit's *tier efficiency*
section tells you when you got that split wrong.

**Schema repair.** If a response fails Pydantic validation, the error text is
fed back once — "fix exactly these fields" — and the repair call is metered
separately as **waste**. Tighten the schema prompt and that number drops. See
`tests/test_schemas.py`.

**Quarantine.** Every tool result passes through `guards.quarantine()` before it
can reach a prompt: wrapped in a labelled untrusted block, injection patterns
flagged inline rather than silently deleted, hard cap on length. There is no
path from a tool to the model that skips it.

---

## The Token Auditor

Most projects print a total token count. That's a number, not an insight. The
auditor answers *where did the tokens go, and which of them were wasted?*

Every call carries a **cause**, and every cause has a class:

| Cause | Meaning | Class |
|---|---|---|
| `planning` | supervisor routing decisions | productive |
| `research` | specialist tool-calling turns | productive |
| `synthesis` | drafting the brief | productive |
| `critique` | critic scoring the draft | productive |
| `finalization` | last validation pass | productive |
| `memory_compaction` | scratchpad summarization | overhead |
| `fallback` | tokens spent on the failover provider | overhead |
| `injection_rescan` | re-processing after a quarantine flag | overhead |
| `schema_repair` | retry after a Pydantic `ValidationError` | **waste** |
| `revision` | any call made during a critic-driven revision pass | **waste** |
| `retry_transient` | rate-limit / 5xx retries | **waste** |

**Waste ratio = `waste_tokens / total_tokens`** — tokens that produced no new
information. It's the headline metric because it is fully deterministic (the
auditor counts tokens, it does not judge text) and directly actionable.

`audit.md` breaks the run down by node, by cause, by what filled the context
(system / scratchpad / tool_output / findings), by revision pass — *"what did
quality cost me?"* — and by tier.

Its **recommendations are rule-based, not LLM-generated**. Deterministic, free,
unit-testable, and they add no tokens to the very run being audited.

```
- schema_repairs > 2         -> tighten the schema prompt / add a few-shot example
- waste_ratio > 20%          -> revision loop dominating; sharpen critic criteria
- scratchpad > 40% of input  -> compact memory more aggressively
- strong tier > 70% of spend -> move extraction calls to the cheap tier
```

### Pricing

`agent/auditor/pricing.py` is a plain dict of `{model: (in_usd_per_1k,
out_usd_per_1k)}`. **Prices change. This repo does not claim authoritative
rates** — every USD figure is only as correct as the table you maintain. A model
with no entry is priced at zero and named in the report so the gap is obvious
rather than silent.

---

## Commands

```bash
python cli.py ask "question"                    # run the graph
python cli.py ask "question" --max-usd 0.50     # halt and finalize at a ceiling
python cli.py ask "question" --max-revisions 1  # tighter loop bound
python cli.py ask "question" --thread-id abc    # resume a halted run
python cli.py audit <run_id>                    # re-render a past run's audit
python cli.py audit <run_id> --json
python cli.py runs                              # list runs with cost and waste
pytest -q                                       # no API key needed
```

---

## Tests run without an API key

`LLMClient` never constructs a model itself; it calls an injected
`model_factory`. That one seam makes the whole orchestration layer testable
offline — CI drives the real graph with canned Pydantic objects and asserts:

- a failing critique routes back and increments the revision counter
- exceeding `MAX_REVISIONS` finalizes with honest limitations
- parallel findings accumulate through the reducer
- a `Claim` with zero evidence is rejected, and the repair path fires once
- poisoned tool output is flagged and capped, not silently dropped
- transient errors retry, then fail over to OpenAI as `fallback`
- auditor arithmetic — totals, per-cause aggregation, waste ratio, every
  recommendation threshold, and a zero-event run that must not divide by zero

CI runs exactly these. No key, no network, no cost. Live runs stay local.

---

## Honesty caveats

1. **No results ship in this repo.** Every number in `audit.md`, `trace.jsonl`
   and `brief.json` comes from your own runs. Nothing is pre-computed.
2. **The critic is an LLM judging an LLM** from the same family, so it is
   probably lenient about failure modes it shares with the writer. A brief that
   passes has *passed the critic* — it has not been verified true. The finalizer
   stamps this into every brief's `limitations`. The one grounding check that
   isn't an LLM's opinion is deterministic: a claim citing a source no
   researcher actually retrieved is dropped and demoted to an open question.
3. **Small corpus, no benchmark.** This demonstrates architecture, not accuracy.
   There is no retrieval quality metric here and none is claimed.
4. **Injection defense is a mitigation, not a guarantee.** Known patterns are
   flagged and output is bounded. A novel injection can still get through.
5. **Cost figures are estimates** from a hand-maintained price table, computed
   from provider-reported usage. Treat them as a relative signal, not a bill.

---

## Layout

```
agent/
├── schemas.py       every Pydantic model — state, agent outputs, telemetry
├── state.py         the graph state TypedDict and its reducers
├── graph.py         StateGraph wiring, conditional edges, the bounded cycle
├── runtime.py       per-run context: llm, meter, run dir, trace writer
├── llm.py           provider routing, retry, failover, repair, metering
├── memory.py        SQLite checkpointer + scratchpad compaction
├── guards.py        injection quarantine + context caps
├── nodes/           supervisor · researcher · synthesizer · critic · finalizer
├── tools/           corpus_search (BM25) · calculator (AST) · fetch_url (off)
└── auditor/         meter · attribute · pricing · report
cli.py               ask / audit / runs
tests/               all offline, all stubbed
```

## Try to break it

The interesting runs are the ones that go wrong on purpose:

- Ask something the corpus only partly supports — watch the critic send it back,
  the revision counter increment, and the audit price the revision as avoidable.
- Temporarily tighten a Pydantic constraint — watch the repair path fire and
  show up as waste.
- Unset `ANTHROPIC_API_KEY` with `OPENAI_API_KEY` set — the run completes on the
  fallback path and the audit attributes those tokens to `fallback`.
- Set `--max-usd` below your typical run cost — the cycle halts and finalizes
  with `limitations: ["Halted at budget ceiling: ..."]`.
