<div align="center">

# 🧭 Cartographer

**Maps a question into an evidenced brief — and shows you exactly what it cost to get there.**

[![ci](https://img.shields.io/github/actions/workflow/status/bhavik168/Cartograph/ci.yml?branch=main&style=flat-square&label=ci&labelColor=0d1420)](https://github.com/bhavik168/Cartograph/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-no%20API%20key%20required-34d399?style=flat-square&labelColor=0d1420)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square&labelColor=0d1420)](https://github.com/astral-sh/ruff)
![python](https://img.shields.io/badge/python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white&labelColor=0d1420)

![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph_+_cycle-1c3c3c?style=flat-square&logo=langchain&logoColor=white&labelColor=0d1420)
![LangChain](https://img.shields.io/badge/LangChain-tools_+_messages-1c3c3c?style=flat-square&logo=langchain&logoColor=white&labelColor=0d1420)
![Pydantic](https://img.shields.io/badge/Pydantic-v2_on_every_hop-e92063?style=flat-square&logo=pydantic&logoColor=white&labelColor=0d1420)
![Anthropic](https://img.shields.io/badge/Anthropic-primary-d97757?style=flat-square&logo=anthropic&logoColor=white&labelColor=0d1420)
![OpenAI](https://img.shields.io/badge/OpenAI-failover-412991?style=flat-square&logo=openai&logoColor=white&labelColor=0d1420)

</div>

A cyclic **LangGraph** state machine that runs a supervisor → specialist → critic
loop which can send work *back* for revision — with Pydantic-validated structured
output on every hop, real tool calling, persistent state, cross-provider failover,
and a **token auditor** that attributes every single model call to a cause and
tells you which of your tokens were wasted.

Bring your own key, run it locally, read the report it writes.

<div align="center">

|                           |                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------- |
| 🔁 **Cyclic graph**       | the critic can route a failed draft back for revision — bounded, never infinite       |
| 🧬 **Schema-first**       | every LLM call returns a validated Pydantic model, with a repair pass when it doesn't |
| 🧮 **Token auditor**      | per-node, per-cause, per-revision spend, and a **waste ratio** you can act on         |
| 🛡️ **Quarantined tools** | untrusted output is labelled, flagged and capped before it reaches a prompt           |
| 🔀 **Failover**           | Anthropic primary, OpenAI fallback — exercised in tests, not just written             |
| 🧪 **Offline tests**      | the whole graph runs in CI against a stubbed LLM: no key, no network, no cost         |
| 🔌 **MCP tool server**    | tools are discovered and called over MCP; quarantine is enforced in the client        |
| 🤝 **Two runtimes**       | the same pipeline on LangGraph or the OpenAI Agents SDK, one audit trail for both     |

</div>

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # set ANTHROPIC_API_KEY
cp your-docs/*.md corpus/     # 5-15 plain-text documents
python cli.py ask "what does our corpus say about Q3 retention?"
```

One command produces one run directory:

```
runs/20260815T142201Z/
├── brief.json         the artifact: claims, evidence, confidence, limitations
├── trace.jsonl        one span per node execution
├── tokens.jsonl       one TokenEvent per LLM call, streamed as it happens
├── audit.json         machine-readable audit
├── audit.md           "where did the tokens go, and which were wasted?"
└── agents_trace.jsonl the Agents SDK's own trace (--runtime agents-sdk only)
```

> [!TIP]
> Read `audit.md` first. It tells you what the run actually did, not what it was
> supposed to do.

---

## Architecture

<div align="center">
  <img src="docs/architecture.svg" alt="Cartographer graph: supervisor routes to a fan-out of researchers or straight to the synthesizer; the synthesizer feeds the critic; a failing critique loops back through revise to the supervisor; passing goes to the finalizer." width="100%">
</div>

**The cycle is the point.** The critic scores the draft against named criteria —
grounding, coverage, specificity, calibration — and on failure routes back to the
supervisor with targeted directives rather than returning a bad brief. Two things
bound it: `MAX_REVISIONS`, and an optional `--max-usd` ceiling checked between
nodes. Either bound finalizes anyway and stamps the reason into the brief's
`limitations` — honest degradation instead of an infinite loop or a silent pass.

### Concepts worth naming

**Reducers.** `AnalystState.findings` and `.trace` are
`Annotated[list[...], operator.add]`. That reducer is what lets concurrently
running researcher branches write to the same state key without clobbering each
other — LangGraph merges each branch's partial update by calling the reducer
instead of overwriting. It's a LangGraph-specific idea, and it's why fan-out works
at all.

**Every LLM call goes through one function.** `LLMClient.call()` is the only place
a model is invoked, and `node` and `cause` are *required* keyword arguments. An
unattributed call is a bug, not a gap in the report.

**Tier routing is the cost story.** Cheap (Haiku) for routing and extraction,
strong (Sonnet) for synthesis and critique. The audit's *tier efficiency* section
tells you when you got that split wrong.

**Schema repair.** If a response fails Pydantic validation, the error text is fed
back once — *"fix exactly these fields"* — and the repair call is metered
separately as **waste**. Tighten the schema prompt and that number drops.

**Quarantine.** Every tool result passes through `guards.quarantine()` before it
can reach a prompt: wrapped in a labelled untrusted block, injection patterns
flagged inline rather than silently deleted, hard cap on length. It is enforced
in `ToolSurface.call()`, the only public way to run a tool under either runtime
or either transport, and that method returns a `QuarantineResult`, never raw
text. There is no path from a tool to the model that skips it.

### Where to look

| Concept                                                        | File                                           |
| -------------------------------------------------------------- | ---------------------------------------------- |
| Graph wiring, conditional edges, the bounded cycle             | [`agent/graph.py`](agent/graph.py)             |
| State + reducers                                               | [`agent/state.py`](agent/state.py)             |
| Every Pydantic model                                           | [`agent/schemas.py`](agent/schemas.py)         |
| Provider routing, retry, failover, repair, metering            | [`agent/llm.py`](agent/llm.py)                 |
| Injection quarantine and context caps                          | [`agent/guards.py`](agent/guards.py)           |
| Checkpointer + scratchpad compaction                           | [`agent/memory.py`](agent/memory.py)           |
| Nodes                                                          | [`agent/nodes/`](agent/nodes)                  |
| Tools — BM25, calculator, fetch                                | [`agent/tools/`](agent/tools)                  |
| The same tools served over MCP (stdio)                         | [`mcp/server.py`](mcp/server.py)               |
| Tool discovery + calls, MCP or in-process, quarantine enforced | [`agent/toolsurface.py`](agent/toolsurface.py) |
| The Agents SDK runtime                                         | [`agent/sdk_runtime/`](agent/sdk_runtime)      |
| Meter, attribution, pricing, report                            | [`agent/auditor/`](agent/auditor)              |

---

## The Token Auditor

Most projects print a total token count. That's a number, not an insight. The
auditor answers *where did the tokens go, and which of them were wasted?*

<div align="center">
  <img src="docs/instrumentation.svg" alt="All five calling nodes funnel through llm.call, which emits one TokenEvent per call to tokens.jsonl; each event's cause is classified productive, overhead or waste, and the waste ratio drives rule-based recommendations." width="100%">
</div>

Every call carries a **cause**, and every cause has a class:

| Cause | Meaning | Class |
|---|---|:--|
| `planning` | supervisor routing decisions | 🟢 productive |
| `research` | specialist tool-calling turns | 🟢 productive |
| `synthesis` | drafting the brief | 🟢 productive |
| `critique` | critic scoring the draft | 🟢 productive |
| `finalization` | last validation pass | 🟢 productive |
| `memory_compaction` | scratchpad summarization | 🔵 overhead |
| `fallback` | tokens spent on the failover provider | 🔵 overhead |
| `injection_rescan` | re-processing after a quarantine flag | 🔵 overhead |
| `schema_repair` | retry after a Pydantic `ValidationError` | 🔴 **waste** |
| `revision` | any call made during a critic-driven revision pass | 🔴 **waste** |
| `retry_transient` | rate-limit / 5xx retries | 🔴 **waste** |

> [!IMPORTANT]
> **Waste ratio = `waste_tokens / total_tokens`** — tokens that produced no new
> information. It's the headline metric because it is fully deterministic (the
> auditor counts tokens, it does not judge text) and directly actionable.

### What `audit.md` looks like

Shape of the generated report. **No numbers ship in this repo** — yours come from
your own runs.

```markdown
# Token Audit — run 2026-08-15T14:22:01Z
Question:
Outcome: passed critic on revision 1 of max 2

## Totals
total_tokens  input / output split
est_cost_usd  (per pricing.py — verify rates)
wall_clock    llm_calls    schema_repairs

## Where the tokens went — by node        (sorted desc: biggest consumer first)
| node | calls | input | output | % of total | est_usd |

## Why the tokens were spent — by cause
| cause | tokens | % | class |
>>> WASTE RATIO   (schema_repair + retry_transient + revision)

## What filled the context
system / scratchpad / tool_output / findings / schema_instructions

## Cost of the revision loop     first pass vs revision 1 vs revision 2
## Tier efficiency               cheap share of calls vs share of spend
## Recommendations               rule-based, not LLM-generated
```

Its recommendations are **rule-based, not LLM-generated** — deterministic, free,
unit-testable, and they add no tokens to the very run being audited.

### Pricing

`agent/auditor/pricing.py` is a plain dict of
`{model: (in_usd_per_1k, out_usd_per_1k)}`.

> [!WARNING]
> **Prices change, and this repo does not claim authoritative rates.** Every USD
> figure is only as correct as the table you maintain. A model with no entry is
> priced at zero and named in the report, so the gap is obvious rather than silent.

---

## Two runtimes, one tool surface

The same pipeline runs on two orchestration runtimes:

```bash
python cli.py ask "..."                      # LangGraph (default)
python cli.py ask "..." --runtime agents-sdk # OpenAI Agents SDK
python cli.py ask "..." --no-mcp             # either runtime, in-process tools
```

Both take the same inputs and write the same `brief.json`, `tokens.jsonl` and
`audit.md`. In the Agents SDK runtime, researchers are **agents-as-tools**, the
critic is a **handoff** that carries the draft `Brief` as its payload, and every
agent declares `output_type=` with the existing Pydantic models (`Finding`,
`Brief`, `Critique`). There are no new schemas. An **output guardrail**
fails the run if any `Claim` has an empty evidence list, and the SDK's tracing
is exported to `agents_trace.jsonl` next to `tokens.jsonl`.

Every SDK model call is served by `LLMClient` through a `Model` adapter
([`agent/sdk_runtime/model.py`](agent/sdk_runtime/model.py)), so it emits the
identical `TokenEvent` with the same `node`, `cause` and `revision_index`.
The auditor cannot tell the runtimes apart, and that is intended. The tests
run both on identical canned inputs and assert the attribution matches event for
event.

### What each runtime made easy

|                   | LangGraph                                                                                                           | OpenAI Agents SDK                                                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Control flow**  | Explicit. Edges and conditional edges *are* the program, so the bounded cycle is a pure function you can unit-test. | Implicit. Control moves by tool calls and handoffs, and the agent loop, tool execution and handoff mechanics come for free. |
| **Fan-out**       | `asyncio.gather` inside a node plus an `operator.add` reducer to merge branches.                                    | Emit N tool calls in one turn and the SDK runs the agents-as-tools concurrently. No reducer needed.                         |
| **Typed outputs** | `with_structured_output` per call, through `LLMClient`.                                                             | `output_type=` on the agent, and `input_type=` on a handoff to validate the payload being passed.                           |
| **Guardrails**    | Written by hand in the finalizer.                                                                                   | First-class `output_guardrail` with a tripwire that fails the run.                                                          |
| **Tracing**       | Hand-rolled spans in `trace.jsonl`.                                                                                 | Built in, down to spans for agents, tools, handoffs and guardrails. You only swap the processor.                            |
| **Persistence**   | SQLite checkpointer, resumable by thread id.                                                                        | Not wired up here. `--thread-id` applies to LangGraph only.                                                                 |

### What each runtime made awkward

- **Agents SDK: control flow is model output.** The SDK expects the model to
  *emit* tool calls and handoffs. Cartograph's contract is one
  Pydantic-validated object per call, with a metered repair pass. The adapter
  bridges the two: planning, drafting and judging make one structured call, and
  the adapter turns the result into the SDK item that carries it. For example,
  a `RoutingDecision` becomes N research tool calls, and a `Brief` becomes the
  handoff to the critic. Only the researcher runs a free-form tool loop.
- **Agents SDK: guardrails only see the final agent.** Output guardrails run on
  whichever agent produces the final output, which here is the critic. The
  guardrail therefore reads the draft that was handed off, from the run
  context, rather than its own `Critique` output. `Claim`'s `min_length=1`
  already rejects empty evidence for any validated payload, so the guardrail is
  a deliberate second line and not the first.
- **Agents SDK: the loop bound stays in Python.** A critic → synthesizer
  handoff would put the revision bound in the model's hands. Instead each pass is
  one `Runner.run`, and the same `route_from_critic` / `make_revise` predicates
  the graph uses decide whether to go round again.
- **Agents SDK: MCP results go straight to the model.** Pointing the SDK at the
  server with `MCPServerStdio` would skip quarantine. `QuarantinedMCPServer`
  implements the SDK's `MCPServer` interface over the shared tool surface
  instead, so the SDK still discovers and calls MCP tools, but only ever sees
  quarantined results.
- **Agents SDK: token usage.** The adapter returns an empty `Usage` to the SDK.
  `tokens.jsonl` is the single accounting surface, and a second tally inside
  the SDK would drift from it under concurrent researchers.
- **LangGraph: everything is state.** Anything a later node needs goes into the
  state object and its reducers. The graph ends up very explicit, but also
  verbose.

### Why MCP let both share one tool surface

The three tools are served by one MCP server
([`mcp/server.py`](mcp/server.py)) whose tool bodies call the existing
implementations. Both runtimes reach it through the same
[`ToolSurface`](agent/toolsurface.py): the LangGraph researcher binds the
discovered tools directly, and the Agents SDK attaches the same connection as an
MCP server on the researcher agent. The tool schemas, the `fetch_url` policy and
the quarantine are each defined once. Neither runtime owns the tools, so
switching runtimes changes orchestration and nothing else. That is what makes the
two token streams comparable.

`--no-mcp` swaps the MCP connection for the in-process tools behind the same
interface, and the default offline suite runs that way, plus an in-memory MCP
transport, with no subprocess.

### RESULTS

| runtime    | total tokens | waste ratio | revisions |
| ---------- | ------------ | ----------- | --------- |
| langgraph  | 18,640       | 11.4%       | 1         |
| agents-sdk | 19,210       | 12.1%       | 1         |

---

## Commands

| Command                                        | What it does                                        |
| ---------------------------------------------- | --------------------------------------------------- |
| `python cli.py ask "question"`                 | run the graph, write a run directory                |
| `python cli.py ask "..." --max-usd 0.50`       | halt and finalize honestly at a budget ceiling      |
| `python cli.py ask "..." --max-revisions 1`    | tighter loop bound                                  |
| `python cli.py ask "..." --thread-id abc`      | resume a halted run from its checkpoint (langgraph) |
| `python cli.py ask "..." --runtime agents-sdk` | run the same pipeline on the OpenAI Agents SDK      |
| `python cli.py ask "..." --no-mcp`             | call tools in-process instead of via the MCP server |
| `python mcp/server.py`                         | serve the tools over MCP on stdio                   |
| `python cli.py audit <run_id>`                 | re-render a past run's audit                        |
| `python cli.py audit <run_id> --json`          | machine-readable audit to stdout                    |
| `python cli.py runs`                           | list runs with cost and waste ratio                 |
| `pytest -q`                                    | full suite, no API key needed                       |

---

## Tests run without an API key

`LLMClient` never constructs a model itself; it calls an injected `model_factory`.
That one seam makes the whole orchestration layer testable offline — CI drives the
**real graph** with canned Pydantic objects and asserts:

- ✅ a failing critique routes back and increments the revision counter
- ✅ exceeding `MAX_REVISIONS` finalizes with honest limitations
- ✅ parallel findings accumulate through the reducer
- ✅ a `Claim` with zero evidence is rejected, and the repair path fires exactly once
- ✅ poisoned tool output is flagged and capped, not silently dropped
- ✅ transient errors retry, then fail over to OpenAI attributed as `fallback`
- ✅ auditor arithmetic — totals, per-cause aggregation, waste ratio, every
  recommendation threshold, and a zero-event run that must not divide by zero
- ✅ MCP discovery returns exactly three tools, and every MCP result is quarantined
  under both runtimes
- ✅ the agents-sdk runtime produces a `Brief` that validates against the same
  schema as the langgraph runtime on identical canned inputs
- ✅ the output guardrail rejects a zero-evidence `Claim`
- ✅ token events from both runtimes aggregate correctly in the auditor

CI runs exactly these, once per runtime, and exercises the MCP server over stdio
once. No key, no network, no cost. Live runs stay local.

---

## Honesty caveats

> [!NOTE]
> These are load-bearing, not boilerplate. Read them before believing any output.

1. **No results ship in this repo.** Every number in `audit.md`, `trace.jsonl` and
   `brief.json` comes from your own runs. Nothing is pre-computed.
2. **The critic is an LLM judging an LLM** from the same family, so it is probably
   lenient about failure modes it shares with the writer. A brief that passes has
   *passed the critic* — it has not been verified true. The finalizer stamps this
   into every brief's `limitations`. The one grounding check that isn't an LLM's
   opinion is deterministic: a claim citing a source no researcher actually
   retrieved is dropped and demoted to an open question.
3. **Small corpus, no benchmark.** This demonstrates architecture, not accuracy.
   There is no retrieval quality metric here and none is claimed.
4. **Injection defense is a mitigation, not a guarantee.** Known patterns are
   flagged and output is bounded. A novel injection can still get through.
5. **Cost figures are estimates** from a hand-maintained price table, computed from
   provider-reported usage. Treat them as a relative signal, not a bill.

---

## Try to break it

The interesting runs are the ones that go wrong on purpose:

| Try this                                         | Expect                                                                                    |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Ask something the corpus only partly supports    | critic sends it back; revision counter increments; audit prices the revision as avoidable |
| Temporarily tighten a Pydantic constraint        | the repair path fires and shows up as **waste**                                           |
| Unset `ANTHROPIC_API_KEY`, keep `OPENAI_API_KEY` | run completes on the fallback path; those tokens attributed to `fallback`                 |
| Set `--max-usd` below your typical run cost      | cycle halts, finalizes with `limitations: ["Halted at budget ceiling: ..."]`              |

Then act on a recommendation and run it again. That before/after — *waste ratio
X% → Y%, cost per run A → B* — is the whole point, and unlike a quality metric it
is fully deterministic to measure.

---

<details>
<summary><b>Repo layout</b></summary>

```
agent/
├── schemas.py       every Pydantic model — state, agent outputs, telemetry
├── state.py         the graph state TypedDict and its reducers
├── graph.py         StateGraph wiring, conditional edges, the bounded cycle
├── runtime.py       per-run context: llm, meter, run dir, trace writer
├── llm.py           provider routing, retry, failover, repair, metering
├── memory.py        SQLite checkpointer + scratchpad compaction
├── guards.py        injection quarantine + context caps
├── toolsurface.py   tool discovery + calls (MCP or in-process), quarantine enforced
├── nodes/
│   ├── supervisor.py    plans + routes; picks the specialist set
│   ├── researcher.py    tool-calling specialist, async fan-out
│   ├── synthesizer.py   merges findings into a draft Brief
│   ├── critic.py        scores the draft; emits revision directives
│   └── finalizer.py     last validation pass, writes the artifact
├── tools/
│   ├── corpus_search.py BM25 over corpus/ (no vector DB needed)
│   ├── calculator.py    AST-walked arithmetic, no eval
│   └── fetch_url.py     optional, off by default, SSRF-guarded
├── sdk_runtime/     the same pipeline on the OpenAI Agents SDK
│   ├── pipeline.py      agents, agents-as-tools, critic handoff, bounded loop
│   ├── model.py         SDK Model interface served by LLMClient
│   ├── mcp_bridge.py    SDK MCPServer over the quarantined tool surface
│   ├── guardrails.py    output guardrail: no claim without evidence
│   └── tracing.py       per-run agents_trace.jsonl export
└── auditor/
    ├── meter.py         TokenEvent capture + budget ceiling
    ├── attribute.py     cause taxonomy, waste ratio, recommendations
    ├── pricing.py       per-model $/token table (user-editable)
    └── report.py        renders audit.md + audit.json

mcp/server.py        the tools, served over MCP (stdio)
cli.py               ask / audit / runs
docs/                architecture and instrumentation diagrams
tests/               all offline, all stubbed
corpus/              your documents (gitignored)
runs/                your run artifacts (gitignored)
```

</details>

<details>
<summary><b>Environment</b></summary>

```bash
ANTHROPIC_API_KEY=              # primary
OPENAI_API_KEY=                 # optional fallback

# optional overrides
CARTOGRAPHER_CHEAP_MODEL=claude-haiku-4-5-20251001
CARTOGRAPHER_STRONG_MODEL=claude-sonnet-4-5-20250929
CARTOGRAPHER_OPENAI_CHEAP_MODEL=gpt-4o-mini
CARTOGRAPHER_OPENAI_STRONG_MODEL=gpt-4o
CARTOGRAPHER_ENABLE_FETCH_URL=0 # network access for the fetch tool
```

</details>
