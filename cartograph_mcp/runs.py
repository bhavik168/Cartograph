"""Runs started over MCP: the existing graph, driven in the background.

Nothing here orchestrates. ``start`` builds the same ``RunContext`` the CLI
builds — an ``LLMClient`` over a ``TokenMeter``, the SQLite checkpointer, the
graph from ``agent.graph.build_graph`` — and hands it to ``graph.astream`` as a
background task. Every LLM call therefore goes through ``LLMClient.call`` with
the node and cause the graph's nodes already set; the MCP layer makes no model
calls of its own, which is why the cause taxonomy has no MCP-specific cause.
Where a run came from is recorded once, at the run level: ``origin: "mcp"`` in
``run.json`` and on the audit.

Observation rides on LangGraph's ``tasks`` stream (which node is running now)
and ``values`` stream (the state after each step). Artifacts are the same files
the CLI writes — ``brief.json``, ``tokens.jsonl``, ``audit.json`` — plus
``run.json``, and ``get_brief`` / ``get_token_audit`` read them from disk, so
the brief on the wire is byte-for-byte the brief on disk.

Budget, in three layers:

1. Per run, the ceiling is ``min(caller's max_usd, server's max_usd, what is
   left of the process-wide ceiling)``.
2. Process-wide, each in-flight run *reserves* its full ceiling, so concurrent
   runs cannot jointly overspend what was left when they started.
3. Enforcement is the graph's own: the meter is checked between revision
   passes, and a run over its ceiling finalizes with the reason in the brief.
   A pass that is already under way completes, so a run can overshoot its
   ceiling by up to one pass. That overshoot is counted against the
   process-wide total.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent.auditor.attribute import Audit
from agent.auditor.meter import TokenMeter
from agent.auditor.report import write_report
from agent.graph import build_graph, describe_outcome
from agent.guards import PathOutsideRoot, resolve_within
from agent.llm import LLMClient, LLMConfig, LLMError
from agent.memory import checkpointer
from agent.runtime import RunContext
from agent.schemas import Brief, Finding, RoutingDecision
from agent.state import MAX_REVISIONS, AnalystState, initial_state
from agent.tools.corpus_search import get_index, search_corpus
from agent.toolsurface import InProcessTools
from cartograph_mcp.config import ServerConfig
from cartograph_mcp.schemas import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    BriefContent,
    BriefResult,
    CartographToolError,
    ErrorCode,
    FinalizedReason,
    RunStarted,
    RunStatus,
    RunStatusName,
    StartBriefArgs,
    StatusContent,
    TokenAuditResult,
)

logger = logging.getLogger("cartograph_mcp")

ORIGIN = "mcp"
RUN_FILE = "run.json"
RECURSION_LIMIT = 50


class CorpusTools(InProcessTools):
    """The in-process tool surface, pinned to the server's corpus root.

    ``ToolSurface.call`` — the only public call path — is inherited unchanged,
    so every result still goes through ``guards.quarantine`` before the
    researcher sees it. Only where ``corpus_search`` reads from changes, and its
    fan-out is clamped to the same bound the MCP tool advertises.
    """

    kind = "in-process (cartograph_mcp)"

    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root

    async def _call_raw(self, name: str, args: dict[str, Any]) -> str:
        if name == "corpus_search":
            k = max(1, min(int(args.get("k", DEFAULT_TOP_K)), MAX_TOP_K))
            return await asyncio.to_thread(
                search_corpus, str(args.get("query", "")), k, str(self.corpus_root)
            )
        return await super()._call_raw(name, args)


def finalized_reason(
    state: AnalystState, max_revisions: int
) -> tuple[FinalizedReason, str | None]:
    """Why a finished run stopped. The only writer of ``halted_reason`` is the
    budget check in ``agent.graph.make_revise``."""
    halted = state.get("halted_reason")
    if halted:
        return "budget_ceiling", halted
    critique = state.get("critique")
    if critique is None:
        return "no_critique", "The run finalized without a critique."
    if critique.passed:
        return "passed_critic", None
    return "max_revisions", (
        f"Failed critic after {state.get('revision_count', 0)} revision(s) "
        f"of max {max_revisions}."
    )


@dataclass
class Run:
    run_id: str
    thread_id: str
    question: str
    max_revisions: int
    max_usd: float
    ctx: RunContext
    status: RunStatusName = "planning"
    current_node: str | None = None
    state: dict = field(default_factory=dict)
    failure: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        return self.status in ("planning", "running")

    @property
    def spent_usd(self) -> float:
        return self.ctx.meter.total_usd


class RunRegistry:
    def __init__(
        self,
        config: ServerConfig,
        *,
        model_factory: Callable[[str, str], Any] | None = None,
        providers: list[str] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory
        self._providers = providers
        self._llm_config = llm_config or LLMConfig()
        self._runs: dict[str, Run] = {}
        self._start_lock = asyncio.Lock()

    # -- budget ----------------------------------------------------------

    def committed_usd(self) -> float:
        """Spent by finished runs, plus the larger of reserve and spend for live ones."""
        return sum(
            max(run.max_usd, run.spent_usd) if run.active else run.spent_usd
            for run in self._runs.values()
        )

    # -- start -----------------------------------------------------------

    async def start(self, args: StartBriefArgs) -> RunStarted:
        if not get_index(str(self.config.corpus_root)).chunks:
            raise CartographToolError(
                ErrorCode.CORPUS_NOT_INDEXED,
                "The corpus root holds no .md or .txt documents; a run would have "
                "no evidence to cite.",
            )

        # Serialised so two concurrent starts cannot both pass the budget and
        # concurrency checks before either registers.
        async with self._start_lock:
            active = sum(1 for run in self._runs.values() if run.active)
            if active >= self.config.max_concurrent_runs:
                raise CartographToolError(
                    ErrorCode.RUN_LIMIT_REACHED,
                    f"{active} run(s) in flight; the server allows "
                    f"{self.config.max_concurrent_runs}.",
                    max_concurrent_runs=self.config.max_concurrent_runs,
                )

            remaining = self.config.total_max_usd - self.committed_usd()
            if remaining <= 0:
                raise CartographToolError(
                    ErrorCode.BUDGET_EXCEEDED,
                    "The server's process-wide USD ceiling is spent or reserved by "
                    "runs in flight.",
                    total_max_usd=self.config.total_max_usd,
                )
            max_usd = min(
                args.max_usd if args.max_usd is not None else float("inf"),
                self.config.run_max_usd,
                remaining,
            )

            run_id = new_run_id()
            run_dir = self.run_dir(run_id)
            thread_id = args.thread_id or run_id
            await self._assert_thread_unused(thread_id)

            meter = TokenMeter(path=run_dir / "tokens.jsonl", max_usd=max_usd)
            llm = LLMClient(
                meter,
                self._llm_config,
                model_factory=self._model_factory,
                providers=self._providers,
            )
            try:
                llm.providers  # noqa: B018 - raises when no provider key is configured
            except LLMError as exc:
                meter.close()
                raise CartographToolError(
                    ErrorCode.PROVIDER_UNAVAILABLE,
                    "No LLM provider key is configured on the server.",
                ) from exc

            max_revisions = (
                args.max_revisions if args.max_revisions is not None else MAX_REVISIONS
            )
            ctx = RunContext(
                llm=llm,
                meter=meter,
                run_dir=run_dir,
                run_id=run_id,
                max_revisions=max_revisions,
                corpus_dir=str(self.config.corpus_root),
                tools=CorpusTools(self.config.corpus_root),
            )
            run = Run(
                run_id=run_id,
                thread_id=thread_id,
                question=args.question,
                max_revisions=max_revisions,
                max_usd=max_usd,
                ctx=ctx,
            )
            self._runs[run_id] = run
            self._write_run_file(run)
            run.task = asyncio.create_task(self._drive(run), name=f"cartograph-run-{run_id}")

        return RunStarted(
            run_id=run_id,
            thread_id=thread_id,
            max_revisions=max_revisions,
            max_usd=max_usd,
        )

    async def _assert_thread_unused(self, thread_id: str) -> None:
        if any(run.thread_id == thread_id for run in self._runs.values()):
            raise CartographToolError(
                ErrorCode.THREAD_EXISTS, "That thread_id belongs to another run."
            )
        async with checkpointer(self.config.checkpoint_db) as saver:
            if saver is not None and await saver.aget_tuple(
                {"configurable": {"thread_id": thread_id}}
            ):
                raise CartographToolError(
                    ErrorCode.THREAD_EXISTS,
                    "That thread_id already has checkpoints; choose an unused one.",
                )

    # -- the background task ---------------------------------------------

    async def _drive(self, run: Run) -> None:
        ctx = run.ctx
        config = {
            "configurable": {"thread_id": run.thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        try:
            async with checkpointer(self.config.checkpoint_db) as saver:
                graph = build_graph(ctx, checkpointer=saver)
                async for mode, data in graph.astream(
                    initial_state(run.question, run.run_id),
                    config,
                    stream_mode=["tasks", "values"],
                ):
                    if mode == "values":
                        run.state = data
                    elif "result" not in data:  # a task starting
                        run.current_node = data["name"]
                    elif data["name"] == "supervisor" and run.status == "planning":
                        run.status = "running"
            run.status = "finalized"
        except asyncio.CancelledError:
            run.status, run.failure = "failed", "Cancelled: the server shut down mid-run."
            raise
        except LLMError:
            logger.exception("run %s: LLM failure", run.run_id)
            run.status, run.failure = "failed", "Every LLM provider failed for a call."
        except Exception:
            logger.exception("run %s: unexpected failure", run.run_id)
            run.status, run.failure = "failed", "Internal error; see the server log."
        finally:
            run.finished_at = time.time()
            try:
                write_report(
                    ctx.run_dir,
                    ctx.meter.events,
                    run_id=run.run_id,
                    question=run.question,
                    outcome=(
                        describe_outcome(run.state, run.max_revisions)
                        if run.status == "finalized"
                        else f"failed — {run.failure}"
                    ),
                    wall_clock_s=run.finished_at - run.started_at,
                    origin=ORIGIN,
                )
                self._write_run_file(run)
            finally:
                ctx.meter.close()
                ctx.close()

    def _write_run_file(self, run: Run) -> None:
        reason, detail = (
            finalized_reason(run.state, run.max_revisions)
            if run.status == "finalized"
            else (None, None)
        )
        run.ctx.write_json(
            RUN_FILE,
            {
                "run_id": run.run_id,
                "origin": ORIGIN,
                "thread_id": run.thread_id,
                "question": run.question,
                "status": run.status,
                "failure": run.failure,
                "finalized_reason": reason,
                "reason_detail": detail,
                "revision_count": run.state.get("revision_count", 0),
                "max_revisions": run.max_revisions,
                "max_usd": run.max_usd,
                "llm_calls": run.ctx.meter.call_count,
                "tokens_spent": run.ctx.meter.total_tokens,
                "est_usd_spent": run.ctx.meter.total_usd,
                "last_node": run.current_node,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
            },
        )

    # -- reads -----------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        try:
            return resolve_within(self.config.runs_root, run_id)
        except PathOutsideRoot as exc:
            raise CartographToolError(
                ErrorCode.PATH_OUTSIDE_ROOT, "That run id resolves outside the runs root."
            ) from exc

    def _record(self, run_id: str) -> dict:
        """``run.json`` of an MCP-started run, or ``UNKNOWN_RUN``."""
        path = self.run_dir(run_id) / RUN_FILE
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = None
        if not isinstance(record, dict) or record.get("origin") != ORIGIN:
            raise CartographToolError(
                ErrorCode.UNKNOWN_RUN, "No run with that id was started by this server."
            )
        return record

    def _finished_record(self, run_id: str) -> dict:
        run = self._runs.get(run_id)
        if run is not None and run.active:
            raise CartographToolError(
                ErrorCode.RUN_IN_PROGRESS,
                f"The run is still {run.status}; poll get_run_status.",
                status=run.status,
                current_node=run.current_node,
            )
        record = self._record(run_id)
        if record.get("status") in ("planning", "running"):
            # On disk as live but not in memory: the server restarted mid-run.
            raise CartographToolError(
                ErrorCode.RUN_FAILED, "The run was interrupted by a server restart."
            )
        return record

    async def status(self, run_id: str) -> RunStatus:
        run = self._runs.get(run_id)
        if run is not None:
            state: dict = run.state
            meter = run.ctx.meter
            reason = (
                finalized_reason(state, run.max_revisions)[0]
                if run.status == "finalized"
                else None
            )
            return RunStatus(
                run_id=run_id,
                status=run.status,
                current_node=run.current_node,
                revision_count=state.get("revision_count", 0),
                max_revisions=run.max_revisions,
                llm_calls=meter.call_count,
                tokens_spent=meter.total_tokens,
                est_usd_spent=meter.total_usd,
                max_usd=run.max_usd,
                finalized_reason=reason,
                failure=run.failure,
                untrusted_content=StatusContent(
                    routing=state.get("routing"), findings=state.get("findings") or []
                ),
                injection_flags=sorted(set(state.get("flags") or [])),
            )

        # Not in memory: a run from before a restart. Counters come from
        # run.json, content from the last checkpoint of its thread.
        record = self._record(run_id)
        values = await self._checkpointed_values(record["thread_id"])
        status = record["status"]
        failure = record.get("failure")
        if status in ("planning", "running"):
            status, failure = "failed", "Interrupted by a server restart."
        routing = values.get("routing")
        return RunStatus(
            run_id=run_id,
            status=status,
            current_node=record.get("last_node"),
            revision_count=record.get("revision_count", 0),
            max_revisions=record["max_revisions"],
            llm_calls=record.get("llm_calls", 0),
            tokens_spent=record.get("tokens_spent", 0),
            est_usd_spent=record.get("est_usd_spent", 0.0),
            max_usd=record["max_usd"],
            finalized_reason=record.get("finalized_reason"),
            failure=failure,
            untrusted_content=StatusContent(
                routing=RoutingDecision.model_validate(routing) if routing else None,
                findings=[Finding.model_validate(f) for f in values.get("findings") or []],
            ),
            injection_flags=sorted(set(values.get("flags") or [])),
        )

    async def _checkpointed_values(self, thread_id: str) -> dict:
        async with checkpointer(self.config.checkpoint_db) as saver:
            if saver is None:
                return {}
            saved = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
        # Channel values come back as the models (or dicts) the graph stored;
        # model_validate accepts either.
        return saved.checkpoint.get("channel_values", {}) if saved else {}

    def brief(self, run_id: str) -> BriefResult:
        record = self._finished_record(run_id)
        if record.get("status") == "failed":
            raise CartographToolError(
                ErrorCode.RUN_FAILED,
                record.get("failure") or "The run ended without a brief.",
            )
        path = self.run_dir(run_id) / "brief.json"
        try:
            brief = Brief.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise CartographToolError(
                ErrorCode.RUN_FAILED, "The run finalized but its brief.json is unreadable."
            ) from exc
        run = self._runs.get(run_id)
        flags = sorted(set((run.state.get("flags") if run else None) or []))
        return BriefResult(
            run_id=run_id,
            finalized_reason=record["finalized_reason"],
            reason_detail=record.get("reason_detail"),
            revision_count=record.get("revision_count", 0),
            max_revisions=record["max_revisions"],
            untrusted_content=BriefContent(brief=brief),
            injection_flags=flags,
        )

    def token_audit(self, run_id: str) -> TokenAuditResult:
        self._finished_record(run_id)
        path = self.run_dir(run_id) / "audit.json"
        try:
            audit = Audit.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            raise CartographToolError(
                ErrorCode.RUN_FAILED, "The run's audit.json is missing or unreadable."
            ) from exc
        return TokenAuditResult(
            run_id=run_id,
            origin=audit.origin,
            llm_calls=audit.llm_calls,
            failed_calls=audit.failed_calls,
            total_tokens=audit.total_tokens,
            productive_tokens=audit.productive_tokens,
            overhead_tokens=audit.overhead_tokens,
            waste_tokens=audit.waste_tokens,
            waste_ratio=audit.waste_ratio,
            est_cost_usd=audit.est_cost_usd,
            by_cause=audit.by_cause,
            by_node=audit.by_node,
        )

    # -- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        tasks = [run.task for run in self._runs.values() if run.task and not run.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def new_run_id() -> str:
    """Sortable like the CLI's ids, plus a suffix: MCP runs can start in the same second."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"
