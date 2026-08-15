"""Provider routing, failover, schema repair, and the single metering choke point.

Every model call in Cartographer goes through :meth:`LLMClient.call`. Nothing
else constructs a chat model, so the auditor cannot miss a call. ``node`` and
``cause`` are required keyword arguments — an unattributed call is a bug, and
the client raises rather than emitting an event it cannot explain.

Layers, outermost first:

1. **Tier routing** — ``cheap`` (Haiku) for planning and extraction, ``strong``
   (Sonnet) for synthesis and critique. This is the cost-optimisation story.
2. **Retry** — exponential backoff on rate limits and transient errors, each
   failed attempt metered as ``retry_transient`` with ``ok=False``.
3. **Failover** — after the primary provider exhausts its attempts (or when its
   key is absent), the equivalent OpenAI model serves the call and its tokens
   are attributed to ``fallback``.
4. **Schema repair** — if the response does not validate against the Pydantic
   model, the validation error is fed back once with an instruction to fix
   exactly those fields. The repair call is metered as ``schema_repair``, which
   the auditor classifies as waste.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agent.auditor.meter import TokenMeter
from agent.schemas import Cause, InputComposition, Provider, Tier, TokenEvent

T = TypeVar("T", bound=BaseModel)

Message = Any  # a LangChain message, or a ("role", "content") tuple


class LLMError(RuntimeError):
    """All providers and attempts exhausted."""


@dataclass
class LLMConfig:
    anthropic_cheap: str = field(
        default_factory=lambda: os.getenv(
            "CARTOGRAPHER_CHEAP_MODEL", "claude-haiku-4-5-20251001"
        )
    )
    anthropic_strong: str = field(
        default_factory=lambda: os.getenv(
            "CARTOGRAPHER_STRONG_MODEL", "claude-sonnet-4-5-20250929"
        )
    )
    openai_cheap: str = field(
        default_factory=lambda: os.getenv("CARTOGRAPHER_OPENAI_CHEAP_MODEL", "gpt-4o-mini")
    )
    openai_strong: str = field(
        default_factory=lambda: os.getenv("CARTOGRAPHER_OPENAI_STRONG_MODEL", "gpt-4o")
    )
    max_attempts: int = 3
    base_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    temperature: float = 0.0
    max_tokens: int = 4096
    enable_repair: bool = True

    def model_for(self, provider: Provider, tier: Tier) -> str:
        return {
            ("anthropic", "cheap"): self.anthropic_cheap,
            ("anthropic", "strong"): self.anthropic_strong,
            ("openai", "cheap"): self.openai_cheap,
            ("openai", "strong"): self.openai_strong,
        }[(provider, tier)]


def available_providers() -> list[Provider]:
    """Providers with a key present, primary first."""
    order: list[Provider] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        order.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        order.append("openai")
    return order


def _default_model_factory(provider: Provider, model: str, cfg: LLMConfig) -> Any:
    """Imported lazily so the test suite runs without provider SDKs installed."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=cfg.temperature, max_tokens=cfg.max_tokens
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=cfg.temperature)


def _usage_from(raw: Any) -> tuple[int, int, int]:
    """Pull (input, output, cached_input) out of a LangChain response.

    Providers disagree on where usage lives; every path here is best-effort and
    falls back to zeros rather than inventing numbers.
    """
    usage = getattr(raw, "usage_metadata", None) or {}
    if not usage:
        meta = getattr(raw, "response_metadata", None) or {}
        usage = meta.get("usage") or meta.get("token_usage") or {}

    def pick(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return 0

    cached = 0
    details = usage.get("input_token_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cache_read", 0) or 0)

    return (
        pick("input_tokens", "prompt_tokens"),
        pick("output_tokens", "completion_tokens"),
        cached,
    )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "rate limit",
        "ratelimit",
        "429",
        "overloaded",
        "timeout",
        "timed out",
        "503",
        "502",
        "500",
        "connection",
        "temporarily",
    )
    return any(marker in text for marker in markers)


class LLMClient:
    """The one place a model is called."""

    def __init__(
        self,
        meter: TokenMeter,
        config: LLMConfig | None = None,
        *,
        model_factory: Callable[[Provider, str], Any] | None = None,
        providers: Sequence[Provider] | None = None,
    ) -> None:
        self.meter = meter
        self.config = config or LLMConfig()
        self._factory = model_factory
        self._providers = list(providers) if providers is not None else None
        self._cache: dict[tuple[Provider, str], Any] = {}
        self.repair_count = 0

    # -- provider plumbing -----------------------------------------------

    @property
    def providers(self) -> list[Provider]:
        if self._providers is not None:
            return self._providers
        found = available_providers()
        if not found:
            raise LLMError(
                "No provider key found. Set ANTHROPIC_API_KEY (primary) or "
                "OPENAI_API_KEY (fallback) in your environment or .env file."
            )
        return found

    def _model(self, provider: Provider, model: str) -> Any:
        key = (provider, model)
        if key not in self._cache:
            if self._factory is not None:
                self._cache[key] = self._factory(provider, model)
            else:
                self._cache[key] = _default_model_factory(provider, model, self.config)
        return self._cache[key]

    # -- the public entry point -------------------------------------------

    async def call(
        self,
        schema: type[T],
        messages: Sequence[Message],
        *,
        node: str,
        cause: Cause,
        tier: Tier = "cheap",
        revision_index: int = 0,
        composition: InputComposition | None = None,
    ) -> T:
        """Return a validated ``schema`` instance, or raise :class:`LLMError`."""
        if not node or not cause:
            raise ValueError("every LLM call must declare both node and cause")

        last_error: Exception | None = None
        primary = self.providers[0]

        for provider in self.providers:
            model = self.config.model_for(provider, tier)
            # Tokens burned on a non-primary provider are failover overhead,
            # regardless of what the caller was trying to do.
            effective_cause: Cause = "fallback" if provider != primary else cause

            for attempt in range(self.config.max_attempts):
                try:
                    return await self._attempt(
                        schema,
                        list(messages),
                        provider=provider,
                        model=model,
                        node=node,
                        cause=effective_cause,
                        tier=tier,
                        revision_index=revision_index,
                        composition=composition,
                    )
                except ValidationError:
                    raise
                except Exception as exc:  # noqa: BLE001 - provider SDK errors vary
                    last_error = exc
                    self._meter_failure(
                        node=node,
                        model=model,
                        tier=tier,
                        provider=provider,
                        revision_index=revision_index,
                    )
                    if not _is_transient(exc) or attempt == self.config.max_attempts - 1:
                        break
                    await asyncio.sleep(self._backoff(attempt))

        raise LLMError(
            f"all providers failed for node={node} cause={cause}: {last_error}"
        ) from last_error

    async def call_with_tools(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any],
        *,
        node: str,
        cause: Cause,
        tier: Tier = "cheap",
        revision_index: int = 0,
        composition: InputComposition | None = None,
    ) -> Any:
        """One turn of a tool-calling loop, metered like any other call.

        Returns the raw ``AIMessage`` so the caller can inspect ``tool_calls``.
        Unstructured, so there is no repair path here — the structured
        extraction happens in a separate :meth:`call` once the loop finishes.
        """
        if not node or not cause:
            raise ValueError("every LLM call must declare both node and cause")

        last_error: Exception | None = None
        primary = self.providers[0]

        for provider in self.providers:
            model = self.config.model_for(provider, tier)
            effective_cause: Cause = "fallback" if provider != primary else cause
            bound = self._model(provider, model).bind_tools(list(tools))

            for attempt in range(self.config.max_attempts):
                try:
                    started = time.perf_counter()
                    response = await bound.ainvoke(list(messages))
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    input_tokens, output_tokens, cached = _usage_from(response)
                    self.meter.record(
                        TokenEvent(
                            node=node,
                            cause=effective_cause,
                            model=model,
                            tier=tier,
                            provider=provider,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached,
                            revision_index=revision_index,
                            latency_ms=latency_ms,
                            ok=True,
                            composition=composition or InputComposition(),
                        )
                    )
                    return response
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    self._meter_failure(
                        node=node,
                        model=model,
                        tier=tier,
                        provider=provider,
                        revision_index=revision_index,
                    )
                    if not _is_transient(exc) or attempt == self.config.max_attempts - 1:
                        break
                    await asyncio.sleep(self._backoff(attempt))

        raise LLMError(
            f"all providers failed for node={node} cause={cause}: {last_error}"
        ) from last_error

    def _backoff(self, attempt: int) -> float:
        delay = min(self.config.base_backoff_s * (2**attempt), self.config.max_backoff_s)
        return delay * (0.5 + random.random() / 2)  # jitter, avoids retry convoys

    def _meter_failure(
        self,
        *,
        node: str,
        model: str,
        tier: Tier,
        provider: Provider,
        revision_index: int,
    ) -> None:
        self.meter.record(
            TokenEvent(
                node=node,
                cause="retry_transient",
                model=model,
                tier=tier,
                provider=provider,
                revision_index=revision_index,
                ok=False,
            )
        )

    # -- one attempt, including its repair pass ---------------------------

    async def _attempt(
        self,
        schema: type[T],
        messages: list[Message],
        *,
        provider: Provider,
        model: str,
        node: str,
        cause: Cause,
        tier: Tier,
        revision_index: int,
        composition: InputComposition | None,
    ) -> T:
        runnable = self._model(provider, model).with_structured_output(
            schema, include_raw=True
        )

        parsed, error = await self._invoke(
            runnable,
            messages,
            schema=schema,
            provider=provider,
            model=model,
            node=node,
            cause=cause,
            tier=tier,
            revision_index=revision_index,
            composition=composition,
        )
        if parsed is not None:
            return parsed

        if not self.config.enable_repair:
            raise LLMError(f"{node}/{cause}: output failed validation: {error}")

        # One repair pass. The model is shown its own failure verbatim; asking
        # it to "fix exactly these fields" repairs far more reliably than
        # re-rolling the original prompt, and it is cheaper.
        self.repair_count += 1
        repair_messages = list(messages) + [
            (
                "user",
                "Your previous response failed schema validation with these "
                f"errors:\n\n{error}\n\nReturn the same content again, corrected "
                "to satisfy exactly those fields. Change nothing else. Do not "
                "explain the fix.",
            )
        ]
        parsed, error = await self._invoke(
            runnable,
            repair_messages,
            schema=schema,
            provider=provider,
            model=model,
            node=node,
            cause="schema_repair",
            tier=tier,
            revision_index=revision_index,
            composition=composition,
        )
        if parsed is not None:
            return parsed
        raise LLMError(f"{node}/{cause}: output failed validation after repair: {error}")

    async def _invoke(
        self,
        runnable: Any,
        messages: list[Message],
        *,
        schema: type[T],
        provider: Provider,
        model: str,
        node: str,
        cause: Cause,
        tier: Tier,
        revision_index: int,
        composition: InputComposition | None,
    ) -> tuple[T | None, str | None]:
        started = time.perf_counter()
        result = await runnable.ainvoke(messages)
        latency_ms = int((time.perf_counter() - started) * 1000)

        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

        # Resolve the outcome *before* metering, so a call that came back
        # unparsable is recorded as ok=False. Metering it optimistically would
        # hide exactly the failures the audit exists to surface.
        value: T | None = None
        error: str | None = None
        if parsing_error is not None:
            error = str(parsing_error)
        elif parsed is None:
            error = "model returned no parsable structured output"
        elif isinstance(parsed, schema):
            value = parsed
        else:
            # Some providers hand back a dict; validate it ourselves so the
            # repair path sees a real Pydantic error message.
            try:
                value = schema.model_validate(parsed)
            except ValidationError as exc:
                error = str(exc)

        input_tokens, output_tokens, cached = _usage_from(raw)
        self.meter.record(
            TokenEvent(
                node=node,
                cause=cause,
                model=model,
                tier=tier,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached,
                revision_index=revision_index,
                latency_ms=latency_ms,
                ok=value is not None,
                composition=composition or InputComposition(),
            )
        )
        return value, error
