"""A stub chat model, so the whole graph runs in CI with no API key.

``LLMClient`` never constructs a model itself — it calls the injected
``model_factory``. That seam is what makes the entire orchestration layer
testable offline: swap the factory, drive canned Pydantic objects through the
real graph, and assert on routing, cycling, and accounting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class StubResponse:
    """Stands in for an AIMessage: carries usage metadata and tool calls."""

    def __init__(
        self,
        content: str = "",
        *,
        tool_calls: list[dict] | None = None,
        input_tokens: int = 100,
        output_tokens: int = 50,
        cached: int = 0,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {"cache_read": cached},
        }


@dataclass
class StubScript:
    """Canned behaviour, keyed by the schema each call asks for.

    ``responses`` maps a schema class to either a value or a callable taking
    the call index. A list is consumed one entry per call, which is how the
    critic is made to fail once and then pass.
    """

    responses: dict[type[BaseModel], Any] = field(default_factory=dict)
    tool_responses: list[StubResponse] = field(default_factory=list)
    errors: dict[type[BaseModel], list[Exception | None]] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    input_tokens: int = 100
    output_tokens: int = 50

    def next_for(self, schema: type[BaseModel]) -> Any:
        index = sum(1 for name, _ in self.calls if name == schema.__name__)
        self.calls.append((schema.__name__, None))

        planned = self.errors.get(schema)
        if planned and index < len(planned) and planned[index] is not None:
            raise planned[index]

        value = self.responses.get(schema)
        if value is None:
            raise AssertionError(f"stub has no response scripted for {schema.__name__}")
        if isinstance(value, list):
            return value[min(index, len(value) - 1)]
        if callable(value) and not isinstance(value, BaseModel):
            return value(index)
        return value

    def next_tool_response(self) -> StubResponse:
        index = sum(1 for name, _ in self.calls if name == "__tools__")
        self.calls.append(("__tools__", None))
        if not self.tool_responses:
            return StubResponse("done")
        return self.tool_responses[min(index, len(self.tool_responses) - 1)]


class _StructuredRunnable:
    def __init__(self, script: StubScript, schema: type[BaseModel]) -> None:
        self._script = script
        self._schema = schema

    async def ainvoke(self, messages: list[Any]) -> dict:
        value = self._script.next_for(self._schema)
        raw = StubResponse(
            input_tokens=self._script.input_tokens,
            output_tokens=self._script.output_tokens,
        )
        # A dict (rather than a model instance) is passed through unvalidated,
        # which exercises the client's own validate-then-repair path exactly as
        # a sloppy provider would.
        return {"raw": raw, "parsed": value, "parsing_error": None}


class StubChatModel:
    def __init__(self, script: StubScript, provider: str, model: str) -> None:
        self.script = script
        self.provider = provider
        self.model = model

    def with_structured_output(self, schema: type[BaseModel], include_raw: bool = False):
        return _StructuredRunnable(self.script, schema)

    def bind_tools(self, tools: list[Any]) -> StubChatModel:
        self._tools = tools
        return self

    async def ainvoke(self, messages: list[Any]) -> StubResponse:
        return self.script.next_tool_response()


def stub_factory(script: StubScript) -> Callable[[str, str], StubChatModel]:
    def factory(provider: str, model: str) -> StubChatModel:
        return StubChatModel(script, provider, model)

    return factory
