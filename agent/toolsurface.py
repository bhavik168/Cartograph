"""The tool surface: how either runtime discovers and calls tools.

Two implementations share one contract:

``MCPTools``        talks to ``mcp/server.py`` over MCP — stdio in real runs,
                    an in-memory transport in tests. The default.
``InProcessTools``  calls the LangChain tools directly. Selected by ``--no-mcp``
                    and by any ``RunContext`` without a surface, which is how
                    the offline suite runs with no subprocess.

The quarantine guarantee lives in the base class. :meth:`ToolSurface.call` is
the only public way to run a tool, and it returns a
:class:`~agent.guards.QuarantineResult`, never raw text: the raw result exists
only inside ``_call_raw`` and is handed straight to ``guards.quarantine``.
Neither runtime can reach an unquarantined result, because no method returns
one.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent import guards
from agent.tools.fetch_url import is_enabled as fetch_enabled

SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "mcp" / "server.py"


@dataclass(frozen=True)
class ToolSpec:
    """What a model needs to know to call a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        """OpenAI function format, which LangChain's ``bind_tools`` accepts for any provider."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _is_active(name: str) -> bool:
    return name != "fetch_url" or fetch_enabled()


class ToolSurface(ABC):
    """Discover and call tools. Every result comes back quarantined."""

    kind: str = "abstract"

    @abstractmethod
    async def discover(self) -> list[ToolSpec]:
        """Every tool the source exposes, active or not."""

    @abstractmethod
    async def _call_raw(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool. Private: its return value must never reach a prompt."""

    async def specs(self) -> list[ToolSpec]:
        """The tools a model may bind this run (``fetch_url`` only when enabled)."""
        return [spec for spec in await self.discover() if _is_active(spec.name)]

    async def bindable(self) -> list[Any]:
        """Tools in a form ``LLMClient.call_with_tools`` can bind."""
        return [spec.as_openai() for spec in await self.specs()]

    async def call(self, name: str, args: dict[str, Any] | None) -> guards.QuarantineResult:
        """Run a tool and return its quarantined result. The only public call path."""
        known = {spec.name for spec in await self.specs()}
        if name not in known:
            raw = f"TOOL ERROR: no such tool {name!r}"
        else:
            try:
                raw = await self._call_raw(name, dict(args or {}))
            except Exception as exc:  # noqa: BLE001 - returned to the model
                raw = f"TOOL ERROR: {type(exc).__name__}: {exc}"
        return guards.quarantine(raw, source=name or "unknown_tool")


class InProcessTools(ToolSurface):
    """The original direct-import path, kept for ``--no-mcp`` and the offline suite."""

    kind = "in-process"

    def _registry(self) -> dict[str, Any]:
        from agent.tools import calculator, corpus_search, fetch_url

        return {t.name: t for t in (corpus_search, calculator, fetch_url)}

    async def discover(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=t.name,
                description=t.description,
                input_schema=t.tool_call_schema.model_json_schema(),
            )
            for t in self._registry().values()
        ]

    async def bindable(self) -> list[Any]:
        # Bind the LangChain tool objects themselves, exactly as before MCP.
        registry = self._registry()
        return [registry[spec.name] for spec in await self.specs()]

    async def _call_raw(self, name: str, args: dict[str, Any]) -> str:
        tool = self._registry()[name]
        return str(await asyncio.to_thread(tool.invoke, args))


class MCPTools(ToolSurface):
    """Tools discovered and called over MCP."""

    kind = "mcp"

    def __init__(self, client: Any) -> None:
        self._client = client
        self._discovered: list[ToolSpec] | None = None

    # -- connections -----------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def stdio(
        cls, script: Path | str = SERVER_SCRIPT, cwd: Path | str | None = None
    ) -> AsyncIterator[MCPTools]:
        """Spawn ``mcp/server.py`` as a subprocess and talk to it over stdio."""
        from mcp import Client, StdioServerParameters

        # The SDK passes only a minimal default environment to the child, so the
        # tool switches are forwarded explicitly.
        env = {k: v for k, v in os.environ.items() if k.startswith("CARTOGRAPHER_")}
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(script)],
            cwd=str(cwd or Path.cwd()),
            env=env,
        )
        async with Client(params) as client:
            yield cls(client)

    @classmethod
    @asynccontextmanager
    async def in_memory(cls, server: Any | None = None) -> AsyncIterator[MCPTools]:
        """Connect to the same server in-process. Real MCP protocol, no subprocess."""
        from mcp import Client

        async with Client(server if server is not None else load_server()) as client:
            yield cls(client)

    # -- the surface -----------------------------------------------------

    async def discover(self) -> list[ToolSpec]:
        if self._discovered is None:
            result = await self._client.list_tools()
            self._discovered = [
                ToolSpec(
                    name=t.name,
                    description=t.description or "",
                    input_schema=dict(t.input_schema or {"type": "object", "properties": {}}),
                )
                for t in result.tools
            ]
        return self._discovered

    async def _call_raw(self, name: str, args: dict[str, Any]) -> str:
        result = await self._client.call_tool(name, args)
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        )
        return f"TOOL ERROR: {text}" if result.is_error else text


def load_server() -> Any:
    """Build the server defined in ``mcp/server.py``.

    Loaded by path under a private module name: the directory is called ``mcp``
    and cannot be imported as a package without shadowing the SDK.
    """
    spec = importlib.util.spec_from_file_location("cartograph_mcp_server", SERVER_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - file ships with the repo
        raise ImportError(f"cannot load MCP server from {SERVER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_server()
