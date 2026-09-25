"""The Agents SDK's view of Cartograph's MCP server, with quarantine on.

The SDK's MCP integration puts a tool's result straight into the next model
input. Pointed at ``mcp/server.py`` directly (``MCPServerStdio``), that would be
an MCP-to-model path that skips ``guards.quarantine``.

This class is the SDK's ``MCPServer`` interface over a
:class:`~agent.toolsurface.ToolSurface`. The agents still discover and call
tools through the SDK's MCP machinery, and they reach the same server the
LangGraph runtime uses. The difference is that every result has already been
through ``ToolSurface.call`` and is quarantined before the SDK sees it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.mcp import MCPServer
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, TextContent, Tool

from agent.guards import QuarantineResult
from agent.toolsurface import ToolSurface

OnResult = Callable[[str, QuarantineResult], None]


class QuarantinedMCPServer(MCPServer):
    def __init__(self, surface: ToolSurface, on_result: OnResult | None = None) -> None:
        super().__init__(use_structured_content=False)
        self.surface = surface
        self._on_result = on_result

    @property
    def name(self) -> str:
        return f"cartograph-tools ({self.surface.kind})"

    async def connect(self) -> None:
        # The surface's connection is owned by the caller (the CLI or a test),
        # which also hands it to the LangGraph runtime. Nothing to open here.
        return None

    async def cleanup(self) -> None:
        return None

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Tool]:
        return [
            Tool(name=spec.name, description=spec.description, input_schema=spec.input_schema)
            for spec in await self.surface.specs()
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        result = await self.surface.call(tool_name, arguments)
        if self._on_result is not None:
            self._on_result(tool_name, result)
        return CallToolResult(content=[TextContent(type="text", text=result.text)])

    async def list_prompts(self) -> ListPromptsResult:
        return ListPromptsResult(prompts=[])

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        raise KeyError(f"{self.name} serves no prompts (asked for {name!r})")
