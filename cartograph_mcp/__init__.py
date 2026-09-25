"""Cartograph as an MCP server: the research graph, callable by a model.

A protocol surface over the graph that already exists — no second agent loop,
no new orchestration. See ``cartograph_mcp.server`` for the tool list.

Distinct from ``mcp/server.py``, which serves the three raw tools to
Cartograph's *own* runtimes and leaves quarantine to that trusted client. Here
the client is an arbitrary model, so this server quarantines on its own side.
"""
