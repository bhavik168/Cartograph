"""The model-facing MCP server (``cartograph_mcp``), fully offline.

Every test reaches the server over the real MCP protocol through the SDK's
in-memory transport, and every LLM call goes through ``LLMClient`` with the
injected stub ``model_factory``. No API key, no network, no subprocess — the one
stdio test is opt-in (``CARTOGRAPH_MCP_STDIO_TEST=1``).
"""

from __future__ import annotations

import os

import pytest

from cartograph_mcp.config import GRAPH_FETCH_SWITCH, ConfigError, ServerConfig

# -- configuration: the permission surface is fixed at boot --------------------


def test_config_refuses_to_boot_without_a_corpus_root(tmp_path):
    with pytest.raises(ConfigError, match="no corpus root"):
        ServerConfig.load(runs_root=tmp_path, env={})


def test_config_refuses_a_corpus_root_that_is_not_a_directory(tmp_path):
    with pytest.raises(ConfigError, match="not an existing directory"):
        ServerConfig.load(corpus_root=tmp_path / "missing", runs_root=tmp_path, env={})


def test_config_refuses_to_boot_on_an_unpriced_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CARTOGRAPHER_CHEAP_MODEL", "mystery-model-1")
    with pytest.raises(ConfigError, match="mystery-model-1"):
        ServerConfig.load(corpus_root=tmp_path, runs_root=tmp_path, env={})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_config_refuses_a_non_positive_or_infinite_ceiling(tmp_path, bad):
    with pytest.raises(ConfigError):
        ServerConfig.load(corpus_root=tmp_path, runs_root=tmp_path, run_max_usd=bad, env={})


def test_config_resolves_the_corpus_root(tmp_path):
    (tmp_path / "corpus").mkdir()
    config = ServerConfig.load(
        corpus_root=tmp_path / "x" / ".." / "corpus", runs_root=tmp_path / "runs", env={}
    )
    assert config.corpus_root == (tmp_path / "corpus").resolve()
    assert config.enable_fetch_url is False


def test_server_flag_overrides_an_inherited_fetch_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(GRAPH_FETCH_SWITCH, "1")  # e.g. leaked from the client's env
    ServerConfig.load(corpus_root=tmp_path, runs_root=tmp_path, env={}).pin_tool_policy()
    assert os.environ[GRAPH_FETCH_SWITCH] == "0"
