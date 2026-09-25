"""Server configuration: the whole permission surface, fixed at boot.

Nothing here can be changed by an MCP caller. A caller can ask for *less* — a
lower ``max_usd``, fewer revisions — but every ceiling below is the most it can
get, whatever it passes.

The server refuses to boot when:

* no corpus root is configured, or it is not an existing directory — there is
  no implicit default, because the corpus root is the read boundary;
* any configured model has no entry in ``agent/auditor/pricing.py`` — an
  unpriced model costs $0.00 to the meter, which silently switches off every
  USD ceiling this server enforces.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agent.auditor.pricing import is_priced
from agent.llm import LLMConfig

ENV_CORPUS_ROOT = "CARTOGRAPH_MCP_CORPUS_ROOT"
ENV_RUNS_ROOT = "CARTOGRAPH_MCP_RUNS_ROOT"
ENV_RUN_MAX_USD = "CARTOGRAPH_MCP_MAX_USD"
ENV_TOTAL_MAX_USD = "CARTOGRAPH_MCP_TOTAL_MAX_USD"
ENV_MAX_CONCURRENT_RUNS = "CARTOGRAPH_MCP_MAX_CONCURRENT_RUNS"
ENV_ENABLE_FETCH_URL = "CARTOGRAPH_MCP_ENABLE_FETCH_URL"

# The switch the graph's tool layer reads (agent.tools.fetch_url.is_enabled).
GRAPH_FETCH_SWITCH = "CARTOGRAPHER_ENABLE_FETCH_URL"

DEFAULT_RUN_MAX_USD = 0.50
DEFAULT_TOTAL_MAX_USD = 5.00
DEFAULT_MAX_CONCURRENT_RUNS = 2


class ConfigError(RuntimeError):
    """The server cannot start safely with this configuration."""


def _positive_usd(name: str, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be a finite amount above zero, got {value!r}")
    return value


def assert_models_priced(llm_config: LLMConfig) -> None:
    """Refuse any model the meter would price at zero."""
    models = {
        llm_config.model_for(provider, tier)
        for provider in ("anthropic", "openai")
        for tier in ("cheap", "strong")
    }
    unpriced = sorted(m for m in models if not is_priced(m))
    if unpriced:
        raise ConfigError(
            "no price entry for " + ", ".join(unpriced) + ". An unpriced model is "
            "metered at $0, which would disable every USD ceiling. Add it to "
            "agent/auditor/pricing.py or configure a priced model."
        )


@dataclass(frozen=True)
class ServerConfig:
    corpus_root: Path
    runs_root: Path
    run_max_usd: float = DEFAULT_RUN_MAX_USD
    total_max_usd: float = DEFAULT_TOTAL_MAX_USD
    max_concurrent_runs: int = DEFAULT_MAX_CONCURRENT_RUNS
    enable_fetch_url: bool = False

    @property
    def checkpoint_db(self) -> Path:
        return self.runs_root / "checkpoints.sqlite"

    @classmethod
    def load(
        cls,
        *,
        corpus_root: Path | str | None = None,
        runs_root: Path | str | None = None,
        run_max_usd: float | None = None,
        total_max_usd: float | None = None,
        max_concurrent_runs: int | None = None,
        enable_fetch_url: bool | None = None,
        env: Mapping[str, str] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> ServerConfig:
        """Explicit arguments win over the environment. Validates everything."""
        env = os.environ if env is None else env

        raw_corpus = corpus_root or env.get(ENV_CORPUS_ROOT)
        if not raw_corpus:
            raise ConfigError(
                f"no corpus root configured: pass --corpus-root or set {ENV_CORPUS_ROOT}"
            )
        corpus = Path(raw_corpus).expanduser().resolve()
        if not corpus.is_dir():
            raise ConfigError(f"corpus root {corpus} is not an existing directory")

        runs = Path(runs_root or env.get(ENV_RUNS_ROOT) or "runs").expanduser().resolve()
        runs.mkdir(parents=True, exist_ok=True)

        try:
            per_run = _positive_usd(
                "max_usd",
                run_max_usd if run_max_usd is not None
                else float(env.get(ENV_RUN_MAX_USD, DEFAULT_RUN_MAX_USD)),
            )
            total = _positive_usd(
                "total_max_usd",
                total_max_usd if total_max_usd is not None
                else float(env.get(ENV_TOTAL_MAX_USD, DEFAULT_TOTAL_MAX_USD)),
            )
            concurrent = (
                max_concurrent_runs if max_concurrent_runs is not None
                else int(env.get(ENV_MAX_CONCURRENT_RUNS, DEFAULT_MAX_CONCURRENT_RUNS))
            )
        except ValueError as exc:
            raise ConfigError(f"unparseable numeric setting: {exc}") from exc
        if per_run > total:
            raise ConfigError(f"max_usd {per_run} exceeds total_max_usd {total}")
        if concurrent < 1:
            raise ConfigError(f"max_concurrent_runs must be at least 1, got {concurrent}")

        fetch = (
            enable_fetch_url if enable_fetch_url is not None
            else env.get(ENV_ENABLE_FETCH_URL, "0") == "1"
        )

        assert_models_priced(llm_config or LLMConfig())

        return cls(
            corpus_root=corpus,
            runs_root=runs,
            run_max_usd=per_run,
            total_max_usd=total,
            max_concurrent_runs=concurrent,
            enable_fetch_url=fetch,
        )

    def pin_tool_policy(self) -> None:
        """Make this config the only authority over ``fetch_url`` in this process.

        The graph's tool layer reads ``CARTOGRAPHER_ENABLE_FETCH_URL`` at call
        time. Overwriting it here — to "0" as well as "1" — means a value
        inherited from the MCP client's environment cannot turn network access
        on behind the server's back.
        """
        os.environ[GRAPH_FETCH_SWITCH] = "1" if self.enable_fetch_url else "0"
