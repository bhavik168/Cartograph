"""Second runtime: the same research pipeline on the OpenAI Agents SDK.

Selected with ``--runtime agents-sdk``. It takes the same inputs, produces the
same ``Brief``, and writes the same ``tokens.jsonl``. Import ``pipeline`` or
``guardrails`` directly. This package ``__init__`` does not import the SDK, so
the LangGraph path never loads it.
"""
