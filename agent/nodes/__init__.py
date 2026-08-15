"""Node factories. Each takes a ``RunContext`` and returns a LangGraph node."""

from agent.nodes.critic import make_critic
from agent.nodes.finalizer import make_finalizer
from agent.nodes.researcher import make_researcher
from agent.nodes.supervisor import make_supervisor
from agent.nodes.synthesizer import make_synthesizer

__all__ = [
    "make_critic",
    "make_finalizer",
    "make_researcher",
    "make_supervisor",
    "make_synthesizer",
]
