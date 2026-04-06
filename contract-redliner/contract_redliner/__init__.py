"""Contract redliner package.

A Python agent that reviews contracts, flags policy violations, and proposes clause rewrites.
Built with Pydantic AI and traced end-to-end with Datadog LLM Observability.
"""

from .agent import run_redliner

__all__ = [
    "run_redliner",
]
