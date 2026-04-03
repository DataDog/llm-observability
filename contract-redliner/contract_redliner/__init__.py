"""Contract redliner package.

A Python agent that reviews contracts, flags policy violations, and proposes clause rewrites.
Built with Pydantic AI and traced end-to-end with Datadog LLM Observability.
"""

from contract_redliner.agent import run_redliner
from contract_redliner.evaluators import clauses_with_issues

__all__ = [
    "run_redliner",
    "clauses_with_issues",
]
