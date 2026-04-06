"""Contract redliner agent using Pydantic AI.

Two agents:
  - segmenter: classifies document type and splits into clauses (output_type=DocumentSegment)
  - redliner: tool-calling agent that reviews each clause (output_type=RedlineResult)

LLM Observability tracing:
  - run_redliner() → @llmobs_agent (top-level [agent] span)
  - _segment()     → @llmobs_task  ([task] child span)
  - Each @redliner.tool uses `with LLMObs.tool()` for explicit [tool] spans
  - generate_proposal / generate_validation use @llm → [llm] spans nested inside [tool] spans
"""

import os

from pydantic_ai import Agent
from ddtrace.llmobs.decorators import agent as llmobs_agent

from .models import (
    ContractClauses,
    RedlineResult,
)
from .policies import POLICY_DB

MODEL = f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-5.4-nano')}"

ClauseExtractor = Agent(
    MODEL,
    output_type=ContractClauses,
    system_prompt=(
        "You are a contract parser. "
        "Extract the important clauses to review. "
        "For each clause, write out the main text of the clause only, "
        "ignoring line numbers and section titles."
    ),
)

RedLiner = Agent(
    MODEL,
    output_type=RedlineResult,
    system_prompt=(
        "You are a contract redlining agent. Your job is to review every "
        "clause and return a complete RedlineResult.\n\n"
        "Follow these steps:\n"
        "1. View the policy index.\n"
        "2. View the relevant policies.\n"
        "3. For each clause where revisions are needed, return the clause "
        "number, risk level, suggested revision, and reasoning.\n\n"
        "Suggested revisions should be no greatly exceed the original clause "
        "in length."
    ),
)


@RedLiner.tool_plain
def get_policy_index() -> str:
    """Get index of available policies."""
    lines = ["Policy Index"]
    for doc_type, policies in POLICY_DB.items():
        lines.append(f"doc_type: {doc_type}")
        for topic in policies:
            lines.append(f" - topic: {topic}")
    return "\n".join(lines)


@RedLiner.tool_plain
def get_policy(doc_type: str, topic: str = "") -> str:
    """Look up policy text by doc_type and topic."""
    return POLICY_DB.get(doc_type, {}).get(topic, "policy not found")


@llmobs_agent(name="ContractRedliner")
def run_redliner(contract_text: str) -> RedlineResult:
    """Run redliner on contract text."""

    print("Extracting clauses...")
    result = ClauseExtractor.run_sync(contract_text)
    print(result.output.clauses)

    numbered_clauses = "\n\n".join(
        f"Clause {i}\n\n{clause}"
        for i, clause in enumerate(result.output.clauses)
    )

    print("Reviewing clauses...")
    agent_result = RedLiner.run_sync(numbered_clauses)

    return agent_result.output
